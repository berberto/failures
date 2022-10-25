from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid', {'font_scale': 2})
import functools
import sys
import os

# conda update -n base -c defaults conda

def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


class Net(nn.Module):
    '''
    Network class
    '''
    def __init__(self, N, layer_type=nn.Linear, scaling="sqrt", lr=5e-2, verbose=False, **kwargs):
        super(Net, self).__init__()

        self.fc1 = layer_type(28 ** 2, N, bias=False, **kwargs)
        # self.fc1 = layer_type(28 ** 2, N, bias=False) # additional hidden layer
        self.fc2 = layer_type(N, 10, bias=False, **kwargs).requires_grad_(False) # frozen weights

        if verbose:
          for name, pars in self.named_parameters():
              print("=====")
              print(name)
              print("-----")
              f_in = pars.data.size()[1]
              print(f"input size = {f_in}")

        if scaling == "lin":
            # initialisation of the weights -- N(1/n, 1/n)
            for name, pars in self.named_parameters():
                f_in = 1.*pars.data.size()[1]
                pars.data.normal_(1./f_in, 1./f_in)
        elif scaling == "sqrt":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            for name, pars in self.named_parameters():
                f_in = 1.*pars.data.size()[1]
                pars.data.normal_(0., 2./np.sqrt(f_in))
        else:
            raise ValueError(f"Invalid scaling option '{scaling}'\nChoose either 'sqrt' or 'lin'")

        # SGD over the first layer 
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=0.)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

def train(model, device, train_loader, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.optimizer.zero_grad()
        output = model(data)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float(), reduction="sum")
        loss.backward()
        model.optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, F.one_hot(target, num_classes=10).float(), reduction='sum').item()# F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    with torch.no_grad():
        weights = list(model.parameters())
        model_norm = (torch.linalg.norm(weights[0]).item(), torch.linalg.norm(weights[1]).item())
    return 100. * correct / len(test_loader.dataset), model_norm[0], model_norm[1]


if __name__ == "__main__":


    # get parameters as inputs
    scaling = str(sys.argv[1])
    ltype = str(sys.argv[2])
    N = int(sys.argv[3])
    dropout = float(sys.argv[4])

    # set (and create) output directory
    out_dir = "outputs/"
    out_dir += f"scaling:{scaling}"
    out_dir += f"_N:{N:04d}"
    out_dir += f"_ltype:{ltype}"
    if ltype == "drop":
        out_dir += f"_dropout:{dropout:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # define network options
    layers = {"full": nn.Linear, "drop": WeightDropLinear}
    net_kwargs = {}
    net_kwargs["layer_type"] = layers[ltype]
    net_kwargs["scaling"] = scaling
    if ltype == "drop":
        net_kwargs["weight_dropout"] = dropout

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    from pprint import pprint
    pprint(net_kwargs)

    # define network model (and put on available device)
    model = Net(N, **net_kwargs).to(device)

    print(model)
    print("")

    # histogram of initial parameters
    weights = list(model.parameters())
    weights_1 = weights[0].data.detach().cpu().numpy(); print(f"weights_1: {weights_1.shape}"); print(f"mean_1 = {np.mean(weights_1):.5f}, std_1 = {np.std(weights_1):.5f}, ")
    weights_2 = weights[1].data.detach().cpu().numpy(); print(f"weights_2: {weights_2.shape}"); print(f"mean_2 = {np.mean(weights_2):.5f}, std_2 = {np.std(weights_2):.5f}, ")
    fig, ax = plt.subplots()
    ax.hist(weights_1.ravel(), density=True, bins="sqrt", alpha=.3, label="w1")
    ax.hist(weights_2.ravel(), density=True, bins="sqrt", alpha=.3, label="w2")
    ax.axvline(0.,c="k")
    ax.legend()
    fig.savefig(f"{out_dir}/init_weights.svg", bbox_inches="tight")
    np.save(f"{out_dir}/init_weights_1.npy", weights_1)
    np.save(f"{out_dir}/init_weights_2.npy", weights_2)
    plt.close(fig)

    # ==================================================
    n_epochs = 100
    batch_size = 256
    lr = 5e-2
    wd = 1e-4
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    use_cuda = True
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
            transforms.ToTensor(),
            ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    # ==================================================

    # train and monitor performance and weight norm 
    test_acc = []
    model_norm1 = []
    model_norm2 = []
    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, epoch, log_interval=1000)
        acc, weight_norm1, weight_norm2 = test(model, device, test_loader)
        test_acc.append(acc)
        model_norm1.append(weight_norm1)
        model_norm2.append(weight_norm2)
    np.save(f"{out_dir}/norm_weights_1.npy", weight_norm1)
    np.save(f"{out_dir}/norm_weights_2.npy", weight_norm2)


    # ==================================================
    fig, ax = plt.subplots()
    ax.plot(test_acc)
    ax.set_xlabel("epoch")
    ax.set_ylabel('accuracy')
    fig.savefig("accuracy.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(model_norm1, label='|w1|')
    ax.plot(model_norm2, label='|w2|')
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel('L2 weight norm')
    fig.savefig("weight_norm.svg", bbox_inches="tight")
    plt.close(fig)