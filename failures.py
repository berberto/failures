from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid', {'font_scale': 2})
import functools


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float())# F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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
        model_norm = (torch.linalg.norm(model.fc1.weight).item(), torch.linalg.norm(model.fc2.weight).item())
    return 100. * correct / len(test_loader.dataset), model_norm[0], model_norm[1]


class LinearExampleDropout(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, drop_p=0.0):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.drop_p = drop_p

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        new_weight = (torch.rand((input.shape[0], *self.weight.shape), device=input.device) > self.drop_p) * self.weight[None, :, :]
        output = torch.bmm(new_weight, input[:, :, None])[:, :, 0] / (1 - self.drop_p)
        if self.bias is None:
            return output
        return output + self.bias

class Net(nn.Module):
    def __init__(self, N, layer_type=nn.Linear):
        super(Net, self).__init__()
        self.fc1 = layer_type(28 ** 2, N)
        self.fc2 = layer_type(N, 10)

        with torch.no_grad():
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x#output


def plot_weights_histograms (model, out_dir=".", name="init_weights"):
    # histogram of initial parameters
    plt.figure(figsize=(10, 5))
    for par_name, par_vals in model.named_parameters():
        weights_ = par_vals.data.detach().cpu().numpy()
        plt.hist(weights_.ravel(), density=True, bins="sqrt", alpha=.3, label=par_name)
        np.save(f"{out_dir}/{name}_{par_name}.npy", weights_)
    plt.axvline(0.,c="k")
    plt.legend()
    plt.savefig(f"{out_dir}/plot_histo_{name}.svg", bbox_inches="tight")


if __name__ == "__main__":

    import sys
    import os
    import numpy as np

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    N = int(sys.argv[1])
    drop_p = float(sys.argv[2])

    # set (and create) output directory
    out_dir = "outputs_RP/"
    out_dir += f"N:{N:04d}"
    out_dir += f"_dropout:{drop_p:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING
    batch_size = 256
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
    #   TRAINING WITHOUT DROPOUT
    lr = 5e-2
    wd = 1e-4
    n_epochs = 100

    test_acc = []
    model_norm1 = []
    model_norm2 = []

    model = Net(N).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, n_epochs)

    plot_weights_histograms(model, out_dir=out_dir, name="full")

    print(model)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm1, weight_norm2 = test(model, device, test_loader)
        test_acc.append(acc)
        model_norm1.append(weight_norm1)
        model_norm2.append(weight_norm2)
        scheduler.step()
    np.save(f"{out_dir}/full_accuracy.npy", np.array(test_acc))
    np.save(f"{out_dir}/full_norm_weights_1.npy", np.array(model_norm1))
    np.save(f"{out_dir}/full_norm_weights_2.npy", np.array(model_norm2))

    # ==================================================
    #   TRAINING WITH DROPOUT
    n_epochs = 100
    drop_p = 0.5
    lr = 5e-3
    wd = 1e-4

    test_acc_p = []
    model_norm1_p = []
    model_norm2_p = []

    model = Net(N, functools.partial(LinearExampleDropout, drop_p=drop_p)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, n_epochs)

    plot_weights_histograms(model, out_dir=out_dir, name="drop")

    print(model)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm1, weight_norm2 = test(model, device, test_loader)
        test_acc_p.append(acc)
        model_norm1_p.append(weight_norm1)
        model_norm2_p.append(weight_norm2)
        scheduler.step()
    np.save(f"{out_dir}/drop_accuracy.npy", np.array(test_acc_p))
    np.save(f"{out_dir}/drop_norm_weights_1.npy", np.array(model_norm1_p))
    np.save(f"{out_dir}/drop_norm_weights_2.npy", np.array(model_norm2_p))

    # ==================================================
    #      PLOTS
    
    test_acc = np.load(f"{out_dir}/full_accuracy.npy")
    test_acc_p = np.load(f"{out_dir}/drop_accuracy.npy")
    model_norm1_p = np.load(f"{out_dir}/drop_norm_weights_1.npy")
    model_norm2_p = np.load(f"{out_dir}/drop_norm_weights_2.npy")
    model_norm1 = np.load(f"{out_dir}/full_norm_weights_1.npy")
    model_norm2 = np.load(f"{out_dir}/full_norm_weights_2.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(test_acc, label='standard')
    plt.plot(test_acc_p, label='p={}'.format(drop_p))
    plt.legend()
    plt.title('accuracy')
    plt.savefig(f'{out_dir}/plot_accuracy.png', bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(model_norm1, label='standard')
    plt.plot(model_norm1_p, label='p={}'.format(drop_p))
    plt.legend()
    plt.title('L2 weight norm (fc1)')
    plt.savefig(f'{out_dir}/plot_L2_weight_norm_fc1.png', bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(model_norm2, label='standard')
    plt.plot(model_norm2_p, label='p={}'.format(drop_p))
    plt.legend()
    plt.title('L2 weight norm (fc2)')
    plt.savefig(f'{out_dir}/plot_L2_weight_norm_fc2.png', bbox_inches="tight")