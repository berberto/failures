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

class LinearRegressionDataset(torch.utils.data.Dataset):
    def __init__ (self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, i):
        return self.X[i], self.y[i]

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.clone().detach().float().to(device)
        target = target.clone().detach().float().view(-1,1).to(device)
        optimizer.zero_grad()
        output = model(data)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss = F.mse_loss(output, target, reduction="mean")
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.clone().detach().float().to(device)
            target = target.clone().detach().float().view(-1,1).to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction="mean").item()# F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    with torch.no_grad():
        model_norm = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_norm.append(torch.linalg.norm(pars).item())
    return (test_loss, ) + tuple(model_norm)


class LinearExampleDropout(nn.Linear):
    def __init__(self, in_features, out_features, drop_p=0.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
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
    def __init__(self, N, layer_type=nn.Linear, scaling="sqrt"):
        super(Net, self).__init__()
        self.fc1 = layer_type(N, 1)

        if scaling == "lin":
            # initialisation of the weights -- N(1/n, 1/n)
            for name, pars in self.named_parameters():
                if "weight" in name:
                    f_in = 1.*pars.data.size()[1]
                    pars.data.normal_(1./f_in, 1./f_in)
        elif scaling == "sqrt":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            for name, pars in self.named_parameters():
                if "weight" in name:
                    f_in = 1.*pars.data.size()[1]
                    pars.data.normal_(0., 2./np.sqrt(f_in))
        else:
            raise ValueError(f"Invalid scaling option '{scaling}'\nChoose either 'sqrt' or 'lin'")

    def forward(self, x):
        x = self.fc1(x)
        return x

    def save(self, filename):
        T.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(T.load(filename, map_location=self.device))


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
    scaling = sys.argv[1]
    N = int(sys.argv[2])
    drop_p = float(sys.argv[3])

    # set (and create) output directory
    out_dir = "outputs_LR/noWD_"
    out_dir += f"init_{scaling}"
    out_dir += f"__N_{N:04d}"
    out_dir += f"__dropout_{drop_p:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING
    
    batch_size = 200
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    use_cuda = True
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


    # ==================================================
    #   GENERATING TRAINING AND TEST DATA

    n_train = 10
    n_test = 1000

    w_star = np.ones(N)/N

    _th = 2*np.pi * np.random.rand(n_train); _v = np.random.randn(n_train, N)
    X_train = (np.cos(_th)**2)[:, None] * w_star[None,:] + (np.sin(_th)**2)[:,None] * _v
    y_train = np.sum(X_train * w_star[None,:], axis=1)

    _th = 2*np.pi * np.random.rand(n_test); _v = np.random.randn(n_test, N)
    X_test = (np.cos(_th)**2)[:, None] * w_star[None,:] + (np.sin(_th)**2)[:,None] * _v
    y_test = np.sum(X_test * w_star[None,:], axis=1)

    print(f"w_star.shape = {w_star.shape}\t max(|w_star.shape|) = {np.max(np.abs(w_star))}")
    print(f"X_train.shape = {X_train.shape}\t max(|X_train.shape|) = {np.max(np.abs(X_train))}")
    print(f"y_train.shape = {y_train.shape}\t max(|y_train.shape|) = {np.max(np.abs(y_train))}")
    print(f"X_test.shape = {X_test.shape}\t max(|X_test.shape|) = {np.max(np.abs(X_test))}")
    print(f"y_test.shape = {y_test.shape}\t max(|y_test.shape|) = {np.max(np.abs(y_test))}")

    dataset1 = LinearRegressionDataset(X_train, y_train)
    dataset2 = LinearRegressionDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # ==================================================
    #   TRAINING WITHOUT DROPOUT
    lr = 1e-4
    wd = 0.
    n_epochs = 200

    train_loss = []
    test_acc = []
    model_norm = []

    model = Net(N, layer_type=nn.Linear, scaling=scaling).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = CosineAnnealingLR(optimizer, n_epochs)

    plot_weights_histograms(model, out_dir=out_dir, name="full")

    print(model)

    for epoch in range(1, n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm = test(model, device, test_loader)
        train_loss.append(loss)
        test_acc.append(acc)
        model_norm.append(weight_norm)
        # scheduler.step()
    np.save(f"{out_dir}/full_train_loss.npy", np.array(train_loss))
    np.save(f"{out_dir}/full_test_loss.npy", np.array(test_acc))
    np.save(f"{out_dir}/full_norm_weights.npy", np.array(model_norm))

    # ==================================================
    #   TRAINING WITH DROPOUT
    n_epochs = 200
    lr = 1e-4
    wd = 0.

    train_loss_p = []
    test_acc_p = []
    model_norm_p = []

    model = Net(N, layer_type=functools.partial(LinearExampleDropout, drop_p=drop_p), scaling=scaling).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = CosineAnnealingLR(optimizer, n_epochs)

    plot_weights_histograms(model, out_dir=out_dir, name="drop")

    print(model)

    for epoch in range(1, n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm = test(model, device, test_loader)
        train_loss_p.append(loss)
        test_acc_p.append(acc)
        model_norm_p.append(weight_norm)
        # scheduler.step()
    np.save(f"{out_dir}/drop_train_loss.npy", np.array(train_loss_p))
    np.save(f"{out_dir}/drop_test_loss.npy", np.array(test_acc_p))
    np.save(f"{out_dir}/drop_norm_weights.npy", np.array(model_norm_p))

    # ==================================================
    #      PLOTS
    
    train_loss = np.load(f"{out_dir}/full_train_loss.npy")
    train_loss_p = np.load(f"{out_dir}/drop_train_loss.npy")
    test_acc = np.load(f"{out_dir}/full_test_loss.npy")
    test_acc_p = np.load(f"{out_dir}/drop_test_loss.npy")
    model_norm_p = np.load(f"{out_dir}/drop_norm_weights.npy")
    model_norm = np.load(f"{out_dir}/full_norm_weights.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='standard')
    plt.plot(train_loss_p, label='p={}'.format(drop_p))
    plt.legend()
    plt.title('Training loss')
    plt.savefig(f'{out_dir}/plot_train_loss.png', bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(model_norm, label='standard')
    plt.plot(model_norm_p, label='p={}'.format(drop_p))
    plt.legend()
    plt.title('L2 weight norm (fc1)')
    plt.savefig(f'{out_dir}/plot_L2_weight_norm_fc1.png', bbox_inches="tight")
