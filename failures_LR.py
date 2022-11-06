from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import functools
import sys
import os
import numpy as np

from networks import LinearWeightDropout, Net
from training_utils import train, test
from data import LinearRegressionDataset


def plot_weights_histograms (model, out_dir=".", name="init_weights"):
    # histogram of initial parameters
    plt.figure(figsize=(6, 4))
    for par_name, par_vals in model.named_parameters():
        weights_ = par_vals.data.detach().cpu().numpy()
        plt.hist(weights_.ravel(), density=True, bins="sqrt", alpha=.3, label=par_name)
        np.save(f"{out_dir}/{name}_{par_name}.npy", weights_)
    plt.axvline(0.,c="k")
    plt.legend()
    plt.savefig(f"{out_dir}/plot_histo_{name}.svg", bbox_inches="tight")


if __name__ == "__main__":

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    scaling = sys.argv[1]
    N = int(sys.argv[2])
    drop_p = float(sys.argv[3])

    # set (and create) output directory
    out_dir = "outputs_LR_2L/"
    out_dir += f"init_{scaling}"
    out_dir += f"__N_{N:04d}"
    out_dir += f"__dropout_{drop_p:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING
    
    n_epochs = 100000
    lr = 1e-4
    wd = 0.

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
    dataset1 = LinearRegressionDataset(N, n_train)
    dataset2 = LinearRegressionDataset(N, n_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # ==================================================
    #   TRAINING WITHOUT DROPOUT

    train_loss = []
    test_acc = []
    model_norm = []

    model = Net(N, layer_type=nn.Linear, scaling=scaling, bias=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = CosineAnnealingLR(optimizer, n_epochs)

    model.save(f"{out_dir}/full_model_init")
    plot_weights_histograms(model, out_dir=out_dir, name="full")

    print(model)

    for epoch in range(n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm, _ = test(model, device, test_loader)
        train_loss.append(loss)
        test_acc.append(acc)
        model_norm.append(weight_norm)
        # scheduler.step()
    np.save(f"{out_dir}/full_train_loss.npy", np.array(train_loss))
    np.save(f"{out_dir}/full_test_loss.npy", np.array(test_acc))
    np.save(f"{out_dir}/full_norm_weights.npy", np.array(model_norm))

    # ==================================================
    #   TRAINING WITH DROPOUT

    train_loss_p = []
    test_acc_p = []
    model_norm_p = []

    model = Net(N, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), bias=False, scaling=scaling).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = CosineAnnealingLR(optimizer, n_epochs)

    model.save(f"{out_dir}/drop_model_init")
    plot_weights_histograms(model, out_dir=out_dir, name="drop")

    print(model)

    for epoch in range(n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, weight_norm, _ = test(model, device, test_loader)
        train_loss_p.append(loss)
        test_acc_p.append(acc)
        model_norm_p.append(weight_norm)
        # scheduler.step()
    np.save(f"{out_dir}/drop_train_loss.npy", np.array(train_loss_p))
    np.save(f"{out_dir}/drop_test_loss.npy", np.array(test_acc_p))
    np.save(f"{out_dir}/drop_norm_weights.npy", np.array(model_norm_p))

    # ==================================================
    #      PLOTS
    
    title = f"init: {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; N ={N:04d}"

    train_loss = np.load(f"{out_dir}/full_train_loss.npy")
    train_loss_p = np.load(f"{out_dir}/drop_train_loss.npy")
    test_acc = np.load(f"{out_dir}/full_test_loss.npy")
    test_acc_p = np.load(f"{out_dir}/drop_test_loss.npy")
    model_norm_p = np.load(f"{out_dir}/drop_norm_weights.npy")
    model_norm = np.load(f"{out_dir}/full_norm_weights.npy")

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='standard')
    plt.plot(train_loss_p, label='p={}'.format(drop_p), ls="--")
    plt.legend()
    plt.title(title)
    plt.ylabel('Training loss')
    plt.xlabel('epoch')
    plt.savefig(f'{out_dir}/plot_train_loss.png', bbox_inches="tight")

    plt.figure(figsize=(6, 4))
    plt.plot(model_norm, label='standard')
    plt.plot(model_norm_p, label='p={}'.format(drop_p), ls="--")
    plt.legend()
    plt.title(title)
    plt.ylabel('L2 weight norm (fc1)')
    plt.xlabel('epoch')
    plt.savefig(f'{out_dir}/plot_L2_weight_norm_fc1.png', bbox_inches="tight")
