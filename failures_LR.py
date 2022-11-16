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
import pickle

from networks import LinearWeightDropout, Net
from training_utils import train, test
from data import LinearRegressionDataset


def plot_weights_histograms (model, out_dir=".", name="init_weights"):
    # histogram of initial parameters
    fig, ax = plt.subplots(figsize=(6, 4))
    for par_name, par_vals in model.named_parameters():
        weights_ = par_vals.data.detach().cpu().numpy()
        ax.hist(weights_.ravel(), density=True, bins="sqrt", alpha=.3, label=par_name)
        np.save(f"{out_dir}/{name}_{par_name}.npy", weights_)
    ax.axvline(0.,c="k")
    ax.legend()
    fig.savefig(f"{out_dir}/plot_histo_{name}.svg", bbox_inches="tight")

def generate_data (N, n_samples, **kwargs):
    dataset = LinearRegressionDataset(N, n_samples)
    loader = torch.utils.data.DataLoader(dataset,**kwargs)
    return loader


if __name__ == "__main__":

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    scaling = sys.argv[1]       # init pars scaling ("lin"=1/N or "sqrt"=1/sqrt(N))
    N = int(sys.argv[2])        # number of input and hidden units
    drop_p = float(sys.argv[3]) # probability of weight drop
    drop_l = sys.argv[4]        # layer(s) with dropout ("1", "12" or "2")
    if not drop_p:
        drop_l = None

    # set (and create) output directory
    out_dir = "outputs_2L/"
    out_dir += f"{scaling}/"
    out_dir += f"N_{N:04d}/"
    out_dir += f"{drop_l}/"
    out_dir += f"q_{drop_p:.2f}"    
    os.makedirs(out_dir, exist_ok=True)

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING

    n_epochs = 1000

    n_train = 100000
    n_test = 1000
    n_skip = min(100, n_epochs) # epochs to skip when saving data

    lr = 1e-4
    wd = 0.

    batch_size = n_train//10
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    use_cuda = True
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


    # ==================================================
    #   TRAINING/TESTING

    train_loss = []
    test_acc = []
    weights_norm = []
    hidden_var = []
    model_weights = []

    # model = Net(N, layer_type=nn.Linear, scaling=scaling, bias=False).to(device)
    model = Net(N, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), bias=False, 
                scaling=scaling, drop_l=drop_l).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    model.save(f"{out_dir}/model_init")

    print(model)

    for epoch in range(n_epochs + 1):
        # train
        train_loader = generate_data(N, n_train, **train_kwargs)
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        # test
        test_loader = generate_data(N, n_test, **test_kwargs)
        acc, model_weights_, hidden_var_ = test(model, device, test_loader)
        # collect statistics
        if epoch % n_skip == 0:
            model_weights.append(model_weights_)
        train_loss.append(loss)
        test_acc.append(acc)
        hidden_var.append(hidden_var_)
    model.save(f"{out_dir}/model_trained")
    np.save(f"{out_dir}/train_loss.npy", np.array(train_loss))
    np.save(f"{out_dir}/test_loss.npy", np.array(test_acc))
    np.save(f"{out_dir}/hidden_var.npy", np.array(hidden_var))
    with open(f"{out_dir}/weights.pkl", "wb") as f:
        pickle.dump(model_weights, f)

    # ==================================================
    #      ANALYSIS

    # get weights and calculate norm
    with open(f"{out_dir}/weights.pkl", "rb") as f:
        model_weights_ = pickle.load(f)
        weights_list = []
        weights_norm = []
        svd_list = []
        for i in range(len(model_weights_[0])):
            weights_list.append( np.array([np.squeeze(w[i]) for w in model_weights_]) )
            weights_norm.append( np.array([np.linalg.norm(w[i]) for w in model_weights_]) )
    
    with open(f"{out_dir}/weights_norm.pkl", "wb") as f:
        pickle.dump(weights_norm)

    W1 = weights_list[0]; norm1 = weights_norm[0]
    W2 = weights_list[1]; norm2 = weights_norm[1]

    # singular value decomposition of W1 for all snapshots
    U, S, V = np.linalg.svd(W1)

    # select dominant modes
    mode = np.argmax(S, axis=1)
    eval1 = S[:, mode]
    Rvec1 = U[:, mode]
    Lvec1 = V[:, mode]

    # calculate the participation ratio
    PR = np.array([np.sum(s)**2/np.sum(s**2) for s in S])

    # save spectral properties for all stored snapshots
    np.save(f"{out_dir}/PR.npy", PR)
    np.save(f"{out_dir}/eigenvalues.npy", S)
    np.save(f"{out_dir}/eval.npy", eval1)
    np.save(f"{out_dir}/Levec.npy", Lvec1)
    np.save(f"{out_dir}/Revec.npy", Rvec1)

    # ==================================================
    #      PLOTS
    
    # re-load saved data
    train_loss = np.load(f"{out_dir}/train_loss.npy")
    test_acc = np.load(f"{out_dir}/test_loss.npy")
    hidden_var = np.load(f"{out_dir}/hidden_var.npy")
    with open(f"{out_dir}/weights_norm.pkl", "rb") as f:
        weights_norm = pickle.load(f)

    saved_epochs = np.arange(0,n_epochs+1,n_skip)
    PR = np.load(f"{out_dir}/PR.npy")
    S = np.load(f"{out_dir}/eigenvalues.npy")
    eval1 = np.load(f"{out_dir}/eval.npy")
    Lvec1 = np.load(f"{out_dir}/Levec.npy")
    Rvec1 = np.load(f"{out_dir}/Revec.npy")

    title = f"init {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; N ={N:04d}; drop {drop_l} wp {drop_p:.2f}"
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.plot(saved_epochs, PR)
    ax.set_ylabel('participation ratio')
    ax.set_xlabel('epoch')
    fig.savefig(f'{out_dir}/plot_PR.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.plot(saved_epochs, eval1/N)
    ax.set_ylabel(r'$\lambda$ / N')
    ax.set_xlabel('epoch')
    fig.savefig(f'{out_dir}/plot_eval1.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss)
    ax.plot(test_acc)
    ax.set_title(title)
    ax.set_ylabel('Training loss')
    ax.set_xlabel('epoch')
    fig.savefig(f'{out_dir}/plot_train_loss.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('L2 weight norm')
    ax.set_xlabel('epoch')
    ax.set_ylim([0,1])
    for i, (norm, c) in enumerate(zip(weights_norm, colors)):
        ax.plot(norm/norm[0], c=c, label=f'layer {i+1}')
    ax.legend(loc='best')
    fig.savefig(f'{out_dir}/plot_weights_norm.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('Hidden layer variance')
    ax.set_xlabel('epoch')
    ax.set_ylim([0,1])
    ln = ax.plot(hidden_var/hidden_var[0])
    fig.savefig(f'{out_dir}/plot_hidden_layer_variance.png', bbox_inches="tight")

