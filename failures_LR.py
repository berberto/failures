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

    training = True
    analysis = True

    # ==================================================
    #   SETUP PARAMETERS

    np.random.seed(1871)

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

    n_epochs = 100000

    n_train = 100000
    n_test = 1000
    n_skip = min(1000, n_epochs//10) # epochs to skip when saving data

    lr = 1e-3
    wd = 0.

    train_kwargs = {'batch_size': 1000}
    test_kwargs = {'batch_size': n_test}
    use_cuda = True
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


    # ==================================================
    #   TRAINING/TESTING

    if training:

        print("\nTRAINING ...")

        train_loss = []
        test_acc = []
        weights_norm = []
        hidden = []
        model_weights = []
        saved_epochs = []

        # model = Net(N, layer_type=nn.Linear, scaling=scaling, bias=False).to(device)
        model = Net(N, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), bias=False, 
                    scaling=scaling, drop_l=drop_l).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        model.save(f"{out_dir}/model_init")

        print(model)
        test_loader = generate_data(N, n_test, **test_kwargs)
        train_loader = generate_data(N, n_train, **train_kwargs)

        for epoch in range(n_epochs + 1):
            # test
            acc, model_weights_, hidden_ = test(model, device, test_loader)
            # train
            loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)

            train_loss.append(loss)
            test_acc.append(acc)
            # collect statistics
            if epoch % n_skip == 0:
                saved_epochs.append(epoch)
                hidden.append(hidden_)
                model_weights.append(model_weights_)
                # save data
                model.save(f"{out_dir}/model_trained")
                np.save(f"{out_dir}/saved_epochs.npy", np.array(saved_epochs))
                np.save(f"{out_dir}/train_loss.npy", np.array(train_loss))
                np.save(f"{out_dir}/test_loss.npy", np.array(test_acc))
                np.save(f"{out_dir}/hidden.npy", np.array(hidden))
                with open(f"{out_dir}/weights.pkl", "wb") as f:
                    pickle.dump(model_weights, f)

    # ==================================================
    #      ANALYSIS

    if analysis:

        print("\nSTATISTICS ...")
        
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
            pickle.dump(weights_norm, f)

        W1 = weights_list[0]; norm1 = weights_norm[0]
        W2 = weights_list[1]; norm2 = weights_norm[1]

        # singular value decomposition of W1 for all snapshots
        U, S, V = np.linalg.svd(W1)

        # select dominant modes
        mode = np.argmax(S, axis=1)
        eval1 = S[np.arange(len(mode)), mode] #  np.take(S, mode, axis=1)
        Rvec1 = U[np.arange(len(mode)), mode] #  np.take(U, mode, axis=1)
        Lvec1 = V[np.arange(len(mode)), mode] #  np.take(V, mode, axis=1)

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
    
    print("\nPLOTTING ...")

    # re-load saved data
    saved_epochs = np.load(f"{out_dir}/saved_epochs.npy")
    train_loss = np.load(f"{out_dir}/train_loss.npy")
    test_acc = np.load(f"{out_dir}/test_loss.npy")
    hidden = np.load(f"{out_dir}/hidden.npy")
    with open(f"{out_dir}/weights_norm.pkl", "rb") as f:
        weights_norm = pickle.load(f)

    PR = np.load(f"{out_dir}/PR.npy")
    S = np.load(f"{out_dir}/eigenvalues.npy")
    eval1 = np.load(f"{out_dir}/eval.npy")
    Lvec1 = np.load(f"{out_dir}/Levec.npy")
    Rvec1 = np.load(f"{out_dir}/Revec.npy")

    title = f"init {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; N ={N:04d}; drop {drop_l} wp {drop_p:.2f}"
    colors = ['C0', 'C1', 'C2', 'C3']

    # PARTICIPATION RATIO AND LARGEST SINGULAR VALUE
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$\lambda_1$')
    ln = ax.plot(saved_epochs, eval1, c='C0', label=r"$\lambda_1$")
    ax1 = ax.twinx()
    ln1 = ax1.plot(saved_epochs, PR, c='C1', label="PR")
    ax1.set_ylabel('participation ratio')
    lns = ln+ln1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=9)
    fig.savefig(f'{out_dir}/plot_eval_PR.png', bbox_inches="tight")

    # SINGULAR VALUES DISTRIBUTION
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_xlabel('eigenvalue')
    ax.set_ylabel('density')
    ax.hist(S[0], density=True, bins="sqrt", label="initial", alpha=0.3)
    ax.hist(S[-1], density=True, bins="sqrt", label="trained", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(f'{out_dir}/plot_eval_distr.png', bbox_inches="tight")

    # TRAIN AND TEST LOSS
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label="train")
    ax.plot(test_acc, label="test")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylabel('Train and test loss')
    ax.set_xlabel('epoch')
    ax.legend(loc="best")
    fig.savefig(f'{out_dir}/plot_loss.png', bbox_inches="tight")

    # NORM OF THE WEIGHTS
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('L2 weight norm')
    ax.set_xlabel('epoch')
    ax.set_ylim([0,1])
    for i, (norm, c) in enumerate(zip(weights_norm, colors)):
        ax.plot(norm/norm[0], c=c, label=f'{i+1}: {norm[0]:.2e}')
    ax.legend(loc='best', title="layer: scale")
    fig.savefig(f'{out_dir}/plot_weights_norm.png', bbox_inches="tight")

    # VARIANCE OF THE HIDDEN LAYER
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('Hidden layer variance')
    ax.set_xlabel('epoch')
    ax.plot(saved_epochs, np.linalg.norm(hidden, axis=1))
    fig.savefig(f'{out_dir}/plot_hidden_layer_variance.png', bbox_inches="tight")

    # HISTOGRAM OF THE HIDDEN LAYER
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_xlabel('Hidden layer activity')
    ax.set_ylabel('density')
    ax.hist(hidden[0], density=True, bins="sqrt", label="initial", alpha=0.3)
    ax.hist(hidden[-1], density=True, bins="sqrt", label="trained", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(f'{out_dir}/plot_hidden_layer_histogram.png', bbox_inches="tight")

