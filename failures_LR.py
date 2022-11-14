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


if __name__ == "__main__":

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    scaling = sys.argv[1]
    N = int(sys.argv[2])
    drop_p = float(sys.argv[3])

    # set (and create) output directory
    out_dir = "outputs_drop_full/"
    out_dir += f"init_{scaling}"
    out_dir += f"__N_{N:04d}"
    out_dir += f"__dropout_{drop_p:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING
    '''
    
    n_epochs = 20000
    lr = 1e-4
    wd = 0.

    batch_size = 100
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

    n_train = 1000
    n_test = 1000
    dataset1 = LinearRegressionDataset(N, n_train)
    dataset2 = LinearRegressionDataset(N, n_test)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # ==================================================
    #   TRAINING WITHOUT DROPOUT

    train_loss = []
    test_acc = []
    model_norms = []
    hidden_var = []
    model_weights = []

    model = Net(N, layer_type=nn.Linear, scaling=scaling, bias=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    model.save(f"{out_dir}/full_model_init")
    # plot_weights_histograms(model, out_dir=out_dir, name="full")

    print(model)

    for epoch in range(n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, model_weights_, hidden_var_ = test(model, device, test_loader)
        weight_norm = np.linalg.norm(model_weights_)
        if epoch % 100 == 0:
            model_weights.append(model_weights_)
        train_loss.append(loss)
        test_acc.append(acc)
        hidden_var.append(hidden_var_)
    model.save(f"{out_dir}/full_model_trained")
    np.save(f"{out_dir}/full_train_loss.npy", np.array(train_loss))
    np.save(f"{out_dir}/full_test_loss.npy", np.array(test_acc))
    np.save(f"{out_dir}/full_norm_weights.npy", np.array(model_norms).T)
    np.save(f"{out_dir}/full_hidden_var.npy", np.array(hidden_var))
    with open(f"{out_dir}/full_weights.pkl", "wb") as f:
        pickle.dump(model_weights, f)

    # ==================================================
    #   TRAINING WITH DROPOUT

    train_loss_p = []
    test_acc_p = []
    model_norms_p = []
    hidden_var_p = []
    model_weights_p = []

    model = Net(N, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), bias=False, scaling=scaling).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    model.save(f"{out_dir}/drop_model_init")
    plot_weights_histograms(model, out_dir=out_dir, name="drop")

    print(model)

    for epoch in range(n_epochs + 1):
        loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
        acc, model_weights_, hidden_var_ = test(model, device, test_loader)
        weight_norm = np.linalg.norm(model_weights_)
        if epoch % 100 == 0:
            model_weights_p.append(model_weights_)
        train_loss_p.append(loss)
        test_acc_p.append(acc)
        model_norms_p.append(weight_norm)
        hidden_var_p.append(hidden_var_)
    model.save(f"{out_dir}/drop_model_trained")
    np.save(f"{out_dir}/drop_train_loss.npy", np.array(train_loss_p))
    np.save(f"{out_dir}/drop_test_loss.npy", np.array(test_acc_p))
    np.save(f"{out_dir}/drop_norm_weights.npy", np.array(model_norms_p).T)
    np.save(f"{out_dir}/drop_hidden_var.npy", np.array(hidden_var_p))
    with open(f"{out_dir}/drop_weights.pkl", "wb") as f:
        pickle.dump(model_weights_p, f)

    # '''

    # ==================================================
    #      ANALYSIS
    '''

    with open(f"{out_dir}/full_weights.pkl", "rb") as f:
        model_weights_ = pickle.load(f)
        weights_list = []
        svd_list = []
        for i in range(len(model_weights_[0])):
            weights_list.append( np.array([np.squeeze(w[i]) for w in model_weights_]) )

    # print(type(weights_list), len(weights_list))
    # print(type(weights_list[0]), weights_list[0].shape)
    # print(type(weights_list[1]), weights_list[1].shape)

    W1 = weights_list[0]
    W2 = weights_list[1]

    U, S, V = np.linalg.svd(W1)

    # print(type(U), U.shape)
    # print(type(S), S.shape)
    # print(type(V), V.shape)

    IPR = np.array([np.sum(s**2)/np.sum(s)**2 for s in S])

    print(IPR)


    # with open(f"{out_dir}/drop_weights.pkl", "rb") as f:
    #     model_weights_p = pickle.load(f)

    '''

    # ==================================================
    #      PLOTS
    '''
    
    title = f"init: {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; N ={N:04d}"

    train_loss = np.load(f"{out_dir}/full_train_loss.npy")
    train_loss_p = np.load(f"{out_dir}/drop_train_loss.npy")
    test_acc = np.load(f"{out_dir}/full_test_loss.npy")
    test_acc_p = np.load(f"{out_dir}/drop_test_loss.npy")
    model_norms = np.load(f"{out_dir}/full_norm_weights.npy")
    model_norms_p = np.load(f"{out_dir}/drop_norm_weights.npy")
    hidden_var = np.load(f"{out_dir}/full_hidden_var.npy")
    hidden_var_p = np.load(f"{out_dir}/drop_hidden_var.npy")

    colors = ['C0', 'C1', 'C2', 'C3']

    fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax = plt.subplots()
    ax.plot(train_loss_p, label='p={}'.format(drop_p), ls="--")
    ax.plot(train_loss, label='standard')
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Training loss')
    ax.set_xlabel('epoch')
    fig.savefig(f'{out_dir}/plot_train_loss.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('L2 weight norm')
    ax.set_xlabel('epoch')
    ax1 = ax#.twinx()
    for i, (norm, norm_p, c) in enumerate(zip(model_norms, model_norms_p, colors)):
        ln = ax.plot(norm/norm[0], c=c, label='l={}, full'.format(i+1))
        ln1 = ax1.plot(norm_p/norm_p[0], c=c, label='l={}, p={}'.format(i+1, drop_p), ls="--")
    lns = ln+ln1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    fig.savefig(f'{out_dir}/plot_L2_weight_norm_fc1.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.set_ylabel('Hidden layer variance')
    ax.set_xlabel('epoch')
    ax1 = ax#.twinx()
    ln = ax.plot(hidden_var/hidden_var[0], c="C0", label='full')
    ln1 = ax1.plot(hidden_var_p/hidden_var_p[0], c="C1", label='p={}'.format(drop_p), ls="--")
    lns = ln+ln1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    fig.savefig(f'{out_dir}/plot_hidden_layer_variance.png', bbox_inches="tight")

    '''
