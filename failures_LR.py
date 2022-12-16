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

from networks import LinearWeightDropout, LinearNet2L, LinearNet3L
from training_utils import train_regressor as train
from training_utils import test_regressor as test
from data import LinearRegressionDataset

from stats_utils import run_statistics


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

def generate_data (w_star, n_samples, **kwargs):
    dataset = LinearRegressionDataset(w_star, n_samples)
    loader = torch.utils.data.DataLoader(dataset,**kwargs)
    return loader


if __name__ == "__main__":

    training = True
    analysis = False
    plotting = False

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    scaling = sys.argv[1]       # init pars scaling ("lin"=1/N or "sqrt"=1/sqrt(N))
    N = int(sys.argv[2])        # number of input and hidden units
    drop_p = float(sys.argv[3]) # probability of weight drop
    if not drop_p:
        drop_l = None
    else:
        drop_l = sys.argv[4]        # layer(s) with dropout, combined in a string ("1", "12", "13" etc)

    d_output = 2
    n_layers = 2

    if n_layers == 2:
        Net = LinearNet2L
    elif n_layers == 3:
        Net = LinearNet3L
    else:
        raise ValueError(f"Invalid number of layers, {n_layers}")

    # set (and create) output directory
    out_dir = f"outputs_{n_layers}L_{d_output}d/"
    out_dir += f"{scaling}/"
    out_dir += f"N_{N:04d}/"
    out_dir += f"{drop_l}/"
    out_dir += f"q_{drop_p:.2f}"    
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory:\n\t{out_dir}\n")

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING

    n_epochs = 100000

    n_train = 100000
    n_test = 1000
    n_skip = min(100, n_epochs//100) # epochs to skip when saving data

    lr = 1e-5
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

        np.random.seed(1871)
        if d_output == 1:
            w_star = np.random.randn(N)
            w_star /= np.linalg.norm(w_star)
        elif d_output == 2:
            u_1 = np.array([1,1])/np.sqrt(2)
            u_2 = np.array([-1,1])/np.sqrt(2)
            v_1 = np.ones(N)/np.sqrt(N)
            v_2 = np.zeros(N); v_2[0] = 1; v_2[2] = -1; v_2 /= np.sqrt(2)
            w_star = 1. * u_1[:,None]*v_1[None,:] \
                   + .2 * u_2[:,None]*v_2[None,:]
        else:
            raise ValueError("invalid value of 'd_output'")
            
        np.save(f"{out_dir}/w_star.npy", w_star)
        test_loader = generate_data(w_star, n_test, **test_kwargs)
        train_loader = generate_data(w_star, n_train, **train_kwargs)

        model = Net(N, d_output=d_output, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), 
                    bias=False, scaling=scaling, drop_l=drop_l).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        model.save(f"{out_dir}/model_init")
        print(model)

        train_loss = []
        test_acc = []
        weights_norm = []
        hidden = []
        model_weights = []
        saved_epochs = []

        for epoch in range(n_epochs + 1):
            # train (except on the first epoch)
            loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
            # test
            acc, model_weights_, hidden_ = test(model, device, test_loader)

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
                with open(f"{out_dir}/hidden.pkl", "wb") as f:
                    pickle.dump(hidden, f)
                with open(f"{out_dir}/weights.pkl", "wb") as f:
                    pickle.dump(model_weights, f)

    # ==================================================
    #      ANALYSIS

    if analysis:

        print("STATISTICS ...")

        run_statistics(out_dir)
        

    # ==================================================
    #      PLOTS
    
    if plotting:
        print("PLOTTING ...")

        # re-load saved data
        saved_epochs = np.load(f"{out_dir}/saved_epochs.npy")
        train_loss = np.load(f"{out_dir}/train_loss.npy")
        test_acc = np.load(f"{out_dir}/test_loss.npy")
        with open(f"{out_dir}/hidden.pkl", "rb") as f:
            hidden = pickle.load(f)
        with open(f"{out_dir}/weights_norm.pkl", "rb") as f:
            weights_norm = pickle.load(f)
        with open(f"{out_dir}/SVDw.pkl", "rb") as f:
            Uw, Sw, Vw = pickle.load(f)
        with open(f"{out_dir}/SVD1.pkl", "rb") as f:
            U1, S1, V1 = pickle.load(f)
        with open(f"{out_dir}/SVD2.pkl", "rb") as f:
            U2, S2, V2 = pickle.load(f)
        with open(f"{out_dir}/SVD3.pkl", "rb") as f:
            U3, S3, V3 = pickle.load(f)

        PR = np.load(f"{out_dir}/PR.npy")

        title = f"init {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; N ={N:04d}; drop {drop_l} wp {drop_p:.2f}"
        colors = ['C0', 'C1', 'C2', 'C3']

        # ALIGNMENT
        V2U1 = np.einsum('...ij,...jk->...ik', V2, U1)
        V3U2 = np.einsum('...ij,...jk->...ik', V3, U2)
        U3Uw = np.einsum('...ij,...jk->...ik', U3, Uw.T)
        V1Vw = np.einsum('...ij,...jk->...ik', V1, Vw.T)
        kwargs=dict(cmap="bwr", vmin=-1, vmax=1, aspect='equal')
        fig, axs_ = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs_.ravel()
        # plt.subplots_adjust(wspace=0.4)
        plt.subplots_adjust(hspace=0.3)
        def plot_frame (frame):
            plt.cla()
            fig.suptitle(title+f" -- epoch {frame*n_skip}")
            ax = axs[0]
            ax.set_title(r"$V^n_3\cdot U^m_2$")
            ax.set_xlabel(r"$m$")
            ax.set_ylabel(r"$n$")
            im = ax.imshow(V3U2[frame, :d_output+2, :d_output+2], **kwargs)#; plt.colorbar(im, ax=ax)
            ax = axs[1]
            ax.set_title(r"$V^n_2\cdot U^m_1$")
            ax.set_xlabel(r"$m$")
            ax.set_ylabel(r"$n$")
            im = ax.imshow(V2U1[frame, :d_output+2, :d_output+2], **kwargs)#; plt.colorbar(im, ax=ax)
            ax = axs[2]
            ax.set_title(r"$V^n_1\cdot \tilde{V}^m$")
            ax.set_xlabel(r"$m$")
            ax.set_ylabel(r"$n$")
            im = ax.imshow(V1Vw[frame, :d_output+2, :d_output+2], **kwargs)#; plt.colorbar(im, ax=ax)
            # ax = axs[3]
            # ax.set_title(r"$U^n_3\cdot \tilde{U}^m$")
            # ax.set_xlabel("frame")
            # ax.set_ylabel(r"$n$")
            # ax.set_xticks(np.arange(3))
            # ax.set_yticks(np.arange(3))
            # im = ax.plot(U3Uw[:frame, 0])#; plt.colorbar(im, ax=ax)
        plot_frame(len(saved_epochs)-1)
        fig.savefig(f'{out_dir}/alignment.png', bbox_inches="tight")
        from matplotlib.animation import FuncAnimation
        duration=6
        frames=range(20)
        dt = duration*1000./20.
        ani = FuncAnimation(fig, plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
        ani.save(f'{out_dir}/alignment.gif')

        fig, axs_ = plt.subplots(1, 3, figsize=(14, 4))
        axs = axs_.ravel()
        # plt.subplots_adjust(wspace=0.4)
        plt.subplots_adjust(hspace=0.3)
        fig.suptitle(title)
        ax = axs[0]
        ax.set_ylim([0,1.1])
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|V^n_3\cdot U^m_2|$")
        dims = V3U2.shape
        for i in range(d_output+1): # range(dims[1]):
            for j in range(d_output+1): #range(dims[2]):
                c = "C0" if i == j else "C1"
                ax.plot(saved_epochs, np.abs(V3U2[:, i, j]), c=c)
        ax = axs[1]
        ax.set_ylim([0,1.1])
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|V^n_2\cdot U^m_1|$")
        dims = V2U1.shape
        for i in range(d_output+1): # range(dims[1]):
            for j in range(d_output+1): #range(dims[2]):
                c = "C0" if i == j else "C1"
                ax.plot(saved_epochs, np.abs(V2U1[:, i, j]), c=c)
        ax = axs[2]
        ax.set_ylim([0,1.1])
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|V^n_1\cdot \tilde{V}^m|$")
        dims = V1Vw.shape
        for i in range(d_output+1): # range(dims[1]):
            for j in range(d_output+1): #range(dims[2]):
                c = "C0" if i == j else "C1"
                ax.plot(saved_epochs, np.abs(V1Vw[:, i, j]), c=c)
        fig.savefig(f'{out_dir}/alignment_vs_epoch.png', bbox_inches="tight")

        # PARTICIPATION RATIO
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('singular values\nparticipation ratio')
        for i, pr in enumerate(PR[:-1]):
            ax.plot(saved_epochs, pr, label=f"{i+1}")
        ax.legend(loc="best", title="layer")
        fig.savefig(f'{out_dir}/plot_s-values_PR.png', bbox_inches="tight")
        plt.close(fig)

        # ALL SINGULAR VALUES
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        fig.suptitle(title)
        ax = axs[0]
        ax.set_title(r"$W_1$")
        ax.set_xlabel('epoch')
        ax.set_ylabel('singular value')
        for s in S1.T:
            ax.plot(saved_epochs, s)
        ax = axs[1]
        ax.set_title(r"$W_2$")
        ax.set_xlabel('epoch')
        ax.set_ylabel('singular value')
        for s in S2.T:
            ax.plot(saved_epochs, s)
        fig.savefig(f'{out_dir}/plot_s-values.png', bbox_inches="tight")
        plt.close(fig)

        # # BIMODALITY
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.set_title(title)
        # ax.set_xlabel(r'$W^{(2)}_j$')
        # ax.set_ylabel(r'$(W^{(1)}\cdot w^*)_j$')
        # ax.scatter(W2[-1], np.sum(W1[-1]*w_star[None,:], axis=1), alpha=0.5, s=.1)
        # fig.savefig(f'{out_dir}/plot_scatter_W.png', bbox_inches="tight")
        # plt.close(fig)

        # # COS OF ANGLE BETWEEN PRINCIPAL COMPOMENTS AND WEIGHTS
        # # (check low rank of W)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.set_title(title)
        # ax.set_xlabel('epoch')
        # ax.set_ylim([0,1.1])
        # ax.grid()
        # # ax.set_xscale('log')
        # ax.set_ylabel(r'$|\cos\theta(u,v)|$')
        # ax.plot(saved_epochs, np.abs(V1_dot_wst), c='C0', label=r'$w^*, v_1$')
        # ax.plot(saved_epochs, np.abs(V2_dot_wst), c='C0', label=r'$w^*, v_2$', ls="--")
        # ax.plot(saved_epochs, np.abs(U1_dot_w2), c='C1', label=r'$w_2, u_1$')
        # ax.plot(saved_epochs, np.abs(U2_dot_w2), c='C1', label=r'$w_2, u_2$', ls="--")
        # ax.legend(loc="upper right", title=r"$u, v$")
        # fig.savefig(f'{out_dir}/plot_evec_theta.png', bbox_inches="tight")
        # plt.close(fig)

        # SINGULAR VALUES DISTRIBUTION
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('singular value')
        ax.set_ylabel('density')
        ax.hist(S1[0], density=True, bins=30, label="initial", alpha=0.3)
        ax.hist(S1[-1], density=True, bins=30, label="trained", alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_s-values_distr.png', bbox_inches="tight")
        plt.close(fig)

        # TRAIN AND TEST LOSS
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(np.arange(len(test_acc)), test_acc, label="test", s=2, c="C1")
        ax.plot(train_loss, label="train", c="C0")
        ax.set_title(title)
        ax.grid()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel('Train and test loss')
        ax.set_xlabel('epoch')
        ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_loss.png', bbox_inches="tight")
        plt.close(fig)

        # NORM OF THE WEIGHTS
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_ylabel('L2 weight norm')
        ax.grid()
        # ax.set_xscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylim([0,1])
        for i, (norm, c) in enumerate(zip(weights_norm, colors)):
            ax.plot(saved_epochs, norm/norm[0], c=c, label=f'{i+1}: {norm[0]:.2f}')
        ax.legend(loc='best', title="layer: init value")
        fig.savefig(f'{out_dir}/plot_weights_norm.png', bbox_inches="tight")
        plt.close(fig)

        # HISTOGRAM OF THE WEIGHTS
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('L2 weight norm (trained)')
        ax.set_ylabel('density')
        ax.set_xlim([-1/np.sqrt(N),1/np.sqrt(N)])
        ax.hist(W1[-1].ravel(), density=True, bins=100, label="W1", alpha=0.3)
        ax.hist(W2[-1].ravel(), density=True, bins=100, label="W2", alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_weights_histogram.png', bbox_inches="tight")
        plt.close(fig)

        # VARIANCE OF THE HIDDEN LAYER
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_ylabel('Hidden layer variance')
        ax.set_xlabel('epoch')
        ax.grid()
        ax.plot(saved_epochs, np.linalg.norm(hidden, axis=1))
        fig.savefig(f'{out_dir}/plot_hidden_layer_variance.png', bbox_inches="tight")
        plt.close(fig)

        # HISTOGRAM OF THE HIDDEN LAYER
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('Hidden layer activity')
        ax.set_ylabel('density')
        ax.hist(hidden[-1,0], density=True, bins="sqrt", label="initial", alpha=0.3)
        ax.hist(hidden[-1,1], density=True, bins="sqrt", label="trained", alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_hidden_layer_histogram.png', bbox_inches="tight")
        plt.close(fig)

