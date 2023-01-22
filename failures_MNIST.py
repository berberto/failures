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

from networks import LinearWeightDropout
from networks import LinearNet2L, LinearNet3L
from networks import ClassifierNet2L, ClassifierNet3L
from training_utils import train_classifier as train
from training_utils import test_classifier as test

from stats_utils import run_statistics, load_statistics

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

    d_output = 10 # 10 digits in MNIST
    n_layers = 2

    if n_layers == 2:
        Net = ClassifierNet2L
        # Net = LinearNet2L
    elif n_layers == 3:
        Net = ClassifierNet3L
        # Net = LinearNet3L
    else:
        raise ValueError(f"Invalid number of layers, {n_layers}")

    # set (and create) output directory
    out_dir = f"outputs_MNIST/{n_layers}L_relu/"
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

    n_epochs = 1000
    n_skip = 10  # epochs to skip when saving data

    lr = 1e-4
    wd = 0.

    train_kwargs = {'batch_size': 100}
    test_kwargs = {'batch_size': 100}
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

        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(x))
                ])
        train_dataset = datasets.MNIST('data', train=True, #download=True,
                            transform=transform)
        test_dataset = datasets.MNIST('data', train=False,
                            transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        model = Net(d_input=28*28, d_output=10, d_hidden=N, layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p), 
                    bias=False, scaling=scaling, drop_l=drop_l).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        model.save(f"{out_dir}/model_init")
        print(model)

        train_loss = []; train_acc = []
        test_loss = []; test_acc = []
        hidden = [np.array([]) for _ in range(n_layers - 1)]
        model_weights = [np.array([]) for _ in range(n_layers)]
        saved_epochs = []

        for epoch in range(n_epochs + 1):
            # train (except on the first epoch)
            train_loss_, train_acc_ = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
            # test
            test_loss_, test_acc_, model_weights_, hidden_ = test(model, device, test_loader)

            train_loss.append(train_loss_); train_acc.append(train_acc_)
            test_loss.append(test_loss_); test_acc.append(test_acc_)
            # collect statistics
            if epoch % n_skip == 0:
                model.save(f"{out_dir}/model_trained")
                
                saved_epochs.append(epoch)
                np.save(f"{out_dir}/saved_epochs.npy", np.array(saved_epochs))
                np.save(f"{out_dir}/train_loss.npy", np.array([train_loss, train_acc]))
                np.save(f"{out_dir}/test_loss.npy", np.array([test_loss, test_acc]))
                
                for l in range(n_layers - 1):
                    hidden[l] = np.vstack((hidden[l], hidden_[l]))
                    np.save( f"{out_dir}/hidden_{l+1}.npy", hidden[l] )
                for l in range(n_layers):
                    model_weights[l] = np.vstack((model_weights[l], model_weights_[l]))
                    np.save( f"{out_dir}/weights_{l+1}.npy", model_weights[l] )

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
        train_loss, train_acc = np.load(f"{out_dir}/train_loss.npy")
        test_loss, test_acc = np.load(f"{out_dir}/test_loss.npy")
        hidden = [np.load( f"{out_dir}/hidden_{l+1}.npy" ) for l in range(n_layers - 1)]
        model_weights = [np.load( f"{out_dir}/weights_{l+1}.npy" ) for l in range(n_layers)]
        
        weights_norm, (Us, Ss, Vs), projs = load_statistics(out_dir)

        title = f"init {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; L={n_layers}; N={N:04d}; drop {drop_l} wp {drop_p:.2f}"
        colors = ['C0', 'C1', 'C2', 'C3']

        # ALIGNMENT
        kwargs=dict(vmin=0, vmax=1, aspect='equal') # cmap="bwr", vmin=-1, 
        cols=max(n_layers-1, 2)
        fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
        # plt.subplots_adjust(wspace=0.4)
        plt.subplots_adjust(hspace=0.3)
        def plot_frame (frame):
            plt.cla()
            fig.suptitle(title+f" -- epoch {frame*n_skip}")
            # plot alignment of intermediate layers
            for l, proj in enumerate(projs):
                ax = axs[l]
                ax.set_xlabel(r"$m$")
                ax.set_ylabel(r"$n$")
                ax.set_title(rf"$|V^n_{l+2}\cdot U^m_{l+1}$|")
                im = ax.imshow(np.abs(proj[frame, :d_output+2, :d_output+2]), **kwargs)#; plt.colorbar(im, ax=ax)

        plot_frame(len(saved_epochs)-1)
        fig.savefig(f'{out_dir}/alignment.png', bbox_inches="tight")
        from matplotlib.animation import FuncAnimation
        duration = 10 # in s
        frames = np.linspace(0,len(saved_epochs)-1,51).astype(int)
        dt = duration*1000./len(frames) # in ms
        ani = FuncAnimation(fig, plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
        ani.save(f'{out_dir}/alignment.gif')


        # fig, axs_ = plt.subplots(1, 3, figsize=(14, 4))
        # axs = axs_.ravel()
        # # plt.subplots_adjust(wspace=0.4)
        # plt.subplots_adjust(hspace=0.3)
        # fig.suptitle(title)
        # ax = axs[0]
        # ax.set_ylim([0,1.1])
        # ax.set_xlabel("epoch")
        # ax.set_ylabel(r"$|V^n_3\cdot U^m_2|$")
        # dims = V3U2.shape
        # for i in range(d_output+1): # range(dims[1]):
        #     for j in range(d_output+1): #range(dims[2]):
        #         c = "C0" if i == j else "C1"
        #         ax.plot(saved_epochs, np.abs(V3U2[:, i, j]), c=c)
        # ax = axs[1]
        # ax.set_ylim([0,1.1])
        # ax.set_xlabel("epoch")
        # ax.set_ylabel(r"$|V^n_2\cdot U^m_1|$")
        # dims = V2U1.shape
        # for i in range(d_output+1): # range(dims[1]):
        #     for j in range(d_output+1): #range(dims[2]):
        #         c = "C0" if i == j else "C1"
        #         ax.plot(saved_epochs, np.abs(V2U1[:, i, j]), c=c)
        # # ax = axs[2]
        # # ax.set_ylim([0,1.1])
        # # ax.set_xlabel("epoch")
        # # ax.set_ylabel(r"$|V^n_1\cdot \tilde{V}^m|$")
        # # dims = V1Vw.shape
        # # for i in range(d_output+1): # range(dims[1]):
        # #     for j in range(d_output+1): #range(dims[2]):
        # #         c = "C0" if i == j else "C1"
        # #         ax.plot(saved_epochs, np.abs(V1Vw[:, i, j]), c=c)
        # fig.savefig(f'{out_dir}/alignment_vs_epoch.png', bbox_inches="tight")

        # PARTICIPATION RATIO AND LARGEST SINGULAR VALUE
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('participation ratio / rank')
        ax.grid()
        PR = lambda S: np.array([np.sum(s)**2/np.sum(s**2)/len(s) for s in S])
        for l, S in enumerate(Ss):
            ax.plot(saved_epochs, PR(S), label=f"{l+1}")
        ax.legend(loc="best", title="layer")
        fig.savefig(f'{out_dir}/plot_s-values_PR.png', bbox_inches="tight")
        plt.close(fig)

        # ALL SINGULAR VALUES
        cols=n_layers
        fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
        if cols == 1:
            axs = [axs]
        fig.suptitle(title)
        for l, S in enumerate(Ss):
            ax = axs[l]
            ax.set_title(rf"$W_{l+1}$")
            ax.set_xlabel('epoch')
            ax.set_ylabel('singular value')
            for s in S.T:
                ax.plot(saved_epochs, s)
        fig.savefig(f'{out_dir}/plot_s-values.png', bbox_inches="tight")
        plt.close(fig)

        # SINGULAR VALUES DISTRIBUTION
        cols=n_layers-1
        fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
        if cols == 1:
            axs = [axs]
        fig.suptitle(title)
        for l, S in enumerate(Ss[:-1]):
            ax = axs[l]
            ax.set_title(rf"$W_{l+1}$")
            ax.set_xlabel('singular value')
            ax.set_ylabel('density')
            ax.hist(S[0], density=True, bins=30, label="initial", alpha=0.4)
            ax.hist(S[-1], density=True, bins=30, label="trained", alpha=0.4)
            ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_eval_distr.png', bbox_inches="tight")
        plt.close(fig)

        # # TRAIN AND TEST LOSS
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.scatter(np.arange(len(test_loss)), test_loss, label="test", s=2, c="C1")
        # # ax.plot(train_loss, label="train", c="C0")
        # ax.scatter(saved_epochs[::10], test_loss, label="test", s=2, c="C1")
        # ax.plot(saved_epochs[::10], train_loss, label="train", c="C0")
        # ax.set_title(title)
        # ax.grid()
        # # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_ylabel('Train and test loss')
        # ax.set_xlabel('epoch')
        # ax.legend(loc="best")
        # fig.savefig(f'{out_dir}/plot_loss.png', bbox_inches="tight")
        # plt.close(fig)

        # # TRAIN AND TEST ACCURACY
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.scatter(np.arange(len(test_acc)), test_acc, label="test", s=2, c="C1")
        # # ax.plot(train_acc, label="train", c="C0")
        # ax.scatter(saved_epochs[::10], test_acc, label="test", s=2, c="C1")
        # ax.plot(saved_epochs[::10], train_acc, label="train", c="C0")
        # ax.set_title(title)
        # ax.grid()
        # # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_ylabel('Train and test accuracy')
        # ax.set_xlabel('epoch')
        # ax.legend(loc="best")
        # fig.savefig(f'{out_dir}/plot_accuracy.png', bbox_inches="tight")
        # plt.close(fig)

        # NORM OF THE WEIGHTS
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_ylabel('L2 weight norm')
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        # ax.set_ylim([0,1])
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
        for l, W in enumerate(model_weights):
            ax.hist(W[-1].ravel(), density=True, bins=100, label=f"W_{l+1}", alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(f'{out_dir}/plot_weights_histogram.png', bbox_inches="tight")
        plt.close(fig)

        # VARIANCE OF THE HIDDEN LAYER
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_ylabel('Hidden layer norm')
        ax.set_xlabel('epoch')
        ax.grid()
        for l, h in enumerate(hidden):
            ax.plot(saved_epochs, np.linalg.norm(h, axis=1), label=f"X_{l+1}")
        ax.legend(loc="best", title="hidden layer")
        fig.savefig(f'{out_dir}/plot_hidden_layer_norm.png', bbox_inches="tight")
        plt.close(fig)

        # HISTOGRAM OF THE HIDDEN LAYER(S)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.set_xlabel('Hidden layer activity')
        ax.set_ylabel('density')
        for i, h in enumerate(hidden):
            ax.hist(h[-1], density=True, bins="sqrt", label=f"X_{l+1}", alpha=0.3)
        ax.legend(loc="best", title="hidden layer")
        fig.savefig(f'{out_dir}/plot_hidden_layer_histogram.png', bbox_inches="tight")
        plt.close(fig)

