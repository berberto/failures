from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import functools
import sys
import os
from os.path import join
import numpy as np
import pickle

from networks import LinearWeightDropout, DeepNet
from training_utils import train_classifier as train
from training_utils import test_classifier as test
from training_utils import append

from stats_utils import run_statistics, load_statistics
from plot_utils import (plot_alignment_layers, plot_singular_values,
                        plot_loss_accuracy, plot_weights,
                        plot_hidden_units, plot_covariance)

if __name__ == "__main__":

    training = True
    analysis = True
    plotting = True

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    dataset_name = sys.argv[1]
    if dataset_name == "MNIST":
        dataset = datasets.MNIST
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10
    else:
        raise NotImplementedError(f"dataset \"{dataset_name}\" not implemented")
    activation = sys.argv[2]    # hidden layer activation function
    scaling = sys.argv[3]       # init pars scaling ("lin"=1/N or "sqrt"=1/sqrt(N))
    N = int(sys.argv[4])        # number of hidden units
    n_layers = int(sys.argv[5]) # number of layers (hidden + 1)
    drop_p = float(sys.argv[6]) # probability of weight drop
    if not drop_p:
        drop_l = None
    else:
        drop_l = sys.argv[7]    # layer(s) with dropout, combined in a string ("1", "12", "13" etc)

    # set (and create) output directory
    out_dir = join(f"outputs_{dataset_name}", f"{n_layers}L_{activation}")
    out_dir = join(out_dir, scaling, f"{drop_l}", f"q_{drop_p:.2f}")
    
    wd = 0.
    if drop_p == 0.:
        try:
            wd = float(sys.argv[8])
        except:
            pass
        out_dir = join(out_dir, f"wd_{wd:.5f}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory:\n\t{out_dir}\n")

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING

    n_epochs = 10000
    n_skip = 100  # epochs to skip when saving data

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
    #   DATASET

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
    train_dataset = dataset('data', train=True, # download=True,
                        transform=transform)
    test_dataset = dataset('data', train=False,
                        transform=transform)
    d_input = np.prod(train_dataset.data.shape[1:])  # shape of flattened input
    d_output = len(np.unique(train_dataset.targets)) # number of classes


    # ==================================================
    #   TRAINING/TESTING

    if training:

        print("\nTRAINING ...")


        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        train_data = torch.flatten(torch.tensor(train_dataset.data), start_dim=1).numpy()
        covariance_XX = np.cov(train_data.T)
        np.save( f"{out_dir}/covariance_XX.npy", covariance_XX )

        model = DeepNet(d_input=d_input, d_output=d_output,
                    d_hidden=(n_layers - 1)*[N],
                    layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p),
                    activation=activation, output_activation='linear',
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
            train_loss.append(train_loss_); train_acc.append(train_acc_)

            # test
            test_loss_, test_acc_, model_weights_, hidden_ = test(model, device, test_loader)
            # collect statistics
            if epoch % n_skip == 0:
                test_loss.append(test_loss_); test_acc.append(test_acc_)
                
                model.save(f"{out_dir}/model_trained")
                
                saved_epochs.append(epoch)
                np.save(f"{out_dir}/saved_epochs.npy", np.array(saved_epochs))
                np.save(f"{out_dir}/train_loss.npy", np.array([train_loss, train_acc]))
                np.save(f"{out_dir}/test_loss.npy", np.array([test_loss, test_acc]))
                
                for l in range(n_layers - 1):
                    hidden[l] = append(hidden[l], hidden_[l])
                    np.save( f"{out_dir}/hidden_{l+1}.npy", hidden[l] )
                for l in range(n_layers):
                    model_weights[l] = append(model_weights[l], model_weights_[l])
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
        covariance_XX = np.load( f"{out_dir}/covariance_XX.npy" )
        
        weights_norm, (Us, Ss, Vs), projs = load_statistics(out_dir)

        title = f"init {'1/N' if scaling == 'lin' else '1/sqrt(N)'}; L={n_layers}; N={N:04d}; drop {drop_l} wp {drop_p:.2f}"

        plot_covariance (covariance_XX, d_output=d_output, out_dir=out_dir, title=title)

        plot_alignment_layers (projs, d_output=d_output, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_singular_values (Ss, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_loss_accuracy (train_loss, test_loss, train_acc, test_acc, test_epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_weights (model_weights, weights_norm, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_hidden_units (hidden, epochs=saved_epochs, out_dir=out_dir, title=title)
