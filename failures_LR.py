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
from os.path import join
import numpy as np
import pickle

from networks import LinearWeightDropout, DeepNet, evaluate
from training_utils import train_regressor as train
from training_utils import test_regressor as test
from training_utils import append
from data import LinearRegressionDataset, SemanticsDataset

from path_utils import get_path
from stats_utils import run_statistics, load_statistics, load_data, load_weights #, diagonal_matrix
from plot_utils import (plot_alignment_layers, plot_alignment_wstar,
                        plot_singular_values, plot_loss_accuracy,
                        plot_weights, plot_hidden_units,
                        plot_covariance)


if __name__ == "__main__":

    training = True
    analysis = True
    plotting = True

    n_epochs = 1000
    n_skip = 5 # epochs to skip when saving data
    sub_dir = "shortrun"
    # n_epochs = 1000000
    # n_skip = 2000 # epochs to skip when saving data
    # sub_dir = "longrun"

    # ==================================================
    #   SETUP PARAMETERS

    # get parameters as inputs
    scaling = sys.argv[1]       # init pars scaling ("lin"=1/N, "sqrt"=1/sqrt(N), or "const"=0.001)
    activation = sys.argv[2]    # hidden layer activation function
    N = int(sys.argv[3])        # number of input and hidden units
    n_layers = int(sys.argv[4]) # number of layers (hidden + 1)
    d_output = int(sys.argv[5]) # output dimension
    drop_p = float(sys.argv[6]) # probability of weight drop
    if not drop_p:
        drop_l = None
    else:
        drop_l = sys.argv[7]    # layer(s) with dropout, comma separated ("1", "1,2", "1,3" etc)

    wd = 0.
    if drop_p == 0.:
        try:
            wd = float(sys.argv[7])
        except:
            pass
    
    base_dir = "test_AS"

    out_dir = get_path(base_dir, n_layers, activation,
                       scaling, N, drop_l, drop_p,
                       wd=wd, sub_dir=sub_dir)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory:\n\t{out_dir}\n")

    # find device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # ==================================================
    #   SETUP TRAINING

    n_train = 10000
    n_test = 10000

    lr = 1e-3

    train_kwargs = {'batch_size': 1000}
    test_kwargs = {'batch_size': n_test}
    use_cuda = True
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # ==================================================
    #   DATASET

    np.random.seed(1871)
    
    # define torch dataset and dataloader
    train_dataset = SemanticsDataset(n_train)
    test_dataset = SemanticsDataset(n_test)

    w_star = train_dataset.w
    np.save( join(out_dir, "w_star.npy"), w_star )

    d_output, d_input = w_star.shape

    # ==================================================
    #   TRAINING/TESTING

    if training:

        print("\nTRAINING ...")

        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset,**test_kwargs)

        # calculate and save data covariances
        train_data = torch.flatten(train_dataset.data, start_dim=1).numpy()
        covariance = np.cov(train_data.T, train_dataset.targets.T)
        np.save( join(out_dir, "covariance.npy"), covariance )

        '''
        train network
        '''
        # define network model
        model = DeepNet(d_input=d_input, d_output=d_output, d_hidden=(n_layers - 1)*[N],
                    layer_type=functools.partial(LinearWeightDropout, drop_p=drop_p),
                    activation=activation, output_activation='linear',
                    bias=False, scaling=scaling, drop_l=drop_l).to(device)

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        model.save( join(out_dir, "model_init") )
        print(model)

        train_loss = []
        test_acc = []
        hidden = [np.array([]) for _ in range(n_layers - 1)]
        model_weights = [np.array([]) for _ in range(n_layers)]
        saved_epochs = []
        covariance_train = []
        covariance_test = []

        for epoch in range(n_epochs + 1):
            # train (except on the first epoch)
            loss = train(model, device, train_loader, optimizer, epoch, log_interval=1000)
            # test
            print(f"Epoch {epoch}:", end=" ")
            acc, model_weights_, hidden_ = test(model, device, test_loader)

            train_loss.append(loss)
            test_acc.append(acc)
            # collect statistics
            if epoch % n_skip == 0:
                model.save( join(out_dir, f"model_trained") )
                
                saved_epochs.append(epoch)
                np.save( join(out_dir, "saved_epochs.npy"), np.array(saved_epochs))
                np.save( join(out_dir, "train_loss.npy"), np.array(train_loss))
                np.save( join(out_dir, "test_loss.npy"), np.array(test_acc))

                # input-input, input-output, output-output covariances from the model
                train_inputs, train_outputs = evaluate(model, device, train_loader)
                covariance_train.append( np.cov(train_inputs.T, train_outputs.T) )
                np.save( join(out_dir, "covariance_train.npy"), np.array(covariance_train) )

                test_inputs, test_outputs = evaluate(model, device, test_loader)
                covariance_test.append( np.cov(test_inputs.T, test_outputs.T) )
                np.save( join(out_dir, "covariance_test.npy"), np.array(covariance_test) )
                
                for l in range(n_layers - 1):
                    hidden[l] = append(hidden[l], hidden_[l])
                    np.save( join(out_dir, f"hidden_{l+1}.npy"), hidden[l] )
                for l in range(n_layers):
                    model_weights[l] = append(model_weights[l], model_weights_[l])
                    np.save( join(out_dir, f"weights_{l+1}.npy"), model_weights[l] )


    # ==================================================
    #      ANALYSIS

    # if analysis:

    #     print("STATISTICS ...")
        
    #     run_statistics(out_dir)


    # ==================================================
    #      PLOTS
    
    if plotting:


        weights_list = load_weights( out_dir )

        weights_norm = [np.linalg.norm(W, axis=(-1,-2)) for W in weights_list]
        with open(f"{out_dir}/weights_norm.pkl", "wb") as f:
            pickle.dump(weights_norm, f)

        # singular value decomposition of W's for all snapshots
        Us = []
        Ss = []
        Vs = []
        for l, W in enumerate(weights_list):
            # calculate the singular value decomposition of the weights
            # if len(W.shape) == 2:
            #     n, d = W.shape
            #     W = np.reshape(W, (n, 1, d))
            print(f"Layer {l+1}, {W.shape}")
            U, S, Vh = np.linalg.svd(W)
            Us.append(U)
            Ss.append(S)
            Vs.append(Vh)

        print("PLOTTING ...")

        # saved_epochs, train_loss, test_loss, hidden, model_weights, \
        # covariance, covariance_train, covariance_test, \
        # weights_norm, (Us, Ss, Vs), projs = load_data(out_dir)

        saved_epochs, train_loss, test_loss, hidden, model_weights, \
        covariance, covariance_train, covariance_test = load_data(out_dir)

        w_star = np.load( join(out_dir, "w_star.npy") )
        d_output, d_input = w_star.shape

        cov_XX = covariance[:d_input,:d_input]
        cov_Xy = covariance[:d_input,-d_output:]
        cov_yy = covariance[-d_output:,-d_output:]
        cov_XX_train = covariance_train[:,:d_input,:d_input]
        cov_Xy_train = covariance_train[:,:d_input,-d_output:]
        cov_yy_train = covariance_train[:,-d_output:,-d_output:]
        cov_XX_test = covariance_test[:,:d_input,:d_input]
        cov_Xy_test = covariance_test[:,:d_input,-d_output:]
        cov_yy_test = covariance_test[:,-d_output:,-d_output:]

        # calculate the product of all matrices
        W_product = model_weights[0]
        for l in range(1, n_layers):
            W_product = np.einsum('...ij,...jk->...ik', model_weights[l], W_product)
        np.save(f"{out_dir}/W_product.npy", W_product)

        title = f"init {scaling}; L={n_layers}; N={N:04d}; drop {drop_l} wp {drop_p:.2f}"

        plot_covariance (covariance, IO=True, d_output=d_output, out_dir=out_dir, title=title, W_product=cov_Xy_test)
        
        # plot_alignment_layers (projs, d_output=d_output, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_alignment_wstar (model_weights, w_star, Us,Vs, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_singular_values (Ss, epochs=saved_epochs, out_dir=out_dir, title=title)#, ext="svg", xlim=[0,100]) #, inset=[0,100]

        try:
            # new version where testing is done only every `n_skip` epochs
            # -- throws an exception if testing done every epoch
            plot_loss_accuracy (train_loss, test_loss, test_epochs=saved_epochs, out_dir=out_dir, title=title) #, xlim=[0,20])
        except:
            plot_loss_accuracy (train_loss, test_loss, out_dir=out_dir, title=title) #, xlim=[0,20]) 

        plot_weights (model_weights, weights_norm, epochs=saved_epochs, out_dir=out_dir, title=title)

        plot_hidden_units (hidden, epochs=saved_epochs, out_dir=out_dir, title=title)
