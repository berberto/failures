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

from theory import LinearNetwork


if __name__ == "__main__":

    training = True
    restart = True
    analysis = True
    plotting = True
    theory = True

    # n_epochs = 1000
    # n_skip = 5 # epochs to skip when saving data
    # sub_dir = "shortrun"
    n_epochs = 1000000
    n_skip = 2000 # epochs to skip when saving data
    sub_dir = "longrun"

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
        print(model)

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        if restart:
            # load the last model saved
            model.load( join(out_dir, f"model_trained"), device )
            # load info snapshots as they were previously saved
            saved_epochs = list(np.load( join(out_dir, "saved_epochs.npy")))
            train_loss = list( np.load( join(out_dir, "train_loss.npy")) )
            test_acc = list( np.load( join(out_dir, "test_loss.npy")) )
            covariance_train = list(np.load( join(out_dir, "covariance_train.npy") ))
            covariance_test = list(np.load( join(out_dir, "covariance_test.npy") ))
            hidden = [np.load( join(out_dir, f"hidden_{l+1}.npy")) for l in range(n_layers - 1)]
            model_weights = [np.load( join(out_dir, f"weights_{l+1}.npy")) for l in range(n_layers)]

            start_epoch = saved_epochs[-1] + 1

        else:
            # save the initial model
            model.save( join(out_dir, "model_init") )
            # initialise info snapshot as empty lists
            saved_epochs = []
            train_loss = []
            test_acc = []
            covariance_train = []
            covariance_test = []
            hidden = [np.array([]) for _ in range(n_layers - 1)]
            model_weights = [np.array([]) for _ in range(n_layers)]
            
            start_epoch = 0


        for epoch in range(start_epoch, n_epochs + 1):

            # collect statistics
            if epoch % n_skip == 0:
                model.save( join(out_dir, f"model_trained") )
                
                print(f"Ep {epoch}", end="\t")

                # test model on training data
                loss, _, _ = test(model, device, train_loader)
                print(f'Train loss: {loss:.6e}', end="\t")
                
                # test model on test data
                acc, model_weights_, hidden_ = test(model, device, test_loader)
                print(f'Test loss: {acc:.6e}')

                train_loss.append(loss)
                test_acc.append(acc)
                
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

            # train
            train(model, device, train_loader, optimizer, epoch, log_interval=1000)

    # ==================================================
    #      ANALYSIS

    # if analysis:

    #     print("STATISTICS ...")
        
    #     run_statistics(out_dir)


    # ==================================================
    #      PLOTS
    
    if plotting:

        print("PRE-PROCESSING...")
        print("\tLoading weights...", end=" ")
        weights_list = load_weights( out_dir )
        print("Done")

        print("\tCalculating weights norm...", end=" ")
        weights_norm = [np.linalg.norm(W, axis=(-1,-2)) for W in weights_list]
        with open(f"{out_dir}/weights_norm.pkl", "wb") as f:
            pickle.dump(weights_norm, f)
        print("Done")

        # singular value decomposition of W's for all snapshots
        print("\tCalculataing SVD of weights...")
        Us = []
        Ss = []
        Vs = []
        for l, W in enumerate(weights_list):
            # calculate the singular value decomposition of the weights
            # if len(W.shape) == 2:
            #     n, d = W.shape
            #     W = np.reshape(W, (n, 1, d))
            print(f"\t\tLayer {l+1}, {W.shape}", end=" ")
            U, S, Vh = np.linalg.svd(W)
            Us.append(U)
            Ss.append(S)
            Vs.append(Vh)
            print("Done")

        # saved_epochs, train_loss, test_loss, hidden, model_weights, \
        # covariance, covariance_train, covariance_test, \
        # weights_norm, (Us, Ss, Vs), projs = load_data(out_dir)

        print("\tImporting saved data...", end=" ")
        saved_epochs, train_loss, test_loss, hidden, model_weights, \
        covariance, covariance_train, covariance_test = load_data(out_dir)
        print("Done")

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
        print("\tCalculating product of all weights...", end=" ")
        W_product = model_weights[0]
        for l in range(1, n_layers):
            W_product = np.einsum('...ij,...jk->...ik', model_weights[l], W_product)
        np.save(f"{out_dir}/W_product.npy", W_product)
        print("Done")

        title = f"{activation}; 1/{scaling}; N={N:04d}; drop {drop_l} w/p {drop_p:.2f}" # ; L={n_layers}

        if theory and n_layers == 2 and activation == "linear":

            print("THEORY")

            # theory predictions
            ln = LinearNetwork(
                    [ weights_list[0][0], weights_list[1][0] ],
                    w_star,
                    q = 1 - drop_p,
                    # the timestep in gradient flow is the learning rate
                    # multiplied by the number of batches in an epoch
                    eta=lr * n_train/train_kwargs['batch_size'],
                    out_dir=join(out_dir, 'theory'))

            try:
                print("\t\tTrying to import existing simulations...", end=" ")
                weights_list_th = load_weights( join(out_dir, 'theory') )
                print("Done")
            except FileNotFoundError as e:
                print("Simulating anew...", end=" ")
                _, weights_list_th = ln.simulate(n_epochs, saved_steps=saved_epochs)
                print("Done")

            print("\t\tCalculataing SVD of weights from simulations...")
            Us_th = []
            Ss_th = []
            Vs_th = []
            for l, W in enumerate(weights_list_th):
                print(f"\t\tLayer {l+1}, {W.shape}", end=" ")
                U, S, Vh = np.linalg.svd(W)
                Us_th.append(U)
                Ss_th.append(S)
                Vs_th.append(Vh)
                print("Done")


        print("PLOTTING ...")

        print("\tcovariance...", end=" ")
        plot_covariance (covariance, IO=True, d_output=d_output, out_dir=out_dir, title=title, W_product=cov_Xy_test)
        print("Done")
        
        # print("\taligment of adjacent layers...", end=" ")
        # plot_alignment_layers (projs, d_output=d_output, epochs=saved_epochs, out_dir=out_dir, title=title)
        # print("Done")

        print("\talignment of SV with true weights...", end=" ")
        plot_alignment_wstar (model_weights, w_star, Us,Vs, epochs=saved_epochs, out_dir=out_dir, title=title)
        print("Done")

        print("\tsingular values...", end=" ")
        if theory and n_layers == 2 and activation == "linear":
            plot_singular_values (Ss, epochs=saved_epochs, theory=Ss_th, out_dir=out_dir, title=title)#, ext="svg", xlim=[0,100]) #, inset=[0,100]
        else:
            plot_singular_values (Ss, epochs=saved_epochs, out_dir=out_dir, title=title)#, ext="svg", xlim=[0,100]) #, inset=[0,100]
        print("Done")

        print("\ttrain and test loss/accuracy...", end=" ")
        try:
            # new version where testing is done only every `n_skip` epochs
            # -- throws an exception if testing done every epoch
            plot_loss_accuracy (train_loss, test_loss, test_epochs=saved_epochs, out_dir=out_dir, title=title) #, xlim=[0,20])
        except:
            plot_loss_accuracy (train_loss, test_loss, out_dir=out_dir, title=title) #, xlim=[0,20]) 
        print("Done")

        print("\tweights statistics...", end=" ")
        plot_weights (model_weights, weights_norm, epochs=saved_epochs, out_dir=out_dir, title=title)
        print("Done")

        print("\thidden activity statistics...", end=" ")
        plot_hidden_units (hidden, epochs=saved_epochs, out_dir=out_dir, title=title)
        print("Done")
