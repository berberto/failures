from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import functools
import sys
import os


def train_regressor (model, device, train_loader, optimizer, epoch, log_interval=100, verbose=False):
    model.train()
    train_loss = 0.
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.clone().detach().float().to(device)
        y = y.clone().detach().float().to(device)
        output = model(X)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss = F.mse_loss(output, y, reduction="mean")
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item() / len(train_loader)
    return train_loss

def test_regressor (model, device, test_loader):
    model.eval()
    test_loss = 0
    n_layers = len(model)
    hidden = (n_layers - 1)*[0]
    with torch.no_grad():
        for X, y in test_loader:
            X = X.clone().detach().float().to(device)
            y = y.clone().detach().float().to(device)
            output, hidden_ = model.forward(X, hidden_layer=True)
            # mean hidden-layer activity
            for i, h_ in enumerate(hidden_):
                # mean over datapoints in batch and over batches
                hidden[i] += np.mean(h_.detach().cpu().numpy(), axis=0) / len(test_loader)                      # mean over batches
            # mean loss (MSE averaged over data points only)
            test_loss += F.mse_loss(output, y, reduction="mean").item() / len(test_loader)

    with torch.no_grad():
        model_weights = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_weights.append(pars.detach().cpu().numpy())
    return test_loss, model_weights, hidden


def train_classifier (model, device, train_loader, optimizer, epoch, log_interval=100, verbose=False, num_classes=10):
    model.train()
    train_loss = 0.
    train_acc = 0.
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.clone().detach().float().to(device)
        y = y.clone().detach().to(device)
        output = model(X)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        # loss = F.cross_entropy(output, y, reduction="sum")
        loss = F.mse_loss(output, F.one_hot(y, num_classes=num_classes).float(), reduction="sum")
        if epoch > 0:
            loss.backward()
            optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item() / len(train_loader.dataset)
        # mean accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_acc += pred.eq(y.view_as(pred)).sum().item() / len(train_loader.dataset)
    return train_loss, train_acc

def test_classifier (model, device, test_loader, num_classes=10):
    model.eval()
    test_loss = 0.
    n_layers = len(model)
    hidden = (n_layers - 1)*[0]
    test_acc = 0.
    with torch.no_grad():
        for X, y in test_loader:
            X = X.clone().detach().float().to(device)
            y = y.clone().detach().to(device)
            output, hidden_ = model.forward(X, hidden_layer=True)
            # mean hidden-layer activity
            for i, h_ in enumerate(hidden_):
                # mean over datapoints in batch and over batches
                hidden[i] += np.mean(h_.detach().cpu().numpy(), axis=0) / len(test_loader)                      # mean over batches
            # mean loss (MSE averaged over data points only)
            # test_loss += F.cross_entropy(output, y, reduction="sum").item() / len(test_loader.dataset)
            test_loss += F.mse_loss(output, F.one_hot(y, num_classes=num_classes).float(), reduction="sum").item() / len(test_loader.dataset)
            # mean accuracy
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_acc += pred.eq(y.view_as(pred)).sum().item() / len(test_loader.dataset)

    print('Test set: Average loss: {:.6e}'.format(test_loss))
    with torch.no_grad():
        model_weights = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_weights.append(pars.detach().cpu().numpy())
    return test_loss, test_acc, model_weights, hidden

def append (stack_1, stack_2):
    if len(stack_1) == 0:
        stack_1 = np.array([stack_2])
    else:
        stack_1 = np.append(stack_1, [stack_2], axis=0)
    return stack_1
