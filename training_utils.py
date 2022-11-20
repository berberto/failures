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


def train(model, device, train_loader, optimizer, epoch, log_interval=100, verbose=False):
    model.train()
    train_loss = 0.
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.clone().detach().float().to(device)
        y = y.clone().detach().float().view(-1,1).to(device)
        output = model(X)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss = F.mse_loss(output, y, reduction="sum")
        if epoch > 0:
            loss.backward()
            optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item() / len(train_loader.dataset)
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    hidden = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.clone().detach().float().to(device)
            y = y.clone().detach().float().view(-1,1).to(device)
            output, hidden_ = model.forward(X, hidden_layer=True)
            hidden_ = np.mean(hidden_.detach().cpu().numpy(), axis=0)
            hidden += hidden_/len(test_loader)
            test_loss += F.mse_loss(output, y, reduction="sum").item() / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    with torch.no_grad():
        model_weights = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_weights.append(pars.detach().cpu().numpy())
    return test_loss, model_weights, hidden

