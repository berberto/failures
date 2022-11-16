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
    loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.clone().detach().float().to(device)
        target = target.clone().detach().float().view(-1,1).to(device)
        optimizer.zero_grad()
        output = model(data)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss += F.mse_loss(output, target, reduction="mean")/len(train_loader)
    if epoch > 0:
        loss.backward()
        optimizer.step()
        # if verbose:
        #     if batch_idx % log_interval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    hidden_sq = 0
    hidden_mean = 0 
    with torch.no_grad():
        for data, target in test_loader:
            data = data.clone().detach().float().to(device)
            target = target.clone().detach().float().view(-1,1).to(device)
            output, hidden = model.forward(data, hidden_layer=True)
            test_loss += F.mse_loss(output, target, reduction="sum").item()# F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            hidden = hidden.detach().cpu().numpy()
            hidden_mean += np.mean(hidden, axis=0)
            hidden_sq += np.mean(hidden**2, axis=0)

            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    hidden_var = (hidden_sq - hidden_mean**2)/len(test_loader.dataset)
    hidden_var = np.mean(hidden_var)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    with torch.no_grad():
        model_weights = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_weights.append(pars.detach().cpu().numpy())
    return test_loss, model_weights, hidden_var

