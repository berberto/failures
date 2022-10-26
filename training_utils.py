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


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.clone().detach().float().to(device)
        target = target.clone().detach().float().view(-1,1).to(device)
        optimizer.zero_grad()
        output = model(data)
        # "sum" or "mean" refers to the sum/average over both batch AND dimension indices
        # e.g. for a batch size of 64 and 10 classes, it is a sum/average of 640 numbers
        loss = F.mse_loss(output, target, reduction="mean")
        if epoch > 0:
            loss.backward()
            optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.clone().detach().float().to(device)
            target = target.clone().detach().float().view(-1,1).to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction="mean").item()# F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    with torch.no_grad():
        model_norm = []
        for name, pars in model.named_parameters():
            if "weight" in name:
                model_norm.append(torch.linalg.norm(pars).item())
    return (test_loss, ) + tuple(model_norm)
