from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid', {'font_scale': 2})
import functools
import sys
import os


class LinearWeightDropout(nn.Linear):
    def __init__(self, in_features, out_features, drop_p=0.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.drop_p = drop_p

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        new_weight = (torch.rand((input.shape[0], *self.weight.shape), device=input.device) > self.drop_p) * self.weight[None, :, :]
        output = torch.bmm(new_weight, input[:, :, None])[:, :, 0] / (1 - self.drop_p)
        if self.bias is None:
            return output
        return output + self.bias


class Net(nn.Module):
    def __init__(self, N, layer_type=nn.Linear, scaling="sqrt", bias=False):
        super(Net, self).__init__()
        self.fc1 = layer_type(N, N, bias=bias)
        self.fc2 = nn.Linear(N, 1, bias=bias)

        torch.manual_seed(1871)

        if scaling == "lin":
            # initialisation of the weights -- N(1/n, 1/n)
            for name, pars in self.named_parameters():
                if "weight" in name:
                    f_in = 1.*pars.data.size()[1]
                    pars.data.normal_(1./f_in, 1./f_in)
        elif scaling == "sqrt":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            for name, pars in self.named_parameters():
                if "weight" in name:
                    f_in = 1.*pars.data.size()[1]
                    pars.data.normal_(0., 2./np.sqrt(f_in))
        else:
            raise ValueError(f"Invalid scaling option '{scaling}'\nChoose either 'sqrt' or 'lin'")

    def forward(self, x, hidden_layer=False):
        h = self.fc1(x)
        # h = F.relu(h)
        out = self.fc2(h)
        if hidden_layer:
            return out, h
        else:
            return out

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))
