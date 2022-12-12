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
    '''
    Linear layer with weights dropout (synaptic failures)
    '''
    def __init__(self, in_features, out_features, drop_p=0.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.drop_p = drop_p

    def forward(self, input):
        new_weight = (torch.rand((input.shape[0], *self.weight.shape), device=input.device) > self.drop_p) * self.weight[None, :, :]
        output = torch.bmm(new_weight, input[:, :, None])[:, :, 0] / (1 - self.drop_p)
        if self.bias is None:
            return output
        return output + self.bias


class Net(nn.Module):
    '''
    Base class for network models.
    Attribute function `init_weights` is a custom weight initialisation.
    '''

    def init_weights (self, scaling):
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
                    pars.data.normal_(0., 1./np.sqrt(f_in))
        else:
            raise ValueError(f"Invalid scaling option '{scaling}'\nChoose either 'sqrt' or 'lin'")

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def __len__ (self):
        return len(self._modules.items())


class LinearNet2L(Net):
    '''
    Base class for feed-forward neural network models with linear layers by default,
    and with the optional argument `layer_type` to select alternative types of layers
    for specific layers (indicated in a string through `drop_l` optional argument).
    '''
    def __init__(self, N, d_output=1, layer_type=nn.Linear, scaling="sqrt", drop_l=None, bias=False):
        super(LinearNet2L, self).__init__()
        
        l1_type = nn.Linear
        l2_type = nn.Linear
        if drop_l is not None:
            if "1" in drop_l:
                l1_type = layer_type
            if "2" in drop_l:
                l2_type = layer_type

        self.fc1 = l1_type(N, N, bias=bias)
        self.fc2 = l2_type(N, d_output, bias=bias)

        self.init_weights (scaling)

    def forward(self, x, hidden_layer=False):
        h1 = self.fc1(x)
        out = self.fc2(h1)
        if hidden_layer:
            return out, [h1]
        else:
            return out

class LinearNet3L(Net):
    '''
    Base class for feed-forward neural network models with linear layers by default,
    and with the optional argument `layer_type` to select alternative types of layers
    for specific layers (indicated in a string through `drop_l` optional argument).
    '''
    def __init__(self, N, d_output=1, layer_type=nn.Linear, scaling="sqrt", drop_l=None, bias=False):
        super(LinearNet3L, self).__init__()
        
        l1_type = nn.Linear
        l2_type = nn.Linear
        l3_type = nn.Linear
        if drop_l is not None:
            if "1" in drop_l:
                l1_type = layer_type
            if "2" in drop_l:
                l2_type = layer_type
            if "3" in drop_l:
                l3_type = layer_type

        self.fc1 = l1_type(N, N, bias=bias)
        self.fc2 = l2_type(N, N, bias=bias)
        self.fc3 = l3_type(N, d_output, bias=bias)

        self.init_weights (scaling)

    def forward(self, x, hidden_layer=False):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        out = self.fc3(h2)
        if hidden_layer:
            return out, [h1,h2]
        else:
            return out

class ClassifierNet2L (LinearNet2L):
    '''
    Feed forward neural network with 2 fully connected hidden layers,
    relu non-linearity and softmax output for classification tasks.
    Adds the ReLU non-linearity to the layers specified as in the 
    LinearNet2L base class.
    '''
    def forward (self, x, hidden_layer=False):
        h1 = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(h1))
        if hidden_layer:
            return out, [h1]
        else:
            return out

class ClassifierNet3L (LinearNet3L):
    '''
    Feed forward neural network with 3 fully connected hidden layers,
    relu non-linearity and softmax output for classification tasks.
    Adds the ReLU non-linearity to the layers specified as in the 
    LinearNet3L base class.
    '''
    def forward (self, x, hidden_layer=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = F.softmax(self.fc3(h2))
        if hidden_layer:
            return out, [h1,h2]
        else:
            return out