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
            # initialisation of the weights -- N(0, 1/n)
            scaling_f = lambda f_in: 1./f_in
        elif scaling == "sqrt":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            scaling_f = lambda f_in: 1./np.sqrt(f_in)
        elif scaling == "const":
            # initialisation of the weights independent of n
            scaling_f = lambda f_in: 0.001
        elif isinstance(scaling, float) and scaling > 0:
            # initialisation of the weights -- N(0, 1/n**alpha)
            '''
            UNTESTED
            '''
            scaling_f = lambda f_in: 1./np.power(f_in, scaling)
        else:
            raise ValueError(
                f"Invalid scaling option '{scaling}'\n" + \
                 "Choose either 'sqrt', 'lin' or a float larger than 0")
        
        for name, pars in self.named_parameters():
            if "weight" in name:
                f_in = 1.*pars.data.size()[1]
                pars.data.normal_(0, scaling_f(f_in))

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
    def __init__(self, d_input, d_output=1, d_hidden=100, layer_type=nn.Linear, scaling="sqrt", drop_l=None, bias=False):
        super(LinearNet2L, self).__init__()
        
        l1_type = nn.Linear
        l2_type = nn.Linear
        if drop_l is not None:
            if "1" in drop_l:
                l1_type = layer_type
            if "2" in drop_l:
                l2_type = layer_type

        self.fc1 = l1_type(d_input, d_hidden, bias=bias)
        self.fc2 = l2_type(d_hidden, d_output, bias=bias)

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
    def __init__(self, d_input, d_output=1, d_hidden=100, layer_type=nn.Linear, scaling="sqrt", drop_l=None, bias=False):
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

        self.fc1 = l1_type(d_input, d_hidden, bias=bias)
        self.fc2 = l2_type(d_hidden, d_hidden, bias=bias)
        self.fc3 = l3_type(d_hidden, d_output, bias=bias)

        self.init_weights (scaling)

    def forward(self, x, hidden_layer=False):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        out = self.fc3(h2)
        if hidden_layer:
            return out, [h1,h2]
        else:
            return out

class DeepNet(Net):
    '''
    General deep feed forward class
    Optional arguments
    - `layer_type` specifies a particular type of layer (e.g. Dropout, DropConnect, etc)
    for layers in `drop_l`.
    - `drop_l` is a string containing comma-separated numbers of all the layers where
    `layer_type` has to be used.
    '''
    def __init__(self, d_input, d_output=1, d_hidden=[100],
                    activation=None,
                    output_activation = None,
                    layer_type=nn.Linear, drop_l=None,
                    scaling="sqrt",
                    bias=False):
        super(DeepNet, self).__init__()

        if activation in [None, 'linear']:
            self.phi = lambda x: x
        elif activation == 'relu':
            self.phi = lambda x: F.relu(x)
        elif activation == 'sigmoid':
            self.phi = lambda x: F.sigmoid(x)
        else:
            raise NotImplementedError("activation function " + \
                            f"\"{activation}\" not implemented")

        if output_activation in [None, 'linear']:
            self.out_phi = lambda x: x
        elif output_activation == 'softmax':
            self.out_phi = lambda x: F.softmax(x, dim=-1)
        else:
            raise NotImplementedError("output activation function " + \
                            f"\"{output_activation}\" not implemented")
        
        self.d_output = d_output
        self.n_layers = len(d_hidden) + 1
        self.layer_dims = [d_input] + d_hidden + [d_output]

        # convert drop_l into a list of strings
        if drop_l == None:
            drop_l = ""
        elif drop_l == "all":
            drop_l = ",".join([str(i+1) for i in range(self.n_layers)])
        drop_l = drop_l.split(",")

        self.layers = nn.ModuleList([layer_type(self.layer_dims[l], self.layer_dims[l+1], bias=bias) if str(l+1) in drop_l \
                                        else nn.Linear(self.layer_dims[l], self.layer_dims[l+1], bias=bias) \
                                        for l in range(self.n_layers)
                                    ])

        self.init_weights (scaling)

    def __len__ (self):
        return len(self.layers)

    def forward(self, x, hidden_layer=False):
        # ModuleList can act as an iterable, or be indexed using ints
        h = []
        for l, layer in enumerate(self.layers):
            x = self.phi(layer(x))
            if hidden_layer and l < self.n_layers - 1:
                h.append(x)
        x = self.out_phi(x)

        if hidden_layer:
            return x, h
        else:
            return x


class ClassifierNet2L (LinearNet2L):
    '''
    Feed forward neural network with 2 fully connected hidden layers,
    relu non-linearity and softmax output for classification tasks.
    Adds the ReLU non-linearity to the layers specified as in the 
    LinearNet2L base class.
    '''
    def forward (self, x, hidden_layer=False):
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.fc1(x))
        # out = F.softmax(self.fc2(h1), dim=1)
        out = self.fc2(h1)
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
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        # out = F.softmax(self.fc3(h2), dim=1)
        out = self.fc3(h2)
        if hidden_layer:
            return out, [h1,h2]
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
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        # out = F.softmax(self.fc3(h2), dim=1)
        out = self.fc3(h2)
        if hidden_layer:
            return out, [h1,h2]
        else:
            return out