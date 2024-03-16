from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from collections import OrderedDict
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

        scaling_arr = scaling.split(",")
        assert len(scaling_arr) in [1, len(list(self.named_parameters()))], \
            "The `scaling` parameter must be a string with one of the available options, "+\
            "or multiple available options comma-separated (as many as the number of layers)"
        
        for l, (name, pars) in enumerate(self.named_parameters()):
            if len(scaling_arr) == 1:
                scaling = scaling_arr[0]
            else:
                scaling = scaling_arr[l]

            if "weight" in name:
                f_in = 1.*pars.data.size()[1]
                if scaling == "lin":
                    # initialisation of the weights -- N(0, 1/n)
                    init_f = lambda f_in: (0., 1./f_in)
                elif scaling == "lin+":
                    # initialisation of the weights -- N(0, 1/n)
                    init_f = lambda f_in: (1./f_in, 1./f_in)
                elif scaling == "sqrt":
                    # initialisation of the weights -- N(0, 1/sqrt(n))
                    init_f = lambda f_in: (0., 1./np.sqrt(f_in))
                elif scaling == "const":
                    # initialisation of the weights independent of n
                    init_f = lambda f_in: (0., 0.001)
                elif scaling == "const+":
                    # initialisation of the weights independent of n
                    init_f = lambda f_in: (0.001, 0.001)
                elif isinstance(scaling, float) and scaling > 0:
                    # initialisation of the weights -- N(0, 1/n**alpha)
                    '''
                    UNTESTED
                    '''
                    init_f = lambda f_in: (0., 1./np.power(f_in, scaling))
                else:
                    raise ValueError(
                        f"Invalid scaling option '{scaling}'\n" + \
                         "Choose either 'sqrt', 'lin' or a float larger than 0")

                mu, sigma = init_f(f_in)
                pars.data.normal_(mu, sigma)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def grad_dict (self):
        return OrderedDict({name:pars.grad for name, pars in self.named_parameters()})

    def __len__ (self):
        return len(self._modules.items())



class RNN (Net):

    def __init__(self, d_input, d_output, d_hidden=[100],
            output_activation = None, # for classification vs regression tasks
            drop_l=None,
            nonlinearity=F.tanh,
            layer_type=nn.Linear,
            bias=False,
        ):

        super(RNN, self).__init__()

        self.i2h = nn.Linear (d_input, d_hidden, bias=False)
        self.h2h = layer_type (d_hidden, d_hidden, bias=bias)
        self.h2o = nn.Linear (d_hidden, d_output, bias=bias)

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

        # convert drop_l into a list of strings
        if drop_l == None:
            drop_l = ""
        elif drop_l == "all":
            drop_l = ",".join([str(i+1) for i in range(self.n_layers)])
        drop_l = drop_l.split(",")

        self.init_weights (scaling)

    def forward (self, x, h0):
        '''
        x
        ---
        seq_length, batch_size, input_num_units
        or
        seq_length, input_num_units
        '''
        ht = h0
        hidden = ht
        output = torch.Tensor([])
        for t, xt in enumerate(x):
            z = self.i2h (xt)
            ht = self.h2h (ht)
            ht = self.phi (ht)
            output = torch.cat([output, self.h2o(ht)], dim=0)
            hidden = torch.cat([hidden, ht], dim=0)

        return hidden, output


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


def evaluate (model, device, loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(loader):
            X = X.clone().detach().float().to(device)
            output = model.forward(X).cpu().numpy()
            X = X.cpu().numpy()
            if batch_idx == 0:
                X_data = X
                y_data = output
            else:
                X_data = np.vstack([X_data, X])
                y_data = np.vstack([y_data, output])
    return X_data, y_data
