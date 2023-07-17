import numpy as np
import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', dest='run', action='store_true', default=False, help='Run code serially')
args = parser.parse_args()
run = args.run

data_vals = ['CIFAR10']
# data_vals = ['MNIST']

a_vals = ["linear", "relu"] # activation function for hidden units

s_vals = ["sqrt", "lin"] # initial parameters scaling

N_vals = [100]    # number of units in hidden layers

l_vals = [2,3,10] # number of layers

p_vals = [0.5] # weight failure probability

with open("pars_classifier.txt", "w") as file:
    for data in data_vals:
        for a in a_vals:
            for s in s_vals:
                for N in N_vals:
                    for l in l_vals:
                        # no drop-out => d = 0
                        _pars = f"{data}  {a}  {s}  {N}  {l}  0.00  0"
                        file.write(_pars+"\n")
                        if run:
                            os.system("python failures_classifier.py "+_pars)
                        for p in p_vals:
                            # layers with failures
                            # f_vals = [",".join([str(j+1) for j in range(i+1)]) for i in range(l)]
                            f_vals = ["all"] # f_vals = [",".join([str(i+1) for i in range(l)])]
                            for f in f_vals:
                            # if dropout, chose option for layers
                                _pars = f"{data}  {a}  {s}  {N}  {l}  {p:.2f}  {f}"
                                file.write(_pars+"\n")
                                if run:
                                    os.system("python failures_classifier.py "+_pars)

