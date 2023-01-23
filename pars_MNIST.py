import numpy as np
import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', dest='run', action='store_true', default=False, help='Run code serially')
args = parser.parse_args()
run = args.run

a_vals = ["linear", "relu"] # activation function for hidden units

s_vals = ["sqrt"] # initial parameters scaling

N_vals = [784] # number of units per hidden layer

l_vals = [2,3] # number of layers

p_vals = [0.5] # weight failure probability

with open("pars_MNIST.txt", "w") as file:
    for a in a_vals:
        for s in s_vals:
            for N in N_vals:
                for l in l_vals:
                    # no drop-out => d = 0
                    _pars = f"{a}\t{s}\t{N}\t{l}\t0.00\t0"
                    file.write(_pars+"\n")
                    if run:
                        os.system("python failures_MNIST.py "+_pars)
                    for p in p_vals:
                        # layers with failures
                        # f_vals = ["".join([str(j+1) for j in range(i+1)]) for i in range(l)]
                        f_vals = ["".join([str(i+1) for i in range(l)])]
                        for f in f_vals:
                        # if dropout, chose option for layers
                            _pars = f"{a}\t{s}\t{N}\t{l}\t{p:.2f}\t{f}"
                            file.write(_pars+"\n")
                            if run:
                                os.system("python failures_MNIST.py "+_pars)

