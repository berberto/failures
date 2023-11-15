import numpy as np
import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', dest='run', action='store_true', default=False, help='Run code serially')
args = parser.parse_args()
run = args.run

s_vals = ["sqrt,lin", "sqrt,lin+", "const", "const+", "lin", "lin+", "sqrt,const", "sqrt,const+"] # initial parameters scaling

a_vals = ["linear", "relu"] # initial parameters scaling

N_vals = [16, 64, 256]#, 1024] # number of units per hidden layer

l_vals = [2] # [2,3,5,10] # number of layers

d_vals = [4] # [1,2] # output dimension

p_vals = [0.5] # weight failure probability

with open("pars_LR.txt", "w") as file:
    for s in s_vals:
        for a in a_vals:
            for N in N_vals:
                for l in l_vals:
                    for d in d_vals:
                        # no drop-out => f = 0
                        _pars = f"{s}  {a}  {N}  {l}  {d}  0.00  0"
                        file.write(_pars+"\n")
                        if run:
                            os.system("python failures_LR.py "+_pars)
                        for p in p_vals:
                            # layers to apply failure
                            # f_vals = [",".join([str(j+1) for j in range(i+1)]) for i in range(l)]
                            f_vals = ["all"] # f_vals = [",".join([str(i+1) for i in range(l)])]
                            for f in f_vals:
                            # if dropout, chose option for layers
                                _pars = f"{s}  {a}  {N}  {l}  {d}  {p:.2f}  {f}"
                                file.write(_pars+"\n")
                                if run:
                                    os.system("python failures_LR.py "+_pars)
