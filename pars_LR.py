import numpy as np
import sys

s_vals = ["sqrt"] # initial parameters scaling

N_vals = [100] # number of units per hidden layer

l_vals = [2,3] # number of layers

d_vals = [1,2] # output dimension

p_vals = [0.5] # weight failure probability

with open("pars_LR.txt", "w") as file:
    for s in s_vals:
        for N in N_vals:
            for l in l_vals:
                for d in d_vals:
                    # no drop-out => f = 0
                    file.write(f"{s}\t{N}\t{l}\t{d}\t0.00\t0\n")
                    for p in p_vals:
                        # layers to apply failure
                        # f_vals = ["".join([str(j+1) for j in range(i+1)]) for i in range(l)]
                        f_vals = ["".join([str(i+1) for i in range(l)])]
                        for f in f_vals:
                        # if dropout, chose option for layers
                            file.write(f"{s}\t{N}\t{l}\t{d}\t{p:.2f}\t{f}\n")
