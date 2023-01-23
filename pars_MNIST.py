import numpy as np
import sys

a_vals = ["linear", "relu"]

s_vals = ["sqrt"]

N_vals = [784]

l_vals = [2,3]

p_vals = [0.5]

with open("pars_MNIST.txt", "w") as file:
    for a in a_vals:
        for s in s_vals:
            for N in N_vals:
                for l in l_vals:
                    # no drop-out => d = 0
                    file.write(f"{a}\t{s}\t{N}\t{l}\t0.00\t0\n")
                    for p in p_vals:
                        # layers with failures
                        # f_vals = ["".join([str(j+1) for j in range(i+1)]) for i in range(l)]
                        f_vals = ["".join([str(i+1) for i in range(l)])]
                        for f in f_vals:
                        # if dropout, chose option for layers
                            file.write(f"{a}\t{s}\t{N}\t{l}\t{p:.2f}\t{f}\n")

