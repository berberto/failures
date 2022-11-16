import numpy as np
import sys

s_vals = ["lin ", "sqrt"]

N_vals = [1000]

p_vals = [0.1, 0.25, 0.50, 0.75]

d_vals = ["1", "12", "2"]

with open("job_array_pars.txt", "w") as f:
    for s in s_vals:
        for N in N_vals:
            # no drop-out => d = 0
            f.write(f"{s}\t{N} \t0.00\t0\n")
            for p in p_vals:
                # if dropout, chose option for layers
                for d in d_vals:
                    f.write(f"{s}\t{N} \t{p:.2f}\t{d}\n")
