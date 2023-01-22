import numpy as np
import sys

s_vals = ["sqrt"]

N_vals = [784]

p_vals = [0.1, 0.5] # [0.1, 0.25, 0.50, 0.75]

l_vals = ["1", "12", "123"]

with open("job_array_pars.txt", "w") as f:
    for s in s_vals:
        for N in N_vals:
            # no drop-out => d = 0
            f.write(f"{s}\t{N} \t0.00\t0\n")
            for p in p_vals:
                # if dropout, chose option for layers
                for l in l_vals:
                    f.write(f"{s}\t{N} \t{p:.2f}\t{l}\n")
