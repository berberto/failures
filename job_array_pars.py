import numpy as np
import sys

s_vals = ["lin ", "sqrt"]

N_vals = [100, 1000]

p_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

with open("job_array_pars.txt", "w") as f:
    for s in s_vals:
        for N in N_vals:
            for p in p_vals:
                f.write(f"{s}\t{N} \t{p:.2f}\n")
                
