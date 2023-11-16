import os
from os.path import join

def get_path (base_dir, n_layers, activation, scaling, N, drop_l, drop_p, wd=0, sub_dir=None):
    # scaling:  init pars scaling ("lin"=1/N, "sqrt"=1/sqrt(N), or "const"=0.001)
    # activation:  hidden layer activation function
    # N:  number of input and hidden units
    # n_layers:  number of layers (hidden + 1)
    # d_output:  output dimension
    # drop_p:  probability of weight drop

    data_dir = base_dir
    data_dir = join(data_dir, f"{n_layers}L_{activation}", scaling)
    data_dir = join(data_dir, f"N_{N:04d}", f"{drop_l}", f"q_{drop_p:.2f}")
    if drop_p == 0.:
        data_dir = join(data_dir, f"wd_{wd:.5f}")
    if sub_dir is not None:
    	data_dir = join(data_dir, sub_dir)
    
    return data_dir