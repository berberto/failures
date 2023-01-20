import numpy as np
import pickle
import os


def run_statistics(out_dir):

    weights_list = [np.load(os.path.join(out_dir,file))  for file in os.listdir(out_dir) if "weights_" in file and ".npy" in file]

    weights_norm = [np.linalg.norm(W, axis=(-1,-2)) for W in weights_list]
    with open(f"{out_dir}/weights_norm.pkl", "wb") as f:
        pickle.dump(weights_norm, f)

    # singular value decomposition of W's for all snapshots
    Us = []
    Ss = []
    Vs = []
    for l, W in enumerate(weights_list):
        # calculate the singular value decomposition of the weights
        # if len(W.shape) == 2:
        #     n, d = W.shape
        #     W = np.reshape(W, (n, 1, d))
        print(f"Layer {l+1}, {W.shape}")
        U, S, Vh = np.linalg.svd(W)
        Us.append(U)
        Ss.append(S)
        Vs.append(Vh)

    pickle.dump( Us, open(f"{out_dir}/Us.pkl", "wb") )
    pickle.dump( Ss, open(f"{out_dir}/Ss.pkl", "wb") )
    pickle.dump( Vs, open(f"{out_dir}/Vs.pkl", "wb") )

    # calculate product between left and right modes in adjacent layers
    projs = []
    for l in range(len(weights_list) - 1):
        projs.append( np.einsum("...ij,...jk->...ik", Vs[l+1], Us[l]) )
    pickle.dump(projs, open(f"{out_dir}/projs.pkl", "wb"))


def load_statistics (out_dir):

    weights_norm = pickle.load( open(f"{out_dir}/weights_norm.pkl", "rb") )
    Us = pickle.load( open(f"{out_dir}/Us.pkl", "rb") )
    Ss = pickle.load( open(f"{out_dir}/Ss.pkl", "rb") )
    Vs = pickle.load( open(f"{out_dir}/Vs.pkl", "rb") )
    projs = pickle.load( open(f"{out_dir}/projs.pkl", "rb") )

    return weights_norm, (Us, Ss, Vs), projs