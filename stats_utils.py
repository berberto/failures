import numpy as np
import pickle
import os
from os.path import join

def load_weights (out_dir):
    return [np.load(join(out_dir,file))  for file in os.listdir(out_dir) if "weights_" in file and ".npy" in file]


def run_statistics(out_dir):

    weights_list = load_weights( out_dir )

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


def load_data (out_dir):
    # re-load saved data
    saved_epochs = np.load( join(out_dir, "saved_epochs.npy") )
    train_loss = np.load( join(out_dir, "train_loss.npy") )
    test_loss = np.load( join(out_dir, "test_loss.npy") )
    model_weights = load_weights (out_dir)
    n_layers = len(model_weights)
    hidden = [np.load( join(out_dir, f"hidden_{l+1}.npy") ) for l in range(n_layers - 1)]
    covariance = np.load( join(out_dir, "covariance.npy") )
    covariance_train = np.load( join(out_dir, "covariance_train.npy") )
    covariance_test = np.load( join(out_dir, "covariance_test.npy") )

    # weights_norm, (Us, Ss, Vs), projs = load_statistics(out_dir)

    return saved_epochs, train_loss, test_loss, hidden, model_weights, \
            covariance, covariance_train, covariance_test #, \
            # weights_norm, (Us, Ss, Vs), projs


def diagonal_matrix (S, n: int, m: int):
    '''
    produces an (n, m) matrix, with main diagonal S

    If S is a (r,) array, with r = min(n,m), it produces the matrix (n,m) with main diagonal equal S

    If S is a (..., r) array, it produces an array (..., n,m).
    '''
    assert isinstance(n, int) and isinstance(m, int), "n and m must be integers"
    rank = min(n,m)
    assert S.shape[-1] == rank, "length of S must be equal to the minimum dimension"

    _S = S.reshape(-1, rank)
    diag_S = np.zeros((_S.shape[0], n, m))

    diag_S[:, :rank, :rank] = np.array([np.diag(s) for s in _S])
    diag_S = np.reshape( diag_S, (*S.shape[:-1],n,m) )

    return diag_S
