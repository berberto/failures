import numpy as np
import pickle


def run_statistics(out_dir):

    # get weights and calculate norm
    with open(f"{out_dir}/weights.pkl", "rb") as f:
        model_weights_ = pickle.load(f)
        weights_list = []
        weights_norm = []
        svd_list = []
        for i in range(len(model_weights_[0])):
            weights_list.append( np.array([w[i] for w in model_weights_]) )
            weights_norm.append( np.array([np.linalg.norm(w[i]) for w in model_weights_]) )

    with open(f"{out_dir}/weights_norm.pkl", "wb") as f:
        pickle.dump(weights_norm, f)

    w_star = np.load(f"{out_dir}/w_star.npy")

    # singular value decomposition of w_star
    Uw, Sw, Vw = np.linalg.svd(np.atleast_2d(w_star))
    with open(f"{out_dir}/SVDw.pkl", "wb") as f:
        pickle.dump([Uw, Sw, Vw], f)

    # singular value decomposition of W1 and W2 for all snapshots
    PR = []
    for i, W in enumerate(weights_list):
        # calculate the singular value decomposition of the weights
        # if len(W.shape) == 2:
        #     n, d = W.shape
        #     W = np.reshape(W, (n, 1, d))
        U, S, Vh = np.linalg.svd(W)
        # calculate the participation ratio of the singular values
        PR.append(np.array([np.sum(s)**2/np.sum(s**2) for s in S]))

        with open(f"{out_dir}/SVD{i+1}.pkl", "wb") as f:
            pickle.dump([U, S, Vh], f)

    np.save(f"{out_dir}/PR.npy", np.array(PR))