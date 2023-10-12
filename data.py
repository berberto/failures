import numpy as np
import torch

class LinearRegressionDataset(torch.utils.data.Dataset):
    '''
    data to be used through a data loader
    - N: dimensionality of the input
    - n_samples: number of samples in the dataset
    '''
    def __init__ (self, w_star, n_samples, cov=None):

        self.w = np.atleast_2d(w_star)
        self.d, self.N = self.w.shape
        if cov is None:
            cov = np.eye(self.N)
        assert len(cov) == self.N, "covariance matrix must have the same dimensions as the input vector"

        X = np.random.multivariate_normal(np.zeros(self.N), cov, size=n_samples)
        y = np.matmul(X, self.w.T)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    @property
    def data(self):
        return self.X

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, i):
        return self.X[i], self.y[i]


class SemanticsDataset (torch.utils.data.Dataset):

    # 0 - grow
    # 1 - move
    # 2 - roots
    # 3 - fly
    # 4 - swims
    # 5 - leaves
    # 6 - petals
    N = 7

    def __init__ (self, n_samples, cov=None):

        # input-output covariance matrix
        IO_cov = np.array([
                [1,1,0,1,0,0,0],
                [1,1,0,0,1,0,0],
                [1,0,1,0,0,1,0],
                [1,0,1,0,0,0,1]
            ]).astype(float).T

        # input-input covariance matrix
        if cov is None:
            cov = np.eye(self.N)

        # target input-output map
        self.w = np.dot( np.linalg.inv(cov), IO_cov ).T

        X = np.random.multivariate_normal(np.zeros(self.N), cov, size=n_samples)
        y = np.matmul(X, self.w.T)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    @property
    def data(self):
        return self.X

    @property
    def targets(self):
        return self.y

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, i):
        return self.X[i], self.y[i]


if __name__ == "__main__":
    # train_dataset = LinearRegressionDataset(np.ones(100), 10000)
    train_dataset = SemanticsDataset(10000)

    import matplotlib.pyplot as plt

    X_train = train_dataset.data
    y_train = train_dataset.targets
    d_input = X_train.shape[1]
    d_output = y_train.shape[1]

    cov = np.cov(X_train.T, y_train.T)
    cov_XX = cov[:d_input,:d_input]
    cov_Xy = cov[:d_input,-d_output:]
    cov_yy = cov[-d_output:,-d_output:]

    fig, ax = plt.subplots(1,3)
    kwargs = dict(vmin=-1, vmax=1, cmap="bwr", aspect="equal")
    im = ax[0].imshow(cov_XX, **kwargs); ax[0].set_title("Input-input cov")
    ax[1].imshow(cov_Xy, **kwargs); ax[1].set_title("Input-output cov")
    ax[2].imshow(cov_yy, **kwargs); ax[2].set_title("Output-output cov")
    fig.colorbar(im, ax=ax.ravel().tolist())
    fig.savefig("AS_cov.png")

    U, S, Vh = np.linalg.svd(train_dataset.w.T)

    fig, ax = plt.subplots(1,4,figsize=(10,3))
    kwargs = dict(vmin=-1, vmax=1, cmap="bwr", aspect="equal")
    im = ax[0].imshow(train_dataset.w.T, **kwargs); ax[0].set_title(r"$\Sigma^{xy}$")
    ax[1].imshow(-U[:,:d_output], **kwargs); ax[1].set_title(r"$U$")
    ax[2].imshow(np.diag(S), aspect="equal"); ax[2].set_title(r"$S$")
    for i,s in enumerate(S):
        ax[2].text(i,i,f"{s:.2f}", c='r',verticalalignment="center",horizontalalignment="center")
    ax[3].imshow(-Vh, **kwargs); ax[3].set_title(r"$V^T$")
    fig.colorbar(im, ax=ax.ravel().tolist())
    fig.savefig("AS_svd.png")
