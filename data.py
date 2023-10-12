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
    data = LinearRegressionDataset(np.ones(100), 10000)

