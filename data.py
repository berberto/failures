import numpy as np
import torch

class LinearRegressionDataset(torch.utils.data.Dataset):
    '''
    data to be used through a data loader
    - N: dimensionality of the input
    - n_samples: number of samples in the dataset
    '''
    def __init__ (self, w_star, n_samples):

        self.w = np.atleast_2d(w_star)
        self.d, self.N = self.w.shape

        X = np.random.randn(n_samples, self.N)
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


if __name__ == "__main__":
    data = LinearRegressionDataset(np.ones(100), 10000)

