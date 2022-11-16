import numpy as np
import torch

class LinearRegressionDataset(torch.utils.data.Dataset):
    '''
    data to be used through a data loader
    - N: dimensionality of the input
    - n_samples: number of samples in the dataset
    '''
    def __init__ (self, N, n_samples):
        w_star = np.ones(N)/np.sqrt(N)

        # _th = 2*np.pi * np.random.rand(n_samples); _v = 0.2*np.random.randn(n_samples, N)
        # X = (np.cos(_th)**2)[:, None] * w_star[None,:] + (np.sin(_th)**2)[:,None] * _v
        X = np.random.randn(n_samples, N)
        y = np.sum(X * w_star[None,:], axis=1)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.w = w_star

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, i):
        return self.X[i], self.y[i]


if __name__ == "__main__":
    data = LinearRegressionDataset(100, 10000)
