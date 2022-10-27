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

        _th = 2*np.pi * np.random.rand(n_samples); _v = np.random.randn(n_samples, N)
        X = (np.cos(_th)**2)[:, None] * w_star[None,:] + (np.sin(_th)**2)[:,None] * _v
        y = np.sum(X * w_star[None,:], axis=1)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        print(f"w.shape = {w_star.shape}\t max(|w|) = {np.max(np.abs(w_star))}")
        print(f"X.shape = {X.shape}\t max(|X|) = {np.max(np.abs(X))}")
        print(f"y.shape = {y.shape}\t max(|y|) = {np.max(np.abs(y))}")

    def __len__ (self):
        return len(self.X)

    def __getitem__ (self, i):
        return self.X[i], self.y[i]