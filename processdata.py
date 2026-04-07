import torch
from scipy.io import loadmat
import numpy as np
import scipy.linalg


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Sensor windows (batch, lags, num_sensors) paired with full-state targets."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def load_data(name):
    """Load a named benchmark matrix as (N, m) with N samples and m state dims."""
    if name == "SST":
        load_X = loadmat("Data/SST_data.mat")["Z"].T
        mean_X = np.mean(load_X, axis=0)
        sst_locs = np.where(mean_X != 0)[0]
        return load_X[:, sst_locs]

    if name == "AO3":
        load_X = np.load("Data/short_svd_O3.npy")
        return load_X

    if name == "ISO":
        load_X = np.load("Data/numpy_isotropic.npy").reshape(-1, 350 * 350)
        return load_X


def qr_place(data_matrix, num_sensors):
    """QR-based sensor placement from an (m x N) state sample matrix."""
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox
