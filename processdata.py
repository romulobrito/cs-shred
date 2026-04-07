import torch
from scipy.io import loadmat
import numpy as np
import scipy.linalg

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


class SpatialWindowDataset(torch.utils.data.Dataset):
    """Lazy dataset that builds (lags, H, W) spatial windows on-the-fly.

    Instead of pre-expanding all overlapping windows into a huge tensor,
    this dataset stores the volume (T, H, W) once and slices windows at
    access time.  Each item returns a 3-tuple:

        (x_spatial, y_target, t_start)

    where:
        x_spatial : (lags, H, W) float32 -- spatial frames for the window
        y_target  : (H*W,)       float32 -- target state at t_start + lags - 1
        t_start   : ()           long    -- absolute window start index (for cache)

    Parameters
    ----------
    volume_thw : np.ndarray
        Scaled + re-zeroed volume with shape ``(T, H, W)``.
    targets_flat : np.ndarray
        Target states (original, non-subsampled) with shape ``(T, H*W)``.
    lags : int
        Window length.
    indices : np.ndarray
        Window start indices for this split (e.g. train_indices).
    """

    def __init__(
        self,
        volume_thw: np.ndarray,
        targets_flat: np.ndarray,
        lags: int,
        indices: np.ndarray,
    ) -> None:
        if volume_thw.ndim != 3:
            raise ValueError("volume_thw must be 3-D (T, H, W)")
        if targets_flat.ndim != 2:
            raise ValueError("targets_flat must be 2-D (T, H*W)")
        self._volume = np.ascontiguousarray(volume_thw)
        self._targets = np.ascontiguousarray(targets_flat)
        self._lags = int(lags)
        self._indices = np.asarray(indices, dtype=np.int64)

    def __getitem__(self, idx: int):
        i = int(self._indices[idx])
        x = torch.tensor(
            self._volume[i : i + self._lags].copy(),
            dtype=torch.float32,
        )
        y = torch.tensor(
            self._targets[i + self._lags - 1].copy(),
            dtype=torch.float32,
        )
        t_start = torch.tensor(i, dtype=torch.long)
        return x, y, t_start

    def __len__(self) -> int:
        return len(self._indices)

def load_data(name):
    '''Takes string denoting data name and returns the corresponding (N x m) array 
    (N samples of m dimensional state)'''
    if name == 'SST':
        load_X = loadmat('Data/SST_data.mat')['Z'].T
        mean_X = np.mean(load_X, axis=0)
        sst_locs = np.where(mean_X != 0)[0]
        return load_X[:, sst_locs]

    if name == 'AO3':
        load_X = np.load('Data/short_svd_O3.npy')
        return load_X

    if name == 'ISO':
        load_X = np.load('Data/numpy_isotropic.npy').reshape(-1, 350*350)
        return load_X


def qr_place(data_matrix, num_sensors):
    '''Takes a (m x N) data matrix consisting of N samples of an m dimensional state and
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T'''
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox

