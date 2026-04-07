import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import spgl1
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


# torch.cuda.empty_cache()
import pylops
from pylops.optimization.sparsity import spgl1
import torch

import warnings
import time as _time_module
warnings.filterwarnings("ignore", message="Linesearch failed with error 1")


# -----------------------------------------------------------------------
# SequentialBlockBatchSampler: cria batches de indices CONTIGUOS
# para que x[:, i, j] no forward seja uma serie temporal real.
# A ORDEM dos blocos eh embaralhada a cada epoca (shuffle_blocks=True),
# mas os indices DENTRO de cada bloco sao sempre sequenciais.
# -----------------------------------------------------------------------
class SequentialBlockBatchSampler(Sampler):
    """Batch sampler that yields contiguous blocks of indices.

    This ensures each batch contains temporally consecutive samples,
    which is critical for CS recovery in the forward pass (the batch
    dimension becomes a valid temporal dimension for FFT-based recovery).

    The ORDER of blocks is shuffled each epoch to aid training convergence,
    but the indices WITHIN each block remain sequential.

    Args:
        data_source_length (int): Total number of samples in the dataset.
        batch_size (int): Number of samples per batch (= temporal window for CS).
        shuffle_blocks (bool): Whether to shuffle the order of blocks each epoch.
        generator (torch.Generator, optional): RNG for reproducible shuffling.
    """

    def __init__(self, data_source_length, batch_size, shuffle_blocks=True, generator=None):
        self.data_source_length = data_source_length
        self.batch_size = batch_size
        self.shuffle_blocks = shuffle_blocks
        self.generator = generator

    def __iter__(self):
        # Create contiguous blocks: [0..B-1], [B..2B-1], ...
        blocks = []
        for start in range(0, self.data_source_length, self.batch_size):
            end = min(start + self.batch_size, self.data_source_length)
            blocks.append(list(range(start, end)))

        # Shuffle the ORDER of blocks (not the indices within each block)
        if self.shuffle_blocks:
            if self.generator is not None:
                perm = torch.randperm(len(blocks), generator=self.generator)
            else:
                perm = torch.randperm(len(blocks))
            blocks = [blocks[i] for i in perm]

        for block in blocks:
            yield block

    def __len__(self):
        return (self.data_source_length + self.batch_size - 1) // self.batch_size


def forward_csshred_contiguous_batches(model, x_tensor, batch_size):
    """Run CS-SHRED forward in contiguous temporal batches.

    The batch dimension is the CS time axis for column-wise recovery; inference
    must use the same contiguous blocking as ``SequentialBlockBatchSampler`` in
    training. ``x_tensor`` may live on CPU; each chunk is passed to ``model``
    which moves tensors as needed.
    """
    n = int(x_tensor.shape[0])
    if n <= 0:
        return model(x_tensor)
    bs = max(1, int(batch_size))
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            chunks.append(model(x_tensor[start:end]))
    return torch.cat(chunks, dim=0)


# -----------------------------------------------------------------------
# Rastreador de estatisticas do Compressed Sensing (CS)
# Acumula metricas por forward pass e imprime resumo consolidado.
# -----------------------------------------------------------------------
_cs_stats = {
    "recovered": 0,
    "skipped": 0,
    "failed": 0,
    "forward_calls": 0,
    "cumulative_recovered": 0,
    "cumulative_skipped": 0,
    "cumulative_failed": 0,
}


###################################### CS-SHRED
def recover_signal(
    x,
    l1_precision,
    opt_tol,
    ls_tol,
    n_sparsity_threshold,
    verbosity,
    observed_rel_error_max=0.30,
    enable_fallback=True,
):
    """Recover a sparse signal using SPGL1 (Basis Pursuit via FFT).

    If the percentage of exact zeros in x exceeds n_sparsity_threshold,
    the function attempts CS recovery. Otherwise returns x unchanged.

    Args:
        x (torch.Tensor): 1-D input signal (may contain zeros from subsampling).
        l1_precision (float): BP tolerance for SPGL1.
        opt_tol (float): Optimality tolerance for SPGL1.
        ls_tol (float): Line-search tolerance for SPGL1.
        n_sparsity_threshold (float): Minimum fraction of zeros to trigger CS.
        verbosity (int): SPGL1 verbosity level.

    Returns:
        torch.Tensor: Recovered (or original) signal.
    """
    x_np = x.cpu().numpy()

    n_sparse = np.where(x_np == 0)[0]
    size = x_np.size
    percentage = len(n_sparse) / size

    if percentage <= n_sparsity_threshold:
        _cs_stats["skipped"] += 1
        return torch.tensor(x_np)

    else:
        try:
            iava = np.nonzero(x_np != 0)[0]
            Rop = pylops.Restriction(x.numel(), iava=iava, dtype="float64")
            y = Rop * x_np

            Fop = pylops.signalprocessing.FFT(x.numel(), dtype="complex128")
            Op = Rop * Fop.H

            # Adaptative: adjusts iter_lim based on tolerances
            if opt_tol > 1e-4 or ls_tol > 1e-4:
                iter_lim = 1000  # Relaxed tolerances = less iterations
            elif opt_tol > 1e-5 or ls_tol > 1e-5:
                iter_lim = 2000  # Medium tolerances = medium iterations
            else:
                iter_lim = 4000  # Rigid tolerances = more iterations

            x_recovered, _, _ = spgl1(
                Op,
                y,
                verbosity=verbosity,
                iter_lim=iter_lim,
                opt_tol=opt_tol,
                bp_tol=l1_precision,
                ls_tol=ls_tol,
                show=False,
            )

            recovered_signal_time = Fop.H * x_recovered
            # Extrair parte real (a inversa da FFT pode ter residuo imaginario numerico)
            recovered_signal_time = np.real(recovered_signal_time).reshape(x.shape)
            if not np.all(np.isfinite(recovered_signal_time)):
                _cs_stats["failed"] += 1
                return torch.tensor(x_np)

            if enable_fallback:
                observed_mask = x_np != 0
                if np.any(observed_mask):
                    denom = np.linalg.norm(x_np[observed_mask]) + 1e-12
                    rel_error_on_observed = np.linalg.norm(
                        recovered_signal_time[observed_mask] - x_np[observed_mask]
                    ) / denom
                    if rel_error_on_observed > observed_rel_error_max:
                        _cs_stats["failed"] += 1
                        return torch.tensor(x_np)

            recovered_signal_tensor = torch.tensor(recovered_signal_time)

            _cs_stats["recovered"] += 1
            return recovered_signal_tensor

        except Exception as e:
            _cs_stats["failed"] += 1
            return torch.tensor(x_np)  # Return original signal if recovery fails


def recover_signal_dct(
    x,
    l1_precision,
    opt_tol,
    ls_tol,
    n_sparsity_threshold,
    verbosity,
    observed_rel_error_max=0.30,
    enable_fallback=True,
):
    """Sparse recovery with SPGL1 using DCT-II synthesis (same gate as recover_signal).

    Forward model: measurements y = R @ D^H @ alpha; recover alpha, then x = D^H @ alpha.
    """
    x_np = x.cpu().numpy()

    n_sparse = np.where(x_np == 0)[0]
    size = x_np.size
    percentage = len(n_sparse) / size

    if percentage <= n_sparsity_threshold:
        _cs_stats["skipped"] += 1
        return torch.tensor(x_np)

    try:
        iava = np.nonzero(x_np != 0)[0]
        Rop = pylops.Restriction(x.numel(), iava=iava, dtype="float64")
        y = Rop * x_np

        Dop = pylops.signalprocessing.DCT(x.numel(), type=2, dtype="float64")
        Op = Rop * Dop.H

        if opt_tol > 1e-4 or ls_tol > 1e-4:
            iter_lim = 1000
        elif opt_tol > 1e-5 or ls_tol > 1e-5:
            iter_lim = 2000
        else:
            iter_lim = 4000

        x_recovered, _, _ = spgl1(
            Op,
            y,
            verbosity=verbosity,
            iter_lim=iter_lim,
            opt_tol=opt_tol,
            bp_tol=l1_precision,
            ls_tol=ls_tol,
            show=False,
        )

        recovered_signal_time = Dop.H * x_recovered
        recovered_signal_time = np.asarray(recovered_signal_time, dtype=np.float64).reshape(
            x.shape
        )
        if not np.all(np.isfinite(recovered_signal_time)):
            _cs_stats["failed"] += 1
            return torch.tensor(x_np)

        if enable_fallback:
            observed_mask = x_np != 0
            if np.any(observed_mask):
                denom = np.linalg.norm(x_np[observed_mask]) + 1e-12
                rel_error_on_observed = np.linalg.norm(
                    recovered_signal_time[observed_mask] - x_np[observed_mask]
                ) / denom
                if rel_error_on_observed > observed_rel_error_max:
                    _cs_stats["failed"] += 1
                    return torch.tensor(x_np)

        recovered_signal_tensor = torch.tensor(recovered_signal_time)

        _cs_stats["recovered"] += 1
        return recovered_signal_tensor

    except Exception:
        _cs_stats["failed"] += 1
        return torch.tensor(x_np)


def _lags_vector_cache_key(basis: str, vec_f32: np.ndarray) -> bytes:
    return ("%s|" % basis).encode("ascii") + np.ascontiguousarray(
        vec_f32, dtype=np.float32
    ).tobytes()


class _LagVectorLruCache:
    """LRU cache for 1D recovered vectors (bytes key -> numpy 1D float64)."""

    def __init__(self, max_entries: int) -> None:
        self._max = int(max_entries)
        self._data: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def get(self, key: bytes):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: bytes, value: np.ndarray) -> None:
        self._data[key] = np.asarray(value, dtype=np.float64).copy()
        self._data.move_to_end(key)
        if self._max <= 0:
            return
        while len(self._data) > self._max:
            self._data.popitem(last=False)


def recover_signal_per_column(
    x,
    l1_precision,
    opt_tol,
    ls_tol,
    n_sparsity_threshold,
    verbosity,
    cs_force_bypass=False,
    global_zero_gate=0.0,
    observed_rel_error_max=0.30,
    enable_fallback=True,
):
    """Recover signals for each (lag, sensor) pair in the input tensor.

    Iterates over all lag/sensor combinations, applies CS recovery where
    applicable, and prints a one-line summary per forward pass.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, lags, sensors).
        l1_precision (float): BP tolerance for SPGL1.
        opt_tol (float): Optimality tolerance for SPGL1.
        ls_tol (float): Line-search tolerance for SPGL1.
        n_sparsity_threshold (float): Minimum fraction of zeros to trigger CS.
        verbosity (int): SPGL1 verbosity level.

    Returns:
        list[torch.Tensor]: List of recovered signal tensors.
    """
    # Reset per-forward-pass counters
    _cs_stats["recovered"] = 0
    _cs_stats["skipped"] = 0
    _cs_stats["failed"] = 0
    t_start = _time_module.time()

    recovered_signals = []
    total_vectors = x.shape[1] * x.shape[2]
    gate_bypass = False
    if not cs_force_bypass:
        global_zero_fraction = torch.mean((x == 0).float()).item()
        gate_bypass = global_zero_fraction <= global_zero_gate

    if cs_force_bypass or gate_bypass:
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                recovered_signals.append(torch.tensor(x[:, i, j].detach().cpu().numpy()))
        _cs_stats["skipped"] = total_vectors
    else:
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                column_data = x[:, i, j]
                recovered_signal = recover_signal(
                    column_data,
                    l1_precision,
                    opt_tol,
                    ls_tol,
                    n_sparsity_threshold,
                    verbosity,
                    observed_rel_error_max=observed_rel_error_max,
                    enable_fallback=enable_fallback,
                )
                recovered_signals.append(recovered_signal)

    # Update cumulative counters
    _cs_stats["forward_calls"] += 1
    _cs_stats["cumulative_recovered"] += _cs_stats["recovered"]
    _cs_stats["cumulative_skipped"] += _cs_stats["skipped"]
    _cs_stats["cumulative_failed"] += _cs_stats["failed"]

    # Print one-line consolidated summary
    elapsed = _time_module.time() - t_start
    total_vecs = _cs_stats["recovered"] + _cs_stats["skipped"] + _cs_stats["failed"]
    cum_total = (
        _cs_stats["cumulative_recovered"]
        + _cs_stats["cumulative_skipped"]
        + _cs_stats["cumulative_failed"]
    )
    print(
        f"[CS #{_cs_stats['forward_calls']:>4d}] "
        f"recuperados={_cs_stats['recovered']}/{total_vecs} "
        f"pulados={_cs_stats['skipped']} "
        f"falhas={_cs_stats['failed']} "
        f"t={elapsed:.1f}s "
        f"| acum: rec={_cs_stats['cumulative_recovered']} "
        f"skip={_cs_stats['cumulative_skipped']} "
        f"fail={_cs_stats['cumulative_failed']} "
        f"total={cum_total} "
        f"| bypass={'yes' if (cs_force_bypass or gate_bypass) else 'no'}"
    )

    return recovered_signals


class CSSHRED(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
        l1_tol=1e-3,
        opt_tol=1e-4, 
        ls_tol=1e-4,
        n_sparsity_threshold=0.75,   
        cs_warmup_epochs=0,
        cs_global_zero_gate=0.0,
        cs_observed_rel_error_max=0.30,
        cs_enable_fallback=True,
        verbosity=-1,
        show_plot=False,
    ):
        super(CSSHRED, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.l1_tol = l1_tol
        self.opt_tol = opt_tol 
        self.ls_tol = ls_tol
        self.n_sparsity_threshold = n_sparsity_threshold
        self.cs_warmup_epochs = int(cs_warmup_epochs)
        self.cs_global_zero_gate = float(cs_global_zero_gate)
        self.cs_observed_rel_error_max = float(cs_observed_rel_error_max)
        self.cs_enable_fallback = bool(cs_enable_fallback)
        self.current_epoch = 0
        self.verbosity_spgl1 = verbosity
        self.show_plot = show_plot

    def set_training_progress(self, epoch, total_epochs):
        if total_epochs <= 0:
            self.current_epoch = 0
            return
        self.current_epoch = int(epoch)

    def forward(self, x):
        cs_force_bypass = (
            self.cs_warmup_epochs > 0 and self.current_epoch <= self.cs_warmup_epochs
        )

        recovered_signals_per_column = recover_signal_per_column(
            x,
            self.l1_tol,
            self.opt_tol,
            self.ls_tol,
            self.n_sparsity_threshold,
            self.verbosity_spgl1,
            cs_force_bypass=cs_force_bypass,
            global_zero_gate=self.cs_global_zero_gate,
            observed_rel_error_max=self.cs_observed_rel_error_max,
            enable_fallback=self.cs_enable_fallback,
        )
        combined_recovered_signal = torch.stack(recovered_signals_per_column, dim=1)
        combined_recovered_signal_expanded = combined_recovered_signal.unsqueeze(-1)
        combined_recovered_signal_expanded = combined_recovered_signal_expanded.squeeze(
            -1
        )
        combined_recovered_signal_expanded = (
            combined_recovered_signal_expanded.permute(0, 2, 1)
            if len(combined_recovered_signal_expanded.shape) > 2
            else combined_recovered_signal_expanded.permute(0, 1)
        )
        combined_recovered_signal_expanded = combined_recovered_signal_expanded.float()
        if self.show_plot:
            num_columns = x.size(1)
            num_channels = x.size(2)
            plt.plot(
                x[:, 0, 0].detach().cpu().numpy(),
                label=f"Subsampled Signal (Column {0}, Channel {0})",
                color="red", linewidth=5
            
            )
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.legend()
            # plt.grid(False)
            # plt.show()
            plt.plot(
                x[:, 0, 1].detach().cpu().numpy()+1,
                label=f"Subsampled Signal (Column {0}, Channel {1})",
                color="green", linewidth=5
            )
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.legend()
            # plt.grid(False)
            # plt.show()
            plt.plot(
                x[:, 0, 2].detach().cpu().numpy()+2,
                label=f"Subsampled Signal (Column {0}, Channel {2})",
                color="blue", linewidth=5
            )
            
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(False)
            plt.show()


        h_0 = torch.zeros(
            self.hidden_layers,
            combined_recovered_signal_expanded.size(0),
            self.hidden_size,
            dtype=torch.float,
        )
        c_0 = torch.zeros(
            self.hidden_layers,
            combined_recovered_signal_expanded.size(0),
            self.hidden_size,
            dtype=torch.float,
        )

        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.cuda()
            )
        if len(x.shape) > 2:
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.unsqueeze(-1).repeat(
                    1, 1, x.shape[2]
                )
            )
        elif len(x.shape) == 2:
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.unsqueeze(-1)
            )

        if combined_recovered_signal_expanded.size(0) != x.size(0):
            combined_recovered_signal_expanded = combined_recovered_signal_expanded[
                : x.size(0)
            ]

        combined_recovered_signal_expanded = combined_recovered_signal_expanded.float()

        _, (h_out, _) = self.lstm(combined_recovered_signal_expanded, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = F.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = F.relu(output)

        output = self.linear3(output)

        return output


class SHRED(torch.nn.Module):
    """SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter"""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
    ):
        super(SHRED, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )

        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        h_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )
        c_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )

        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


class CSSHREDLAGS(nn.Module):
    """CS-SHRED 1D along lags per sensor: same tensor layout as SHRED.

    For each (batch, sensor), applies 1D CS to ``x[b, :, s]`` (length ``lags``),
    producing ``x_rec`` with shape ``(batch, lags, num_sensors)``, then runs the
    same LSTM + MLP backbone as :class:`SHRED`.

    ``basis`` selects FFT (``recover_signal``) or DCT (``recover_signal_dct``).
    Optional LRU cache keys on the input float32 vector to reduce repeated SPGL1
    cost across epochs.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
        l1_tol=1e-3,
        opt_tol=1e-4,
        ls_tol=1e-4,
        n_sparsity_threshold=0.75,
        cs_observed_rel_error_max=0.30,
        cs_enable_fallback=True,
        verbosity=-1,
        basis="fft",
        cache_max_entries=100000,
        cs_warmup_epochs=0,
    ):
        super(CSSHREDLAGS, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.l1_tol = l1_tol
        self.opt_tol = opt_tol
        self.ls_tol = ls_tol
        self.n_sparsity_threshold = float(n_sparsity_threshold)
        self.cs_observed_rel_error_max = float(cs_observed_rel_error_max)
        self.cs_enable_fallback = bool(cs_enable_fallback)
        self.verbosity_spgl1 = verbosity

        b = str(basis).strip().lower()
        if b not in ("fft", "dct"):
            raise ValueError("basis must be 'fft' or 'dct', got %s" % basis)
        self._cs_basis = b

        self.cs_warmup_epochs = int(cs_warmup_epochs)
        self.current_epoch = 0

        cm = int(cache_max_entries)
        self._lags_cache = _LagVectorLruCache(cm) if cm > 0 else None

    def set_training_progress(self, epoch, total_epochs):
        if total_epochs <= 0:
            self.current_epoch = 0
            return
        self.current_epoch = int(epoch)

    def clear_lags_cache(self):
        if self._lags_cache is not None:
            self._lags_cache._data.clear()

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("CSSHREDLAGS expects x of shape (batch, lags, sensors)")

        device = x.device
        dtype = x.dtype
        bsz, _lags, n_sens = x.shape

        cs_bypass = (
            self.cs_warmup_epochs > 0 and self.current_epoch <= self.cs_warmup_epochs
        )

        x_np = x.detach().cpu().numpy()
        out = np.empty_like(x_np, dtype=np.float64)

        for b in range(bsz):
            for s in range(n_sens):
                vec = x_np[b, :, s]
                if cs_bypass:
                    out[b, :, s] = vec
                    continue

                zero_frac = float(np.mean(vec == 0.0))
                if zero_frac <= self.n_sparsity_threshold:
                    out[b, :, s] = vec
                    continue

                key = _lags_vector_cache_key(self._cs_basis, vec.astype(np.float32))
                if self._lags_cache is not None:
                    hit = self._lags_cache.get(key)
                    if hit is not None:
                        out[b, :, s] = hit
                        continue

                vt = torch.tensor(vec, dtype=torch.float32)
                if self._cs_basis == "fft":
                    rec = recover_signal(
                        vt,
                        self.l1_tol,
                        self.opt_tol,
                        self.ls_tol,
                        self.n_sparsity_threshold,
                        self.verbosity_spgl1,
                        observed_rel_error_max=self.cs_observed_rel_error_max,
                        enable_fallback=self.cs_enable_fallback,
                    )
                else:
                    rec = recover_signal_dct(
                        vt,
                        self.l1_tol,
                        self.opt_tol,
                        self.ls_tol,
                        self.n_sparsity_threshold,
                        self.verbosity_spgl1,
                        observed_rel_error_max=self.cs_observed_rel_error_max,
                        enable_fallback=self.cs_enable_fallback,
                    )

                rec_np = rec.detach().numpy().astype(np.float64).reshape(-1)
                out[b, :, s] = rec_np
                if self._lags_cache is not None:
                    self._lags_cache.set(key, rec_np)

        x_rec = torch.as_tensor(out, dtype=dtype, device=device)

        h_0 = torch.zeros(
            (self.hidden_layers, x_rec.size(0), self.hidden_size),
            dtype=dtype,
            device=device,
        )
        c_0 = torch.zeros(
            (self.hidden_layers, x_rec.size(0), self.hidden_size),
            dtype=dtype,
            device=device,
        )

        _, (h_out, _) = self.lstm(x_rec, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        o = self.linear1(h_out)
        o = self.dropout(o)
        o = F.relu(o)
        o = self.linear2(o)
        o = self.dropout(o)
        o = F.relu(o)
        o = self.linear3(o)
        return o


# -----------------------------------------------------------------------
# WindowCsSHRED: CS row-wise integrado na arquitetura + SHRED backbone.
#
# Fases:
# 1. prepare_input(volume_thw) -> (N, lags, num_sensors) via CS + sensores
# 2. forward(x) -> (B, output_size) via LSTM + decoder
#
# O CS (recover_frame_rowwise) roda SEM gradiente (solver SPGL1 fixo).
# Apenas o backbone LSTM+MLP e treinavel.
# -----------------------------------------------------------------------
class WindowCsSHRED(torch.nn.Module):
    """SHRED com CS row-wise integrado como primeiro estagio da arquitetura.

    A preparacao dos dados (CS por frame + extracao de sensores) e um metodo
    do modelo, nao um preprocessamento externo.  O forward recebe tensores
    (batch, lags, num_sensors) ja extraidos por ``prepare_input``.

    Args:
        input_size: Numero de sensores (= num_sensors).
        output_size: Dimensao do estado espacial (= dim_h * dim_w).
        hidden_size: Tamanho do hidden state da LSTM.
        hidden_layers: Numero de camadas LSTM.
        l1: Neuroenios na primeira camada linear.
        l2: Neuroenios na segunda camada linear.
        dropout: Taxa de dropout.
        sensor_flat_indices: Indices lineares dos sensores em H*W (numpy int64).
        dim_h: Altura do grid espacial.
        dim_w: Largura do grid espacial.
        lags: Tamanho da janela temporal.
        cs_l1_tol: BP tolerance para SPGL1 (row-wise).
        cs_opt_tol: Optimality tolerance para SPGL1.
        cs_ls_tol: Line-search tolerance para SPGL1.
        cs_iter_lim: Limite de iteracoes do SPGL1.
        cs_obs_rel_error_max: Erro relativo maximo nos pontos observados.
        cs_enable_fallback: Se True, retorna original quando erro e alto.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
        sensor_flat_indices=None,
        dim_h=1,
        dim_w=1,
        lags=50,
        cs_l1_tol=1e-2,
        cs_opt_tol=1e-3,
        cs_ls_tol=1e-3,
        cs_iter_lim=200,
        cs_obs_rel_error_max=0.30,
        cs_enable_fallback=True,
    ):
        super(WindowCsSHRED, self).__init__()

        # -- SHRED backbone (LSTM + decoder) --
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        # -- Spatial / sensor config --
        if sensor_flat_indices is None:
            raise ValueError("sensor_flat_indices is required")
        self._sensor_flat_indices = np.asarray(sensor_flat_indices, dtype=np.int64)
        self._dim_h = int(dim_h)
        self._dim_w = int(dim_w)
        self._lags = int(lags)

        # -- CS solver parameters (fixed, no gradient) --
        self._cs_l1_tol = float(cs_l1_tol)
        self._cs_opt_tol = float(cs_opt_tol)
        self._cs_ls_tol = float(cs_ls_tol)
        self._cs_iter_lim = int(cs_iter_lim)
        self._cs_obs_rel_error_max = float(cs_obs_rel_error_max)
        self._cs_enable_fallback = bool(cs_enable_fallback)

        # -- Frame-level cache (time_index -> recovered (H, W) ndarray) --
        self._frame_cache = {}

    # -----------------------------------------------------------------
    # CS stem: recover_frame_rowwise per frame (numpy, no gradient)
    # -----------------------------------------------------------------
    def _recover_single_frame(self, frame_hw):
        """Apply row-wise CS on one (H, W) frame.  Returns (H, W) numpy."""
        from cs_preprocess import recover_frame_rowwise

        recovered, _stats = recover_frame_rowwise(
            np.asarray(frame_hw, dtype=np.float64),
            l1_tol=self._cs_l1_tol,
            opt_tol=self._cs_opt_tol,
            ls_tol=self._cs_ls_tol,
            iter_lim=self._cs_iter_lim,
            obs_rel_error_max=self._cs_obs_rel_error_max,
            enable_fallback=self._cs_enable_fallback,
            verbosity=0,
        )
        return recovered

    def clear_cs_cache(self):
        """Remove all cached recovered frames."""
        self._frame_cache.clear()

    # -----------------------------------------------------------------
    # prepare_input: CS + sensor extraction  (the "architecture stem")
    # -----------------------------------------------------------------
    def prepare_input(self, volume_thw):
        """Apply CS row-wise per frame, then build sensor windows.

        This is the first stage of the WindowCsSHRED architecture.
        It runs WITHOUT gradient (solver SPGL1) and produces tensors
        ready for the LSTM backbone.

        Args:
            volume_thw: numpy array ``(T, H, W)`` -- scaled + re-zeroed
                (zeros mark missing data for CS detection).

        Returns:
            numpy array ``(T - lags, lags, num_sensors)`` sensor windows.
        """
        if volume_thw.ndim != 3:
            raise ValueError("volume_thw must be 3-D (T, H, W)")
        t_dim, h_dim, w_dim = volume_thw.shape
        if h_dim != self._dim_h or w_dim != self._dim_w:
            raise ValueError(
                f"spatial dims ({h_dim}, {w_dim}) != expected "
                f"({self._dim_h}, {self._dim_w})"
            )
        if t_dim <= self._lags:
            raise ValueError("T must be > lags")

        self.clear_cs_cache()

        recovered = np.empty_like(volume_thw)
        frames_with_cs = 0
        for t in range(t_dim):
            frame = volume_thw[t]
            if np.any(frame == 0):
                rec = self._recover_single_frame(frame)
                recovered[t] = rec
                frames_with_cs += 1
            else:
                recovered[t] = frame
            self._frame_cache[t] = recovered[t]

        print(
            f"[WindowCsSHRED.prepare_input] T={t_dim}, "
            f"frames_with_cs={frames_with_cs}/{t_dim}, "
            f"lags={self._lags}, sensors={len(self._sensor_flat_indices)}"
        )

        recovered_flat = recovered.reshape(t_dim, h_dim * w_dim)

        n_windows = t_dim - self._lags
        idx = self._sensor_flat_indices
        out = np.empty(
            (n_windows, self._lags, idx.size), dtype=recovered_flat.dtype
        )
        for i in range(n_windows):
            out[i] = recovered_flat[i : i + self._lags][:, idx]

        return out

    # -----------------------------------------------------------------
    # forward_full: end-to-end (volume -> predictions) for inference
    # -----------------------------------------------------------------
    def forward_full(self, volume_thw, device=None):
        """End-to-end inference: volume -> CS -> sensors -> LSTM -> output.

        Convenience method that chains ``prepare_input`` and ``forward``.

        Args:
            volume_thw: numpy ``(T, H, W)`` scaled + re-zeroed.
            device: torch device (default: model device).

        Returns:
            torch.Tensor ``(N, output_size)`` predictions.
        """
        sensor_windows = self.prepare_input(volume_thw)
        if device is None:
            device = next(self.parameters()).device
        x = torch.tensor(sensor_windows, dtype=torch.float32).to(device)
        return self.forward(x)

    # -----------------------------------------------------------------
    # forward: LSTM + decoder (trainable, receives sensor windows)
    # -----------------------------------------------------------------
    def forward(self, x):
        """Run the SHRED backbone on sensor windows.

        Args:
            x: torch.Tensor ``(batch, lags, num_sensors)``.

        Returns:
            torch.Tensor ``(batch, output_size)``.
        """
        h_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )
        c_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )

        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


# -----------------------------------------------------------------------
# BatchCsSHRED: CS row-wise 2-D DENTRO do forward(), por batch.
#
# Diferenca critica em relacao a WindowCsSHRED:
#   - WindowCsSHRED.prepare_input() roda ANTES do treino (pre-processamento).
#   - BatchCsSHRED.forward() recebe frames espaciais (B, lags, H, W) e
#     executa CS + extracao de sensores + LSTM + decoder a cada batch.
#
# Isso torna a arquitetura pronta para, no futuro (Passo 2), substituir
# o solver SPGL1 por um modulo diferenciavel (e.g. ISTA-Net) e treinar
# end-to-end com gradientes atravessando o CS.
#
# Para o Passo 1 atual, o SPGL1 e determinístico e usa um cache por frame
# (keyed pelo indice temporal absoluto) para evitar recomputacao.
# -----------------------------------------------------------------------
class BatchCsSHRED(torch.nn.Module):
    """SHRED com CS row-wise 2-D integrado no forward(), executado por batch.

    O forward() recebe frames espaciais ``(B, lags, H, W)``, aplica CS
    row-wise em cada frame, extrai sensores, e alimenta o backbone
    LSTM + decoder.  O CS roda em NumPy/CPU (SPGL1 sem gradiente);
    apenas o LSTM + decoder e treinavel.

    Para Passo 2 (futuro): substituir ``_recover_single_frame`` por um
    modulo PyTorch diferenciavel e remover o cache.

    Args:
        input_size: Numero de sensores (= num_sensors).
        output_size: Dimensao do estado espacial (= dim_h * dim_w).
        hidden_size: Tamanho do hidden state da LSTM.
        hidden_layers: Numero de camadas LSTM.
        l1: Neuronios na primeira camada linear.
        l2: Neuronios na segunda camada linear.
        dropout: Taxa de dropout.
        sensor_flat_indices: Indices lineares dos sensores em H*W (numpy int64).
        dim_h: Altura do grid espacial.
        dim_w: Largura do grid espacial.
        lags: Tamanho da janela temporal.
        cs_l1_tol: BP tolerance para SPGL1 (row-wise).
        cs_opt_tol: Optimality tolerance para SPGL1.
        cs_ls_tol: Line-search tolerance para SPGL1.
        cs_iter_lim: Limite de iteracoes do SPGL1.
        cs_obs_rel_error_max: Erro relativo maximo nos pontos observados.
        cs_enable_fallback: Se True, retorna original quando erro e alto.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
        sensor_flat_indices=None,
        dim_h=1,
        dim_w=1,
        lags=50,
        cs_l1_tol=1e-2,
        cs_opt_tol=1e-3,
        cs_ls_tol=1e-3,
        cs_iter_lim=200,
        cs_obs_rel_error_max=0.30,
        cs_enable_fallback=True,
    ):
        super(BatchCsSHRED, self).__init__()

        # -- SHRED backbone (LSTM + decoder) --
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        # -- Spatial / sensor config --
        if sensor_flat_indices is None:
            raise ValueError("sensor_flat_indices is required")
        self._sensor_flat_indices = np.asarray(sensor_flat_indices, dtype=np.int64)
        self._dim_h = int(dim_h)
        self._dim_w = int(dim_w)
        self._lags = int(lags)

        # -- CS solver parameters (fixed, no gradient) --
        self._cs_l1_tol = float(cs_l1_tol)
        self._cs_opt_tol = float(cs_opt_tol)
        self._cs_ls_tol = float(cs_ls_tol)
        self._cs_iter_lim = int(cs_iter_lim)
        self._cs_obs_rel_error_max = float(cs_obs_rel_error_max)
        self._cs_enable_fallback = bool(cs_enable_fallback)

        # -- Frame cache: absolute time index -> recovered (H, W) ndarray --
        self._frame_cache = {}
        self._cs_calls = 0
        self._cache_hits = 0

    # -----------------------------------------------------------------
    # CS: recover one (H, W) frame via row-wise SPGL1 (parallel)
    # -----------------------------------------------------------------
    def _recover_single_frame(self, frame_hw):
        """Apply row-wise CS on one (H, W) frame.  Returns (H, W) numpy.

        Uses parallel row recovery when multiple rows need CS.
        """
        from cs_preprocess import recover_frame_rowwise_parallel

        recovered, _stats = recover_frame_rowwise_parallel(
            np.asarray(frame_hw, dtype=np.float64),
            l1_tol=self._cs_l1_tol,
            opt_tol=self._cs_opt_tol,
            ls_tol=self._cs_ls_tol,
            iter_lim=self._cs_iter_lim,
            obs_rel_error_max=self._cs_obs_rel_error_max,
            enable_fallback=self._cs_enable_fallback,
        )
        return recovered

    def clear_cs_cache(self):
        """Remove all in-memory cached recovered frames and reset counters."""
        self._frame_cache.clear()
        self._cs_calls = 0
        self._cache_hits = 0

    # -----------------------------------------------------------------
    # Disk cache: persist / load recovered frames between trials
    # -----------------------------------------------------------------
    def save_frame_cache_to_disk(self, path: str) -> int:
        """Save in-memory frame cache to a .npz file on disk.

        Returns the number of frames saved.
        """
        if not self._frame_cache:
            return 0
        keys = sorted(self._frame_cache.keys())
        arrays = {str(k): self._frame_cache[k] for k in keys}
        np.savez_compressed(path, **arrays)
        return len(arrays)

    def load_frame_cache_from_disk(self, path: str) -> int:
        """Load frame cache from a .npz file into memory.

        Returns the number of frames loaded.  Existing entries are not
        overwritten (in-memory cache takes precedence).
        """
        if not os.path.exists(path):
            return 0
        data = np.load(path)
        loaded = 0
        for key in data.files:
            t_abs = int(key)
            if t_abs not in self._frame_cache:
                self._frame_cache[t_abs] = data[key]
                loaded += 1
        data.close()
        return loaded

    # -----------------------------------------------------------------
    # warm_cache: pre-recover all frames with multiprocessing (spawn)
    # -----------------------------------------------------------------
    def warm_cache(self, volume_thw, max_workers=4):
        """Pre-recover all frames that contain zeros using multiprocessing.

        Uses ``spawn`` context to avoid fork() memory duplication and to
        bypass the GIL, achieving true parallelism.  Each worker receives
        a single frame (~500 KB) and returns the recovered result.

        Frames already in ``_frame_cache`` are skipped.

        Args:
            volume_thw: np.ndarray of shape ``(T, H, W)``.
            max_workers: number of parallel processes (default 4).

        Returns:
            int: number of frames recovered in this call.
        """
        import multiprocessing as mp
        from cs_preprocess import _recover_frame_worker

        t_total = volume_thw.shape[0]

        frames_to_recover = []
        for t in range(t_total):
            if t in self._frame_cache:
                continue
            frame = volume_thw[t]
            if np.any(frame == 0):
                frames_to_recover.append(t)

        if not frames_to_recover:
            print(
                f"[warm_cache] All {t_total} frames already cached "
                f"(cache_size={len(self._frame_cache)}). Nothing to do."
            )
            return 0

        already_cached = len(self._frame_cache)
        print(
            f"[warm_cache] {len(frames_to_recover)} frames need CS "
            f"(of {t_total} total, {already_cached} already cached). "
            f"Using {max_workers} workers (spawn)."
        )

        cs_kwargs = {
            "l1_tol": self._cs_l1_tol,
            "opt_tol": self._cs_opt_tol,
            "ls_tol": self._cs_ls_tol,
            "iter_lim": self._cs_iter_lim,
            "obs_rel_error_max": self._cs_obs_rel_error_max,
            "enable_fallback": self._cs_enable_fallback,
        }

        args_list = [
            (t, volume_thw[t].copy(), cs_kwargs)
            for t in frames_to_recover
        ]

        ctx = mp.get_context("spawn")
        n_done = 0
        n_total = len(args_list)

        with ctx.Pool(processes=max_workers) as pool:
            for t_abs, recovered in pool.imap_unordered(
                _recover_frame_worker, args_list, chunksize=1
            ):
                self._frame_cache[t_abs] = recovered
                n_done += 1
                # Frequent progress so long SPGL1 runs do not look "stuck" (no output).
                if n_done <= 5 or n_done % 5 == 0 or n_done == n_total:
                    print(
                        f"[warm_cache] {n_done}/{n_total} frames recovered "
                        f"(cache_size={len(self._frame_cache)})",
                        flush=True,
                    )

        print(
            f"[warm_cache] Done. {n_done} new frames recovered. "
            f"Total cache: {len(self._frame_cache)} frames."
        )
        return n_done

    # -----------------------------------------------------------------
    # forward: CS per frame + sensor extraction + LSTM + decoder
    # -----------------------------------------------------------------
    def forward(self, x, t_indices=None):
        """Forward pass with CS integrated per batch.

        Args:
            x: torch.Tensor ``(B, lags, H, W)`` spatial frames (any device).
            t_indices: torch.LongTensor ``(B,)`` absolute window start index
                for each sample.  Used as cache key.  Optional; when None
                the cache is bypassed and CS runs unconditionally.

        Returns:
            torch.Tensor ``(B, output_size)`` on model device (GPU if available).
        """
        if x.dim() != 4:
            raise ValueError(
                f"BatchCsSHRED.forward expects 4-D input (B, lags, H, W), "
                f"got {x.dim()}-D"
            )
        batch_size, seq_len, h_dim, w_dim = x.shape

        # -- CS stem: recover frames on CPU/NumPy --
        x_np = x.detach().cpu().numpy()
        frames_recovered_this_call = 0

        for b in range(batch_size):
            for t_local in range(seq_len):
                t_abs = None
                if t_indices is not None:
                    t_abs = int(t_indices[b].item()) + t_local

                if t_abs is not None and t_abs in self._frame_cache:
                    x_np[b, t_local] = self._frame_cache[t_abs]
                    self._cache_hits += 1
                    continue

                frame = x_np[b, t_local]
                if np.any(frame == 0):
                    frame = self._recover_single_frame(frame)
                    frames_recovered_this_call += 1

                x_np[b, t_local] = frame
                if t_abs is not None:
                    self._frame_cache[t_abs] = frame.copy()

        self._cs_calls += 1
        if self._cs_calls <= 2 or self._cs_calls % 50 == 0:
            print(
                f"[BatchCsSHRED.forward] call={self._cs_calls}, "
                f"batch={batch_size}, cs_recovered={frames_recovered_this_call}, "
                f"cache_size={len(self._frame_cache)}, "
                f"cache_hits={self._cache_hits}"
            )

        # -- Extract sensors: (B, lags, H*W) -> (B, lags, num_sensors) --
        x_flat = x_np.reshape(batch_size, seq_len, h_dim * w_dim)
        sensor_data = x_flat[:, :, self._sensor_flat_indices]

        # -- Move to model device for LSTM --
        device = next(self.parameters()).device
        x_tensor = torch.tensor(sensor_data, dtype=torch.float32, device=device)

        # -- LSTM + decoder --
        h_0 = torch.zeros(
            self.hidden_layers, batch_size, self.hidden_size,
            dtype=torch.float32, device=device,
        )
        c_0 = torch.zeros(
            self.hidden_layers, batch_size, self.hidden_size,
            dtype=torch.float32, device=device,
        )

        _, (h_out, _) = self.lstm(x_tensor, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


def fit_spatial(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    step_epoch=50,
    verbose=False,
    patience=5,
    generator=None,
    batch_sampler=None,
):
    """Training loop for spatial models (BatchCsSHRED).

    Like ``fit()`` but handles 3-element datasets ``(x, y, t_indices)``
    and validates in batches (spatial windows are too large to pass all
    at once).  Targets are moved to the model device inside the loop so
    the dataset can stay on CPU.

    Returns
    -------
    numpy array of validation errors (same contract as ``fit``).
    """
    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, generator=generator,
        )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(epoch=epoch, total_epochs=num_epochs)

        for _k, data in enumerate(train_loader):
            x_spatial = data[0]
            y_target = data[1].to(device)
            t_idx = data[2] if len(data) > 2 else None

            model.train()
            outputs = model(x_spatial, t_idx)
            optimizer.zero_grad()
            loss = criterion(outputs, y_target)
            loss.backward()
            optimizer.step()

        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(valid_dataset, batch_size=batch_size)
                all_val_out = []
                all_val_y = []
                for vdata in val_loader:
                    vx = vdata[0]
                    vy = vdata[1].to(device)
                    vt = vdata[2] if len(vdata) > 2 else None
                    vout = model(vx, vt)
                    all_val_out.append(vout)
                    all_val_y.append(vy)
                val_outputs = torch.cat(all_val_out, dim=0)
                val_y = torch.cat(all_val_y, dim=0)
                val_error = torch.linalg.norm(
                    val_outputs - val_y
                ) / torch.linalg.norm(val_y)
                val_error_list.append(val_error)

            if verbose:
                print(
                    f"Training epoch {epoch}  "
                    f"Error {val_error_list[-1]:.6f}"
                )

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def fit_spatial_composite(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    lambL2=1.0,
    lambL1=0.01,
    lambdaSNR=0.03,
    step_epoch=50,
    verbose=False,
    patience=5,
    generator=None,
    batch_sampler=None,
    use_adamw=False,
    use_scheduler=False,
):
    """Training loop for BatchCsSHRED with composite loss (MSE+L1+SNR).

    Same data handling as ``fit_spatial`` (3-element datasets on CPU) but
    uses the composite loss from ``fit_csshred_model``.

    Parameters
    ----------
    lambL2 : float
        Weight for MSE term.
    lambL1 : float
        Weight for L1 sparsity term (outputs vs zeros).
    lambdaSNR : float
        Weight for SNR term.
    use_adamw : bool
        If True use AdamW (with weight_decay=1e-5); otherwise Adam.
    use_scheduler : bool
        If True use ReduceLROnPlateau on validation loss.

    Returns
    -------
    numpy array of validation errors.
    """
    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size, generator=generator,
        )

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    weight_decay_manual = 1e-4

    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5,
        )

    device = next(model.parameters()).device

    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(epoch=epoch, total_epochs=num_epochs)

        for _k, data in enumerate(train_loader):
            x_spatial = data[0]
            y_target = data[1].to(device)
            t_idx = data[2] if len(data) > 2 else None

            model.train()
            outputs = model(x_spatial, t_idx)
            optimizer.zero_grad()

            loss_mse = criterion_mse(outputs, y_target)
            loss_l1 = criterion_l1(outputs, torch.zeros_like(outputs))
            snr = calculate_snr(y_target, outputs)

            l2_reg = sum(torch.norm(p, p=2) for p in model.parameters())

            if snr > 0:
                loss = (
                    torch.clamp(1.0 / (snr + 1e-8), max=100.0) * lambdaSNR
                    + lambL2 * loss_mse
                    + lambL1 * loss_l1
                    + weight_decay_manual * l2_reg
                )
            else:
                loss = (
                    -snr * lambdaSNR
                    + lambL2 * loss_mse
                    + lambL1 * loss_l1
                    + weight_decay_manual * l2_reg
                )

            loss.backward()
            optimizer.step()

        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(valid_dataset, batch_size=batch_size)
                all_val_out = []
                all_val_y = []
                for vdata in val_loader:
                    vx = vdata[0]
                    vy = vdata[1].to(device)
                    vt = vdata[2] if len(vdata) > 2 else None
                    vout = model(vx, vt)
                    all_val_out.append(vout)
                    all_val_y.append(vy)
                val_outputs = torch.cat(all_val_out, dim=0)
                val_y = torch.cat(all_val_y, dim=0)

                val_mse = criterion_mse(val_outputs, val_y)
                val_l1 = criterion_l1(val_outputs, torch.zeros_like(val_outputs))
                val_snr = calculate_snr(val_y, val_outputs)

                if val_snr > 0:
                    val_loss = (
                        torch.clamp(1.0 / (val_snr + 1e-8), max=100.0) * lambdaSNR
                        + lambL2 * val_mse
                        + lambL1 * val_l1
                    )
                else:
                    val_loss = (
                        -val_snr * lambdaSNR
                        + lambL2 * val_mse
                        + lambL1 * val_l1
                    )

                val_error = torch.linalg.norm(
                    val_outputs - val_y
                ) / torch.linalg.norm(val_y)
                val_error_list.append(val_error)

            if scheduler is not None:
                scheduler.step(val_loss)

            if verbose:
                print(
                    f"Training epoch {epoch}  "
                    f"Error {val_error_list[-1]:.6f}  "
                    f"ValLoss {val_loss.item():.6f}  "
                    f"SNR {val_snr.item():.2f}dB"
                )

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


class SDN(torch.nn.Module):
    """SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter"""

    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


def fit(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    step_epoch=50,
    verbose=False,
    patience=5,
    generator=None,
    batch_sampler=None,
):
    """Function for training SHRED and SDN models.

    When batch_sampler is provided, it overrides shuffle and batch_size
    in the DataLoader (used by CS-SHRED-FAIR for contiguous batches).
    """
    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, generator=generator)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(epoch=epoch, total_epochs=num_epochs)

        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(
                    val_outputs - valid_dataset.Y
                ) / torch.linalg.norm(valid_dataset.Y)
                # val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print("Training epoch " + str(epoch))
                print("Error " + str(val_error_list[-1]))
                

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def total_variation_regularization(outputs):
    h_diff = torch.abs(outputs[:, :-1] - outputs[:, 1:])
    v_diff = torch.abs(outputs[:-1, :] - outputs[1:, :])
    total_var = torch.sum(h_diff) + torch.sum(v_diff)
    return total_var


def abrupt_transition_regularization(outputs):
    h_diff = torch.abs(outputs[:, :-1] - outputs[:, 1:])
    total_diff = torch.sum(h_diff)
    return total_diff


def calculate_snr(original_signal, noisy_signal):
    signal_power = torch.mean(original_signal**2)
    noise_power = torch.mean((noisy_signal - original_signal) ** 2)
    eps = 1e-10
    snr_db = 10 * torch.log10(signal_power / (noise_power + eps))

    return snr_db


def fit_csshred_model(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    lambL2=1,
    lambL1=0.01,
    lambdaSNR=0.03,
    step_epoch=20,
    verbose=False,
    patience=5,
    generator=None,
):
    # Resetar estatisticas do CS para cada chamada de fit
    _cs_stats["recovered"] = 0
    _cs_stats["skipped"] = 0
    _cs_stats["failed"] = 0
    _cs_stats["forward_calls"] = 0
    _cs_stats["cumulative_recovered"] = 0
    _cs_stats["cumulative_skipped"] = 0
    _cs_stats["cumulative_failed"] = 0

    # Usar SequentialBlockBatchSampler para CS-SHRED:
    # Batches contiguos fazem x[:, i, j] ser uma serie temporal real,
    # permitindo que o CS (FFT) opere na dimensao correta (temporal).
    batch_sampler = SequentialBlockBatchSampler(
        len(train_dataset), batch_size,
        shuffle_blocks=True, generator=generator,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    val_error_list = []
    train_error_list = []
    patience_counter = 0  
    lambL2 = lambL2  
    lambL1 = lambL1  
    lambdaSNR = lambdaSNR  
    best_params = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(epoch=epoch, total_epochs=num_epochs)
        train_losses = []

        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()

            lossMSE = criterion(outputs, data[1])
            lossL1 = criterion2(outputs, torch.zeros_like(outputs))
            snr = calculate_snr(data[1], outputs)

            # L2 regularization
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)

            # Adjusting the loss to incentivize the maximization of the SNR
            if snr > 0:
                loss = (
                    torch.clamp(1 / (snr + 1e-8), max=100.0) * lambdaSNR
                    + lambL2 * lossMSE
                    + lambL1 * lossL1
                    + weight_decay * l2_reg
                )  # The higher the SNR, the lower the loss
            else:
                loss = (
                    - snr * lambdaSNR
                    + lambL2 * lossMSE
                    + lambL1 * lossL1
                    + weight_decay * l2_reg
                )  # Inverse of the SNR: the lower the SNR, the higher the loss

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_error_list.append(avg_train_loss)

        # Calculating the validation error after the end of the epoch
        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = forward_csshred_contiguous_batches(
                    model, valid_dataset.X, batch_size,
                )

                # Calculating the validation SNR
                val_snr = calculate_snr(valid_dataset.Y, val_outputs)
                
                # CORRIGIDO: Calcular MSE e L1 usando dados de VALIDACAO, nao do treinamento
                # As variaveis lossMSE e lossL1 do loop de treinamento nao existem neste escopo
                val_lossMSE = criterion(val_outputs, valid_dataset.Y)
                val_lossL1 = criterion2(val_outputs, torch.zeros_like(val_outputs))

                # Adjusting the validation loss to incentivize the maximization of the SNR
                if val_snr > 0:
                    val_loss = (
                        torch.clamp(1 / (val_snr + 1e-8), max=100.0) * lambdaSNR 
                        + lambL2 * val_lossMSE 
                        + lambL1 * val_lossL1
                    )
                else:
                    val_loss = (
                        -val_snr * lambdaSNR 
                        + lambL2 * val_lossMSE 
                        + lambL1 * val_lossL1
                    )

                val_error_list.append(val_loss)
            scheduler.step(val_loss)

            if verbose:
                print(f"LambdaL2: {lambL2}, LambdaL1: {lambL1}, LambdaSNR: {lambdaSNR}")
                print("Training epoch " + str(epoch))
                print("Training Error: " + str(avg_train_loss))
                print("Validation Error:" + str(val_loss.item()))
                print("Validation SNR:" + str(val_snr.item()))

            if len(val_error_list) > 0:
                if val_loss == torch.min(torch.tensor(val_error_list)):
                    patience_counter = 0
                    best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return (
                    torch.tensor(train_error_list).cpu(),
                    torch.tensor(val_error_list).cpu(),
                )

    model.load_state_dict(best_params)
    return torch.tensor(train_error_list).detach().cpu().numpy(),torch.tensor(val_error_list).detach().cpu().numpy()


def fit_shred_composite(*args, **kwargs):
    """Train ``SHRED`` with the same composite objective as ``fit_csshred_model``.

    Ablation V3 (reviewer): SNR-guided loss (MSE + MAE + piecewise SNR term) without
    the compressed-sensing block in the forward pass. The optimization loop, AdamW,
    scheduler, and contiguous-batch sampler are identical to ``fit_csshred_model``;
    only the model class differs (``SHRED`` vs ``CSSHRED``). Tensor shapes follow the
    same (batch, lags, sensors) layout as the reference 1D CS-SHRED pipeline.
    """
    return fit_csshred_model(*args, **kwargs)


def forecast(forecaster, reconstructor, test_dataset):
    """Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts
    throughout the test dataset."""
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(scaled_output)

    device = "cuda" if next(reconstructor.parameters()).is_cuda else "cpu"
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = (
            reconstructor(
                forecasted_vals[i : i + test_dataset.X.shape[1]].reshape(
                    1, test_dataset.X.shape[1], test_dataset.X.shape[2]
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
