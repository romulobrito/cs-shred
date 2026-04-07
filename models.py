from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import pylops
from pylops.optimization.sparsity import spgl1
import warnings

warnings.filterwarnings("ignore", message="Linesearch failed with error 1")


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
    """Run model forward in contiguous temporal batches (training-style blocking).

    Used for ``CSSHREDLAGS`` and for ``SHRED`` when trained with
    ``fit_csshred_model`` (same ``SequentialBlockBatchSampler`` layout).
    ``x_tensor`` may live on CPU; each chunk is passed to ``model``.
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


# ---------------------------------------------------------------------------
# CS statistics (optional diagnostics from recover_signal / per-column paths)
# ---------------------------------------------------------------------------
_cs_stats = {
    "recovered": 0,
    "skipped": 0,
    "failed": 0,
    "forward_calls": 0,
    "cumulative_recovered": 0,
    "cumulative_skipped": 0,
    "cumulative_failed": 0,
}


# --- SPGL1 recovery helpers (FFT / DCT), used by CSSHREDLAGS ----------------------------
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

            # Adaptive: set iter_lim from tolerances
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
            # Real part only (IFFT can leave tiny imaginary numerical residue)
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

        except Exception:
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


class SHRED(torch.nn.Module):
    """LSTM shallow decoder from sensor windows to full state (same layout as CSSHREDLAGS)."""

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
    """Compressed sensing along the lag axis per sensor; same tensor layout as SHRED.

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
    """Train SHRED or CSSHREDLAGS with plain MSE and Adam.

    If ``batch_sampler`` is set, it replaces shuffle/batch_size in the training
    DataLoader (contiguous batches for lag-wise CS consistency).
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
    """Train with composite loss (MSE + L1-on-output + SNR term) and AdamW.

    Uses :class:`SequentialBlockBatchSampler` so each batch is a contiguous time
    block (required for consistent lag-wise CS in ``CSSHREDLAGS``). Validation
    runs through :func:`forward_csshred_contiguous_batches`. Works with ``SHRED``
    (ablation, no CS in forward) or ``CSSHREDLAGS``.
    """
    # Reset CS counters for this training run
    _cs_stats["recovered"] = 0
    _cs_stats["skipped"] = 0
    _cs_stats["failed"] = 0
    _cs_stats["forward_calls"] = 0
    _cs_stats["cumulative_recovered"] = 0
    _cs_stats["cumulative_skipped"] = 0
    _cs_stats["cumulative_failed"] = 0

    # Contiguous batches so each batch is a true temporal slice for lag-axis CS
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
                
                # Validation MSE/L1 from validation tensors (not training-loop variables)
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
    return (
        torch.tensor(train_error_list).detach().cpu().numpy(),
        torch.tensor(val_error_list).detach().cpu().numpy(),
    )


def fit_shred_composite(*args, **kwargs):
    """Train ``SHRED`` with the same composite objective as ``fit_csshred_model``.

    Ablation V3 (reviewer): SNR-guided loss (MSE + MAE + piecewise SNR term) without
    the compressed-sensing block in the forward pass. The optimization loop, AdamW,
    scheduler, and contiguous-batch sampler are identical to ``fit_csshred_model``;
    only the model class differs (plain ``SHRED`` vs ``CSSHREDLAGS``). Shapes follow
    the usual (batch, lags, sensors) layout.
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
