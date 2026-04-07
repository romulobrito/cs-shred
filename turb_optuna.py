import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from math import log10
import optuna
import json
import sys
import os
from datetime import datetime

# Repository root (this script lives at repo root alongside lpips_eval.py).
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _default_turb_npy_path() -> str:
    """Default TURB-Rot .npy under codigos_para_gerar_imagem_cs-shred/data/."""
    return os.environ.get(
        "TURB_NPY_PATH",
        os.path.join(
            _REPO_ROOT,
            "codigos_para_gerar_imagem_cs-shred",
            "data",
            "turb_vy_combined.npy",
        ),
    )


def _default_optuna_results_base() -> str:
    return os.environ.get(
        "TURB_OPTUNA_RESULTS_DIR",
        os.path.join(_REPO_ROOT, "results", "turb_optuna"),
    )


import models
from processdata import TimeSeriesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Minimum fraction of zeros to trigger SPGL1 along lags (CSSHREDLAGS; see models.recover_signal).
# Override: export CS_N_SPARSITY_THRESHOLD=0.85
CS_N_SPARSITY_THRESHOLD = float(os.environ.get("CS_N_SPARSITY_THRESHOLD", "0.75"))

# Turb study: CSSHREDLAGS, SHRED (MSE via fit), or SHRED-COMPOSITE (SHRED + fit_csshred_model composite loss).
# No CS-SHRED (spatial CS block) in this bundle.
ABLATION_MODEL_TYPE = os.environ.get("ABLATION_MODEL_TYPE", "CSSHREDLAGS").strip().upper()
if ABLATION_MODEL_TYPE not in ("CSSHREDLAGS", "SHRED", "SHRED-COMPOSITE"):
    raise ValueError(
        "ABLATION_MODEL_TYPE must be CSSHREDLAGS, SHRED, or SHRED-COMPOSITE, got: %s"
        % ABLATION_MODEL_TYPE
    )

CSSHREDLAGS_BASIS = os.environ.get("CSSHREDLAGS_BASIS", "fft").strip().lower()
if CSSHREDLAGS_BASIS not in ("fft", "dct"):
    CSSHREDLAGS_BASIS = "fft"
CSSHREDLAGS_CACHE_MAX = int(os.environ.get("CSSHREDLAGS_CACHE_MAX", "100000"))
CSSHREDLAGS_TRAIN_LOSS = os.environ.get("CSSHREDLAGS_TRAIN_LOSS", "mse").strip().lower()
if CSSHREDLAGS_TRAIN_LOSS not in ("mse", "full"):
    CSSHREDLAGS_TRAIN_LOSS = "mse"

# Ablation loss mode: conditions lambL1/lambdaSNR in Optuna; used by fit_csshred_model when
# CSSHREDLAGS_TRAIN_LOSS=full or when ABLATION_MODEL_TYPE=SHRED-COMPOSITE.
#   "full"     -> MSE + L1 + SNR
#   "mse_only" -> MSE only
#   "mse_l1"   -> MSE + L1
#   "mse_snr"  -> MSE + SNR
ABLATION_LOSS_MODE = os.environ.get("ABLATION_LOSS_MODE", "full").strip().lower()

TURB_TIME_SLICE = int(os.environ.get("TURB_TIME_SLICE", "650"))
TURB_COL_SUB_FRAC = float(os.environ.get("TURB_COL_SUB_FRAC", "0.3"))
TURB_SNAP_SUB_FRAC = float(os.environ.get("TURB_SNAP_SUB_FRAC", "0.3"))

print(
    "device=%s model=%s ablation=%s CS_N_SPARSITY_THRESHOLD=%s TURB_TIME_SLICE=%s"
    % (
        device,
        ABLATION_MODEL_TYPE,
        ABLATION_LOSS_MODE,
        CS_N_SPARSITY_THRESHOLD,
        TURB_TIME_SLICE,
    )
)
if ABLATION_MODEL_TYPE == "CSSHREDLAGS":
    print(
        "CSSHREDLAGS_BASIS=%s CSSHREDLAGS_CACHE_MAX=%s CSSHREDLAGS_TRAIN_LOSS=%s"
        % (CSSHREDLAGS_BASIS, CSSHREDLAGS_CACHE_MAX, CSSHREDLAGS_TRAIN_LOSS)
    )

def load_data(npy_file_path, time_slice):
    data_array = np.load(npy_file_path)
    data_array = data_array[:,:,: time_slice]
    data_array = np.transpose(data_array, (2,0,1))
    return data_array


def visualize_data(matrix, subsampled, results_dir, save_plots=True):
    """
    Visualize the last temporal slice of the original and subsampled data arrays,
    saving the plots as PNG files.
    
    Parameters
    ----------
    matrix : np.ndarray
        Original data array of shape (time, x, y).
    subsampled : np.ndarray
        Subsampled data array of shape (x, y, time).
    results_dir : str
        Directory to save the plots.
    save_plots : bool
        Whether to save the plots to disk.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix[-1, :, :], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_plots:
        plt.savefig(os.path.join(results_dir, "last_temporal_slice.png"))
        plt.savefig(os.path.join(results_dir, "last_temporal_slice.pdf"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(subsampled[:, :, -1], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice (Subsampled)")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_plots:
        plt.savefig(os.path.join(results_dir, "last_temporal_subsampled_slice.png"))
        plt.savefig(os.path.join(results_dir, "last_temporal_subsampled_slice.pdf"))
    plt.close()



def subsample(snapshot, num_cols_subsample, num_snapshots_subsample):
    np.random.seed(1001)

    snapshot = np.transpose(snapshot, (1, 2, 0))
    # print('snapshot after transpose', snapshot.shape)
    dim_x, dim_y, dim_t = snapshot.shape
    snapshot_subsampled = snapshot.copy()

    num_snapshots_subsample = min(num_snapshots_subsample, dim_t - 1)

    num_cols_subsample = min(num_cols_subsample, dim_y - 1)
    cols_to_subsample = np.random.choice(dim_y, size=num_cols_subsample, replace=False)
    cols_to_subsample = np.sort(cols_to_subsample)

    available_snapshots = np.arange(dim_t - 1)
    snapshots_to_subsample = np.random.choice(
        available_snapshots, 
        size=num_snapshots_subsample - 1, 
        replace=False
    )
    snapshots_to_subsample = np.append(snapshots_to_subsample, dim_t - 1)
    snapshots_to_subsample = np.sort(snapshots_to_subsample)

    mask = np.zeros((dim_x, dim_y, dim_t), dtype=bool)

    for t in snapshots_to_subsample:
        mask[:, cols_to_subsample, t] = True

    total_points = dim_x * dim_y * dim_t
    masked_points = np.sum(mask)
    if masked_points / total_points > 0.95:
        return snapshot_subsampled

    snapshot_subsampled[mask] = 0

    if np.all(snapshot_subsampled == 0):
        return snapshot

    return snapshot_subsampled


def plot_dynamics_at_sensors(
    trace_A, num_sensors, locations="c", show_plot=False, seed=101,
    save_plot=False, save_path=None, file_name="plot_din.png", model_type="SHRED",
):
    """
    Select sensor locations and optionally plot their temporal dynamics and spatial positions.
    
    Parameters
    ----------
    trace_A : np.ndarray
        3D array (x, y, time) representing the field.
    num_sensors : int
        Number of sensors to select.
    locations : str
        Sensor placement strategy ('a', 'b', or 'c').
    show_plot : bool
        Whether to display the plot interactively.
    seed : int
        Random seed for reproducibility.
    save_plot : bool
        Whether to save the plot to disk.
    save_path : str
        Directory to save the plot.
    file_name : str
        Name of the plot file.
    model_type : str
        Model type for file naming.
    
    Returns
    -------
    sensor_locations : np.ndarray
        Indices of selected sensor locations.
    sensor_positions_x : np.ndarray
        X coordinates of sensors (normalized).
    sensor_positions_y : np.ndarray
        Y coordinates of sensors (normalized).
    """
    np.random.seed(seed)

    dim_x, dim_y, dim_t = trace_A.shape

    if locations == "a":
        central_x = 0.5
        central_y = 0.5
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

        sensor_positions_x = np.append(sensor_positions_x, central_x)
        sensor_positions_y = np.append(sensor_positions_y, central_y)

    elif locations == "b":
        sensor_positions_x = [0.5]
        sensor_positions_y = [0.5]
        sensor_locations = None

    else:
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

    # Plot and save if requested
    if save_plot and save_path:
        sensor_temperature_history = []
        for t in range(dim_t):
            sensor_temperatures = [
                trace_A[int(dim_x * x), int(dim_y * y), t]
                for x, y in zip(sensor_positions_x, sensor_positions_y)
            ]
            sensor_temperature_history.append(sensor_temperatures)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        y = np.linspace(0, 1, int(dim_x))
        x = np.linspace(0, 1, int(dim_y))
        X, Y = np.meshgrid(x, y)

        cmap = ax1.pcolormesh(
            X, Y, trace_A[:, :, -1].real, shading="auto", cmap="viridis"
        )
        fig.colorbar(cmap, ax=ax1, label=r"$|v_y|$")
        ax1.set_title("Velocity field $|v_y|$")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        ax1.scatter(
            sensor_positions_x,
            sensor_positions_y,
            color="k",
            label="Sensor Positions",
        )
        ax1.legend()

        sensor_temperature_history = np.array(sensor_temperature_history)
        for i, sensor_data in enumerate(sensor_temperature_history.T):
            ax2.plot(range(dim_t), sensor_data, label=f"Sensor {i+1}")

        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Amplitude Velocity $v_y$")
        ax2.set_title("Dynamics at Sensor Positions")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, file_name)
        plt.savefig(save_file)
        plt.savefig(save_file.replace('.png', '.pdf'))
        print(f"Plot saved to {save_file}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return sensor_locations, sensor_positions_x, sensor_positions_y




def evaluate_model(model, test_dataset, test_dataset_test, sc, sc_test, batch_size=64):
    """Inverse-transform predictions with sc_test; CSSHREDLAGS needs contiguous batch forward."""
    if isinstance(model, models.CSSHREDLAGS):
        raw = models.forward_csshred_contiguous_batches(
            model, test_dataset.X, batch_size,
        )
    else:
        raw = model(test_dataset.X)
    test_recons = sc_test.inverse_transform(raw.detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc_test.inverse_transform(test_dataset_test.Y.detach().cpu().numpy())

    error_norm = np.linalg.norm(test_recons - test_ground_truth_test) / np.linalg.norm(
        test_ground_truth_test
    )

    ssim_score = ssim(
        test_ground_truth_test, test_recons, data_range=test_recons.max() - test_recons.min()
    )
    last_snapshot_idx = test_recons.shape[0] - 1
    last_ground_truth = test_ground_truth_test[last_snapshot_idx]
    last_reconstruction = test_recons[last_snapshot_idx]
    ssim_last_snapshot = ssim(
        last_ground_truth,
        last_reconstruction,
        data_range=last_ground_truth.max() - last_ground_truth.min(),
    )
    mse_last = mean_squared_error(last_ground_truth.flatten(), last_reconstruction.flatten())
    max_pixel_last = np.max(last_ground_truth)
    psnr_last_snapshot = (
        20 * log10(max_pixel_last / np.sqrt(mse_last)) if mse_last > 0 else float("inf")
    )
    error_norm_last_snapshot = np.linalg.norm(last_reconstruction - last_ground_truth) / np.linalg.norm(
        last_ground_truth
    )

    print("Normalized Error (global):", error_norm)
    print("SSIM (global):", ssim_score)
    print("SSIM (last snapshot):", ssim_last_snapshot)
    print("PSNR (last snapshot):", psnr_last_snapshot, "dB")
    print("Normalized Error (last snapshot):", error_norm_last_snapshot)

    return (
        test_recons,
        test_ground_truth,
        test_ground_truth_test,
        error_norm,
        ssim_score,
        ssim_last_snapshot,
        psnr_last_snapshot,
        error_norm_last_snapshot,
    )


def objective(trial):
    npy_file_path = _default_turb_npy_path()
    global global_results_dir
    results_dir = global_results_dir
    os.makedirs(results_dir, exist_ok=True)

    matrix = load_data(npy_file_path, time_slice=TURB_TIME_SLICE)

    model_type = ABLATION_MODEL_TYPE
    seed = 915
    
    # ============================================================================
    # REPRODUCIBILITY: Set all random seeds for deterministic results
    # ============================================================================
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

    # --- Architecture ---
    # With the corrected forward (LSTM sees (batch, lags, sensors) properly),
    # larger architectures should now converge; include paper value (256, 3).
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    hidden_layers = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # --- Optimizer ---
    # Paper used lr=1.49e-3; widen range but center lower than before.
    lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
    # --- Loss weights (conditioned on ABLATION_LOSS_MODE) ---
    lambL2 = trial.suggest_float("lambL2", 1e-5, 1, log=True)
    if ABLATION_LOSS_MODE in ("full", "mse_l1"):
        lambL1 = trial.suggest_float("lambL1", 1e-5, 1, log=True)
    else:
        lambL1 = 0.0
    if ABLATION_LOSS_MODE in ("full", "mse_snr"):
        lambdaSNR = trial.suggest_float("lambdaSNR", 1e-2, 2, log=True)
    else:
        lambdaSNR = 0.0
    # --- Decoder ---
    l1 = trial.suggest_categorical("l1", [300, 400, 500])
    l2 = trial.suggest_categorical("l2", [300, 400, 500])
    # --- Data ---
    lags = trial.suggest_categorical("lags", [15, 20, 40])
    num_sensors = trial.suggest_categorical("num_sensors", [5])
    num_epochs = trial.suggest_int("num_epochs", 500, 2000)
    step_epoch = trial.suggest_int("step_epoch", 10, 50)
    _dev_epoch_cap_raw = os.environ.get("OPTUNA_DEV_EPOCH_CAP", "").strip()
    if _dev_epoch_cap_raw:
        _cap = int(_dev_epoch_cap_raw)
        if _cap < 1:
            raise ValueError("OPTUNA_DEV_EPOCH_CAP must be >= 1, got %s" % _dev_epoch_cap_raw)
        num_epochs = min(int(num_epochs), _cap)
    if model_type == "CSSHREDLAGS":
        l1_tol = trial.suggest_float("l1_tol", 1e-5, 1, log=True)
        opt_tol = trial.suggest_float("opt_tol", 1e-5, 1, log=True)
        ls_tol = trial.suggest_float("ls_tol", 1e-5, 1, log=True)
    else:
        l1_tol = 1e-3
        opt_tol = 1e-4
        ls_tol = 1e-4
    # --- Regularization ---
    # Was fixed at [0.01, 0.011]; now explore meaningful range.
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    print(
        f"Trial {trial.number}: hidden={hidden_size} layers={hidden_layers} batch={batch_size} "
        f"lr={lr:.6f} lags={lags} epochs={num_epochs}"
    )

    global all_data_in, train_data_in, valid_data_in, test_data_in
    global train_data_out, valid_data_out, test_data_out, train_dataset, valid_dataset, test_dataset

    snapshot = matrix.copy()
    num_cols_subsample = int(snapshot.shape[2] * TURB_COL_SUB_FRAC)
    num_snapshots_subsample = int(snapshot.shape[0] * TURB_SNAP_SUB_FRAC)
    snapshot = subsample(snapshot, num_cols_subsample, num_snapshots_subsample)

    sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
        snapshot,
        num_sensors,
        locations="c",
        show_plot=False,
        seed=seed,
        model_type=model_type,
    )

    trace_A = snapshot.copy()
    # Full field for targets: matrix is (time, x, y); transpose to (x, y, time).
    trace_A_ori = np.transpose(matrix.copy(), (1, 2, 0))

    dim_x, dim_y, dim_t = trace_A.shape
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])

    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T

    n = load_X.shape[0]
    m = load_X.shape[1]

    if n - lags <= 0:
        raise ValueError("Invalid lags: exceeds data length.")

    total_size = train_size + val_size + test_size
    if total_size > n - lags:
        train_size = (train_size * (n - lags)) // total_size
        val_size = (val_size * (n - lags)) // total_size
        test_size = (test_size * (n - lags)) // total_size

    np.random.seed(seed)

    last_snapshot_idx = n - lags - 1

    available_indices = np.arange(0, n - lags)
    available_indices = available_indices[available_indices != last_snapshot_idx]
    
    train_indices = np.random.choice(available_indices, size=train_size, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    mask[last_snapshot_idx] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]

    valid_indices = valid_test_indices[:val_size]
    test_indices = valid_test_indices[val_size : val_size + test_size]
    if last_snapshot_idx not in test_indices:
        if len(test_indices) > 0:
            test_indices = test_indices[:-1]
        test_indices = np.append(test_indices, last_snapshot_idx)

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    # Re-zero: preserve subsampling zeros after MinMaxScaler so that
    # recover_signal can detect missing entries in the 1-D batch signal.
    _sub_mask = (load_X == 0)
    transformed_X[_sub_mask] = 0.0

    sc_test = MinMaxScaler()
    sc_test = sc_test.fit(load_X_test[train_indices])
    transformed_X_test = sc_test.transform(load_X_test)

    all_data_in = np.zeros((n - lags, lags, num_sensors))
    all_data_in_test = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(
        device
    )
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(
        device
    )
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(
        device
    )
    # Model input from subsampled field; targets from full (scaled) field.
    test_data_in_test = torch.tensor(
        all_data_in[test_indices], dtype=torch.float32
    ).to(device)

    train_data_out = torch.tensor(
        transformed_X_test[train_indices + lags - 1], dtype=torch.float32
    ).to(device)
    valid_data_out = torch.tensor(
        transformed_X_test[valid_indices + lags - 1], dtype=torch.float32
    ).to(device)
    test_data_out = torch.tensor(
        transformed_X[test_indices + lags - 1], dtype=torch.float32
    ).to(device)
    test_data_out_test = torch.tensor(
        transformed_X_test[test_indices + lags - 1], dtype=torch.float32
    ).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    test_dataset_test = TimeSeriesDataset(test_data_in_test, test_data_out_test)

    train_error = None
    if model_type == "CSSHREDLAGS":
        model = models.CSSHREDLAGS(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
            l1_tol=l1_tol,
            opt_tol=opt_tol,
            ls_tol=ls_tol,
            n_sparsity_threshold=CS_N_SPARSITY_THRESHOLD,
            verbosity=0,
            basis=CSSHREDLAGS_BASIS,
            cache_max_entries=CSSHREDLAGS_CACHE_MAX,
        ).to(device)
        if CSSHREDLAGS_TRAIN_LOSS == "full":
            train_error, validation_errors = models.fit_csshred_model(
                model,
                train_dataset,
                valid_dataset,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                lambL2=lambL2,
                step_epoch=step_epoch,
                lambL1=lambL1,
                lambdaSNR=lambdaSNR,
                verbose=False,
                patience=15,
                generator=generator,
            )
        else:
            validation_errors = models.fit(
                model,
                train_dataset,
                valid_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                step_epoch=step_epoch,
                verbose=False,
                patience=15,
                generator=generator,
            )
    elif model_type == "SHRED":
        model = models.SHRED(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
        ).to(device)
        validation_errors = models.fit(
            model,
            train_dataset,
            valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            step_epoch=step_epoch,
            verbose=False,
            patience=15,
            generator=generator,
        )
    elif model_type == "SHRED-COMPOSITE":
        # SHRED backbone without CS block; same optimizer/loss path as CS-SHRED (fit_csshred_model).
        model = models.SHRED(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
        ).to(device)
        train_error, validation_errors = models.fit_csshred_model(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambL2=lambL2,
            step_epoch=step_epoch,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            verbose=False,
            patience=15,
            generator=generator,
        )
    else:
        raise RuntimeError("unexpected model_type: %s" % model_type)

    validation_errors = [float(val) for val in validation_errors]

    test_recons, test_ground_truth, test_ground_truth_test, error_norm, ssim_score, ssim_last_snapshot, psnr_last_snapshot, error_norm_last_snapshot = evaluate_model(
        model, test_dataset, test_dataset_test, sc, sc_test, batch_size=batch_size,
    )

    best_trial_artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
    os.makedirs(best_trial_artifacts_dir, exist_ok=True)

    # Composite objective: align with MSE-heavy training; emphasize last snapshot via beta.
    alpha = 1.0
    beta = 1.5
    composite_metric = alpha * error_norm + beta * error_norm_last_snapshot

    best_trial_file = os.path.join(results_dir, "current_best_trial.json")
    is_best = False
    
    if os.path.exists(best_trial_file):
        with open(best_trial_file, 'r') as f:
            best_trial_info = json.load(f)
        best_composite = best_trial_info.get("composite_metric", float('inf'))
        if composite_metric < best_composite:
            is_best = True
            print(
                f"Trial {trial.number} new best: composite={composite_metric:.6f} "
                f"(was {best_composite:.6f})"
            )
        else:
            print(
                f"Trial {trial.number} skipped: composite={composite_metric:.6f} "
                f">= {best_composite:.6f}"
            )
    else:
        is_best = True
        print(f"Trial {trial.number} first candidate: composite={composite_metric:.6f}")
    
    if is_best:
        if os.path.exists(best_trial_file):
            # Remove previous best artifacts before overwriting.
            for file in ["test_recons.npy", "test_ground_truth.npy", "matrix.npy", "snapshot.npy",
                        "sensor_positions_x.npy", "sensor_positions_y.npy", "validation_errors.npy", "train_error.npy"]:
                file_path = os.path.join(best_trial_artifacts_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        np.save(os.path.join(best_trial_artifacts_dir, "test_recons.npy"), test_recons)
        np.save(os.path.join(best_trial_artifacts_dir, "test_ground_truth.npy"), test_ground_truth_test)
        np.save(os.path.join(best_trial_artifacts_dir, "matrix.npy"), matrix)
        np.save(os.path.join(best_trial_artifacts_dir, "snapshot.npy"), snapshot)
        np.save(os.path.join(best_trial_artifacts_dir, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(best_trial_artifacts_dir, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(best_trial_artifacts_dir, "validation_errors.npy"), validation_errors)
        if train_error is not None:
            np.save(os.path.join(best_trial_artifacts_dir, "train_error.npy"), train_error)
        
        alpha = 1.0
        beta = 1.5
        composite_metric = alpha * error_norm + beta * error_norm_last_snapshot
        best_trial_info = {
            "trial_number": trial.number,
            "composite_metric": float(composite_metric),
            "error_norm": float(error_norm),
            "ssim_score": float(ssim_score),
            "ssim_last_snapshot": float(ssim_last_snapshot),
            "psnr_last_snapshot": float(psnr_last_snapshot),
            "error_norm_last_snapshot": float(error_norm_last_snapshot),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        with open(best_trial_file, 'w') as f:
            json.dump(best_trial_info, f, indent=4)
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "w") as f:
        results = {
            "trial_number": trial.number,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
            "batch_size": batch_size,
            "lr": float(lr),
            "lambL2": float(lambL2),
            "lambL1": float(lambL1),
            "lambdaSNR": float(lambdaSNR),
            "dropout": float(dropout), 
            "l1_tol": float(l1_tol),
            "opt_tol": float(opt_tol),
            "ls_tol": float(ls_tol),
            "l1": l1,
            "l2": l2,
            "lags": lags,
            "num_sensors": num_sensors,
            "num_epochs": num_epochs,  
            "step_epoch": step_epoch,
            "error_norm": float(error_norm),
            "validation_errors": np.mean(validation_errors),           
            "ssim_score": float(ssim_score),
            "ssim_last_snapshot": float(ssim_last_snapshot),
            "psnr_last_snapshot": float(psnr_last_snapshot),
            "error_norm_last_snapshot": float(error_norm_last_snapshot),
            
        }
        json.dump(results, f, indent=4)

    alpha = 1.0
    beta = 1.5
    composite_metric = alpha * error_norm + beta * error_norm_last_snapshot
    print(
        f"Trial {trial.number} done: composite={composite_metric:.6f} "
        f"err={error_norm:.6f} err_last={error_norm_last_snapshot:.6f}"
    )
    
    return float(composite_metric)


def save_best_trial_artifacts(best_trial, results_dir, npy_file_path, model_type="SHRED", seed=915):
    """
    Re-run the best trial and save numpy arrays and figures for notebooks.

    model_type: CSSHREDLAGS, SHRED, or SHRED-COMPOSITE (must match the study).
    """
    artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"save_best_trial_artifacts: writing to {artifacts_dir}")
    
    # ============================================================================
    # REPRODUCIBILITY: Set all random seeds
    # ============================================================================
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    hidden_size = best_trial.params.get("hidden_size")
    hidden_layers = best_trial.params.get("hidden_layers")
    batch_size = best_trial.params.get("batch_size")
    lr = best_trial.params.get("lr")
    lambL2 = best_trial.params.get("lambL2")
    lambL1 = best_trial.params.get("lambL1")
    lambdaSNR = best_trial.params.get("lambdaSNR")
    l1 = best_trial.params.get("l1")
    l2 = best_trial.params.get("l2")
    lags = best_trial.params.get("lags")
    num_sensors = best_trial.params.get("num_sensors")
    num_epochs = best_trial.params.get("num_epochs")
    step_epoch = best_trial.params.get("step_epoch")
    dropout = best_trial.params.get("dropout")
    l1_tol = best_trial.params.get("l1_tol")
    opt_tol = best_trial.params.get("opt_tol")
    ls_tol = best_trial.params.get("ls_tol")
    
    matrix = load_data(npy_file_path, time_slice=TURB_TIME_SLICE)
    snapshot = matrix.copy()
    num_cols_subsample = int(snapshot.shape[2] * TURB_COL_SUB_FRAC)
    num_snapshots_subsample = int(snapshot.shape[0] * TURB_SNAP_SUB_FRAC)
    snapshot = subsample(snapshot, num_cols_subsample, num_snapshots_subsample)

    visualize_data(matrix, snapshot, artifacts_dir, save_plots=True)

    sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
        snapshot,
        num_sensors,
        locations="c",
        show_plot=False,
        seed=seed,
        save_plot=True,
        save_path=artifacts_dir,
        file_name=f"plot_din_{model_type}.png",
        model_type=model_type,
    )

    trace_A = snapshot.copy()
    # Original data needs to be transposed to (x, y, time) format
    trace_A_ori = np.transpose(matrix, (1, 2, 0))  # Original data (x, y, time)
    
    dim_x, dim_y, dim_t = trace_A.shape
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])
    
    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T
    
    n = load_X.shape[0]
    m = load_X.shape[1]  # Number of spatial points (flattened)
    
    total_size = train_size + val_size + test_size
    if total_size > n - lags:
        train_size = (train_size * (n - lags)) // total_size
        val_size = (val_size * (n - lags)) // total_size
        test_size = (test_size * (n - lags)) // total_size
    
    np.random.seed(seed)

    last_snapshot_idx = n - lags - 1

    available_indices = np.arange(0, n - lags)
    available_indices = available_indices[available_indices != last_snapshot_idx]
    
    train_indices = np.random.choice(available_indices, size=train_size, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    mask[last_snapshot_idx] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
    
    valid_indices = valid_test_indices[:val_size]
    test_indices = valid_test_indices[val_size : val_size + test_size]
    if last_snapshot_idx not in test_indices:
        if len(test_indices) > 0:
            test_indices = test_indices[:-1]
        test_indices = np.append(test_indices, last_snapshot_idx)
    
    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    _sub_mask = (load_X == 0)
    transformed_X[_sub_mask] = 0.0

    sc_test = MinMaxScaler()
    sc_test = sc_test.fit(load_X_test[train_indices])
    transformed_X_test = sc_test.transform(load_X_test)
    
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    all_data_in_test = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]
    
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
    test_data_in_test = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X_test[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X_test[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out_test = torch.tensor(transformed_X_test[test_indices + lags - 1], dtype=torch.float32).to(device)
    
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    test_dataset_test = TimeSeriesDataset(test_data_in_test, test_data_out_test)
    
    _basis = os.environ.get("CSSHREDLAGS_BASIS", "fft").strip().lower()
    if _basis not in ("fft", "dct"):
        _basis = "fft"
    _cache_max = int(os.environ.get("CSSHREDLAGS_CACHE_MAX", "100000"))
    _tl = os.environ.get("CSSHREDLAGS_TRAIN_LOSS", "mse").strip().lower()
    if _tl not in ("mse", "full"):
        _tl = "mse"

    train_error = None
    if model_type == "CSSHREDLAGS":
        model = models.CSSHREDLAGS(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
            l1_tol=l1_tol,
            opt_tol=opt_tol,
            ls_tol=ls_tol,
            n_sparsity_threshold=CS_N_SPARSITY_THRESHOLD,
            verbosity=0,
            basis=_basis,
            cache_max_entries=_cache_max,
        ).to(device)
        if _tl == "full":
            train_error, validation_errors = models.fit_csshred_model(
                model,
                train_dataset,
                valid_dataset,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                lambL2=lambL2,
                step_epoch=step_epoch,
                lambL1=lambL1,
                lambdaSNR=lambdaSNR,
                verbose=False,
                patience=15,
                generator=generator,
            )
        else:
            validation_errors = models.fit(
                model,
                train_dataset,
                valid_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                step_epoch=step_epoch,
                verbose=False,
                patience=15,
                generator=generator,
            )
    elif model_type == "SHRED":
        model = models.SHRED(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
        ).to(device)
        validation_errors = models.fit(
            model,
            train_dataset,
            valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            step_epoch=step_epoch,
            verbose=False,
            patience=15,
            generator=generator,
        )
    elif model_type == "SHRED-COMPOSITE":
        model = models.SHRED(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
        ).to(device)
        train_error, validation_errors = models.fit_csshred_model(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambL2=lambL2,
            step_epoch=step_epoch,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            verbose=False,
            patience=15,
            generator=generator,
        )
    else:
        raise ValueError(
            "model_type must be CSSHREDLAGS, SHRED, or SHRED-COMPOSITE, got: %s" % model_type
        )

    validation_errors = [float(val) for val in validation_errors]

    test_recons, test_ground_truth, test_ground_truth_test, error_norm, ssim_score, ssim_last_snapshot, psnr_last_snapshot, error_norm_last_snapshot = evaluate_model(
        model, test_dataset, test_dataset_test, sc, sc_test, batch_size=batch_size,
    )

    np.save(os.path.join(artifacts_dir, "test_recons.npy"), test_recons)
    np.save(os.path.join(artifacts_dir, "test_ground_truth.npy"), test_ground_truth_test)  # Original data
    np.save(os.path.join(artifacts_dir, "matrix.npy"), matrix)
    np.save(os.path.join(artifacts_dir, "snapshot.npy"), snapshot)
    np.save(os.path.join(artifacts_dir, "sensor_positions_x.npy"), sensor_positions_x)
    np.save(os.path.join(artifacts_dir, "sensor_positions_y.npy"), sensor_positions_y)
    np.save(os.path.join(artifacts_dir, "validation_errors.npy"), validation_errors)
    if train_error is not None:
        np.save(os.path.join(artifacts_dir, "train_error.npy"), train_error)

    print(
        f"save_best_trial_artifacts done: err={error_norm:.6f} "
        f"ssim_g={ssim_score:.6f} psnr_last={psnr_last_snapshot:.2f} dB"
    )


base_results_dir = _default_optuna_results_base()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if ABLATION_MODEL_TYPE == "CSSHREDLAGS":
    _model_run_suffix = "_csshredlags"
    if CSSHREDLAGS_TRAIN_LOSS != "full":
        _model_run_suffix += "_mse"
elif ABLATION_MODEL_TYPE == "SHRED-COMPOSITE":
    _model_run_suffix = "_shred_composite"
else:
    _model_run_suffix = "_shred"
_mode_suffix = "_%s" % ABLATION_LOSS_MODE if ABLATION_LOSS_MODE != "full" else ""
results_dir = os.path.join(
    base_results_dir, "test_%s%s%s" % (timestamp, _model_run_suffix, _mode_suffix)
)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

global_results_dir = results_dir

study = optuna.create_study(direction="minimize")
_n_trials = int(os.environ.get("OPTUNA_N_TRIALS", "20"))
study.optimize(objective, n_trials=_n_trials)

print("Best trial:")
trial = study.best_trial

best_params = {
    "trial_number": trial.number,
    "model_type": ABLATION_MODEL_TYPE,
    "loss_mode": ABLATION_LOSS_MODE,
    "hidden_size": trial.params.get("hidden_size"),
    "hidden_layers": trial.params.get("hidden_layers"),
    "batch_size": trial.params.get("batch_size"),
    "lr": float(trial.params.get("lr")),
    "lambL2": float(trial.params.get("lambL2")),
    "lambL1": float(trial.params.get("lambL1", 0.0)),
    "lambdaSNR": float(trial.params.get("lambdaSNR", 0.0)),
    "l1": trial.params.get("l1"),
    "l2": trial.params.get("l2"),
    "lags": trial.params.get("lags"),
    "num_sensors": trial.params.get("num_sensors"),
    "num_epochs": trial.params.get("num_epochs"),
    "dropout": float(trial.params.get("dropout")),
    "step_epoch": trial.params.get("step_epoch"),
    "n_sparsity_threshold": float(CS_N_SPARSITY_THRESHOLD),
}
if ABLATION_MODEL_TYPE == "CSSHREDLAGS":
    best_params["l1_tol"] = float(trial.params.get("l1_tol"))
    best_params["opt_tol"] = float(trial.params.get("opt_tol"))
    best_params["ls_tol"] = float(trial.params.get("ls_tol"))
    best_params["csshredlags_train_loss"] = CSSHREDLAGS_TRAIN_LOSS
    best_params["csshredlags_basis"] = CSSHREDLAGS_BASIS
    best_params["csshredlags_cache_max"] = CSSHREDLAGS_CACHE_MAX

results_file = os.path.join(results_dir, f"{trial.number}_results.json")
with open(results_file, "r") as f:
    results = json.load(f)

current_best_file = os.path.join(results_dir, "current_best_trial.json")
composite_metric = None
alpha = 1.0
beta = 1.5
if os.path.exists(current_best_file):
    with open(current_best_file, 'r') as f:
        current_best_info = json.load(f)
    composite_metric = current_best_info.get("composite_metric")
    alpha = current_best_info.get("alpha", 1.0)
    beta = current_best_info.get("beta", 0.5)

best_params.update({
    "error_norm": results.get("error_norm"),
    "ssim_score": results.get("ssim_score"),
    "ssim_last_snapshot": results.get("ssim_last_snapshot"),
    "psnr_last_snapshot": results.get("psnr_last_snapshot"),
    "error_norm_last_snapshot": results.get("error_norm_last_snapshot"),
    "composite_metric": float(composite_metric) if composite_metric is not None else None,
    "alpha": float(alpha),
    "beta": float(beta),
})

with open(os.path.join(results_dir, "best_trial_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

print("Best trial parameters and metrics saved to 'best_trial_params.json'")

seed = 915
best_trial_artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
os.makedirs(best_trial_artifacts_dir, exist_ok=True)

best_trial_file = os.path.join(results_dir, "current_best_trial.json")
if os.path.exists(best_trial_file):
    with open(best_trial_file, 'r') as f:
        best_trial_info = json.load(f)
    best_trial_number = best_trial_info.get("trial_number")
else:
    print("warning: current_best_trial.json missing; using Optuna best trial id")
    best_trial_number = trial.number

if os.path.exists(os.path.join(best_trial_artifacts_dir, "test_recons.npy")):
    if os.path.exists(best_trial_file):
        optuna_metrics = {
            "SSIM_global": float(best_trial_info.get("ssim_score", 0.0)),
            "SSIM_last_snapshot": float(best_trial_info.get("ssim_last_snapshot", 0.0)),
            "PSNR_last_snapshot": float(best_trial_info.get("psnr_last_snapshot", 0.0)),
            "Normalized_Error_global": float(best_trial_info.get("error_norm", 0.0)),
            "Normalized_Error_last_snapshot": float(best_trial_info.get("error_norm_last_snapshot", 0.0)),
            "source": "Optuna (during optimization)",
            "trial_number": int(best_trial_number)
        }
    else:
        optuna_metrics = {
            "SSIM_global": float(results.get("ssim_score", 0.0)),
            "SSIM_last_snapshot": float(results.get("ssim_last_snapshot", 0.0)),
            "PSNR_last_snapshot": float(results.get("psnr_last_snapshot", 0.0)),
            "Normalized_Error_global": float(results.get("error_norm", 0.0)),
            "Normalized_Error_last_snapshot": float(results.get("error_norm_last_snapshot", 0.0)),
            "source": "Optuna (during optimization)",
            "trial_number": int(trial.number)
        }
    
    try:
        import lpips_eval

        tre = np.load(os.path.join(best_trial_artifacts_dir, "test_recons.npy"))
        tgt = np.load(os.path.join(best_trial_artifacts_dir, "test_ground_truth.npy"))
        mtx = np.load(os.path.join(best_trial_artifacts_dir, "matrix.npy"))
        _, dim_hp, dim_wp = mtx.shape
        lp_mean, lp_last = lpips_eval.lpips_test_metrics(
            np.asarray(tre),
            np.asarray(tgt),
            int(dim_hp),
            int(dim_wp),
        )
        optuna_metrics["LPIPS_global_mean"] = float(lp_mean)
        optuna_metrics["LPIPS_last_snapshot"] = float(lp_last)
        print(f"LPIPS Alex mean={lp_mean:.6f} last={lp_last:.6f}")
    except Exception as exc:
        print(f"warning: could not compute LPIPS for results.json: {exc}")

    results_json_path = os.path.join(best_trial_artifacts_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(optuna_metrics, f, indent=4)

    matrix = np.load(os.path.join(best_trial_artifacts_dir, "matrix.npy"))
    snapshot = np.load(os.path.join(best_trial_artifacts_dir, "snapshot.npy"))
    
    visualize_data(matrix, snapshot, best_trial_artifacts_dir, save_plots=True)

    model_type = ABLATION_MODEL_TYPE
    plot_dynamics_at_sensors(
        snapshot,
        trial.params.get("num_sensors"),
        locations="c",
        show_plot=False,
        seed=seed,
        save_plot=True,
        save_path=best_trial_artifacts_dir,
        file_name=f"plot_din_{model_type}.png",
        model_type=model_type,
    )

else:
    print(
        f"warning: best-trial artifacts missing under {best_trial_artifacts_dir}; "
        "running save_best_trial_artifacts fallback (metrics may differ slightly)."
    )
    npy_file_path = _default_turb_npy_path()
    save_best_trial_artifacts(
        trial, results_dir, npy_file_path, model_type=ABLATION_MODEL_TYPE, seed=915,
    )

