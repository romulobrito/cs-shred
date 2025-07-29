import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
import json
import os
import time

import models
from processdata import TimeSeriesDataset

# data: velo_256 velo_257.h5 velo_258.h5
# source: https://smart-turb.roma2.infn.it/init/routes/#/logging/view_dataset/1/tabfile


# Path to the .npy file
npy_file_path = r"./data/turb_vy_combined.npy"
save_path = r"./turb-git-csshred"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Function to load data from a .npy file
def load_data(npy_file_path, time_slice):
    """
    Load a 3D spatiotemporal array from a .npy file, truncate it to a given number of time steps,
    and transpose it to (time, x, y) format.

    Parameters
    ----------
    npy_file_path : str
        Path to the .npy file containing the data.
    time_slice : int
        Number of time steps to keep (truncate along the last axis).

    Returns
    -------
    data_array : np.ndarray
        Array of shape (time, x, y) with the loaded and transposed data.
    """
    data_array = np.load(npy_file_path)
    data_array = data_array[:, :, :time_slice]
    data_array = np.transpose(data_array, (2, 0, 1))
    print("Loaded data dimensions:", data_array.shape)
    return data_array


# Visualization of 2D or 3D data
def visualize_data(matrix, subsampled):
    """
    Visualize the last temporal slice of the original and subsampled data arrays,
    saving the plots as PNG files.

    Parameters
    ----------
    matrix : np.ndarray
        Original data array of shape (time, x, y).
    subsampled : np.ndarray
        Subsampled data array of shape (x, y, time).
    """
    # Plot for the last temporal slice of the matrix
    plt.imshow(matrix[-1, :, :], cmap="viridis", origin="lower")
    # plt.colorbar()
    plt.title("Last Temporal Slice")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(save_path + r"/last_temporal_slice.png")
    plt.show()

    # Plot for the last temporal slice of the subsampled matrix
    plt.imshow(subsampled[:, :, -1], cmap="viridis", origin="lower")
    # plt.colorbar()
    plt.title("Last Temporal Slice (Subsampled)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(save_path + r"/last_temporal_slice_subsampled.png")
    plt.show()


def subsample(snapshot, num_cols_subsample, num_snapshots_subsample):
    """
    Randomly subsample columns and time snapshots from a 3D array, setting selected entries to zero.

    Parameters
    ----------
    snapshot : np.ndarray
        Input array of shape (time, x, y).
    num_cols_subsample : int
        Number of columns to subsample (set to zero).
    num_snapshots_subsample : int
        Number of time snapshots to subsample (set to zero).

    Returns
    -------
    snapshot_subsampled : np.ndarray
        Subsampled array with selected entries set to zero.
    """
    np.random.seed(1001)

    print("snapshot", snapshot.shape)

    snapshot = np.transpose(snapshot, (1, 2, 0))
    print("snapshot after transpose", snapshot.shape)
    dim_x, dim_y, dim_t = snapshot.shape
    snapshot_subsampled = snapshot.copy()

    # Ensure that num_snapshots_subsample is less than dim_t
    num_snapshots_subsample = min(num_snapshots_subsample, dim_t - 1)

    # Randomly choose the columns to be subsampled
    # Ensure that no column is subsampled
    num_cols_subsample = min(
        num_cols_subsample, dim_y - 1
    )  # Always keep at least one column
    cols_to_subsample = np.random.choice(dim_y, size=num_cols_subsample, replace=False)
    cols_to_subsample = np.sort(cols_to_subsample)

    # Randomly choose the snapshots, excluding the first one initially
    available_snapshots = np.arange(dim_t - 1)
    snapshots_to_subsample = np.random.choice(
        available_snapshots, size=num_snapshots_subsample - 1, replace=False
    )
    # Add the last snapshot
    snapshots_to_subsample = np.append(snapshots_to_subsample, dim_t - 1)
    snapshots_to_subsample = np.sort(snapshots_to_subsample)

    # Create a mask initially False (keep all data)
    mask = np.zeros((dim_x, dim_y, dim_t), dtype=bool)

    # Apply subsampling only on the selected columns and snapshots
    for t in snapshots_to_subsample:
        mask[:, cols_to_subsample, t] = True

    # Check if we are not zeroing too many data
    total_points = dim_x * dim_y * dim_t
    masked_points = np.sum(mask)
    if masked_points / total_points > 0.95:  # If more than 95% of the points are masked
        print("Warning: Too many points are being masked. Adjusting...")
        return snapshot_subsampled  # Return without subsampling

    # Apply the mask
    snapshot_subsampled[mask] = 0

    # Final check
    if np.all(snapshot_subsampled == 0):
        print("Warning: All values were zeroed. Returning original data...")
        return snapshot

    print("Shape of the snapshot after subsampling:", snapshot_subsampled.shape)
    print(f"Percentage of data kept: {100 * (1 - np.sum(mask)/mask.size):.2f}%")
    return snapshot_subsampled


# Configuration of the sensors
def plot_dynamics_at_sensors(
    trace_A,
    num_sensors,
    locations="c",
    show_plot=False,
    save_plot=True,
    save_path=save_path,
    file_name="plot_din.png",
    seed=101,
    auto_close_time=5,
):
    """
    Select sensor locations and plot their temporal dynamics and spatial positions.

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
    save_plot : bool
        Whether to save the plot to disk.
    save_path : str
        Directory to save the plot.
    file_name : str
        Name of the plot file.
    seed : int
        Random seed for reproducibility.
    auto_close_time : int
        Time to keep the plot open if shown interactively.

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

    print("Shape of the snapshot after subsampling:", trace_A.shape)

    if locations == "a":
        central_x, central_y = 0.5, 0.5
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

        sensor_positions_x = np.append(sensor_positions_x, central_x)
        sensor_positions_y = np.append(sensor_positions_y, central_y)

    elif locations == "b":
        sensor_positions_x, sensor_positions_y = [0.5], [0.5]

    else:
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

    sensor_temperature_history = []
    for t in range(dim_t):
        sensor_temperatures = [
            trace_A[int(dim_x * x), int(dim_y * y), t]
            for x, y in zip(sensor_positions_x, sensor_positions_y)
        ]
        sensor_temperature_history.append(sensor_temperatures)
    if show_plot or save_plot:
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

        if save_plot:
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = os.path.join(save_path, file_name)
            else:
                save_file = file_name
            plt.savefig(save_file)
            print(f"Plot saved to {save_file}")

        if show_plot:
            plt.show()
            # time.sleep(auto_close_time)  # wait for a specific time
            # plt.close(fig)
        else:
            plt.close(fig)

    return sensor_locations, sensor_positions_x, sensor_positions_y


# Preparation of the data for training and validation
def prepare_datasets(trace_A, trace_A_ori, num_sensors, sensor_locations, lags):
    """
    Prepare PyTorch datasets for training, validation, and testing from the original and subsampled data.

    Parameters
    ----------
    trace_A : np.ndarray
        Subsampled data array (x, y, time).
    trace_A_ori : np.ndarray
        Original data array (x, y, time).
    num_sensors : int
        Number of sensors.
    sensor_locations : np.ndarray
        Indices of sensor locations.
    lags : int
        Number of time lags for input sequences.

    Returns
    -------
    train_dataset : TimeSeriesDataset
        Training dataset.
    valid_dataset : TimeSeriesDataset
        Validation dataset.
    test_dataset_test : TimeSeriesDataset
        Test dataset.
    sc : MinMaxScaler
        Fitted scaler for normalization.
    load_X_shape_1 : int
        Number of spatial points (flattened).
    """
    trace_A_ori = np.transpose(trace_A_ori, (1, 2, 0))
    num_sensors = num_sensors

    dim_x, dim_y, dim_t = trace_A.shape

    lags = lags
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])

    # print("train_size", train_size)
    # print("val_size", val_size)
    # print("test_size", test_size)

    # print("trace_A", trace_A.shape)
    # print("trace_A", trace_A_ori.shape)

    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T

    load_X_shape_0, load_X_shape_1 = load_X.shape
    print(load_X_shape_0, load_X_shape_1)

    # Ensuring that the sizes do not exceed the real size of the data
    total_size = train_size + val_size + test_size
    if total_size > load_X_shape_0 - lags:
        train_size = (train_size * (load_X_shape_0 - lags)) // total_size
        val_size = (val_size * (load_X_shape_0 - lags)) // total_size
        test_size = (test_size * (load_X_shape_0 - lags)) // total_size

    train_indices = np.random.choice(
        load_X_shape_0 - lags, size=train_size, replace=False
    )
    mask = np.ones(load_X_shape_0 - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, load_X_shape_0 - lags)[np.where(mask != 0)[0]]

    valid_indices = valid_test_indices[:val_size]
    test_indices = valid_test_indices[val_size : val_size + test_size]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    transformed_X_test = sc.transform(load_X_test)

    all_data_in = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    all_data_in_test = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    for i in range(load_X_shape_0 - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]

    print("device:", device)
    print("Number of sensors:{}".format(num_sensors))

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(
        device
    )
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(
        device
    )
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(
        device
    )
    test_data_in_test = torch.tensor(
        all_data_in_test[test_indices], dtype=torch.float32
    ).to(device)

    train_data_out = torch.tensor(
        transformed_X[train_indices + lags - 1], dtype=torch.float32
    ).to(device)
    valid_data_out = torch.tensor(
        transformed_X[valid_indices + lags - 1], dtype=torch.float32
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

    return train_dataset, valid_dataset, test_dataset, test_dataset_test, sc, load_X_shape_1


# Training and validation of the model
def train_and_validate_model(
    type_model,
    model,
    train_dataset,
    valid_dataset,
    num_epochs,
    batch_size,
    lr,
    lambL2,
    lambL1,
    lambdaSNR,
    step_epoch,
    verbose,
    patience,
):
    """
    Train and validate a CS-SHRED or SHRED model.

    Parameters
    ----------
    type_model : str
        Model type ('CS-SHRED' or 'SHRED').
    model : torch.nn.Module
        Model instance.
    train_dataset : Dataset
        Training dataset.
    valid_dataset : Dataset
        Validation dataset.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    lambL2 : float
        L2 regularization coefficient.
    lambL1 : float
        L1 regularization coefficient.
    lambdaSNR : float
        SNR regularization coefficient.
    step_epoch : int
        Epoch interval for validation.
    verbose : bool
        Verbosity flag.
    patience : int
        Early stopping patience.

    Returns
    -------
    train_error : np.ndarray or None
        Training error history (CS-SHRED only).
    validation_errors : np.ndarray
        Validation error history.
    """
    if type_model == "CS-SHRED":
        train_error, validation_errors = models.fit_csshred_model(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambL2=lambL2,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            step_epoch=step_epoch,
            verbose=verbose,
            patience=patience,
        )
        return train_error, validation_errors
    else:
        validation_errors = models.fit(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            step_epoch=step_epoch,
            verbose=True,
            patience=patience,
        )
        return validation_errors


def evaluate_model(
    model, test_dataset, test_dataset_test, sc, json_save_path=save_path + "/error_results.json"
):
    """
    Evaluate the model on the test dataset, computing normalized error and SSIM,
    and save results to a JSON file.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    test_dataset : Dataset
        Test dataset (subsampled data).
    test_dataset_test : Dataset
        Test dataset (original data).
    sc : MinMaxScaler
        Scaler used for normalization.
    json_save_path : str
        Path to save the results JSON.

    Returns
    -------
    test_recons : np.ndarray
        Model reconstructions (denormalized).
    test_ground_truth_test : np.ndarray
        Ground truth data (denormalized, original).
    error_norm : float
        Normalized error.
    """
    # Perform the prediction with the model and transform the data back to the original format
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc.inverse_transform(test_dataset_test.Y.detach().cpu().numpy())

    # Check the dimensions of the arrays
    if test_recons.ndim != test_ground_truth_test.ndim:
        raise ValueError(
            "The dimensions of the reconstructed and ground truth data do not correspond."
        )

    # Calculate the normalized error using original data
    error_norm = np.linalg.norm(test_recons - test_ground_truth_test) / np.linalg.norm(
        test_ground_truth_test
    )

    # Calculate the SSIM for each snapshot using original data
    ssim_scores = []
    for i in range(test_recons.shape[0]):
        ssim_score = ssim(
            test_ground_truth_test[i],
            test_recons[i],
            data_range=test_ground_truth_test[i].max() - test_ground_truth_test[i].min(),
        )
        ssim_scores.append(ssim_score)
    
    # SSIM mean of all samples
    mean_ssim = np.mean(ssim_scores)
    
    # SSIM of the last snapshot (most important to evaluate final quality)
    last_snapshot_ssim = ssim_scores[-1]

    print("Mean SSIM (all samples):", mean_ssim)
    print("Last Snapshot SSIM:", last_snapshot_ssim)
    print("Normalized Error:", error_norm)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

    # Save the results in a JSON file
    results = {
        "Normalized_Error": float(error_norm),
        "SSIM": {
            "mean_all_samples": float(mean_ssim), 
            "last_snapshot": float(last_snapshot_ssim),
            "all_snapshots": ssim_scores
        },
    }
    with open(json_save_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return test_recons, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim


def add_model_info_to_json(json_file_path, model_type, model_params, config_params):
    """
    Add model and configuration information to a JSON file, creating it if necessary.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file.
    model_type : str
        Model type ('CS-SHRED' or 'SHRED').
    model_params : dict
        Model hyperparameters.
    config_params : dict
        Experiment configuration parameters.
    """
    # Check if the JSON file already exists
    if os.path.exists(json_file_path):
        # If it exists, load the content
        with open(json_file_path, "r") as json_file:
            results = json.load(json_file)
    else:
        # If it does not exist, create an empty dictionary
        results = {}

    # Add the new information
    results["Model_Type"] = model_type
    results["Model_Parameters"] = model_params
    results["Configuration_Parameters"] = config_params

    # Save the updated JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Model information added to {json_file_path}")


# Common parameters
seed = 915
verbose = True
patience = 15
np.random.seed(seed)
# Choice of model CS-SHRED/SHRED
model_type = "CS-SHRED"

# Loading the data
matrix = load_data(npy_file_path, time_slice=650)

begin_time = time.time()


# Subsampling and visualization of the data
num_cols_subsample = int(matrix.shape[2] * 0.3)  # % of columns will be subsampled
num_snapshots_subsample = int(
    matrix.shape[0] * 0.3
)  #  % of snapshots will be subsampled
snapshot = subsample(matrix, num_cols_subsample, num_snapshots_subsample)
# visualize_data(matrix, snapshot)



# Parameters optimized by Optuna 
hidden_size = 256
hidden_layers = 3
batch_size = 8
lr = 0.00016405841062596682
lambL2 = 0.5777770618168943
lambL1 = 9.888011620382175e-05
lambdaSNR = 0.8644875563549309
l1 = 500
l2 = 500
lags = 30
num_sensors = 5
num_epochs = 1312
step_epoch = 34
l1_tol = 0.0007228888870258154
opt_tol = 0.00017764675865100242
ls_tol = 0.11549664495934957
dropout = 0.007879458596486309
patience = 11


# SHRED Turb
# hidden_size	=128
# hidden_layers	=3
# batch_size	=128
# lr	=0.005878894964721222
# lambL2	=0.2358966667175767
# lambL1	=0.0032743141604346126
# lambdaSNR	=0.0764446281191567
# dropout	=0.01043954387600347
# l1_tol	=0.000036726695633784415
# opt_tol	=0.000023951009540287297
# ls_tol	=0.003605803632395063
# l1	=300
# l2	=500
# lags	=15
# num_sensors	=5
# num_epochs	=1871
# step_epoch	=23
# patience = 15


# Configuration of the sensors
sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
    snapshot,
    num_sensors,
    locations="c",
    show_plot=False,
    save_plot=True,
    save_path=save_path,
    file_name=f"plot_din_{model_type}.png",
    seed=seed,
)

# Preparation of the datasets
train_dataset, valid_dataset, test_dataset, test_dataset_test, sc, load_X_shape_1 = prepare_datasets(
    snapshot, matrix, num_sensors, sensor_locations, lags
)


# Training and validation of the model
if model_type == "CS-SHRED":
    # Instantiation and configuration of the CS-SHRED model
    model = models.CSSHRED(
        num_sensors,
        load_X_shape_1,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=dropout,
        l1_tol=l1_tol,
        opt_tol=opt_tol,
        ls_tol=ls_tol,
        n_sparsity_threshold=num_snapshots_subsample,
        verbosity=0,
        show_plot=False,
    ).to(device)
    train_error, validation_errors = train_and_validate_model(
        model_type,
        model,
        train_dataset,
        valid_dataset,
        num_epochs,
        batch_size,
        lr,
        lambL2,
        lambL1,
        lambdaSNR,
        step_epoch,
        verbose,
        patience,
    )
else:
    # Instantiation and configuration of the SHRED model
    model = models.SHRED(  # 64
        num_sensors,
        load_X_shape_1,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=dropout,
    ).to(device)
    validation_errors = train_and_validate_model(
        model_type,
        model,
        train_dataset,
        valid_dataset,
        num_epochs,
        batch_size,
        lr,
        lambL2,
        lambL1,
        lambdaSNR,
        step_epoch,
        verbose,
        patience,
    )


# Evaluation of the model
test_recons, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim = evaluate_model(
    model, test_dataset, test_dataset_test, sc, json_save_path=save_path + r"error_results.json"
)

end_time = time.time()
total_time = (end_time - begin_time) / 60
print(f"Tempo de execução: {total_time:.2f} minutos")

# Definition of the model parameters
model_params = {
    "hidden_size": hidden_size,
    "hidden_layers": hidden_layers,
    "batch_size": batch_size,
    "lr": lr,
    "lambL2": lambL2,
    "lambL1": lambL1,
    "lambdaSNR": lambdaSNR,
    "l1": l1,
    "l2": l2,
    "lags": lags,
    "num_sensors": num_sensors,
    "num_epochs": num_epochs,
    "step_epoch": step_epoch,
    "dropout": dropout,
    "l1_tol": l1_tol,
    "opt_tol": opt_tol,
    "ls_tol": ls_tol,
}

# Definition of the configuration parameters
config_params = {
    "seed": seed,
    "verbose": verbose,
    "patience": patience,
    "num_cols_subsample": num_cols_subsample,
    "num_snapshots_subsample": num_snapshots_subsample,
    "total_time": total_time,
}

# Path to the JSON file
json_file_path = save_path + r"/error_results.json"


add_model_info_to_json(json_file_path, model_type, model_params, config_params)


if model_type == "CS-SHRED":

    def save_to_numpy(
        test_recons,
        test_ground_truth,
        matrix,
        snapshot,
        sensor_positions_x,
        sensor_positions_y,
        train_error,
        validation_errors,
        model,
        directory=save_path,
    ):
        # Check if the 'results' directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save each dataset in a separate .npy file
        np.save(os.path.join(directory, "test_recons.npy"), test_recons)
        np.save(os.path.join(directory, "test_ground_truth.npy"), test_ground_truth)
        np.save(os.path.join(directory, "matrix.npy"), matrix)
        np.save(os.path.join(directory, "snapshot.npy"), snapshot)
        np.save(os.path.join(directory, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(directory, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(directory, "train_error.npy"), train_error)
        np.save(os.path.join(directory, "validation_errors.npy"), validation_errors)

        print(f"Results saved in {directory}")

    save_to_numpy(
        test_recons,
        test_ground_truth_test,
        matrix,
        snapshot,
        sensor_positions_x,
        sensor_positions_y,
        train_error,
        validation_errors,
        model,
    )

else:

    def save_to_numpy(
        test_recons,
        test_ground_truth,
        matrix,
        snapshot,
        sensor_positions_x,
        sensor_positions_y,
        validation_errors,
        model,
        directory=save_path,
    ):
        # Check if the 'results' directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save each dataset in a separate .npy file
        np.save(os.path.join(directory, "test_recons.npy"), test_recons)
        np.save(os.path.join(directory, "test_ground_truth.npy"), test_ground_truth)
        np.save(os.path.join(directory, "matrix.npy"), matrix)
        np.save(os.path.join(directory, "snapshot.npy"), snapshot)
        np.save(os.path.join(directory, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(directory, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(directory, "validation_errors.npy"), validation_errors)

        print(f"Results saved in {directory}")

    save_to_numpy(
        test_recons,
        test_ground_truth_test,
        matrix,
        snapshot,
        sensor_positions_x,
        sensor_positions_y,
        validation_errors,
        model,
    )
