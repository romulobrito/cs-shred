import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
from math import log10
import json
import os
import time

from sklearn.metrics import mean_squared_error
from processdata import TimeSeriesDataset


def plot_csshred_results(
    train_error, validation_errors, test_recons, test_ground_truth_test, matrix, subsampled,
    save_path=r'./results/csshred/oldroyd/'
):
    """
    Plot results for CS-SHRED model with corrected data handling.
    
    Parameters
    ----------
    train_error : np.ndarray
        Training error history.
    validation_errors : np.ndarray
        Validation error history.
    test_recons : np.ndarray
        Model reconstructions.
    test_ground_truth_test : np.ndarray
        Ground truth data (original, not subsampled).
    matrix : np.ndarray
        Original data matrix.
    subsampled : np.ndarray
        Subsampled data.
    save_path : str
        Path to save plots and results.
    """
    os.makedirs(save_path, exist_ok=True)
    
    dim_x, dim_y, dim_t = subsampled.shape

    # Reshape ground truth to match expected format
    test_ground_truth_test = test_ground_truth_test.copy().reshape(-1, dim_x, dim_y)
    test_ground_truth_test = np.transpose(test_ground_truth_test.copy(), (1, 2, 0))
   
    # Reshape reconstructions
    test_recons = test_recons.reshape(-1, dim_x, dim_y)
   
    # Reshape subsampled data
    subsampled = subsampled.reshape(dim_x, dim_y, -1)

    print("test_recons", test_recons.shape)
    print("test_ground_truth_test", test_ground_truth_test.shape)
    print("subsampled", subsampled.shape)

    # Plot Training Error
    plt.figure(figsize=(12, 6))
    plt.plot(train_error, label="Training Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Error Over Epochs")
    plt.legend()
    training_error_plot_path = os.path.join(save_path, "training_error.pdf")
    plt.savefig(training_error_plot_path)
    plt.show()

    # Plot Validation Error
    plt.figure(figsize=(12, 6))
    plt.plot(validation_errors, label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Error Over Epochs")
    plt.legend()
    validation_error_plot_path = os.path.join(save_path, "validation_error.pdf")
    plt.savefig(validation_error_plot_path)
    plt.show()

    # Plot Last Snapshot - Reconstructed and Original
    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(test_recons[-1, :, :], cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    fig1 = plt.gcf()
    cbar1 = fig1.colorbar(img1, ax=ax1, label=r"$tr(C)$")
    plt.title("Reconstructed - Last Snapshot")
    
    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(test_ground_truth_test[:, :, -1], cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    fig2 = plt.gcf()
    cbar2 = fig2.colorbar(img2, ax=ax2, label=r"$tr(C)$")
    plt.title("Original - Last Snapshot")
    
    last_snapshot_plot_path = os.path.join(save_path, "last_snapshot_comparison.pdf")
    plt.savefig(last_snapshot_plot_path)
    plt.show()

    # Plot Last Snapshot - Reconstructed and Subsampled
    plt.figure(figsize=(14, 6))
    
    ax3 = plt.subplot(1, 2, 1)
    img3 = ax3.imshow(test_recons[-1, :, :], cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    fig3 = plt.gcf()
    cbar3 = fig3.colorbar(img3, ax=ax3, label=r"$tr(C)$")
    plt.title("Reconstructed - Last Snapshot")
    
    ax4 = plt.subplot(1, 2, 2)
    img4 = ax4.imshow(subsampled[:, :, -1], cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    fig4 = plt.gcf()
    cbar4 = fig4.colorbar(img4, ax=ax4, label=r"$tr(C)$")
    plt.title("Subsampled - Last Snapshot")
    
    subsampled_snapshot_plot_path = os.path.join(save_path, "subsampled_snapshot_comparison.pdf")
    plt.savefig(subsampled_snapshot_plot_path)
    plt.show()
    
    # Calculate SSIM, MSE and PSNR for the last snapshot
    ssim_value_last = ssim(
        test_ground_truth_test[:, :, -1],
        test_recons[-1],
        data_range=test_ground_truth_test[:, :, -1].max() - test_ground_truth_test[:, :, -1].min(),
    )
    
    test_recons_mse = np.transpose(test_recons.copy(), (1,2,0))
    mse_value_last = np.linalg.norm(test_recons_mse - test_ground_truth_test) / np.linalg.norm(test_ground_truth_test)
    
    mse = mean_squared_error(test_ground_truth_test[:, :, -1], test_recons[-1])
    max_pixel = np.max(test_ground_truth_test[:, :, -1])
    psnr_value_last = 20 * log10(max_pixel / np.sqrt(mse))
    
    # Calculate normalized error
    error_norm_last = np.linalg.norm(test_recons[-1] - test_ground_truth_test[:, :, -1]) / np.linalg.norm(test_ground_truth_test[:, :, -1])
    
    print(f"SSIM for the last snapshot: {ssim_value_last}")
    print(f"MSE for the last snapshot: {mse_value_last}")
    print(f"PSNR for the last snapshot: {psnr_value_last} dB")
    print(f"Normalized Error for the last snapshot: {error_norm_last}")

    # Calculate averages for all snapshots
    ssim_values = []
    psnr_values = []
    error_norms = []
    for i in range(test_recons.shape[0]):
        ssim_value = ssim(
            test_ground_truth_test[:, :, i],
            test_recons[i],
            data_range=test_ground_truth_test[:, :, i].max() - test_ground_truth_test[:, :, i].min(),
        )
        ssim_values.append(ssim_value)
        
        mse = mean_squared_error(test_ground_truth_test[:, :, i], test_recons[i])
        max_pixel = np.max(test_ground_truth_test[:, :, i])
        psnr = 20 * log10(max_pixel / np.sqrt(mse))
        psnr_values.append(psnr)

        # Calculate normalized error for each snapshot
        error_norm = np.linalg.norm(test_recons[i] - test_ground_truth_test[:, :, i]) / np.linalg.norm(test_ground_truth_test[:, :, i])
        error_norms.append(error_norm)

    mean_ssim_value = np.mean(ssim_values)
    mean_psnr_value = np.mean(psnr_values)
    mean_error_norm = np.mean(error_norms)
    print(f"Mean SSIM for all snapshots: {mean_ssim_value}")
    print(f"Mean PSNR for all snapshots: {mean_psnr_value} dB")
    print(f"Mean Normalized Error for all snapshots: {mean_error_norm}")

    # Save results in JSON
    results = {
        "SSIM_last_snapshot": float(ssim_value_last),
        "MSE_last_snapshot": float(mse_value_last),
        "PSNR_last_snapshot": float(psnr_value_last),
        "Normalized_Error_last_snapshot": float(error_norm_last),  
        "Mean_SSIM_all_snapshots": float(mean_ssim_value),
        "Mean_PSNR_all_snapshots": float(mean_psnr_value),
        "Mean_Normalized_Error_all_snapshots": float(mean_error_norm), 
    }
    
    json_file_path = os.path.join(save_path, "results.json")
    with open(json_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Results saved to {json_file_path}")
    print(f"Plots saved to {save_path}")


def plot_shred_results(
    validation_errors, test_recons, test_ground_truth_test, matrix, subsampled,
    save_path=r'./results/shred/oldroyd/'
):
    """
    Plot results for SHRED model with corrected data handling.
    
    Parameters
    ----------
    validation_errors : np.ndarray
        Validation error history.
    test_recons : np.ndarray
        Model reconstructions.
    test_ground_truth_test : np.ndarray
        Ground truth data (original, not subsampled).
    matrix : np.ndarray
        Original data matrix.
    subsampled : np.ndarray
        Subsampled data.
    save_path : str
        Path to save plots and results.
    """
    os.makedirs(save_path, exist_ok=True)
    
    dim_x, dim_y, dim_t = subsampled.shape

    # Reshape ground truth to match expected format
    test_ground_truth_test = test_ground_truth_test.copy().reshape(-1, dim_x, dim_y)
    test_ground_truth_test = np.transpose(test_ground_truth_test.copy(), (1, 2, 0))
   
    # Reshape reconstructions
    test_recons = test_recons.reshape(-1, dim_x, dim_y)
   
    # Reshape subsampled data
    subsampled = subsampled.reshape(dim_x, dim_y, -1)

    print("test_recons", test_recons.shape)
    print("test_ground_truth_test", test_ground_truth_test.shape)
    print("subsampled", subsampled.shape)

    # Plot Validation Error
    plt.figure(figsize=(12, 6))
    plt.plot(validation_errors, label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Error Over Epochs")
    plt.legend()
    validation_error_plot_path = os.path.join(save_path, "validation_error.png")
    plt.savefig(validation_error_plot_path)
    plt.show()

    # Plot Last Snapshot - Reconstructed and Original
    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(test_recons[-1, :, :], cmap="viridis", origin="lower")
    fig1 = plt.gcf()
    cbar1 = fig1.colorbar(img1, ax=ax1, label=r"$tr(C)$")
    plt.title("Reconstructed - Last Snapshot")
    
    ax2 = plt.subplot(1, 2, 2)
    img2 = ax2.imshow(test_ground_truth_test[:, :, -1], cmap="viridis", origin="lower")
    fig2 = plt.gcf()
    cbar2 = fig2.colorbar(img2, ax=ax2, label=r"$tr(C)$")
    plt.title("Original - Last Snapshot")
    
    last_snapshot_plot_path = os.path.join(save_path, "last_snapshot_comparison.png")
    plt.savefig(last_snapshot_plot_path)
    plt.show()

    # Plot Last Snapshot - Reconstructed and Subsampled
    plt.figure(figsize=(14, 6))
    
    ax3 = plt.subplot(1, 2, 1)
    img3 = ax3.imshow(test_recons[-1, :, :], cmap="viridis", origin="lower")
    fig3 = plt.gcf()
    cbar3 = fig3.colorbar(img3, ax=ax3, label=r"$tr(C)$")
    plt.title("Reconstructed - Last Snapshot")
    
    ax4 = plt.subplot(1, 2, 2)
    img4 = ax4.imshow(subsampled[:, :, -1], cmap="viridis", origin="lower")
    fig4 = plt.gcf()
    cbar4 = fig4.colorbar(img4, ax=ax4, label=r"$tr(C)$")
    plt.title("Subsampled - Last Snapshot")
    
    subsampled_snapshot_plot_path = os.path.join(save_path, "subsampled_snapshot_comparison.png")
    plt.savefig(subsampled_snapshot_plot_path)
    plt.show()
    
    # Calculate SSIM, MSE and PSNR for the last snapshot
    ssim_value_last = ssim(
        test_ground_truth_test[:, :, -1],
        test_recons[-1],
        data_range=test_ground_truth_test[:, :, -1].max() - test_ground_truth_test[:, :, -1].min(),
    )
    
    test_recons_mse = np.transpose(test_recons.copy(), (1,2,0))
    mse_value_last = np.linalg.norm(test_recons_mse - test_ground_truth_test) / np.linalg.norm(test_ground_truth_test)
    
    mse = mean_squared_error(test_ground_truth_test[:, :, -1], test_recons[-1])
    max_pixel = np.max(test_ground_truth_test[:, :, -1])
    psnr_value_last = 20 * log10(max_pixel / np.sqrt(mse))
    
    # Calculate normalized error
    error_norm_last = np.linalg.norm(test_recons[-1] - test_ground_truth_test[:, :, -1]) / np.linalg.norm(test_ground_truth_test[:, :, -1])
    
    print(f"SSIM for the last snapshot: {ssim_value_last}")
    print(f"MSE for the last snapshot: {mse_value_last}")
    print(f"PSNR for the last snapshot: {psnr_value_last} dB")
    print(f"Normalized Error for the last snapshot: {error_norm_last}")

    # Calculate averages for all snapshots
    ssim_values = []
    psnr_values = []
    error_norms = []
    for i in range(test_recons.shape[0]):
        ssim_value = ssim(
            test_ground_truth_test[:, :, i],
            test_recons[i],
            data_range=test_ground_truth_test[:, :, i].max() - test_ground_truth_test[:, :, i].min(),
        )
        ssim_values.append(ssim_value)
        
        mse = mean_squared_error(test_ground_truth_test[:, :, i], test_recons[i])
        max_pixel = np.max(test_ground_truth_test[:, :, i])
        psnr = 20 * log10(max_pixel / np.sqrt(mse))
        psnr_values.append(psnr)

        # Calculate normalized error for each snapshot
        error_norm = np.linalg.norm(test_recons[i] - test_ground_truth_test[:, :, i]) / np.linalg.norm(test_ground_truth_test[:, :, i])
        error_norms.append(error_norm)

    mean_ssim_value = np.mean(ssim_values)
    mean_psnr_value = np.mean(psnr_values)
    mean_error_norm = np.mean(error_norms)
    print(f"Mean SSIM for all snapshots: {mean_ssim_value}")
    print(f"Mean PSNR for all snapshots: {mean_psnr_value} dB")
    print(f"Mean Normalized Error for all snapshots: {mean_error_norm}")

    # Save results in JSON
    results = {
        "SSIM_last_snapshot": float(ssim_value_last),
        "MSE_last_snapshot": float(mse_value_last),
        "PSNR_last_snapshot": float(psnr_value_last),
        "Normalized_Error_last_snapshot": float(error_norm_last),  
        "Mean_SSIM_all_snapshots": float(mean_ssim_value),
        "Mean_PSNR_all_snapshots": float(mean_psnr_value),
        "Mean_Normalized_Error_all_snapshots": float(mean_error_norm), 
    }
    
    json_file_path = os.path.join(save_path, "results.json")
    with open(json_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Results saved to {json_file_path}")
    print(f"Plots saved to {save_path}")


def main():
    """
    Main function to run the plotting scripts for both CS-SHRED and SHRED models.
    """
    print("=== OLDROYD PLOTTING SCRIPT (CORRECTED) ===")
    
    # CS-SHRED Results
    print("\n--- CS-SHRED Results ---")
    csshred_save_path = r"./results/csshred/oldroyd_paper"
    csshred_load_path = r"./results/csshred/oldroyd_paper"
    
    try:
        # Load CS-SHRED files
        test_recons = np.load(csshred_load_path + r"/test_recons.npy")
        test_ground_truth_test = np.load(csshred_load_path + r"/test_ground_truth.npy")
        matrix = np.load(csshred_load_path + r"/matrix.npy")
        snapshot = np.load(csshred_load_path + r"/snapshot.npy")
        sensor_positions_x = np.load(csshred_load_path + r"/sensor_positions_x.npy")
        sensor_positions_y = np.load(csshred_load_path + r"/sensor_positions_y.npy")
        train_error = np.load(csshred_load_path + r"/train_error.npy")
        validation_errors = np.load(csshred_load_path + r"/validation_errors.npy")
        
        print("✅ CS-SHRED files loaded successfully")
        
        # Plot CS-SHRED results
        plot_csshred_results(
            train_error, validation_errors, test_recons, test_ground_truth_test, matrix, snapshot,
            save_path=csshred_save_path
        )
        
    except FileNotFoundError as e:
        print(f"❌ CS-SHRED files not found: {e}")
        print("Make sure the CS-SHRED results are available in the specified path.")
    
    # SHRED Results
    print("\n--- SHRED Results ---")
    shred_save_path = r"./results/shred/oldroyd"
    shred_load_path = r"./results/shred/oldroyd"
    
    try:
        # Load SHRED files
        test_recons = np.load(shred_load_path + r"/test_recons.npy")
        test_ground_truth_test = np.load(shred_load_path + r"/test_ground_truth.npy")
        matrix = np.load(shred_load_path + r"/matrix.npy")
        snapshot = np.load(shred_load_path + r"/snapshot.npy")
        sensor_positions_x = np.load(shred_load_path + r"/sensor_positions_x.npy")
        sensor_positions_y = np.load(shred_load_path + r"/sensor_positions_y.npy")
        validation_errors = np.load(shred_load_path + r"/validation_errors.npy")
        
        print("✅ SHRED files loaded successfully")
        
        # Plot SHRED results
        plot_shred_results(
            validation_errors, test_recons, test_ground_truth_test, matrix, snapshot,
            save_path=shred_save_path
        )
        
    except FileNotFoundError as e:
        print(f"❌ SHRED files not found: {e}")
        print("Make sure the SHRED results are available in the specified path.")
    
    print("\n=== PLOTTING COMPLETED ===")


if __name__ == "__main__":
    main() 