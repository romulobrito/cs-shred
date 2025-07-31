import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import lpips

# Path configuration
save_path = "./turb-git-csshred"

# Loading saved data
test_recons = np.load(os.path.join(save_path, "test_recons.npy"))
test_ground_truth = np.load(os.path.join(save_path, "test_ground_truth.npy"))
matrix = np.load(os.path.join(save_path, "matrix.npy"))

print("=== DIMENSION VERIFICATION ===")
print(f"test_recons shape: {test_recons.shape}")
print(f"test_ground_truth shape: {test_ground_truth.shape}")
print(f"matrix shape: {matrix.shape}")

# Ensure dimensions are correct
# test_recons: (n_samples, spatial_dim)
# test_ground_truth: (n_samples, spatial_dim) 
# Need to convert to (n_samples, height, width)

# Assuming spatial data is 256x256 = 65536
spatial_dim = 256
n_samples = test_recons.shape[0]

# Reshape to correct format: (n_samples, height, width)
test_recons_reshaped = test_recons.reshape(n_samples, spatial_dim, spatial_dim)
test_ground_truth_reshaped = test_ground_truth.reshape(n_samples, spatial_dim, spatial_dim)

print("\n=== AFTER RESHAPE ===")
print(f"test_recons_reshaped shape: {test_recons_reshaped.shape}")
print(f"test_ground_truth_reshaped shape: {test_ground_truth_reshaped.shape}")

# ================================
# CORRECT SSIM CALCULATION
# ================================
print("\n=== CALCULATING METRICS ===")

ssim_values = []
psnr_values = []
error_norms = []

for i in range(n_samples):
    # Correct SSIM - both arrays 2D (height, width)
    ground_truth_2d = test_ground_truth_reshaped[i]
    reconstructed_2d = test_recons_reshaped[i]
    
    # Calculate SSIM
    data_range = ground_truth_2d.max() - ground_truth_2d.min()
    ssim_value = ssim(
        ground_truth_2d,
        reconstructed_2d,
        data_range=data_range
    )
    ssim_values.append(ssim_value)
    
    # Calculate PSNR
    mse = mean_squared_error(ground_truth_2d, reconstructed_2d)
    if mse == 0:
        psnr_value = float('inf')
    else:
        max_pixel = np.max(ground_truth_2d)
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    psnr_values.append(psnr_value)
    
    # Calculate normalized error
    error_norm = np.linalg.norm(reconstructed_2d - ground_truth_2d) / np.linalg.norm(ground_truth_2d)
    error_norms.append(error_norm)

# Final metrics
mean_ssim = np.mean(ssim_values)
last_ssim = ssim_values[-1]
mean_psnr = np.mean(psnr_values)
last_psnr = psnr_values[-1]
mean_error = np.mean(error_norms)
last_error = error_norms[-1]

print(f"SSIM for the last snapshot: {last_ssim}")
print(f"PSNR for the last snapshot: {last_psnr} dB")
print(f"Normalized Error for the last snapshot: {last_error}")
print(f"Mean SSIM for all snapshots: {mean_ssim}")
print(f"Mean PSNR for all snapshots: {mean_psnr} dB")
print(f"Mean Normalized Error for all snapshots: {mean_error}")

# ================================
# CORRECT LPIPS CALCULATION
# ================================
print("\n=== CALCULATING LPIPS ===")

# Create LPIPS model (suppressing deprecation warnings)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
loss_fn = lpips.LPIPS(net='alex', use_dropout=True)

# For LPIPS, we need tensors with format:
# (batch_size, channels, height, width)
# Since we have monochromatic data, channels = 1

# Convert to tensors and normalize appropriately
def prepare_for_lpips(array_2d):
    """Prepare 2D array for LPIPS"""
    # Normalize to [0, 1]
    array_norm = (array_2d - array_2d.min()) / (array_2d.max() - array_2d.min())
    # Convert to tensor (1, 1, H, W) - batch=1, channels=1
    tensor = torch.from_numpy(array_norm).unsqueeze(0).unsqueeze(0).float()
    # Replicate to 3 channels (RGB) as expected by LPIPS
    tensor_rgb = tensor.repeat(1, 3, 1, 1)
    return tensor_rgb

# Calculate LPIPS for the last snapshot (most important)
last_ground_truth = test_ground_truth_reshaped[-1]
last_reconstruction = test_recons_reshaped[-1]

img0_lpips = prepare_for_lpips(last_ground_truth)
img1_lpips = prepare_for_lpips(last_reconstruction)

print(f"Tensor shapes for LPIPS:")
print(f"Ground truth: {img0_lpips.shape}")
print(f"Reconstruction: {img1_lpips.shape}")

# Calculate LPIPS
with torch.no_grad():
    lpips_value = loss_fn(img0_lpips, img1_lpips).item()

print(f"LPIPS (perceptual loss) for last snapshot: {lpips_value}")

# Calculate mean LPIPS for all snapshots (optional)
lpips_values = []
print("Calculating LPIPS for all snapshots...")

for i in range(min(n_samples, 10)):  # Limit to 10 samples to save time
    gt_tensor = prepare_for_lpips(test_ground_truth_reshaped[i])
    rec_tensor = prepare_for_lpips(test_recons_reshaped[i])
    
    with torch.no_grad():
        lpips_val = loss_fn(gt_tensor, rec_tensor).item()
    lpips_values.append(lpips_val)

mean_lpips = np.mean(lpips_values)
print(f"Mean LPIPS for {len(lpips_values)} snapshots: {mean_lpips}")

# ================================
# SAVE RESULTS
# ================================
results = {
    "Normalized_Error": {
        "last_snapshot": float(last_error),
        "mean_all_snapshots": float(mean_error),
        "all_snapshots": [float(x) for x in error_norms]
    },
    "SSIM": {
        "last_snapshot": float(last_ssim),
        "mean_all_snapshots": float(mean_ssim),
        "all_snapshots": [float(x) for x in ssim_values]
    },
    "PSNR": {
        "last_snapshot": float(last_psnr),
        "mean_all_snapshots": float(mean_psnr),
        "all_snapshots": [float(x) for x in psnr_values]
    },
    "LPIPS": {
        "last_snapshot": float(lpips_value),
        "mean_sample": float(mean_lpips) if lpips_values else None,
        "sample_snapshots": [float(x) for x in lpips_values] if lpips_values else []
    }
}

# Save results
json_file_path = os.path.join(save_path, "results_corrected.json")
with open(json_file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\nCorrected results saved to {json_file_path}")

# ================================
# VISUALIZATION
# ================================
print("\n=== CREATING VISUALIZATIONS ===")

# Comparative plot of the last snapshot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ground truth
im1 = axes[0].imshow(last_ground_truth, cmap='viridis', origin='lower')
axes[0].set_title('Original - Last Snapshot')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0])

# Reconstruction
im2 = axes[1].imshow(last_reconstruction, cmap='viridis', origin='lower')
axes[1].set_title('Reconstructed - Last Snapshot')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1])

# Difference
difference = np.abs(last_ground_truth - last_reconstruction)
im3 = axes[2].imshow(difference, cmap='hot', origin='lower')
axes[2].set_title('Absolute Difference')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(save_path, "comparison_corrected.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot of metrics over time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SSIM
axes[0,0].plot(ssim_values, 'b-', linewidth=2)
axes[0,0].set_title('SSIM over snapshots')
axes[0,0].set_xlabel('Snapshot')
axes[0,0].set_ylabel('SSIM')
axes[0,0].grid(True, alpha=0.3)

# PSNR
axes[0,1].plot(psnr_values, 'g-', linewidth=2)
axes[0,1].set_title('PSNR over snapshots')
axes[0,1].set_xlabel('Snapshot')
axes[0,1].set_ylabel('PSNR (dB)')
axes[0,1].grid(True, alpha=0.3)

# Normalized error
axes[1,0].plot(error_norms, 'r-', linewidth=2)
axes[1,0].set_title('Normalized Error over snapshots')
axes[1,0].set_xlabel('Snapshot')
axes[1,0].set_ylabel('Normalized Error')
axes[1,0].grid(True, alpha=0.3)

# LPIPS (if calculated for multiple snapshots)
if lpips_values:
    axes[1,1].plot(lpips_values, 'm-', linewidth=2)
    axes[1,1].set_title('LPIPS over snapshots')
    axes[1,1].set_xlabel('Snapshot')
    axes[1,1].set_ylabel('LPIPS')
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, f'LPIPS (last snapshot):\n{lpips_value:.4f}', 
                   ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
    axes[1,1].set_title('LPIPS')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "metrics_evolution_corrected.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ANALYSIS COMPLETED ===")
print("All plots and results have been saved with correct dimensions!") 