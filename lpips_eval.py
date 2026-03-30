#!/usr/bin/env python3
"""
Shared LPIPS evaluation for flattened test reconstructions (TURB and similar).

Matches analyze_all_snapshots.py: Alex backbone, per-image min-max to [0, 1],
single-channel tensors shaped (N, 1, H, W). Lower LPIPS is better.

All comments and docstrings are ASCII-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _minmax01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mn = float(arr.min())
    mx = float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)


def lpips_distance_single(
    gt: np.ndarray,
    rec: np.ndarray,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Pairwise LPIPS for 2D arrays gt, rec with the same shape."""
    g = _minmax01(gt)
    r = _minmax01(rec)
    t0 = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(r).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        d = loss_fn(t0, t1)
    return float(d.item())


def lpips_test_metrics(
    test_recons: np.ndarray,
    test_ground_truth: np.ndarray,
    height: int,
    width: int,
) -> tuple[float, float]:
    """
    Mean and last-sample LPIPS over test rows.

    test_recons / test_ground_truth: shape (n_samples, height * width).
    """
    if test_recons.ndim != 2 or test_ground_truth.ndim != 2:
        raise ValueError("Expected 2D arrays (n_samples, n_pixels).")
    n_pix = height * width
    if test_recons.shape[1] != n_pix or test_ground_truth.shape[1] != n_pix:
        raise ValueError(
            f"Expected {n_pix} columns (H*W={height}*{width}), "
            f"got {test_recons.shape[1]} and {test_ground_truth.shape[1]}."
        )
    n = int(test_recons.shape[0])
    import lpips  # local import so module load works without lpips installed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    loss_fn.eval()

    vals: list[float] = []
    rec_cube = test_recons.reshape(n, height, width)
    gt_cube = test_ground_truth.reshape(n, height, width)
    for i in range(n):
        vals.append(
            lpips_distance_single(gt_cube[i], rec_cube[i], loss_fn, device)
        )
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr[-1])


def update_results_json_lpips(artifact_dir: Path) -> Mapping[str, Any]:
    """
    Load test_recons.npy, test_ground_truth.npy, matrix.npy from artifact_dir,
    compute LPIPS mean and last snapshot, merge into results.json, return the dict.
    """
    artifact_dir = artifact_dir.resolve()
    matrix = np.load(artifact_dir / "matrix.npy", mmap_mode="r")
    _, dim_h, dim_w = matrix.shape
    test_recons = np.load(artifact_dir / "test_recons.npy")
    test_gt = np.load(artifact_dir / "test_ground_truth.npy")
    mean_l, last_l = lpips_test_metrics(
        np.asarray(test_recons),
        np.asarray(test_gt),
        int(dim_h),
        int(dim_w),
    )
    results_path = artifact_dir / "results.json"
    payload: dict[str, Any] = {}
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload["LPIPS_global_mean"] = mean_l
    payload["LPIPS_last_snapshot"] = last_l
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    return payload
