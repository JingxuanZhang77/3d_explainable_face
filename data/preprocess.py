"""Point-cloud preprocessing utilities reused across training/inference/XAI."""

from __future__ import annotations

import numpy as np
import torch


def ensure_bn3(points: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return an array shaped (B, N, 3) without modifying caller storage."""

    if isinstance(points, torch.Tensor):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        if arr.ndim == 3 and arr.shape[-2] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f"Expected shape (B,N,3); got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def zero_mean_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Apply zero-mean and unit-sphere normalisation to batched points."""

    arr = ensure_bn3(points).copy()
    centroid = arr.mean(axis=1, keepdims=True)
    arr -= centroid
    radius = np.linalg.norm(arr, axis=2, keepdims=True).max(axis=1, keepdims=True)
    arr /= (radius + 1e-6)
    return arr


def zero_mean_unit_sphere_tensor(points: torch.Tensor) -> torch.Tensor:
    """Torch version keeping gradients when possible."""

    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.size(-1) != 3:
        if points.ndim == 3 and points.size(-2) == 3 and points.size(-1) != 3:
            points = points.transpose(1, 2)
        else:
            raise ValueError(f"Expected shape (B,N,3); got {tuple(points.shape)}")

    centroid = points.mean(dim=1, keepdim=True)
    centred = points - centroid
    radius, _ = torch.linalg.norm(centred, ord=2, dim=2, keepdim=True).max(dim=1, keepdim=True)
    return centred / (radius + 1e-6)
