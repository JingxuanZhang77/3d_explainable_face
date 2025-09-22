"""Export utilities for colouring point clouds with saliency scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np


def _ensure_n3(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError("Only single clouds supported; got batch size >1")
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3); got {arr.shape}")
    return arr


def save_colored_pointcloud(points: np.ndarray, saliency: np.ndarray, path: str | Path, cmap: str = "viridis") -> None:
    pts = _ensure_n3(points)
    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim == 2:
        if sal.shape[0] != 1:
            raise ValueError("Batch export not supported")
        sal = sal[0]
    if sal.shape[0] != pts.shape[0]:
        raise ValueError("Mismatch between points and saliency length")

    sal_norm = sal - sal.min()
    if sal_norm.max() > 0:
        sal_norm /= sal_norm.max()
    colors = (cm.get_cmap(cmap)(sal_norm)[:, :3] * 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    with path.open("w", encoding="ascii") as fh:
        fh.write("\n".join(header) + "\n")
        for (x, y, z), (r, g, b) in zip(pts, colors):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
