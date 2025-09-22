"""Utilities for selecting salient points and re-embedding them."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from data.preprocess import ensure_bn3, zero_mean_unit_sphere, zero_mean_unit_sphere_tensor


def select_topk_points(points_bn3: np.ndarray, saliency: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Return top-k points per sample according to saliency."""

    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0,1]")

    pts = ensure_bn3(points_bn3)
    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim == 1:
        sal = sal[None, :]
    if sal.shape[0] != pts.shape[0] or sal.shape[1] != pts.shape[1]:
        raise ValueError("Saliency shape must match points batch")

    k = max(1, int(round(pts.shape[1] * keep_ratio)))
    indices = np.argsort(-sal, axis=1)[:, :k]

    batch_indices = np.arange(pts.shape[0])[:, None]
    selected = pts[batch_indices, indices]
    return selected


def renorm_unit_sphere(points_bn3: np.ndarray) -> np.ndarray:
    return zero_mean_unit_sphere(points_bn3)


def embed_points(backbone: torch.nn.Module, points_bn3: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Project selected points through the frozen backbone."""

    if isinstance(points_bn3, np.ndarray):
        pts = torch.from_numpy(points_bn3.astype(np.float32))
    else:
        pts = points_bn3.to(dtype=torch.float32)

    pts = zero_mean_unit_sphere_tensor(pts).to(next(backbone.parameters()).device)
    with torch.no_grad():
        embeddings = backbone(pts)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)
