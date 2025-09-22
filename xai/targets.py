"""Helpers for defining attribution targets."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    return arr


def pick_top1_id_for_query(full_embed: np.ndarray, ids: Sequence[str], templates: np.ndarray) -> tuple[str, float]:
    """Return the top-1 identity and cosine similarity for an embedding."""

    emb = _ensure_2d(np.asarray(full_embed, dtype=np.float32))
    tmpl = _ensure_2d(np.asarray(templates, dtype=np.float32))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    tmpl = tmpl / (np.linalg.norm(tmpl, axis=1, keepdims=True) + 1e-6)

    sims = emb @ tmpl.T
    best_idx = int(np.argmax(sims[0]))
    return ids[best_idx], float(sims[0, best_idx])


def make_score_fn_cos_to_template(template_vec: np.ndarray | torch.Tensor, device: torch.device | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a score function mapping embeddings to scalar cosine similarity."""

    if isinstance(template_vec, np.ndarray):
        template = torch.from_numpy(template_vec.astype(np.float32))
    else:
        template = template_vec.detach().clone()

    if template.ndim == 1:
        template = template.unsqueeze(0)
    if template.ndim != 2:
        raise ValueError(f"Expected template shaped (1,D) or (B,D), got {tuple(template.shape)}")

    template = template / (template.norm(dim=1, keepdim=True) + 1e-6)
    if device is None:
        device = template.device
    else:
        template = template.to(device)

    def score_fn(emb: torch.Tensor) -> torch.Tensor:
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-6)
        sims = torch.sum(emb_norm * template, dim=1)
        return sims.sum()

    return score_fn
