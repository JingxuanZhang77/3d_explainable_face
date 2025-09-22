"""Template building utilities for full-face and compact embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from data.preprocess import zero_mean_unit_sphere
from models.load_backbone import LoadedModel
from xai.point_gradcam import PointGradCAM
from xai.selection import embed_points, renorm_unit_sphere, select_topk_points
from xai.targets import make_score_fn_cos_to_template


def _get_id(path: str | Path) -> str:
    name = Path(path).stem
    return name.split("_")[0]


def build_id_templates(gallery_features: np.ndarray, gallery_paths: Sequence[str]) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Return per-identity averaged templates from gallery embeddings."""

    feats = np.asarray(gallery_features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("gallery_features must be 2D")
    if len(gallery_paths) != feats.shape[0]:
        raise ValueError("gallery_paths length mismatch")

    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)

    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    for vec, path in zip(feats, gallery_paths):
        buckets[_get_id(path)].append(vec)

    ids = sorted(buckets.keys())
    templates = []
    id2tmpl: Dict[str, np.ndarray] = {}
    for identity in ids:
        stacked = np.stack(buckets[identity])
        mean_vec = stacked.mean(axis=0)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-6)
        templates.append(mean_vec)
        id2tmpl[identity] = mean_vec

    return ids, np.stack(templates), id2tmpl


def build_compact_templates(
    loaded: LoadedModel,
    gallery_files: Sequence[str],
    id2template: Dict[str, np.ndarray],
    keep_ratio: float,
    layer: str = "conv5.2",
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], Dict[str, str], np.ndarray]:
    """Create compact templates by XAI-guided point selection."""

    backbone = loaded.backbone
    device = next(backbone.parameters()).device
    grad_cam = PointGradCAM(backbone, layer=layer)

    per_id: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_embeddings: List[np.ndarray] = []
    meta = {
        "keep_ratio": f"{keep_ratio:.4f}",
        "layer": layer,
        "fingerprint": loaded.meta.get("fingerprint", ""),
        "checkpoint": loaded.meta.get("checkpoint", ""),
    }

    for path in gallery_files:
        identity = _get_id(path)
        template = id2template.get(identity)
        if template is None:
            continue

        data = np.load(path)
        points = zero_mean_unit_sphere(data["points"].astype(np.float32))
        pts_tensor = torch.from_numpy(points).to(device=device, dtype=torch.float32)

        score_fn = make_score_fn_cos_to_template(template, device=device)
        saliency = grad_cam.attribute(pts_tensor, score_fn)[0]

        selected = select_topk_points(points, saliency, keep_ratio)
        selected = renorm_unit_sphere(selected)
        embedding = embed_points(backbone, selected)[0].cpu().numpy()
        per_id[identity].append(embedding)
        sample_embeddings.append(embedding)

    ids = sorted(per_id.keys())
    templates = []
    id2compact = {}
    for identity in ids:
        stack = np.stack(per_id[identity])
        mean_vec = stack.mean(axis=0)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-6)
        templates.append(mean_vec)
        id2compact[identity] = mean_vec

    return ids, np.stack(templates), id2compact, meta, np.stack(sample_embeddings)


def save_templates(path: str | Path, ids: Sequence[str], templates: np.ndarray, meta: Dict[str, str]) -> None:
    payload = {
        "ids": np.array(ids),
        "templates": templates.astype(np.float32),
        "meta_json": json.dumps(meta, sort_keys=True),
    }
    np.savez(path, **payload)


def load_templates(path: str | Path) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    templates = data["templates"].astype(np.float32)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    return ids, templates, meta
