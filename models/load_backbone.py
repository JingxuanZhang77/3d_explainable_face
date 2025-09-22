"""Helpers for loading trained face recognition models/backbones."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from train_feature_extractor import Face3DModel


def _load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=device)


def _fingerprint(state: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@dataclass
class LoadedModel:
    backbone: torch.nn.Module
    model: torch.nn.Module
    meta: Dict[str, Any]


def load_backbone(checkpoint_path: str, device: Optional[str] = None) -> LoadedModel:
    """Load an inference model (ArcFace head removed) and expose its backbone."""

    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = _load_checkpoint(checkpoint_path, resolved_device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    feature_dim = int(ckpt.get("feature_dim", 512) or 512)
    num_classes = int(ckpt.get("num_classes", 1) or 1)

    model = Face3DModel(num_classes=num_classes, feature_dim=feature_dim, use_arcface=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(resolved_device)
    model.eval()

    meta = {
        "checkpoint": checkpoint_path,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "device": str(resolved_device),
        "fingerprint": _fingerprint(model.state_dict()),
    }

    return LoadedModel(backbone=model.backbone, model=model, meta=meta)


def load_full_model(checkpoint_path: str, device: Optional[str] = None, strict: bool = True) -> LoadedModel:
    """Load the full training model (with ArcFace head when available)."""

    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = _load_checkpoint(checkpoint_path, resolved_device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    feature_dim = int(ckpt.get("feature_dim", 512) or 512)
    num_classes = int(ckpt.get("num_classes", 1) or 1)

    model = Face3DModel(num_classes=num_classes, feature_dim=feature_dim, use_arcface=True)
    model.load_state_dict(state_dict, strict=strict)
    model.to(resolved_device)
    model.eval()

    meta = {
        "checkpoint": checkpoint_path,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "device": str(resolved_device),
        "fingerprint": _fingerprint(model.state_dict()),
    }

    return LoadedModel(backbone=model.backbone, model=model, meta=meta)
