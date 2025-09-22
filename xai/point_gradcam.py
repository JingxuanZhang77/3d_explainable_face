"""Point-GradCAM implementation tailored for point-cloud backbones."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def _resolve_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]  # type: ignore[index]
        else:
            layer = getattr(layer, part)
    return layer


class PointGradCAM:
    def __init__(self, model: torch.nn.Module, layer: str = "backbone.conv5.2") -> None:
        self.model = model
        self.layer_name = layer
        self.layer = _resolve_layer(model, layer)
        self.device = next(model.parameters()).device
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()
            if not output.requires_grad:
                return

            def grad_hook(grad):
                self.gradients = grad.detach()

            output.register_hook(grad_hook)

        self.layer.register_forward_hook(forward_hook)

    def attribute(self, points_bn3: np.ndarray | torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        if isinstance(points_bn3, np.ndarray):
            pts = torch.from_numpy(points_bn3.astype(np.float32))
        else:
            pts = points_bn3.to(dtype=torch.float32)

        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
        if pts.ndim != 3 or pts.size(-1) != 3:
            raise ValueError(f"Expected points shaped (B,N,3); got {tuple(pts.shape)}")

        pts = pts.to(self.device)
        pts.requires_grad_(True)

        embeddings = self.model(pts)
        score = score_fn(embeddings)
        if score.ndim != 0:
            score = score.sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        acts = self.activations  # (B, C, N)
        grads = self.gradients  # (B, C, N)
        weights = grads.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))  # (B, N)

        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        norm = torch.where((cam_max - cam_min) > 0, cam_max - cam_min, torch.ones_like(cam_max))
        saliency = ((cam - cam_min) / norm).clamp(min=0.0, max=1.0)

        return saliency.detach().cpu().numpy()
