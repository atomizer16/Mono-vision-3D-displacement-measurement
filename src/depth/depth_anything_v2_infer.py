"""Depth Anything V2 inference wrapper.

This module only predicts *relative* depth (``depth_rel``). It does **not** perform
absolute metric scaling; that step is intentionally delegated to the fusion module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(slots=True)
class DepthInferMeta:
    """Runtime metadata returned together with ``depth_rel``."""

    infer_latency_ms: float
    input_size_hw: tuple[int, int]
    orig_size_hw: tuple[int, int]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "infer_latency_ms": self.infer_latency_ms,
            "input_size_hw": self.input_size_hw,
            "orig_size_hw": self.orig_size_hw,
            "model_name": self.model_name,
        }


class DepthAnythingV2Infer:
    """End-to-end wrapper for model loading, preprocessing, inference, postprocessing.

    Input:
        frame_bgr: BGR image in shape ``(H, W, 3)``.

    Output:
        tuple[depth_rel, meta]
        - depth_rel: float32 array in shape ``(H, W)``
        - meta: dict with latency/input size and a visualization image under
          ``meta['depth_vis_bgr']`` for debugging.
    """

    def __init__(self, config: Mapping[str, Any], model: Any | None = None) -> None:
        self.config = config
        model_cfg = config.get("model", {})
        infer_cfg = config.get("inference", {})

        self.model_name = str(model_cfg.get("name", "depth-anything-v2"))
        self.weights = str(model_cfg.get("weights", ""))

        input_size = infer_cfg.get("input_size", [518, 518])
        self.input_size_hw = (int(input_size[0]), int(input_size[1]))

        norm_cfg = infer_cfg.get("normalize", {})
        self.mean = np.asarray(norm_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(norm_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        self.device = str(infer_cfg.get("device", "cpu"))
        self._model = model

    def load_model(self) -> None:
        """Load model weights when an external model instance is not injected."""
        if self._model is not None:
            return

        import torch

        loaded = torch.load(self.weights, map_location=self.device)
        if hasattr(loaded, "eval"):
            self._model = loaded.eval()
        elif isinstance(loaded, Mapping) and "model" in loaded and hasattr(loaded["model"], "eval"):
            self._model = loaded["model"].eval()
        else:
            raise TypeError("Unsupported weights content. Expected a torch.nn.Module or {'model': module}.")

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """BGR uint8 -> normalized NCHW float32 tensor-like ndarray."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (H, W, 3).")

        orig_hw = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))
        resized = cv2.resize(frame_bgr, (self.input_size_hw[1], self.input_size_hw[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        return x, orig_hw

    def infer(self, x_nchw: np.ndarray) -> np.ndarray:
        """Run forward pass and return raw relative depth in input resolution."""
        self.load_model()

        import torch

        x = torch.from_numpy(x_nchw).to(self.device)
        with torch.inference_mode():
            pred = self._model(x)

        if isinstance(pred, (tuple, list)):
            pred = pred[0]

        pred = pred.squeeze().detach().float().cpu().numpy()
        return np.asarray(pred, dtype=np.float32)

    def postprocess(self, raw_depth: np.ndarray, orig_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Resize depth to original resolution and build a debug visualization map."""
        depth_rel = cv2.resize(raw_depth, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_CUBIC)
        depth_rel = np.asarray(depth_rel, dtype=np.float32)

        finite = np.isfinite(depth_rel)
        if np.any(finite):
            dmin = float(np.min(depth_rel[finite]))
            dmax = float(np.max(depth_rel[finite]))
            denom = max(dmax - dmin, 1e-6)
            depth_01 = np.clip((depth_rel - dmin) / denom, 0.0, 1.0)
        else:
            depth_01 = np.zeros_like(depth_rel, dtype=np.float32)

        depth_u8 = (depth_01 * 255.0).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        return depth_rel, depth_vis

    def run(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Run full pipeline and return ``depth_rel`` and ``meta``."""
        t0 = time.perf_counter()
        x_nchw, orig_hw = self.preprocess(frame_bgr)
        raw_depth = self.infer(x_nchw)
        depth_rel, depth_vis = self.postprocess(raw_depth, orig_hw)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        meta_obj = DepthInferMeta(
            infer_latency_ms=float(latency_ms),
            input_size_hw=self.input_size_hw,
            orig_size_hw=orig_hw,
            model_name=self.model_name,
        )
        meta = meta_obj.to_dict()
        meta["depth_vis_bgr"] = depth_vis

        return depth_rel, meta
