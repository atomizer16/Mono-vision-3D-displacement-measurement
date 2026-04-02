"""Relative-depth postprocessing utilities.

This module intentionally works on ``depth_rel`` only. Absolute metric conversion
(e.g., ``Z_abs`` recovery) is handled by fusion/calibration modules.
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


def smooth_depth_rel(depth_rel: np.ndarray, config: Mapping[str, Any] | None = None) -> np.ndarray:
    """Apply optional median and/or bilateral smoothing on ``depth_rel``."""
    cfg = dict(config or {})
    out = np.asarray(depth_rel, dtype=np.float32)

    median_ksize = int(cfg.get("median_ksize", 0))
    if median_ksize > 1:
        if median_ksize % 2 == 0:
            median_ksize += 1
        out = cv2.medianBlur(out, median_ksize)

    bilateral = cfg.get("bilateral", {})
    if bool(bilateral.get("enabled", False)):
        d = int(bilateral.get("d", 5))
        sigma_color = float(bilateral.get("sigma_color", 0.1))
        sigma_space = float(bilateral.get("sigma_space", 5.0))
        out = cv2.bilateralFilter(out, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return np.asarray(out, dtype=np.float32)


def build_confidence_mask(
    frame_bgr: np.ndarray,
    depth_rel: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Build confidence mask; optionally suppress edge and low-texture regions."""
    cfg = dict(config or {})
    h, w = depth_rel.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    edge_cfg = cfg.get("edge_suppression", {})
    if bool(edge_cfg.get("enabled", False)):
        low = int(edge_cfg.get("canny_low", 80))
        high = int(edge_cfg.get("canny_high", 160))
        dilate_k = int(edge_cfg.get("dilate_ksize", 3))
        edges = cv2.Canny(gray, threshold1=low, threshold2=high)
        if dilate_k > 1:
            kernel = np.ones((dilate_k, dilate_k), dtype=np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        mask[edges > 0] = 0

    texture_cfg = cfg.get("low_texture_suppression", {})
    if bool(texture_cfg.get("enabled", False)):
        win = int(texture_cfg.get("window", 7))
        if win % 2 == 0:
            win += 1
        var_thr = float(texture_cfg.get("variance_threshold", 12.0))
        gray_f = gray.astype(np.float32)

        mean = cv2.GaussianBlur(gray_f, (win, win), 0)
        mean2 = cv2.GaussianBlur(gray_f * gray_f, (win, win), 0)
        local_var = np.maximum(mean2 - mean * mean, 0.0)
        mask[local_var < var_thr] = 0

    finite = np.isfinite(depth_rel)
    mask[~finite] = 0

    return mask


def postprocess_depth(
    frame_bgr: np.ndarray,
    depth_rel: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Postprocess relative depth and return ``depth_rel`` + metadata."""
    cfg = dict(config or {})
    smooth_cfg = cfg.get("smoothing", {})
    conf_cfg = cfg.get("confidence", {})

    depth_smoothed = smooth_depth_rel(depth_rel=depth_rel, config=smooth_cfg)
    conf_mask = build_confidence_mask(frame_bgr=frame_bgr, depth_rel=depth_smoothed, config=conf_cfg)

    return depth_smoothed, {"confidence_mask": conf_mask}
