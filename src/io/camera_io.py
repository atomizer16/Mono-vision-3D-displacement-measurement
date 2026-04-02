"""Camera parameter I/O helpers for project-wide reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Load camera intrinsics YAML produced by calibration module."""

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = {
        "camera_matrix",
        "dist_coeffs",
        "reprojection_error",
        "image_width",
        "image_height",
    }
    missing = required_keys - set(data.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Missing calibration keys: {missing_text}")

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    metadata = {
        "reprojection_error": float(data["reprojection_error"]),
        "image_width": int(data["image_width"]),
        "image_height": int(data["image_height"]),
    }

    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")

    return camera_matrix, dist_coeffs, metadata
