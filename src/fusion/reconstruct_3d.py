"""3D reconstruction utilities from pixel observations and metric depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.common.data_models import Detection, Node3D


@dataclass(slots=True)
class CameraIntrinsics:
    """Compact camera intrinsics representation.

    Attributes are the usual pinhole intrinsics in pixel units.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k: Sequence[float] | np.ndarray) -> "CameraIntrinsics":
        """Build intrinsics from [fx, fy, cx, cy] or a 3x3 camera matrix."""
        arr = np.asarray(k, dtype=np.float64)
        if arr.shape == (4,):
            fx, fy, cx, cy = arr.tolist()
            return cls(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
        if arr.shape == (3, 3):
            return cls(fx=float(arr[0, 0]), fy=float(arr[1, 1]), cx=float(arr[0, 2]), cy=float(arr[1, 2]))
        raise ValueError(f"K must be [fx, fy, cx, cy] or 3x3 matrix, got shape {arr.shape}")


def pixel_to_camera_point(
    u: float,
    v: float,
    z_abs: float,
    intrinsics: CameraIntrinsics | Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert a pixel and absolute depth into camera coordinates.

    Equations:
    - X = (u - cx) * Z / fx
    - Y = (v - cy) * Z / fy
    - Z = Z_abs
    """
    k = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_k(intrinsics)
    z = float(z_abs)
    x = (float(u) - k.cx) * z / k.fx
    y = (float(v) - k.cy) * z / k.fy
    return x, y, z


def _sample_depth(depth_abs: np.ndarray, u: float, v: float) -> float:
    h, w = depth_abs.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    ui = int(np.clip(ui, 0, w - 1))
    vi = int(np.clip(vi, 0, h - 1))
    return float(depth_abs[vi, ui])


def reconstruct_nodes_3d(
    detections: Iterable[Detection],
    depth_abs: np.ndarray,
    intrinsics: CameraIntrinsics | Sequence[float] | np.ndarray,
) -> list[Node3D]:
    """Reconstruct 3D nodes for tracked detections from absolute depth map."""
    depth_abs = np.asarray(depth_abs, dtype=np.float32)
    k = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_k(intrinsics)

    nodes: list[Node3D] = []
    for det in detections:
        u, v = det.center_uv
        z_abs = _sample_depth(depth_abs, u=u, v=v)
        if not np.isfinite(z_abs):
            continue
        x, y, z = pixel_to_camera_point(u=u, v=v, z_abs=z_abs, intrinsics=k)
        nodes.append(Node3D(track_id=det.track_id, X=x, Y=y, Z=z, frame_id=det.frame_id, timestamp=det.timestamp))
    return nodes
