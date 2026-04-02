"""Camera-to-world extrinsic estimation and coordinate transform utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class ExtrinsicResult:
    """Rigid transform from camera frame to world/engineering frame."""

    R: np.ndarray
    t: np.ndarray
    reprojection_error: float


def solve_extrinsic_from_points(
    world_points_xyz: Sequence[Sequence[float]] | np.ndarray,
    image_points_uv: Sequence[Sequence[float]] | np.ndarray,
    k: Sequence[float] | np.ndarray,
    dist_coeffs: Sequence[float] | np.ndarray | None = None,
) -> ExtrinsicResult:
    """Estimate camera pose from control points or ArUco-board corner correspondences.

    ``world_points_xyz`` are engineering/world coordinates.
    ``image_points_uv`` are pixel points in the same order.
    """
    obj = np.asarray(world_points_xyz, dtype=np.float64)
    img = np.asarray(image_points_uv, dtype=np.float64)
    if obj.ndim != 2 or obj.shape[1] != 3:
        raise ValueError("world_points_xyz must be shape (N,3)")
    if img.ndim != 2 or img.shape[1] != 2:
        raise ValueError("image_points_uv must be shape (N,2)")
    if obj.shape[0] < 4:
        raise ValueError("At least 4 points are required for solvePnP")
    if obj.shape[0] != img.shape[0]:
        raise ValueError("world_points and image_points must have same length")

    k_arr = np.asarray(k, dtype=np.float64)
    if k_arr.shape == (4,):
        fx, fy, cx, cy = k_arr.tolist()
        k_arr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if k_arr.shape != (3, 3):
        raise ValueError("k must be [fx,fy,cx,cy] or 3x3 matrix")

    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, k_arr, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")

    rmat, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k_arr, dist)
    proj = proj.reshape(-1, 2)
    reproj = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))

    return ExtrinsicResult(R=rmat.astype(np.float64), t=tvec.reshape(3).astype(np.float64), reprojection_error=reproj)


def camera_to_world(pc_xyz: Sequence[float] | np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform camera-frame point Pc to world frame by Pw = R*Pc + t."""
    pc = np.asarray(pc_xyz, dtype=np.float64).reshape(3)
    r_arr = np.asarray(r, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t, dtype=np.float64).reshape(3)
    return r_arr @ pc + t_arr


def batch_camera_to_world(pc_xyz: Sequence[Sequence[float]] | np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorized camera-to-world transformation for Nx3 points."""
    pcs = np.asarray(pc_xyz, dtype=np.float64)
    if pcs.ndim != 2 or pcs.shape[1] != 3:
        raise ValueError("pc_xyz must be shape (N,3)")
    r_arr = np.asarray(r, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t, dtype=np.float64).reshape(1, 3)
    return (r_arr @ pcs.T).T + t_arr
