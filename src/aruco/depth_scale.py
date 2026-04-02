"""Depth scale fitting using ArUco anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .aruco_pose import ArucoMarkerPose


@dataclass(slots=True)
class ScaleFitResult:
    """Per-frame depth scaling result."""

    scale_a: float
    scale_b: float
    fit_rmse: float
    num_anchors: int
    quality: str


@dataclass(slots=True)
class DepthAnchor:
    marker_id: int
    z_rel: float
    z_abs_m: float


def _sample_marker_depth(depth_rel: np.ndarray, corners: np.ndarray) -> float:
    h, w = depth_rel.shape[:2]
    polygon = corners.reshape(-1, 2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    values = depth_rel[mask > 0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    rmse = float(np.sqrt(np.mean((pred - ys) ** 2)))
    return float(a), float(b), rmse


def _ransac_linear_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    residual_threshold: float = 0.08,
    max_trials: int = 64,
    random_state: int = 7,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(xs)

    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(max_trials):
        sample_idx = rng.choice(n, size=2, replace=False)
        x_s = xs[sample_idx]
        y_s = ys[sample_idx]
        if np.isclose(x_s[0], x_s[1]):
            continue

        a = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        b = y_s[0] - a * x_s[0]
        residuals = np.abs((a * xs + b) - ys)
        inliers = residuals <= residual_threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 2:
        return _linear_fit(xs, ys)

    return _linear_fit(xs[best_inliers], ys[best_inliers])


class DepthScaleCalibrator:
    """Builds and stabilizes relative->absolute depth scale from ArUco anchors."""

    def __init__(self, config: Mapping[str, Any]):
        self.min_markers = int(config.get("min_markers", 2))
        self.ransac_threshold = float(config.get("ransac_residual_threshold_m", 0.08))
        self.ransac_max_trials = int(config.get("ransac_max_trials", 64))

        self._last_stable: ScaleFitResult | None = None

    def build_anchors(self, depth_rel: np.ndarray, marker_poses: Sequence[ArucoMarkerPose]) -> list[DepthAnchor]:
        anchors: list[DepthAnchor] = []
        for pose in marker_poses:
            z_rel = _sample_marker_depth(depth_rel, pose.corners)
            if not np.isfinite(z_rel):
                continue
            anchors.append(DepthAnchor(marker_id=pose.marker_id, z_rel=float(z_rel), z_abs_m=float(pose.z_abs_m)))
        return anchors

    def fit_frame(
        self,
        depth_rel: np.ndarray,
        marker_poses: Sequence[ArucoMarkerPose],
    ) -> ScaleFitResult:
        anchors = self.build_anchors(depth_rel=depth_rel, marker_poses=marker_poses)
        num_anchors = len(anchors)

        if num_anchors < self.min_markers:
            if self._last_stable is not None:
                return ScaleFitResult(
                    scale_a=self._last_stable.scale_a,
                    scale_b=self._last_stable.scale_b,
                    fit_rmse=self._last_stable.fit_rmse,
                    num_anchors=num_anchors,
                    quality="fallback_stable",
                )
            return ScaleFitResult(
                scale_a=1.0,
                scale_b=0.0,
                fit_rmse=float("inf"),
                num_anchors=num_anchors,
                quality="insufficient_no_history",
            )

        xs = np.array([a.z_rel for a in anchors], dtype=np.float64)
        ys = np.array([a.z_abs_m for a in anchors], dtype=np.float64)

        if num_anchors >= 3:
            scale_a, scale_b, rmse = _ransac_linear_fit(
                xs,
                ys,
                residual_threshold=self.ransac_threshold,
                max_trials=self.ransac_max_trials,
            )
            quality = "good_ransac"
        else:
            scale_a, scale_b, rmse = _linear_fit(xs, ys)
            quality = "good_linear"

        result = ScaleFitResult(
            scale_a=scale_a,
            scale_b=scale_b,
            fit_rmse=rmse,
            num_anchors=num_anchors,
            quality=quality,
        )
        self._last_stable = result
        return result


def apply_depth_scale(depth_rel: np.ndarray, fit: ScaleFitResult) -> np.ndarray:
    """Convert relative depth map to metric depth with current fit parameters."""

    return fit.scale_a * depth_rel + fit.scale_b
