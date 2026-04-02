"""ArUco marker detection and per-marker absolute depth estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import cv2
import numpy as np


@dataclass(slots=True)
class ArucoMarkerPose:
    """Pose and quality information for one detected ArUco marker."""

    marker_id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    z_abs_m: float
    area_px: float


def _resolve_dictionary(dictionary_name: str) -> Any:
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def _marker_area(corners: np.ndarray) -> float:
    contour = corners.reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.contourArea(contour))


def _marker_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_marker_poses(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    config: Mapping[str, Any],
) -> list[ArucoMarkerPose]:
    """Detect ArUco markers and estimate each marker pose in camera coordinates.

    Notes:
    - Marker detection uses ``cv2.aruco.detectMarkers``.
    - Pose estimation uses ``cv2.aruco.estimatePoseSingleMarkers`` when all marker
      lengths are equal, otherwise falls back to per-marker ``cv2.solvePnP``.
    - ``tvec[2]`` is returned as ``z_abs_m`` (absolute depth in meters).
    """

    dictionary_name = str(config.get("dictionary", "DICT_4X4_50"))
    aruco_dict = _resolve_dictionary(dictionary_name)
    params = cv2.aruco.DetectorParameters()

    corners_list, ids, _ = cv2.aruco.detectMarkers(image_bgr, aruco_dict, parameters=params)
    if ids is None or len(ids) == 0:
        return []

    ids_flat = ids.flatten().astype(int)

    whitelist_cfg: Optional[Iterable[int]] = config.get("id_whitelist")
    whitelist = set(int(v) for v in whitelist_cfg) if whitelist_cfg else None
    min_marker_area_px = float(config.get("min_marker_area_px", 0.0))

    marker_lengths_by_id: Dict[int, float] = {
        int(k): float(v) for k, v in dict(config.get("marker_length_by_id_m", {})).items()
    }
    default_length = float(config.get("marker_length_m", 0.08))

    filtered: list[tuple[int, np.ndarray, float, float]] = []
    for marker_id, corners in zip(ids_flat, corners_list):
        if whitelist is not None and marker_id not in whitelist:
            continue

        area_px = _marker_area(corners)
        if area_px < min_marker_area_px:
            continue

        marker_length = marker_lengths_by_id.get(marker_id, default_length)
        filtered.append((marker_id, corners, area_px, marker_length))

    if not filtered:
        return []

    lengths = np.array([item[3] for item in filtered], dtype=np.float32)
    all_same_length = bool(np.allclose(lengths, lengths[0], rtol=1e-6, atol=1e-9))

    results: list[ArucoMarkerPose] = []
    if all_same_length:
        corners = [item[1] for item in filtered]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            float(lengths[0]),
            camera_matrix,
            dist_coeffs,
        )
        for (marker_id, marker_corners, area_px, _), rvec, tvec in zip(filtered, rvecs, tvecs):
            z_abs_m = float(tvec.reshape(3)[2])
            results.append(
                ArucoMarkerPose(
                    marker_id=marker_id,
                    corners=marker_corners,
                    rvec=rvec.reshape(3),
                    tvec=tvec.reshape(3),
                    z_abs_m=z_abs_m,
                    area_px=area_px,
                )
            )
        return results

    for marker_id, marker_corners, area_px, marker_length in filtered:
        object_points = _marker_object_points(marker_length)
        image_points = marker_corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue

        tvec_reshaped = tvec.reshape(3)
        results.append(
            ArucoMarkerPose(
                marker_id=marker_id,
                corners=marker_corners,
                rvec=rvec.reshape(3),
                tvec=tvec_reshaped,
                z_abs_m=float(tvec_reshaped[2]),
                area_px=area_px,
            )
        )

    return results
