"""Unified data models for the mono-vision 3D displacement pipeline."""

from dataclasses import dataclass
from typing import Tuple


Matrix3x3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
Vector = Tuple[float, ...]
ImageSize = Tuple[int, int]
BBoxXYXY = Tuple[float, float, float, float]
PixelUV = Tuple[float, float]


@dataclass(slots=True)
class CameraParams:
    """Camera calibration and extrinsic parameters."""

    K: Matrix3x3
    dist: Vector
    rvec: Tuple[float, float, float]
    tvec: Tuple[float, float, float]
    image_size: ImageSize


@dataclass(slots=True)
class Detection:
    """2D detection and tracking result in image space."""

    track_id: int
    cls: str
    conf: float
    bbox_xyxy: BBoxXYXY
    center_uv: PixelUV
    frame_id: int
    timestamp: float


@dataclass(slots=True)
class Node3D:
    """3D point reconstructed for a tracked target."""

    track_id: int
    X: float
    Y: float
    Z: float
    frame_id: int
    timestamp: float


@dataclass(slots=True)
class Displacement:
    """Per-frame displacement relative to a defined baseline frame/time."""

    track_id: int
    dX: float
    dY: float
    dZ: float
    d3d: float
    frame_id: int
    timestamp: float
