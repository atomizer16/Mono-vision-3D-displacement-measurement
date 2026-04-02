"""Chessboard-based camera intrinsic calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


@dataclass(slots=True)
class CalibrationResult:
    """Calibration outputs needed by downstream modules."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    image_size: tuple[int, int]


def _build_object_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def collect_calibration_points(
    image_paths: Sequence[Path],
    board_size: tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[Path]]:
    """Read images in batch and detect chessboard corners for calibration."""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_template = _build_object_points(board_size=board_size, square_size=square_size)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    valid_image_paths: list[Path] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] skip unreadable image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            print(f"[WARN] chessboard not found: {image_path}")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(obj_template.copy())
        img_points.append(refined)
        valid_image_paths.append(image_path)

    if not obj_points or image_size is None:
        raise ValueError("No valid chessboard detections found in the provided images.")

    return obj_points, img_points, image_size, valid_image_paths


def calibrate_camera_from_images(
    image_paths: Sequence[Path],
    board_size: tuple[int, int],
    square_size: float,
) -> CalibrationResult:
    """Estimate camera intrinsics with OpenCV calibrateCamera."""

    obj_points, img_points, image_size, _valid_image_paths = collect_calibration_points(
        image_paths=image_paths,
        board_size=board_size,
        square_size=square_size,
    )
    ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=float(ret),
        image_size=image_size,
    )


def save_calibration_yaml(result: CalibrationResult, output_path: Path) -> None:
    """Persist camera intrinsics and metadata to YAML."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.reshape(-1).tolist(),
        "reprojection_error": result.reprojection_error,
        "image_width": int(result.image_size[0]),
        "image_height": int(result.image_size[1]),
    }
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def discover_images(images_dir: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    """Collect calibration images from a directory."""

    search_patterns = tuple(patterns or ("*.jpg", "*.jpeg", "*.png", "*.bmp"))
    image_paths: list[Path] = []
    for pattern in search_patterns:
        image_paths.extend(images_dir.glob(pattern))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return image_paths


def run_chessboard_calibration(
    images_dir: Path,
    board_size: tuple[int, int],
    square_size: float,
    output_path: Path = Path("artifacts/camera_intrinsics.yaml"),
) -> CalibrationResult:
    """High-level helper used by scripts and tests."""

    image_paths = discover_images(images_dir)
    result = calibrate_camera_from_images(
        image_paths=image_paths,
        board_size=board_size,
        square_size=square_size,
    )
    save_calibration_yaml(result, output_path)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chessboard camera intrinsic calibration")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing calibration images")
    parser.add_argument("--board-cols", type=int, required=True, help="Chessboard inner corners per row")
    parser.add_argument("--board-rows", type=int, required=True, help="Chessboard inner corners per column")
    parser.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size in user units")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/camera_intrinsics.yaml"),
        help="Output YAML path",
    )
    args = parser.parse_args()

    calib_result = run_chessboard_calibration(
        images_dir=args.images_dir,
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size,
        output_path=args.output,
    )
    print(f"Calibration done. RMS reprojection error: {calib_result.reprojection_error:.6f}")
    print(f"Saved intrinsics to: {args.output}")
