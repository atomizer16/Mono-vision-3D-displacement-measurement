"""Calibration validation with per-image reprojection error diagnostics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.calibration.chessboard_calib import collect_calibration_points, discover_images
from src.io.camera_io import load_camera_intrinsics


def validate_reprojection_errors(
    images_dir: Path,
    intrinsics_path: Path,
    board_size: tuple[int, int],
    square_size: float,
    warn_threshold: float = 0.5,
) -> tuple[list[tuple[Path, float]], float]:
    """Compute reprojection error per image and report mean error."""

    image_paths = discover_images(images_dir)
    obj_points, img_points, image_size, valid_image_paths = collect_calibration_points(
        image_paths=image_paths,
        board_size=board_size,
        square_size=square_size,
    )

    camera_matrix, dist_coeffs, _meta = load_camera_intrinsics(intrinsics_path)

    _, refined_K, refined_dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    errors: list[tuple[Path, float]] = []

    for idx, objp in enumerate(obj_points):
        projected, _ = cv2.projectPoints(objp, rvecs[idx], tvecs[idx], refined_K, refined_dist)
        err = cv2.norm(img_points[idx], projected, cv2.NORM_L2) / len(projected)
        errors.append((valid_image_paths[idx], float(err)))

    mean_error = float(np.mean([e for _, e in errors])) if errors else float("nan")

    for image_path, err in errors:
        tag = "WARN" if err > warn_threshold else "OK"
        print(f"[{tag}] {image_path.name}: {err:.4f} px")
    print(f"Mean reprojection error: {mean_error:.4f} px")
    print(f"Warning threshold: {warn_threshold:.4f} px")

    return errors, mean_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate camera calibration reprojection error")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing calibration images")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("artifacts/camera_intrinsics.yaml"),
        help="Path to calibration YAML",
    )
    parser.add_argument("--board-cols", type=int, required=True, help="Chessboard inner corners per row")
    parser.add_argument("--board-rows", type=int, required=True, help="Chessboard inner corners per column")
    parser.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size in user units")
    parser.add_argument("--warn-threshold", type=float, default=0.5, help="Warning threshold in pixels")
    args = parser.parse_args()

    validate_reprojection_errors(
        images_dir=args.images_dir,
        intrinsics_path=args.intrinsics,
        board_size=(args.board_cols, args.board_rows),
        square_size=args.square_size,
        warn_threshold=args.warn_threshold,
    )
