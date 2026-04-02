"""Offline pipeline runner: video -> detect/track -> depth -> aruco scale -> 3D -> displacement/export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.aruco.aruco_pose import detect_marker_poses
from src.aruco.depth_scale import DepthScaleCalibrator, apply_depth_scale
from src.depth.depth_anything_v2_infer import DepthAnythingV2Infer
from src.detection.botsort_tracker import BoTSORTTracker
from src.detection.yolo11_detector import YOLO11SegDetector
from src.fusion.displacement import DisplacementEstimator
from src.fusion.reconstruct_3d import CameraIntrinsics, reconstruct_nodes_3d
from src.io.camera_io import load_camera_intrinsics
from src.io.exporter import DisplacementExporter
from src.io.overlay import build_quality_alerts, draw_frame_alerts, draw_track_overlay


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def _quality_score(
    num_anchors: int,
    min_anchors: int,
    fit_rmse: float,
    reproj_error: float,
    id_switch_rate: float,
) -> float:
    anchor_term = min(1.0, num_anchors / max(min_anchors, 1))
    fit_term = max(0.0, 1.0 - fit_rmse / 0.20)
    reproj_term = max(0.0, 1.0 - reproj_error / 1.0)
    switch_term = max(0.0, 1.0 - id_switch_rate)
    score = 0.35 * anchor_term + 0.30 * fit_term + 0.20 * reproj_term + 0.15 * switch_term
    return float(max(0.0, min(1.0, score)))


def run(args: argparse.Namespace) -> None:
    yolo_cfg = _load_yaml(args.yolo_config)
    depth_cfg = _load_yaml(args.depth_config)
    aruco_cfg = _load_yaml(args.aruco_config)
    output_cfg = _load_yaml(args.output_config)

    camera_matrix, dist_coeffs, camera_meta = load_camera_intrinsics(args.camera_file)
    reproj_error = float(camera_meta.get("reprojection_error", 0.0))

    detector = YOLO11SegDetector(
        weights=yolo_cfg.get("detection", {}).get("weights", "weights/yolo11-seg.pt"),
        conf_thres=float(yolo_cfg.get("detection", {}).get("conf_thres", 0.25)),
        iou_thres=float(yolo_cfg.get("detection", {}).get("iou_thres", 0.45)),
    )
    tracker = BoTSORTTracker(
        iou_thres=float(yolo_cfg.get("tracking", {}).get("iou_thres", 0.3)),
        max_age=int(yolo_cfg.get("tracking", {}).get("max_age", 30)),
        min_hits=int(yolo_cfg.get("tracking", {}).get("min_hits", 3)),
        reconnect_frames=int(yolo_cfg.get("tracking", {}).get("reconnect", {}).get("reconnect_frames", 15)),
    )
    depth = DepthAnythingV2Infer(config=depth_cfg)
    calibrator = DepthScaleCalibrator(config=aruco_cfg)
    displacement = DisplacementEstimator(filter_type="lowpass")
    exporter = DisplacementExporter()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 1e-3 else 30.0

    out_writer = None
    if args.output_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(str(args.output_video), fourcc, fps, (width, height))

    seen_track_ids: set[int] = set()
    id_switch_events = 0
    frame_id = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_id / fps
        raw_dets = detector.detect(frame)
        tracked = tracker.update(raw_dets, frame_id=frame_id, timestamp=timestamp)

        current_ids = {d.track_id for d in tracked}
        new_ids = current_ids - seen_track_ids
        if frame_id > 0 and len(new_ids) > 0 and len(seen_track_ids) > 0:
            id_switch_events += len(new_ids)
        seen_track_ids.update(current_ids)

        depth_rel, _ = depth.run(frame)
        marker_poses = detect_marker_poses(frame, camera_matrix, dist_coeffs, aruco_cfg)
        fit = calibrator.fit_frame(depth_rel=depth_rel, marker_poses=marker_poses)
        depth_abs = apply_depth_scale(depth_rel, fit)

        nodes = reconstruct_nodes_3d(
            detections=tracked,
            depth_abs=depth_abs,
            intrinsics=CameraIntrinsics.from_k(camera_matrix),
        )
        disp_results = displacement.update(nodes)
        disp_by_track = {item.track_id: item.filtered for item in disp_results}
        node_by_track = {node.track_id: node for node in nodes}

        id_switch_rate = id_switch_events / max(frame_id + 1, 1)
        quality_flags = {
            "insufficient_aruco": fit.num_anchors < int(aruco_cfg.get("min_markers", 2)),
            "high_reprojection_error": reproj_error > float(args.reprojection_error_thres),
            "reprojection_error": reproj_error,
            "frequent_id_switch": id_switch_rate > float(args.id_switch_rate_thres),
            "id_switch_count": id_switch_events,
        }

        quality_score = _quality_score(
            num_anchors=fit.num_anchors,
            min_anchors=int(aruco_cfg.get("min_markers", 2)),
            fit_rmse=fit.fit_rmse,
            reproj_error=reproj_error,
            id_switch_rate=id_switch_rate,
        )

        for det in tracked:
            node = node_by_track.get(det.track_id)
            disp = disp_by_track.get(det.track_id)
            if node is None or disp is None:
                continue
            exporter.add_row(node=node, displacement=disp, quality_score=quality_score)
            draw_track_overlay(frame, det, node, disp, quality_score=quality_score)

        alerts = build_quality_alerts(quality_flags)
        draw_frame_alerts(frame, alerts)

        if out_writer is not None:
            out_writer.write(frame)
        if args.show:
            cv2.imshow("pipeline", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_id += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    export_cfg = output_cfg.get("export", {})
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(export_cfg.get("csv", True)):
        exporter.write_csv(out_dir / "displacement.csv")
    if bool(export_cfg.get("json", True)):
        exporter.write_json(out_dir / "displacement.json")

    quality_report = {
        "frames": frame_id,
        "elapsed_sec": time.perf_counter() - t0,
        "unique_tracks": len(seen_track_ids),
        "id_switch_events": id_switch_events,
        "reprojection_error": reproj_error,
        "warnings": build_quality_alerts(quality_flags),
    }
    with (out_dir / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full mono-vision displacement pipeline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera-file", type=Path, default=Path("data/calibration/camera_params.yaml"))
    parser.add_argument("--yolo-config", type=Path, default=Path("configs/yolo.yaml"))
    parser.add_argument("--depth-config", type=Path, default=Path("configs/depth.yaml"))
    parser.add_argument("--aruco-config", type=Path, default=Path("configs/aruco.yaml"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/output.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--reprojection-error-thres", type=float, default=0.5)
    parser.add_argument("--id-switch-rate-thres", type=float, default=0.2)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
