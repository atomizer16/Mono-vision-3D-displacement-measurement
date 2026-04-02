"""Overlay rendering for tracked 3D state and quality."""

from __future__ import annotations

from typing import Any, Mapping

import cv2

from src.common.data_models import Detection, Displacement, Node3D


def _put_lines(frame: Any, x: int, y: int, lines: list[str], color: tuple[int, int, int]) -> None:
    for i, line in enumerate(lines):
        y_i = y + i * 16
        cv2.putText(frame, line, (x, y_i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y_i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_track_overlay(
    frame_bgr: Any,
    detection: Detection,
    node: Node3D,
    displacement: Displacement,
    quality_score: float,
) -> None:
    """Draw one tracked target's id, XYZ and displacement values."""
    x1, y1, x2, y2 = detection.bbox_xyxy
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.rectangle(frame_bgr, p1, p2, (0, 255, 255), 2)

    anchor_x = max(int(x1), 8)
    anchor_y = max(int(y1) - 10, 20)

    lines = [
        f"id={detection.track_id} q={quality_score:.2f}",
        f"X/Y/Z=({node.X:.3f},{node.Y:.3f},{node.Z:.3f})m",
        f"dX/dY/dZ=({displacement.dX:.3f},{displacement.dY:.3f},{displacement.dZ:.3f})m",
    ]
    draw_color = (0, 255, 0) if quality_score >= 0.6 else (0, 165, 255)
    _put_lines(frame_bgr, anchor_x, anchor_y, lines, draw_color)


def draw_frame_alerts(frame_bgr: Any, alerts: list[str]) -> None:
    """Draw frame-level quality alerts on top-left corner."""
    if not alerts:
        return
    _put_lines(frame_bgr, 8, 20, [f"[WARN] {m}" for m in alerts], (0, 0, 255))


def build_quality_alerts(meta: Mapping[str, Any]) -> list[str]:
    """Create warning messages from quality monitor metadata."""
    alerts: list[str] = []
    if meta.get("insufficient_aruco"):
        alerts.append("Aruco anchors insufficient")
    if meta.get("high_reprojection_error"):
        reproj_err = float(meta.get("reprojection_error", 0.0))
        alerts.append(f"Reprojection error high ({reproj_err:.3f})")
    if meta.get("frequent_id_switch"):
        switches = int(meta.get("id_switch_count", 0))
        alerts.append(f"Frequent track id switches ({switches})")
    return alerts
