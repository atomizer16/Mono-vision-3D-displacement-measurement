"""YOLO11-SEG detector wrapper.

This module wraps Ultralytics YOLO11 segmentation inference and returns
normalized per-instance outputs for downstream tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DetectorOutput:
    """Single instance output from YOLO11-SEG."""

    cls_id: int
    cls_name: str
    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    center_uv: tuple[float, float]
    mask_polygon_uv: list[tuple[float, float]]


class YOLO11SegDetector:
    """Load YOLO11-SEG weights and output class/mask/bbox/confidence."""

    def __init__(
        self,
        weights: str | Path = "weights/yolo11-seg.pt",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.weights = str(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Ultralytics is required for YOLO11SegDetector. Install with `pip install ultralytics`."
            ) from exc

        self._model = YOLO(self.weights)

    @staticmethod
    def _bbox_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _polygon_center(mask_polygon_uv: list[tuple[float, float]]) -> tuple[float, float] | None:
        if not mask_polygon_uv:
            return None
        sx = sum(p[0] for p in mask_polygon_uv)
        sy = sum(p[1] for p in mask_polygon_uv)
        n = float(len(mask_polygon_uv))
        return sx / n, sy / n

    def detect(self, frame: Any) -> list[DetectorOutput]:
        """Run YOLO11-SEG inference and return normalized instance outputs."""
        results = self._model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        boxes = result.boxes
        masks = result.masks

        mask_polygons: list[list[tuple[float, float]]] = []
        if masks is not None:
            for poly in masks.xy:
                mask_polygons.append([(float(x), float(y)) for x, y in poly.tolist()])

        outputs: list[DetectorOutput] = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bbox_xyxy = (x1, y1, x2, y2)

            polygon = mask_polygons[idx] if idx < len(mask_polygons) else []
            center_uv = self._polygon_center(polygon) or self._bbox_center(bbox_xyxy)

            outputs.append(
                DetectorOutput(
                    cls_id=cls_id,
                    cls_name=str(names[cls_id]),
                    bbox_xyxy=bbox_xyxy,
                    conf=conf,
                    center_uv=center_uv,
                    mask_polygon_uv=polygon,
                )
            )

        return outputs


# Backward-compatible alias if old imports still reference YOLO11Detector.
YOLO11Detector = YOLO11SegDetector
