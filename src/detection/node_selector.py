"""Node target selector for steel-frame displacement measurement."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from src.common.data_models import Detection

KeypointRefiner = Callable[[Detection], tuple[float, float] | None]


class NodeSelector:
    """Filter and refine detections to steel-frame node targets."""

    def __init__(
        self,
        target_classes: set[str] | None = None,
        roi_xyxy: tuple[float, float, float, float] | None = None,
        conf_thres: float = 0.25,
        keypoint_refiner: KeypointRefiner | None = None,
    ) -> None:
        self.target_classes = target_classes
        self.roi_xyxy = roi_xyxy
        self.conf_thres = conf_thres
        self.keypoint_refiner = keypoint_refiner

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _in_roi(self, u: float, v: float) -> bool:
        if self.roi_xyxy is None:
            return True
        x1, y1, x2, y2 = self.roi_xyxy
        return x1 <= u <= x2 and y1 <= v <= y2

    def select(self, tracked: list[Detection]) -> list[Detection]:
        """Select node targets by class/ROI/confidence and provide center (u,v)."""
        selected: list[Detection] = []

        for det in tracked:
            if det.conf < self.conf_thres:
                continue
            if self.target_classes and det.cls not in self.target_classes:
                continue

            center_uv = det.center_uv or self._bbox_center(det.bbox_xyxy)
            if self.keypoint_refiner is not None:
                refined = self.keypoint_refiner(det)
                if refined is not None:
                    center_uv = refined

            if not self._in_roi(*center_uv):
                continue

            selected.append(replace(det, center_uv=center_uv))

        return selected
