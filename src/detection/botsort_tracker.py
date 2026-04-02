"""Detection-driven BoT-SORT style tracker.

This tracker consumes YOLO11-SEG instance outputs and assigns stable track IDs.
It keeps recently lost tracks for short-term reconnection during occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.data_models import Detection
from src.detection.yolo11_detector import DetectorOutput


@dataclass(slots=True)
class _Track:
    track_id: int
    cls_name: str
    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    center_uv: tuple[float, float]
    hits: int
    age: int
    time_since_update: int


class BoTSORTTracker:
    """Simple BoT-SORT style association over frame-level detections."""

    def __init__(
        self,
        iou_thres: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        reconnect_frames: int = 15,
    ) -> None:
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.min_hits = min_hits
        self.reconnect_frames = reconnect_frames

        self._next_track_id = 1
        self._active_tracks: dict[int, _Track] = {}
        self._lost_tracks: dict[int, _Track] = {}

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _match(self, det: DetectorOutput, candidates: dict[int, _Track]) -> int | None:
        best_track_id: int | None = None
        best_iou = self.iou_thres
        for track_id, track in candidates.items():
            if track.cls_name != det.cls_name:
                continue
            iou = self._iou(det.bbox_xyxy, track.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        return best_track_id

    def update(
        self,
        detections: list[DetectorOutput],
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        """Associate detections and return stable tracked detections."""
        for track in self._active_tracks.values():
            track.age += 1
            track.time_since_update += 1
        for track in self._lost_tracks.values():
            track.age += 1
            track.time_since_update += 1

        output: list[Detection] = []
        used_active: set[int] = set()
        used_lost: set[int] = set()

        for det in detections:
            track_id = self._match(det, {k: v for k, v in self._active_tracks.items() if k not in used_active})

            if track_id is None:
                reconnect_pool = {
                    k: v
                    for k, v in self._lost_tracks.items()
                    if k not in used_lost and v.time_since_update <= self.reconnect_frames
                }
                track_id = self._match(det, reconnect_pool)
                if track_id is not None:
                    track = self._lost_tracks.pop(track_id)
                    self._active_tracks[track_id] = track
                    used_lost.add(track_id)

            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks[track_id] = _Track(
                    track_id=track_id,
                    cls_name=det.cls_name,
                    bbox_xyxy=det.bbox_xyxy,
                    conf=det.conf,
                    center_uv=det.center_uv,
                    hits=1,
                    age=1,
                    time_since_update=0,
                )
            else:
                track = self._active_tracks[track_id]
                track.bbox_xyxy = det.bbox_xyxy
                track.conf = det.conf
                track.center_uv = det.center_uv
                track.hits += 1
                track.time_since_update = 0

            used_active.add(track_id)
            track = self._active_tracks[track_id]

            if track.hits >= self.min_hits:
                output.append(
                    Detection(
                        track_id=track.track_id,
                        cls=track.cls_name,
                        conf=track.conf,
                        bbox_xyxy=track.bbox_xyxy,
                        center_uv=track.center_uv,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        stale_ids: list[int] = []
        for track_id, track in self._active_tracks.items():
            if track_id in used_active:
                continue
            if track.time_since_update <= self.reconnect_frames:
                self._lost_tracks[track_id] = track
            if track.time_since_update > self.max_age:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self._active_tracks.pop(track_id, None)
            self._lost_tracks.pop(track_id, None)

        prune_lost = [tid for tid, tr in self._lost_tracks.items() if tr.time_since_update > self.max_age]
        for tid in prune_lost:
            self._lost_tracks.pop(tid, None)

        return output
