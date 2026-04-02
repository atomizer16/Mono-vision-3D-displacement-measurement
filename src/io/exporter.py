"""CSV/JSON exporters for 3D reconstruction and displacement rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from src.common.data_models import Displacement, Node3D

EXPORT_FIELDS = (
    "timestamp",
    "frame_id",
    "track_id",
    "X",
    "Y",
    "Z",
    "dX",
    "dY",
    "dZ",
    "d3d",
    "quality_score",
)


class DisplacementExporter:
    """Collect per-frame rows and export to CSV/JSON."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add_row(
        self,
        node: Node3D,
        displacement: Displacement,
        quality_score: float,
    ) -> dict[str, Any]:
        row = {
            "timestamp": float(node.timestamp),
            "frame_id": int(node.frame_id),
            "track_id": int(node.track_id),
            "X": float(node.X),
            "Y": float(node.Y),
            "Z": float(node.Z),
            "dX": float(displacement.dX),
            "dY": float(displacement.dY),
            "dZ": float(displacement.dZ),
            "d3d": float(displacement.d3d),
            "quality_score": float(quality_score),
        }
        self._rows.append(row)
        return row

    def extend_rows(self, rows: list[Mapping[str, Any]]) -> None:
        for row in rows:
            self._rows.append({k: row[k] for k in EXPORT_FIELDS})

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def write_csv(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(EXPORT_FIELDS))
            writer.writeheader()
            writer.writerows(self._rows)
        return out_path

    def write_json(self, path: str | Path, indent: int = 2) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self._rows, f, ensure_ascii=False, indent=indent)
        return out_path
