"""Per-track 3D displacement with optional temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.common.data_models import Displacement, Node3D


@dataclass(slots=True)
class FilteredNode3D:
    """Raw + filtered 3D coordinates for a track observation."""

    track_id: int
    raw_xyz: tuple[float, float, float]
    filtered_xyz: tuple[float, float, float]
    frame_id: int
    timestamp: float


@dataclass(slots=True)
class FilteredDisplacement:
    """Raw + filtered displacement for one tracked target."""

    track_id: int
    raw: Displacement
    filtered: Displacement


class _LowPass3D:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.state: tuple[float, float, float] | None = None

    def update(self, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if self.state is None:
            self.state = value
            return value
        px, py, pz = self.state
        x, y, z = value
        fx = self.alpha * x + (1.0 - self.alpha) * px
        fy = self.alpha * y + (1.0 - self.alpha) * py
        fz = self.alpha * z + (1.0 - self.alpha) * pz
        self.state = (fx, fy, fz)
        return self.state


class _Kalman1D:
    def __init__(self, process_var: float, measure_var: float) -> None:
        self.q = process_var
        self.r = measure_var
        self.x: float | None = None
        self.p = 1.0

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class _Kalman3D:
    def __init__(self, process_var: float, measure_var: float) -> None:
        self.fx = _Kalman1D(process_var=process_var, measure_var=measure_var)
        self.fy = _Kalman1D(process_var=process_var, measure_var=measure_var)
        self.fz = _Kalman1D(process_var=process_var, measure_var=measure_var)

    def update(self, value: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = value
        return self.fx.update(x), self.fy.update(y), self.fz.update(z)


class DisplacementEstimator:
    """Track baseline 3D coordinates and emit raw + filtered displacement.

    Parameters
    ----------
    filter_type:
        ``"lowpass"`` or ``"kalman"``.
    lowpass_alpha:
        Smoothing factor for low-pass filter, larger means less smoothing.
    kalman_process_var / kalman_measure_var:
        Scalar 1D Kalman parameters applied per axis.
    """

    def __init__(
        self,
        filter_type: str = "lowpass",
        lowpass_alpha: float = 0.25,
        kalman_process_var: float = 1e-4,
        kalman_measure_var: float = 1e-2,
    ) -> None:
        if filter_type not in {"lowpass", "kalman"}:
            raise ValueError("filter_type must be 'lowpass' or 'kalman'")
        self.filter_type = filter_type
        self.lowpass_alpha = lowpass_alpha
        self.kalman_process_var = kalman_process_var
        self.kalman_measure_var = kalman_measure_var

        self.baseline_raw: dict[int, tuple[float, float, float]] = {}
        self.baseline_filtered: dict[int, tuple[float, float, float]] = {}
        self.filters: dict[int, _LowPass3D | _Kalman3D] = {}

    def _get_filter(self, track_id: int) -> _LowPass3D | _Kalman3D:
        filt = self.filters.get(track_id)
        if filt is not None:
            return filt
        if self.filter_type == "lowpass":
            filt = _LowPass3D(alpha=self.lowpass_alpha)
        else:
            filt = _Kalman3D(process_var=self.kalman_process_var, measure_var=self.kalman_measure_var)
        self.filters[track_id] = filt
        return filt

    @staticmethod
    def _disp(track_id: int, xyz: tuple[float, float, float], baseline: tuple[float, float, float], frame_id: int, ts: float) -> Displacement:
        dx = xyz[0] - baseline[0]
        dy = xyz[1] - baseline[1]
        dz = xyz[2] - baseline[2]
        d3d = sqrt(dx * dx + dy * dy + dz * dz)
        return Displacement(track_id=track_id, dX=dx, dY=dy, dZ=dz, d3d=d3d, frame_id=frame_id, timestamp=ts)

    def update(self, nodes: list[Node3D]) -> list[FilteredDisplacement]:
        """Update estimator for one frame and return raw+filtered displacement."""
        out: list[FilteredDisplacement] = []
        for node in nodes:
            track_id = node.track_id
            raw_xyz = (node.X, node.Y, node.Z)
            filt_xyz = self._get_filter(track_id).update(raw_xyz)

            if track_id not in self.baseline_raw:
                self.baseline_raw[track_id] = raw_xyz
            if track_id not in self.baseline_filtered:
                self.baseline_filtered[track_id] = filt_xyz

            raw_disp = self._disp(track_id, raw_xyz, self.baseline_raw[track_id], node.frame_id, node.timestamp)
            filt_disp = self._disp(track_id, filt_xyz, self.baseline_filtered[track_id], node.frame_id, node.timestamp)
            out.append(FilteredDisplacement(track_id=track_id, raw=raw_disp, filtered=filt_disp))
        return out

    def update_nodes(self, nodes: list[Node3D]) -> list[FilteredNode3D]:
        """Return raw + filtered node coordinates for evaluation/diagnostics."""
        rows: list[FilteredNode3D] = []
        for node in nodes:
            raw_xyz = (node.X, node.Y, node.Z)
            filt_xyz = self._get_filter(node.track_id).update(raw_xyz)
            rows.append(
                FilteredNode3D(
                    track_id=node.track_id,
                    raw_xyz=raw_xyz,
                    filtered_xyz=filt_xyz,
                    frame_id=node.frame_id,
                    timestamp=node.timestamp,
                )
            )
        return rows
