# Mono-vision-3D-displacement-measurement

Source code of "3D vibration displacement measurement based on monocular computer vision".

## Recommended project structure

```text
.
├── apps/
│   ├── offline_video.py
│   └── realtime_stream.py
├── configs/
│   ├── aruco.yaml
│   ├── camera.yaml
│   ├── depth.yaml
│   ├── output.yaml
│   └── yolo.yaml
└── src/
    ├── aruco/        # ArUco detection, PnP, and scale recovery
    ├── calibration/  # Camera calibration and parameter I/O
    ├── common/       # Shared data models and common types
    ├── depth/        # Depth Anything V2 inference and relative depth processing
    ├── detection/    # YOLO11 detection and BoT-SORT tracking
    ├── fusion/       # Coordinate transforms, 3D recovery, displacement calculation
    └── io/           # Config/log/result export (CSV/JSON)
```

## Unified data structures

Implemented in `src/common/data_models.py`:

- `CameraParams(K, dist, rvec, tvec, image_size)`
- `Detection(track_id, cls, conf, bbox_xyxy, center_uv, frame_id, timestamp)`
- `Node3D(track_id, X, Y, Z, frame_id, timestamp)`
- `Displacement(track_id, dX, dY, dZ, d3d, frame_id, timestamp)`

## Coordinate conventions

- **Image coordinates**: `(u, v)` in **pixels**, origin at image top-left.
- **Camera coordinates**: `(Xc, Yc, Zc)` in **meters**, right-handed camera frame.
- **World/engineering coordinates**: `(Xw, Yw, Zw)` in **meters**, defined by calibration and/or ArUco reference frame.

### Displacement baseline definition

In this project, displacement is defined **relative to a baseline frame/time**.

- Default baseline: the first valid observation frame for each `track_id`.
- Optional baseline: a user-defined reference timestamp (for example after warm-up).

All reported `dX, dY, dZ, d3d` follow that baseline policy and should be kept consistent for a full run.

## Configuration split

- `configs/camera.yaml`: calibration parameter file path.
- `configs/depth.yaml`: model weights, input size, normalization settings.
- `configs/yolo.yaml`: confidence/NMS/tracking thresholds.
- `configs/aruco.yaml`: dictionary type, marker side length, ID whitelist.
- `configs/output.yaml`: output frequency, smoothing window, export formats.
