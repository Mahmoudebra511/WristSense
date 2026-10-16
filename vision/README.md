# Webcam vision pipeline (Section 5.4)

Writes one row per second to `vision.csv` per the [session data
contract](../docs/schema.md):

| column | meaning |
|---|---|
| ts_ms | PC clock, epoch ms |
| face_present | 1 if a face was detected this second |
| head_yaw / head_pitch | head rotation in degrees (0 = calibrated neutral) |
| eye_closure | 0 = open, 1 = closed (relative to calibrated baseline) |
| conf | detection confidence 0..1 |

Privacy by construction: **no video, no frames are stored** — only these
numbers per second.

## Install

```bash
pip install -r requirements-vision.txt   # opencv-python, mediapipe, numpy
```

The pipeline runs without OpenCV/MediaPipe using `--simulate`.

## Run

```bash
# synthetic (no camera, no mediapipe) — good for a quick look
python -m vision.run --simulate --duration 20 --participant p01

# real webcam with 10 s calibration
python -m vision.run --camera-index 0 --duration 120

# skip calibration (baseline = neutral defaults)
python -m vision.run --no-calibration --duration 60

# attach to a session the hub/logger already created
python -m vision.run --session-dir data/raw/p01_20261020T141530_laptop --duration 120
```

During calibration, look straight ahead with your eyes open for ~10 seconds.
The pipeline neutralises per-person anatomy so yaw/pitch/eye_closure are
comparable across users.

Ctrl-C finalises `meta.json` (streams, row count, measured 1 Hz rate,
calibration baselines in `notes`) and prints a validation report.

## Architecture

| file | responsibility |
|---|---|
| `pipeline.py` | FrameResult, Detector/Camera protocols, 10 s calibration, 1 Hz pipeline |
| `detector.py` | MediaPipe FaceMesh face detector (head pose via solvePnP, eye closure via EAR) |
| `writer.py` | `vision.csv` sink |
| `run.py` | CLI |

`VisionPipeline` takes its clock/sleep/camera/detector as arguments, so it is
tested with scripted frames — no camera, no mediapipe required.

## Notes

* `eye_closure` is computed from the eye-aspect-ratio relative to the
  calibration baseline; clamp to [0, 1].
* Glasses can throw off MediaPipe landmarks — the pilot should test the
  glasses case explicitly (Section 5.4).
* Pilot labelling video (if any) is recorded separately, not by this program
  (Section 5.4).
