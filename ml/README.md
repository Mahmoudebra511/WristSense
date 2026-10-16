# ML feature pipeline + baseline model + early fusion + replay (Section 5.5 — Mahmoud's part)

Deterministic, one-command tools for feature extraction, baseline modeling,
sensor fusion, and state-machine replay from recorded sessions.

## Module layout

| file | responsibility |
|---|---|
| `windowing.py` | sliding windows (4 s, 50% overlap) + decision windows |
| `features.py` | 15 features per IMU window |
| `labels.py` | label alignment + transition window dropping |
| `pipeline.py` | one-command runner: read → window → feature → label → write |
| `baseline.py` | Random Forest / LightGBM, leave-one-session-out evaluation |
| `fusion.py` | early fusion table: one row per 30 s decision window |
| `state_machine.py` | on-task state machine with hysteresis timers |
| `replay.py` | replay tool: run state machine on recorded/synthetic sessions |

## 1. Feature pipeline

```bash
python -m ml.pipeline data/raw/p01_20261020T141530_laptop --out features.csv
```

Produces one row per kept IMU window with 15 features plus `task_state` and
`activity`.

## 2. Baseline model

```bash
python -m ml.baseline data/processed/ --cv
python -m ml.baseline data/processed/ --model models/baseline.joblib
```

Model classes: `ON_TASK`, `DISTRACTED`, `AWAY`. Evaluation: leave-one-session-out,
macro-F1, accuracy, per-class precision/recall/F1, confusion matrix.

## 3. Early fusion

```bash
python -m ml.fusion data/processed/ --out fused.csv --window-s 30
```

One row per 30-second decision window with `imu_*`, `logger_*`, `vision_*`
aggregates plus `task_state`.

## 4. State machine + replay (NEW)

The state machine implements the project document's decision hierarchy
(presence → context → phone → engagement → time) with hysteresis timers:

| parameter | default | meaning |
|---|---|---|
| `grace_s` | 90 | stillness up to this stays ON_TASK |
| `confirm_s` | 20 | new signal must persist this long before switching |
| `alert_delay_s` | 120 | sustained distraction must last this long before alert |
| `away_threshold_s` | 30 | away from desk longer than this → AWAY |

```bash
# replay a fused session with a trained model
python -m ml.replay --fused fused.csv --model models/baseline.joblib --print-states

# replay a scripted synthetic session (no model needed)
python -m ml.replay --synthetic --duration 600 --window-s 30 --seed 1

# tune parameters and compare false alerts / hour
python -m ml.replay --fused fused.csv --grace-s 60 --confirm-s 15 --alert-delay-s 90 --out replay.json
```

Outputs per-window state timeline, state transitions, alerts, false-alerts-per-hour,
detection delay, and per-state agreement vs ground truth.

## 5. Requirements

```
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0          # optional, only with --lightgbm
joblib
```

Declared in `requirements-ml.txt`. Everything else in AAC is pure standard
library.

## Next steps (not yet implemented)

* `evaluate.py` — false alerts/hour, detection delay, ablation study
* `state_machine_tune.py` — automated parameter sweep for the state machine
