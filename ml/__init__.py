"""AAC ML package — feature pipeline, baseline model, early fusion, replay,
evaluation, and state-machine tuning (project document, Section 5.5).

This is Mahmoud's part: the longest part because it is the core of the project.

Pipeline contract (deterministic, one command per step):

    python -m ml.pipeline data/raw/p01_... --out features.csv
    python -m ml.fusion data/processed/ --out fused.csv
    python -m ml.baseline data/processed/ --cv --model models/baseline.joblib
    python -m ml.replay --fused fused.csv --model models/baseline.joblib
    python -m ml.evaluate data/processed/ --fused fused.csv --cv
    python -m ml.state_machine_tune data/processed/ --fused fused.csv
"""

from . import windowing, features, labels, pipeline, baseline, fusion, state_machine, replay, evaluate, state_machine_tune

__all__ = [
    "windowing",
    "features",
    "labels",
    "pipeline",
    "baseline",
    "fusion",
    "state_machine",
    "replay",
    "evaluate",
    "state_machine_tune",
]
