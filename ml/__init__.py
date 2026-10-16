"""AAC ML package — feature pipeline, baseline model, early fusion, and replay
(project document, Section 5.5).

This is Mahmoud's part: the longest part because it is the core of the project.

Pipeline contract (deterministic, one command per step):

    python -m ml.pipeline data/raw/p01_... --out features.csv
    python -m ml.fusion data/processed/ --out fused.csv
    python -m ml.baseline data/processed/ --cv --model models/baseline.joblib
    python -m ml.replay --fused fused.csv --model models/baseline.joblib
"""

from . import windowing, features, labels, pipeline, baseline, fusion, state_machine, replay

__all__ = [
    "windowing",
    "features",
    "labels",
    "pipeline",
    "baseline",
    "fusion",
    "state_machine",
    "replay",
]
