"""AAC vision pipeline — webcam face/head/eye analysis.

Produces ``vision.csv`` (one row per second) per the aacdata session contract.
"""

from .pipeline import (
    CalibrationResult,
    Detector,
    FrameResult,
    NullDetector,
    SyntheticCamera,
    SyntheticDetector,
    VisionCsvWriter,
    VisionPipeline,
    clamp,
)
from .detector import MediaPipeFaceDetector, OpenCVCamera

__all__ = [
    "CalibrationResult",
    "Detector",
    "FrameResult",
    "NullDetector",
    "SyntheticCamera",
    "SyntheticDetector",
    "VisionCsvWriter",
    "VisionPipeline",
    "MediaPipeFaceDetector",
    "OpenCVCamera",
    "clamp",
]
