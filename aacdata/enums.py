"""Controlled vocabularies for the AAC project.

Every string enum the system writes to disk is declared here exactly once.
Readers should compare against these members, never against raw literals.
"""

from __future__ import annotations

from enum import Enum


class Wrist(str, Enum):
    """Which wrist the band is worn on. Fixed per session and recorded."""

    LEFT = "left"
    RIGHT = "right"


class Mode(str, Enum):
    """Operating mode chosen by the user at session start (Section 2)."""

    LAPTOP = "laptop"
    PAPER = "paper"


class TaskState(str, Enum):
    """Ground-truth state written by a human labeler in ``labels.csv``.

    ``UNCERTAIN`` is deliberately *not* here: it is produced by the state
    machine at runtime, not observed by the labeler. See ``RuntimeState``.
    """

    ON_TASK = "ON_TASK"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    BREAK = "BREAK"


class RuntimeState(str, Enum):
    """The state set the state machine can emit (adds UNCERTAIN)."""

    ON_TASK = "ON_TASK"
    UNCERTAIN = "UNCERTAIN"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    BREAK = "BREAK"


class AppCategory(str, Enum):
    """Coarse class of the active laptop window. No titles/content stored."""

    STUDY = "study"
    ENTERTAINMENT = "entertainment"
    SOCIAL = "social"
    OTHER = "other"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Activity vocabulary (labels.csv -> activity column)
# ---------------------------------------------------------------------------
# PROVISIONAL. The exact set of 11 activities is frozen in the pilot protocol
# document (docs/pilot_protocol.md). Until then the validator treats an unknown
# activity as a *warning*, not an error, so early data is never rejected.
PROVISIONAL_ACTIVITIES: tuple[str, ...] = (
    # productive / on-task
    "reading_paper",
    "writing_notes",
    "page_turn",
    "typing",
    "coding",
    "watching_course",
    # off-task
    "phone_use",
    "entertainment_screen",
    "social_screen",
    # presence / body
    "walking",
    "away",
    "idle_still",
    "drinking",
    "talking",
    # fallback
    "other",
)

# Streams that are always present in every session.
REQUIRED_STREAM_NAMES: tuple[str, ...] = ("imu",)

# Streams that may be absent depending on mode / phase but are known today.
OPTIONAL_STREAM_NAMES: tuple[str, ...] = ("logger", "vision", "phone")
