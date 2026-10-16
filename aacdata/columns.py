"""Column-level specification of every file in a session folder.

This is the authoritative description of the raw on-disk layout. The hub,
the logger, the vision program, the feature pipeline and the replay tool all
import from here so that "training preprocessing" and "runtime preprocessing"
refer to the same columns.

Units are fixed on purpose:

* acceleration in g (1 g = 9.80665 m/s^2), gravity included
* angular rate in degrees per second (dps)
* time in integer milliseconds since the Unix epoch, UTC, on the *unified PC
  clock* (see docs/schema.md, "Clocks and time").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # "int" | "float" | "bool" | "str"
    unit: str = ""
    required: bool = True
    nullable: bool = False  # may be empty/NaN (e.g. no face detected)
    description: str = ""


@dataclass(frozen=True)
class Stream:
    name: str
    filename: str
    rate_hz: float | None  # nominal sampling rate, None for event tables
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
TS_MS = Column(
    "ts_ms",
    "int",
    unit="ms",
    description="Sample time on the unified PC clock (epoch ms, UTC).",
)

T_DEV_MS = Column(
    "t_dev_ms",
    "int",
    unit="ms",
    description=(
        "Device-reported monotonic time in ms. Kept next to ts_ms so a lost "
        "BLE packet or a clock drift can be diagnosed later."
    ),
)


# ---------------------------------------------------------------------------
# imu.csv — the wrist band, target >= 50 Hz
# ---------------------------------------------------------------------------
IMU = Stream(
    name="imu",
    filename="imu.csv",
    rate_hz=50.0,
    columns=(
        TS_MS,
        T_DEV_MS,
        Column("ax", "float", unit="g", description="Accelerometer X (gravity included)."),
        Column("ay", "float", unit="g", description="Accelerometer Y (gravity included)."),
        Column("az", "float", unit="g", description="Accelerometer Z (gravity included)."),
        Column("gx", "float", unit="dps", description="Gyroscope X (deg/s)."),
        Column("gy", "float", unit="dps", description="Gyroscope Y (deg/s)."),
        Column("gz", "float", unit="dps", description="Gyroscope Z (deg/s)."),
        Column(
            "seq",
            "int",
            required=False,
            nullable=True,
            description="Device sample counter; lets us detect dropped samples.",
        ),
    ),
)


# ---------------------------------------------------------------------------
# logger.csv — laptop context, 1 Hz
# ---------------------------------------------------------------------------
LOGGER = Stream(
    name="logger",
    filename="logger.csv",
    rate_hz=1.0,
    columns=(
        TS_MS,
        Column("keys", "int", description="Keystrokes counted in this second (no characters)."),
        Column("mouse_moves", "int", description="Mouse-move events in this second."),
        Column("mouse_clicks", "int", description="Mouse clicks in this second."),
        Column("win_switches", "int", description="Active-window changes in this second."),
        Column(
            "category",
            "str",
            description="AppCategory of the active window (study/entertainment/social/other/unknown).",
        ),
        Column(
            "time_since_input_s",
            "float",
            unit="s",
            description="Seconds since the last keyboard/mouse input (0 if there was input this second).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# vision.csv — webcam, 1 Hz, numbers only (no video)
# ---------------------------------------------------------------------------
VISION = Stream(
    name="vision",
    filename="vision.csv",
    rate_hz=1.0,
    columns=(
        TS_MS,
        Column("face_present", "bool", description="1 if a face was detected in this second."),
        Column("head_yaw", "float", unit="deg", nullable=True,
               description="Head rotation about the vertical axis; NaN when no face."),
        Column("head_pitch", "float", unit="deg", nullable=True,
               description="Head rotation about the lateral axis; NaN when no face."),
        Column("eye_closure", "float", nullable=True,
               description="0 = open, 1 = closed; NaN when no face."),
        Column("conf", "float", required=False, nullable=True,
               description="Detector confidence 0..1, optional."),
    ),
)


# ---------------------------------------------------------------------------
# phone.csv — deferred (Section 5.8). Declared now so the schema is stable.
# ---------------------------------------------------------------------------
PHONE = Stream(
    name="phone",
    filename="phone.csv",
    rate_hz=1.0,
    columns=(
        TS_MS,
        Column("screen_on", "bool", description="1 if the phone screen is on."),
        Column("app_category", "str",
               description="Coarse app category; no app names or content."),
    ),
)


# ---------------------------------------------------------------------------
# labels.csv — human annotation intervals (event table, not periodic)
# ---------------------------------------------------------------------------
LABELS = Stream(
    name="labels",
    filename="labels.csv",
    rate_hz=None,
    columns=(
        Column("start_ms", "int", unit="ms", description="Interval start on the unified PC clock."),
        Column("end_ms", "int", unit="ms", description="Interval end (exclusive) on the PC clock."),
        Column("activity", "str", description="Activity label from the pilot vocabulary."),
        Column("task_state", "str", description="Ground-truth TaskState (never UNCERTAIN)."),
        Column("labeler", "str", required=False, nullable=True, description="Who annotated it."),
        Column("notes", "str", required=False, nullable=True, description="Free-form note."),
    ),
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
STREAMS: dict[str, Stream] = {
    "imu": IMU,
    "logger": LOGGER,
    "vision": VISION,
    "phone": PHONE,
}

# name -> Stream for every file we know how to read, including the event table.
ALL_FILES: dict[str, Stream] = {**STREAMS, "labels": LABELS}
