# AAC — Adaptive Attention Companion

A system that decides, in real time, whether a student is still **on-task**
while studying, nudges them when distraction persists, and reports focus time
and distractions at the end of a session.

Three data sources cover each other's blind spots:

* a **wrist band** (IMU) — what the hand is doing
* a **laptop logger** — what is on screen and whether there is input
* a **webcam** — presence and coarse engagement

The project document (`AAC — Project Document`) is the source of truth for the
design. This repository is the implementation, built in the phases it defines.

## Repository layout

```
aac/
├── aacdata/     # the session data contract (this repo's shared schema code)
├── schema/      # JSON Schema for meta.json
├── scripts/     # CLI tools (generate example, validate sessions)
├── tests/       # contract tests
├── docs/        # schema spec, on-task definition, pilot protocol
├── firmware/    # band code            (phase 1)
├── hub/         # BLE receive + session writing   (phase 1)
├── logger/      # laptop logger        (phase 1)
├── vision/      # webcam pipeline      (phase 1)
├── ml/          # features, models, evaluation, replay   (phase 3-4)
├── dashboard/   # session report       (phase 5)
├── examples/    # synthetic sessions for testing
└── data/        # recordings (kept out of git)
```

## Status

| Phase (doc §6) | State |
|---|---|
| 0. Setup — repo, schema, frozen on-task definition | **in progress** (schema done) |
| 1. Data-collection tools | not started |
| 2. Pilot | not started |
| 3. Baseline ML | not started |
| 4. Fusion and state machine | not started |
| 5. Actions and dashboard | not started |
| 6. Wider evaluation | not started |

## The session data contract

A recording session is one folder:

```
data/raw/p01_20261020T141530_laptop/
├── meta.json     # describes the session (required)
├── imu.csv       # wrist band, >= 50 Hz (required)
├── logger.csv    # laptop context, 1 Hz (optional)
├── vision.csv    # webcam numbers, 1 Hz (optional)
└── labels.csv    # human ground truth (optional until labelled)
```

* Full specification: [`docs/schema.md`](docs/schema.md)
* Debugging aid: [`docs/schema_diagram.html`](docs/schema_diagram.html)
* Code that defines it: [`aacdata/columns.py`](aacdata/columns.py),
  [`aacdata/enums.py`](aacdata/enums.py)
* JSON Schema: [`schema/meta.schema.json`](schema/meta.schema.json)
* Frozen **on-task definition**: [`docs/on_task_definition.md`](docs/on_task_definition.md)

## Quick start (no dependencies)

The schema tooling is pure standard library, so it runs on a bare Python
install, before the ML stack exists.

```bash
# generate a synthetic example session under examples/
python scripts/make_example_session.py --broken

# validate it
python scripts/validate_session.py examples/p00_20261016T090000_laptop --require-labels

# validate everything
python scripts/validate_session.py examples

# run the contract tests
python -m unittest discover -s tests -v
```

Exit code `0` means no errors; warnings never fail a run.

Using the schema from code:

```python
from aacdata.validate import validate_session
from aacdata.session import Session

report = validate_session("data/raw/p01_20261020T141530_laptop", require_labels=True)
print(report.info["imu_rate_hz"])   # measured, not assumed

sess = Session.create("data/raw", "p02", wrist="right", mode="laptop")
```

## Conventions

* **One clock.** Everything is stamped with the PC clock as epoch
  milliseconds (UTC); the band's own time is kept alongside.
* **Measure, don't assume.** The configured sampling rate and the measured one
  are separate values; the validator reports the measured rate and the largest
  gap.
* **Change the contract in one place.** Edit `aacdata/`, bump `SCHEMA_VERSION`,
  update `docs/schema.md`.
* **Privacy by construction.** Only counts, categories and angles are stored —
  never keystrokes, window titles, or video.
