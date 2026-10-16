# Laptop logger (Section 5.3)

Writes one row per second to `logger.csv`, describing laptop activity:

| column | what it is |
|---|---|
| `keys` | keystrokes in this second — **counted, never recorded** |
| `mouse_moves` | mouse-move events in this second |
| `mouse_clicks` | clicks in this second |
| `win_switches` | active-window changes in this second |
| `category` | `study` / `entertainment` / `social` / `other` / `unknown` |
| `time_since_input_s` | seconds since the last keyboard/mouse input |

Only the **category** is stored. Window titles are read locally to decide the
category and are never written to disk. Characters are never recorded.

## Install

```bash
pip install pynput
```

`pynput` is the only dependency. If it is missing, the logger still runs and
logs the active window, but the input columns stay zero.

## Run

```bash
# a new logger-only session under data/raw/
python -m logger.run --participant p01 --mode laptop --duration 120

# attach to a session the hub already created
python -m logger.run --session-dir data/raw/p01_20261020T141530_laptop

# no OS access: synthetic activity, good for a quick look
python -m logger.run --simulate --duration 10 --print

# until Ctrl-C
python -m logger.run --participant p01
```

At the end the logger finalises `meta.json` (declares the `logger` stream,
row count, sampling rate, the config hash) and validates the session, printing
a report.

> **Tip:** if you run the logger from a terminal, that terminal is the
> foreground window and will be categorised. Run it minimised, or add its
> process to `exclude_processes` so it is ignored (the previous category is
> kept).

## Configure categories

Edit `logger/config.json`. Rules are case-insensitive regular expressions:

```json
{
  "poll_hz": 4,
  "default_category": "other",
  "categories": {
    "study":         { "processes": ["code\\.exe"], "titles": ["overleaf", "arxiv"] },
    "entertainment": { "processes": ["vlc\\.exe"],  "titles": ["youtube", "netflix"] },
    "social":        { "processes": ["discord\\.exe"], "titles": ["telegram"] }
  },
  "exclude_processes": []
}
```

Matching order (so results are predictable):

1. every category's `processes`, in the order declared in the file;
2. every category's `titles`, in the same order;
3. `default_category` if nothing matched.

A **process match always beats a title match**, so `code.exe` stays `study`
even if its window title mentions something else. If the process is in
`exclude_processes`, the sample is ignored and the previous category is kept.

The SHA-256 of the config file is stored in `meta.json → app_whitelist_hash`,
so the exact rules in force for a session are traceable.

## How it fits the schema

* Writes `logger.csv` exactly as `aacdata/columns.py` defines it.
* Updates `meta.json`: `streams.logger`, `sampling_rate_hz.logger`,
  `row_counts.logger`, `app_whitelist_hash`, `end_utc`, `duration_s` and `pc`.
* A logger-only capture is marked `partial: true` in `meta.json`, so the
  validator does not demand an IMU stream that the hub has not produced yet.
  Once the hub adds `imu.csv` and declares it, the session stops being partial.

## Architecture

| file | responsibility |
|---|---|
| `config.py` | load rules, build the `Categorizer` |
| `active_window.py` | read the foreground window (Windows via ctypes) |
| `input_tracker.py` | keyboard/mouse counters (pynput; null and synthetic fallbacks) |
| `recorder.py` | the 1 Hz aggregation loop (`Recorder.tick`) |
| `writer.py` | `logger.csv` sink, plus an in-memory sink for tests |
| `run.py` | the CLI |

`Recorder` takes its clock, sleep and sources as arguments, so the loop is
tested with a fake clock and scripted windows — no OS, no waiting.
