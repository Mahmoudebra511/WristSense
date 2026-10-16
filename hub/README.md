# PC hub (Section 5.2)

Receives IMU packets from the wrist band over BLE, stamps every sample with
the unified PC clock, writes `imu.csv` + `meta.json` per the
[session data contract](../docs/schema.md), reconnects automatically on BLE
drops, and detects the *three sharp shakes* sync marker.

## Hardware it pairs with

ESP32-S3-N16R8 + MPU-6050 (I2C) running the band firmware. The firmware must
emit the versioned binary packet defined in `hub/packet.py`:

| field | size | meaning |
|---|---|---|
| magic `AAC\x01` | 4 bytes | packet start + self-describing version |
| device_ts_ms | u64 | band's own monotonic ms |
| seq_start | u32 | sequence number of first sample in the batch |
| sample_count | u16 | samples in this packet |
| samples | count × 28 bytes | seq u32 + ax..gz f32 (g, dps) |

MAX30102 is unused in v1 (out of scope).

## Install

```bash
pip install -r requirements-hub.txt   # bleak for real hardware
```

The hub runs without bleak using `--simulate`.

## Run

```bash
# no hardware: synthetic IMU, one minute
python -m hub.run --simulate --duration 60 --participant p01 --wrist left

# inject fake shake spikes every 8 s (exercises the sync-marker detector)
python -m hub.run --simulate --duration 60 --shake-every 8

# real band
python -m hub.run --address AA:BB:CC:DD:EE:FF --duration 60

# attach to a session the logger already created
python -m hub.run --session-dir data/raw/p01_20261020T141530_laptop --duration 60
```

Ctrl-C finalises `meta.json` (streams, row count, measured rate, sync
markers, PC clock) and prints a validation report.

## What the hub writes

`imu.csv` — exactly the columns in `aacdata/columns.py` (ts_ms, t_dev_ms,
ax..gz, seq). `ts_ms` is stamped by the PC on arrival; `t_dev_ms` is the
band's own time, kept for clock-drift diagnosis.

`meta.json` — gains `streams.imu`, `sampling_rate_hz.imu` (**measured**, not
assumed), `row_counts.imu`, `sync_markers_ms`, `end_utc`, `duration_s`, `pc`.

## Sync marker detection

Three acceleration-magnitude spikes (> 2.5 g, gravity included) within 2 s
emit one marker at the PC-clock time of the first spike, followed by a 5 s
cooldown. The band is expected to physically do "three sharp shakes" at the
start/end of a labelled block (Section 5.2 of the project document).

## Architecture

| file | responsibility |
|---|---|
| `packet.py` | packet encode/decode, BLE sources (bleak + synthetic), sync detector, Hub |
| `writer.py` | `imu.csv` sink |
| `run.py` | CLI |

`Hub.stream()` is async and takes its clock as an argument, so it can be
driven with simulated time in tests.
