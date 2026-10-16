"""AAC PC hub — BLE receive, timestamp, session writing, sync detection.

Receives versioned IMU packets from the wrist band (ESP32-S3 + MPU-6050),
stamps every sample with the unified PC clock, writes ``imu.csv`` and
``meta.json`` per the :mod:`aacdata` session contract, reconnects
automatically on BLE drops, and detects the ``three sharp shakes`` sync
marker (Section 5.2 of the project document).

The BLE transport is swappable: :class:`BleakPacketSource` talks to real
hardware (needs ``bleak``); :class:`SyntheticPacketSource` generates
deterministic fake packets for testing and demos.
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import socket
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Protocol

from aacdata.columns import IMU
from aacdata.constants import SCHEMA_VERSION, ON_TASK_DEFINITION_VERSION
from aacdata.models import Device, Meta, PC
from aacdata.session import Session, iso_utc
from aacdata.validate import validate_session
from hub.writer import ImuCsvWriter

# ---------------------------------------------------------------------------
# Versioned binary packet:  magic | version | device_ts_ms | seq_start | count
# then `count` IMU samples (seq + 6 floats), all little-endian.
# ---------------------------------------------------------------------------
MAGIC = b"AAC\x01"  # 4 bytes: literal + version byte, so version is self-describing
HEADER_STRUCT = struct.Struct("<4sQIH")  # magic, device_ts_ms, seq_start, sample_count
SAMPLE_STRUCT = struct.Struct("<I6f")     # seq, ax, ay, az, gx, gy, gz
HEADER_SIZE = HEADER_STRUCT.size          # 4+8+4+2 = 18
SAMPLE_SIZE = SAMPLE_STRUCT.size          # 4+24 = 28


@dataclass(frozen=True)
class ImuSample:
    seq: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class Packet:
    version: int
    device_ts_ms: int
    seq_start: int
    samples: list[ImuSample] = field(default_factory=list)

    def encode(self) -> bytes:
        """Encode to bytes (firmware-side helper; the PC hub only decodes)."""
        buf = bytearray()
        buf += HEADER_STRUCT.pack(MAGIC, self.device_ts_ms, self.seq_start, len(self.samples))
        for s in self.samples:
            buf += SAMPLE_STRUCT.pack(s.seq, s.ax, s.ay, s.az, s.gx, s.gy, s.gz)
        return bytes(buf)

    @staticmethod
    def decode(buffer: bytes) -> "Packet | None":
        """Decode one packet from the front of a byte buffer.

        Returns ``None`` if the buffer holds fewer than one full packet or the
        magic/version is wrong. Caller removes the consumed bytes.
        """
        if len(buffer) < HEADER_SIZE:
            return None
        magic, device_ts_ms, seq_start, count = HEADER_STRUCT.unpack_from(buffer, 0)
        if magic != MAGIC:
            return None  # resynchronise: caller drops one byte and retries
        needed = HEADER_SIZE + count * SAMPLE_SIZE
        if len(buffer) < needed:
            return None
        samples = []
        off = HEADER_SIZE
        for _ in range(count):
            seq, ax, ay, az, gx, gy, gz = SAMPLE_STRUCT.unpack_from(buffer, off)
            samples.append(ImuSample(seq, ax, ay, az, gx, gy, gz))
            off += SAMPLE_SIZE
        return Packet(version=MAGIC[3], device_ts_ms=device_ts_ms, seq_start=seq_start, samples=samples)


# ---------------------------------------------------------------------------
# Packet source abstraction
# ---------------------------------------------------------------------------
class PacketSource(Protocol):
    """Async iterator of :class:`Packet` from some transport."""

    async def __aiter__(self) -> AsyncIterator[Packet]:
        ...


class SyntheticPacketSource:
    """Deterministic fake IMU stream. Useful without hardware and in tests.

    Emits 50 Hz packets of ``batch`` samples. Every ``shake_every_s`` seconds
    it inserts a sharp acceleration spike; three spikes within ``window_ms``
    form a sync marker (exercised by :class:`SyncMarkerDetector`).
    """

    def __init__(
        self,
        rate_hz: float = 50.0,
        batch: int = 10,
        shake_every_s: float | None = None,
        seed: int = 0,
    ):
        self.rate_hz = rate_hz
        self.batch = batch
        self.shake_every_s = shake_every_s
        self._rng = __import__("random").Random(seed)
        self._seq = 0
        self._device_t = 0.0
        self._wall_t = 0.0
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def __aiter__(self) -> AsyncIterator[Packet]:
        interval = self.batch / self.rate_hz
        next_shake = self.shake_every_s if self.shake_every_s else float("inf")
        while self._running:
            samples = []
            for _ in range(self.batch):
                # mild 1 Hz hand rhythm on Z, gravity on Y, small noise
                rhythm = math.sin(2 * math.pi * 1.0 * self._wall_t)
                ax = self._rng.gauss(0.0, 0.02)
                ay = 1.0 + 0.04 * rhythm + self._rng.gauss(0.0, 0.02)
                az = 0.1 * rhythm + self._rng.gauss(0.0, 0.02)
                gx = 2.0 * rhythm + self._rng.gauss(0.0, 0.5)
                gy = self._rng.gauss(0.0, 0.5)
                gz = self._rng.gauss(0.0, 0.5)
                if self.shake_every_s and self._wall_t >= next_shake:
                    ax += 3.0  # sharp shake spike
                    next_shake += self.shake_every_s
                samples.append(
                    ImuSample(
                        seq=self._seq, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz
                    )
                )
                self._seq += 1
                self._device_t += 1000.0 / self.rate_hz
                self._wall_t += 1.0 / self.rate_hz
            yield Packet(
                version=1,
                device_ts_ms=int(self._device_t),
                seq_start=samples[0].seq if samples else self._seq,
                samples=samples,
            )
            await asyncio.sleep(interval)


class BleakPacketSource:
    """Real BLE source via bleak. Notifications are bridged through a queue."""

    def __init__(self, address: str, char_uuid: str, service_uuid: str | None = None):
        self.address = address
        self.char_uuid = char_uuid
        self.service_uuid = service_uuid
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def __aiter__(self) -> AsyncIterator[Packet]:
        try:
            from bleak import BleakClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "bleak is not installed; use SyntheticPacketSource or pip install bleak"
            ) from exc

        buffer = bytearray()

        def handle_notification(_sender: int, data: bytearray) -> None:
            # BLE callbacks run on the bleak loop; put raw bytes on the queue.
            self._queue.put_nowait(bytes(data))

        async with BleakClient(self.address) as client:
            if not client.is_connected:
                await client.connect()
            await client.start_notify(self.char_uuid, handle_notification)
            try:
                while True:
                    chunk = await self._queue.get()
                    if chunk is None:  # sentinel -> stop
                        return
                    buffer += chunk
                    while True:
                        pkt = Packet.decode(buffer)
                        if pkt is None:
                            if len(buffer) > HEADER_SIZE + 64 * SAMPLE_SIZE:
                                # likely garbage; resync by dropping one byte
                                del buffer[0]
                            break
                        yield pkt
                        del buffer[: HEADER_SIZE + len(pkt.samples) * SAMPLE_SIZE]
            finally:
                await client.stop_notify(self.char_uuid)

    def stop(self) -> None:
        self._queue.put_nowait(None)


# ---------------------------------------------------------------------------
# Sync marker detection: three sharp shakes within a window
# ---------------------------------------------------------------------------
class SyncMarkerDetector:
    """Detects 'three sharp shakes' in the IMU stream.

    A shake is an accelerometer magnitude spike beyond ``spike_g`` (gravity
    included, so ~1 g at rest). Three shakes within ``window_ms`` emit one
    sync marker at the PC-clock time of the first shake, followed by a
    ``cooldown_ms`` quiet period to avoid double-counting.
    """

    def __init__(
        self,
        spike_g: float = 2.5,
        window_ms: float = 2000.0,
        cooldown_ms: float = 5000.0,
        clock=time.time,
    ):
        self.spike_g = spike_g
        self.window_ms = window_ms
        self.cooldown_ms = cooldown_ms
        self._clock = clock
        self._spikes: list[float] = []  # PC-clock epoch seconds
        self._cooldown_until = 0.0
        self.markers_ms: list[int] = []

    def observe(self, sample: ImuSample, ts_ms: int) -> int | None:
        """Feed one sample. Returns the marker PC-clock ms if one was emitted."""
        now_s = self._clock()
        if now_s < self._cooldown_until:
            return None
        mag = math.sqrt(sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az)
        if mag > self.spike_g:
            self._spikes.append(now_s)
        cutoff = now_s - self.window_ms / 1000.0
        self._spikes = [t for t in self._spikes if t >= cutoff]
        if len(self._spikes) >= 3:
            marker_ms = int(self._spikes[0] * 1000)
            self.markers_ms.append(marker_ms)
            self._spikes = []
            self._cooldown_until = now_s + self.cooldown_ms / 1000.0
            return marker_ms
        return None


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------
class Hub:
    def __init__(
        self,
        session: Session,
        source: PacketSource,
        writer: ImuCsvWriter | None = None,
        clock=time.time,
        sync_detector: SyncMarkerDetector | None = None,
        sample_period_ms: float = 20.0,
    ):
        self.session = session
        self.source = source
        self.writer = writer or ImuCsvWriter(session.path / IMU.filename)
        self.clock = clock
        self.sync_detector = sync_detector or SyncMarkerDetector(clock=clock)
        self.sample_period_ms = sample_period_ms
        self._start_wall: float | None = None
        self.rows = 0
        self._stop = False
        # PC<->device clock anchor, set on the first sample:
        #   ts_ms = pc_base_ms + (device_sample_ms - dev_base_ms)
        self._pc_base_ms: int | None = None
        self._dev_base_ms: int | None = None

    def stop(self) -> None:
        self._stop = True
        if hasattr(self.source, "stop"):
            self.source.stop()

    async def stream(self, duration_s: float | None = None) -> None:
        """Main loop: consume packets, stamp, write, detect sync markers."""
        self._start_wall = self.clock()
        start = self._start_wall
        async for packet in self.source:
            if self._stop:
                break
            for sample in packet.samples:
                device_sample_ms = packet.device_ts_ms + (sample.seq - packet.seq_start) * self.sample_period_ms
                if self._pc_base_ms is None:
                    self._pc_base_ms = int(self.clock() * 1000)
                    self._dev_base_ms = device_sample_ms
                ts_ms = self._pc_base_ms + int(device_sample_ms - self._dev_base_ms)
                self.writer.write(
                    {
                        "ts_ms": ts_ms,
                        "t_dev_ms": device_sample_ms,
                        "ax": sample.ax,
                        "ay": sample.ay,
                        "az": sample.az,
                        "gx": sample.gx,
                        "gy": sample.gy,
                        "gz": sample.gz,
                        "seq": sample.seq,
                    }
                )
                self.rows += 1
                self.sync_detector.observe(sample, ts_ms)
            if duration_s is not None and self.clock() - start >= duration_s:
                break

    def finalize(self) -> Meta:
        """Close the writer and write/finalize ``meta.json``."""
        self.writer.close()
        meta = self.session.load_meta()
        now = datetime.now(timezone.utc)
        meta.streams = {**meta.streams, "imu": IMU.filename}
        # measured rate from the validator's point of view: rows / elapsed wall time
        elapsed = max(1e-6, self.clock() - (self._start_wall or now.timestamp()))
        meta.sampling_rate_hz = {**meta.sampling_rate_hz, "imu": round(self.rows / elapsed, 3)}
        meta.row_counts = {**meta.row_counts, "imu": self.rows}
        meta.sync_markers_ms = self.sync_detector.markers_ms
        meta.end_utc = iso_utc(now)
        meta.duration_s = max(0.0, (now - datetime.fromtimestamp(self._start_wall or now.timestamp(), timezone.utc)).total_seconds())
        meta.pc = PC(os=platform.system(), host=socket.gethostname())
        meta.partial = "imu" not in meta.streams
        self.session.write_meta(meta)
        return meta

    def report(self):
        return validate_session(self.session.path)
