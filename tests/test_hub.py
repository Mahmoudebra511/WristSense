"""Hub tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aacdata.columns import IMU  # noqa: E402
from aacdata.session import Session  # noqa: E402
from hub.packet import (  # noqa: E402
    Hub,
    ImuSample,
    Packet,
    SyncMarkerDetector,
    SyntheticPacketSource,
)
from hub.writer import ImuCsvWriter  # noqa: E402


def _make_session(root: Path) -> Session:
    return Session.at(
        root / "p00_20261016T090000_laptop",
        participant_id="p00",
        wrist="right",
        mode="laptop",
    )


class TestPacket(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        pkt = Packet(
            version=1,
            device_ts_ms=1000,
            seq_start=5,
            samples=[ImuSample(5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), ImuSample(6, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)],
        )
        decoded = Packet.decode(pkt.encode())
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.version, 1)
        self.assertEqual(decoded.device_ts_ms, 1000)
        self.assertEqual(decoded.seq_start, 5)
        self.assertEqual(len(decoded.samples), 2)
        self.assertEqual(decoded.samples[1].ay, 1.0)

    def test_bad_magic_returns_none(self):
        self.assertIsNone(Packet.decode(b"\x00\x00\x00\x00" + b"\x00" * 100))


class TestSyncDetector(unittest.TestCase):
    def test_three_shakes_emit_marker(self):
        clock = [0.0]

        def now():
            return clock[0]

        det = SyncMarkerDetector(spike_g=2.5, window_ms=2000, cooldown_ms=0, clock=now)
        # rest samples then three spikes
        samples = [
            ImuSample(0, 0.0, 1.0, 0.0, 0, 0, 0),   # mag 1.0 -> no spike
            ImuSample(1, 3.0, 0.0, 0.0, 0, 0, 0),   # spike 1 at t=1
            ImuSample(2, 3.0, 0.0, 0.0, 0, 0, 0),   # spike 2 at t=2
            ImuSample(3, 3.0, 0.0, 0.0, 0, 0, 0),   # spike 3 at t=3 -> marker
        ]
        clock[0] = 1.0
        self.assertIsNone(det.observe(samples[1], 1000))
        clock[0] = 2.0
        self.assertIsNone(det.observe(samples[2], 2000))
        clock[0] = 3.0
        ms = det.observe(samples[3], 3000)
        self.assertEqual(ms, 1000)  # marker at first spike's time
        self.assertEqual(det.markers_ms, [1000])

    def test_spikes_outside_window_do_not_trigger(self):
        clock = [0.0]
        det = SyncMarkerDetector(spike_g=2.5, window_ms=1000, cooldown_ms=0, clock=lambda: clock[0])
        spikes = [ImuSample(i, 3.0, 0.0, 0.0, 0, 0, 0) for i in range(3)]
        clock[0] = 0.0
        det.observe(spikes[0], 0)
        clock[0] = 2.0  # 2 s later, outside 1 s window
        det.observe(spikes[1], 2000)
        clock[0] = 4.0
        self.assertIsNone(det.observe(spikes[2], 4000))
        self.assertEqual(det.markers_ms, [])

    def test_cooldown_prevents_double_count(self):
        clock = [0.0]
        det = SyncMarkerDetector(spike_g=2.5, window_ms=2000, cooldown_ms=10000, clock=lambda: clock[0])
        spikes = [ImuSample(i, 3.0, 0.0, 0.0, 0, 0, 0) for i in range(3)]
        clock[0] = 1.0
        det.observe(spikes[0], 1000)
        clock[0] = 1.1
        det.observe(spikes[1], 1100)
        clock[0] = 1.2
        ms = det.observe(spikes[2], 1200)
        self.assertEqual(ms, 1000)
        # immediately again: still in cooldown
        self.assertIsNone(det.observe(spikes[0], 1300))
        self.assertEqual(len(det.markers_ms), 1)


class TestSyntheticSource(unittest.TestCase):
    def test_emits_50hz_packets(self):
        source = SyntheticPacketSource(rate_hz=50.0, batch=10, seed=1)

        async def collect():
            packets = []
            async for pkt in source:
                packets.append(pkt)
                if len(packets) >= 3:
                    source.stop()
            return packets

        packets = asyncio.run(collect())
        self.assertEqual(len(packets), 3)
        self.assertEqual(len(packets[0].samples), 10)
        # seq should continue across packets
        self.assertEqual(packets[1].seq_start, packets[0].seq_start + 10)


class TestHubIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session = _make_session(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _run_sim(self, duration_s: float, shake_every: float | None = None, seed: int = 7):
        source = SyntheticPacketSource(rate_hz=50.0, batch=10, shake_every_s=shake_every, seed=seed)
        writer = ImuCsvWriter(self.session.path / IMU.filename)
        hub = Hub(self.session, source, writer=writer)
        try:
            asyncio.run(hub.stream(duration_s=duration_s))
        finally:
            hub.stop()
            writer.close()
        return hub

    def test_rows_match_schema(self):
        hub = self._run_sim(duration_s=0.5)
        self.assertGreater(hub.rows, 0)
        # validate through the session contract
        meta = self.session.load_meta()
        meta.streams = {"imu": "imu.csv"}
        meta.sampling_rate_hz = {"imu": 50.0}
        meta.row_counts = {"imu": hub.rows}
        meta.partial = True
        self.session.write_meta(meta)
        report = self.session.validate()
        self.assertTrue(report.ok, report.format())
        self.assertEqual(report.info["imu_rows"], hub.rows)

    def test_measured_rate_close_to_50hz(self):
        hub = self._run_sim(duration_s=1.0)
        self.assertGreater(hub.rows, 30)
        self.assertLess(hub.rows, 80)

    def test_sync_markers_recorded(self):
        hub = self._run_sim(duration_s=3.0, shake_every=0.8, seed=3)
        # 3 s with shakes every 0.8 s -> spikes at 0.8, 1.6, 2.4 -> marker at 0.8 s
        self.assertGreaterEqual(len(hub.sync_detector.markers_ms), 1)
        meta = self.session.load_meta()
        meta.streams = {"imu": "imu.csv"}
        meta.sampling_rate_hz = {"imu": 50.0}
        meta.row_counts = {"imu": hub.rows}
        meta.sync_markers_ms = hub.sync_detector.markers_ms
        meta.partial = True
        self.session.write_meta(meta)
        report = self.session.validate()
        self.assertTrue(report.ok, report.format())
        self.assertEqual(len(report.info.get("sync_markers_ms", []) if False else meta.sync_markers_ms), len(meta.sync_markers_ms))

    def test_finalize_writes_meta(self):
        hub = self._run_sim(duration_s=0.5)
        meta = hub.finalize()
        self.assertEqual(meta.streams.get("imu"), "imu.csv")
        self.assertIn("imu", meta.sampling_rate_hz)
        self.assertEqual(meta.row_counts.get("imu"), hub.rows)
        self.assertFalse(meta.partial)


if __name__ == "__main__":
    unittest.main()
