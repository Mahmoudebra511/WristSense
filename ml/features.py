"""Feature computation per IMU window (project document, Section 5.5).

Features computed on a single sliding window of IMU samples:

1. RMS of acceleration after gravity removal
2. RMS of gyroscope
3. Mean and standard deviation of gyroscope magnitude
4. Dominant frequency and spectral entropy (FFT on accel magnitude)
5. Energy in frequency bands: 0.5–3 Hz and 3–8 Hz
6. Zero-crossing rate
7. Gravity direction / hand tilt (mean of the low-frequency accel vector)
8. Peak count and maximum jerk (for page-turn detection)
9. Stillness duration (fraction of samples near-static)

Requires numpy (declared in requirements-ml.txt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY_G = 9.80665  # m/s^2, but IMU is already in g; we work in g units.
STILLNESS_THRESHOLD_G = 0.05  # |accel_no_gravity| below this => still


@dataclass(frozen=True)
class FeatureVector:
    """Named feature values for one window."""

    rms_accel_nogravity: float
    rms_gyro: float
    gyro_mag_mean: float
    gyro_mag_std: float
    dominant_freq_hz: float
    spectral_entropy: float
    band_energy_0_5_3_hz: float
    band_energy_3_8_hz: float
    zero_crossing_rate: float
    gravity_x: float
    gravity_y: float
    gravity_z: float
    peak_count: int
    max_jerk: float
    stillness_duration_s: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}rms_accel_nogravity": self.rms_accel_nogravity,
            f"{prefix}rms_gyro": self.rms_gyro,
            f"{prefix}gyro_mag_mean": self.gyro_mag_mean,
            f"{prefix}gyro_mag_std": self.gyro_mag_std,
            f"{prefix}dominant_freq_hz": self.dominant_freq_hz,
            f"{prefix}spectral_entropy": self.spectral_entropy,
            f"{prefix}band_energy_0_5_3_hz": self.band_energy_0_5_3_hz,
            f"{prefix}band_energy_3_8_hz": self.band_energy_3_8_hz,
            f"{prefix}zero_crossing_rate": self.zero_crossing_rate,
            f"{prefix}gravity_x": self.gravity_x,
            f"{prefix}gravity_y": self.gravity_y,
            f"{prefix}gravity_z": self.gravity_z,
            f"{prefix}peak_count": float(self.peak_count),
            f"{prefix}max_jerk": self.max_jerk,
            f"{prefix}stillness_duration_s": self.stillness_duration_s,
        }

    @classmethod
    def header(cls, prefix: str = "") -> list[str]:
        return list(cls("", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0).as_dict(prefix).keys())


def _remove_gravity(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate gravity as the mean acceleration vector and subtract it.

    Returns ``(accel_no_gravity, gravity_vector)``. This is a simple
    low-frequency estimate: over a 4 s window the mean is dominated by
    gravity plus slow hand tilt.
    """
    gravity = accel.mean(axis=0)
    return accel - gravity, gravity


def _jerk(accel: np.ndarray, dt: float) -> np.ndarray:
    """Jerk = derivative of acceleration magnitude."""
    mag = np.linalg.norm(accel, axis=1)
    return np.diff(mag) / dt


def compute_features(
    accel: np.ndarray,
    gyro: np.ndarray,
    ts_ms: np.ndarray,
    sample_rate_hz: float = 50.0,
) -> FeatureVector:
    """Compute all features for one window.

    Parameters
    ----------
    accel:
        Array shape ``(n, 3)`` in g (gravity included).
    gyro:
        Array shape ``(n, 3)`` in degrees per second.
    ts_ms:
        Array shape ``(n,)`` of timestamps in ms.
    sample_rate_hz:
        Nominal sampling rate (Hz).
    """
    n = accel.shape[0]
    dt = 1.0 / sample_rate_hz

    # --- 1. RMS of acceleration after gravity removal ----------------------
    accel_no_gravity, gravity = _remove_gravity(accel)
    rms_accel_nogravity = float(np.sqrt(np.mean(accel_no_gravity ** 2)))

    # --- 2. RMS of gyroscope ----------------------------------------------
    rms_gyro = float(np.sqrt(np.mean(gyro ** 2)))

    # --- 3. Mean and std of gyroscope magnitude ---------------------------
    gyro_mag = np.linalg.norm(gyro, axis=1)
    gyro_mag_mean = float(np.mean(gyro_mag))
    gyro_mag_std = float(np.std(gyro_mag))

    # --- 4-5. FFT-based features on accel magnitude -----------------------
    accel_mag = np.linalg.norm(accel_no_gravity, axis=1)
    # Detrend and apply a light window to reduce leakage.
    accel_mag = accel_mag - accel_mag.mean()
    spectrum = np.abs(np.fft.rfft(accel_mag))
    freqs = np.fft.rfftfreq(n, d=dt)

    # Avoid the DC bin for dominant frequency.
    mag_no_dc = spectrum.copy()
    if len(mag_no_dc) > 1:
        mag_no_dc[0] = 0.0
    dominant_freq_hz = float(freqs[int(np.argmax(mag_no_dc))]) if len(freqs) > 1 else 0.0

    # Spectral entropy: treat normalised power spectrum as a probability dist.
    power = spectrum ** 2
    total_power = power.sum()
    if total_power > 0:
        prob = power / total_power
        prob = prob[prob > 0]
        spectral_entropy = float(-np.sum(prob * np.log(prob)))
    else:
        spectral_entropy = 0.0

    # --- 5. Band energy (0.5–3 Hz and 3–8 Hz) -----------------------------
    def band_energy(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            return 0.0
        band_power = power[mask].sum()
        return float(band_power / total_power) if total_power > 0 else 0.0

    band_0_5_3 = band_energy(0.5, 3.0)
    band_3_8 = band_energy(3.0, 8.0)

    # --- 6. Zero-crossing rate --------------------------------------------
    signs = np.sign(accel_mag)
    signs[signs == 0] = 1.0
    zero_crossing_rate = float(np.sum(np.diff(signs) != 0) / max(n - 1, 1))

    # --- 7. Gravity direction (hand tilt) ---------------------------------
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm > 0:
        gravity_dir = gravity / gravity_norm
    else:
        gravity_dir = np.array([0.0, 0.0, 1.0])
    gravity_x, gravity_y, gravity_z = (float(v) for v in gravity_dir)

    # --- 8. Peak count and max jerk ---------------------------------------
    # Peak count: local maxima in acceleration magnitude.
    if n >= 3:
        is_peak = (accel_mag[1:-1] > accel_mag[:-2]) & (accel_mag[1:-1] > accel_mag[2:])
        peak_count = int(is_peak.sum())
    else:
        peak_count = 0

    jerk = _jerk(accel, dt)
    max_jerk = float(np.max(np.abs(jerk))) if jerk.size else 0.0

    # --- 9. Stillness duration --------------------------------------------
    still_mask = np.linalg.norm(accel_no_gravity, axis=1) < STILLNESS_THRESHOLD_G
    stillness_duration_s = float(still_mask.sum() * dt)

    return FeatureVector(
        rms_accel_nogravity=rms_accel_nogravity,
        rms_gyro=rms_gyro,
        gyro_mag_mean=gyro_mag_mean,
        gyro_mag_std=gyro_mag_std,
        dominant_freq_hz=dominant_freq_hz,
        spectral_entropy=spectral_entropy,
        band_energy_0_5_3_hz=band_0_5_3,
        band_energy_3_8_hz=band_3_8,
        zero_crossing_rate=zero_crossing_rate,
        gravity_x=gravity_x,
        gravity_y=gravity_y,
        gravity_z=gravity_z,
        peak_count=peak_count,
        max_jerk=max_jerk,
        stillness_duration_s=stillness_duration_s,
    )
