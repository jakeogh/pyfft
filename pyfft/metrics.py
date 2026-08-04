#!/usr/bin/env python3
# tab-width:4

"""
ADC characterization metrics.

analyze_tone performs standard single-tone FFT analysis: component powers
are integrated over window main lobes on a Parseval-consistent scale
(sum(power)/enbw_bins), harmonics are folded through Nyquist, and SNR /
THD / SINAD / ENOB / SFDR are reported together with the per-component
peaks. noise_floor summarizes a band statistically for noise work, and
waveform_stats covers the time domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .peaks import HARMONIC_RECENTER_BINS
from .peaks import Harmonic
from .peaks import Peak
from .peaks import parabolic_refine
from .peaks import fold_frequency
from .spectrum import Spectrum


@dataclass(frozen=True)
class WaveformStats:
    n: int
    duration: float
    mean: float
    rms: float
    ac_rms: float
    min: float
    max: float
    peak_to_peak: float


def waveform_stats(y: np.ndarray, samplerate: float) -> WaveformStats:
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"expected a non-empty 1-D record, got shape {y.shape}")
    if samplerate <= 0.0:
        raise ValueError(f"samplerate must be positive, got {samplerate}")
    lo = float(np.min(y))
    hi = float(np.max(y))
    return WaveformStats(
        n=y.size,
        duration=y.size / samplerate,
        mean=float(np.mean(y)),
        rms=float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))),
        ac_rms=float(np.std(y)),
        min=lo,
        max=hi,
        peak_to_peak=hi - lo,
    )


def codes_to_volts(codes: np.ndarray, *, bits: int, full_scale: float) -> np.ndarray:
    """Convert raw ADC codes to volts: codes * full_scale / 2**bits."""
    return np.asarray(codes, dtype=np.float64) * (full_scale / (1 << bits))


@dataclass(frozen=True)
class NoiseFloor:
    median_db: float
    mean_db: float
    std_db: float
    min_db: float
    max_db: float
    peak_to_peak_db: float
    rms: float              # integrated band rms
    asd_median: float       # median amplitude spectral density, V/sqrt(Hz)


def noise_floor(spec: Spectrum, low: float | None = None, high: float | None = None) -> NoiseFloor:
    """Statistical noise-floor summary over [low, high), DC bin excluded."""
    band = spec[low:high]
    start = 1 if band.bin0 == 0 else 0
    if len(band) <= start:
        raise ValueError(f"no bins in band [{low}, {high})")
    db = band.db[start:]
    return NoiseFloor(
        median_db=float(np.median(db)),
        mean_db=float(np.mean(db)),
        std_db=float(np.std(db)),
        min_db=float(np.min(db)),
        max_db=float(np.max(db)),
        peak_to_peak_db=float(np.ptp(db)),
        rms=float(np.sqrt(np.sum(band.power[start:]) / band.window.enbw_bins)),
        asd_median=float(np.median(band.asd[start:])),
    )


@dataclass(frozen=True)
class ToneAnalysis:
    fundamental: Peak
    harmonics: tuple[Harmonic, ...]
    worst_spur: Peak | None
    signal_rms: float
    noise_rms: float
    distortion_rms: float
    snr_db: float
    thd_db: float
    thd_percent: float
    sinad_db: float
    enob: float
    sfdr_db: float
    noise_floor_db: float


def _region_ms(power: np.ndarray, mask: np.ndarray, enbw_bins: float) -> float:
    return float(np.sum(power[mask]) / enbw_bins)


def analyze_tone(
    spec: Spectrum,
    *,
    fundamental: float | None = None,
    max_order: int = 10,
    lobe_bins: int | None = None,
) -> ToneAnalysis:
    """
    Single-tone ADC analysis of a spectrum.

    The fundamental is the strongest bin outside the DC lobe unless given.
    Each component claims +-lobe_bins (default: the window main lobe) and
    the residual is noise. Harmonics 2..max_order are folded through
    Nyquist. SFDR is in dBc against the largest non-fundamental bin,
    harmonics included.
    """
    n = len(spec)
    if n < 8:
        raise ValueError(f"spectrum too short for tone analysis: {n} bins")
    lobe = spec.window.lobe_bins if lobe_bins is None else lobe_bins
    if lobe < 1:
        raise ValueError(f"lobe_bins must be >= 1, got {lobe}")
    power = spec.power
    db = spec.db
    f = spec.frequencies
    bw = spec.binwidth
    enbw_bins = spec.window.enbw_bins
    bins_abs = spec.bin0 + np.arange(n)

    dc_mask = bins_abs <= lobe
    if not np.any(~dc_mask):
        raise ValueError("spectrum is entirely inside the DC lobe")

    if fundamental is None:
        i0 = int(np.argmax(np.where(dc_mask, -np.inf, spec.amplitude)))
    else:
        i0 = int(np.argmin(np.abs(f - fundamental)))
    f0, fund_db = parabolic_refine(db, i0, f, bw)
    if f0 <= 0.0:
        raise ValueError(f"fundamental resolves to {f0} Hz")
    fund_mask = np.abs(np.arange(n) - i0) <= lobe

    claimed = dc_mask | fund_mask
    harmonics: list[Harmonic] = []
    harm_mask = np.zeros(n, dtype=bool)
    for h in range(2, max_order + 1):
        fh = fold_frequency(f0 * h, spec.samplerate)
        center = round(fh / bw) - spec.bin0
        if center < 0 or center >= n:
            continue
        lo = max(0, center - HARMONIC_RECENTER_BINS)
        hi = min(n, center + HARMONIC_RECENTER_BINS + 1)
        i = lo + int(np.argmax(db[lo:hi]))
        this = (np.abs(np.arange(n) - i) <= lobe) & ~claimed
        hfreq, hdb = parabolic_refine(db, i, f, bw)
        harmonics.append(Harmonic(
            order=h,
            index=i,
            frequency=hfreq,
            amplitude=10.0 ** (hdb / 20.0),
            db=hdb,
            snr_db=math.nan,    # filled after the noise floor is known
        ))
        harm_mask |= this
        claimed |= this

    noise_mask = ~claimed
    if not np.any(noise_mask):
        raise ValueError("no noise bins left after masking DC, fundamental, and harmonics")

    signal_ms = _region_ms(power, fund_mask & ~dc_mask, enbw_bins)
    dist_ms = _region_ms(power, harm_mask, enbw_bins)
    noise_ms = _region_ms(power, noise_mask, enbw_bins)
    if signal_ms <= 0.0:
        raise ValueError("fundamental has no power; not a tone record")
    if noise_ms <= 0.0:
        raise ValueError("noise region has zero power; degenerate spectrum")

    floor_db = float(np.median(db[noise_mask]))
    harmonics = [
        Harmonic(h.order, h.index, h.frequency, h.amplitude, h.db, h.db - floor_db)
        for h in harmonics
    ]

    snr_db = 10.0 * math.log10(signal_ms / noise_ms)
    sinad_db = 10.0 * math.log10(signal_ms / (noise_ms + dist_ms))
    thd_db = 10.0 * math.log10(dist_ms / signal_ms) if dist_ms > 0.0 else -math.inf
    thd_percent = 100.0 * math.sqrt(dist_ms / signal_ms)
    enob = (sinad_db - 1.76) / 6.02

    spur_db_arr = np.where(fund_mask | dc_mask, -np.inf, db)
    ispur = int(np.argmax(spur_db_arr))
    if np.isfinite(spur_db_arr[ispur]):
        sfreq, sdb = parabolic_refine(db, ispur, f, bw)
        worst_spur = Peak(
            index=ispur,
            frequency=sfreq,
            amplitude=10.0 ** (sdb / 20.0),
            db=sdb,
            snr_db=sdb - floor_db,
            phase=float(spec.phase[ispur]) if spec.phase is not None else None,
        )
        sfdr_db = fund_db - sdb
    else:
        worst_spur = None
        sfdr_db = math.inf

    fund_peak = Peak(
        index=i0,
        frequency=f0,
        amplitude=10.0 ** (fund_db / 20.0),
        db=fund_db,
        snr_db=fund_db - floor_db,
        phase=float(spec.phase[i0]) if spec.phase is not None else None,
    )
    return ToneAnalysis(
        fundamental=fund_peak,
        harmonics=tuple(harmonics),
        worst_spur=worst_spur,
        signal_rms=math.sqrt(signal_ms),
        noise_rms=math.sqrt(noise_ms),
        distortion_rms=math.sqrt(dist_ms),
        snr_db=snr_db,
        thd_db=thd_db,
        thd_percent=thd_percent,
        sinad_db=sinad_db,
        enob=enob,
        sfdr_db=sfdr_db,
        noise_floor_db=floor_db,
    )
