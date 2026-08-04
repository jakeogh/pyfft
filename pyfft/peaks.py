#!/usr/bin/env python3
# tab-width:4

"""
Peak and harmonic detection on a Spectrum.

Detection runs on the dB spectrum: vectorized local-maximum candidates,
then a dominance test over the min-distance neighborhood. Interior peaks
are refined by parabolic interpolation of the dB values for sub-bin
frequency and amplitude estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spectrum import Spectrum

HARMONIC_RECENTER_BINS = 3


@dataclass(frozen=True)
class Peak:
    index: int          # bin index within the analyzed Spectrum
    frequency: float    # parabolic-refined when interior
    amplitude: float
    db: float
    snr_db: float       # relative to the median floor of the analyzed band
    phase: float | None


@dataclass(frozen=True)
class Harmonic:
    order: int
    index: int
    frequency: float
    amplitude: float
    db: float
    snr_db: float


def parabolic_refine(db: np.ndarray, i: int, frequencies: np.ndarray, binwidth: float) -> tuple[float, float]:
    """Parabolic interpolation around interior bin i: (frequency, db)."""
    if i <= 0 or i >= db.size - 1:
        return float(frequencies[i]), float(db[i])
    a, b, c = float(db[i - 1]), float(db[i]), float(db[i + 1])
    denom = a - 2.0 * b + c    # strictly negative at a strict local maximum
    if denom >= 0.0:
        return float(frequencies[i]), b
    delta = 0.5 * (a - c) / denom
    return float(frequencies[i]) + delta * binwidth, b - 0.25 * (a - c) * delta


def _floor_db(spec: Spectrum) -> float:
    start = 1 if spec.bin0 == 0 else 0
    if len(spec) <= start:
        raise ValueError("empty spectrum")
    return float(np.median(spec.db[start:]))


def _make_peak(spec: Spectrum, i: int, floor: float, refine: bool) -> Peak:
    if refine:
        freq, db = parabolic_refine(spec.db, i, spec.frequencies, spec.binwidth)
    else:
        freq, db = float(spec.frequencies[i]), float(spec.db[i])
    return Peak(
        index=i,
        frequency=freq,
        amplitude=10.0 ** (db / 20.0),
        db=db,
        snr_db=db - floor,
        phase=float(spec.phase[i]) if spec.phase is not None else None,
    )


def find_peaks(
    spec: Spectrum,
    *,
    count: int = 10,
    min_distance_hz: float | None = None,
    threshold_db: float | None = None,
    refine: bool = True,
) -> list[Peak]:
    """
    Strongest local maxima above threshold, sorted by level descending.

    threshold_db defaults to the median floor + 10 dB. min_distance_hz
    defaults to max(10 bins, 1% of the top frequency) so a leakage skirt
    is not reported as multiple peaks. The DC bin never qualifies.
    """
    db = spec.db
    if db.size < 3:
        return []
    floor = _floor_db(spec)
    if threshold_db is None:
        threshold_db = floor + 10.0
    bw = spec.binwidth
    if min_distance_hz is None:
        min_distance_hz = max(10.0 * bw, 0.01 * float(spec.frequencies[-1]))
    bin_distance = max(1, int(min_distance_hz / bw))

    interior = np.nonzero(
        (db[1:-1] > db[:-2]) & (db[1:-1] > db[2:]) & (db[1:-1] > threshold_db)
    )[0] + 1

    hits: list[int] = []
    for i in interior:
        lo = max(1 if spec.bin0 == 0 else 0, i - bin_distance)
        hi = min(db.size, i + bin_distance + 1)
        if db[i] == np.max(db[lo:hi]):
            hits.append(int(i))
    hits.sort(key=lambda i: db[i], reverse=True)
    return [_make_peak(spec, i, floor, refine) for i in hits[:count]]


def fold_frequency(frequency: float, samplerate: float) -> float:
    """Alias a frequency into the first Nyquist zone [0, samplerate/2]."""
    f = math.fmod(frequency, samplerate)
    if f < 0.0:
        f += samplerate
    return samplerate - f if f > samplerate / 2.0 else f


def find_harmonics(
    spec: Spectrum,
    fundamental: float | None = None,
    *,
    max_order: int = 10,
    search_bins: int | None = None,
    min_snr_db: float = 6.0,
    fold: bool = False,
) -> list[Harmonic]:
    """
    Detected harmonics of the fundamental (the strongest peak when None).

    Each harmonic is the local maximum within +-search_bins of the ideal
    h * f0 bin, reported when it clears the floor by min_snr_db. With
    fold=True harmonics beyond Nyquist are aliased back in-band instead
    of ending the scan.
    """
    if fundamental is None:
        top = find_peaks(spec, count=1)
        if not top:
            raise ValueError("no fundamental peak found above the floor")
        fundamental = top[0].frequency
    if fundamental <= 0.0:
        raise ValueError(f"fundamental must be positive, got {fundamental}")
    floor = _floor_db(spec)
    if search_bins is None:
        search_bins = max(HARMONIC_RECENTER_BINS, spec.window.lobe_bins)
    f = spec.frequencies
    db = spec.db
    bw = spec.binwidth
    out: list[Harmonic] = []
    for h in range(2, max_order + 1):
        target = fundamental * h
        if fold:
            target = fold_frequency(target, spec.samplerate)
        elif target > f[-1]:
            break
        center = round(target / bw) - spec.bin0
        if center < 0 or center >= f.size:
            continue
        lo = max(1 if spec.bin0 == 0 else 0, center - search_bins)
        hi = min(f.size, center + search_bins + 1)
        i = lo + int(np.argmax(db[lo:hi]))
        if db[i] - floor < min_snr_db:
            continue
        freq, level = parabolic_refine(db, i, f, bw)
        out.append(Harmonic(
            order=h,
            index=i,
            frequency=freq,
            amplitude=10.0 ** (level / 20.0),
            db=level,
            snr_db=level - floor,
        ))
    return out
