#!/usr/bin/env python3
# tab-width:4

"""
Spectrum computation for real-valued ADC records.

compute_spectrum  one FFT of the whole record, phase preserved
average_spectrum  Welch-style reduction over overlapping segments

average_spectrum runs as batched 2-D rffts over zero-copy strided segment
views, multithreaded inside pocketfft (workers), with a bounded scratch
buffer so arbitrarily long records stream in constant memory. float32 input
stays float32 end to end; accumulation is float64.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.fft
import scipy.signal
from numpy.lib.stride_tricks import sliding_window_view

from .spectrum import Spectrum
from .windows import Window
from .windows import WindowSpec

next_fast_len = scipy.fft.next_fast_len

Detrend = Literal["none", "mean", "linear"]
AverageMode = Literal["power", "amplitude", "max", "min"]

DEFAULT_MAX_BATCH_BYTES = 1 << 28


def fft_frequencies(nfft: int, samplerate: float) -> np.ndarray:
    """Bin-center frequencies of a real FFT: k * samplerate / nfft."""
    return np.arange(nfft // 2 + 1) * (samplerate / nfft)


def _as_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"expected a 1-D record, got shape {y.shape}")
    if y.dtype == np.float32 or y.dtype == np.float64:
        return y
    return y.astype(np.float64)


def _detrended(y: np.ndarray, detrend: Detrend) -> np.ndarray:
    if detrend == "none":
        return y
    if detrend == "mean":
        return y - y.mean()
    if detrend == "linear":
        return scipy.signal.detrend(y, type="linear")
    raise ValueError(f"unknown detrend {detrend!r}")


def _single_sided(mag: np.ndarray, wsum: float, nfft: int) -> np.ndarray:
    mag *= 2.0 / wsum
    mag[0] *= 0.5
    if nfft % 2 == 0:
        mag[-1] *= 0.5
    return mag


def compute_spectrum(
    y: np.ndarray,
    samplerate: float,
    *,
    window: WindowSpec = "hann",
    detrend: Detrend = "mean",
    nfft: int | None = None,
    workers: int = -1,
) -> Spectrum:
    """
    Amplitude spectrum of one record, with phase (degrees).

    nfft > len(y) zero-pads after windowing (frequency interpolation);
    nfft < len(y) is an error, use average_spectrum to reduce a long record.
    """
    if samplerate <= 0.0:
        raise ValueError(f"samplerate must be positive, got {samplerate}")
    y = _as_float(y)
    n = y.size
    if nfft is None:
        nfft = n
    if nfft < n:
        raise ValueError(f"nfft {nfft} < record length {n}; use average_spectrum to reduce")
    win = Window.create(window, n)
    w = win.samples.astype(y.dtype, copy=False)
    yw = _detrended(y, detrend) * w
    x = scipy.fft.rfft(yw, n=nfft, workers=workers, overwrite_x=True)
    amplitude = _single_sided(np.abs(x), win.sum, nfft)
    phase = np.angle(x, deg=True)
    return Spectrum(
        amplitude=amplitude,
        samplerate=float(samplerate),
        nfft=nfft,
        window=win,
        phase=phase,
    )


def average_spectrum(
    y: np.ndarray,
    samplerate: float,
    nfft: int,
    *,
    overlap: float = 0.75,
    step: int | None = None,
    window: WindowSpec = "hann",
    detrend: Detrend = "mean",
    mode: AverageMode = "power",
    workers: int = -1,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> Spectrum:
    """
    Reduce a long record to one spectrum over overlapping nfft-point segments.

    mode "power"      rms average (Welch), the default for noise work
         "amplitude"  linear magnitude average
         "max"/"min"  peak/valley hold across segments

    step overrides overlap when given. Phase is not defined for a reduction
    and is None on the result.
    """
    if samplerate <= 0.0:
        raise ValueError(f"samplerate must be positive, got {samplerate}")
    y = _as_float(y)
    if nfft < 2:
        raise ValueError(f"nfft must be >= 2, got {nfft}")
    if y.size < nfft:
        raise ValueError(f"record length {y.size} < nfft {nfft}")
    if step is None:
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        step = max(1, round(nfft * (1.0 - overlap)))
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    win = Window.create(window, nfft)
    w = win.samples.astype(y.dtype, copy=False)
    segments = sliding_window_view(y, nfft)[::step]
    nseg = segments.shape[0]

    nbins = nfft // 2 + 1
    if mode == "power" or mode == "amplitude":
        acc = np.zeros(nbins, dtype=np.float64)
    elif mode == "max":
        acc = np.full(nbins, -np.inf, dtype=np.float64)
    elif mode == "min":
        acc = np.full(nbins, np.inf, dtype=np.float64)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    rows = max(1, min(nseg, max_batch_bytes // (nfft * y.dtype.itemsize * 6)))
    buf = np.empty((rows, nfft), dtype=y.dtype)
    for b0 in range(0, nseg, rows):
        batch = segments[b0:b0 + rows]
        k = batch.shape[0]
        scratch = buf[:k]
        if detrend == "mean":
            np.subtract(batch, batch.mean(axis=1, keepdims=True), out=scratch)
        elif detrend == "linear":
            scratch[...] = scipy.signal.detrend(batch, axis=-1, type="linear")
        elif detrend == "none":
            scratch[...] = batch
        else:
            raise ValueError(f"unknown detrend {detrend!r}")
        scratch *= w
        x = scipy.fft.rfft(scratch, axis=1, workers=workers, overwrite_x=True)
        mag = np.abs(x)
        if mode == "power":
            np.square(mag, out=mag)
            acc += mag.sum(axis=0, dtype=np.float64)
        elif mode == "amplitude":
            acc += mag.sum(axis=0, dtype=np.float64)
        elif mode == "max":
            np.maximum(acc, mag.max(axis=0), out=acc)
        else:
            np.minimum(acc, mag.min(axis=0), out=acc)

    if mode == "power":
        amplitude = np.sqrt(acc / nseg)
    elif mode == "amplitude":
        amplitude = acc / nseg
    else:
        amplitude = acc
    amplitude = _single_sided(amplitude, win.sum, nfft)
    return Spectrum(
        amplitude=amplitude,
        samplerate=float(samplerate),
        nfft=nfft,
        window=win,
        averages=nseg,
    )
