#!/usr/bin/env python3
# tab-width:4

"""
Window - a precomputed FFT window with its scaling constants.

Every spectral quantity in this library is corrected by these constants:
coherent gain (sum) for amplitudes, ENBW for densities, and the main-lobe
half-width for tone-power integration and harmonic masking.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft
import scipy.signal

WindowSpec = str | float | tuple | np.ndarray

_LOBE_REF_LEN = 4096    # main-lobe width in bins is length-invariant; measure on a capped copy
_LOBE_PAD = 16


def _lobe_halfwidth_bins(samples: np.ndarray) -> int:
    """
    Half-width of the window's spectral main lobe, in bins: the first local
    minimum of |W| at least 6 dB below the peak, so flat-top passband
    ripple does not read as a null.
    """
    ref = samples if len(samples) <= _LOBE_REF_LEN else samples[:: len(samples) // _LOBE_REF_LEN]
    mag = np.abs(scipy.fft.rfft(ref, _LOBE_PAD * len(ref)))
    deep = (mag[1:-1] < mag[:-2]) & (mag[1:-1] <= mag[2:]) & (mag[1:-1] < 0.5 * mag[0])
    minima = np.nonzero(deep)[0] + 1
    first_null = int(minima[0]) if minima.size else len(mag) - 1
    return max(1, math.ceil(first_null / _LOBE_PAD))


class Window:
    """
    Precomputed window samples plus the derived constants:

    sum            coherent gain * n, the amplitude normalization divisor
    sumsq          sum of squares, for noise-power normalization
    coherent_gain  sum / n
    enbw_bins      equivalent noise bandwidth in bins, n * sumsq / sum**2
    lobe_bins      main-lobe half-width in bins (first spectral null)
    """

    __slots__ = ("samples", "label", "n", "sum", "sumsq", "coherent_gain",
                 "enbw_bins", "lobe_bins")

    def __init__(self, samples: np.ndarray, label: str) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or samples.size < 2:
            raise ValueError(f"window must be a 1-D array of at least 2 samples, got shape {samples.shape}")
        self.samples = samples
        self.label = label
        self.n = samples.size
        self.sum = float(np.sum(samples))
        self.sumsq = float(np.sum(samples * samples))
        if self.sum == 0.0:
            raise ValueError("window sums to zero, amplitude normalization impossible")
        self.coherent_gain = self.sum / self.n
        self.enbw_bins = self.n * self.sumsq / (self.sum * self.sum)
        self.lobe_bins = _lobe_halfwidth_bins(samples)

    @classmethod
    def create(cls, spec: WindowSpec, n: int) -> Window:
        """Build a periodic (DFT-even) window of length n from a scipy window spec."""
        if isinstance(spec, np.ndarray):
            if spec.size != n:
                raise ValueError(f"window array length {spec.size} != nfft {n}")
            return cls(spec, "custom")
        samples = scipy.signal.get_window(spec, n, fftbins=True)
        label = spec if isinstance(spec, str) else repr(spec)
        return cls(samples, label)

    def enbw_hz(self, samplerate: float) -> float:
        """Equivalent noise bandwidth in Hz."""
        return self.enbw_bins * samplerate / self.n

    def __repr__(self) -> str:
        return (f"Window({self.label!r}, n={self.n}, enbw_bins={self.enbw_bins:.4f}, "
                f"lobe_bins={self.lobe_bins})")
