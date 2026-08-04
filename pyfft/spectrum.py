#!/usr/bin/env python3
# tab-width:4

"""
Spectrum - single-sided spectrum of a real signal on a uniform bin grid.

The stored canonical quantity is peak amplitude per bin, window-corrected
(2|X|/sum(w), DC and Nyquist not doubled). Everything else - dB, rms, power,
PSD, ASD - derives lazily from it. bin0 tracks the absolute rfft bin of the
first element so slices keep exact DC/Nyquist identification and the
frequency grid stays derived, never stored.

Frequency-range selection:
    spec[low:high]   band [low, high) in Hz, bounds sorted, either side open
    spec[frequency]  SpectrumPoint at the closest bin
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .windows import Window

DB_FLOOR = 1e-20


class SpectrumPoint(NamedTuple):
    frequency: float
    amplitude: float
    db: float
    phase: float | None


@dataclass(frozen=True, repr=False)
class Spectrum:
    amplitude: np.ndarray
    samplerate: float
    nfft: int
    window: Window
    phase: np.ndarray | None = None     # degrees
    bin0: int = 0
    averages: int = 1

    @property
    def binwidth(self) -> float:
        return self.samplerate / self.nfft

    @property
    def enbw_hz(self) -> float:
        return self.window.enbw_bins * self.binwidth

    @cached_property
    def frequencies(self) -> np.ndarray:
        return (self.bin0 + np.arange(self.amplitude.size)) * self.binwidth

    @cached_property
    def db(self) -> np.ndarray:
        """20*log10(amplitude): dBV peak for volt-scaled input."""
        return 20.0 * np.log10(np.maximum(self.amplitude, DB_FLOOR))

    @cached_property
    def rms(self) -> np.ndarray:
        r = self.amplitude * (0.5 ** 0.5)
        if self.bin0 == 0 and r.size:
            r[0] = self.amplitude[0]
        if self.nfft % 2 == 0 and self.bin0 + r.size == self.nfft // 2 + 1 and r.size:
            r[-1] = self.amplitude[-1]
        return r

    @cached_property
    def power(self) -> np.ndarray:
        """Mean-square per bin (Vrms^2)."""
        return self.rms * self.rms

    @cached_property
    def psd(self) -> np.ndarray:
        """Power spectral density, V^2/Hz."""
        return self.power / self.enbw_hz

    @cached_property
    def asd(self) -> np.ndarray:
        """Amplitude spectral density, V/sqrt(Hz) - datasheet noise density."""
        return np.sqrt(self.psd)

    def __len__(self) -> int:
        return self.amplitude.size

    def _slice(self, i0: int, i1: int) -> Spectrum:
        return replace(
            self,
            amplitude=self.amplitude[i0:i1],
            phase=self.phase[i0:i1] if self.phase is not None else None,
            bin0=self.bin0 + i0,
        )

    def _band_indices(self, low: float | None, high: float | None) -> tuple[int, int]:
        if low is not None and high is not None and low > high:
            low, high = high, low
        f = self.frequencies
        i0 = 0 if low is None else int(np.searchsorted(f, low, side="left"))
        i1 = f.size if high is None else int(np.searchsorted(f, high, side="left"))
        return i0, i1

    def __getitem__(self, arg: slice | float | int) -> Spectrum | SpectrumPoint:
        if isinstance(arg, slice):
            if arg.step is not None:
                raise ValueError("frequency slices take no step")
            return self._slice(*self._band_indices(arg.start, arg.stop))
        if isinstance(arg, (float, int)):
            return self.closest(float(arg))
        raise TypeError(f"Spectrum[] selects a band [low:high] or the closest bin [frequency], not {arg!r}")

    def closest(self, frequency: float) -> SpectrumPoint:
        f = self.frequencies
        if f.size == 0:
            raise ValueError("empty spectrum")
        i = int(np.searchsorted(f, frequency))
        if i == f.size or (i > 0 and frequency - f[i - 1] < f[i] - frequency):
            i -= 1
        return SpectrumPoint(
            float(f[i]),
            float(self.amplitude[i]),
            float(self.db[i]),
            float(self.phase[i]) if self.phase is not None else None,
        )

    def dominant(self, low: float | None = None, high: float | None = None) -> SpectrumPoint:
        """Largest-amplitude bin, optionally restricted to [low, high)."""
        band = self[low:high]
        if len(band) == 0:
            raise ValueError(f"no bins in band [{low}, {high})")
        i = int(np.argmax(band.amplitude))
        return SpectrumPoint(
            float(band.frequencies[i]),
            float(band.amplitude[i]),
            float(band.db[i]),
            float(band.phase[i]) if band.phase is not None else None,
        )

    def dominant_frequency(self, low: float | None = None, high: float | None = None) -> float:
        return self.dominant(low, high).frequency

    def band_rms(self, low: float | None = None, high: float | None = None) -> float:
        """
        Integrated rms over [low, high): sqrt(sum(psd) * binwidth).
        Parseval-consistent for both noise and coherent tones.
        """
        band = self[low:high]
        if len(band) == 0:
            raise ValueError(f"no bins in band [{low}, {high})")
        return float(np.sqrt(np.sum(band.power) / self.window.enbw_bins))

    def noise_floor_db(self) -> float:
        """Median bin level in dB, DC bin excluded."""
        start = 1 if self.bin0 == 0 else 0
        if self.amplitude.size <= start:
            raise ValueError("empty spectrum")
        return float(np.median(self.db[start:]))

    def cut_dc(self, return_index: bool = False) -> Spectrum | int:
        """
        Cut the DC leakage skirt: drop every bin before the first local
        minimum of the amplitude. Returns the cut Spectrum, or the cut
        index with return_index=True.
        """
        a = self.amplitude
        rising = np.nonzero(np.diff(a) > 0)[0]
        below = rising[a[rising] < a[0]] if rising.size else rising    # skirt must have descended
        idx = int(below[0]) if below.size else 0
        if return_index:
            return idx
        return self._slice(idx, a.size)

    def save(self, path: str | Path) -> None:
        """Write the spectrum to a compressed .npz."""
        arrays: dict[str, np.ndarray | np.generic] = {
            "amplitude": self.amplitude,
            "samplerate": np.float64(self.samplerate),
            "nfft": np.int64(self.nfft),
            "bin0": np.int64(self.bin0),
            "averages": np.int64(self.averages),
            "window_samples": self.window.samples,
            "window_label": np.str_(self.window.label),
        }
        if self.phase is not None:
            arrays["phase"] = self.phase
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Spectrum:
        d = np.load(path)
        return cls(
            amplitude=d["amplitude"],
            samplerate=float(d["samplerate"]),
            nfft=int(d["nfft"]),
            window=Window(d["window_samples"], str(d["window_label"])),
            phase=d["phase"] if "phase" in d.files else None,
            bin0=int(d["bin0"]),
            averages=int(d["averages"]),
        )

    def __repr__(self) -> str:
        f = self.frequencies
        span = f"{f[0]:g}..{f[-1]:g} Hz" if f.size else "empty"
        return (f"Spectrum({len(self)} bins, {span}, binwidth={self.binwidth:g} Hz, "
                f"window={self.window.label!r}, averages={self.averages})")


def cut_dc_common(spectra: list[Spectrum]) -> list[Spectrum]:
    """Cut the DC skirt from every spectrum at the deepest common index."""
    if not spectra:
        raise ValueError("no spectra given")
    idx = max(s.cut_dc(return_index=True) for s in spectra)
    return [s._slice(idx, len(s)) for s in spectra]
