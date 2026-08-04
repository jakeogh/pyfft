#!/usr/bin/env python3
# tab-width:4

"""
pyfft - spectra, peaks, and ADC metrics for high-speed ADC records.

    import pyfft
    spec = pyfft.average_spectrum(y, samplerate, nfft=1 << 16)
    plot.add_plot(spec.frequencies, spec.db)
    tone = pyfft.analyze_tone(spec)
"""

from .compute import average_spectrum as average_spectrum
from .compute import compute_spectrum as compute_spectrum
from .compute import fft_frequencies as fft_frequencies
from .compute import next_fast_len as next_fast_len
from .metrics import NoiseFloor as NoiseFloor
from .metrics import ToneAnalysis as ToneAnalysis
from .metrics import WaveformStats as WaveformStats
from .metrics import analyze_tone as analyze_tone
from .metrics import codes_to_volts as codes_to_volts
from .metrics import noise_floor as noise_floor
from .metrics import waveform_stats as waveform_stats
from .peaks import Harmonic as Harmonic
from .peaks import Peak as Peak
from .peaks import find_harmonics as find_harmonics
from .peaks import find_peaks as find_peaks
from .peaks import fold_frequency as fold_frequency
from .spectrum import Spectrum as Spectrum
from .spectrum import SpectrumPoint as SpectrumPoint
from .spectrum import cut_dc_common as cut_dc_common
from .windows import Window as Window
from .windows import WindowSpec as WindowSpec
