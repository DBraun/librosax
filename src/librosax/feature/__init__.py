"""
Feature extraction
==================

Spectral features
-----------------
.. autosummary::
    :toctree: generated/

    spectral_centroid
    spectral_bandwidth
    spectral_rolloff
    spectral_flatness
    spectral_contrast
    rms
    zero_crossing_rate

Mel-frequency representations
-----------------------------
.. autosummary::
    :toctree: generated/

    melspectrogram
    mfcc

Chromagram
----------
.. autosummary::
    :toctree: generated/

    chroma_stft
    chroma_cqt
    chroma_filter

Constant-Q transform
--------------------
.. autosummary::
    :toctree: generated/

    cqt
    cqt2010
    cqt_frequencies

Tonal features
--------------
.. autosummary::
    :toctree: generated/

    tonnetz

Pitch and frequency
-------------------
.. autosummary::
    :toctree: generated/

    hz_to_octs
    note_to_hz
"""

from .spectral import (
    chroma_stft,
    chroma_cqt,
    chroma_vqt,
    chroma_cens,
    melspectrogram,
    mfcc,
    rms,
    spectral_centroid,
    spectral_bandwidth,
    spectral_contrast,
    spectral_flatness,
    spectral_rolloff,
    poly_features,
    tonnetz,
    zero_crossing_rate,

    hz_to_octs,
    chroma_filter,
    note_to_hz,
    cqt_frequencies,
    cqt,
    cqt2010,
)

from .temporal import (
    delta,
    stack_memory,
)

from .rhythm import (
    tempogram,
    fourier_tempogram,
    tempogram_ratio,
    tempo,
)

from . import inverse

__all__ = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_rolloff",
    "spectral_flatness",
    "poly_features",
    "rms",
    "zero_crossing_rate",
    "chroma_stft",
    "chroma_cqt",
    "chroma_vqt",
    "chroma_cens",
    "melspectrogram",
    "mfcc",
    "tonnetz",
    "hz_to_octs",
    "chroma_filter",
    "note_to_hz",
    "cqt_frequencies",
    "cqt",
    "cqt2010",
    # Temporal features
    "delta",
    "stack_memory",
    # Rhythm features
    "tempogram",
    "fourier_tempogram",
    "tempogram_ratio",
    "tempo",
]