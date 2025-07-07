"""
Librosax
========

A JAX implementation of audio processing functions from librosa and TorchLibrosa.

This library provides JAX-compatible implementations of many functions from:
- librosa (https://github.com/librosa/librosa)
- TorchLibrosa (https://github.com/qiuqiangkong/torchlibrosa)

We gratefully acknowledge the librosa development team and Qiuqiang Kong for their
foundational work in audio processing.

Spectral representations
------------------------
.. autosummary::
    :toctree: generated/

    stft
    istft

Magnitude scaling
-----------------
.. autosummary::
    :toctree: generated/

    amplitude_to_db
    power_to_db

Frequency utilities
-------------------
.. autosummary::
    :toctree: generated/

    fft_frequencies

Submodules
----------
.. autosummary::
    :toctree: _autosummary

    feature
    layers
"""

from .version import version as __version__

# Only expose core functionality at the top level
from .core import (
    stft,
    istft,
    power_to_db,
    amplitude_to_db,
    fft_frequencies,
)
