"""
Librosax
========

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
