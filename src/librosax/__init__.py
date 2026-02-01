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

# Exception classes
from .util.exceptions import (
    LibrosaxError,
    ParameterError,
)

# Core functionality at the top level
from .core import (
    # Spectral representations
    stft,
    istft,
    # Magnitude scaling
    power_to_db,
    amplitude_to_db,
    # Frequency utilities
    fft_frequencies,
    cqt_frequencies,
    mel_frequencies,
    tempo_frequencies,
    fourier_tempo_frequencies,
    # Time/sample conversions
    frames_to_samples,
    frames_to_time,
    samples_to_frames,
    samples_to_time,
    time_to_samples,
    time_to_frames,
    blocks_to_samples,
    blocks_to_frames,
    blocks_to_time,
    samples_like,
    times_like,
    # Note/frequency conversions
    note_to_hz,
    note_to_midi,
    midi_to_hz,
    midi_to_note,
    hz_to_note,
    hz_to_midi,
    hz_to_mel,
    hz_to_octs,
    hz_to_fjs,
    mel_to_hz,
    octs_to_hz,
    A4_to_tuning,
    tuning_to_A4,
    # Indian classical music (svara) conversions
    midi_to_svara_h,
    midi_to_svara_c,
    note_to_svara_h,
    note_to_svara_c,
    hz_to_svara_h,
    hz_to_svara_c,
    # Frequency weighting
    A_weighting,
    B_weighting,
    C_weighting,
    D_weighting,
    Z_weighting,
    frequency_weighting,
    multi_frequency_weighting,
)
