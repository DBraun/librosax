#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rhythm and tempo features

This module provides wrappers around librosa.feature for rhythm and tempo analysis.
"""

import librosa.feature as _lf
from jax import numpy as jnp
import numpy as np
from typing import Optional, Union, Callable


__all__ = [
    'tempogram',
    'fourier_tempogram',
    'tempogram_ratio',
    'tempo',
]


def tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: str = 'hann',
    norm: Optional[Union[float, Callable]] = np.inf,
) -> np.ndarray:
    """Compute the tempogram: local autocorrelation of the onset strength envelope.

    This is a wrapper around librosa.feature.tempogram.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series. Required if `onset_envelope` is not provided.
    sr : float > 0
        Sampling rate of the audio time series
    onset_envelope : np.ndarray [shape=(n,)] or None
        Optional pre-computed onset strength envelope
    hop_length : int > 0
        Number of audio samples between successive onset measurements
    win_length : int > 0
        Length of the onset autocorrelation window (in frames/onset measurements)
    center : bool
        If `True`, onset autocorrelation windows are centered.
        If `False`, windows are left-aligned.
    window : str, tuple, number, callable, or list-like
        A window specification as in `get_window`
    norm : {np.inf, float > 0, None, callable}
        Normalization mode. Set to `None` to disable normalization.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length, n)]
        Tempogram matrix

    See Also
    --------
    fourier_tempogram
    librosa.feature.tempogram
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if onset_envelope is not None:
        onset_envelope = np.asarray(onset_envelope)

    result = _lf.tempogram(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        window=window,
        norm=norm,
    )

    return result


def fourier_tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: str = 'hann',
) -> np.ndarray:
    """Compute the Fourier tempogram: Fourier transform of the onset strength envelope.

    This is a wrapper around librosa.feature.fourier_tempogram.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series. Required if `onset_envelope` is not provided.
    sr : float > 0
        Sampling rate of the audio time series
    onset_envelope : np.ndarray [shape=(n,)] or None
        Optional pre-computed onset strength envelope
    hop_length : int > 0
        Number of audio samples between successive onset measurements
    win_length : int > 0
        Length of the onset window (in frames/onset measurements)
    center : bool
        If `True`, onset windows are centered.
        If `False`, windows are left-aligned.
    window : str, tuple, number, callable, or list-like
        A window specification as in `get_window`

    Returns
    -------
    ftgram : np.ndarray [shape=(..., win_length // 2 + 1, n), dtype=complex]
        Fourier tempogram matrix

    See Also
    --------
    tempogram
    librosa.feature.fourier_tempogram
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if onset_envelope is not None:
        onset_envelope = np.asarray(onset_envelope)

    result = _lf.fourier_tempogram(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        window=window,
    )

    return result


def tempogram_ratio(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    tg: Optional[np.ndarray] = None,
    bpm: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: Optional[np.ndarray] = None,
    factors: Optional[np.ndarray] = None,
    aggregate: Optional[Callable] = None,
    prior: Optional[Callable] = None,
    center: bool = True,
    window: str = 'hann',
    kind: str = 'linear',
    fill_value: float = 0,
    norm: Optional[Union[float, Callable]] = np.inf,
) -> np.ndarray:
    """Compute tempogram ratio: the ratio of absolute energy in tempo harmonics.

    This is a wrapper around librosa.feature.tempogram_ratio.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series (if tempogram not provided)
    sr : float > 0
        Sampling rate of the audio time series
    onset_envelope : np.ndarray [shape=(n,)] or None
        Pre-computed onset strength envelope (if tempogram not provided)
    tg : np.ndarray [shape=(..., win_length, n)] or None
        Pre-computed tempogram. If not provided, it will be computed from
        `y` or `onset_envelope`.
    bpm : np.ndarray or None
        Optional BPM values to evaluate
    hop_length : int > 0
        Number of audio samples between successive onset measurements
    win_length : int > 0
        Length of the tempogram window
    start_bpm : float > 0
        Initial guess of the BPM
    std_bpm : float > 0
        Standard deviation of tempo distribution
    max_tempo : float > 0 or None
        Maximum tempo to consider (in BPM)
    freqs : np.ndarray or None
        Optional frequency values
    factors : np.ndarray or None
        Optional factors for ratio computation
    aggregate : callable or None
        Aggregation function
    prior : callable or None
        Prior distribution over tempo
    center : bool
        If `True`, tempogram windows are centered.
    window : str, tuple, number, callable, or list-like
        A window specification
    kind : str
        Interpolation kind
    fill_value : float
        Fill value for interpolation
    norm : {np.inf, float > 0, None, callable}
        Normalization mode

    Returns
    -------
    ratio : np.ndarray
        Tempogram ratio features

    See Also
    --------
    tempogram
    librosa.feature.tempogram_ratio
    """
    # Convert JAX arrays to numpy if needed
    if tg is not None:
        tg = np.asarray(tg)
    if y is not None:
        y = np.asarray(y)
    if onset_envelope is not None:
        onset_envelope = np.asarray(onset_envelope)
    if bpm is not None:
        bpm = np.asarray(bpm)
    if freqs is not None:
        freqs = np.asarray(freqs)
    if factors is not None:
        factors = np.asarray(factors)

    result = _lf.tempogram_ratio(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        tg=tg,
        bpm=bpm,
        hop_length=hop_length,
        win_length=win_length,
        start_bpm=start_bpm,
        std_bpm=std_bpm,
        max_tempo=max_tempo,
        freqs=freqs,
        factors=factors,
        aggregate=aggregate,
        prior=prior,
        center=center,
        window=window,
        kind=kind,
        fill_value=fill_value,
        norm=norm,
    )

    return result


def tempo(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: float = 320.0,
    aggregate: Optional[Callable] = None,
    prior: Optional[Callable] = None,
) -> Union[float, np.ndarray]:
    """Estimate the tempo (beats per minute) from an audio signal.

    This is a wrapper around librosa.feature.tempo.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series
    sr : float > 0
        Sampling rate of the audio time series
    onset_envelope : np.ndarray [shape=(n,)] or None
        Optional pre-computed onset strength envelope
    hop_length : int > 0
        Number of audio samples between successive onset measurements
    start_bpm : float > 0
        Initial guess of the BPM
    std_bpm : float > 0
        Standard deviation of tempo distribution
    ac_size : float > 0
        Length (in seconds) of the onset autocorrelation window
    max_tempo : float > 0
        Maximum tempo to consider (in BPM)
    aggregate : callable or None
        Aggregation function. If provided, used to combine tempo estimates.
    prior : callable or None
        Prior distribution over tempo

    Returns
    -------
    tempo : float or np.ndarray
        Estimated tempo (in beats per minute)

    See Also
    --------
    tempogram
    librosa.feature.tempo
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if onset_envelope is not None:
        onset_envelope = np.asarray(onset_envelope)

    result = _lf.tempo(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        start_bpm=start_bpm,
        std_bpm=std_bpm,
        ac_size=ac_size,
        max_tempo=max_tempo,
        aggregate=aggregate,
        prior=prior,
    )

    return result
