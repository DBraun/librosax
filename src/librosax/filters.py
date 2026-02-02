#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Filter bank construction

This module wraps librosa.filters to provide filter bank construction functions.
For now, we use librosa's implementations directly as they work well with JAX arrays.
"""

from typing import Optional, Union, Callable
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import librosa.filters as _librosa_filters


def mel(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[Union[str, float]] = "slaney",
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Create a Mel filter-bank.

    This wraps librosa.filters.mel and converts the result to a JAX array.

    Args:
        sr: Sample rate of the incoming signal.
        n_fft: Number of FFT components.
        n_mels: Number of Mel bands to generate.
        fmin: Lowest frequency (in Hz).
        fmax: Highest frequency (in Hz). If None, use sr / 2.0.
        htk: Use HTK formula instead of Slaney.
        norm: Normalization method. Options: None, 'slaney', or a number.
        dtype: The data type of the output array.

    Returns:
        Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1).

    Examples:
        >>> mel_fb = librosax.filters.mel(sr=22050, n_fft=2048, n_mels=128)
        >>> mel_fb.shape
        (128, 1025)
    """
    # Call librosa's implementation and convert to JAX array
    mel_basis = _librosa_filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
    )
    return jnp.asarray(mel_basis, dtype=dtype)


def chroma(
    *,
    sr: float,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    norm: Optional[float] = 2,
    base_c: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Create a chroma filter bank.

    This wraps librosa.filters.chroma and converts the result to a JAX array.

    Args:
        sr: Sample rate of the incoming signal.
        n_fft: Number of FFT components.
        n_chroma: Number of chroma bins to produce.
        tuning: Tuning deviation from A440 in fractional chroma bins.
        ctroct: Center octave for Gaussian weighting.
        octwidth: Width of Gaussian weighting (in octaves). If None, disable weighting.
        norm: Normalization factor for each filter.
        base_c: If True, the first chroma bin is C. If False, the first bin is A.
        dtype: The data type of the output array.

    Returns:
        Chroma filter matrix of shape (n_chroma, n_fft // 2 + 1).

    Examples:
        >>> chroma_fb = librosax.filters.chroma(sr=22050, n_fft=2048)
        >>> chroma_fb.shape
        (12, 1025)
    """
    chroma_basis = _librosa_filters.chroma(
        sr=sr,
        n_fft=n_fft,
        n_chroma=n_chroma,
        tuning=tuning,
        ctroct=ctroct,
        octwidth=octwidth,
        norm=norm,
        base_c=base_c,
    )
    return jnp.asarray(chroma_basis, dtype=dtype)


def constant_q(
    *,
    sr: float,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    window: str = "hann",
    filter_scale: float = 1,
    pad_fft: bool = True,
    norm: Optional[float] = 1,
    dtype: jnp.dtype = jnp.complex64,
    gamma: float = 0,
    **kwargs
):
    """Create a Constant-Q filter bank.

    This wraps librosa.filters.constant_q.

    Args:
        sr: Sample rate of the incoming signal.
        fmin: Minimum frequency. If None, defaults to C1 (~32.7 Hz).
        n_bins: Number of frequency bins.
        bins_per_octave: Number of bins per octave.
        window: Window function for each filter.
        filter_scale: Scale factor for filter bandwidth.
        pad_fft: Pad filter FFT to the nearest power of 2.
        norm: Normalization factor for each filter.
        dtype: The data type of the output array.
        gamma: Bandwidth offset for variable-Q filterbank.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (filters, lengths) where:
        - filters: CQ filter bank (sparse matrix)
        - lengths: Length of each filter

    Examples:
        >>> cqt_fb, lengths = librosax.filters.constant_q(sr=22050, fmin=32.7, n_bins=84)
    """
    # Call librosa's implementation which returns (filters, lengths)
    result = _librosa_filters.constant_q(
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
        pad_fft=pad_fft,
        norm=norm,
        dtype=dtype,
        gamma=gamma,
        **kwargs
    )
    # Return the tuple as-is (filters, lengths)
    return result


def get_window(
    window: Union[str, tuple, float, ArrayLike],
    Nx: int,
    *,
    fftbins: bool = True,
) -> jnp.ndarray:
    """Return a window of a given length and type.

    This wraps librosa.filters.get_window.

    Args:
        window: Window specification.
        Nx: Length of the window.
        fftbins: If True, create a periodic window for use with FFT/STFT.

    Returns:
        Window array of length Nx.

    Examples:
        >>> window = librosax.filters.get_window('hann', 1024)
        >>> window.shape
        (1024,)
    """
    win = _librosa_filters.get_window(window, Nx, fftbins=fftbins)
    return jnp.asarray(win)


def semitone_filterbank(
    *,
    center_freqs: Optional[jnp.ndarray] = None,
    tuning: float = 0.0,
    sample_rates: Optional[jnp.ndarray] = None,
    flayout: str = "ba",
    **kwargs
):
    """Create a multirate semitone filterbank.

    This wraps librosa.filters.semitone_filterbank.

    Args:
        center_freqs: Optional array of center frequencies. If None, uses MIDI range.
        tuning: Tuning deviation from A440 in fractional bins.
        sample_rates: Optional array of sample rates for each filter.
        flayout: Filter layout - 'ba' for numerator/denominator form.
        **kwargs: Additional arguments passed to the filterbank constructor.

    Returns:
        Tuple of (filters, sample_rates) where filters is a list of filter coefficients.

    Examples:
        >>> filters, srs = librosax.filters.semitone_filterbank()
    """
    # Convert JAX arrays to numpy if needed
    if center_freqs is not None:
        center_freqs = np.asarray(center_freqs)
    if sample_rates is not None:
        sample_rates = np.asarray(sample_rates)

    result = _librosa_filters.semitone_filterbank(
        center_freqs=center_freqs,
        tuning=tuning,
        sample_rates=sample_rates,
        flayout=flayout,
        **kwargs
    )
    return result


def window_bandwidth(
    window: Union[str, tuple],
    n: Optional[int] = 1000,
) -> float:
    """Get the bandwidth of a window function.

    This wraps librosa.filters.window_bandwidth.

    Args:
        window: Window specification.
        n: Number of samples for window bandwidth estimation.

    Returns:
        Bandwidth of the window (in frequency bins).

    Examples:
        >>> librosax.filters.window_bandwidth('hann')
        1.5
    """
    return _librosa_filters.window_bandwidth(window, n=n)


def cq_to_chroma(
    *,
    n_input: int,
    bins_per_octave: int = 12,
    n_chroma: int = 12,
    fmin: Optional[float] = None,
    window: Optional[ArrayLike] = None,
    base_c: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Convert a Constant-Q basis to a chroma representation.

    This wraps librosa.filters.cq_to_chroma.

    Args:
        n_input: Number of input CQ bins.
        bins_per_octave: Number of bins per octave in the CQ representation.
        n_chroma: Number of chroma bins to produce.
        fmin: Minimum frequency of the CQ representation.
        window: Optional window function to apply to each chroma bin.
        base_c: If True, the first chroma bin is C.
        dtype: The data type of the output array.

    Returns:
        CQ-to-chroma transformation matrix of shape (n_chroma, n_input).

    Examples:
        >>> cq_to_chr = librosax.filters.cq_to_chroma(n_input=84, bins_per_octave=12)
        >>> cq_to_chr.shape
        (12, 84)
    """
    cq2chr = _librosa_filters.cq_to_chroma(
        n_input=n_input,
        bins_per_octave=bins_per_octave,
        n_chroma=n_chroma,
        fmin=fmin,
        window=window,
        base_c=base_c,
    )
    return jnp.asarray(cq2chr, dtype=dtype)


def diagonal_filter(
    window: ArrayLike,
    n: int,
    *,
    slope: float = 1.0,
    angle: Optional[float] = None,
    zero_mean: bool = False,
) -> jnp.ndarray:
    """Create a diagonal filter for time-frequency analysis.

    This wraps librosa.filters.diagonal_filter.

    Args:
        window: Window function to use for the filter.
        n: Length of the filter.
        slope: Slope of the filter (frequency bins per time step).
        angle: Alternative to slope; angle of the filter in radians.
        zero_mean: If True, make the filter have zero mean.

    Returns:
        Diagonal filter.

    Examples:
        >>> window = jnp.ones(5)
        >>> diag_filt = librosax.filters.diagonal_filter(window, n=100, slope=0.5)
    """
    filt = _librosa_filters.diagonal_filter(
        window=window, n=n, slope=slope, angle=angle, zero_mean=zero_mean
    )
    return jnp.asarray(filt)


def constant_q_lengths(
    *,
    sr: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    window: str = "hann",
    filter_scale: float = 1,
    gamma: float = 0,
) -> jnp.ndarray:
    """Return length of each filter in a constant-Q filterbank.

    This wraps librosa.filters.constant_q_lengths.

    Args:
        sr: Sample rate.
        fmin: Minimum frequency (required).
        n_bins: Number of frequency bins.
        bins_per_octave: Number of bins per octave.
        window: Window function for each filter.
        filter_scale: Scale factor for filter bandwidth.
        gamma: Bandwidth offset for variable-Q filterbank.

    Returns:
        Array of filter lengths.

    Examples:
        >>> lengths = librosax.filters.constant_q_lengths(sr=22050, fmin=32.7, n_bins=84)
    """
    lengths = _librosa_filters.constant_q_lengths(
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
    )
    return jnp.asarray(lengths)


def wavelet(
    *,
    freqs: ArrayLike,
    sr: float = 22050,
    window: str = "hann",
    filter_scale: float = 1,
    pad_fft: bool = True,
    norm: Optional[float] = 1,
    dtype: jnp.dtype = jnp.complex64,
    gamma: float = 0,
    alpha: Optional[float] = None,
    **kwargs
):
    """Construct a Morlet wavelet filter bank.

    This wraps librosa.filters.wavelet.

    Args:
        freqs: Center frequencies of the filters.
        sr: Sample rate.
        window: Window function for each filter.
        filter_scale: Scale factor for filter bandwidth.
        pad_fft: Pad filter FFT to the nearest power of 2.
        norm: Normalization factor.
        dtype: Output data type.
        gamma: Bandwidth offset for variable-Q filterbank.
        alpha: Alternate bandwidth parameter.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (filters, lengths) where:
        - filters: Wavelet filter bank (array)
        - lengths: Length of each filter

    Examples:
        >>> freqs = librosax.note_to_hz(['C2', 'C3', 'C4'])
        >>> wavelets, lengths = librosax.filters.wavelet(freqs=freqs)
    """
    # librosa returns tuple (filters, lengths)
    result = _librosa_filters.wavelet(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        pad_fft=pad_fft,
        norm=norm,
        dtype=dtype,
        gamma=gamma,
        alpha=alpha,
        **kwargs
    )
    return result


def wavelet_lengths(
    *,
    freqs: ArrayLike,
    sr: float = 22050,
    window: str = "hann",
    filter_scale: float = 1,
    gamma: Optional[float] = 0,
    alpha: Optional[float] = None,
):
    """Return the length of each filter in a wavelet filter bank.

    This wraps librosa.filters.wavelet_lengths.

    Args:
        freqs: Center frequencies of the filters.
        sr: Sample rate.
        window: Window function for each filter.
        filter_scale: Scale factor for filter bandwidth.
        gamma: Bandwidth offset for variable-Q filterbank.
        alpha: Alternate bandwidth parameter.

    Returns:
        Tuple of (lengths, bw_Q) where:
        - lengths: Array of filter lengths
        - bw_Q: Bandwidth parameter

    Examples:
        >>> freqs = librosax.note_to_hz(['C2', 'C3', 'C4'])
        >>> lengths, bw = librosax.filters.wavelet_lengths(freqs=freqs)
    """
    result = _librosa_filters.wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
    )
    return result


def window_sumsquare(
    window: Union[str, tuple, float, ArrayLike],
    n_frames: int,
    *,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_fft: int = 2048,
    norm: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute the sum-square envelope of a window function.

    This wraps librosa.filters.window_sumsquare.

    Args:
        window: Window specification.
        n_frames: Number of analysis frames.
        hop_length: Hop length between frames.
        win_length: Window length. If None, defaults to n_fft.
        n_fft: FFT window size.
        norm: Normalization factor.
        dtype: Output data type.

    Returns:
        Window sum-square envelope.

    Examples:
        >>> wss = librosax.filters.window_sumsquare('hann', n_frames=100, hop_length=512)
    """
    wss = _librosa_filters.window_sumsquare(
        window=window,
        n_frames=n_frames,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        norm=norm,
        dtype=dtype,
    )
    return jnp.asarray(wss)


def __float_window(window_spec: Union[str, tuple, Callable]) -> Callable:
    """Decorate a window function to support fractional input lengths.

    This is a private function used internally for testing and special cases.
    It wraps window functions to handle fractional lengths by:
    1. Creating a window of length ceil(n) for fractional n
    2. Setting values from floor(n) onwards to 0

    For integer n, behavior is unchanged.

    Args:
        window_spec: Window specification (name, tuple, or callable).

    Returns:
        A wrapped window function that supports fractional lengths.

    Examples:
        >>> wrapped = librosax.filters.__float_window('hann')
        >>> window = wrapped(10.5)  # Creates window of length 11 with last value zeroed
    """
    def _wrap(n, *args, **kwargs):
        """Wrap the window"""
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        # Convert to numpy and make a copy for mutability
        window = np.array(window, copy=True)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode="constant")

        window[n_min:] = 0.0

        return window

    return _wrap


# Export all functions
__all__ = [
    "mel",
    "chroma",
    "constant_q",
    "constant_q_lengths",
    "get_window",
    "semitone_filterbank",
    "window_bandwidth",
    "window_sumsquare",
    "cq_to_chroma",
    "diagonal_filter",
    "wavelet",
    "wavelet_lengths",
]
