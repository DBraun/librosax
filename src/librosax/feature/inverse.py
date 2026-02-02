#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inverse feature transforms

This module provides inverse transforms for converting features back to audio
or intermediate representations.
"""

from typing import Optional, Union, Callable
import jax.numpy as jnp
import numpy as np
import librosa.feature.inverse as _inv


def mel_to_stft(
    M: jnp.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
    **kwargs
) -> jnp.ndarray:
    """Approximate STFT magnitude from a Mel power spectrogram.

    This is the inverse of melspectrogram. It uses pseudo-inverse of the mel
    basis to map mel-scaled features back to STFT-scaled features.

    Args:
        M: Mel spectrogram with shape (..., n_mels, t).
        sr: Sample rate.
        n_fft: FFT window size.
        power: Exponent for the magnitude spectrogram (2.0 for power, 1.0 for magnitude).
        **kwargs: Additional arguments for mel filter bank (fmin, fmax, htk, norm).

    Returns:
        STFT magnitude with shape (..., 1 + n_fft // 2, t).

    Examples:
        >>> M = librosax.feature.melspectrogram(y, sr=sr)
        >>> S_approx = librosax.feature.inverse.mel_to_stft(M, sr=sr)
    """
    M_np = np.asarray(M)
    result = _inv.mel_to_stft(M_np, sr=sr, n_fft=n_fft, power=power, **kwargs)
    # Return numpy array for compatibility with numpy testing utilities
    # Inverse transforms are typically used at pipeline endpoints
    return result


def mel_to_audio(
    M: jnp.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
    **kwargs
) -> jnp.ndarray:
    """Approximate audio signal from a Mel power spectrogram.

    This uses Griffin-Lim algorithm to reconstruct phase and invert the STFT.

    Args:
        M: Mel spectrogram with shape (..., n_mels, t).
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length. If None, defaults to n_fft.
        window: Window function.
        center: If True, center frames.
        pad_mode: Padding mode.
        power: Exponent for the magnitude spectrogram.
        n_iter: Number of Griffin-Lim iterations.
        length: If provided, trim or pad the output to this length.
        **kwargs: Additional arguments for mel filter bank.

    Returns:
        Audio signal with shape (..., length).

    Examples:
        >>> M = librosax.feature.melspectrogram(y, sr=sr)
        >>> y_approx = librosax.feature.inverse.mel_to_audio(M, sr=sr)
    """
    M_np = np.asarray(M)
    result = _inv.mel_to_audio(
        M_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        n_iter=n_iter,
        length=length,
        **kwargs
    )
    # Return numpy array for compatibility with numpy testing utilities
    # Inverse transforms are typically used at pipeline endpoints
    return result


def mfcc_to_mel(
    mfcc: jnp.ndarray,
    *,
    n_mels: int = 128,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    ref: Union[float, Callable] = 1.0,
    lifter: int = 0,
) -> jnp.ndarray:
    """Convert Mel-frequency cepstral coefficients to a Mel power spectrogram.

    This uses the inverse discrete cosine transform (IDCT).

    Args:
        mfcc: MFCC features with shape (..., n_mfcc, t).
        n_mels: Number of Mel bands to generate.
        dct_type: Discrete cosine transform type (1, 2, or 3).
        norm: Normalization mode for DCT.
        ref: Reference value for dB conversion.
        lifter: Cepstral liftering coefficient. If > 0, applies de-liftering.

    Returns:
        Mel spectrogram with shape (..., n_mels, t).

    Examples:
        >>> mfcc = librosax.feature.mfcc(y, sr=sr)
        >>> M_approx = librosax.feature.inverse.mfcc_to_mel(mfcc)
    """
    mfcc_np = np.asarray(mfcc)
    result = _inv.mfcc_to_mel(
        mfcc_np, n_mels=n_mels, dct_type=dct_type, norm=norm, ref=ref, lifter=lifter
    )
    # Return numpy array for compatibility with numpy testing utilities
    # Users can convert to JAX if needed
    return result


def mfcc_to_audio(
    mfcc: jnp.ndarray,
    *,
    n_mels: int = 128,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    ref: Union[float, Callable] = 1.0,
    lifter: int = 0,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
    **kwargs
) -> jnp.ndarray:
    """Convert Mel-frequency cepstral coefficients to an audio signal.

    This applies IDCT to get Mel spectrogram, then uses Griffin-Lim to reconstruct audio.

    Args:
        mfcc: MFCC features with shape (..., n_mfcc, t).
        n_mels: Number of Mel bands.
        dct_type: Discrete cosine transform type.
        norm: Normalization mode for DCT.
        ref: Reference value for dB conversion.
        lifter: Cepstral liftering coefficient.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length.
        window: Window function.
        center: If True, center frames.
        pad_mode: Padding mode.
        power: Exponent for the magnitude spectrogram.
        n_iter: Number of Griffin-Lim iterations.
        length: If provided, trim or pad the output to this length.
        **kwargs: Additional arguments for mel filter bank.

    Returns:
        Audio signal with shape (..., length).

    Examples:
        >>> mfcc = librosax.feature.mfcc(y, sr=sr)
        >>> y_approx = librosax.feature.inverse.mfcc_to_audio(mfcc, sr=sr)
    """
    mfcc_np = np.asarray(mfcc)
    result = _inv.mfcc_to_audio(
        mfcc_np,
        n_mels=n_mels,
        dct_type=dct_type,
        norm=norm,
        ref=ref,
        lifter=lifter,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        n_iter=n_iter,
        length=length,
        **kwargs
    )
    # Return numpy array for compatibility with numpy testing utilities
    # Inverse transforms are typically used at pipeline endpoints
    return result


__all__ = [
    "mel_to_stft",
    "mel_to_audio",
    "mfcc_to_mel",
    "mfcc_to_audio",
]
