#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio signal generation and processing utilities"""
from typing import Optional
from jax import numpy as jnp
import numpy as np
import librosa
from ..util.exceptions import ParameterError


__all__ = [
    "tone",
    "autocorrelate",
]


def tone(
    frequency: float,
    *,
    sr: float = 22050,
    length: Optional[int] = None,
    duration: Optional[float] = None,
    phi: Optional[float] = None,
) -> jnp.ndarray:
    """Construct a pure tone (cosine) signal at a given frequency.

    Parameters
    ----------
    frequency : float > 0
        frequency
    sr : number > 0
        desired sampling rate of the output signal
    length : int > 0
        desired number of samples in the output signal.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    duration : float > 0
        desired duration in seconds.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    phi : float or None
        phase offset, in radians. If unspecified, defaults to ``-jnp.pi * 0.5``.

    Returns
    -------
    tone_signal : jnp.ndarray [shape=(length,), dtype=float64]
        Synthesized pure sine tone signal

    Raises
    ------
    ParameterError
        - If ``frequency`` is not provided.
        - If neither ``length`` nor ``duration`` are provided.

    Examples
    --------
    Generate a pure sine tone A4

    >>> tone = librosax.tone(440, duration=1, sr=22050)

    Or generate the same signal using `length`

    >>> tone = librosax.tone(440, sr=22050, length=22050)
    """
    if frequency is None:
        raise ParameterError('"frequency" must be provided')

    # Compute signal length
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
        length = int(duration * sr)

    if phi is None:
        phi = -jnp.pi * 0.5

    y: jnp.ndarray = jnp.cos(2 * jnp.pi * frequency * jnp.arange(length) / sr + phi)
    return y


def autocorrelate(
    y: jnp.ndarray, *, max_size: Optional[int] = None, axis: int = -1
) -> jnp.ndarray:
    """Bounded-lag auto-correlation.

    This function computes the autocorrelation of a signal using FFT for efficiency.
    The autocorrelation is bounded to a maximum lag specified by ``max_size``.

    Args:
        y: Array to autocorrelate.
        max_size: Maximum correlation lag. If None, defaults to ``y.shape[axis]`` (unbounded).
        axis: Axis along which to autocorrelate. Default is -1 (last axis).

    Returns:
        Autocorrelated array. If ``max_size`` is specified, the shape along ``axis``
        will be ``max_size``. Otherwise, it matches ``y.shape[axis]``.

    Examples:
        Compute full autocorrelation of a white noise signal

        >>> import numpy as np
        >>> y = np.random.randn(256)
        >>> z = librosax.autocorrelate(y)
        >>> z.shape
        (256,)

        Compute autocorrelation with a maximum lag of 32 samples

        >>> z = librosax.autocorrelate(y, max_size=32)
        >>> z.shape
        (32,)

        Autocorrelate along the time axis of a batch of signals

        >>> y = np.random.randn(10, 256)  # 10 signals of length 256
        >>> z = librosax.autocorrelate(y, axis=-1)
        >>> z.shape
        (10, 256)
    """
    # Use librosa's implementation which is well-optimized
    y_np = np.asarray(y)
    result = librosa.autocorrelate(y_np, max_size=max_size, axis=axis)
    return jnp.asarray(result)
