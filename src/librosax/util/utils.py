#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Utility functions for array processing

This module provides utility functions that wrap librosa.util for JAX compatibility.
"""

from typing import Optional
import jax.numpy as jnp
import numpy as np
import librosa.util as _librosa_util


def localmax(x: jnp.ndarray, *, axis: int = 0) -> jnp.ndarray:
    """Find local maxima in an array.

    An element ``x[i]`` is considered a local maximum if:
    - ``x[i] > x[i-1]`` (strict inequality)
    - ``x[i] >= x[i+1]``

    Note that the first element ``x[0]`` will never be considered a local maximum.

    Args:
        x: Input array.
        axis: Axis along which to search for local maxima.

    Returns:
        Boolean array of same shape as x, True where x is a local maximum.

    Examples:
        >>> x = jnp.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosax.util.localmax(x)
        array([False, False, False,  True, False,  True, False,  True])

        Find local maxima in each row of a matrix

        >>> x = jnp.array([[1, 0, 1], [2, -1, 0], [-2, 1, 0]])
        >>> librosax.util.localmax(x, axis=1)
        array([[False, False,  True],
               [ True, False, False],
               [False,  True, False]])
    """
    # Convert to numpy for processing, then back to JAX
    x_np = np.asarray(x)
    result = _librosa_util.localmax(x_np, axis=axis)
    return jnp.asarray(result)


def localmin(x: jnp.ndarray, *, axis: int = 0) -> jnp.ndarray:
    """Find local minima in an array.

    An element ``x[i]`` is considered a local minimum if:
    - ``x[i] < x[i-1]`` (strict inequality)
    - ``x[i] <= x[i+1]``

    Note that the first element ``x[0]`` will never be considered a local minimum.

    Args:
        x: Input array.
        axis: Axis along which to search for local minima.

    Returns:
        Boolean array of same shape as x, True where x is a local minimum.

    Examples:
        >>> x = jnp.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosax.util.localmin(x)
        array([False,  True, False, False,  True, False,  True, False])
    """
    x_np = np.asarray(x)
    result = _librosa_util.localmin(x_np, axis=axis)
    return jnp.asarray(result)


def peak_pick(
    x: jnp.ndarray,
    *,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    delta: float = 0.0,
    wait: int = 10,
) -> jnp.ndarray:
    """Peak picking using adaptive local thresholding.

    This function identifies peaks in a 1-d signal using a combination of:
    - Local maximum detection
    - Adaptive thresholding based on local mean
    - Minimum separation between peaks

    Args:
        x: Input signal (1-dimensional).
        pre_max: Number of samples before n to use for max filtering.
        post_max: Number of samples after n to use for max filtering.
        pre_avg: Number of samples before n to use for mean filtering.
        post_avg: Number of samples after n to use for mean filtering.
        delta: Threshold offset for mean filtering.
        wait: Minimum number of samples between peaks.

    Returns:
        Array of indices where peaks occur.

    Examples:
        >>> x = jnp.array([0, 1, 2, 1, 0, 1, 0])
        >>> librosax.util.peak_pick(x, pre_max=1, post_max=1, pre_avg=1, post_avg=1)
        array([2, 5])
    """
    x_np = np.asarray(x)
    result = _librosa_util.peak_pick(
        x_np,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )
    return jnp.asarray(result)


def normalize(
    x: jnp.ndarray,
    *,
    norm: Optional[float] = jnp.inf,
    axis: int = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> jnp.ndarray:
    """Normalize an array along a specified axis.

    Args:
        x: Input array.
        norm: Normalization order (e.g., 1, 2, jnp.inf). If None, no normalization.
        axis: Axis along which to normalize.
        threshold: Only normalize if norm is above this threshold.
        fill: If True, entries with norm below threshold are set to 0.

    Returns:
        Normalized array of same shape as x.

    Examples:
        >>> x = jnp.array([[1, 2], [3, 4]])
        >>> librosax.util.normalize(x, norm=2, axis=0)
        array([[0.316..., 0.447...],
               [0.948..., 0.894...]])
    """
    x_np = np.asarray(x)
    result = _librosa_util.normalize(
        x_np, norm=norm, axis=axis, threshold=threshold, fill=fill
    )
    return jnp.asarray(result)


def valid_audio(y: jnp.ndarray) -> bool:
    """Validate whether a variable contains valid audio data.

    This function checks if the input is a valid audio signal by verifying:
    - The input is a numpy/JAX array
    - The input has numeric dtype (not string, object, etc.)
    - The input is 1D (mono) or 2D (multi-channel)
    - All values are finite (no NaN or Inf)

    Args:
        y: Input audio signal to validate.

    Returns:
        True if valid audio signal, False otherwise.

    Examples:
        >>> # Valid mono audio
        >>> y = jnp.array([0.1, 0.2, 0.3])
        >>> librosax.util.valid_audio(y)
        True

        >>> # Invalid: contains NaN
        >>> y_bad = jnp.array([0.1, jnp.nan, 0.3])
        >>> librosax.util.valid_audio(y_bad)
        False
    """
    y_np = np.asarray(y)
    return _librosa_util.valid_audio(y_np)


__all__ = [
    "localmax",
    "localmin",
    "peak_pick",
    "normalize",
    "valid_audio",
]
