#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Utility functions for array processing

``localmax``, ``localmin`` and ``normalize`` are implemented natively in JAX,
so they compose with ``jit`` and ``vmap`` rather than forcing a round trip
through NumPy.  ``localmax`` and ``localmin`` return boolean masks and so carry
no gradient; ``normalize`` is differentiable.  ``peak_pick`` is inherently
sequential — each selection suppresses a window of its successors — and
delegates to librosa on host memory.
"""

from typing import Literal

import jax.numpy as jnp
import librosa.util as _librosa_util
import numpy as np

from librosax.core.audio import fix_length
from librosax.core.spectrum import normalize, tiny

_PeakPickMethod = Literal["greedy", "dp_count", "dp_value"]


def _local_extrema(x: jnp.ndarray, axis: int, maximum: bool) -> jnp.ndarray:
    """Compare each element against its neighbors along ``axis``.

    The comparison against the previous element is strict and the comparison
    against the next element is not, so plateaus resolve to their final sample.
    The first element is never selected.

    Args:
        x: Input array.
        axis: Axis along which neighbors are taken.
        maximum: If ``True`` find maxima, otherwise minima.

    Returns:
        Boolean array of the same shape as ``x``.
    """
    xi = jnp.moveaxis(x, axis, -1)

    if xi.shape[-1] < 2:
        return jnp.zeros(x.shape, dtype=bool)

    if maximum:
        strict_prev = xi[..., 1:] > xi[..., :-1]
        loose_next = xi[..., :-1] >= xi[..., 1:]
    else:
        strict_prev = xi[..., 1:] < xi[..., :-1]
        loose_next = xi[..., :-1] <= xi[..., 1:]

    first = jnp.zeros(xi.shape[:-1] + (1,), dtype=bool)
    interior = strict_prev[..., :-1] & loose_next[..., 1:]
    last = strict_prev[..., -1:]

    out = jnp.concatenate([first, interior, last], axis=-1)
    return jnp.moveaxis(out, -1, axis)


def localmax(x: jnp.ndarray, *, axis: int = 0) -> jnp.ndarray:
    """Find local maxima in an array.

    An element ``x[i]`` is considered a local maximum if:

    - ``x[i] > x[i-1]`` (strict inequality)
    - ``x[i] >= x[i+1]``

    Note that the first element ``x[0]`` will never be considered a local
    maximum.

    Args:
        x: Input array.
        axis: Axis along which to search for local maxima.

    Returns:
        Boolean array of same shape as x, True where x is a local maximum.

    Examples:
        >>> x = jnp.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosax.util.localmax(x)
        Array([False, False, False,  True, False,  True, False,  True], dtype=bool)

        Find local maxima in each row of a matrix

        >>> x = jnp.array([[1, 0, 1], [2, -1, 0], [-2, 1, 0]])
        >>> librosax.util.localmax(x, axis=1)
        Array([[False, False,  True],
               [False, False,  True],
               [False,  True, False]], dtype=bool)
    """
    return _local_extrema(x, axis=axis, maximum=True)


def localmin(x: jnp.ndarray, *, axis: int = 0) -> jnp.ndarray:
    """Find local minima in an array.

    An element ``x[i]`` is considered a local minimum if:

    - ``x[i] < x[i-1]`` (strict inequality)
    - ``x[i] <= x[i+1]``

    Note that the first element ``x[0]`` will never be considered a local
    minimum.

    Args:
        x: Input array.
        axis: Axis along which to search for local minima.

    Returns:
        Boolean array of same shape as x, True where x is a local minimum.

    Examples:
        >>> x = jnp.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosax.util.localmin(x)
        Array([False,  True, False, False,  True, False,  True, False], dtype=bool)
    """
    return _local_extrema(x, axis=axis, maximum=False)


def peak_pick(
    x: jnp.ndarray,
    *,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    delta: float = 0.0,
    wait: int = 10,
    sparse: bool = True,
    method: _PeakPickMethod = "greedy",
    axis: int = -1,
) -> jnp.ndarray:
    """Peak picking using adaptive local thresholding.

    A sample ``n`` is selected as a peak if:

    1. ``x[n] == max(x[n - pre_max:n + post_max])``
    2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
    3. ``n - previous_n > wait``

    Selection is a sequential decision problem, so this function is not
    JIT-compatible: the input is moved to host memory and handed to librosa.

    Args:
        x: Input signal.
        pre_max: Number of samples before n to use for max filtering.
        post_max: Number of samples after n to use for max filtering.
        pre_avg: Number of samples before n to use for mean filtering.
        post_avg: Number of samples after n to use for mean filtering.
        delta: Threshold offset for mean filtering.
        wait: Minimum number of samples between peaks.
        sparse: If ``True`` (default), return the indices of the peaks, which
            requires ``x`` to be one-dimensional. If ``False``, return a boolean
            mask with the same shape as ``x``.
        method: Peak selection strategy. ``"greedy"`` (default) takes each
            qualifying peak as it is encountered. ``"dp_count"`` and
            ``"dp_value"`` use dynamic programming to maximize, respectively,
            the number of selected peaks and the sum of their values, subject to
            the same ``wait`` constraint.
        axis: Axis along which to pick peaks.

    Returns:
        Indices of the peaks if ``sparse`` is ``True``, otherwise a boolean mask.

    Examples:
        >>> x = jnp.array([0.0, 1, 2, 1, 0, 1, 0])
        >>> librosax.util.peak_pick(
        ...     x, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=1
        ... )
        Array([0, 2, 5], dtype=int32)
    """
    result = _librosa_util.peak_pick(
        np.asarray(x),
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
        sparse=sparse,
        method=method,
        axis=axis,
    )
    return jnp.asarray(result)


def valid_audio(y: jnp.ndarray) -> bool:
    """Validate whether a variable contains valid audio data.

    A valid audio buffer is a real-valued floating point array with at least one
    dimension and no non-finite entries.

    Args:
        y: Input audio signal to validate.

    Returns:
        ``True`` if ``y`` is a valid audio buffer.

    Raises:
        librosa.util.exceptions.ParameterError: If ``y`` is not a valid audio
            buffer. Invalid input raises rather than returning ``False``, which
            matches ``librosa.util.valid_audio``.

    Examples:
        >>> librosax.util.valid_audio(jnp.array([0.1, 0.2, 0.3]))
        True
    """
    return _librosa_util.valid_audio(np.asarray(y))


__all__ = [
    "fix_length",
    "localmax",
    "localmin",
    "normalize",
    "peak_pick",
    "tiny",
    "valid_audio",
]
