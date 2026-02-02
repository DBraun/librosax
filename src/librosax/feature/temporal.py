#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Temporal feature functions"""

from typing import Optional
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import librosa.feature


def delta(
    data: jnp.ndarray,
    *,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = "interp",
    **kwargs
) -> jnp.ndarray:
    """Compute delta features: local estimate of the derivative.

    This is a wrapper around librosa.feature.delta for full compatibility.

    Delta features are computed using a centered finite difference approximation:

        delta[t] = (data[t+1] - data[t-1]) / 2

    This can be extended to higher orders (acceleration, etc.) by iterative
    application.

    Args:
        data: Input data matrix. Shape: (..., T) where T is the time dimension.
        width: Number of frames over which to compute the delta features.
            Must be an odd positive integer.
        order: Order of the difference operator (1=velocity, 2=acceleration).
        axis: Axis along which to compute deltas. Default is -1 (time axis).
        mode: Mode for handling boundaries:
            - 'interp': Linearly interpolate at boundaries
            - 'nearest': Use nearest values at boundaries
            - 'mirror': Mirror the data at boundaries
            - 'constant': Use constant values at boundaries
            - 'wrap': Wrap around at boundaries
        **kwargs: Additional keyword arguments passed to the boundary mode.

    Returns:
        Delta features with the same shape as input data.

    Examples:
        Compute MFCC deltas

        >>> y, sr = librosax.load('audio.wav')
        >>> mfcc = librosax.mfcc(y=y, sr=sr)
        >>> mfcc_delta = librosax.delta(mfcc)
        >>> mfcc_delta.shape == mfcc.shape
        True

        Compute delta-deltas (acceleration)

        >>> mfcc_delta2 = librosax.delta(mfcc, order=2)
    """
    # Use librosa's implementation for perfect compatibility
    data_np = np.asarray(data)
    result = librosa.feature.delta(
        data_np, width=width, order=order, axis=axis, mode=mode, **kwargs
    )
    return jnp.asarray(result)


def stack_memory(
    data: jnp.ndarray,
    *,
    n_steps: int = 2,
    delay: int = 1,
    **kwargs
) -> jnp.ndarray:
    """Stack delayed copies of a feature matrix into a single matrix.

    This is a wrapper around librosa.feature.stack_memory for full compatibility.

    Args:
        data: Input feature matrix. Shape: (..., d, T) where d is the feature
            dimension and T is the time dimension.
        n_steps: Number of time steps to stack. Must be >= 1.
        delay: Delay (in frames) between each stacked copy. Can be positive or negative.
            - Positive delay: looks backward in time
            - Negative delay: looks forward in time
        **kwargs: Additional keyword arguments (for future compatibility).

    Returns:
        Stacked feature matrix with shape (..., d * n_steps, T).
        Each output frame at time t contains:
        [data[:, t], data[:, t-delay], data[:, t-2*delay], ...]

    Examples:
        Stack MFCCs with 2-frame context

        >>> y, sr = librosax.load('audio.wav')
        >>> mfcc = librosax.mfcc(y=y, sr=sr)
        >>> mfcc.shape
        (20, 100)
        >>> mfcc_stack = librosax.stack_memory(mfcc, n_steps=3, delay=1)
        >>> mfcc_stack.shape
        (60, 100)
        >>> # Output contains [mfcc[t], mfcc[t-1], mfcc[t-2]]
    """
    # Use librosa's implementation for perfect compatibility
    data_np = np.asarray(data)
    result = librosa.feature.stack_memory(data_np, n_steps=n_steps, delay=delay, **kwargs)
    return jnp.asarray(result)


__all__ = ["delta", "stack_memory"]
