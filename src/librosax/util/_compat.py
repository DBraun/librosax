#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Interoperability helpers for delegating to librosa.

Several librosax functions are thin wrappers around librosa, which is written
against NumPy and compiles parts of its hot paths with numba.  A ``jax.Array``
satisfies the buffer protocol but is not an ``np.ndarray``, and numba refuses to
type it, so every array that crosses the boundary into librosa must first be
materialized as NumPy.
"""

from typing import Any

import jax
import numpy as np

__all__ = ["as_numpy"]


def as_numpy(x: Any) -> Any:
    """Materialize a JAX array as NumPy, passing everything else through.

    Window and filter arguments in librosa accept strings, tuples, floats and
    callables in addition to arrays, so non-array inputs are returned unchanged
    rather than coerced.

    Args:
        x: Value bound for a librosa call.

    Returns:
        ``np.asarray(x)`` if ``x`` is a JAX array, otherwise ``x`` itself.
    """
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return x
