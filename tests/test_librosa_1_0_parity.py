#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Tests for the librosa 1.0.0 API changes tracked by librosax.

Each test names the librosa pull request it follows, and compares against the
installed librosa where librosax delegates or claims numeric equivalence.
"""

import warnings

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import pytest
import scipy.signal

import librosax
import librosax.feature


@pytest.fixture
def y_mono():
    """A ten-second click track at 22050 Hz."""
    sr = 22050
    t = np.arange(10 * sr)
    return (np.sin(2 * np.pi * 220 * t / sr) * (t % 5512 < 400)).astype(np.float32)


# -- librosa 1.0.0 expired the constant_q deprecation --#


def test_constant_q_is_gone():
    for module in (librosa.filters, librosax.filters):
        assert not hasattr(module, "constant_q")
        assert not hasattr(module, "constant_q_lengths")


@pytest.mark.parametrize(
    "call",
    [
        lambda f: librosax.filters.wavelet(freqs=f, sr=11025),
        lambda f: librosax.filters.wavelet_lengths(freqs=f, sr=11025),
    ],
)
def test_filters_never_hand_jax_arrays_to_librosa(call, monkeypatch):
    # librosa's numba kernels cannot type a jax.Array, but the failure only
    # surfaces when no earlier NumPy call has warmed the same ufunc loop.
    # Assert the coercion itself rather than depending on process ordering.
    seen = []

    for name in ("wavelet", "wavelet_lengths"):
        original = getattr(librosa.filters, name)

        def spy(*args, _original=original, **kwargs):
            seen.extend(
                v for v in list(args) + list(kwargs.values()) if isinstance(v, jax.Array)
            )
            return _original(*args, **kwargs)

        monkeypatch.setattr(librosa.filters, name, spy)

    freqs = librosax.cqt_frequencies(n_bins=12, fmin=130.81, bins_per_octave=12)
    assert isinstance(freqs, jax.Array)

    call(freqs)
    assert seen == [], f"jax.Array reached librosa: {seen}"


def test_filters_agree_on_jax_and_numpy_input():
    freqs = librosax.cqt_frequencies(n_bins=12, fmin=130.81, bins_per_octave=12)

    F_jax, lengths_jax = librosax.filters.wavelet(freqs=freqs, sr=11025)
    F_np, lengths_np = librosax.filters.wavelet(freqs=np.asarray(freqs), sr=11025)

    assert np.allclose(F_jax, F_np)
    assert np.allclose(lengths_jax, lengths_np)
