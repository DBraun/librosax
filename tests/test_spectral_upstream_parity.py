"""Regression coverage for librosa 1.0.1 spectral features."""

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import pytest

import librosax


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name",
    ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_contrast", "poly_features"],
)
def test_spectral_feature_dtype_and_values(name, dtype):
    rng = np.random.default_rng(0)
    S = np.abs(rng.standard_normal((2, 1025, 8))).astype(dtype)

    with jax.enable_x64():
        actual = getattr(librosax.feature, name)(S=jnp.asarray(S))

    expected = getattr(librosa.feature, name)(S=S)
    assert actual.dtype == expected.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize("name", ["mfcc", "spectral_contrast"])
@pytest.mark.parametrize("top_db, use_callable", [(None, True), (0.0, False)])
def test_spectral_features_forward_db_options_per_channel(name, top_db, use_callable):
    rng = np.random.default_rng(0)
    channel_scales = np.array([1.0, 1e-4], dtype=np.float32)[:, None]
    y = rng.standard_normal((2, 4096), dtype=np.float32) * channel_scales
    upstream = getattr(librosa.feature, name)
    feature = getattr(librosax.feature, name)
    jax_ref = jnp.max if use_callable else 100.0
    numpy_ref = np.max if use_callable else 100.0

    with jax.enable_x64():
        actual = jax.jit(lambda y: feature(y=y, ref=jax_ref, top_db=top_db))(jnp.asarray(y))

    expected = upstream(y=y, ref=numpy_ref, top_db=top_db)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)


def test_fractional_window_forwards_fftbins():
    actual = librosax.filters.__float_window("hann")(8.5, fftbins=False)
    expected = np.pad(librosa.filters.get_window("hann", 8, fftbins=False), (0, 1))

    np.testing.assert_allclose(actual, expected)
