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


# -- librosa #2051: axes-aware decibel conversion --#


@pytest.mark.parametrize("fn_name", ["power_to_db", "amplitude_to_db"])
@pytest.mark.parametrize("shape", [(), (17,), (5, 7), (3, 5, 7), (2, 3, 5, 7)])
def test_db_matches_librosa_default_axes(fn_name, shape):
    rs = np.random.RandomState(0)
    S = np.abs(rs.randn(*shape)).astype(np.float64) + 1e-3

    expected = getattr(librosa, fn_name)(S)
    actual = getattr(librosax, fn_name)(jnp.asarray(S))

    assert np.allclose(np.asarray(actual), expected, atol=1e-5)


@pytest.mark.parametrize("fn_name", ["power_to_db", "amplitude_to_db"])
@pytest.mark.parametrize("ref", [np.max, np.median])
@pytest.mark.parametrize("axes", ["auto", None, -1, (-2, -1)])
def test_db_callable_ref_matches_librosa(fn_name, ref, axes):
    rs = np.random.RandomState(1)
    S = np.abs(rs.randn(3, 5, 7)).astype(np.float64) + 1e-3

    expected = getattr(librosa, fn_name)(S, ref=ref, axes=axes)
    actual = getattr(librosax, fn_name)(jnp.asarray(S), ref=ref, axes=axes)

    assert np.allclose(np.asarray(actual), expected, atol=1e-5)


def test_power_to_db_top_db_is_per_slice():
    # Two spectrograms whose peaks differ by far more than top_db.  Reducing
    # top_db globally flattens the quiet one to a constant; reducing it per
    # slice keeps its internal structure.
    quiet = jnp.reshape(jnp.linspace(1e-9, 2e-9, 16), (4, 4))
    S = jnp.stack([jnp.full((4, 4), 1.0), quiet])

    per_slice = librosax.power_to_db(S, top_db=80.0, amin=1e-12)
    global_reduction = librosax.power_to_db(S, top_db=80.0, amin=1e-12, axes=None)

    assert np.ptp(np.asarray(per_slice[1])) > 1.0
    assert np.ptp(np.asarray(global_reduction[1])) == 0.0

    # librosa's default is the per-slice reduction.
    assert np.allclose(
        np.asarray(per_slice),
        librosa.power_to_db(np.asarray(S), top_db=80.0, amin=1e-12),
        atol=1e-4,
    )


def test_power_to_db_rejects_ref_without_axis():
    with pytest.raises(librosax.ParameterError):
        librosax.power_to_db(jnp.ones((3, 3)), ref=lambda x: x.max())


def test_power_to_db_rejects_negative_top_db():
    with pytest.raises(librosax.ParameterError):
        librosax.power_to_db(jnp.ones((3, 3)), top_db=-1.0)


# -- librosa #1947 / #1948: note and MIDI conversion --#


def test_note_to_hz_keeps_tuning_deviation_by_default():
    assert np.isclose(
        float(librosax.note_to_hz("C2-32")), float(librosa.note_to_hz("C2-32"))
    )
    assert not np.isclose(
        float(librosax.note_to_hz("C2-32")), float(librosax.note_to_hz("C2"))
    )


def test_note_to_hz_round_midi_quantizes():
    assert np.isclose(
        float(librosax.note_to_hz("C2-32", round_midi=True)),
        float(librosax.note_to_hz("C2")),
    )


def test_note_to_midi_empty_string_is_nan():
    assert np.isnan(librosax.note_to_midi(""))
    assert np.isnan(librosa.note_to_midi(""))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_midi_to_note_nonfinite_is_empty(bad):
    assert librosax.midi_to_note(bad) == ""
    assert librosax.midi_to_svara_h(bad, Sa=60) == ""
    assert librosax.midi_to_svara_c(bad, Sa=60, mela=1) == ""


def test_note_midi_roundtrip_through_empty():
    assert librosax.midi_to_note(librosax.note_to_midi("")) == ""


# -- librosa #1955: non-finite weighting curves --#


@pytest.mark.parametrize("kind", ["A", "B", "C", "D"])
def test_weighting_warns_at_zero_hz(kind):
    fn = getattr(librosax, f"{kind}_weighting")
    with pytest.warns(UserWarning, match=f"{kind}-weighting returned non-finite"):
        weights = fn(jnp.asarray([0.0, 440.0]), min_db=None)
    assert not np.all(np.isfinite(np.asarray(weights)))


@pytest.mark.parametrize("kind", ["A", "B", "C", "D"])
def test_weighting_silent_when_finite(kind):
    fn = getattr(librosax, f"{kind}_weighting")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn(jnp.asarray([100.0, 440.0]), min_db=None)


@pytest.mark.parametrize("kind", ["A", "B", "C", "D"])
def test_weighting_matches_librosa(kind):
    freqs = np.geomspace(20.0, 10000.0, 64)
    expected = getattr(librosa, f"{kind}_weighting")(freqs)
    actual = getattr(librosax, f"{kind}_weighting")(jnp.asarray(freqs))
    assert np.allclose(np.asarray(actual), expected, atol=1e-4)


# -- librosa #2087: rms buffer validation --#


def test_rms_rejects_scalar_input():
    with pytest.raises(librosax.ParameterError):
        librosax.feature.rms(y=jnp.asarray(1.0))


def test_rms_rejects_integer_input():
    with pytest.raises(librosax.ParameterError):
        librosax.feature.rms(y=jnp.arange(2048))


def test_rms_still_jits():
    y = jnp.asarray(np.random.RandomState(2).randn(4096).astype(np.float32))
    fn = jax.jit(librosax.feature.rms, static_argnames=("frame_length", "hop_length"))
    assert np.all(np.isfinite(np.asarray(fn(y=y))))


# -- librosa #1927: peak picking --#


@pytest.mark.parametrize("method", ["greedy", "dp_count", "dp_value"])
def test_peak_pick_methods_match_librosa(method, y_mono):
    onset = np.abs(librosa.stft(y_mono)).sum(axis=0).astype(np.float64)
    kwargs = dict(
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10, method=method
    )

    expected = librosa.util.peak_pick(onset, **kwargs)
    actual = librosax.util.peak_pick(jnp.asarray(onset), **kwargs)

    assert np.array_equal(np.asarray(actual), expected)


def test_peak_pick_dense_is_multidimensional(y_mono):
    S = np.abs(librosa.stft(y_mono)).astype(np.float64)
    kwargs = dict(
        pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.0, wait=2, sparse=False
    )

    expected = librosa.util.peak_pick(S, **kwargs)
    actual = librosax.util.peak_pick(jnp.asarray(S), **kwargs)

    assert actual.shape == S.shape
    assert np.array_equal(np.asarray(actual), expected)


def test_peak_pick_rejects_unknown_method():
    with pytest.raises(librosa.ParameterError):
        librosax.util.peak_pick(
            jnp.arange(16.0),
            pre_max=1,
            post_max=1,
            pre_avg=1,
            post_avg=1,
            delta=0.0,
            wait=1,
            method="nonsense",
        )


# -- librosa #1964: signal mixing helpers --#


@pytest.fixture
def signals():
    y1 = np.asarray(librosax.tone(440.0, sr=22050, duration=0.1), dtype=np.float32)
    y2 = np.asarray(librosax.tone(550.0, sr=22050, duration=0.1), dtype=np.float32)
    y3 = np.asarray(librosax.tone(660.0, sr=22050, duration=0.05), dtype=np.float32)
    return y1, y2, y3


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_to_mono_matches_librosa(signals, pad, norm):
    y1, y2, y3 = signals
    expected = librosa.to_mono(y1, y2, y3, pad=pad, norm=norm)
    actual = librosax.to_mono(
        *(jnp.asarray(y) for y in (y1, y2, y3)), pad=pad, norm=norm
    )
    assert np.allclose(np.asarray(actual), expected, atol=1e-6)


def test_to_mono_downmixes_channels(signals):
    y1, y2, _ = signals
    stereo = np.stack([y1, y2])
    expected = librosa.to_mono(stereo)
    actual = librosax.to_mono(jnp.asarray(stereo))
    assert np.allclose(np.asarray(actual), expected, atol=1e-6)


@pytest.mark.parametrize("downmix", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_to_stereo_matches_librosa(signals, downmix, norm):
    y1, y2, _ = signals
    expected = librosa.to_stereo(left=y1, right=y2, downmix=downmix, norm=norm)
    actual = librosax.to_stereo(
        left=jnp.asarray(y1), right=jnp.asarray(y2), downmix=downmix, norm=norm
    )
    assert np.allclose(np.asarray(actual), expected, atol=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_to_stereo_one_sided(signals, side):
    y1, _, _ = signals
    kwargs = {"left": None, "right": None, side: y1}
    expected = librosa.to_stereo(**kwargs)
    kwargs[side] = jnp.asarray(y1)
    actual = librosax.to_stereo(**kwargs)
    assert np.allclose(np.asarray(actual), expected, atol=1e-6)


def test_to_stereo_requires_a_channel():
    with pytest.raises(librosax.ParameterError):
        librosax.to_stereo(left=None, right=None)


@pytest.mark.parametrize("downmix", [True, False])
def test_to_multi_matches_librosa(signals, downmix):
    y1, y2, _ = signals
    expected = librosa.to_multi(y1, y2, downmix=downmix)
    actual = librosax.to_multi(
        jnp.asarray(y1), jnp.asarray(y2), downmix=downmix
    )
    assert np.allclose(np.asarray(actual), expected, atol=1e-6)


def test_to_multi_rejects_mismatched_layouts(signals):
    y1, y2, _ = signals
    with pytest.raises(librosax.ParameterError):
        librosax.to_multi(
            jnp.asarray(y1), jnp.asarray(np.stack([y1, y2])), downmix=False
        )


def test_mixing_rejects_empty_input():
    with pytest.raises(librosax.ParameterError):
        librosax.to_mono()


@pytest.mark.parametrize("size", [3, 7, 12])
def test_fix_length_matches_librosa(size):
    x = np.arange(7, dtype=np.float32)
    assert np.array_equal(
        np.asarray(librosax.fix_length(jnp.asarray(x), size=size)),
        librosa.util.fix_length(x, size=size),
    )


# -- librosa #2004 / #1986: new rhythm features --#


def test_hybrid_tempogram_matches_librosa(y_mono):
    expected = librosa.feature.hybrid_tempogram(y=y_mono, sr=22050)
    actual = librosax.feature.hybrid_tempogram(y=jnp.asarray(y_mono), sr=22050)
    assert np.allclose(np.asarray(actual), expected, atol=1e-5, equal_nan=True)


def test_metrogram_matches_librosa(y_mono):
    tg = librosa.feature.tempogram(y=y_mono, sr=22050)
    freqs = librosa.tempo_frequencies(tg.shape[-2], sr=22050)

    expected = librosa.feature.metrogram(tg=tg, freqs=freqs)
    actual = librosax.feature.metrogram(tg=jnp.asarray(tg), freqs=jnp.asarray(freqs))

    assert np.allclose(np.asarray(actual), expected, atol=1e-5, equal_nan=True)


def test_metrogram_unaggregated_shape(y_mono):
    tg = librosa.feature.tempogram(y=y_mono, sr=22050)
    freqs = librosa.tempo_frequencies(tg.shape[-2], sr=22050)

    out = librosax.feature.metrogram(
        tg=jnp.asarray(tg), freqs=jnp.asarray(freqs), aggregate=None
    )
    assert out.shape == (4, tg.shape[-2], tg.shape[-1])


# -- librosa #2066: automatic CQT bin count --#


@pytest.mark.parametrize("cqt_fn", ["cqt", "cqt2010"])
@pytest.mark.parametrize("bins_per_octave", [12, 24, 36])
def test_cqt_infers_same_n_bins_as_librosa(cqt_fn, bins_per_octave):
    # Clipping on the center frequency rather than the filter cutoff admits one
    # extra bin whose passband crosses Nyquist, which librosa drops.
    sr = 22050
    y = np.random.RandomState(3).randn(sr).astype(np.float32)

    expected = librosa.cqt(
        y=y, sr=sr, n_bins=None, bins_per_octave=bins_per_octave, hop_length=512
    ).shape[-2]
    actual = getattr(librosax.feature, cqt_fn)(
        jnp.asarray(y), sr=sr, n_bins=None, bins_per_octave=bins_per_octave
    ).shape[-2]

    assert actual == expected


def test_cqt_inferred_bins_stay_below_nyquist():
    sr = 22050
    fmin = float(librosax.note_to_hz("C1"))
    bins_per_octave = 24
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    cutoff_ratio = 1 + 0.5 * librosax.filters.window_bandwidth("hann") / Q

    y = jnp.asarray(np.random.RandomState(3).randn(sr).astype(np.float32))
    n_bins = librosax.feature.cqt(
        y, sr=sr, n_bins=None, bins_per_octave=bins_per_octave
    ).shape[-2]

    top = fmin * 2 ** ((n_bins - 1) / bins_per_octave)
    just_past = fmin * 2 ** (n_bins / bins_per_octave)
    assert top * cutoff_ratio <= sr / 2
    assert just_past * cutoff_ratio > sr / 2


def test_cqt2010_rejects_hop_that_cannot_cover_the_octaves():
    y = jnp.asarray(np.random.RandomState(3).randn(22050).astype(np.float32))
    with pytest.raises(librosax.ParameterError, match="must be a multiple of"):
        librosax.feature.cqt2010(y, sr=22050, hop_length=128, n_bins=None)


def test_cqt_rejects_bins_above_nyquist():
    y = jnp.asarray(np.random.RandomState(4).randn(22050).astype(np.float32))
    with pytest.raises(librosax.ParameterError):
        librosax.feature.cqt(y, sr=22050, n_bins=200)


# -- librosa 1.0.0 expired the constant_q deprecation --#


def test_constant_q_is_gone():
    for module in (librosa.filters, librosax.filters):
        assert not hasattr(module, "constant_q")
        assert not hasattr(module, "constant_q_lengths")


# -- JAX-native reimplementations --#


@pytest.mark.parametrize("shape,axis", [((32,), 0), ((8, 32), 0), ((8, 32), 1), ((4, 8, 32), -1)])
@pytest.mark.parametrize("fn_name", ["localmax", "localmin"])
def test_local_extrema_match_librosa(fn_name, shape, axis):
    x = np.random.RandomState(5).randint(0, 4, size=shape).astype(np.float64)

    expected = getattr(librosa.util, fn_name)(x, axis=axis)
    actual = getattr(librosax.util, fn_name)(jnp.asarray(x), axis=axis)

    assert np.array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("fn_name", ["localmax", "localmin"])
def test_local_extrema_are_jittable(fn_name):
    fn = jax.jit(getattr(librosax.util, fn_name), static_argnames=("axis",))
    x = jnp.asarray(np.random.RandomState(6).randn(64))
    assert np.array_equal(
        np.asarray(fn(x)), np.asarray(getattr(librosax.util, fn_name)(x))
    )


@pytest.mark.parametrize("max_size", [None, 32])
@pytest.mark.parametrize("shape", [(512,), (3, 512)])
def test_autocorrelate_matches_librosa(max_size, shape):
    x = np.random.RandomState(7).randn(*shape)

    expected = librosa.autocorrelate(x, max_size=max_size)
    actual = librosax.autocorrelate(jnp.asarray(x), max_size=max_size)

    assert actual.shape == expected.shape
    assert np.allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max())


def test_autocorrelate_is_jittable():
    x = jnp.asarray(np.random.RandomState(8).randn(512))
    fn = jax.jit(librosax.autocorrelate, static_argnames=("max_size", "axis"))
    assert np.allclose(np.asarray(fn(x)), np.asarray(librosax.autocorrelate(x)))


def test_cqt_honors_window_choice():
    # Every window other than "hann" used to fall back to a rectangular window.
    y = jnp.asarray(np.random.RandomState(9).randn(22050).astype(np.float32))
    kwargs = dict(sr=22050, n_bins=24, bins_per_octave=12)

    hann = np.asarray(librosax.feature.cqt(y, window="hann", **kwargs))
    hamming = np.asarray(librosax.feature.cqt(y, window="hamming", **kwargs))
    boxcar = np.asarray(librosax.feature.cqt(y, window="boxcar", **kwargs))

    assert not np.allclose(hann, hamming)
    assert not np.allclose(hann, boxcar)
    assert not np.allclose(hamming, boxcar)


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


def test_plimit_intervals_accepts_a_list():
    assert np.allclose(
        np.asarray(librosax.plimit_intervals(primes=[3, 5], bins_per_octave=7)),
        librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7),
    )


def test_spectral_contrast_accepts_a_list_of_freqs(y_mono):
    S = np.abs(librosa.stft(y_mono, n_fft=2048))
    freq = list(librosa.fft_frequencies(sr=22050, n_fft=2048))

    actual = librosax.feature.spectral_contrast(S=jnp.asarray(S), sr=22050, freq=freq)
    expected = librosa.feature.spectral_contrast(S=S, sr=22050, freq=np.asarray(freq))

    assert np.allclose(np.asarray(actual), expected, atol=1e-4)


def test_chroma_stft_estimates_tuning(y_mono):
    # Tuning estimation crosses into librosa through jax.pure_callback.
    actual = librosax.feature.chroma_stft(y=jnp.asarray(y_mono), sr=22050)
    expected = librosa.feature.chroma_stft(y=y_mono, sr=22050)

    assert actual.shape == expected.shape
    assert np.allclose(np.asarray(actual), expected, atol=1e-4)


# -- util.normalize must stay equivalent to librosa's --#


@pytest.mark.parametrize("norm", [np.inf, -np.inf, 0, 1, 2, 3.5, None])
@pytest.mark.parametrize("axis", [0, -1, None])
@pytest.mark.parametrize("fill", [None, False, True])
def test_normalize_matches_librosa(norm, axis, fill):
    if norm == 0 and fill is True:
        pytest.skip("librosa rejects this combination")

    rs = np.random.RandomState(11)
    # Include an all-zero slice so the sub-threshold `fill` branch is exercised.
    S = np.concatenate([rs.randn(4, 5), np.zeros((4, 1))], axis=-1)

    expected = librosa.util.normalize(S, norm=norm, axis=axis, fill=fill)
    actual = librosax.util.normalize(jnp.asarray(S), norm=norm, axis=axis, fill=fill)

    assert np.allclose(np.asarray(actual), expected, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("norm", [2, 3.0])
@pytest.mark.parametrize("axis", [0, None])
def test_normalize_fill_produces_unit_norm(norm, axis):
    # A previous implementation filled with 1.0 regardless of the norm order,
    # which is only unit-norm for the max/min norms.
    out = np.asarray(
        librosax.util.normalize(jnp.zeros((2, 3)), norm=norm, axis=axis, fill=True)
    )
    length = np.sum(np.abs(out) ** norm, axis=axis) ** (1.0 / norm)
    assert np.allclose(length, 1.0)


def test_normalize_accepts_numpy_scalar_norm():
    S = jnp.asarray(np.random.RandomState(12).randn(4, 5))
    assert np.allclose(
        np.asarray(librosax.util.normalize(S, norm=np.float32(2), axis=0)),
        np.asarray(librosax.util.normalize(S, norm=2, axis=0)),
    )


def test_normalize_accepts_integer_input():
    out = np.asarray(librosax.util.normalize(jnp.array([[1, 2], [3, 4]]), norm=2, axis=0))
    assert np.allclose(np.sum(out**2, axis=0), 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"norm": "bogus"},
        {"norm": -2},
        {"threshold": -1.0},
        {"fill": "yes"},
        {"norm": 0, "fill": True},
    ],
)
def test_normalize_rejects_bad_parameters(kwargs):
    with pytest.raises(librosax.ParameterError):
        librosax.util.normalize(jnp.ones((4, 5)), **kwargs)


def test_tiny_matches_librosa():
    for x in (jnp.zeros(3, dtype=jnp.float32), jnp.arange(3)):
        assert librosax.util.tiny(x) == librosa.util.tiny(np.asarray(x))


# -- power_to_db input validation --#


def test_power_to_db_rejects_nonpositive_amin():
    with pytest.raises(librosax.ParameterError, match="amin must be strictly positive"):
        librosax.power_to_db(jnp.ones((3, 3)), amin=0.0)


def test_power_to_db_rectifies_complex_input():
    rs = np.random.RandomState(13)
    D = (rs.randn(4, 5) + 1j * rs.randn(4, 5)).astype(np.complex64)

    with pytest.warns(UserWarning, match="complex input"):
        actual = librosax.power_to_db(jnp.asarray(D))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = librosa.power_to_db(D)

    assert not np.iscomplexobj(np.asarray(actual))
    assert np.allclose(np.asarray(actual), expected, atol=1e-4)


# -- regressions found in review of this branch --#


def test_autocorrelate_accepts_array_likes():
    # The JAX rewrite briefly dropped the input coercion.
    expected = librosa.autocorrelate(np.array([1.0, 2.0, 3.0]))
    actual = librosax.autocorrelate([1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(actual), expected, atol=1e-5)


@pytest.mark.parametrize(
    "window",
    ["hann", "hamming", ("kaiser", 4.0), np.hanning],
    ids=["str", "str-hamming", "tuple", "callable"],
)
def test_cqt_accepts_every_window_spec(window):
    # A CQT filter bank has a different length per bin, so only specs that can
    # be evaluated at an arbitrary length are meaningful here; librosa has the
    # same restriction.
    y = jnp.asarray(np.random.RandomState(14).randn(22050).astype(np.float32))
    C = librosax.feature.cqt(y, sr=22050, n_bins=12, bins_per_octave=12, window=window)
    assert C.shape[-2] == 12
    assert np.all(np.isfinite(np.asarray(C)))


def test_cqt_reports_a_fixed_length_window_clearly():
    # librosa raises "unhashable type: 'numpy.ndarray'" for this; routing
    # through filters.get_window gives the actual reason.
    y = jnp.asarray(np.random.RandomState(14).randn(22050).astype(np.float32))
    with pytest.raises(librosax.ParameterError, match="Window size mismatch"):
        librosax.feature.cqt(y, sr=22050, n_bins=12, window=np.hanning(2048))


@pytest.mark.parametrize("fn", ["to_mono", "to_multi"])
def test_mixing_helpers_do_not_warn_on_numpy_float64(fn):
    y = np.arange(8, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        getattr(librosax, fn)(y)


def test_to_stereo_does_not_warn_on_numpy_float64():
    y = np.arange(8, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        librosax.to_stereo(left=y, right=None)


def test_util_exposes_librosa_array_helpers():
    # Code ported from librosa reaches for these under `util`.
    for name in ("fix_length", "tiny", "normalize", "peak_pick", "localmax", "localmin"):
        assert hasattr(librosax.util, name), name
        assert hasattr(librosa.util, name), name


# -- window specifications accepted by stft / istft --#


@pytest.mark.parametrize(
    "window",
    ["hann", "hamming", "sqrt_hann", ("kaiser", 4.0), 4.0, np.hanning, np.hanning(2048)],
    ids=["str", "str-hamming", "sqrt_hann", "tuple", "float", "callable", "ndarray"],
)
def test_stft_accepts_every_window_spec(window):
    y = jnp.asarray(np.random.RandomState(15).randn(8192).astype(np.float32))

    S = librosax.stft(y, n_fft=2048, hop_length=512, window=window)
    assert S.shape == (1025, 17)
    assert np.all(np.isfinite(np.asarray(S)))

    y_hat = librosax.istft(S, n_fft=2048, hop_length=512, window=window)
    assert np.all(np.isfinite(np.asarray(y_hat)))


def test_stft_accepts_a_jax_window_array():
    y = jnp.asarray(np.random.RandomState(15).randn(8192).astype(np.float32))
    expected = librosax.stft(y, n_fft=2048, hop_length=512, window=np.hanning(2048))
    actual = librosax.stft(y, n_fft=2048, hop_length=512, window=jnp.hanning(2048))
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_stft_callable_and_array_windows_agree():
    # np.hanning as a callable and its evaluated array must select the same
    # window; both are the symmetric variant.
    y = jnp.asarray(np.random.RandomState(16).randn(8192).astype(np.float32))
    kwargs = dict(n_fft=2048, hop_length=512)

    from_callable = librosax.stft(y, window=np.hanning, **kwargs)
    from_array = librosax.stft(y, window=np.hanning(2048), **kwargs)

    assert np.allclose(np.asarray(from_callable), np.asarray(from_array), atol=1e-5)


def test_np_hanning_is_the_symmetric_hann():
    # np.hanning differs from the periodic window "hann" selects, so the two
    # must not be silently interchangeable.
    y = jnp.asarray(np.random.RandomState(17).randn(8192).astype(np.float32))
    kwargs = dict(n_fft=2048, hop_length=512)

    symmetric = np.asarray(librosax.stft(y, window=np.hanning, **kwargs))
    periodic = np.asarray(librosax.stft(y, window="hann", **kwargs))
    explicit = np.asarray(
        librosax.stft(y, window=scipy.signal.get_window("hann", 2048, fftbins=False), **kwargs)
    )

    assert not np.allclose(symmetric, periodic)
    assert np.allclose(symmetric, explicit, atol=1e-5)


def test_stft_rejects_a_window_of_the_wrong_length():
    y = jnp.asarray(np.random.RandomState(18).randn(8192).astype(np.float32))
    with pytest.raises(librosa.ParameterError, match="Window size mismatch"):
        librosax.stft(y, n_fft=2048, hop_length=512, window=np.hanning(1024))


@pytest.mark.parametrize("window", ["hann", np.hanning, np.hanning(2048)])
def test_stft_roundtrips_under_every_window_spec(window):
    n = 8192
    y = np.random.RandomState(19).randn(n).astype(np.float32)

    S = librosax.stft(jnp.asarray(y), n_fft=2048, hop_length=512, window=window)
    y_hat = np.asarray(
        librosax.istft(S, n_fft=2048, hop_length=512, window=window, length=n)
    )

    interior = slice(2048, n - 2048)
    assert np.allclose(y_hat[interior], y[interior], atol=1e-4)


def test_spectrogram_layer_accepts_a_callable_window():
    from librosax.layers import Spectrogram

    y = jnp.asarray(np.random.RandomState(20).randn(2, 8192).astype(np.float32))
    layer = Spectrogram(n_fft=2048, hop_length=512, window=np.hanning)
    out = layer(y)
    assert np.all(np.isfinite(np.asarray(out)))
