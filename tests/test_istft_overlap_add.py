import jax
import librosa
import numpy as np
import pytest
import torch

import librosax


def _reference_istft(spectrum, window, hop_length, length):
    n_fft = window.size
    n_frames = spectrum.shape[-1]
    full_length = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros((*spectrum.shape[:-2], full_length), dtype=window.dtype)
    envelope = np.zeros(full_length, dtype=window.dtype)
    for i in range(n_frames):
        samples = slice(i * hop_length, i * hop_length + n_fft)
        signal[..., samples] += np.fft.irfft(spectrum[..., i], n=n_fft) * window
        envelope[samples] += window**2

    signal /= np.where(envelope > 1e-10, envelope, 1)
    start = n_fft // 2
    if length is None:
        return signal[..., start : full_length - start]
    signal = signal[..., start : start + length]
    return np.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(0, length - signal.shape[-1])])


_devices = [jax.devices("cpu")[0]] + [
    device for device in jax.devices() if device.platform == "gpu"
]


@pytest.mark.parametrize("device", _devices, ids=str)
@pytest.mark.parametrize("n_frames", [2, 4, 8])
@pytest.mark.parametrize("window, win_length", [("boxcar", 64), ("hann", 64), ("hann", 48)])
@pytest.mark.parametrize("extra_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_istft_overlap_add(device, n_frames, window, win_length, extra_samples, dtype):
    n_fft, hop_length = 64, 16
    output_length = hop_length * (n_frames - 1)
    length = None if extra_samples is None else output_length + extra_samples
    rng = np.random.default_rng(42)
    shape = (2, 3, n_fft // 2 + 1, n_frames)
    spectrum = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    window_values = librosa.filters.get_window(window, win_length).astype(spectrum.real.dtype)
    pad = (n_fft - win_length) // 2
    padded_window = np.pad(window_values, (pad, n_fft - win_length - pad))
    expected = _reference_istft(spectrum, padded_window, hop_length, length)
    expected_torch = torch.istft(
        torch.from_numpy(spectrum.reshape(-1, *shape[-2:])),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.from_numpy(window_values),
        length=length,
    ).numpy().reshape(expected.shape)

    def reconstruct(D):
        return librosax.istft(
            D,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=length,
        )

    with jax.enable_x64(dtype == np.complex128):
        D = jax.device_put(spectrum, device)
        for actual in (reconstruct(D), jax.jit(reconstruct)(D)):
            assert actual.shape == (*shape[:-2], length if length is not None else output_length)
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(actual, expected_torch, atol=1e-6, rtol=1e-6)


def test_istft_pads_beyond_the_final_frame():
    n_fft, hop_length, length = 65, 16, 53
    spectrum = np.zeros((n_fft // 2 + 1, 2), dtype=np.complex64)
    spectrum[0, -1] = n_fft
    expected = _reference_istft(spectrum, np.ones(n_fft, dtype=np.float32), hop_length, length)

    with jax.enable_x64(False):
        actual = librosax.istft(
            spectrum, n_fft=n_fft, hop_length=hop_length, window="boxcar", length=length
        )

    assert actual.shape == (length,)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(actual[49:], 0)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("n_samples", [64, 80, 95, 8197])
def test_istft_roundtrip_across_waveform_lengths(center, n_samples):
    y = np.random.default_rng(19).standard_normal(n_samples, dtype=np.float32)
    n_fft, hop_length = 64, 16
    kwargs = dict(n_fft=n_fft, hop_length=hop_length, window="boxcar", center=center)

    spectrum = librosax.stft(y, **kwargs)
    reconstruct = jax.jit(lambda D: librosax.istft(D, length=n_samples, **kwargs))
    reconstructed = reconstruct(spectrum)

    expected = y.copy()
    if not center:
        covered = n_fft + hop_length * (spectrum.shape[-1] - 1)
        expected[covered:] = 0
    np.testing.assert_allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("length", [None, 49, 88])
def test_istft_uncentered_hann_matches_librosa(length):
    n_fft, hop_length = 64, 16
    spectrum = np.ones((n_fft // 2 + 1, 2), dtype=np.complex64)
    spectrum[0] = [n_fft, 2 * n_fft]
    kwargs = dict(n_fft=n_fft, hop_length=hop_length, win_length=48, center=False, length=length)

    expected = librosa.istft(spectrum, **kwargs)
    actual = librosax.istft(spectrum, **kwargs)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
