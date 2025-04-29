from itertools import product

from jax import numpy as jnp
from jax import random
import librosa
import librosa.feature
import numpy as np
import pytest
import torch
import torchlibrosa

import librosax
from librosax.layers import (
    Spectrogram,
    MFCC,
    LogMelFilterBank,
    SpecAugmentation,
    DropStripes,
)


@pytest.mark.parametrize(
    "n_fft,hop_length,win_length,window,center,pad_mode",
    product(
        [1024, 2048],
        [None, 256, 320],
        [None, 512],
        ["hann"],
        [False, True],
        ["constant", "reflect"],
    ),
)
def test_stft(
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    pad_mode: str,
):
    C = 2
    duration_samp = 44_100
    x = np.random.uniform(-1, 1, size=(C, duration_samp)) * 0.5

    librosa_res = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    jax_res = librosax.stft(
        jnp.array(x),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    jax_res = np.array(jax_res)
    jax_res = jax_res[..., : librosa_res.shape[-1]]  # todo: avoid this

    np.testing.assert_allclose(librosa_res, jax_res, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "n_fft,hop_length,win_length,window,center,pad_mode,length",
    product(
        [1024, 2048],
        [None, 256, 320],
        [None, 512],
        ["hann"],
        [True],  # todo: need to test center==False
        ["constant", "reflect"],
        [None],
    ),
)
def test_istft(
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    pad_mode: str,
    length: int,
):
    C = 2
    duration_samp = 44_100
    x = np.random.uniform(-1, 1, size=(C, duration_samp)) * 0.5

    stft_matrix = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    librosa_res = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        length=length,
    )
    stft_matrix = jnp.array(stft_matrix)
    jax_res = librosax.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        length=length,
    )

    np.testing.assert_allclose(librosa_res, jax_res, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "n_fft,hop_length,win_length,window,center,pad_mode",
    product(
        [1024, 2048],
        [None, 256, 320],
        [None, 512],
        ["hann", "sqrt_hann"],
        [True],  # todo: need to test center==False
        ["constant", "reflect"],
    ),
)
def test_istft2(
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    pad_mode: str,
):
    """
    Test that librosax.istft undoes librosax.stft
    """
    C = 2
    duration_samp = hop_length * 128 if hop_length is not None else n_fft * 32
    # duration_samp = 44_100  # todo: use this instead of value above
    x = random.uniform(random.key(0), shape=(C, duration_samp), minval=-0.5, maxval=0.5)
    length = x.shape[-1]

    stft_matrix = librosax.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    y = librosax.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        length=length,
    )

    x = np.array(x)
    y = np.array(y)

    np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-5)


def test_mel_spec():
    np.random.seed(42)
    sr = 22_050
    x = np.random.uniform(-1, 1, size=(1, sr))  # fmt: skip

    n_fft = 2048
    hop_length = 512
    win_length = n_fft
    window = "hann"
    n_mels = 64
    fmin = 0.0
    fmax = sr / 2
    is_log = True
    pad_mode = "constant"

    # Compute the spectrogram.
    S, _ = Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        pad_mode=pad_mode,
    ).init_with_output({"params": random.key(0)}, jnp.array(x))
    S = np.array(S)
    S_librosa = torchlibrosa.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        pad_mode=pad_mode,
    )(torch.from_numpy(x).to(torch.float32))
    S_librosa = S_librosa.detach().cpu().numpy()
    S_librosa = S_librosa.squeeze(1)
    assert S.shape == S_librosa.shape
    np.testing.assert_allclose(
        S, S_librosa, atol=1e-2, rtol=1e-5
    )  # todo: not a great atol

    # Compute the log-mel spectrogram.
    logmel_spec, _ = LogMelFilterBank(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, is_log=is_log
    ).init_with_output({"params": random.key(0)}, S)

    logmel_spec_librosa = torchlibrosa.LogmelFilterBank(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, is_log=is_log
    )(torch.from_numpy(S).to(torch.float32))
    logmel_spec_librosa = logmel_spec_librosa.detach().cpu().numpy()

    np.testing.assert_allclose(logmel_spec, logmel_spec_librosa, atol=5e-3, rtol=1.3e-3)

    spec_aug_x, _ = SpecAugmentation(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        deterministic=False,
    ).init_with_output({"params": random.key(0)}, logmel_spec)

    kwargs = {
        "sr": sr,
        "n_mfcc": 13,
        "dct_type": 2,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "fmin": fmin,
        "fmax": fmax,
        "norm": "ortho",
        "lifter": 22,
    }

    assert S.ndim == 3
    mfcc_features, _ = MFCC(
        **kwargs,
    ).init_with_output({"params": random.key(0)}, S)
    mfcc_features_librosa = librosa.feature.mfcc(
        y=x,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode=pad_mode,
        **kwargs,
    )

    np.testing.assert_allclose(
        mfcc_features, mfcc_features_librosa, atol=6.6e-2, rtol=1.7e-1
    )


def test_drop_stripes():

    drop_stripes = DropStripes(axis=2, drop_width=2, stripes_num=2, deterministic=False)
    B, C, H, W = 2, 3, 9, 16
    x = jnp.ones((B, C, H, W))
    x, variables = drop_stripes.init_with_output({"params": random.key(0)}, x)
    print(x)
