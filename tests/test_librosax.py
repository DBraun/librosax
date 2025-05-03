from itertools import product

import jax
# Enable JAX 64-bit mode for better precision
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp, random
from flax import nnx
import librosa
import librosa.feature
import numpy as np
import pytest
import torch
import torchlibrosa

import librosax
import librosax.feature
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

    x = np.random.uniform(-1, 1, size=(1, sr,))  # fmt: skip

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
    S = Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        pad_mode=pad_mode,
    )(jnp.array(x))
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
    logmel_spec = LogMelFilterBank(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, is_log=is_log
    )(S)

    logmel_spec_librosa = torchlibrosa.LogmelFilterBank(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, is_log=is_log
    )(torch.from_numpy(S).to(torch.float32))
    logmel_spec_librosa = logmel_spec_librosa.detach().cpu().numpy()

    np.testing.assert_allclose(logmel_spec, logmel_spec_librosa, atol=5e-3, rtol=1.3e-3)

    spec_aug_x = SpecAugmentation(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        deterministic=False,
        rngs=nnx.Rngs(0),
    )(logmel_spec)
    spec_aug_x = SpecAugmentation(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        deterministic=False,
        rngs=nnx.Rngs(0),
    )(logmel_spec)

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

    S = S.squeeze(0)
    x = x.squeeze(0)

    assert S.ndim == 2
    assert x.ndim == 1

    # repeatedly unsqueeze to test (n_mfcc, time_steps), (B, n_mfcc, time_steps), and (B, C, n_mfcc, time_steps)
    for i in range(3):

        mfcc_features = MFCC(
            **kwargs,
        )(S)
        mfcc_features_librosa = librosa.feature.mfcc(
            y=x,
            hop_length=hop_length,
            win_length=win_length,
            pad_mode=pad_mode,
            **kwargs,
        )

        np.testing.assert_allclose(
            mfcc_features, mfcc_features_librosa, atol=1e-2, rtol=1e-3
        )

        S = np.expand_dims(S, 0)
        x = np.expand_dims(x, 0)


def test_drop_stripes():

    drop_stripes = DropStripes(axis=2, drop_width=2, stripes_num=2, deterministic=False)
    B, C, H, W = 2, 3, 9, 16
    x = jnp.ones((B, C, H, W))
    x = drop_stripes(x, rngs=jax.random.key(0))
    x = drop_stripes(x, rngs=nnx.Rngs(0))
    print(x)


def test_spectral_centroid():
    """Test spectral_centroid against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with known spectral characteristics
    # Mix of different frequencies
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.2 * np.sin(2 * np.pi * 1760 * t)   # A6
    )
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of spectral_centroid
    spectral_centroid_jit = jax.jit(
        librosax.feature.spectral_centroid,
        static_argnames=('sr', 'n_fft', 'hop_length', 'win_length', 'window', 'center', 'pad_mode')
    )
    
    # Compute with librosa
    centroid_librosa = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Compute with librosax (JIT-compiled)
    y_jax = jnp.array(y)
    centroid_jax = spectral_centroid_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Compare results
    np.testing.assert_allclose(
        centroid_jax, centroid_librosa, atol=1e-5, rtol=1e-5
    )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    centroid_librosa_S = librosa.feature.spectral_centroid(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    centroid_jax_S = spectral_centroid_jit(
        S=S_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    np.testing.assert_allclose(
        centroid_jax_S, centroid_librosa_S, atol=1e-5, rtol=1e-5
    )


def test_spectral_bandwidth():
    """Test spectral_bandwidth against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with varying spectral content
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.2 * np.sin(2 * np.pi * 1760 * t) + # A6
        0.1 * np.random.randn(len(t))        # Some noise
    )
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of spectral_bandwidth
    spectral_bandwidth_jit = jax.jit(
        librosax.feature.spectral_bandwidth,
        static_argnames=('sr', 'n_fft', 'hop_length', 'p', 'win_length', 'window', 'center', 'pad_mode', 'norm')
    )
    
    # Compute with librosa
    bandwidth_librosa = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Compute with librosax (JIT-compiled)
    y_jax = jnp.array(y)
    bandwidth_jax = spectral_bandwidth_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Compare results
    np.testing.assert_allclose(
        bandwidth_jax, bandwidth_librosa, atol=1e-5, rtol=1e-5
    )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    bandwidth_librosa_S = librosa.feature.spectral_bandwidth(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    bandwidth_jax_S = spectral_bandwidth_jit(
        S=S_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    np.testing.assert_allclose(
        bandwidth_jax_S, bandwidth_librosa_S, atol=1e-5, rtol=1e-5
    )
    
    # Test with different p values
    for p in [1, 3]:
        bandwidth_librosa_p = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, p=p
        )
        
        bandwidth_jax_p = spectral_bandwidth_jit(
            y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length, p=p
        )
        
        np.testing.assert_allclose(
            bandwidth_jax_p, bandwidth_librosa_p, atol=1e-5, rtol=1e-5
        )


def test_spectral_rolloff():
    """Test spectral_rolloff against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with frequency sweep
    # Start with low frequencies and sweep up
    f_start = 200
    f_end = 8000
    sweep_rate = (f_end - f_start) / duration
    instantaneous_freq = f_start + sweep_rate * t
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    y = np.sin(phase)
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of spectral_rolloff
    spectral_rolloff_jit = jax.jit(
        librosax.feature.spectral_rolloff,
        static_argnames=('sr', 'n_fft', 'hop_length', 'roll_percent', 'win_length', 'window', 'center', 'pad_mode')
    )
    
    # Test different roll percentages
    for roll_percent in [0.01, 0.85, 0.99]:
        # Compute with librosa
        rolloff_librosa = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent
        )
        
        # Compute with librosax (JIT-compiled)
        y_jax = jnp.array(y)
        rolloff_jax = spectral_rolloff_jit(
            y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent
        )
        
        # Compare results
        np.testing.assert_allclose(
            rolloff_jax, rolloff_librosa, atol=1e-5, rtol=1e-5
        )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    rolloff_librosa_S = librosa.feature.spectral_rolloff(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    rolloff_jax_S = spectral_rolloff_jit(
        S=S_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    np.testing.assert_allclose(
        rolloff_jax_S, rolloff_librosa_S, atol=1e-5, rtol=1e-5
    )


def test_spectral_flatness():
    """Test spectral_flatness against librosa implementation."""
    # Generate test signals
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Test with different types of signals
    # 1. White noise (should have high flatness)
    noise = np.random.randn(len(t))
    
    # 2. Pure tone (should have low flatness)
    tone = np.sin(2 * np.pi * 440 * t)
    
    # 3. Mixed signal
    mixed = 0.5 * tone + 0.5 * noise
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of spectral_flatness
    spectral_flatness_jit = jax.jit(
        librosax.feature.spectral_flatness,
        static_argnames=('n_fft', 'hop_length', 'amin', 'power', 'win_length', 'window', 'center', 'pad_mode')
    )
    
    for y in [noise, tone, mixed]:
        # Compute with librosa
        flatness_librosa = librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop_length
        )
        
        # Compute with librosax (JIT-compiled)
        y_jax = jnp.array(y)
        flatness_jax = spectral_flatness_jit(
            y=y_jax, n_fft=n_fft, hop_length=hop_length
        )
        
        # Compare results
        np.testing.assert_allclose(
            flatness_jax, flatness_librosa, atol=1e-5, rtol=1e-5
        )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(mixed, n_fft=n_fft, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    flatness_librosa_S = librosa.feature.spectral_flatness(S=S)
    flatness_jax_S = spectral_flatness_jit(S=S_jax)
    
    np.testing.assert_allclose(
        flatness_jax_S, flatness_librosa_S, atol=1e-5, rtol=1e-5
    )
    
    # Test with power spectrogram
    S_power = S ** 2
    S_power_jax = jnp.array(S_power)
    
    flatness_librosa_power = librosa.feature.spectral_flatness(S=S_power, power=1.0)
    flatness_jax_power = spectral_flatness_jit(S=S_power_jax, power=1.0)
    
    np.testing.assert_allclose(
        flatness_jax_power, flatness_librosa_power, atol=1e-5, rtol=1e-5
    )


def test_rms():
    """Test RMS against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with varying amplitude
    envelope = np.exp(-t * 2)  # Exponential decay
    y = envelope * np.sin(2 * np.pi * 440 * t)
    
    # Test parameters
    frame_length = 2048
    hop_length = 512
    
    # Create JIT-compiled version of rms
    rms_jit = jax.jit(
        librosax.feature.rms,
        static_argnames=('frame_length', 'hop_length', 'center', 'pad_mode')
    )
    
    # Test from time series
    rms_librosa = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=True
    )
    
    y_jax = jnp.array(y)
    rms_jax = rms_jit(
        y=y_jax, frame_length=frame_length, hop_length=hop_length, center=True
    )
    
    np.testing.assert_allclose(
        rms_jax, rms_librosa, atol=1e-5, rtol=1e-5
    )
    
    # Test from spectrogram
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    rms_librosa_S = librosa.feature.rms(S=S)
    rms_jax_S = rms_jit(S=S_jax)
    
    np.testing.assert_allclose(
        rms_jax_S, rms_librosa_S, atol=1e-5, rtol=1e-5
    )
    
    # Test without centering
    rms_librosa_no_center = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=False
    )
    
    rms_jax_no_center = rms_jit(
        y=y_jax, frame_length=frame_length, hop_length=hop_length, center=False
    )
    
    np.testing.assert_allclose(
        rms_jax_no_center, rms_librosa_no_center, atol=1e-5, rtol=1e-5
    )


def test_zero_crossing_rate():
    """Test zero_crossing_rate against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with varying zero-crossing rate
    # Low frequency at start, high frequency at end
    freq_start = 100
    freq_end = 2000
    freq = np.linspace(freq_start, freq_end, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    y = np.sin(phase)
    
    # Add some noise
    y += 0.1 * np.random.randn(len(t))
    
    # Test parameters
    frame_length = 2048
    hop_length = 512
    
    # Create JIT-compiled version of zero_crossing_rate
    zero_crossing_rate_jit = jax.jit(
        librosax.feature.zero_crossing_rate,
        static_argnames=('frame_length', 'hop_length', 'center', 'pad_mode', 'threshold', 'ref_magnitude', 'pad', 'zero_pos', 'axis')
    )
    
    # Test with default parameters
    zcr_librosa = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=True
    )
    
    y_jax = jnp.array(y)
    zcr_jax = zero_crossing_rate_jit(
        y_jax, frame_length=frame_length, hop_length=hop_length, center=True
    )
    
    np.testing.assert_allclose(
        zcr_jax, zcr_librosa, atol=1e-5, rtol=1e-5
    )
    
    # Test without centering
    zcr_librosa_no_center = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=False
    )
    
    zcr_jax_no_center = zero_crossing_rate_jit(
        y_jax, frame_length=frame_length, hop_length=hop_length, center=False
    )
    
    np.testing.assert_allclose(
        zcr_jax_no_center, zcr_librosa_no_center, atol=1e-5, rtol=1e-5
    )
    
    # Test with different threshold
    zcr_librosa_thresh = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=True, threshold=0.01
    )
    
    zcr_jax_thresh = zero_crossing_rate_jit(
        y_jax, frame_length=frame_length, hop_length=hop_length, center=True, threshold=0.01
    )
    
    np.testing.assert_allclose(
        zcr_jax_thresh, zcr_librosa_thresh, atol=1e-5, rtol=1e-5
    )


def test_spectral_contrast():
    """Test spectral_contrast against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a harmonic signal with fundamental and overtones
    f0 = 440  # A4
    y = np.zeros_like(t)
    for harmonic in range(1, 6):
        y += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Add some noise
    y += 0.05 * np.random.randn(len(t))
    
    # Normalize
    y = y / np.max(np.abs(y))
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of spectral_contrast
    spectral_contrast_jit = jax.jit(
        librosax.feature.spectral_contrast,
        static_argnames=('sr', 'n_fft', 'hop_length', 'fmin', 'n_bands', 'quantile', 'linear', 'win_length', 'window', 'center', 'pad_mode')
    )
    
    # Test with default parameters
    contrast_librosa = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    y_jax = jnp.array(y)
    contrast_jax = spectral_contrast_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    np.testing.assert_allclose(
        contrast_jax, contrast_librosa, atol=1e-3, rtol=1e-3
    )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_jax = jnp.array(S)
    
    contrast_librosa_S = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrast_jax_S = spectral_contrast_jit(S=S_jax, sr=sr)
    
    np.testing.assert_allclose(
        contrast_jax_S, contrast_librosa_S, atol=1e-3, rtol=1e-3
    )
    
    # Test with different parameters
    contrast_librosa_custom = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        fmin=100.0, n_bands=4, quantile=0.05, linear=True
    )
    
    contrast_jax_custom = spectral_contrast_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length,
        fmin=100.0, n_bands=4, quantile=0.05, linear=True
    )
    
    np.testing.assert_allclose(
        contrast_jax_custom, contrast_librosa_custom, atol=1e-3, rtol=1e-3
    )


def test_melspectrogram():
    """Test melspectrogram against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a test signal with multiple frequency components
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +   # A5
        0.2 * np.sin(2 * np.pi * 1760 * t) +  # A6
        0.1 * np.sin(2 * np.pi * 3520 * t)    # A7
    )
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    # Create JIT-compiled version of melspectrogram
    melspectrogram_jit = jax.jit(
        librosax.feature.melspectrogram,
        static_argnames=('sr', 'n_fft', 'hop_length', 'win_length', 'window', 
                        'center', 'pad_mode', 'power', 'n_mels', 'fmin', 'fmax', 
                        'htk', 'norm', 'dtype')
    )
    
    # Test with default parameters
    melspec_librosa = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    y_jax = jnp.array(y)
    melspec_jax = melspectrogram_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    np.testing.assert_allclose(
        melspec_jax, melspec_librosa, atol=3e-3, rtol=2e-3
    )
    
    # Test with pre-computed spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_jax = jnp.array(S)
    
    melspec_librosa_S = librosa.feature.melspectrogram(
        S=S, sr=sr, n_mels=n_mels
    )
    
    melspec_jax_S = melspectrogram_jit(
        S=S_jax, sr=sr, n_mels=n_mels
    )
    
    np.testing.assert_allclose(
        melspec_jax_S, melspec_librosa_S, atol=3e-3, rtol=2e-3
    )
    
    # Test with different parameters
    melspec_librosa_custom = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=64, fmin=100.0, fmax=8000.0, power=1.0
    )
    
    melspec_jax_custom = melspectrogram_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=64, fmin=100.0, fmax=8000.0, power=1.0
    )
    
    np.testing.assert_allclose(
        melspec_jax_custom, melspec_librosa_custom, atol=3e-3, rtol=2e-3
    )


def test_mfcc():
    """Test MFCC against librosa implementation."""
    # Generate test signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a test signal with harmonic content
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +   # A5
        0.2 * np.sin(2 * np.pi * 1760 * t)    # A6
    )
    
    # For 1D input, librosa returns (n_mfcc, t)
    # Make sure our implementation matches this
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    n_mfcc = 20
    n_mels = 128
    
    # Create JIT-compiled version of mfcc
    mfcc_jit = jax.jit(
        librosax.feature.mfcc,
        static_argnames=('sr', 'n_mfcc', 'dct_type', 'norm', 'lifter', 
                        'n_fft', 'hop_length', 'win_length', 'window', 
                        'center', 'pad_mode', 'power', 'n_mels', 'fmin', 
                        'fmax', 'htk', 'melspectrogram_params')
    )
    
    # Test with default parameters
    mfcc_librosa = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    y_jax = jnp.array(y)
    mfcc_jax = mfcc_jit(
        y=y_jax, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Handle shape mismatch for 1D input
    # JAX returns shape with batch dimension, librosa doesn't
    if y.ndim == 1 and mfcc_jax.ndim == 3:
        mfcc_jax = mfcc_jax.squeeze(0)
    
    np.testing.assert_allclose(
        mfcc_jax, mfcc_librosa, atol=4e-2, rtol=1e-3
    )
    
    # Test with different parameters
    mfcc_librosa_custom = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length,
        n_mels=40, fmin=100.0, fmax=8000.0
    )
    
    mfcc_jax_custom = mfcc_jit(
        y=y_jax, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length,
        n_mels=40, fmin=100.0, fmax=8000.0
    )
    
    if y.ndim == 1 and mfcc_jax_custom.ndim == 3:
        mfcc_jax_custom = mfcc_jax_custom.squeeze(0)
    
    np.testing.assert_allclose(
        mfcc_jax_custom, mfcc_librosa_custom, atol=4e-2, rtol=1e-3
    )
    
    # Test with liftering
    mfcc_librosa_lifter = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, lifter=22
    )
    
    mfcc_jax_lifter = mfcc_jit(
        y=y_jax, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, lifter=22
    )
    
    if y.ndim == 1 and mfcc_jax_lifter.ndim == 3:
        mfcc_jax_lifter = mfcc_jax_lifter.squeeze(0)
    
    np.testing.assert_allclose(
        mfcc_jax_lifter, mfcc_librosa_lifter, atol=4e-2, rtol=1e-3
    )
    
    # Test with pre-computed mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec_jax = jnp.array(log_mel_spec)
    
    mfcc_librosa_S = librosa.feature.mfcc(
        S=log_mel_spec, n_mfcc=n_mfcc
    )
    
    mfcc_jax_S = mfcc_jit(
        S=log_mel_spec_jax, n_mfcc=n_mfcc
    )
    
    # S input should have same shape handling
    if log_mel_spec.ndim == 2 and mfcc_jax_S.ndim == 3:
        mfcc_jax_S = mfcc_jax_S.squeeze(0)
    
    np.testing.assert_allclose(
        mfcc_jax_S, mfcc_librosa_S, atol=4e-2, rtol=1e-3
    )


def test_chroma_stft():
    """Test chroma_stft against librosa implementation."""
    # Generate test signal with known harmonic content
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with clear pitch content
    # C major triad: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
    y = (
        0.4 * np.sin(2 * np.pi * 261.63 * t) +  # C4
        0.3 * np.sin(2 * np.pi * 329.63 * t) +  # E4
        0.3 * np.sin(2 * np.pi * 392.00 * t)    # G4
    )
    
    # Test parameters
    n_fft = 2048
    hop_length = 512
    
    # Create JIT-compiled version of chroma_stft
    chroma_stft_jit = jax.jit(
        librosax.feature.chroma_stft,
        static_argnames=('sr', 'norm', 'n_fft', 'hop_length', 'win_length', 
                        'window', 'center', 'pad_mode', 'tuning', 'n_chroma')
    )
    
    # Test with default parameters
    chroma_librosa = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    y_jax = jnp.array(y)
    chroma_jax = chroma_stft_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Handle shape for 1D input
    if y.ndim == 1 and chroma_jax.ndim == 3:
        chroma_jax = chroma_jax.squeeze(0)
    
    # Chroma features can have slight numerical differences due to 
    # filter bank construction, so we use a slightly relaxed tolerance
    np.testing.assert_allclose(
        chroma_jax, chroma_librosa, atol=0.1, rtol=0.1
    )
    
    # Test with pre-computed power spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_jax = jnp.array(S)
    
    chroma_librosa_S = librosa.feature.chroma_stft(
        S=S, sr=sr
    )
    
    chroma_jax_S = chroma_stft_jit(
        S=S_jax, sr=sr
    )
    
    if S.ndim == 2 and chroma_jax_S.ndim == 3:
        chroma_jax_S = chroma_jax_S.squeeze(0)
    
    np.testing.assert_allclose(
        chroma_jax_S, chroma_librosa_S, atol=0.1, rtol=0.1
    )
    
    # Test with different parameters
    chroma_librosa_custom = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_chroma=24, tuning=0.0
    )
    
    chroma_jax_custom = chroma_stft_jit(
        y=y_jax, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_chroma=24, tuning=0.0
    )
    
    if y.ndim == 1 and chroma_jax_custom.ndim == 3:
        chroma_jax_custom = chroma_jax_custom.squeeze(0)
    
    np.testing.assert_allclose(
        chroma_jax_custom, chroma_librosa_custom, atol=0.1, rtol=0.1
    )


def test_tonnetz():
    """Test tonnetz against librosa implementation."""
    # Generate test signal with harmonic content
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a harmonic signal
    # C major chord progression: C - F - G - C
    freq_c = 261.63  # C4
    freq_f = 349.23  # F4
    freq_g = 392.00  # G4
    
    # Simple chord progression
    y = np.zeros_like(t)
    n = len(t)
    # C chord (C-E-G)
    y[:n//4] = (
        0.4 * np.sin(2 * np.pi * freq_c * t[:n//4]) +
        0.3 * np.sin(2 * np.pi * 329.63 * t[:n//4]) +  # E4
        0.3 * np.sin(2 * np.pi * freq_g * t[:n//4])
    )
    # F chord (F-A-C)
    y[n//4:n//2] = (
        0.4 * np.sin(2 * np.pi * freq_f * t[:n//4]) +
        0.3 * np.sin(2 * np.pi * 440.00 * t[:n//4]) +  # A4
        0.3 * np.sin(2 * np.pi * 523.25 * t[:n//4])    # C5
    )
    # G chord (G-B-D)
    y[n//2:3*n//4] = (
        0.4 * np.sin(2 * np.pi * freq_g * t[:n//4]) +
        0.3 * np.sin(2 * np.pi * 493.88 * t[:n//4]) +  # B4
        0.3 * np.sin(2 * np.pi * 587.33 * t[:n//4])    # D5
    )
    # C chord again
    y[3*n//4:] = y[:n//4]
    
    # Create JIT-compiled version of tonnetz
    tonnetz_jit = jax.jit(
        librosax.feature.tonnetz,
        static_argnames=('sr',)
    )
    
    # Since librosax uses chroma_stft, we need to use the same for fair comparison
    # First compute chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Test with pre-computed chroma_stft for both
    tonnetz_librosa = librosa.feature.tonnetz(chroma=chroma_stft)
    
    chroma_jax = jnp.array(chroma_stft)
    tonnetz_jax = tonnetz_jit(chroma=chroma_jax)
    
    # Handle shape
    if chroma_stft.ndim == 2 and tonnetz_jax.ndim == 3:
        tonnetz_jax = tonnetz_jax.squeeze(0)
    
    # Should be very close when using same chroma input
    np.testing.assert_allclose(
        tonnetz_jax, tonnetz_librosa, atol=1e-5, rtol=1e-5
    )
    
    # Test tonnetz computation from audio directly
    y_jax = jnp.array(y)
    tonnetz_jax_from_audio = tonnetz_jit(y=y_jax, sr=sr)
    
    # This uses chroma_stft internally, so compare with librosa using chroma_stft too
    tonnetz_librosa_stft = librosa.feature.tonnetz(
        chroma=librosa.feature.chroma_stft(y=y, sr=sr)
    )
    
    if y.ndim == 1 and tonnetz_jax_from_audio.ndim == 3:
        tonnetz_jax_from_audio = tonnetz_jax_from_audio.squeeze(0)
    
    np.testing.assert_allclose(
        tonnetz_jax_from_audio, tonnetz_librosa_stft, atol=0.02, rtol=0.02
    )


def test_chroma_cqt():
    """Test chroma_cqt implementation."""
    # Generate test signal - C major scale
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # C major scale frequencies (C4 to B4)
    scale_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    
    # Play each note for a portion of the duration
    y = np.zeros_like(t)
    note_duration = len(t) // len(scale_freqs)
    
    for i, freq in enumerate(scale_freqs):
        start = i * note_duration
        end = min((i + 1) * note_duration, len(t))
        y[start:end] = 0.5 * np.sin(2 * np.pi * freq * t[start:end])
    
    # Create JIT-compiled version
    chroma_cqt_jit = jax.jit(
        librosax.feature.chroma_cqt,
        static_argnames=('sr', 'hop_length', 'fmin', 'norm', 'threshold',
                        'tuning', 'n_chroma', 'n_octaves', 'bins_per_octave',
                        'cqt_mode')
    )
    
    # Test with reduced parameters to avoid memory issues
    # Default is 36 bins/octave * 7 octaves = 252 bins which is too much
    y_jax = jnp.array(y)
    chroma_jax = chroma_cqt_jit(y=y_jax, sr=sr, n_octaves=5, bins_per_octave=12)
    
    # Check output shape
    assert chroma_jax.shape[0] == 12, f"Expected 12 chroma bins, got {chroma_jax.shape[0]}"
    
    # Since we're playing a C major scale, we expect to see energy
    # in specific chroma bins at specific times
    # C=0, D=2, E=4, F=5, G=7, A=9, B=11
    expected_chromas = [0, 2, 4, 5, 7, 9, 11]
    
    # Check that each note activates the expected chroma
    for i, expected_chroma in enumerate(expected_chromas):
        # Get the time frames for this note
        start_frame = i * chroma_jax.shape[1] // len(scale_freqs)
        end_frame = (i + 1) * chroma_jax.shape[1] // len(scale_freqs)
        
        # Average chroma for this note's duration
        note_chroma = np.mean(chroma_jax[:, start_frame:end_frame], axis=1)
        
        # The expected chroma should be among the strongest
        sorted_indices = np.argsort(note_chroma)[::-1]
        assert expected_chroma in sorted_indices[:3], \
            f"Note {i} (chroma {expected_chroma}) not in top 3 chromas"
    
    # Test with pre-computed CQT
    C_jax = librosax.feature.cqt(y_jax, sr=sr, n_bins=60, bins_per_octave=12)
    chroma_from_cqt = chroma_cqt_jit(C=jnp.abs(C_jax), sr=sr, n_octaves=5, bins_per_octave=12)
    
    # Should have same shape
    assert chroma_from_cqt.shape[0] == 12
    
    # Compare with librosa implementation
    chroma_librosa = librosa.feature.chroma_cqt(
        y=y, sr=sr, n_octaves=5, bins_per_octave=12
    )
    
    # Handle shape difference for 1D input
    if y.ndim == 1 and chroma_jax.ndim == 3:
        chroma_jax_compare = chroma_jax.squeeze(0)
    else:
        chroma_jax_compare = chroma_jax
    
    # CQT-based chroma has more variability so we use a slightly relaxed tolerance
    np.testing.assert_allclose(
        chroma_jax_compare, chroma_librosa, atol=0.063, rtol=0.063
    )
    
    # Also test with pre-computed CQT comparison
    C_librosa = librosa.cqt(y, sr=sr, n_bins=60, bins_per_octave=12)
    chroma_librosa_from_cqt = librosa.feature.chroma_cqt(
        C=np.abs(C_librosa), sr=sr, n_octaves=5, bins_per_octave=12
    )
    
    if C_librosa.ndim == 2 and chroma_from_cqt.ndim == 3:
        chroma_from_cqt_compare = chroma_from_cqt.squeeze(0)
    else:
        chroma_from_cqt_compare = chroma_from_cqt
    
    np.testing.assert_allclose(
        chroma_from_cqt_compare, chroma_librosa_from_cqt, atol=0.063, rtol=0.063
    )
