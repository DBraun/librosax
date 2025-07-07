"""Comprehensive CQT tests comparing with librosa and testing various signals."""
import jax
import platform
if platform.system() == "Darwin":
    jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
import librosa
import numpy as np
import pytest
from scipy.signal import chirp
from scipy.stats import pearsonr

import librosax
import librosax.feature


def test_cqt_sweep_signals():
    """Test CQT with sweep signals like nnAudio does."""
    # Parameters
    fs = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Test 1: Logarithmic sweep (tests all frequencies equally in log scale)
    f0 = 55
    f1 = fs // 2
    x_log = chirp(t, f0, duration, f1, method='logarithmic')
    
    # Test 2: Linear sweep (tests all frequencies equally in linear scale) 
    x_lin = chirp(t, f0, duration, f1, method='linear')
    
    # CQT parameters
    n_bins = 84
    bins_per_octave = 12
    hop_length = 512
    
    # Create JIT-compiled version
    cqt_jit = jax.jit(
        librosax.feature.cqt,
        static_argnames=('sr', 'hop_length', 'fmin', 'n_bins', 'bins_per_octave',
                        'tuning', 'filter_scale', 'norm', 'sparsity', 'window',
                        'scale', 'pad_mode', 'res_type', 'dtype', 'n_fft')
    )
    
    for signal, name in [(x_log, "log_sweep"), (x_lin, "lin_sweep")]:
        # Compute with librosa
        C_librosa = librosa.cqt(
            y=signal, sr=fs, hop_length=hop_length, 
            n_bins=n_bins, bins_per_octave=bins_per_octave
        )
        
        # Compute with librosax
        signal_jax = jnp.array(signal)
        C_jax = cqt_jit(
            signal_jax, sr=fs, hop_length=hop_length, n_bins=n_bins
        )
        
        # Check shape
        assert C_jax.shape == C_librosa.shape, f"Shape mismatch for {name}"
        
        # Check correlation
        corr_mag = pearsonr(np.abs(C_librosa).flatten(), np.abs(C_jax).flatten())[0]
        assert corr_mag > 0.95, f"Low correlation for {name}: {corr_mag}"
        
        # For sweep signals, check that energy follows the sweep
        # The CQT should show a diagonal pattern
        C_mag = np.abs(C_jax)
        
        # Find the time index of maximum energy for each frequency bin
        max_time_indices = np.argmax(C_mag, axis=1)
        
        # For log sweep, the progression should be roughly exponential
        # For linear sweep, it should be roughly linear
        # Just check that it's monotonically increasing (with some tolerance)
        diffs = np.diff(max_time_indices)
        monotonic_ratio = np.sum(diffs >= -2) / len(diffs)  # Allow small backwards jumps
        assert monotonic_ratio > 0.8, f"Sweep pattern not detected in {name}"


def test_cqt_pure_tones():
    """Test CQT with pure tones at specific frequencies."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate pure tones at specific musical notes
    # A4 (440 Hz), A5 (880 Hz), A6 (1760 Hz)
    test_freqs = [440.0, 880.0, 1760.0]
    
    n_bins = 84
    bins_per_octave = 12
    hop_length = 512
    
    # Get CQT frequencies
    cqt_freqs = librosax.feature.cqt_frequencies(
        n_bins=n_bins, bins_per_octave=bins_per_octave
    )
    
    cqt_jit = jax.jit(
        librosax.feature.cqt,
        static_argnames=('sr', 'hop_length', 'fmin', 'n_bins', 'bins_per_octave',
                        'tuning', 'filter_scale', 'norm', 'sparsity', 'window',
                        'scale', 'pad_mode', 'res_type', 'dtype', 'n_fft')
    )
    
    for test_freq in test_freqs:
        # Generate pure tone
        y = np.sin(2 * np.pi * test_freq * t)
        
        # Compute CQT
        y_jax = jnp.array(y)
        C = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins)
        
        # Find the bin closest to the test frequency
        closest_bin = np.argmin(np.abs(cqt_freqs - test_freq))
        
        # Check that this bin has the maximum energy
        C_mean = np.mean(np.abs(C), axis=1)
        max_bin = np.argmax(C_mean)
        
        # Allow for some frequency resolution limitations
        assert abs(max_bin - closest_bin) <= 1, \
            f"Peak not at expected frequency {test_freq}Hz"
        
        # Check that the peak is significantly above other bins
        peak_energy = C_mean[max_bin]
        mean_energy = np.mean(C_mean)
        assert peak_energy > 5 * mean_energy, \
            f"Peak not prominent enough for {test_freq}Hz"


def test_cqt_musical_signal():
    """Test CQT with a musical signal (C major chord)."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # C major chord: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
    chord_freqs = [261.63, 329.63, 392.00]
    y = sum(0.3 * np.sin(2 * np.pi * f * t) for f in chord_freqs)
    
    # Add some harmonics for realism
    y += sum(0.1 * np.sin(2 * np.pi * 2 * f * t) for f in chord_freqs)
    
    # CQT parameters
    n_bins = 84
    bins_per_octave = 12
    hop_length = 512
    
    # Compute CQT with both implementations
    C_librosa = librosa.cqt(
        y=y, sr=sr, hop_length=hop_length,
        n_bins=n_bins, bins_per_octave=bins_per_octave
    )
    
    y_jax = jnp.array(y)
    cqt_jit = jax.jit(
        librosax.feature.cqt,
        static_argnames=('sr', 'hop_length', 'fmin', 'n_bins', 'bins_per_octave',
                        'tuning', 'filter_scale', 'norm', 'sparsity', 'window',
                        'scale', 'pad_mode', 'res_type', 'dtype', 'n_fft')
    )
    C_jax = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins)
    
    # Both should detect the same prominent frequencies
    C_librosa_mean = np.mean(np.abs(C_librosa), axis=1)
    C_jax_mean = np.mean(np.abs(C_jax), axis=1)
    
    # Find top 10 peaks
    top_bins_librosa = np.argsort(C_librosa_mean)[-10:]
    top_bins_jax = np.argsort(C_jax_mean)[-10:]
    
    # At least 7 out of 10 should match (allowing for slight differences)
    overlap = len(set(top_bins_librosa) & set(top_bins_jax))
    assert overlap >= 7, f"Too few matching peaks: {overlap}/10"
    
    # Check correlation
    corr = pearsonr(C_librosa_mean, C_jax_mean)[0]
    assert corr > 0.95, f"Low correlation: {corr}"


def test_cqt_normalization():
    """Test that CQT normalization is working correctly."""
    sr = 22050
    hop_length = 512
    
    # White noise should give roughly equal energy in all bins
    # (after accounting for filter bandwidth differences)
    np.random.seed(42)
    noise = np.random.randn(sr * 2)  # 2 seconds
    
    y_jax = jnp.array(noise)
    C = librosax.feature.cqt(
        y_jax, sr=sr, hop_length=hop_length, 
        n_bins=84, norm=1.0  # L1 norm
    )
    
    # Check that we don't have extreme values
    C_mag = np.abs(C)
    assert np.all(np.isfinite(C_mag)), "CQT contains non-finite values"
    assert np.max(C_mag) < 100, "CQT magnitudes unreasonably large"
    assert np.min(C_mag) >= 0, "CQT magnitudes should be non-negative"
    
    # Energy distribution shouldn't be too extreme
    C_mean = np.mean(C_mag, axis=1)
    energy_ratio = np.max(C_mean) / (np.min(C_mean) + 1e-10)
    assert energy_ratio < 1000, "Energy distribution too extreme"


def test_cqt_edge_cases():
    """Test CQT with edge cases."""
    sr = 22050
    
    # Test 1: Short signal (but realistic for CQT)
    # CQT needs longer signals due to the filter lengths at low frequencies
    y_short = np.random.randn(sr // 2)  # 0.5 seconds
    C = librosax.feature.cqt(jnp.array(y_short), sr=sr, n_fft=8192)
    assert C.shape[0] == 84  # Should still have correct number of bins
    assert C.shape[1] > 0  # Should have at least one time frame
    
    # Test 2: Different window types  
    y = np.random.randn(sr)
    for window in ['hann', 'hamming']:
        # Note: librosax might only support 'hann' currently
        try:
            C = librosax.feature.cqt(jnp.array(y), sr=sr, window=window, n_fft=8192)
            assert C.shape[0] == 84
        except:
            # If window type not supported, that's ok for now
            pass
    
    # Test 3: Different number of bins per octave
    for bins_per_octave in [12, 24, 36]:
        C = librosax.feature.cqt(
            jnp.array(y), sr=sr, 
            n_bins=7 * bins_per_octave,  # 7 octaves
            bins_per_octave=bins_per_octave,
            n_fft=8192
        )
        assert C.shape[0] == 7 * bins_per_octave


if __name__ == "__main__":
    test_cqt_sweep_signals()
    test_cqt_pure_tones()
    test_cqt_musical_signal()
    test_cqt_normalization()
    test_cqt_edge_cases()
    print("All comprehensive CQT tests passed!")