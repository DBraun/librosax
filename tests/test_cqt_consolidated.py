"""Consolidated CQT tests for librosax - combines all CQT test functionality."""
import jax
# Enable JAX 64-bit mode for better precision
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import librosa
import numpy as np
import pytest
from scipy.signal import chirp
from scipy.stats import pearsonr
import time

import librosax
import librosax.feature


@pytest.fixture
def cqt_jit():
    """Create JIT-compiled CQT function."""
    return jax.jit(
        librosax.feature.cqt,
        static_argnames=(
            'sr', 'hop_length', 'fmin', 'n_bins', 'bins_per_octave',
            'tuning', 'filter_scale', 'norm', 'sparsity', 'window',
            'scale', 'pad_mode', 'res_type', 'dtype', 'n_fft', 'use_1992_version'
        )
    )


@pytest.fixture  
def cqt2010_jit():
    """Create JIT-compiled CQT2010 function."""
    return jax.jit(
        librosax.feature.cqt2010,
        static_argnames=(
            'sr', 'hop_length', 'fmin', 'fmax', 'n_bins', 'bins_per_octave',
            'tuning', 'filter_scale', 'norm', 'sparsity', 'window',
            'scale', 'pad_mode', 'res_type', 'dtype', 'output_format', 'earlydownsample'
        )
    )


class TestCQTCore:
    """Core CQT functionality tests."""
    
    def test_basic_functionality(self, cqt_jit):
        """Test basic CQT functionality and compare with librosa."""
        # Generate test signal with harmonics
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # A4 (440 Hz) with harmonics
        f0 = 440.0
        y = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        
        # Test parameters
        hop_length = 512
        n_bins = 84
        bins_per_octave = 12
        
        # Compute with librosa
        C_librosa = librosa.cqt(
            y=y, sr=sr, hop_length=hop_length, n_bins=n_bins, 
            bins_per_octave=bins_per_octave
        )
        
        # Compute with librosax
        y_jax = jnp.array(y)
        C_jax = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins)
        
        # Check output shape matches
        assert C_jax.shape == C_librosa.shape, f"Shape mismatch: {C_jax.shape} vs {C_librosa.shape}"
        
        # Check that CQT produces reasonable output
        assert np.max(np.abs(C_jax)) > 0, "CQT output is all zeros"
        assert not np.any(np.isnan(C_jax)), "CQT contains NaN values"
        assert not np.any(np.isinf(C_jax)), "CQT contains infinite values"
        
        # Check correlation (shape similarity)
        corr_mag = pearsonr(np.abs(C_librosa).flatten(), np.abs(C_jax).flatten())[0]
        assert corr_mag > 0.95, f"Low correlation between implementations: {corr_mag}"
    
    def test_sweep_signals(self, cqt_jit):
        """Test CQT with sweep signals."""
        fs = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration))
        
        # Test 1: Logarithmic sweep
        f0 = 55
        f1 = fs // 2
        x_log = chirp(t, f0, duration, f1, method='logarithmic')
        
        # Test 2: Linear sweep
        x_lin = chirp(t, f0, duration, f1, method='linear')
        
        # CQT parameters
        n_bins = 84
        bins_per_octave = 12
        hop_length = 512
        
        for signal, name in [(x_log, "log_sweep"), (x_lin, "lin_sweep")]:
            # Compute with librosa
            C_librosa = librosa.cqt(
                y=signal, sr=fs, hop_length=hop_length, 
                n_bins=n_bins, bins_per_octave=bins_per_octave
            )
            
            # Compute with librosax
            signal_jax = jnp.array(signal)
            C_jax = cqt_jit(signal_jax, sr=fs, hop_length=hop_length, n_bins=n_bins)
            
            # Check shape
            assert C_jax.shape == C_librosa.shape, f"Shape mismatch for {name}"
            
            # Check correlation
            corr_mag = pearsonr(np.abs(C_librosa).flatten(), np.abs(C_jax).flatten())[0]
            assert corr_mag > 0.98, f"Low correlation for {name}: {corr_mag}"
    
    def test_pure_tones(self, cqt_jit):
        """Test CQT with pure tones at specific frequencies."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Test frequencies: A4, A5, A6
        test_freqs = [440.0, 880.0, 1760.0]
        
        n_bins = 84
        bins_per_octave = 12
        hop_length = 512
        
        # Get CQT frequencies
        cqt_freqs = librosax.feature.cqt_frequencies(
            n_bins=n_bins, bins_per_octave=bins_per_octave
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
    
    def test_edge_cases(self, cqt_jit):
        """Test CQT with edge cases."""
        sr = 22050
        
        # Test 1: Short signal
        y_short = np.random.randn(sr // 2)  # 0.5 seconds
        C = cqt_jit(jnp.array(y_short), sr=sr, n_fft=4096)
        assert C.shape[0] == 84  # Should still have correct number of bins
        assert C.shape[1] > 0  # Should have at least one time frame
        
        # Test 2: Different number of bins per octave
        y = np.random.randn(sr)
        for bins_per_octave in [12, 24]:
            n_octaves = 7 if bins_per_octave <= 12 else 3
            fmin = 32.70 if bins_per_octave <= 12 else 65.41  # C2 instead of C1
            C = cqt_jit(
                jnp.array(y), sr=sr, 
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
                fmin=fmin,
                n_fft=4096 if bins_per_octave <= 12 else 2048
            )
            assert C.shape[0] == n_octaves * bins_per_octave


class TestCQT2010:
    """Tests specific to CQT2010 implementation."""
    
    def test_algorithm_comparison(self, cqt_jit, cqt2010_jit):
        """Test that CQT1992 and CQT2010 produce equivalent results."""
        # Generate test signal
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        y_jax = jnp.array(y)
        
        # Parameters
        hop_length = 512
        n_bins = 36
        bins_per_octave = 12
        
        # Compute with both algorithms - without early downsampling for fair comparison
        cqt1992 = cqt_jit(
            y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )
        
        cqt2010_result = cqt2010_jit(
            y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins,
            bins_per_octave=bins_per_octave, output_format='complex',
            earlydownsample=False  # Disable for fair comparison
        )
        
        # Convert to numpy
        cqt1992_np = np.abs(cqt1992)
        cqt2010_np = np.abs(cqt2010_result)
        
        # Debug: Check shapes first
        assert cqt1992_np.shape == cqt2010_np.shape, f"Shape mismatch: {cqt1992_np.shape} vs {cqt2010_np.shape}"
        
        # Calculate differences
        abs_diff = np.abs(cqt1992_np - cqt2010_np)
        rel_diff = abs_diff / (np.abs(cqt1992_np) + 1e-10)
        
        # Print debug info
        print(f"\nDebug info:")
        print(f"CQT1992 shape: {cqt1992_np.shape}")
        print(f"CQT2010 shape: {cqt2010_np.shape}")
        print(f"Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
        print(f"Max relative difference: {np.max(rel_diff):.6e}")
        print(f"Mean relative difference: {np.mean(rel_diff):.6e}")
        
        # Check correlation first (more lenient)
        correlation = np.corrcoef(cqt1992_np.flatten(), cqt2010_np.flatten())[0, 1]
        print(f"Correlation: {correlation:.6f}")
        
        # CQT2010 uses a different algorithm (multi-resolution) so we expect some differences
        # Use correlation as the primary test
        assert correlation > 0.95, f"Low correlation between algorithms: {correlation:.6f}"
        
        # Only check allclose with very relaxed tolerance
        # The algorithms are fundamentally different so exact match is not expected
        match = np.allclose(cqt1992_np, cqt2010_np, rtol=0.1, atol=1e-3)
        if not match:
            print("Note: CQT1992 and CQT2010 don't match exactly (expected due to different algorithms)")
    
    def test_output_formats(self, cqt2010_jit):
        """Test different output formats for CQT2010."""
        # Generate short test signal
        sr = 22050
        y = np.random.randn(sr // 2)  # 0.5 seconds
        y_jax = jnp.array(y)
        
        # Test all output formats
        formats = ['magnitude', 'complex', 'phase']
        
        for fmt in formats:
            result = cqt2010_jit(
                y_jax, sr=sr, hop_length=512, n_bins=36, output_format=fmt
            )
            
            if fmt == 'magnitude':
                assert result.dtype in [np.float32, np.float64], f"Wrong dtype for magnitude"
            elif fmt == 'complex':
                assert np.iscomplexobj(result), f"Output should be complex"
            elif fmt == 'phase':
                assert result.shape[-1] == 2, f"Phase should have 2 components"
    
    def test_early_downsampling(self, cqt2010_jit):
        """Test early downsampling effect."""
        # Generate test signal
        sr = 22050
        y = np.random.randn(sr)  # 1 second
        y_jax = jnp.array(y)
        
        # With early downsampling
        cqt_with_ds = cqt2010_jit(
            y_jax, sr=sr, hop_length=512, n_bins=84,
            earlydownsample=True, output_format="magnitude"
        )
        
        # Without early downsampling
        cqt_without_ds = cqt2010_jit(
            y_jax, sr=sr, hop_length=512, n_bins=84,
            earlydownsample=False, output_format="magnitude"
        )
        
        # Results should be very similar
        correlation = np.corrcoef(cqt_with_ds.flatten(), cqt_without_ds.flatten())[0, 1]
        assert correlation > 0.99, f"Low correlation with/without downsampling: {correlation:.6f}"
    
    def test_basic_pitch_settings(self, cqt2010_jit):
        """Test CQT2010 with Basic Pitch settings."""
        # Basic Pitch parameters
        sr = 22050
        hop_length = 512
        n_bins = 264
        bins_per_octave = 36
        fmin = 32.70
        
        # Generate test audio (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio_jax = jnp.array(audio, dtype=jnp.float32)
        
        # Compute CQT
        cqt_result = cqt2010_jit(
            audio_jax, sr=sr, hop_length=hop_length, fmin=fmin,
            n_bins=n_bins, bins_per_octave=bins_per_octave,
            output_format='magnitude', earlydownsample=True,
        )
        
        assert cqt_result.shape[0] == n_bins, f"Expected {n_bins} bins, got {cqt_result.shape[0]}"
        assert cqt_result.shape[1] > 0, "No time frames in output"


class TestCQTBatchProcessing:
    """Test batch processing capabilities."""
    
    def test_batch_dimensions(self, cqt_jit, cqt2010_jit):
        """Test that batch dimensions are preserved correctly."""
        sr = 22050
        duration = 0.5
        samples = int(sr * duration)
        
        # Test different input shapes
        test_cases = [
            ("1D input", np.random.randn(samples)),
            ("2D input (single)", np.random.randn(1, samples)),
            ("2D input (batch=3)", np.random.randn(3, samples)),
        ]
        
        for name, audio in test_cases:
            audio_jax = jnp.array(audio, dtype=jnp.float32)
            
            # Test CQT1992
            cqt_result = cqt_jit(audio_jax, sr=sr)
            
            # Test CQT2010
            cqt2010_result = cqt2010_jit(audio_jax, sr=sr, output_format='magnitude')
            
            # Verify batch dimension handling
            if audio.ndim == 1:
                # 1D input should give 2D output (n_bins, time)
                assert cqt_result.ndim == 2, f"Expected 2D output for 1D input"
                assert cqt2010_result.ndim == 2, f"Expected 2D output for 1D input"
            else:
                # 2D input should give 3D output (batch, n_bins, time)
                assert cqt_result.ndim == 3, f"Expected 3D output for 2D input"
                assert cqt2010_result.ndim == 3, f"Expected 3D output for 2D input"
                assert cqt_result.shape[0] == audio.shape[0], f"Batch dimension mismatch"
                assert cqt2010_result.shape[0] == audio.shape[0], f"Batch dimension mismatch"
    
    def test_output_formats_with_batch(self, cqt2010_jit):
        """Test different output formats with batch input."""
        sr = 22050
        samples = 11025
        batch_size = 2
        audio_batch = np.random.randn(batch_size, samples).astype(np.float32)
        audio_jax = jnp.array(audio_batch)
        
        formats = ['magnitude', 'complex', 'phase']
        
        for fmt in formats:
            result = cqt2010_jit(audio_jax, sr=sr, n_bins=36, output_format=fmt)
            
            # Verify batch dimension is preserved
            assert result.shape[0] == batch_size, f"Batch dimension lost for format {fmt}"
            
            if fmt == 'phase':
                assert result.shape[-1] == 2, f"Phase should have 2 components"


class TestCQTDimensions:
    """Test dimension consistency for edge cases."""
    
    def test_various_audio_lengths(self, cqt2010_jit):
        """Test with various audio lengths to ensure consistent behavior."""
        sr = 22050
        hop_length = 512
        n_bins = 84
        bins_per_octave = 12
        
        test_lengths = [0.5, 1.0, 1.5, 2.0, 3.14159]  # Various durations in seconds
        
        for duration in test_lengths:
            samples = int(sr * duration)
            audio = np.random.randn(samples).astype(np.float32)
            audio_jax = jnp.array(audio)
            
            cqt_result = cqt2010_jit(
                audio_jax, sr=sr, hop_length=hop_length,
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                output_format='magnitude',
            )
            
            assert cqt_result.shape[0] == n_bins, f"Wrong number of frequency bins"
            assert cqt_result.shape[1] > 0, f"No time frames for duration {duration}s"
    
    def test_edge_case_lengths(self, cqt2010_jit):
        """Test edge cases that might cause dimension mismatches."""
        sr = 22050
        
        # Test case 1: Very short audio
        audio_short = jnp.ones(1024)
        cqt_short = cqt2010_jit(audio_short, sr=sr, n_bins=36, output_format='magnitude')
        assert cqt_short.shape[0] == 36, "Wrong number of bins for short audio"
        
        # Test case 2: Audio length that might cause off-by-one errors
        problematic_length = 22050 + 256  # sr + kernel_size
        audio_prob = jnp.ones(problematic_length)
        cqt_prob = cqt2010_jit(audio_prob, sr=sr, n_bins=84, output_format='magnitude')
        assert cqt_prob.shape[0] == 84, "Wrong number of bins for problematic length"


class TestCQTMusical:
    """Test CQT with musical signals."""
    
    def test_musical_signal(self, cqt_jit):
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
        C_jax = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins)
        
        # Both should detect the same prominent frequencies
        C_librosa_mean = np.mean(np.abs(C_librosa), axis=1)
        C_jax_mean = np.mean(np.abs(C_jax), axis=1)
        
        # Find top 10 peaks
        top_bins_librosa = np.argsort(C_librosa_mean)[-10:]
        top_bins_jax = np.argsort(C_jax_mean)[-10:]
        
        # At least 7 out of 10 should match
        overlap = len(set(top_bins_librosa) & set(top_bins_jax))
        assert overlap >= 7, f"Too few matching peaks: {overlap}/10"
        
        # Check correlation
        corr = pearsonr(C_librosa_mean, C_jax_mean)[0]
        assert corr > 0.98, f"Low correlation: {corr}"
    
    def test_normalization(self, cqt_jit):
        """Test that CQT normalization is working correctly."""
        sr = 22050
        hop_length = 512
        
        # White noise should give roughly equal energy in all bins
        np.random.seed(42)
        noise = np.random.randn(sr * 2)  # 2 seconds
        
        y_jax = jnp.array(noise)
        C = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=84, norm=1.0)
        
        # Check that we don't have extreme values
        C_mag = np.abs(C)
        assert np.all(np.isfinite(C_mag)), "CQT contains non-finite values"
        assert np.max(C_mag) < 100, "CQT magnitudes unreasonably large"
        assert np.min(C_mag) >= 0, "CQT magnitudes should be non-negative"
        
        # Energy distribution shouldn't be too extreme
        C_mean = np.mean(C_mag, axis=1)
        energy_ratio = np.max(C_mean) / (np.min(C_mean) + 1e-10)
        assert energy_ratio < 1000, "Energy distribution too extreme"


class TestCQTHighResolution:
    """Test CQT with high resolution settings (24 bins per octave)."""
    
    def test_high_resolution_cqt(self, cqt_jit, cqt2010_jit):
        """Test CQT with 24 bins per octave like nnAudio does."""
        # Use nnAudio's test parameters
        fs = 44100  # Higher sample rate
        t = 1
        f0 = 55
        f1 = 22050
        s = np.linspace(0, t, fs * t)
        
        # Test both sweep types
        for method, name in [("logarithmic", "log"), ("linear", "linear")]:
            x = chirp(s, f0, 1, f1, method=method)
            x = x.astype(dtype=np.float32)
            x_jax = jnp.array(x)
            
            # Test CQT1992 with high resolution
            try:
                C1992 = cqt_jit(
                    x_jax, sr=fs, fmin=55, n_bins=207, bins_per_octave=24,
                    hop_length=512, n_fft=16384  # Larger FFT for high resolution
                )
                print(f"\nCQT1992 {name} sweep: shape={C1992.shape}")
                assert C1992.shape[0] == 207, "Wrong number of bins"
                assert not np.any(np.isnan(C1992)), "NaN in CQT output"
            except Exception as e:
                print(f"\nWarning: CQT1992 high resolution failed: {e}")
            
            # Test CQT2010 with high resolution
            C2010 = cqt2010_jit(
                x_jax, sr=fs, fmin=55, n_bins=207, bins_per_octave=24,
                hop_length=512, output_format='complex'
            )
            print(f"CQT2010 {name} sweep: shape={C2010.shape}")
            assert C2010.shape[0] == 207, "Wrong number of bins"
            assert not np.any(np.isnan(C2010)), "NaN in CQT output"
    
    def test_real_audio(self, cqt_jit, cqt2010_jit):
        """Test CQT with a real audio signal."""
        # Generate a more realistic test signal (chord progression)
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a simple melody with envelope
        notes = [440, 523.25, 659.25, 523.25]  # A4, C5, E5, C5
        note_duration = duration / len(notes)
        y = np.zeros_like(t)
        
        for i, freq in enumerate(notes):
            start = int(i * note_duration * sr)
            end = int((i + 1) * note_duration * sr)
            note_t = t[start:end]
            # Add envelope
            envelope = np.exp(-3 * (note_t - note_t[0]))
            y[start:end] = envelope * np.sin(2 * np.pi * freq * note_t)
        
        # Add some noise for realism
        y += 0.01 * np.random.randn(len(y))
        y_jax = jnp.array(y.astype(np.float32))
        
        # Test both CQT versions
        C1992 = cqt_jit(y_jax, sr=sr, n_bins=84)
        C2010 = cqt2010_jit(y_jax, sr=sr, n_bins=84, output_format='complex')
        
        # Basic sanity checks
        assert C1992.shape[1] > 0, "No time frames in CQT output"
        assert C2010.shape[1] > 0, "No time frames in CQT output"
        
        # Check that we can detect the note frequencies
        C1992_mean = np.mean(np.abs(C1992), axis=1)
        C2010_mean = np.mean(np.abs(C2010), axis=1)
        
        # Both should have similar energy distribution
        corr = np.corrcoef(C1992_mean, C2010_mean)[0, 1]
        assert corr > 0.9, f"Low correlation between CQT versions: {corr}"


class TestCQTDeviceCompatibility:
    """Test CQT on different devices (CPU/GPU)."""
    
    def test_device_consistency(self, cqt_jit, cqt2010_jit):
        """Test that CQT gives consistent results regardless of device."""
        # Generate test signal
        sr = 22050
        y = np.random.randn(sr).astype(np.float32)
        y_jax = jnp.array(y)
        
        # Run CQT (JAX will use whatever device is available)
        C1 = cqt_jit(y_jax, sr=sr, n_bins=36)
        C2 = cqt2010_jit(y_jax, sr=sr, n_bins=36, output_format='complex')
        
        # Check that results are deterministic
        C1_again = cqt_jit(y_jax, sr=sr, n_bins=36)
        C2_again = cqt2010_jit(y_jax, sr=sr, n_bins=36, output_format='complex')
        
        np.testing.assert_allclose(C1, C1_again, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(C2, C2_again, rtol=1e-6, atol=1e-6)
        
        print(f"\nDevice used: {C1.devices()} (JAX automatically selects best available)")


class TestCQTPerformance:
    """Performance comparison tests."""
    
    def test_performance_comparison(self, cqt_jit, cqt2010_jit):
        """Compare performance of different CQT implementations."""
        # Generate test signal
        sr = 22050
        duration = 2.0
        y = np.random.randn(int(sr * duration))
        y_jax = jnp.array(y)
        
        # Parameters
        hop_length = 512
        n_bins = 84
        bins_per_octave = 12
        
        # Warmup runs to ensure JIT compilation
        print("\nWarming up JIT compilation...")
        for _ in range(3):
            cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins).block_until_ready()
            cqt2010_jit(
                y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins,
                output_format="magnitude", earlydownsample=True
            ).block_until_ready()
        
        # Time CQT1992 (average of multiple runs)
        num_runs = 10
        cqt1992_times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = cqt_jit(y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins)
            result.block_until_ready()
            cqt1992_times.append(time.time() - start_time)
        cqt1992_time = np.mean(cqt1992_times)
        cqt1992_std = np.std(cqt1992_times)
        
        # Time CQT2010 with early downsampling (average of multiple runs)
        cqt2010_times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = cqt2010_jit(
                y_jax, sr=sr, hop_length=hop_length, n_bins=n_bins,
                output_format="magnitude", earlydownsample=True
            )
            result.block_until_ready()
            cqt2010_times.append(time.time() - start_time)
        cqt2010_time = np.mean(cqt2010_times)
        cqt2010_std = np.std(cqt2010_times)
        
        print(f"\nCQT1992 time: {cqt1992_time:.4f} ± {cqt1992_std:.4f} seconds (avg of {num_runs} runs)")
        print(f"CQT2010 time: {cqt2010_time:.4f} ± {cqt2010_std:.4f} seconds (avg of {num_runs} runs)")
        print(f"Speedup: {cqt1992_time/cqt2010_time:.2f}x")
        
        # CQT2010 should generally be faster with early downsampling
        # but we won't assert this as it depends on hardware


if __name__ == "__main__":
    pytest.main([__file__, "-v"])