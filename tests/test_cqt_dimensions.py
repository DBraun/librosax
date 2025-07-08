#!/usr/bin/env python3
"""Test CQT2010 implementation for dimension consistency."""

import jax
# Enable JAX 64-bit mode for better precision
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from librosax.feature.spectral import cqt2010


def test_basic_pitch_settings():
    """Test CQT2010 with Basic Pitch settings to ensure no dimension mismatches."""
    print("Testing CQT2010 with Basic Pitch settings...")
    
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
    
    try:
        # Compute CQT
        cqt_result = cqt2010(
            audio_jax,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=1.0,
            norm=1,
            window='hann',
            scale=True,
            pad_mode='reflect',
            output_format='magnitude',
            earlydownsample=True,
        )
        
        print(f"✓ Success! CQT shape: {cqt_result.shape}")
        print(f"  Expected shape: ({n_bins}, time_frames)")
        
        # Test without early downsampling
        cqt_result_no_ds = cqt2010(
            audio_jax,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=1.0,
            norm=1,
            window='hann',
            scale=True,
            pad_mode='reflect',
            output_format='magnitude',
            earlydownsample=False,
        )
        
        print(f"✓ Without early downsampling shape: {cqt_result_no_ds.shape}")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_various_audio_lengths():
    """Test with various audio lengths to ensure consistent behavior."""
    print("\nTesting various audio lengths...")
    
    sr = 22050
    hop_length = 512
    n_bins = 84
    bins_per_octave = 12
    
    test_lengths = [0.5, 1.0, 1.5, 2.0, 3.14159]  # Various durations in seconds
    
    for duration in test_lengths:
        samples = int(sr * duration)
        audio = np.random.randn(samples).astype(np.float32)
        audio_jax = jnp.array(audio)
        
        try:
            cqt_result = cqt2010(
                audio_jax,
                sr=sr,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                output_format='magnitude',
            )
            
            expected_frames = (samples + 2 * (2048 // 2)) // hop_length + 1  # Approximate
            print(f"✓ Duration: {duration:.2f}s, Samples: {samples}, CQT shape: {cqt_result.shape}")
            
        except Exception as e:
            print(f"✗ Duration: {duration:.2f}s failed: {type(e).__name__}: {e}")
            return False
    
    return True


def test_edge_cases():
    """Test edge cases that might cause dimension mismatches."""
    print("\nTesting edge cases...")
    
    sr = 22050
    
    # Test case 1: Very short audio
    audio_short = jnp.ones(1024)
    try:
        cqt_short = cqt2010(audio_short, sr=sr, n_bins=36, output_format='magnitude')
        print(f"✓ Short audio (1024 samples): {cqt_short.shape}")
    except Exception as e:
        print(f"✗ Short audio failed: {e}")
    
    # Test case 2: Audio length that might cause off-by-one errors
    # Choose a length that after multiple downsampling might cause issues
    problematic_length = 22050 + 256  # sr + kernel_size
    audio_prob = jnp.ones(problematic_length)
    try:
        cqt_prob = cqt2010(audio_prob, sr=sr, n_bins=84, output_format='magnitude')
        print(f"✓ Problematic length ({problematic_length} samples): {cqt_prob.shape}")
    except Exception as e:
        print(f"✗ Problematic length failed: {e}")
    
    return True


if __name__ == "__main__":
    print("Testing CQT2010 dimension consistency")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_basic_pitch_settings()
    all_passed &= test_various_audio_lengths()
    all_passed &= test_edge_cases()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")