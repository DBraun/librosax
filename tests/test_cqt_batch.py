#!/usr/bin/env python3
"""Test CQT batch processing to verify batch dimension handling."""

import jax
# Enable JAX 64-bit mode for better precision
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from librosax.feature.spectral import cqt, cqt2010


def test_batch_processing():
    """Test that batch dimensions are preserved correctly."""
    print("Testing batch processing...")
    
    sr = 22050
    duration = 0.5
    samples = int(sr * duration)
    
    # Test different input shapes
    test_cases = [
        ("1D input", np.random.randn(samples)),
        ("2D input (single)", np.random.randn(1, samples)),
        ("2D input (batch=3)", np.random.randn(3, samples)),
        ("2D input (batch=5)", np.random.randn(5, samples)),
    ]
    
    for name, audio in test_cases:
        audio_jax = jnp.array(audio, dtype=jnp.float32)
        
        print(f"\n{name}:")
        print(f"  Input shape: {audio_jax.shape}")
        
        # Test CQT1992
        cqt_result = cqt(audio_jax, sr=sr, output_format='magnitude')
        print(f"  CQT1992 output shape: {cqt_result.shape}")
        
        # Test CQT2010
        cqt2010_result = cqt2010(audio_jax, sr=sr, output_format='magnitude')
        print(f"  CQT2010 output shape: {cqt2010_result.shape}")
        
        # Verify batch dimension handling
        if audio.ndim == 1:
            # 1D input should give 2D output (n_bins, time)
            assert cqt_result.ndim == 2, f"Expected 2D output for 1D input, got {cqt_result.ndim}D"
            assert cqt2010_result.ndim == 2, f"Expected 2D output for 1D input, got {cqt2010_result.ndim}D"
        else:
            # 2D input should give 3D output (batch, n_bins, time)
            assert cqt_result.ndim == 3, f"Expected 3D output for 2D input, got {cqt_result.ndim}D"
            assert cqt2010_result.ndim == 3, f"Expected 3D output for 2D input, got {cqt2010_result.ndim}D"
            assert cqt_result.shape[0] == audio.shape[0], f"Batch dimension mismatch"
            assert cqt2010_result.shape[0] == audio.shape[0], f"Batch dimension mismatch"
        
        print("  ✓ Batch dimension handling correct")


def test_basic_pitch_batch():
    """Test Basic Pitch configuration with batch processing."""
    print("\nTesting Basic Pitch configuration with batches...")
    
    sr = 22050
    hop_length = 512
    n_bins = 264
    bins_per_octave = 36
    
    # Create batch of 4 audio signals
    batch_size = 4
    duration = 1.0
    samples = int(sr * duration)
    audio_batch = np.random.randn(batch_size, samples).astype(np.float32)
    audio_jax = jnp.array(audio_batch)
    
    print(f"Input shape: {audio_jax.shape}")
    
    # Process with CQT2010
    cqt_result = cqt2010(
        audio_jax,
        sr=sr,
        hop_length=hop_length,
        fmin=32.70,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        output_format='magnitude',
        earlydownsample=True,
    )
    
    print(f"CQT output shape: {cqt_result.shape}")
    expected_shape = (batch_size, n_bins, -1)  # -1 for time frames
    
    assert cqt_result.shape[0] == batch_size, f"Batch size mismatch"
    assert cqt_result.shape[1] == n_bins, f"Number of bins mismatch"
    print(f"✓ Basic Pitch batch processing works correctly")


def test_output_formats_with_batch():
    """Test different output formats with batch input."""
    print("\nTesting output formats with batch input...")
    
    sr = 22050
    samples = 11025
    batch_size = 2
    audio_batch = np.random.randn(batch_size, samples).astype(np.float32)
    audio_jax = jnp.array(audio_batch)
    
    formats = ['magnitude', 'complex', 'phase']
    
    for fmt in formats:
        result = cqt2010(audio_jax, sr=sr, n_bins=36, output_format=fmt)
        print(f"\nOutput format '{fmt}':")
        print(f"  Shape: {result.shape}")
        print(f"  Dtype: {result.dtype}")
        
        # Verify batch dimension is preserved
        assert result.shape[0] == batch_size, f"Batch dimension lost for format {fmt}"
        
        if fmt == 'phase':
            assert result.shape[-1] == 2, f"Phase should have 2 components"
        
        print("  ✓ Batch dimension preserved")


if __name__ == "__main__":
    print("Testing CQT batch processing")
    print("=" * 50)
    
    test_batch_processing()
    test_basic_pitch_batch()
    test_output_formats_with_batch()
    
    print("\n" + "=" * 50)
    print("✓ All batch processing tests passed!")