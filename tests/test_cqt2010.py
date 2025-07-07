#!/usr/bin/env python3
"""Test script for the new CQT2010 implementation in librosax."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from librosax.feature.spectral import cqt, cqt2010
import librosa

# Enable JAX 64-bit mode for better precision
jax.config.update("jax_enable_x64", True)


def generate_test_signal(sr=22050, duration=2.0):
    """Generate a test signal with multiple frequency components."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with several pure tones
    frequencies = [440, 880, 1320, 1760]  # A4, A5, E6, A6
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += 0.25 * np.sin(2 * np.pi * freq * t)
        
    return jnp.array(signal)


def compare_implementations():
    """Compare CQT1992 and CQT2010 implementations."""
    print("Comparing CQT implementations...")
    
    # Generate test signal
    sr = 22050
    duration = 2.0
    y = generate_test_signal(sr, duration)
    
    # Parameters
    hop_length = 512
    n_bins = 84
    bins_per_octave = 12
    
    # Time CQT1992
    start_time = time.time()
    cqt1992_complex = cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        output_format="complex"
    )
    cqt1992_mag = jnp.abs(cqt1992_complex)
    cqt1992_time = time.time() - start_time
    
    # Time CQT2010
    start_time = time.time()
    cqt2010_mag = cqt2010(
        y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        output_format="magnitude",
        earlydownsample=True
    )
    cqt2010_time = time.time() - start_time
    
    print(f"\nCQT1992v2 time: {cqt1992_time:.4f} seconds")
    print(f"CQT2010v2 time: {cqt2010_time:.4f} seconds")
    print(f"Speedup: {cqt1992_time/cqt2010_time:.2f}x")
    
    print(f"\nCQT1992 shape: {cqt1992_mag.shape}")
    print(f"CQT2010 shape: {cqt2010_mag.shape}")
    
    # Check if implementations match using allclose
    if cqt1992_mag.shape == cqt2010_mag.shape:
        # Convert to numpy for testing
        cqt1992_np = np.array(cqt1992_mag)
        cqt2010_np = np.array(cqt2010_mag)
        
        # Test with different tolerances
        rtol_values = [1e-1, 1e-2, 1e-3, 1e-4]
        print("\nTesting implementation match with np.allclose:")
        for rtol in rtol_values:
            match = np.allclose(cqt1992_np, cqt2010_np, rtol=rtol, atol=1e-8)
            print(f"  rtol={rtol}: {'PASS' if match else 'FAIL'}")
            
        # Calculate detailed statistics
        abs_diff = np.abs(cqt1992_np - cqt2010_np)
        rel_diff = abs_diff / (np.abs(cqt1992_np) + 1e-10)
        
        print(f"\nDetailed comparison statistics:")
        print(f"  Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"  Mean absolute difference: {np.mean(abs_diff):.6e}")
        print(f"  Max relative difference: {np.max(rel_diff):.6e}")
        print(f"  Mean relative difference: {np.mean(rel_diff):.6e}")
        
        # Calculate correlation
        correlation = jnp.corrcoef(cqt1992_mag.flatten(), cqt2010_mag.flatten())[0, 1]
        print(f"  Correlation coefficient: {correlation:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot CQT1992
    im1 = axes[0].imshow(
        cqt1992_mag,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[0].set_title('CQT1992v2 (FFT-based)')
    axes[0].set_ylabel('CQT Bin')
    plt.colorbar(im1, ax=axes[0], label='Magnitude')
    
    # Plot CQT2010
    im2 = axes[1].imshow(
        cqt2010_mag,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[1].set_title('CQT2010v2 (Multi-resolution)')
    axes[1].set_ylabel('CQT Bin')
    axes[1].set_xlabel('Time Frame')
    plt.colorbar(im2, ax=axes[1], label='Magnitude')
    
    plt.tight_layout()
    plt.savefig('cqt_comparison.png', dpi=150)
    print("\nPlot saved as 'cqt_comparison.png'")
    
    return cqt1992_mag, cqt2010_mag


def test_output_formats():
    """Test different output formats."""
    print("\nTesting output formats...")
    
    # Generate short test signal
    sr = 22050
    y = generate_test_signal(sr, 0.5)
    
    # Test all output formats for CQT2010
    formats = ['magnitude', 'complex', 'phase']
    
    for fmt in formats:
        result = cqt2010(
            y,
            sr=sr,
            hop_length=512,
            n_bins=36,
            output_format=fmt
        )
        print(f"\nOutput format '{fmt}':")
        print(f"  Shape: {result.shape}")
        print(f"  Dtype: {result.dtype}")
        
        if fmt == 'complex':
            print(f"  Example value: {result[0, 0]:.4f}")
        elif fmt == 'phase':
            print(f"  Phase shape last dim: {result.shape[-1]}")


def test_algorithm_equivalence():
    """Test that CQT1992 and CQT2010 produce equivalent results."""
    print("\nTesting algorithm equivalence...")
    
    # Test with different signal types
    test_cases = [
        ("Pure tone 440Hz", lambda sr, dur: np.sin(2 * np.pi * 440 * np.linspace(0, dur, int(sr * dur)))),
        ("Multiple tones", generate_test_signal),
        ("White noise", lambda sr, dur: np.random.randn(int(sr * dur)) * 0.1),
    ]
    
    sr = 22050
    duration = 0.5
    hop_length = 512
    n_bins = 36
    bins_per_octave = 12
    
    for test_name, signal_func in test_cases:
        print(f"\n{test_name}:")
        y = jnp.array(signal_func(sr, duration))
        
        # Compute with both algorithms - without early downsampling for fair comparison
        cqt1992 = cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            output_format="magnitude"
        )
        
        cqt2010_result = cqt2010(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            output_format="magnitude",
            earlydownsample=False  # Disable for fair comparison
        )
        
        # Convert to numpy
        cqt1992_np = np.array(cqt1992)
        cqt2010_np = np.array(cqt2010_result)
        
        # Test with allclose
        tolerances = [(1e-1, "Low precision"), (1e-2, "Medium precision"), 
                      (1e-3, "High precision"), (1e-4, "Very high precision")]
        
        for rtol, desc in tolerances:
            match = np.allclose(cqt1992_np, cqt2010_np, rtol=rtol, atol=1e-8)
            print(f"  {desc} (rtol={rtol}): {'PASS' if match else 'FAIL'}")
            if match:
                break
        
        # Compute statistics
        abs_diff = np.abs(cqt1992_np - cqt2010_np)
        rel_diff = abs_diff / (np.abs(cqt1992_np) + 1e-10)
        
        print(f"  Max absolute error: {np.max(abs_diff):.2e}")
        print(f"  Mean absolute error: {np.mean(abs_diff):.2e}")
        print(f"  Max relative error: {np.max(rel_diff):.2e}")


def test_normalization_types():
    """Test different normalization types."""
    print("\nTesting normalization types...")
    
    # Generate short test signal
    sr = 22050
    y = generate_test_signal(sr, 0.5)
    
    # Test all normalization types
    norm_types = ['librosa', 'convolutional', 'wrap']
    
    for norm_type in norm_types:
        result = cqt(
            y,
            sr=sr,
            hop_length=512,
            n_bins=36,
            output_format='magnitude',
            normalization_type=norm_type
        )
        print(f"\nNormalization type '{norm_type}':")
        print(f"  Mean magnitude: {jnp.mean(result):.6f}")
        print(f"  Max magnitude: {jnp.max(result):.6f}")


def test_librosa_comparison():
    """Compare librosax CQT with librosa CQT."""
    print("\nComparing with librosa CQT...")
    
    # Generate test signal
    sr = 22050
    duration = 1.0
    y_np = np.array(generate_test_signal(sr, duration))
    y_jax = jnp.array(y_np)
    
    # Parameters
    hop_length = 512
    n_bins = 84
    bins_per_octave = 12
    fmin = 32.70  # C1
    
    # Compute with librosa
    start_time = time.time()
    cqt_librosa = np.abs(librosa.cqt(
        y_np,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=1.0,
        norm=1,
        window='hann',
        scale=True,
        pad_mode='constant'
    ))
    librosa_time = time.time() - start_time
    
    # Compute with librosax CQT1992
    start_time = time.time()
    cqt_librosax = cqt(
        y_jax,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=1.0,
        norm=1,
        window='hann',
        scale=True,
        pad_mode='constant',
        output_format='magnitude'
    )
    librosax_time = time.time() - start_time
    
    # Compute with librosax CQT2010
    start_time = time.time()
    cqt_librosax_2010 = cqt2010(
        y_jax,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=1.0,
        norm=1,
        window='hann',
        scale=True,
        pad_mode='constant',
        output_format='magnitude',
        earlydownsample=True
    )
    librosax_2010_time = time.time() - start_time
    
    print(f"\nComputation times:")
    print(f"  librosa CQT: {librosa_time:.4f} seconds")
    print(f"  librosax CQT1992: {librosax_time:.4f} seconds")
    print(f"  librosax CQT2010: {librosax_2010_time:.4f} seconds")
    
    # Convert to numpy for comparison
    cqt_librosax_np = np.array(cqt_librosax)
    cqt_librosax_2010_np = np.array(cqt_librosax_2010)
    
    # Adjust shapes if needed (librosa might have different padding)
    min_frames = min(cqt_librosa.shape[1], cqt_librosax_np.shape[1])
    cqt_librosa = cqt_librosa[:, :min_frames]
    cqt_librosax_np = cqt_librosax_np[:, :min_frames]
    cqt_librosax_2010_np = cqt_librosax_2010_np[:, :min_frames]
    
    print(f"\nShapes:")
    print(f"  librosa: {cqt_librosa.shape}")
    print(f"  librosax CQT1992: {cqt_librosax_np.shape}")
    print(f"  librosax CQT2010: {cqt_librosax_2010_np.shape}")
    
    # Test librosax CQT1992 vs librosa
    print("\nlibrosax CQT1992 vs librosa:")
    rtol_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    for rtol in rtol_values:
        match = np.allclose(cqt_librosa, cqt_librosax_np, rtol=rtol, atol=1e-8)
        print(f"  rtol={rtol}: {'PASS' if match else 'FAIL'}")
        if match:
            break
    
    # Detailed statistics
    abs_diff = np.abs(cqt_librosa - cqt_librosax_np)
    rel_diff = abs_diff / (np.abs(cqt_librosa) + 1e-10)
    print(f"  Max absolute error: {np.max(abs_diff):.2e}")
    print(f"  Mean absolute error: {np.mean(abs_diff):.2e}")
    print(f"  Max relative error: {np.max(rel_diff):.2e}")
    
    # Test librosax CQT2010 vs librosa
    print("\nlibrosax CQT2010 vs librosa:")
    for rtol in rtol_values:
        match = np.allclose(cqt_librosa, cqt_librosax_2010_np, rtol=rtol, atol=1e-8)
        print(f"  rtol={rtol}: {'PASS' if match else 'FAIL'}")
        if match:
            break
    
    # Detailed statistics
    abs_diff_2010 = np.abs(cqt_librosa - cqt_librosax_2010_np)
    rel_diff_2010 = abs_diff_2010 / (np.abs(cqt_librosa) + 1e-10)
    print(f"  Max absolute error: {np.max(abs_diff_2010):.2e}")
    print(f"  Mean absolute error: {np.mean(abs_diff_2010):.2e}")
    print(f"  Max relative error: {np.max(rel_diff_2010):.2e}")


def test_early_downsampling():
    """Test early downsampling effect."""
    print("\nTesting early downsampling...")
    
    # Generate test signal
    sr = 22050
    y = generate_test_signal(sr, 1.0)
    
    # With early downsampling
    start_time = time.time()
    cqt_with_ds = cqt2010(
        y,
        sr=sr,
        hop_length=512,
        n_bins=84,
        earlydownsample=True,
        output_format="magnitude"
    )
    time_with_ds = time.time() - start_time
    
    # Without early downsampling
    start_time = time.time()
    cqt_without_ds = cqt2010(
        y,
        sr=sr,
        hop_length=512,
        n_bins=84,
        earlydownsample=False,
        output_format="magnitude"
    )
    time_without_ds = time.time() - start_time
    
    print(f"\nWith early downsampling: {time_with_ds:.4f} seconds")
    print(f"Without early downsampling: {time_without_ds:.4f} seconds")
    print(f"Speedup: {time_without_ds/time_with_ds:.2f}x")
    
    # Convert to numpy for testing
    cqt_with_np = np.array(cqt_with_ds)
    cqt_without_np = np.array(cqt_without_ds)
    
    # Test with allclose
    print("\nTesting early downsampling match with np.allclose:")
    rtol_values = [1e-1, 1e-2, 1e-3, 1e-4]
    for rtol in rtol_values:
        match = np.allclose(cqt_with_np, cqt_without_np, rtol=rtol, atol=1e-8)
        print(f"  rtol={rtol}: {'PASS' if match else 'FAIL'}")
    
    # Detailed statistics
    abs_diff = np.abs(cqt_with_np - cqt_without_np)
    print(f"\nDetailed statistics:")
    print(f"  Max absolute difference: {np.max(abs_diff):.6e}")
    print(f"  Mean absolute difference: {np.mean(abs_diff):.6e}")
    
    # Compare results
    correlation = jnp.corrcoef(cqt_with_ds.flatten(), cqt_without_ds.flatten())[0, 1]
    print(f"  Correlation: {correlation:.6f}")
    
    # Difference in low frequencies
    low_freq_diff = jnp.mean(jnp.abs(cqt_with_ds[:12] - cqt_without_ds[:12]))
    high_freq_diff = jnp.mean(jnp.abs(cqt_with_ds[-12:] - cqt_without_ds[-12:]))
    print(f"  Low frequency difference: {low_freq_diff:.6f}")
    print(f"  High frequency difference: {high_freq_diff:.6f}")


if __name__ == "__main__":
    print("Testing librosax CQT implementations")
    print("=" * 50)
    
    # Run tests
    compare_implementations()
    test_algorithm_equivalence()
    test_librosa_comparison()
    test_output_formats()
    test_normalization_types()
    test_early_downsampling()
    
    print("\n" + "=" * 50)
    print("All tests completed!")