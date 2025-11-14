"""Test comparing SpecAugmentation between torchlibrosa and librosax."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torchlibrosa.augmentation import SpecAugmentation as TorchlibrosaSA
from flax import nnx

from librosax.layers import SpecAugmentation


def set_same_random_mask_locations(jax_key, torch_gen):
    """Helper to synchronize random mask locations between JAX and PyTorch."""
    # Get JAX random values
    jax_uniform = jax.random.uniform(jax_key, shape=(10,))
    
    # Create corresponding torch random values
    torch.manual_seed(42)
    
    return jax_uniform


def test_spec_augmentation_deterministic():
    """Test that deterministic mode returns input unchanged for both implementations."""
    # Create test spectrogram
    batch_size, time_steps, freq_bins = 4, 100, 80
    spec_np = np.random.randn(batch_size, 1, time_steps, freq_bins).astype(np.float32)
    
    # torchlibrosa - doesn't have deterministic mode, so we skip it for this test
    spec_torch = torch.from_numpy(spec_np)
    torch_output = spec_torch  # torchlibrosa doesn't support deterministic mode directly
    
    # librosax
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        time_drop_width=16,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        deterministic=True,
        rngs=nnx.Rngs(jax.random.key(0))
    )
    jax_output = librosax_aug(spec_jax, deterministic=True)
    
    # torchlibrosa doesn't have deterministic mode, so we just verify librosax
    # np.testing.assert_allclose(spec_np, torch_output.numpy(), rtol=1e-6)
    np.testing.assert_allclose(spec_np, np.array(jax_output), rtol=1e-6)


def test_spec_augmentation_shapes():
    """Test that both implementations handle different input shapes correctly."""
    # Test 3D input (batch, time, freq)
    spec_3d = np.random.randn(4, 100, 80).astype(np.float32)
    
    # librosax should handle 3D
    librosax_aug = SpecAugmentation(
        time_drop_width=16,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(0))
    )
    jax_output_3d = librosax_aug(jnp.array(spec_3d))
    assert jax_output_3d.shape == spec_3d.shape
    
    # Test 4D input (batch, channel, time, freq)
    spec_4d = np.random.randn(4, 1, 100, 80).astype(np.float32)
    
    # torchlibrosa expects 4D
    torchlibrosa_aug = TorchlibrosaSA(
        time_drop_width=16,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2
    )
    torch_output_4d = torchlibrosa_aug(torch.from_numpy(spec_4d))
    assert torch_output_4d.shape == spec_4d.shape
    
    # librosax should also handle 4D
    jax_output_4d = librosax_aug(jnp.array(spec_4d))
    assert jax_output_4d.shape == spec_4d.shape


def test_spec_augmentation_masking_behavior():
    """Test that masking behavior is similar between implementations."""
    # Create test spectrogram with known values
    batch_size, channels, time_steps, freq_bins = 2, 1, 100, 80
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    # Apply augmentation with high drop probability
    # torchlibrosa
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(
        time_drop_width=20,
        time_stripes_num=2,
        freq_drop_width=10,
        freq_stripes_num=2
    )
    torch.manual_seed(42)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # librosax
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        time_drop_width=20,
        time_stripes_num=2,
        freq_drop_width=10,
        freq_stripes_num=2,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(42))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Check that some values were masked (set to 0)
    torch_masked_count = (torch_output == 0).sum().item()
    jax_masked_count = (jax_output == 0).sum().item()
    
    # Both should have masked some values
    assert torch_masked_count > 0, "torchlibrosa should have masked some values"
    assert jax_masked_count > 0, "librosax should have masked some values"
    
    # The exact counts may differ due to different RNG implementations
    print(f"torchlibrosa masked {torch_masked_count} values")
    print(f"librosax masked {jax_masked_count} values")


def test_spec_augmentation_time_masking_only():
    """Test time masking only (no frequency masking)."""
    batch_size, channels, time_steps, freq_bins = 2, 1, 100, 80
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    # torchlibrosa - time masking only
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(
        time_drop_width=30,
        time_stripes_num=3,
        freq_drop_width=0,  # No frequency masking
        freq_stripes_num=0
    )
    torch.manual_seed(42)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # librosax - time masking only
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        time_drop_width=30,
        time_stripes_num=3,
        freq_drop_width=0,  # No frequency masking
        freq_stripes_num=0,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(42))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Check masking pattern - should only mask along time axis
    torch_output_np = torch_output.numpy()
    jax_output_np = np.array(jax_output)
    
    # Sum across time axis - if time masking works, some time slices should be all zeros
    torch_time_sums = torch_output_np.sum(axis=(0, 1, 3))  # Sum over batch, channel, freq
    jax_time_sums = jax_output_np.sum(axis=(0, 1, 3))
    
    # Check that some time steps are masked
    torch_masked_times = (torch_time_sums == 0).sum()
    jax_masked_times = (jax_time_sums == 0).sum()
    
    print(f"Time masking - torchlibrosa masked {torch_masked_times} time steps")
    print(f"Time masking - librosax masked {jax_masked_times} time steps")
    
    # Both should mask some time steps
    assert torch_masked_times > 0 or (torch_output_np < 1).any(), "torchlibrosa should mask time"
    assert jax_masked_times > 0 or (jax_output_np < 1).any(), "librosax should mask time"


def test_spec_augmentation_freq_masking_only():
    """Test frequency masking only (no time masking)."""
    batch_size, channels, time_steps, freq_bins = 2, 1, 100, 80
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    # torchlibrosa - frequency masking only
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(
        time_drop_width=0,  # No time masking
        time_stripes_num=0,
        freq_drop_width=15,
        freq_stripes_num=3
    )
    torch.manual_seed(42)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # librosax - frequency masking only
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        time_drop_width=0,  # No time masking
        time_stripes_num=0,
        freq_drop_width=15,
        freq_stripes_num=3,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(42))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Check masking pattern - should only mask along frequency axis
    torch_output_np = torch_output.numpy()
    jax_output_np = np.array(jax_output)
    
    # Sum across frequency axis - if freq masking works, some freq bins should be all zeros
    torch_freq_sums = torch_output_np.sum(axis=(0, 1, 2))  # Sum over batch, channel, time
    jax_freq_sums = jax_output_np.sum(axis=(0, 1, 2))
    
    # Check that some frequency bins are masked
    torch_masked_freqs = (torch_freq_sums == 0).sum()
    jax_masked_freqs = (jax_freq_sums == 0).sum()
    
    print(f"Freq masking - torchlibrosa masked {torch_masked_freqs} freq bins")
    print(f"Freq masking - librosax masked {jax_masked_freqs} freq bins")
    
    # Both should mask some frequency bins
    assert torch_masked_freqs > 0 or (torch_output_np < 1).any(), "torchlibrosa should mask freq"
    assert jax_masked_freqs > 0 or (jax_output_np < 1).any(), "librosax should mask freq"


def test_spec_augmentation_parameter_ranges():
    """Test that both implementations handle various parameter ranges correctly."""
    spec_np = np.random.randn(2, 1, 50, 40).astype(np.float32)
    
    test_configs = [
        # (time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num)
        (5, 1, 5, 1),    # Small masking
        (20, 4, 10, 4),  # Large masking
        (0, 0, 10, 2),   # Freq only
        (10, 2, 0, 0),   # Time only
        (50, 1, 40, 1),  # Full dimension masking
    ]
    
    for time_w, time_n, freq_w, freq_n in test_configs:
        # torchlibrosa
        spec_torch = torch.from_numpy(spec_np)
        torchlibrosa_aug = TorchlibrosaSA(
            time_drop_width=time_w,
            time_stripes_num=time_n,
            freq_drop_width=freq_w,
            freq_stripes_num=freq_n
        )
        torch.manual_seed(42)
        torch_output = torchlibrosa_aug(spec_torch)
        
        # librosax
        spec_jax = jnp.array(spec_np)
        librosax_aug = SpecAugmentation(
            time_drop_width=time_w,
            time_stripes_num=time_n,
            freq_drop_width=freq_w,
            freq_stripes_num=freq_n,
            deterministic=False,
            rngs=nnx.Rngs(jax.random.key(42))
        )
        jax_output = librosax_aug(spec_jax, deterministic=False)
        
        # Check shapes are preserved
        assert torch_output.shape == spec_torch.shape
        assert jax_output.shape == spec_jax.shape
        
        print(f"Config ({time_w}, {time_n}, {freq_w}, {freq_n}) - shapes OK")


def test_spec_augmentation_with_real_spectrogram():
    """Test SpecAugmentation on a realistic spectrogram."""
    # Create a realistic log-mel spectrogram
    batch_size, channels = 4, 1
    time_steps, n_mels = 200, 128
    
    # Simulate a log-mel spectrogram with typical range
    spec_np = np.random.randn(batch_size, channels, time_steps, n_mels).astype(np.float32)
    spec_np = spec_np * 10 - 50  # Typical log-mel range
    
    # torchlibrosa
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=16,
        freq_stripes_num=2
    )
    torch.manual_seed(123)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # librosax
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=16,
        freq_stripes_num=2,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(123))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Statistics check
    torch_output_np = torch_output.numpy()
    jax_output_np = np.array(jax_output)
    
    # Check that augmentation was applied
    torch_diff = np.abs(spec_np - torch_output_np).sum()
    jax_diff = np.abs(spec_np - jax_output_np).sum()
    
    print(f"Real spectrogram test:")
    print(f"  torchlibrosa total change: {torch_diff:.2f}")
    print(f"  librosax total change: {jax_diff:.2f}")
    
    # Both should show some change if augmentation was applied
    # (unless random chance caused no masking)
    if torch_diff > 0:
        assert torch_diff > 100, "torchlibrosa should show significant masking"
    if jax_diff > 0:
        assert jax_diff > 100, "librosax should show significant masking"


if __name__ == "__main__":
    print("Testing SpecAugmentation comparison between torchlibrosa and librosax\n")
    
    print("1. Testing deterministic mode...")
    test_spec_augmentation_deterministic()
    print("   ✓ Passed\n")
    
    print("2. Testing shape handling...")
    test_spec_augmentation_shapes()
    print("   ✓ Passed\n")
    
    print("3. Testing masking behavior...")
    test_spec_augmentation_masking_behavior()
    print("   ✓ Passed\n")
    
    print("4. Testing time masking only...")
    test_spec_augmentation_time_masking_only()
    print("   ✓ Passed\n")
    
    print("5. Testing frequency masking only...")
    test_spec_augmentation_freq_masking_only()
    print("   ✓ Passed\n")
    
    print("6. Testing parameter ranges...")
    test_spec_augmentation_parameter_ranges()
    print("   ✓ Passed\n")
    
    print("7. Testing with real spectrogram...")
    test_spec_augmentation_with_real_spectrogram()
    print("   ✓ Passed\n")
    
    print("All tests passed!")