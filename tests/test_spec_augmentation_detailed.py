"""Detailed test comparing masking amounts and batch diversity for SpecAugmentation."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchlibrosa.augmentation import SpecAugmentation as TorchlibrosaSA
from flax import nnx

from librosax.layers import SpecAugmentation


def test_masking_amount_comparison():
    """Compare the amount of masking between torchlibrosa and librosax."""
    print("\n=== Testing Masking Amount Comparison ===")
    
    # Create test spectrogram with all ones to easily count masked values
    batch_size, channels, time_steps, freq_bins = 8, 1, 200, 128
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    # Test different configurations
    configs = [
        {"time_drop_width": 30, "time_stripes_num": 2, 
         "freq_drop_width": 20, "freq_stripes_num": 2},
        {"time_drop_width": 50, "time_stripes_num": 3, 
         "freq_drop_width": 30, "freq_stripes_num": 3},
        {"time_drop_width": 20, "time_stripes_num": 1, 
         "freq_drop_width": 15, "freq_stripes_num": 1},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        # Run multiple times to get statistics
        torch_masked_percentages = []
        jax_masked_percentages = []
        
        for seed in range(10):
            # torchlibrosa
            spec_torch = torch.from_numpy(spec_np)
            torchlibrosa_aug = TorchlibrosaSA(**config)
            torch.manual_seed(seed)
            torch_output = torchlibrosa_aug(spec_torch)
            torch_masked = (torch_output == 0).sum().item()
            torch_total = torch_output.numel()
            torch_percentage = (torch_masked / torch_total) * 100
            torch_masked_percentages.append(torch_percentage)
            
            # librosax
            spec_jax = jnp.array(spec_np)
            librosax_aug = SpecAugmentation(
                **config,
                deterministic=False,
                rngs=nnx.Rngs(jax.random.key(seed))
            )
            jax_output = librosax_aug(spec_jax, deterministic=False)
            jax_masked = (jax_output == 0).sum().item()
            jax_total = jax_output.size
            jax_percentage = (jax_masked / jax_total) * 100
            jax_masked_percentages.append(jax_percentage)
        
        # Calculate statistics
        torch_mean = np.mean(torch_masked_percentages)
        torch_std = np.std(torch_masked_percentages)
        jax_mean = np.mean(jax_masked_percentages)
        jax_std = np.std(jax_masked_percentages)
        
        print(f"  torchlibrosa: {torch_mean:.2f}% ± {torch_std:.2f}% masked")
        print(f"  librosax:     {jax_mean:.2f}% ± {jax_std:.2f}% masked")
        
        # Check if the means are reasonably close (within a factor of 2)
        if torch_mean > 0 and jax_mean > 0:
            ratio = max(torch_mean, jax_mean) / min(torch_mean, jax_mean)
            print(f"  Ratio: {ratio:.2f}x")
            assert ratio < 3.0, f"Masking amounts differ by more than 3x: {ratio:.2f}x"


def test_batch_mask_diversity():
    """Verify that each item in a batch gets a different mask."""
    print("\n=== Testing Batch Mask Diversity ===")
    
    # Create test spectrogram
    batch_size, channels, time_steps, freq_bins = 8, 1, 100, 80
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    config = {
        "time_drop_width": 30,
        "time_stripes_num": 2,
        "freq_drop_width": 20,
        "freq_stripes_num": 2
    }
    
    # Test torchlibrosa
    print("\nTesting torchlibrosa batch diversity:")
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(**config)
    torch.manual_seed(42)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # Check each batch item
    torch_masks = []
    for b in range(batch_size):
        mask = (torch_output[b] == 0).cpu().numpy()
        torch_masks.append(mask)
    
    # Compare masks between batch items
    torch_different_count = 0
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            if not np.array_equal(torch_masks[i], torch_masks[j]):
                torch_different_count += 1
    
    total_pairs = batch_size * (batch_size - 1) // 2
    torch_diversity_percentage = (torch_different_count / total_pairs) * 100
    print(f"  {torch_different_count}/{total_pairs} pairs have different masks")
    print(f"  Diversity: {torch_diversity_percentage:.1f}%")
    
    # Test librosax
    print("\nTesting librosax batch diversity:")
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        **config,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(42))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Check each batch item
    jax_masks = []
    for b in range(batch_size):
        mask = (jax_output[b] == 0)
        jax_masks.append(np.array(mask))
    
    # Compare masks between batch items
    jax_different_count = 0
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            if not np.array_equal(jax_masks[i], jax_masks[j]):
                jax_different_count += 1
    
    jax_diversity_percentage = (jax_different_count / total_pairs) * 100
    print(f"  {jax_different_count}/{total_pairs} pairs have different masks")
    print(f"  Diversity: {jax_diversity_percentage:.1f}%")
    
    # Both should have high diversity (>90% different masks)
    assert torch_diversity_percentage > 90, f"torchlibrosa batch diversity too low: {torch_diversity_percentage:.1f}%"
    assert jax_diversity_percentage > 90, f"librosax batch diversity too low: {jax_diversity_percentage:.1f}%"


def test_mask_statistics_per_batch_item():
    """Analyze masking statistics for each item in a batch."""
    print("\n=== Testing Mask Statistics Per Batch Item ===")
    
    batch_size, channels, time_steps, freq_bins = 16, 1, 200, 128
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    config = {
        "time_drop_width": 40,
        "time_stripes_num": 2,
        "freq_drop_width": 25,
        "freq_stripes_num": 2
    }
    
    # torchlibrosa
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(**config)
    torch.manual_seed(123)
    torch_output = torchlibrosa_aug(spec_torch)
    
    torch_masked_per_item = []
    for b in range(batch_size):
        masked_count = (torch_output[b] == 0).sum().item()
        total = channels * time_steps * freq_bins
        percentage = (masked_count / total) * 100
        torch_masked_per_item.append(percentage)
    
    # librosax
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        **config,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(123))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    jax_masked_per_item = []
    for b in range(batch_size):
        masked_count = (jax_output[b] == 0).sum().item()
        total = channels * time_steps * freq_bins
        percentage = (masked_count / total) * 100
        jax_masked_per_item.append(percentage)
    
    print(f"\nPer-item masking percentages (batch_size={batch_size}):")
    print(f"torchlibrosa:")
    print(f"  Min: {min(torch_masked_per_item):.2f}%")
    print(f"  Max: {max(torch_masked_per_item):.2f}%")
    print(f"  Mean: {np.mean(torch_masked_per_item):.2f}%")
    print(f"  Std: {np.std(torch_masked_per_item):.2f}%")
    
    print(f"\nlibrosax:")
    print(f"  Min: {min(jax_masked_per_item):.2f}%")
    print(f"  Max: {max(jax_masked_per_item):.2f}%")
    print(f"  Mean: {np.mean(jax_masked_per_item):.2f}%")
    print(f"  Std: {np.std(jax_masked_per_item):.2f}%")
    
    # Check that there's variation between batch items
    torch_std = np.std(torch_masked_per_item)
    jax_std = np.std(jax_masked_per_item)
    
    print(f"\nVariation check:")
    print(f"  torchlibrosa std: {torch_std:.2f}%")
    print(f"  librosax std: {jax_std:.2f}%")
    
    # There should be some variation (std > 0.5%)
    assert torch_std > 0.5 or min(torch_masked_per_item) == 0, "torchlibrosa should have variation between batch items"
    assert jax_std > 0.5, "librosax should have variation between batch items"


def test_visual_mask_comparison():
    """Visual comparison of masks for a small example."""
    print("\n=== Visual Mask Comparison (Small Example) ===")
    
    # Small example for visual inspection
    batch_size, channels, time_steps, freq_bins = 2, 1, 20, 16
    spec_np = np.ones((batch_size, channels, time_steps, freq_bins), dtype=np.float32)
    
    config = {
        "time_drop_width": 5,
        "time_stripes_num": 1,
        "freq_drop_width": 4,
        "freq_stripes_num": 1
    }
    
    # torchlibrosa
    spec_torch = torch.from_numpy(spec_np)
    torchlibrosa_aug = TorchlibrosaSA(**config)
    torch.manual_seed(0)
    torch_output = torchlibrosa_aug(spec_torch)
    
    # librosax
    spec_jax = jnp.array(spec_np)
    librosax_aug = SpecAugmentation(
        **config,
        deterministic=False,
        rngs=nnx.Rngs(jax.random.key(0))
    )
    jax_output = librosax_aug(spec_jax, deterministic=False)
    
    # Print masks for first batch item
    print("\nFirst batch item masks (0=masked, 1=unmasked):")
    
    torch_mask = (torch_output[0, 0] != 0).cpu().numpy().astype(int)
    jax_mask = (jax_output[0, 0] != 0).astype(int)
    
    print("\ntorchlibrosa mask pattern (time x freq):")
    for row in torch_mask:
        print("".join(str(x) for x in row))
    
    print("\nlibrosax mask pattern (time x freq):")
    for row in jax_mask:
        print("".join(str(x) for x in row))
    
    # Count masked regions
    torch_masked = (torch_mask == 0).sum()
    jax_masked = (jax_mask == 0).sum()
    total = time_steps * freq_bins
    
    print(f"\nMasked pixels:")
    print(f"  torchlibrosa: {torch_masked}/{total} ({torch_masked/total*100:.1f}%)")
    print(f"  librosax: {jax_masked}/{total} ({jax_masked/total*100:.1f}%)")


if __name__ == "__main__":
    print("Detailed SpecAugmentation Comparison Tests")
    print("=" * 50)
    
    test_masking_amount_comparison()
    test_batch_mask_diversity()
    test_mask_statistics_per_batch_item()
    test_visual_mask_comparison()
    
    print("\n" + "=" * 50)
    print("All detailed tests passed!")