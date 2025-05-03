# CQT Implementation in librosax

This document provides a quick reference for the Constant-Q Transform (CQT) implementation in librosax. For comprehensive documentation including testing coverage and detailed comparisons, see [CQT_COMPREHENSIVE.md](CQT_COMPREHENSIVE.md).

## Overview

librosax provides two CQT implementations:
- `cqt()`: The default CQT1992v2 algorithm using FFT-based convolution
- `cqt2010()`: The memory-efficient CQT2010v2 algorithm using multi-resolution approach

Both implementations support multiple output formats, normalization options, and batch processing.

## Key Features

### 1. **Multiple Algorithms**

#### CQT1992v2 (`cqt()`)
- Uses FFT-based convolution with pre-computed CQT kernels
- Efficient for moderate-length signals
- Default choice for general use

#### CQT2010v2 (`cqt2010()`)
- Uses multi-resolution approach with downsampling
- Creates small kernel for top octave only
- Iteratively downsamples for lower octaves
- Memory usage: ~1MB per second of audio
- Ideal for long signals or memory-constrained applications

### 2. **Output Formats**

Both functions support three output formats via the `output_format` parameter:
- `'magnitude'`: Returns magnitude spectrum (default for most applications)
- `'complex'`: Returns complex-valued CQT
- `'phase'`: Returns phase as (cos, sin) pairs

### 3. **Normalization Options**

The `cqt()` function supports different normalization types:
- `'librosa'`: Standard librosa normalization with sqrt(filter_lengths) scaling (default)
- `'convolutional'`: No additional normalization
- `'wrap'`: Multiplies output by 2.0

### 4. **Early Downsampling**

The `cqt2010()` function includes early downsampling optimization:
- Enabled with `earlydownsample=True` (default)
- Reduces computation for low frequencies
- Provides 1.3-2x speedup with minimal accuracy impact

### 5. **Batch Processing**

Both implementations handle batch processing automatically:
- 1D input: Single audio signal
- 2D input: Batch of audio signals
- Output preserves input batch dimension

## Usage Examples

### Basic Usage

```python
import jax.numpy as jnp
from librosax.feature.spectral import cqt, cqt2010

# Load audio
audio = jnp.array(audio_data)  # Can be 1D or 2D (batch)

# Standard CQT
C = cqt(audio, sr=22050, hop_length=512, n_bins=84)

# Memory-efficient CQT2010
C_2010 = cqt2010(audio, sr=22050, hop_length=512, n_bins=84)
```

### Basic Pitch Configuration

For compatibility with Basic Pitch:

```python
cqt_result = cqt2010(
    audio,
    sr=22050,
    hop_length=512,
    fmin=32.70,  # C1
    n_bins=264,  # Basic Pitch uses 264 bins
    bins_per_octave=36,  # 3 bins per semitone
    filter_scale=1.0,
    norm=1,
    window='hann',
    scale=True,
    pad_mode='reflect',
    output_format='magnitude',
    earlydownsample=True,
)
```

### Different Output Formats

```python
# Complex-valued CQT
C_complex = cqt(audio, sr=22050, output_format='complex')

# Phase representation
C_phase = cqt(audio, sr=22050, output_format='phase')

# Magnitude (default)
C_mag = cqt(audio, sr=22050, output_format='magnitude')
```

### Normalization Options

```python
# Different normalizations (only for cqt())
C_librosa = cqt(audio, normalization_type='librosa')  # Default
C_conv = cqt(audio, normalization_type='convolutional')
C_wrap = cqt(audio, normalization_type='wrap')
```

## Technical Details

### Implementation Approach

1. **Kernel Creation**: CQT kernels are created using scipy's window functions for numerical precision
2. **Convolution**: 
   - `cqt()`: Uses JAX's FFT-based STFT computation
   - `cqt2010()`: Uses JAX's `lax.conv_general_dilated` for strided convolution
3. **Padding**: Consistent padding ensures predictable output sizes across octaves
4. **Data Types**: Automatic conversion to float32 for compatibility

### Memory and Performance

- **CQT1992v2**: Higher memory usage but simpler implementation
- **CQT2010v2**: ~1MB per second of audio, 2-5x faster for long signals
- **Early Downsampling**: Additional 1.3x speedup with minimal quality loss

### JAX Optimization

Both implementations are JIT-compilable for optimal performance:
```python
from jax import jit

# JIT compile with static arguments
cqt_jit = jit(cqt, static_argnames=['sr', 'hop_length', 'n_bins', 'output_format'])
```

## Testing

Comprehensive tests are consolidated in:
- `test_cqt_consolidated.py`: All CQT tests including performance, dimensions, and comprehensive validation
- `test_cqt_nnaudio_ground_truth.py`: Ground truth comparison with nnAudio

Run tests with:
```bash
cd tests
pytest test_cqt_consolidated.py -v
pytest test_cqt_nnaudio_ground_truth.py -v
```

## Choosing the Right Implementation

### Use `cqt()` when:
- You need standard CQT for general audio analysis
- You want maximum compatibility with existing code
- You need specific normalization options

### Use `cqt2010()` when:
- Processing long audio files
- Memory efficiency is important
- You need Basic Pitch compatibility
- Working with low-frequency content

## API Reference

### `cqt()`
```python
cqt(y, *, sr=22050, hop_length=512, fmin=None, n_bins=84, 
    bins_per_octave=12, tuning=0.0, filter_scale=1.0, 
    norm=1.0, window='hann', scale=True, pad_mode='constant',
    output_format='complex', normalization_type='librosa')
```

### `cqt2010()`
```python
cqt2010(y, *, sr=22050, hop_length=512, fmin=None, fmax=None,
        n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1.0,
        norm=1.0, window='hann', scale=True, pad_mode='reflect',
        output_format='magnitude', earlydownsample=True)
```

Both functions accept 1D or 2D input and preserve batch dimensions in the output.