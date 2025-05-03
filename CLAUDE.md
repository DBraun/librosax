# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Librosax is a JAX implementation of audio processing functions inspired by [librosa](https://github.com/librosa/librosa), [TorchLibrosa](https://github.com/qiuqiangkong/torchlibrosa/), and [nnAudio](https://github.com/KinWaiCheuk/nnAudio). It provides:

- Core audio processing functions (STFT, iSTFT, magnitude scaling)
- Feature extraction for audio analysis (spectral features, mel-frequency representations, chromagram, CQT)
- Neural network layers for audio processing (Spectrogram, MFCC, data augmentation)

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests

# Run specific test file
python -m pytest tests/test_librosax.py

# Download test data (required before running tests)
python scripts/download_test_data.py
```

### Documentation
```bash
# Build documentation
cd docs
make html
```

### Installation
```bash
# Install in development mode
pip install -e ".[dev]"

# Set PYTHONPATH for development
export PYTHONPATH=src:$PYTHONPATH
```

## Architecture

### Core Structure

The structure follows `librosa` closely.

- `src/librosax/` - Main package directory
  - `__init__.py` - Top-level API (`stft`, `istft`, `power_to_db`, `amplitude_to_db`, `fft_frequencies`)
  - `core/` - Core functionality (spectrum, convert, notation, intervals)
  - `feature/` - Feature extraction functions (spectral analysis, mel-frequency, chromagram, CQT)
  - `layers/` - Neural network layers using `flax.nnx.Module` (`Spectrogram`, `MFCC`, `LogMelFilterBank`, `DropStripes`, `SpecAugmentation`)
  - `util/` - Utility functions (decorators, exceptions, file handling)

### Import Patterns
- Core functions are available at top level: `import librosax; librosax.stft()`
- Feature functions: `import librosax.feature; librosax.feature.melspectrogram()`
- Neural layers: `from librosax.layers import Spectrogram, MFCC`

### Test Data
Tests require downloading ground truth data using `python scripts/download_test_data.py`.
Test files compare outputs with librosa and TorchLibrosa implementations for accuracy validation.

## JAX-Specific Considerations

- Some tests require 64-bit mode from JAX
- Functions are designed to work with JAX arrays and be JIT-compilable, although you should use `static_argnames`
- Random number generation uses JAX's random module
- Neural network layers use Flax/NNX module system
