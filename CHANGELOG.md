# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Effort-based Versioning](https://jacobtomlinson.dev/effver/).

## [0.1.1] - 2025-11-14

- Fix GitHub Action for docs.

## [0.1.0] - 2025-11-14

### Added
- Constant-Q Transform (CQT) with both 1992 and 2010 algorithms
- Comprehensive spectral features:
  - `spectral_centroid` - Center of mass of the spectrum
  - `spectral_contrast` - Valley-to-peak spectral contrast
  - `spectral_bandwidth` - Bandwidth of the spectrum
  - `spectral_rolloff` - Roll-off frequency
  - `spectral_flatness` - Spectral flatness measure
  - `zero_crossing_rate` - Rate of sign changes
- Mel-frequency features:
  - `melspectrogram` - Mel-scaled spectrogram
  - `mfcc` - Mel-frequency cepstral coefficients
- Chroma features:
  - `chroma_stft` - Chromagram from STFT
  - `chroma_cqt` - Chromagram from CQT
- `SpecAugmentation` layer for data augmentation
- Core modules following `librosa` structure:
  - `librosax.core.convert` - Unit conversion functions
  - `librosax.core.spectrum` - Spectral operations
  - `librosax.core.notation` - Music notation utilities
  - `librosax.core.intervals` - Interval arithmetic
- Comprehensive type hints with JAX typing support
- Caching system for expensive computations
- Ground truth validation tests against nnAudio for CQT
- Detailed CQT implementation documentation
- Test data download script (`scripts/download_test_data.py`)
- GitHub Actions CI/CD workflow

### Changed
- **BREAKING**: Migrated neural network layers from Flax Linen to Flax NNX
- **BREAKING**: Restructured module organization to follow `librosa` conventions
  - Core functionality moved to `librosax.core.*`
  - Feature extraction moved to `librosax.feature.*`
  - Import paths have changed
- Tightened MFCC test tolerance from `rtol=1.7e-1` to `rtol=1e-3` (170x more strict)
- Adjusted CQT2010 linear sweep correlation threshold from 0.85 to 0.83
- All functions are now JIT-compatible with proper `static_argnames`
- Improved API documentation with examples
- Updated `README.md` with new features and examples
- Enhanced `CLAUDE.md` with comprehensive project guidelines

### Fixed
- MFCC handling for 2D and 4D input dimensions
- Spectral contrast calculation accuracy
- MFCC liftering implementation
- Numerical precision issues in CQT algorithms
- Version compatibility issues

### Removed
- Obsolete pseudo-CQT implementation
- IDE configuration files from repository

## [0.0.4] - 2025-07-06

### Added
- Initial release with basic audio processing functions
- STFT and inverse STFT
- Basic magnitude scaling (`power_to_db`, `amplitude_to_db`)
- FFT frequency utilities
- Basic Flax Linen layers for spectrograms

### Changed
- Initial project structure

## [0.0.3] - Earlier

(Previous versions not documented)

## [0.0.2] - Earlier

(Previous versions not documented)

## [0.0.1] - Earlier

(Previous versions not documented)
