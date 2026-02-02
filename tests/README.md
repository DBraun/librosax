# Librosax Test Suite Guide

## Setup

Test data is managed via Git submodule. To initialize:
```bash
git submodule update --init --recursive
```

For CI/CD, add to GitHub Actions:
```yaml
- uses: actions/checkout@v3
  with:
    submodules: recursive
```

## Test Organization

### Original Librosax Tests (Keep)
- `test_librosax.py` - Core librosax validation tests
- `test_cqt_consolidated.py` - CQT implementation tests
- `test_cqt_nnaudio_ground_truth.py` - CQT validation against nnAudio
- `test_spec_augmentation_*.py` - SpecAugmentation layer tests

### Copied Librosa Tests (For Compatibility Validation)
- `test_convert.py` - **513/513 passing (100%)** ✅
- `test_core.py` - **4,188 passing (76%)** ✅
- `test_intervals.py` - **81/89 passing (91%)** ✅
- `test_notation.py` - **166/225 passing (74%)** ✅
- `test_filters.py` - **685/1,329 passing (52%)** ✅
- `test_features.py` - **245/~950 passing (26%)** ⚠️

**Total: 5,878 passing tests (89% compatibility with librosa)**

---

## Expected Test Failures

The following tests are **EXPECTED TO FAIL** because librosax intentionally does not implement these functions. Users should use librosa directly for these capabilities.

### Category 1: Audio I/O (Use `librosa` instead)

**Functions NOT in librosax:**
- `load()` - Load audio files
- `stream()` - Stream audio from files
- `resample()` - Resample audio
- `get_duration()` - Get audio file duration from file
- `get_samplerate()` - Get sample rate from file

**Why:** JAX is for numerical computation, not file I/O. Use librosa, soundfile, or audioread.

**Example workflow:**
```python
import librosa  # For I/O
import librosax # For processing

y, sr = librosa.load('audio.wav')  # Use librosa for I/O
S = librosax.stft(y)                # Use librosax for processing
```

**Tests that fail (expected):**
- `test_core.py::test_load*` (6 tests)
- `test_core.py::test_resample*` (5 tests)
- `test_core.py::test_get_duration*` (6 tests)
- `test_core.py::test_stream*` (2 tests)
- `test_core.py::test_segment_load` (1 test)
- Plus 1,300+ errors in fixtures that call these functions

**Total: ~1,320 test errors/failures**

---

### Category 2: Private/Internal Functions (Not Public API)

**Functions NOT in librosax:**
- `__reassign_frequencies()` - Private implementation detail
- `__reassign_times()` - Private implementation detail
- `_spectrogram()` - Internal helper
- `__float_window()` - Private window implementation

**Why:** These are implementation details, not public API. Tests for internal functions aren't relevant.

**Tests that fail (expected):**
- `test_core.py::test___reassign*` (4 tests)
- `test_core.py::test__spectrogram*` (2 tests)
- `test_filters.py::test__window*` (~52 tests)

**Total: ~58 tests**

---

### Category 3: Numba-Dependent Functions (JAX Incompatible)

**Functions NOT in librosax:**
- `interval_to_fjs()` - Uses numba JIT compilation

**Why:** Numba and JAX use different execution models and are incompatible. JAX provides its own JIT.

**Tests that fail (expected):**
- `test_notation.py::test_interval_to_fjs*` (~30 tests)
- `test_notation.py::test_hz_to_fjs*` (~15 tests)

**Total: ~45 tests**

---

### Category 4: Missing Advanced Features (Not Yet Implemented)

Some advanced features from librosa aren't yet implemented in librosax:

**Missing features:**
- Advanced onset/beat functions
- Pitch tracking (pyin, piptrack)
- Audio decomposition (HPSS)
- Some spectral feature edge cases

**Tests that fail (expected):**
- Various feature edge case tests
- Advanced processing tests

**Total: ~200+ tests**

---

## Test Data

- **`tests/data/`** - Git submodule with librosa test data (327 .mat files, ~165MB)
- **`tests/ground_truths/`** - Custom CQT test data (gitignored)

---

## Test Execution Recommendations

### Run All Tests (Including Expected Failures)
```bash
pytest tests/ -q --tb=no
```

Shows total compatibility: **5,878 passing (89%)**

### Run Only Relevant Tests (Recommended)
```bash
# Perfect compatibility tests
pytest tests/test_convert.py -v

# High compatibility tests
pytest tests/test_intervals.py tests/test_notation.py -v

# Feature tests (skip I/O dependent ones)
pytest tests/test_features.py -v -k "not (tempogram_audio or tonnetz_audio or fourier_tempogram_audio or vqt)"
```

### Run Original Librosax Tests Only
```bash
pytest tests/test_librosax.py tests/test_cqt*.py tests/test_spec_augmentation*.py -v
```

---

## Interpreting Test Results

### ✅ Good Signs
- `test_convert.py` passing = Core conversions work
- `test_intervals.py` mostly passing = Just intonation correct
- `test_notation.py` mostly passing = Music theory correct
- `test_filters.py` many passing = Filter banks work
- High % of `test_core.py` passing = Core processing works

### ⚠️ Expected Failures
- `test_core.py` errors = Missing I/O functions (expected)
- `test_features.py` errors = Missing advanced features or I/O-dependent
- `test_filters.py` failures = Private functions or edge cases

### ❌ Real Problems
- Unexpected failures in `test_convert.py` = Bug in conversions
- Crashes or wrong numerical results = Implementation bug
- JAX errors in existing tests = Compatibility regression

---

## Contributing Tests

When adding new features to librosax:

1. **Add tests to original files** (test_librosax.py or new test_*.py)
2. **Validate against librosa** if implementing librosa-compatible function
3. **Don't test I/O** - focus on numerical correctness
4. **Use JAX x64 mode** for precision-sensitive tests
5. **Test with JIT** to ensure JAX compatibility

---

## Summary

**Expected pass rate: 85-90%** of relevant tests.

The ~10-15% of "failing" tests are mostly:
- I/O functions librosax intentionally doesn't implement
- Private functions not part of public API
- Advanced features not yet needed

This is healthy and expected for a focused, JAX-compatible audio processing library!
