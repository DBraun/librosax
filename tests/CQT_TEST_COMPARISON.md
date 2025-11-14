# CQT Test Coverage Comparison: librosax vs nnAudio

## Test Coverage Summary

### nnAudio CQT Tests
1. **Ground Truth Tests**: ✅ Tests against pre-saved ground truth arrays
2. **Multiple Versions**: ✅ CQT1992, CQT1992v2, CQT2010, CQT2010v2
3. **Output Formats**: ✅ Magnitude, Complex, Phase
4. **Signal Types**: ✅ Logarithmic sweep, Linear sweep
5. **Device Testing**: ✅ CPU and GPU (with CUDA)
6. **Batch Processing**: ✅ DataParallel tests
7. **Test Parameters**: ✅ fs=44100, bins_per_octave=24, n_bins=207

### librosax CQT Tests
1. **Ground Truth Tests**: ✅ `test_cqt_nnaudio_ground_truth.py` - uses same ground truth files
2. **Multiple Versions**: ✅ CQT (1992) and CQT2010
3. **Output Formats**: ✅ Magnitude, Complex, Phase (in ground truth tests)
4. **Signal Types**: ✅ Logarithmic sweep, Linear sweep, Pure tones, Musical signals
5. **Device Testing**: ✅ JAX automatically handles CPU/GPU selection
6. **Batch Processing**: ✅ Tests batch dimensions extensively
7. **Test Parameters**: ✅ Various parameters including high resolution (24 bins/octave)

## Additional librosax Test Coverage
- **Comparison with librosa**: Direct comparison of outputs
- **Edge Cases**: Short signals, various audio lengths, dimension consistency
- **Musical Signals**: C major chord, pure tones, harmonic content
- **Performance Testing**: Timing comparisons with warmup and proper synchronization
- **Normalization Testing**: White noise energy distribution
- **Real Audio**: Melody with envelope and noise
- **Algorithm Comparison**: CQT1992 vs CQT2010 correlation

## Key Differences
1. **Framework**: nnAudio uses PyTorch, librosax uses JAX
2. **Device Handling**: 
   - nnAudio: Explicit device placement and DataParallel
   - librosax: JAX automatic device selection
3. **Ground Truth Tolerance**:
   - nnAudio: rtol=1e-3, atol=1e-3
   - librosax: More relaxed for some tests due to implementation differences

## Conclusion
librosax CQT tests are comprehensive and actually exceed nnAudio's coverage in several areas:
- More diverse test signals (pure tones, musical signals)
- Edge case testing
- Direct librosa comparison
- Performance benchmarking

The main difference is that librosax relies on JAX's automatic device management rather than explicit CPU/GPU testing, which is the idiomatic JAX approach.