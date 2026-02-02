Scope and Design Philosophy
===========================

What Librosax IS
----------------

**Librosax is a JAX-based audio processing and feature extraction library.**

It provides:

Core Audio Processing
^^^^^^^^^^^^^^^^^^^^^

* **Spectral analysis**: STFT, iSTFT, magnitude scaling (power_to_db, amplitude_to_db)
* **Time-frequency transforms**: Constant-Q Transform (CQT), mel spectrograms
* **Feature extraction**: Spectral features, chromagrams, MFCC, rhythm features
* **Filter banks**: Mel filters, chroma filters, constant-Q filters

Music Information Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Frequency conversions**: Hz ↔ MIDI ↔ note ↔ mel ↔ octaves
* **Time conversions**: Frames ↔ samples ↔ time
* **Notation**: Key signatures, scales, intervals, just intonation
* **Tuning**: A4 reference, svara (Indian classical music)

Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^

* **Flax NNX modules**: Spectrogram, MFCC, LogMelFilterBank layers
* **Data augmentation**: SpecAugmentation, DropStripes

JAX-Compatible Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* All functions work with JAX arrays
* JIT-compilable (with proper ``static_argnames``)
* Compatible with JAX transformations (grad, vmap, jit)
* GPU/TPU acceleration support via JAX


What Librosax is NOT
--------------------

**Librosax is NOT an audio I/O library.**

NOT Implemented (Use librosa or other libraries instead)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Audio File I/O**

* **Reading audio files**: ``load()``, ``stream()``
* **Writing audio files**: ``output.write_wav()``
* **Audio codecs**: MP3, FLAC, OGG, WAV decoding/encoding

*Why not?*

* File I/O is not compatible with JAX's functional paradigm
* Existing libraries (soundfile, audioread, librosa) handle this well
* JAX is designed for numerical computation, not file handling

*What to use instead:*

.. code-block:: python

    import librosa
    import librosax

    # Load audio with librosa
    y, sr = librosa.load('audio.wav')

    # Process with librosax
    S = librosax.stft(y)
    mel = librosax.feature.melspectrogram(S=S)

**Audio Effects and Manipulation**

* **Resampling**: ``resample()`` - use librosa or scipy
* **Time stretching**: ``effects.time_stretch()`` - use librosa
* **Pitch shifting**: ``effects.pitch_shift()`` - use librosa
* **Harmonic-percussive separation**: ``decompose.hpss()`` - use librosa
* **Audio effects**: reverb, EQ, filters - use specialized libraries

*Why not?*

* These are better handled by libraries specialized for audio DSP
* Many require stateful processing or non-JAX-compatible operations
* Librosax focuses on analysis, not manipulation

**Display and Visualization**

* **Plotting**: ``display.specshow()``, ``display.waveshow()``
* **Time-frequency display utilities**

*Why not?*

* Visualization is separate from computation
* Use matplotlib directly or librosa.display

*What to use instead:*

.. code-block:: python

    import librosax
    import librosa.display
    import matplotlib.pyplot as plt

    # Compute with librosax
    mel = librosax.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosax.power_to_db(mel)

    # Visualize with librosa.display
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.show()


Recommended Workflow
--------------------

For Audio Analysis Tasks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import librosa          # For I/O and visualization
    import librosax         # For processing and features
    import jax.numpy as jnp

    # 1. Load audio (use librosa)
    y, sr = librosa.load('audio.wav')

    # 2. Process with librosax (JAX-compatible)
    S = librosax.stft(y)
    mel = librosax.feature.melspectrogram(S=S, sr=sr)
    mfcc = librosax.feature.mfcc(S=mel)

    # 3. Compute deltas (temporal features)
    mfcc_delta = librosax.feature.delta(mfcc)
    mfcc_delta2 = librosax.feature.delta(mfcc, order=2)

    # 4. Use with JAX transformations
    from jax import jit, vmap

    @jit
    def process_batch(audio_batch):
        return vmap(librosax.feature.melspectrogram)(audio_batch)

    # 5. Visualize (use librosa.display)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')

For Machine Learning Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import librosax
    import jax
    import jax.numpy as jnp
    from flax import nnx

    # Use librosax layers in neural networks
    class AudioModel(nnx.Module):
        def __init__(self, rngs):
            # Librosax provides JAX/Flax layers
            self.spec = librosax.layers.Spectrogram(
                n_fft=2048,
                hop_length=512,
                rngs=rngs
            )
            self.mfcc = librosax.layers.MFCC(
                sr=22050,
                n_mfcc=20,
                rngs=rngs
            )
            # ... rest of model

        def __call__(self, audio):
            # End-to-end differentiable!
            S = self.spec(audio)
            features = self.mfcc(S)
            return features

    # JIT-compile the entire pipeline
    model = AudioModel(rngs=nnx.Rngs(0))
    process = jax.jit(model)


Hybrid Approach: Best of Both Worlds
------------------------------------

**Use librosax for:**

* Feature extraction (mel, MFCC, chroma, spectral features)
* Spectral analysis (STFT, CQT)
* JAX/Flax neural network layers
* GPU-accelerated batch processing
* Differentiable audio pipelines
* Conversion utilities (time, frequency, notation)

**Use librosa for:**

* Loading/saving audio files
* Resampling and audio effects
* Visualization (specshow, waveshow)
* Advanced decomposition (HPSS, NMF)
* Beat tracking and onset detection (if not using JAX)

**The two libraries are complementary, not replacements!**


Migration Guide: From Librosa to Librosax
-----------------------------------------

If you have existing librosa code and want to use librosax for JAX compatibility:

What Changes
^^^^^^^^^^^^

.. code-block:: python

    # BEFORE (librosa only)
    import librosa
    y, sr = librosa.load('audio.wav')
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(S=mel)

    # AFTER (librosa + librosax)
    import librosa  # Keep for I/O
    import librosax # Use for processing
    y, sr = librosa.load('audio.wav')
    mel = librosax.feature.melspectrogram(y=y, sr=sr)  # Now JAX-compatible!
    mfcc = librosax.feature.mfcc(S=mel)

What Stays the Same
^^^^^^^^^^^^^^^^^^^

* Function signatures (mostly identical)
* Parameter names and defaults
* Return value shapes and meanings
* Numerical results (validated against librosa)

What's Different
^^^^^^^^^^^^^^^^

* Returns JAX arrays instead of NumPy arrays
* Can be JIT-compiled with ``jax.jit()``
* Works with JAX transformations (vmap, grad, etc.)
* GPU/TPU acceleration available
* No file I/O capabilities


Design Principles
-----------------

1. **Pure Computation**: Librosax handles mathematical transformations, not I/O
2. **JAX-First**: All implementations use JAX arrays and operations
3. **API Compatible**: Match librosa's interface where possible
4. **Complementary**: Designed to work alongside librosa, not replace it
5. **Differentiable**: Support gradient-based optimization and learning


Future Scope
------------

Planned
^^^^^^^

* More feature extraction functions (as requested)
* Additional neural network layers
* More efficient JAX implementations of key algorithms

Not Planned
^^^^^^^^^^^

* Audio I/O (use librosa, soundfile)
* Audio effects (use librosa, pedalboard)
* Visualization (use librosa.display, matplotlib)
* Onset detection (unless pure JAX implementation requested)
* Beat tracking (unless pure JAX implementation requested)


FAQ
---

**Q: Why not implement everything librosa has?**

A: Librosax focuses on JAX-compatible numerical computation. File I/O, visualization, and effects are better handled by specialized libraries. This keeps librosax focused and maintainable.

**Q: Can I use both librosa and librosax together?**

A: Yes! This is the recommended approach. Use librosa for I/O and visualization, librosax for processing and feature extraction.

**Q: Will librosax replace librosa?**

A: No. They serve different purposes and complement each other. Librosa is a comprehensive audio analysis library; librosax is a JAX-compatible processing library.

**Q: What if I need a function that librosax doesn't have?**

A: Use librosa! You can mix and match. Load with librosa, process with librosax, visualize with librosa.display.
