from typing import Optional, Union, Callable

import jax
import librosa
import numpy as np
from jax import numpy as jnp
from jax.scipy import signal as jssignal

from librosax.core.spectrum import fft_frequencies, normalize, power_to_db, stft, _spectrogram, _dct_flexible
from librosax.core.convert import note_to_hz
from librosax import filters
from librosax.util.exceptions import ParameterError
from functools import partial

__all__ = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_rolloff",
    "spectral_flatness",
    "poly_features",
    "rms",
    "zero_crossing_rate",
    "chroma_stft",
    "chroma_cqt",
    "chroma_vqt",
    "chroma_cens",
    "melspectrogram",
    "mfcc",
    "tonnetz",
    "cqt",
    "cqt2010",
    "cqt_frequencies",
]


def spectral_centroid(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq: Optional[jnp.ndarray] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> jnp.ndarray:
    """Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    More precisely, the centroid at frame t is defined as:
        centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    where S is a magnitude spectrogram, and freq is the array of frequencies
    (e.g., FFT frequencies in Hz) of the rows of S.

    Users should ensure S is real-valued and non-negative.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        freq: Center frequencies for spectrogram bins. If None, FFT bin center
            frequencies are used.
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode

    Returns:
        Spectral centroid frequencies with shape ``(..., 1, N)`` where N is the
        number of frames. The frequency axis is reduced to size 1.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Ensure freq has the right shape for broadcasting
    if freq.ndim == 1:
        # Reshape freq to match S dimensions
        # S has shape (..., frequency, time)
        freq = freq.reshape((-1,) + (1,) * (S.ndim - freq.ndim))

    # Column-normalize S
    # norm=1 means L1 norm (sum of absolute values)
    centroid = jnp.sum(freq * normalize(S, norm=1, axis=-2), axis=-2, keepdims=True)

    return centroid


def spectral_bandwidth(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: Optional[jnp.ndarray] = None,
    centroid: Optional[jnp.ndarray] = None,
    norm: bool = True,
    p: float = 2,
) -> jnp.ndarray:
    """Compute p'th-order spectral bandwidth.

    The spectral bandwidth at frame t is computed by:
        (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)

    Users should ensure S is real-valued and non-negative.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        freq: Center frequencies for spectrogram bins. If None, FFT bin center
            frequencies are used.
        centroid: Pre-computed centroid frequencies
        norm: Normalize per-frame spectral energy (sum to one)
        p: Power to raise deviation from spectral centroid

    Returns:
        Frequency bandwidth with shape ``(..., 1, N)``.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # If we don't have a centroid provided, compute it
    if centroid is None:
        centroid = spectral_centroid(
            y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Calculate deviation from centroid
    if freq.ndim == 1:
        # Use outer subtraction
        # centroid has shape (..., 1, time), extract (..., time)
        centroid_squeezed = centroid[..., 0, :]
        # subtract.outer gives shape (time, freq), need to swap axes
        deviation = jnp.abs(jnp.subtract.outer(centroid_squeezed, freq).swapaxes(-2, -1))
    else:
        deviation = jnp.abs(freq - centroid)

    # Column-normalize S if requested
    if norm:
        S = normalize(S, norm=1, axis=-2)

    # Compute bandwidth
    bw = jnp.sum(S * deviation**p, axis=-2, keepdims=True) ** (1.0 / p)

    return bw


def spectral_rolloff(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: Optional[jnp.ndarray] = None,
    roll_percent: float = 0.85,
) -> jnp.ndarray:
    """Compute roll-off frequency.

    The roll-off frequency is defined for each frame as the center frequency
    for a spectrogram bin such that at least roll_percent (0.85 by default)
    of the energy of the spectrum in this frame is contained in this bin and
    the bins below. Users should ensure S is real-valued and non-negative.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        freq: Center frequencies for spectrogram bins. If None, FFT bin center
            frequencies are used. Assumed to be sorted in increasing order.
        roll_percent: Roll-off percentage (0 < roll_percent < 1)

    Returns:
        Roll-off frequency with shape ``(..., 1, N)``.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    if not 0.0 < roll_percent < 1.0:
        raise ParameterError("roll_percent must lie in the range (0, 1)")

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        # Reshape freq to match S dimensions
        freq = freq.reshape((-1,) + (1,) * (S.ndim - freq.ndim))

    # Compute cumulative energy
    total_energy = jnp.cumsum(S, axis=-2)

    # Get threshold energy
    threshold = roll_percent * total_energy[..., -1, :]

    # Reshape threshold for broadcasting
    threshold = jnp.expand_dims(threshold, axis=-2)

    # Find where cumulative energy exceeds threshold
    # Use where to set values below threshold to nan
    ind = jnp.where(total_energy < threshold, jnp.nan, 1)

    # Get the minimum frequency that meets the threshold
    rolloff = jnp.nanmin(ind * freq, axis=-2, keepdims=True)

    return rolloff


def spectral_flatness(
    *,
    y: Optional[jnp.ndarray] = None,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    amin: float = 1e-10,
    power: float = 2.0,
) -> jnp.ndarray:
    """Compute spectral flatness.

    Spectral flatness (or tonality coefficient) is a measure to quantify
    how much noise-like a sound is, as opposed to being tone-like.
    A high spectral flatness (closer to 1.0) indicates the spectrum is
    similar to white noise. Users should ensure S is real-valued and
    non-negative.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        amin: Minimum threshold for S (added noise floor for numerical stability)
        power: Exponent for the magnitude spectrogram (e.g., 1 for energy, 2 for power)

    Returns:
        Spectral flatness with shape ``(..., 1, N)``.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Apply power and threshold
    S_thresh = jnp.maximum(amin, S**power)

    # Compute geometric mean (using log for numerical stability)
    gmean = jnp.exp(jnp.mean(jnp.log(S_thresh), axis=-2, keepdims=True))

    # Compute arithmetic mean
    amean = jnp.mean(S_thresh, axis=-2, keepdims=True)

    # Spectral flatness is the ratio of geometric to arithmetic mean
    flatness = gmean / amean

    return flatness


def frame(x: jnp.ndarray, frame_length: int, hop_length: int, axis: int = -1) -> jnp.ndarray:
    """Slice a data array into overlapping frames.

    This implementation uses JAX operations to create a sliding window view.

    Args:
        x: Input array to frame
        frame_length: Length of each frame
        hop_length: Number of steps to advance between frames
        axis: The axis along which to frame (default: -1)

    Returns:
        jnp.ndarray: Framed view of the input array with an additional dimension.
        The output shape has frames as a new dimension after the framed axis.
    """
    if axis < 0:
        axis = x.ndim + axis

    # Get the shape and create index arrays
    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length

    # For 1D input, we can use a simpler approach
    if x.ndim == 1:
        indices = jnp.arange(frame_length)[None, :] + (jnp.arange(n_frames) * hop_length)[:, None]
        frames = x[indices]
        # Shape is now (n_frames, frame_length), need to swap to (frame_length, n_frames)
        return frames.T

    # Use JAX's dynamic_slice with vmap for multidimensional arrays
    def get_frame(i):
        start_indices = [0] * x.ndim
        start_indices[axis] = i * hop_length
        slice_sizes = list(x.shape)
        slice_sizes[axis] = frame_length
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)

    frames = jax.vmap(get_frame)(jnp.arange(n_frames))

    # Rearrange dimensions to match librosa's output
    # frames currently has shape (n_frames, ...), we need (..., frame_length, n_frames)
    # Move the frame dimension to be the last dimension
    perm = list(range(1, frames.ndim)) + [0]
    frames = jnp.transpose(frames, perm)

    return frames


def abs2(x: jnp.ndarray, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Compute the squared magnitude of a real or complex array.

    Args:
        x: Input array (real or complex)
        dtype: Optional output data type

    Returns:
        jnp.ndarray: Squared magnitude
    """
    if jnp.iscomplexobj(x):
        result = jnp.real(x)**2 + jnp.imag(x)**2
    else:
        result = jnp.square(x)

    if dtype is not None:
        result = result.astype(dtype)

    return result


def rms(
    *,
    y: Optional[jnp.ndarray] = None,
    S: Optional[jnp.ndarray] = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute root-mean-square (RMS) value for each frame.

    Computing the RMS value from audio samples is faster as it doesn't require
    a STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed.

    Args:
        y: (optional) Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        S: (optional) Spectrogram magnitude with shape ``(..., F, N)``
        frame_length: Length of analysis frame (in samples) for energy calculation
        hop_length: Hop length for STFT
        center: If True and operating on time-domain input (y), pad the signal
            by frame_length//2 on either side. Has no effect on spectrogram input.
        pad_mode: Padding mode for centered analysis
        dtype: Data type of the output array

    Returns:
        RMS value with shape ``(..., 1, N)``.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    if y is not None:
        if center:
            # Pad the signal
            pad_width = [(0, 0) for _ in range(y.ndim)]
            pad_width[-1] = (frame_length // 2, frame_length // 2)
            y = jnp.pad(y, pad_width, mode=pad_mode)

        # Frame the signal
        x = frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = jnp.mean(abs2(x, dtype=dtype), axis=-2, keepdims=True)

    elif S is not None:
        # Note: Runtime checks are not compatible with JIT compilation
        # Users should ensure S.shape[-2] == frame_length // 2 + 1
        if S.shape[-2] != frame_length // 2 + 1:
            raise ParameterError(
                f"Since S.shape[-2] is {S.shape[-2]}, "
                f"frame_length is expected to be {S.shape[-2] * 2 - 2} or {S.shape[-2] * 2 - 1}; "
                f"found {frame_length}"
            )

        # Power spectrogram
        x = abs2(S, dtype=dtype)

        # Adjust the DC and sr/2 component
        # Create a copy to modify
        x = x.at[..., 0, :].multiply(0.5)
        if frame_length % 2 == 0:
            x = x.at[..., -1, :].multiply(0.5)

        # Calculate power
        power = 2 * jnp.sum(x, axis=-2, keepdims=True) / frame_length**2
    else:
        raise ParameterError("Either y or S must be input.")

    return jnp.sqrt(power)


def zero_crossings(
    y: jnp.ndarray,
    *,
    threshold: float = 1e-10,
    ref_magnitude: Optional[Union[float, Callable]] = None,
    pad: bool = True,
    zero_pos: bool = True,
    axis: int = -1,
) -> jnp.ndarray:
    """Find zero-crossings of a signal.

    Zero-crossings are indices i such that sign(y[i]) != sign(y[i+1]).

    Args:
        y: Input array
        threshold: If non-zero, values where -threshold <= y <= threshold are
            clipped to 0.
        ref_magnitude: If numeric, the threshold is scaled relative to ref_magnitude.
            If callable, the threshold is scaled relative to ref_magnitude(abs(y)).
        pad: If True, then y[0] is considered a valid zero-crossing.
        zero_pos: If True then the value 0 is interpreted as having positive sign.
            If False, then 0, -1, and +1 all have distinct signs.
        axis: Axis along which to compute zero-crossings.

    Returns:
        jnp.ndarray: Boolean array indicating zero-crossings
    """
    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(jnp.abs(y))
    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    # Apply threshold
    y_thresh = jnp.where(jnp.abs(y) <= threshold, 0, y)

    # Get sign of consecutive elements
    if zero_pos:
        # Use signbit for zero_pos mode (0 has positive sign)
        signs = jnp.signbit(y_thresh)
    else:
        # Use sign for regular mode
        signs = jnp.sign(y_thresh)

    # Compute differences along the specified axis
    # Zero crossing occurs when signs differ
    diff = jnp.diff(signs, axis=axis)

    # Create output array with proper shape
    z_shape = list(y.shape)
    z_shape[axis] = y.shape[axis] - 1
    z = diff != 0

    # Pad to match input shape if requested
    if pad:
        pad_width = [(0, 0) for _ in range(y.ndim)]
        pad_width[axis] = (1, 0)
        z = jnp.pad(z, pad_width, mode='constant', constant_values=True)
    else:
        # Need to pad with False to match shape
        pad_width = [(0, 0) for _ in range(y.ndim)]
        pad_width[axis] = (1, 0)
        z = jnp.pad(z, pad_width, mode='constant', constant_values=False)

    return z


def zero_crossing_rate(
    y: jnp.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    **kwargs,
) -> jnp.ndarray:
    """Compute the zero-crossing rate of an audio time series.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        frame_length: Length of the frame over which to compute zero crossing rates
        hop_length: Number of samples to advance for each frame
        center: If True, frames are centered by padding the edges of y.
            Uses edge-value copies instead of zero-padding.
        **kwargs: Additional keyword arguments to pass to zero_crossings

    Returns:
        Zero crossing rate with shape ``(..., 1, N)``.

        - ``(T,)`` → ``(1, N)``
        - ``(B, T)`` → ``(B, 1, N)``
    """
    if center:
        # Pad with edge values
        pad_width = [(0, 0) for _ in range(y.ndim)]
        pad_width[-1] = (frame_length // 2, frame_length // 2)
        y = jnp.pad(y, pad_width, mode='edge')

    # Frame the signal
    y_framed = frame(y, frame_length=frame_length, hop_length=hop_length)

    # Set default pad=False for zero_crossings within frames
    kwargs.setdefault('pad', False)
    kwargs['axis'] = -2

    # Compute zero crossings for each frame
    crossings = zero_crossings(y_framed, **kwargs)

    # Average over frame dimension
    zcr = jnp.mean(crossings, axis=-2, keepdims=True)

    return zcr


def spectral_contrast(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: Optional[jnp.ndarray] = None,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> jnp.ndarray:
    """Compute spectral contrast.

    Each frame of a spectrogram S is divided into sub-bands.
    For each sub-band, the energy contrast is estimated by comparing
    the mean energy in the top quantile (peak energy) to that of the
    bottom quantile (valley energy). High contrast values generally
    correspond to clear, narrow-band signals, while low contrast values
    correspond to broad-band noise.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        freq: Center frequencies for spectrogram bins. If None, FFT bin center
            frequencies are used.
        fmin: Frequency cutoff for the first bin [0, fmin]
            Subsequent bins will cover [fmin, 2*fmin], [2*fmin, 4*fmin], etc.
        n_bands: Number of frequency bands
        quantile: Quantile for determining peaks and valleys
        linear: If True, return the linear difference of magnitudes: peaks - valleys.
            If False, return the logarithmic difference: log(peaks) - log(valleys).

    Returns:
        Spectral contrast with shape ``(..., n_bands + 1, N)``.

        - ``(T,)`` → ``(n_bands + 1, N)``
        - ``(B, T)`` → ``(B, n_bands + 1, N)``
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    freq = jnp.atleast_1d(freq)

    if freq.ndim != 1 or len(freq) != S.shape[-2]:
        raise ParameterError(f"freq.shape mismatch: expected ({S.shape[-2]},)")

    if n_bands < 1 or not isinstance(n_bands, int):
        raise ParameterError("n_bands must be a positive integer")

    if not 0.0 < quantile < 1.0:
        raise ParameterError("quantile must lie in the range (0, 1)")

    if fmin <= 0:
        raise ParameterError("fmin must be a positive number")

    # Create octave bands
    octa = jnp.zeros(n_bands + 2)
    octa = octa.at[1:].set(fmin * (2.0 ** jnp.arange(0, n_bands + 1)))

    # Calculate maximum possible quantile index size
    # The largest band is typically the last one, which can extend to all remaining bins
    # from the lowest frequency of that band to the Nyquist frequency
    # In the worst case, this could be nearly all frequency bins
    max_possible_band_size = S.shape[-2]  # Conservative upper bound
    # Use Python int() to ensure it's a static value
    max_quantile_size = int(max_possible_band_size * float(quantile)) + 1
    
    # Process bands using scan for JIT compatibility
    def process_band(carry, k):
        S, freq, octa, sr, n_bands, quantile = carry
        
        f_low = octa[k]
        f_high = jax.lax.cond(
            k < n_bands,
            lambda: octa[k + 1],
            lambda: sr / 2
        )
        
        # Create band mask
        band_mask = jnp.logical_and(freq >= f_low, freq <= f_high)
        
        # Find first and last indices
        # Use cumsum trick to find positions
        cumsum = jnp.cumsum(band_mask)
        has_true = cumsum[-1] > 0
        
        # Find first True (where cumsum goes from 0 to 1)
        first_idx = jnp.where(has_true,
                             jnp.argmax(cumsum > 0),
                             0)
        
        # Find last True
        reverse_cumsum = jnp.cumsum(band_mask[::-1])
        last_idx = jnp.where(has_true,
                            len(band_mask) - 1 - jnp.argmax(reverse_cumsum > 0),
                            0)
        
        # Adjust boundaries as in librosa
        # For k > 0, include one bin below
        first_idx = jax.lax.cond(
            jnp.logical_and(k > 0, first_idx > 0),
            lambda: first_idx - 1,
            lambda: first_idx
        )
        
        # For the last band, extend to the end
        last_idx = jax.lax.cond(
            k == n_bands,
            lambda: len(freq) - 1,
            lambda: last_idx
        )
        
        # Update band mask with adjusted boundaries
        indices = jnp.arange(len(freq))
        band_mask_adjusted = jnp.logical_and(indices >= first_idx, indices <= last_idx)
        
        # For non-final bands, exclude the last bin
        band_mask_final = jax.lax.cond(
            k < n_bands,
            lambda: jnp.logical_and(band_mask_adjusted, indices < last_idx),
            lambda: band_mask_adjusted
        )
        
        # Count bins for quantile calculation (use original band mask count)
        n_bins_for_quantile = jnp.sum(band_mask_adjusted)
        
        # Extract sub-band values
        # Use masking approach
        sub_band_values = jnp.where(band_mask_final[:, None], S, -jnp.inf)
        
        # Sort along frequency axis
        sorted_sub = jnp.sort(sub_band_values, axis=0)
        
        # Find where valid values start (first non-(-inf) value)
        valid_mask = sorted_sub > -jnp.inf
        n_valid = jnp.sum(valid_mask, axis=0)
        
        # Calculate quantile index
        n_idx = jnp.maximum(jnp.rint(quantile * n_bins_for_quantile).astype(jnp.int32), 1)
        
        # For each time frame, we need to extract bottom n_idx and top n_idx values
        # max_quantile_size is captured from the outer scope as a static value
        
        # Create index arrays for gathering
        time_frames = S.shape[-1]
        frame_indices = jnp.arange(time_frames)
        idx_range = jnp.arange(max_quantile_size)
        
        # Find first valid index per frame
        first_valid_idx = jnp.argmax(valid_mask, axis=0)
        
        # Valley indices: first n_idx valid values
        valley_indices = first_valid_idx[None, :] + idx_range[:, None]
        valley_indices = jnp.clip(valley_indices, 0, sorted_sub.shape[0] - 1)
        
        # Gather valley values
        valley_values = sorted_sub[valley_indices, frame_indices]
        valley_mask = idx_range[:, None] < jnp.minimum(n_idx, n_valid)[None, :]
        valley_values = jnp.where(valley_mask, valley_values, 0)
        valley = jnp.sum(valley_values, axis=0) / jnp.maximum(jnp.sum(valley_mask, axis=0), 1)
        
        # Peak indices: last n_idx valid values
        last_valid_idx = first_valid_idx + n_valid - 1
        peak_start_idx = jnp.maximum(last_valid_idx - n_idx + 1, first_valid_idx)
        
        peak_indices = peak_start_idx[None, :] + idx_range[:, None]
        peak_indices = jnp.clip(peak_indices, 0, sorted_sub.shape[0] - 1)
        
        # Gather peak values
        peak_values = sorted_sub[peak_indices, frame_indices]
        peak_mask = idx_range[:, None] < jnp.minimum(n_idx, n_valid)[None, :]
        peak_values = jnp.where(peak_mask, peak_values, 0)
        peak = jnp.sum(peak_values, axis=0) / jnp.maximum(jnp.sum(peak_mask, axis=0), 1)
        
        return carry, (valley, peak)
    
    # Process all bands
    carry = (S, freq, octa, sr, n_bands, quantile)
    _, (valleys, peaks) = jax.lax.scan(
        process_band, carry, jnp.arange(n_bands + 1)
    )
    
    # Compute contrast
    if linear:
        contrast = peaks - valleys
    else:
        contrast = power_to_db(peaks) - power_to_db(valleys)
    
    return contrast


def melspectrogram(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[Union[str, float]] = "slaney",
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute a mel-scaled spectrogram.

    If a time-series input y is provided, its magnitude spectrogram S is
    first computed, and then mapped onto the mel scale by mel_f.dot(S**power).

    By default, power=2 operates on a power spectrum.

    Note:
        For JAX JIT compilation, all arguments except ``y`` and ``S`` should be marked as static:
        ``sr``, ``n_fft``, ``hop_length``, ``win_length``, ``window``, ``center``, ``pad_mode``,
        ``power``, ``n_mels``, ``fmin``, ``fmax``, ``htk``, ``norm``, ``dtype``

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude with shape ``(..., F, N)``
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        power: Exponent for the magnitude melspectrogram.
            e.g., 1 for energy, 2 for power, etc.
            If 0, return the STFT magnitude directly.
        n_mels: Number of mel bands to generate
        fmin: Lowest frequency (in Hz)
        fmax: Highest frequency (in Hz). If None, use fmax = sr / 2.0
        htk: Use HTK formula instead of Slaney
        norm: {None, "slaney", float > 0}
            If "slaney", divide the triangular mel weights by the width of the
            mel band (area normalization).
            If numeric, use norm as a mel exponent normalization.
            See librosax.filters.mel for details.
        dtype: Data type of the output array

    Returns:
        Mel spectrogram with shape ``(..., n_mels, N)``.

        - ``(T,)`` → ``(n_mels, N)``
        - ``(B, T)`` → ``(B, n_mels, N)``
    """
    if fmax is None:
        fmax = sr / 2

    # Compute the spectrogram magnitude
    if S is None:
        S, n_fft = _spectrogram(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

        # Apply power scaling if needed
        if power != 1.0:
            S = jnp.power(S, power).astype(dtype)
        else:
            S = S.astype(dtype)
    else:
        # When S is provided, it's already at the desired power scale
        # So just convert dtype
        S = S.astype(dtype)
        # We need to infer n_fft from the spectrogram shape
        n_fft = 2 * (S.shape[-2] - 1)

    # Build mel filter matrix (already returns JAX array)
    mel_basis = filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
        dtype=dtype,
    )

    # Apply mel filterbank
    # mel_basis shape: (n_mels, 1 + n_fft/2)
    # S shape: (..., 1 + n_fft/2, t)
    # Use einsum for flexible dimensions
    melspec = jnp.einsum("...ft,mf->...mt", S, mel_basis)

    return melspec


def mfcc(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: int = 0,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    melspectrogram_params: Optional[dict] = None,
) -> jnp.ndarray:
    """Compute Mel-frequency cepstral coefficients (MFCCs).

    MFCCs are computed from the log-power mel spectrogram.

    Note:
        For JAX JIT compilation, all arguments except ``y`` and ``S`` should be marked as static.
        This includes all the melspectrogram parameters and MFCC-specific parameters:
        ``sr``, ``n_mfcc``, ``dct_type``, ``norm``, ``lifter``, plus all other kwargs.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Audio sampling rate
        S: (optional) log-power mel spectrogram with shape ``(..., n_mels, N)``
        n_mfcc: Number of MFCCs to return (default: 20)
        dct_type: Discrete cosine transform (DCT) type (default: 2)
        norm: If "ortho", use orthonormal DCT basis. Default: "ortho"
        lifter: If lifter>0, apply liftering (cepstral filtering) to the MFCCs.
            If lifter=0, no liftering is applied.
        n_fft: FFT window size (used if y is provided)
        hop_length: Hop length for STFT (used if y is provided)
        win_length: Window length (used if y is provided)
        window: Window function (used if y is provided)
        center: If True, pad the signal (used if y is provided)
        pad_mode: Padding mode (used if y is provided)
        power: Exponent for the magnitude melspectrogram (used if y is provided)
        n_mels: Number of mel bands (used if y is provided)
        fmin: Lowest frequency in Hz (used if y is provided)
        fmax: Highest frequency in Hz (used if y is provided)
        htk: Use HTK formula for mel scale (used if y is provided)
        melspectrogram_params: Additional keyword arguments for melspectrogram
            (used if y is provided)

    Returns:
        MFCCs with shape ``(..., n_mfcc, N)``.

        - ``(T,)`` → ``(n_mfcc, N)``
        - ``(B, T)`` → ``(B, n_mfcc, N)``
    """
    if S is None:
        # Compute mel spectrogram if not provided
        mel_params = dict(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm="slaney",  # Default mel normalization for librosa
        )

        # Override with any custom parameters
        if melspectrogram_params is not None:
            mel_params.update(melspectrogram_params)

        S = melspectrogram(**mel_params)

        # Convert to log scale
        S = power_to_db(S)

    # Note: S has shape (..., n_mels, t)
    # Need to transpose for DCT which expects (..., t, n_mels)
    S_transposed = jnp.moveaxis(S, -2, -1)

    # Apply DCT using the shared _dct_flexible function
    mfccs = _dct_flexible(S_transposed, type=dct_type, norm=norm, axis=-1)

    # Keep only the first n_mfcc coefficients
    mfccs = mfccs[..., :n_mfcc]

    # Transpose back to (..., n_mfcc, t)
    mfccs = jnp.moveaxis(mfccs, -1, -2)

    # Apply liftering if requested
    if lifter > 0:
        # Create liftering coefficients
        # Shape: (n_mfcc,) which will broadcast correctly with (..., n_mfcc, t)
        LI = jnp.sin(jnp.pi * jnp.arange(1, 1 + n_mfcc, dtype=mfccs.dtype) / lifter)
        LI = 1 + (lifter / 2) * LI

        # Reshape for broadcasting: (n_mfcc,) -> (n_mfcc, 1) for shape (..., n_mfcc, t)
        # LI needs to align with the n_mfcc dimension
        shape = [1] * (mfccs.ndim - 2) + [n_mfcc, 1]
        LI = LI.reshape(shape)

        mfccs = mfccs * LI

    return mfccs


def hz_to_octs(frequencies: jnp.ndarray, *, tuning: float = 0.0, bins_per_octave: int = 12) -> jnp.ndarray:
    """Convert frequencies (Hz) to octave numbers.

    Args:
        frequencies: Array of frequencies in Hz
        tuning: Tuning deviation from A440 in fractional bins
        bins_per_octave: Number of bins per octave (default: 12)

    Returns:
        jnp.ndarray: Octave numbers (C1 = 0, C2 = 1, ..., A4 = 4.75)
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    # C1 is 4 octaves below A4
    octs = jnp.log2(frequencies / (A440 / 16))
    return octs


def chroma_filter(
    *,
    sr: float,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2,
    norm: Optional[float] = 2,
    base_c: bool = True,
    dtype: jnp.dtype = jnp.float64,
) -> jnp.ndarray:
    """Create a chroma filter bank.

    Creates a linear transformation matrix to project FFT bins onto chroma bins.

    Args:
        sr: Sampling rate
        n_fft: Number of FFT bins
        n_chroma: Number of chroma bins to produce (default: 12)
        tuning: Tuning deviation from A440 in fractional bins (default: 0.0)
        ctroct: Center of Gaussian weighting in octaves (default: 5.0)
        octwidth: Gaussian half-width for weighting. None for flat weighting (default: 2)
        norm: Normalization factor for filter weights. None for no normalization (default: 2)
        base_c: If True, start filter bank at C. If False, start at A (default: True)
        dtype: Data type for filter bank

    Returns:
        jnp.ndarray: Chroma filter bank [shape=(n_chroma, 1 + n_fft/2)]
    """
    wts = jnp.zeros((n_chroma, n_fft), dtype=dtype)

    # Get the FFT bins, not counting the DC component
    frequencies = jnp.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = jnp.concatenate([jnp.array([frqbins[0] - 1.5 * n_chroma]), frqbins])

    binwidthbins = jnp.concatenate([jnp.maximum(frqbins[1:] - frqbins[:-1], 1.0), jnp.array([1])])

    # Create the chroma matrix
    D = jnp.arange(0, n_chroma, dtype=dtype)[jnp.newaxis, :] - frqbins[:, jnp.newaxis]

    n_chroma2 = jnp.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # remainder are positive
    D = jnp.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = jnp.exp(-0.5 * (2 * D / binwidthbins[:, jnp.newaxis]) ** 2)

    # Transpose to match expected shape
    wts = wts.T

    # normalize each column
    wts = normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts = wts * jnp.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2))

    if base_c:
        wts = jnp.roll(wts, -3 * (n_chroma // 12), axis=0)

    # remove aliasing columns, only take up to n_fft/2 + 1
    return wts[:, : int(1 + n_fft / 2)]


def cqt_frequencies(
    *,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.70,
    tuning: float = 0.0,
) -> jnp.ndarray:
    """Compute the center frequencies of Constant-Q bins.

    Args:
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
        fmin: Minimum frequency (Hz)
        tuning: Tuning deviation from A440 in fractions of a bin

    Returns:
        jnp.ndarray: Center frequencies for each CQT bin
    """
    correction = 2.0 ** (tuning / bins_per_octave)

    # Generate geometric sequence of frequencies
    frequencies = fmin * correction * (2.0 ** (jnp.arange(n_bins) / bins_per_octave))

    return frequencies


def cqt_frequencies_np(
    *,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.70,
    tuning: float = 0.0,
) -> np.ndarray:
    """Compute the center frequencies of Constant-Q bins using numpy.
    
    This is used internally for filter bank creation to avoid JIT issues.
    """
    correction = 2.0 ** (tuning / bins_per_octave)
    frequencies = fmin * correction * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    return frequencies


def _create_cqt_kernels(
    Q: float,
    sr: float,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    norm: Optional[float],
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True,
    gamma: float = 0.0,
    n_fft_fixed: Optional[int] = None,
) -> tuple[jnp.ndarray, int, jnp.ndarray, jnp.ndarray]:
    """Create CQT kernels following nnAudio's implementation.
    
    This function creates the CQT kernels in time domain, similar to nnAudio's
    create_cqt_kernels function. Uses numpy with float64 for better precision.
    
    Returns:
        kernels: Complex CQT kernels in time domain
        fftLen: FFT length used
        lengths: Length of each filter
        freqs: Center frequencies for each bin
    """
    # Use numpy for better precision
    # Calculate frequencies for each bin
    if (fmax is not None) and (n_bins is None):
        n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
        freqs_np = fmin * 2.0 ** (np.arange(n_bins, dtype=np.float64) / float(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs_np = fmin * 2.0 ** (np.arange(n_bins, dtype=np.float64) / float(bins_per_octave))
    else:
        # If both are given, use fmax to calculate n_bins
        if fmax is not None:
            n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
        freqs_np = fmin * 2.0 ** (np.arange(n_bins, dtype=np.float64) / float(bins_per_octave))
    
    # Calculate filter lengths using nnAudio's formula
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths_np = np.ceil(Q * sr / (freqs_np + gamma / alpha)).astype(np.float64)
    
    # Calculate FFT length
    max_len = int(np.max(lengths_np))
    if n_fft_fixed is None:
        # Calculate based on max length, but use a power of 2
        fftLen = int(2 ** (np.ceil(np.log2(max_len))))
    else:
        # If fixed FFT length is provided, ensure it's at least as large as max kernel
        fftLen = max(n_fft_fixed, int(2 ** (np.ceil(np.log2(max_len)))))
    
    # Create kernels in numpy with float64
    kernels_np = np.zeros((n_bins, fftLen), dtype=np.complex128)
    
    # Create each kernel using numpy
    for k in range(n_bins):
        freq = freqs_np[k]
        l = int(lengths_np[k])
        
        # Center the kernel following nnAudio's logic
        if l % 2 == 1:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        
        # Apply window using scipy to match nnAudio exactly
        if window == "hann":
            from scipy.signal import get_window
            window_dispatch = get_window(window, int(l), fftbins=True).astype(np.float64)
        else:
            window_dispatch = np.ones(int(l), dtype=np.float64)
        
        # Create complex sinusoid following nnAudio exactly
        sig = window_dispatch * np.exp(np.r_[-l // 2 : l // 2] * 1j * 2 * np.pi * freq / sr) / l
        
        # Normalize if requested - match nnAudio's use of np.linalg.norm
        if norm:
            sig = sig / np.linalg.norm(sig, norm)
        # Note: if norm is None or 0, no normalization is applied
        
        # Place kernel in the array
        kernels_np[k, start:start + l] = sig
    
    # Convert to JAX arrays at the end
    kernels = jnp.array(kernels_np, dtype=jnp.complex64)
    lengths = jnp.array(lengths_np, dtype=jnp.float32)
    freqs = jnp.array(freqs_np, dtype=jnp.float32)
    
    return kernels, fftLen, lengths, freqs


def cqt(
    y: jnp.ndarray,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: Optional[float] = 0.0,
    filter_scale: float = 1.0,
    norm: Optional[float] = 1.0,
    sparsity: float = 0.01,
    window: str = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    res_type: Optional[str] = None,
    dtype: jnp.dtype = jnp.complex64,
    n_fft: Optional[int] = None,
    use_1992_version: bool = True,
    output_format: str = "complex",
    normalization_type: str = "librosa",
) -> jnp.ndarray:
    """Compute the constant-Q transform following nnAudio's CQT1992v2 implementation.

    This implementation follows nnAudio's CQT1992v2 algorithm which computes the CQT
    efficiently by convolving the time-domain signal with CQT kernels.

    Note:
        For JAX JIT compilation, all arguments except ``y`` should be marked as static:
        ``sr``, ``hop_length``, ``fmin``, ``n_bins``, ``bins_per_octave``, ``tuning``,
        ``filter_scale``, ``norm``, ``sparsity``, ``window``, ``scale``, ``pad_mode``,
        ``res_type``, ``dtype``, ``n_fft``, ``use_1992_version``, ``output_format``,
        ``normalization_type``

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Sampling rate
        hop_length: Number of samples between successive CQT columns
        fmin: Minimum frequency (default: C1 = 32.70 Hz)
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
        tuning: Tuning offset in fractions of a bin
        filter_scale: Filter scale factor (Q = filter_scale / (2^(1/bins_per_octave) - 1))
        norm: Normalization type for basis functions (1, 2, or None)
        sparsity: Sparsification factor (not implemented)
        window: Window function
        scale: If True, scale by sqrt(filter_lengths) following librosa normalization
        pad_mode: Padding mode
        res_type: Resampling type (not used in 1992 version)
        dtype: Complex data type
        n_fft: FFT size (if None, calculated automatically)
        use_1992_version: If True, use CQT1992v2 algorithm (recommended)
        output_format: Output format ('complex', 'magnitude', 'phase')
        normalization_type: Normalization type ('librosa', 'convolutional', 'wrap')

    Returns:
        CQT matrix with shape ``(..., n_bins, N)``. Format depends on output_format.

        - ``(T,)`` → ``(n_bins, N)``
        - ``(B, T)`` → ``(B, n_bins, N)``
    """
    if y is None:
        raise ParameterError("y cannot be None - audio input is required for CQT")

    if fmin is None:
        fmin = note_to_hz("C1")

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Calculate Q factor
    Q = float(filter_scale) / (2.0 ** (1.0 / bins_per_octave) - 1.0)

    # Create CQT kernels
    cqt_kernels, kernel_width, lengths, freqs = _create_cqt_kernels(
        Q=Q,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        norm=norm,
        window=window,
        n_fft_fixed=n_fft,
    )

    # Track if we need to squeeze batch dimension later
    squeeze_batch = False
    if y.ndim == 1:
        y = y[jnp.newaxis, :]
        squeeze_batch = True
    # y is now 2D (batch, time)

    # Pad the signal if center is True
    if pad_mode == "constant":
        y = jnp.pad(y, ((0, 0), (kernel_width // 2, kernel_width // 2)), mode="constant")
    elif pad_mode == "reflect":
        y = jnp.pad(y, ((0, 0), (kernel_width // 2, kernel_width // 2)), mode="reflect")

    # Take FFT of kernels once
    cqt_kernels_fft = jnp.fft.fft(cqt_kernels, axis=1)
    # Extract only the positive frequencies
    cqt_kernels_fft = cqt_kernels_fft[:, :kernel_width // 2 + 1]
    
    # Compute STFT for all batch elements at once
    _, _, D = jssignal.stft(
        y,
        window="boxcar",  # rectangular window since CQT kernels already have windows
        nperseg=kernel_width,
        noverlap=kernel_width - hop_length,
        nfft=kernel_width,
        boundary=None,  # No padding, we already padded
        padded=False,
        axis=-1,
    )
    
    # Multiply in frequency domain
    # D shape: (batch, freq_bins, time_frames) for batched input
    # cqt_kernels_fft shape: (n_bins, freq_bins)
    # Result shape: (batch, n_bins, time_frames)
    C = jnp.einsum('bf,...ft->...bt', cqt_kernels_fft.conj(), D)

    # Apply normalization based on type
    if normalization_type == "librosa":
        if scale:
            # Apply sqrt(lengths) normalization
            # When using FFT-based convolution vs nnAudio's direct convolution,
            # we need an additional scaling factor to match the output magnitude
            # This is empirically determined to match nnAudio's output
            # The factor depends on the kernel width and hop length
            fft_correction = 1.0 / jnp.sqrt(kernel_width) * 32.0
            scale_factors = jnp.sqrt(lengths)[:, jnp.newaxis] * fft_correction
            C = C * scale_factors
    elif normalization_type == "convolutional":
        # No additional normalization
        pass
    elif normalization_type == "wrap":
        # Apply wrap normalization
        C = C * 2.0
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    # Format output based on output_format
    if output_format.lower() == "magnitude":
        result = jnp.abs(C)
    elif output_format.lower() == "complex":
        result = C
    elif output_format.lower() == "phase":
        phase = jnp.angle(C)
        result = jnp.stack([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    # Remove batch dimension if we added it
    if squeeze_batch:
        result = result[0]
        
    return result


def _create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = 256,
    transition_bandwidth: float = 0.03,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Create a lowpass filter for downsampling.
    
    Args:
        band_center: Center frequency (normalized to Nyquist)
        kernel_length: Length of the filter kernel
        transition_bandwidth: Width of the transition band
        dtype: Data type for the filter
        
    Returns:
        Filter kernel coefficients
    """
    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)
    
    key_frequencies = [0.0, passband_max, stopband_min, 1.0]
    gain_at_key_frequencies = [1.0, 1.0, 0.0, 0.0]
    
    # Use scipy for filter design
    import scipy.signal
    filter_kernel = scipy.signal.firwin2(kernel_length, key_frequencies, gain_at_key_frequencies)
    
    return jnp.array(filter_kernel, dtype=dtype)


def _next_power_of_2(n: int) -> int:
    """Calculate the next power of 2."""
    return int(2 ** np.ceil(np.log2(n)))


def _early_downsample_count(
    nyquist_hz: float,
    filter_cutoff_hz: float, 
    hop_length: int,
    n_octaves: int
) -> int:
    """Compute the number of early downsampling operations."""
    downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist_hz / filter_cutoff_hz)) - 1) - 1)
    num_twos = int(np.ceil(np.log2(hop_length)))
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    
    return min(downsample_count1, downsample_count2)


def _get_early_downsample_params(
    sr: float,
    hop_length: int,
    fmax_t: float,
    Q: float,
    n_octaves: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple:
    """Compute downsampling parameters for early downsampling."""
    window_bandwidth = 1.5  # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    
    downsample_count = _early_downsample_count(sr / 2, filter_cutoff, hop_length, n_octaves)
    downsample_factor = 2 ** downsample_count
    
    hop_length //= downsample_factor
    new_sr = sr / float(downsample_factor)
    
    if downsample_factor != 1:
        early_downsample_filter = _create_lowpass_filter(
            band_center=1 / downsample_factor,
            kernel_length=256,
            transition_bandwidth=0.03,
            dtype=dtype,
        )
    else:
        early_downsample_filter = None
        
    return new_sr, hop_length, downsample_factor, early_downsample_filter


@partial(jax.jit, static_argnames=('n', 'axis'))
def _downsample_by_n(x: jnp.ndarray, filter_kernel: jnp.ndarray, n: int, axis: int = -1) -> jnp.ndarray:
    """Downsample signal by factor n using the given filter.
    
    This matches nnAudio's approach using strided convolution with padding.
    """
    # Calculate padding to match nnAudio
    padding = (filter_kernel.shape[0] - 1) // 2
    
    # Apply padding
    if axis == -1 or axis == len(x.shape) - 1:
        x_padded = jnp.pad(x, padding, mode='constant')
    else:
        # Handle other axes if needed
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (padding, padding)
        x_padded = jnp.pad(x, pad_width, mode='constant')
    
    # Apply filter and downsample in one step using strided convolution
    # This ensures consistent output sizes
    filtered = jnp.convolve(x_padded, filter_kernel, mode='valid')
    
    # Downsample
    return filtered[::n]


def _get_cqt_complex(
    x: jnp.ndarray,
    cqt_kernels_real: jnp.ndarray,
    cqt_kernels_imag: jnp.ndarray, 
    hop_length: int,
    pad_length: int,
    pad_mode: str = "constant",
) -> jnp.ndarray:
    """Compute CQT using time-domain convolution.
    
    This implementation matches nnAudio's approach using strided convolution.
    
    Args:
        x: Input signal [shape=(batch, time)]
        cqt_kernels_real: Real part of CQT kernels [shape=(n_bins, 1, kernel_length)]
        cqt_kernels_imag: Imaginary part of CQT kernels [shape=(n_bins, 1, kernel_length)]
        hop_length: Hop size (stride)
        pad_length: Padding length
        pad_mode: Padding mode
        
    Returns:
        Complex CQT [shape=(batch, n_bins, time, 2)]
    """
    # Pad the signal - matches nnAudio's center padding
    if pad_mode == "constant":
        x_padded = jnp.pad(x, ((0, 0), (pad_length, pad_length)), mode="constant")
    elif pad_mode == "reflect":
        x_padded = jnp.pad(x, ((0, 0), (pad_length, pad_length)), mode="reflect")
    else:
        x_padded = x
    
    # Use JAX's conv_general_dilated_local for 1D strided convolution
    # This ensures consistent output sizes like PyTorch's conv1d
    from jax import lax
    
    # Reshape inputs for conv_general_dilated
    # x needs shape: (batch, in_channels=1, time)
    # kernels need shape: (out_channels, in_channels=1, kernel_length)
    x_reshaped = x_padded[:, jnp.newaxis, :]  # Add channel dimension
    
    # Ensure kernels have the right shape
    if cqt_kernels_real.ndim == 2:
        # Shape is (n_bins, kernel_length), need to add channel dimension
        kernels_real = cqt_kernels_real[:, jnp.newaxis, :]
        kernels_imag = cqt_kernels_imag[:, jnp.newaxis, :]
    else:
        # Already has channel dimension
        kernels_real = cqt_kernels_real
        kernels_imag = cqt_kernels_imag
    
    # Perform strided convolution
    # dimension_numbers: (batch, channel, spatial)
    cqt_real = lax.conv_general_dilated(
        x_reshaped,
        kernels_real,
        window_strides=(hop_length,),
        padding='VALID',
        dimension_numbers=('NCH', 'OIH', 'NCH')
    )
    
    cqt_imag = -lax.conv_general_dilated(
        x_reshaped,
        kernels_imag,
        window_strides=(hop_length,),
        padding='VALID',
        dimension_numbers=('NCH', 'OIH', 'NCH')
    )
    
    # The output shape from conv is (batch, n_bins, time)
    # No need to squeeze - the conv already gives us the right shape
    
    # Stack real and imaginary parts
    return jnp.stack([cqt_real, cqt_imag], axis=-1)


def cqt2010(
    y: jnp.ndarray,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    filter_scale: float = 1.0,
    norm: Optional[float] = 1.0,
    sparsity: float = 0.01,  # todo: not used
    window: str = "hann",
    scale: bool = True,
    pad_mode: str = "reflect",
    res_type: Optional[str] = None,  # todo: not used
    dtype: jnp.dtype = jnp.complex64,  # todo: not used
    output_format: str = "magnitude",
    earlydownsample: bool = True,
) -> jnp.ndarray:
    """Compute constant-Q transform using the 2010 algorithm with multi-resolution.
    
    This implementation follows nnAudio's CQT2010v2 algorithm which is more
    memory-efficient than CQT1992. It creates a small CQT kernel for the top
    octave and uses downsampling to compute lower octaves.
    
    Args:
        y: Audio time series
        sr: Sampling rate  
        hop_length: Number of samples between successive CQT columns
        fmin: Minimum frequency (default: C1 = 32.70 Hz)
        fmax: Maximum frequency (default: inferred from n_bins)
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
        tuning: Tuning offset in fractions of a bin
        filter_scale: Filter scale factor
        norm: Normalization type for basis functions
        sparsity: Sparsification factor (not implemented)
        window: Window function
        scale: If True, scale the output
        pad_mode: Padding mode
        res_type: Resampling type (not used)
        dtype: Complex data type
        output_format: Output format ('magnitude', 'complex', 'phase')
        earlydownsample: If True, use early downsampling optimization
        
    Returns:
        CQT matrix
    """
    if fmin is None:
        fmin = note_to_hz("C1")
        
    # Apply tuning
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    
    # Calculate Q factor
    Q = float(filter_scale) / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    
    # Calculate number of octaves
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)
    
    # Get the top octave parameters
    fmin_t = fmin * 2 ** (n_octaves - 1)
    
    # Calculate kernel parameters
    if fmax is not None:
        # If fmax is specified, adjust n_bins
        n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
        n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        fmin_t = fmin * 2 ** (n_octaves - 1)
    
    # Check remainder to calculate top bin frequency
    remainder = n_bins % bins_per_octave
    if remainder == 0:
        fmax_t = fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
    else:
        fmax_t = fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
        
    # Adjust fmin_t
    fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
    
    # Check Nyquist
    if fmax_t > sr / 2:
        raise ValueError(f"The top bin {fmax_t}Hz exceeds Nyquist frequency")
        
    # Get early downsample parameters if enabled
    sample_rate = sr
    hop = hop_length
    downsample_factor = 1.0
    early_downsample_filter = None
    
    if earlydownsample:
        sample_rate, hop, downsample_factor, early_downsample_filter = _get_early_downsample_params(
            sr, hop_length, fmax_t, Q, n_octaves, dtype=jnp.float32
        )
        
    # Create CQT kernels for top octave only
    cqt_kernels, n_fft, lengths, freqs = _create_cqt_kernels(
        Q=Q,
        sr=sample_rate,
        fmin=fmin_t,
        n_bins=n_filters,
        bins_per_octave=bins_per_octave,
        norm=norm,
        window=window,
    )
    
    # Create lowpass filter for octave downsampling
    lowpass_filter = _create_lowpass_filter(
        band_center=0.5,
        kernel_length=256, 
        transition_bandwidth=0.001
    )
    
    # Track if we need to squeeze the batch dimension later
    squeeze_batch = False
    if y.ndim == 1:
        y = y[jnp.newaxis, :]
        squeeze_batch = True
    
    # Convert to float32 if needed to match kernel dtype
    if y.dtype != jnp.float32:
        y = y.astype(jnp.float32)
        
    # Apply early downsampling if enabled
    if earlydownsample and early_downsample_filter is not None:
        # Process each batch element
        if y.shape[0] == 1:
            y = _downsample_by_n(y[0], early_downsample_filter, int(downsample_factor))[jnp.newaxis, :]
        else:
            # Use vmap for batch processing
            downsample_fn = lambda y_single: _downsample_by_n(y_single, early_downsample_filter, int(downsample_factor))
            y = jax.vmap(downsample_fn)(y)
        
    # Split kernels
    cqt_kernels_real = jnp.real(cqt_kernels)[:, jnp.newaxis, :]
    cqt_kernels_imag = jnp.imag(cqt_kernels)[:, jnp.newaxis, :]
    
    # Get CQT for top octave
    CQT = _get_cqt_complex(y, cqt_kernels_real, cqt_kernels_imag, hop, n_fft // 2, pad_mode)
    
    # Process remaining octaves
    y_down = y
    for i in range(n_octaves - 1):
        hop = hop // 2
        # Downsample by 2
        if y_down.shape[0] == 1:
            y_down = _downsample_by_n(y_down[0], lowpass_filter, 2)[jnp.newaxis, :]
        else:
            # Use vmap for batch processing
            downsample_fn = lambda y_single: _downsample_by_n(y_single, lowpass_filter, 2)
            y_down = jax.vmap(downsample_fn)(y_down)
        
        # Get CQT for this octave
        CQT1 = _get_cqt_complex(y_down, cqt_kernels_real, cqt_kernels_imag, hop, n_fft // 2, pad_mode)
        
        # Concatenate (lower frequencies first)
        CQT = jnp.concatenate([CQT1, CQT], axis=1)
        
    # Remove unwanted bins
    CQT = CQT[:, -n_bins:, :]
    
    # Apply scaling
    CQT = CQT * downsample_factor
    
    # Get all frequency lengths
    all_freqs = fmin * 2.0 ** (jnp.arange(n_bins) / float(bins_per_octave))
    all_lengths = jnp.ceil(Q * sr / all_freqs)
    
    # Normalize
    if scale:
        CQT = CQT * jnp.sqrt(all_lengths).reshape((-1, 1, 1))
        
    # Format output based on requested format
    if output_format.lower() == "magnitude":
        result = jnp.sqrt(jnp.sum(CQT ** 2, axis=-1))
    elif output_format.lower() == "complex":
        result = CQT[:, :, :, 0] + 1j * CQT[:, :, :, 1]
    elif output_format.lower() == "phase":
        phase = jnp.arctan2(CQT[:, :, :, 1], CQT[:, :, :, 0])
        result = jnp.stack([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    # Remove batch dimension only if we added it
    if squeeze_batch:
        result = result[0]
        
    return result


def chroma_cqt(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    C: Optional[jnp.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    norm: Optional[Union[float, str]] = jnp.inf,
    threshold: float = 0.0,
    tuning: Optional[float] = 0.0,
    n_chroma: int = 12,
    n_octaves: int = 7,
    window: Optional[jnp.ndarray] = None,  # todo: not used yet
    bins_per_octave: int = 36,
    cqt_mode: str = "full",  # todo: not used yet
    **kwargs,
) -> jnp.ndarray:
    """Chromagram from a constant-Q transform.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Sampling rate
        C: Pre-computed CQT spectrogram with shape ``(..., n_cqt_bins, N)``
        hop_length: Number of samples between successive CQT columns
        fmin: Minimum frequency. Default: C1 ~= 32.70 Hz
        norm: Normalization mode for chroma
        threshold: Pre-normalization energy threshold
        tuning: Tuning deviation from A440 in fractional bins
        n_chroma: Number of chroma bins to produce
        n_octaves: Number of octaves to analyze above fmin
        window: Optional weighting window
        bins_per_octave: Number of bins per octave in the CQT
        cqt_mode: CQT mode ('full' or 'hybrid')
        **kwargs: Additional parameters for cqt

    Returns:
        Chromagram with shape ``(..., n_chroma, N)``.

        - ``(T,)`` → ``(n_chroma, N)``
        - ``(B, T)`` → ``(B, n_chroma, N)``
    """
    if y is None and C is None:
        raise ParameterError("Either y or C must be provided")

    if fmin is None:
        fmin = note_to_hz("C1")

    if C is None:
        C = cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_octaves * bins_per_octave,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            output_format="magnitude",
            **kwargs,
        )

    # Map CQT bins to chroma bins
    # This is a simplified version - proper implementation would use
    # cq_to_chroma matrix from librosa
    n_cqt_bins = C.shape[-2]

    # Create a proper mapping matrix from CQT bins to chroma bins
    # The key insight is that if bins_per_octave is a multiple of n_chroma,
    # then we can group consecutive CQT bins into chroma bins
    bins_per_chroma = bins_per_octave // n_chroma

    # Create the mapping matrix
    # Vectorized version for JAX compatibility
    bin_indices = jnp.arange(n_cqt_bins)
    chroma_indices = (bin_indices // bins_per_chroma) % n_chroma

    cq_to_chr = jnp.zeros((n_chroma, n_cqt_bins))
    cq_to_chr = cq_to_chr.at[chroma_indices, bin_indices].set(1.0)

    # Apply the mapping
    # C shape: (..., n_cqt_bins, t)
    # cq_to_chr shape: (n_chroma, n_cqt_bins)
    # Result shape: (..., n_chroma, t)
    chroma = jnp.einsum("...ct,bc->...bt", C, cq_to_chr)

    # Apply threshold
    if threshold > 0:
        chroma = jnp.where(chroma < threshold, 0, chroma)

    # Normalize
    if norm is not None:
        chroma = normalize(chroma, norm=norm, axis=-2)

    return chroma


def tonnetz(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    chroma: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Compute the tonal centroid features (tonnetz).

    This representation projects chroma features onto a 6-dimensional basis
    representing the perfect fifth, minor third, and major third each as
    two-dimensional coordinates.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Sampling rate of y
        chroma: Normalized energy for each chroma bin at each frame
            with shape ``(..., n_chroma, N)``. If None, a chroma_stft is computed.
        **kwargs: Additional keyword arguments to chroma_stft,
            if chroma is not pre-computed.

    Returns:
        Tonnetz features with shape ``(..., 6, N)``.

        - ``(T,)`` → ``(6, N)``
        - ``(B, T)`` → ``(B, 6, N)``

        Tonnetz dimensions:

        - 0: Fifth x-axis
        - 1: Fifth y-axis
        - 2: Minor x-axis
        - 3: Minor y-axis
        - 4: Major x-axis
        - 5: Major y-axis
    """
    if y is None and chroma is None:
        raise ParameterError(
            "Either the audio samples or the chromagram must be "
            "passed as an argument."
        )

    if chroma is None:
        # Use chroma_stft instead of chroma_cqt for now
        chroma = chroma_stft(y=y, sr=sr, **kwargs)

    # Generate Transformation matrix
    n_chroma = chroma.shape[-2]
    dim_map = jnp.linspace(0, 12, num=n_chroma, endpoint=False)

    # Interval scaling factors
    scale = jnp.array([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])

    # Create the transformation matrix
    V = scale[:, jnp.newaxis] * dim_map[jnp.newaxis, :]

    # Even rows compute sin() offset
    V = V.at[::2].add(-0.5)

    # Radii for each dimension
    R = jnp.array([1, 1, 1, 1, 0.5, 0.5])  # Fifths, Minor, Major

    # Compute the projection matrix
    phi = R[:, jnp.newaxis] * jnp.cos(jnp.pi * V)

    # Normalize chroma features
    chroma_norm = normalize(chroma, norm=1, axis=-2)

    # Do the transform to tonnetz
    # phi shape: (6, n_chroma)
    # chroma_norm shape: (..., n_chroma, t)
    # tonnetz shape: (..., 6, t)
    tonnetz = jnp.einsum("pc,...ct->...pt", phi, chroma_norm)

    return tonnetz


def chroma_stft(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    norm: Optional[Union[float, str]] = jnp.inf,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    tuning: Optional[float] = None,
    n_chroma: int = 12,
    **kwargs,
) -> jnp.ndarray:
    """Compute a chromagram from a power spectrogram or waveform.

    Args:
        y: Audio time series. The last axis must be time.

            - ``(T,)`` - single waveform
            - ``(B, T)`` - batch of waveforms

        sr: Sampling rate
        S: Power spectrogram with shape ``(..., F, N)`` (optional if y is provided)
        norm: Column-wise normalization. See `normalize` for details.
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
        window: Window specification
        center: Center the frames
        pad_mode: Padding mode
        tuning: Tuning deviation from A440 in fractional bins.
            If None, tuning will be automatically estimated (not implemented yet).
        n_chroma: Number of chroma bins to produce
        **kwargs: Additional arguments to chroma_filter (ctroct, octwidth, norm, base_c)

    Returns:
        Chromagram with shape ``(..., n_chroma, N)``.

        - ``(T,)`` → ``(n_chroma, N)``
        - ``(B, T)`` → ``(B, n_chroma, N)``
    """
    # Get power spectrogram
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # _spectrogram returns magnitude, but chroma needs power
    if y is not None:
        S = S ** 2

    # Estimate tuning if not provided
    if tuning is None:
        def _estimate_tuning(S_arr):
            return np.array([librosa.estimate_tuning(S=S_arr, sr=sr, bins_per_octave=n_chroma)])
        tuning = jax.pure_callback(
            _estimate_tuning,
            jax.ShapeDtypeStruct((1,), jnp.float64),
            S
        )[0]

    # Get the filter bank
    chromafb = chroma_filter(
        sr=sr,
        n_fft=n_fft,
        tuning=tuning,
        n_chroma=n_chroma,
        **kwargs
    )

    # Apply the filter bank
    # chromafb shape: (n_chroma, 1 + n_fft/2)
    # S shape: (..., 1 + n_fft/2, t)
    # Use einsum for flexible dimensions
    raw_chroma = jnp.einsum("...ft,cf->...ct", S, chromafb)

    # Normalize
    if norm is not None:
        chroma = normalize(raw_chroma, norm=norm, axis=-2)
    else:
        chroma = raw_chroma

    return chroma


def poly_features(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    order: int = 1,
    freq: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.

    This function wraps librosa.feature.poly_features for compatibility.

    Args:
        y: Audio time series.
        sr: Audio sampling rate of y.
        S: Spectrogram magnitude. If None, compute from y.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length. If None, defaults to n_fft.
        window: Window function.
        center: If True, center frames.
        pad_mode: Padding mode.
        order: Order of the polynomial (1=linear, 2=quadratic, etc).
        freq: Optional center frequencies for each row of S. If None, use FFT bin frequencies.

    Returns:
        Polynomial coefficients with shape (..., order+1, t).

    Examples:
        Compute first-order (linear) coefficients

        >>> y, sr = librosax.load('audio.wav')
        >>> coeffs = librosax.feature.poly_features(y=y, sr=sr, order=1)
        >>> coeffs.shape
        (2, t)  # intercept and slope for each frame
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if S is not None:
        S = np.asarray(S)
    if freq is not None:
        freq = np.asarray(freq)

    result = librosa.feature.poly_features(
        y=y,
        sr=sr,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        order=order,
        freq=freq,
    )

    return jnp.asarray(result)


def chroma_vqt(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    V: Optional[jnp.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    intervals: str = "ji5",
    norm: Optional[float] = jnp.inf,
    threshold: float = 0.0,
    n_octaves: int = 7,
    bins_per_octave: int = 12,
    gamma: float = 0,
) -> jnp.ndarray:
    """Chromagram using Variable-Q Transform (VQT).

    This is a wrapper around librosa.feature.chroma_vqt for compatibility.

    Args:
        y: Audio time series.
        sr: Sampling rate.
        V: Pre-computed VQT spectrogram (optional).
        hop_length: Hop length for VQT.
        fmin: Minimum frequency. If None, defaults to C1.
        intervals: Interval specification ('ji3', 'ji5', 'equal', etc).
        norm: Normalization type.
        threshold: Pre-normalization energy threshold.
        n_octaves: Number of octaves.
        bins_per_octave: Number of bins per octave in the VQT.
        gamma: Bandwidth offset for variable-Q filterbank.

    Returns:
        Chromagram with shape (..., n_chroma, t).

    Examples:
        >>> y, sr = librosax.load('audio.wav')
        >>> chroma = librosax.feature.chroma_vqt(y=y, sr=sr)
        >>> chroma.shape
        (12, t)
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if V is not None:
        V = np.asarray(V)

    result = librosa.feature.chroma_vqt(
        y=y,
        sr=sr,
        V=V,
        hop_length=hop_length,
        fmin=fmin,
        intervals=intervals,
        norm=norm,
        threshold=threshold,
        n_octaves=n_octaves,
        bins_per_octave=bins_per_octave,
        gamma=gamma,
    )

    return jnp.asarray(result)


def chroma_cens(
    *,
    y: Optional[jnp.ndarray] = None,
    sr: float = 22050,
    C: Optional[jnp.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    tuning: Optional[float] = None,
    n_chroma: int = 12,
    n_octaves: int = 7,
    bins_per_octave: int = 36,
    cqt_mode: str = "full",
    window: Optional[jnp.ndarray] = None,
    norm: Optional[float] = 2,
    win_len_smooth: Optional[int] = 41,
    smoothing_window: str = "hann",
) -> jnp.ndarray:
    """Compute Chroma Energy Normalized Statistics (CENS).

    This is a wrapper around librosa.feature.chroma_cens.

    CENS is a variant of chroma features with additional temporal smoothing
    and energy normalization.

    Args:
        y: Audio time series.
        sr: Sampling rate.
        C: Pre-computed constant-Q chromagram.
        hop_length: Hop length.
        fmin: Minimum frequency. If None, defaults to C1.
        tuning: Tuning deviation from A440 in fractional bins.
        n_chroma: Number of chroma bins.
        n_octaves: Number of octaves.
        bins_per_octave: Number of bins per octave in the CQT.
        cqt_mode: CQT mode.
        window: Optional weighting window.
        norm: Normalization type.
        win_len_smooth: Length of temporal smoothing window.
        smoothing_window: Type of smoothing window.

    Returns:
        CENS chromagram with shape (..., n_chroma, t).

    Examples:
        >>> y, sr = librosax.load('audio.wav')
        >>> cens = librosax.feature.chroma_cens(y=y, sr=sr)
    """
    # Convert JAX arrays to numpy if needed
    if y is not None:
        y = np.asarray(y)
    if C is not None:
        C = np.asarray(C)
    if window is not None:
        window = np.asarray(window)

    result = librosa.feature.chroma_cens(
        y=y,
        sr=sr,
        C=C,
        hop_length=hop_length,
        fmin=fmin,
        tuning=tuning,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
        bins_per_octave=bins_per_octave,
        cqt_mode=cqt_mode,
        window=window,
        norm=norm,
        win_len_smooth=win_len_smooth,
        smoothing_window=smoothing_window,
    )

    return jnp.asarray(result)
