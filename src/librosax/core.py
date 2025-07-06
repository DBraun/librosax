from typing import Any, Callable, Optional, Union

import jax
from jax import numpy as jnp
from jax.scipy import signal
import librosa
import numpy as np
from scipy.signal import get_window


def stft(
    waveform: jnp.ndarray,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
):
    """Compute the Short-Time Fourier Transform (STFT) of a waveform.

    This function computes the STFT of the given waveform using JAX's ``scipy.signal.stft`` implementation.

    Note:
        For JAX JIT compilation, the following arguments should be marked as static:
        ``n_fft``, ``hop_length``, ``win_length``, ``window``, ``center``, ``pad_mode``

    Args:
        waveform: Input signal waveform.
        n_fft: FFT size.
        hop_length: Number of samples between successive frames. Default is ``win_length // 4``.
        win_length: Window size. Default is ``n_fft``.
        window: Window function type. Default is ``"hann"``.
        center: If ``True``, the waveform is padded so that frames are centered. Default is ``True``.
        pad_mode: Padding mode for the waveform. Must be one of ``["constant", "reflect"]``. Default is ``"constant"``.

    Returns:
        jnp.ndarray: Complex STFT matrix.

    Raises:
        AssertionError: If pad_mode is not one of ``["constant", "reflect"]``.
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    assert pad_mode in [
        "constant",
        "reflect",
    ], f"Pad mode '{pad_mode}' has not been tested with librosax."

    boundary = {
        "constant": "zeros",
        "reflect": "even",
    }[pad_mode]

    # Pad the window to n_fft size
    if window == "sqrt_hann":
        win = np.sqrt(get_window("hann", win_length))
    else:
        win = get_window(window, win_length)

    padded_win = np.zeros(n_fft)
    start = (n_fft - win_length) // 2
    padded_win[start : start + win_length] = win
    padded_win = jnp.array(padded_win)

    _, _, Zxx = signal.stft(
        waveform,
        window=padded_win,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=boundary if center else None,
        padded=False,
        axis=-1,
    )
    Zxx = Zxx * win_length / 2.0
    return Zxx


def istft(
    stft_matrix: jnp.ndarray,
    hop_length: int = None,
    win_length: int = None,
    n_fft: int = None,
    window: str = "hann",
    center: bool = True,
    length: int = None,
):
    """Compute the Inverse Short-Time Fourier Transform (ISTFT).

    This function reconstructs a waveform from an STFT matrix using JAX's ``scipy.signal.istft`` implementation.

    Args:
        stft_matrix: The STFT matrix from which to compute the inverse.
        hop_length: Number of samples between successive frames. Default is ``win_length // 4``.
        win_length: Window size. Default is ``n_fft``.
        n_fft: FFT size. Default is ``(stft_matrix.shape[-2] - 1) * 2``.
        window: Window function type. Default is ``"hann"``.
        center: If ``True``, assumes the waveform was padded so that frames were centered. Default is ``True``.
        length: Target length for the reconstructed signal. If None, the entire signal is returned.

    Returns:
        jnp.ndarray: Reconstructed time-domain signal.

    Raises:
        AssertionError: If center is ``False`` because the function is only tested for ``center=True``.
    """
    assert center, "Only tested for `center==True`"

    if n_fft is None:
        n_fft = (stft_matrix.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    # Pad the window to n_fft size
    if window == "sqrt_hann":
        win = np.sqrt(get_window("hann", win_length))
    else:
        win = get_window(window, win_length)

    padded_win = np.zeros(n_fft)
    start = (n_fft - win_length) // 2
    padded_win[start : start + win_length] = win
    padded_win = jnp.array(padded_win)

    _, reconstructed_signal = signal.istft(
        stft_matrix,
        window=padded_win,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=center,
    )

    reconstructed_signal = reconstructed_signal * 2.0 / win_length

    # Trim or pad the output signal to the desired length
    if length is not None:
        if length > reconstructed_signal.shape[-1]:
            # Pad the signal if it is shorter than the desired length
            pad_width = length - reconstructed_signal.shape[-1]
            reconstructed_signal = jnp.pad(
                reconstructed_signal,
                ((0, 0) * (reconstructed_signal.ndim - 1), (0, pad_width)),
                mode="constant",
            )
        else:
            # Trim the signal if it is longer than the desired length
            reconstructed_signal = reconstructed_signal[..., :length]

    return reconstructed_signal


def power_to_db(
    x: jnp.ndarray,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
    ref: float = 1.0,
) -> jnp.ndarray:
    """Convert a power spectrogram to decibel (dB) units.

    This function is a JAX implementation of ``librosa.power_to_db``.

    Args:
        x: Input power spectrogram.
        amin: Minimum threshold for input values. Default is 1e-10.
        top_db: Threshold the output at top_db below the peak. Default is 80.0.
        ref: Reference value for scaling. Default is 1.0.

    Returns:
        jnp.ndarray: dB-scaled spectrogram.

    Raises:
        librosa.util.exceptions.ParameterError: If ``top_db`` is negative.
    """
    log_spec = 10.0 * jnp.log10(jnp.maximum(amin, x))
    log_spec = log_spec - 10.0 * jnp.log10(jnp.maximum(amin, ref))

    if top_db is not None:
        if top_db < 0:
            raise librosa.util.exceptions.ParameterError("top_db must be non-negative")
        log_spec = jnp.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def amplitude_to_db(
    S: jnp.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
) -> Union[jnp.floating[Any], jnp.ndarray]:
    """Convert an amplitude spectrogram to decibel (dB) units.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,
    but is provided for convenience.

    Args:
        S: Input amplitude spectrogram.
        ref: Reference value for scaling. If scalar, the amplitude |S| is scaled relative
            to ref: 20 * log10(S / ref). If callable, the reference value is computed
            as ref(S). Default is 1.0.
        amin: Minimum threshold for input values. Default is 1e-5.
        top_db: Threshold the output at top_db below the peak. Default is 80.0.

    Returns:
        jnp.ndarray: dB-scaled spectrogram.

    See Also:
        power_to_db, db_to_amplitude
    """
    magnitude = jnp.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = jnp.abs(ref)

    power = jnp.square(magnitude)

    db: jnp.ndarray = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db


def fft_frequencies(sr: float = 22050, n_fft: int = 2048) -> jnp.ndarray:
    """Alternative interface for np.fft.rfftfreq, compatible with JAX.
    
    Args:
        sr: Audio sampling rate
        n_fft: FFT window size
        
    Returns:
        jnp.ndarray: Frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
    """
    return jnp.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def normalize(
    S: jnp.ndarray,
    *,
    norm: Optional[float] = jnp.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> jnp.ndarray:
    """Normalize an array along a chosen axis.
    
    Given a norm and a target axis, the input array is scaled so that:
        norm(S, axis=axis) == 1
        
    Args:
        S: The array to normalize
        norm: {jnp.inf, -jnp.inf, 0, float > 0, None}
            - jnp.inf: maximum absolute value
            - -jnp.inf: minimum absolute value  
            - 0: number of non-zeros (the support)
            - float: corresponding l_p norm
            - None: no normalization is performed
        axis: Axis along which to compute the norm
        threshold: Only the columns (or rows) with norm at least threshold are normalized.
            By default, the threshold is determined from the numerical precision of S.dtype.
        fill: If None, columns with norm below threshold are left as is.
            If False, columns with norm below threshold are set to 0.
            If True, columns with norm below threshold are filled uniformly such that norm is 1.
            
    Returns:
        jnp.ndarray: Normalized array
    """
    # Avoid div-by-zero
    if threshold is None:
        threshold = jnp.finfo(S.dtype).tiny
        
    if norm is None:
        return S
        
    # All norms only depend on magnitude
    mag = jnp.abs(S).astype(jnp.float32)
    
    # Compute the appropriate norm
    if norm == jnp.inf:
        length = jnp.max(mag, axis=axis, keepdims=True)
    elif norm == -jnp.inf:
        length = jnp.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")
        length = jnp.sum(mag > 0, axis=axis, keepdims=True).astype(mag.dtype)
    elif isinstance(norm, (int, float)) and norm > 0:
        length = jnp.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)
    else:
        raise ValueError(f"Unsupported norm: {repr(norm)}")
        
    # Handle small values
    small_idx = length < threshold
    
    if fill is None:
        # Leave small indices un-normalized
        length = jnp.where(small_idx, 1.0, length)
        Snorm = S / length
    elif fill:
        # Fill with uniform values
        # This is a simplified version - in practice you might want different fill strategies
        length = jnp.where(small_idx, jnp.nan, length)
        Snorm = S / length
        # For simplicity, we'll just use a constant fill value
        if axis is None:
            fill_norm = 1.0
        else:
            fill_norm = 1.0 / (S.shape[axis] ** (1.0 / norm) if norm > 0 else S.shape[axis])
        Snorm = jnp.where(jnp.isnan(Snorm), fill_norm, Snorm)
    else:
        # Set small values to zero
        length = jnp.where(small_idx, jnp.inf, length)
        Snorm = S / length
        
    return Snorm


def _spectrogram(
    y: Optional[jnp.ndarray] = None,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> tuple[jnp.ndarray, int]:
    """Helper function to compute magnitude spectrogram.
    
    This function is used internally by spectral feature extractors.
    Either y or S must be provided. If y is provided, the magnitude 
    spectrogram is computed. If S is provided, it is used directly.
    
    Args:
        y: Audio time series
        S: Pre-computed spectrogram magnitude
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        win_length: Window size
        window: Window function
        center: If True, pad the signal so that frames are centered
        pad_mode: Padding mode
        
    Returns:
        tuple: (magnitude spectrogram, n_fft used)
    """
    if S is not None:
        # Use the provided spectrogram
        n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Compute a magnitude spectrogram from scratch
        if y is None:
            raise ValueError("Either y or S must be provided")
            
        D = stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        S = jnp.abs(D)
        
    return S, n_fft


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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude
        n_fft: FFT window size
        hop_length: Hop length for STFT
        freq: Center frequencies for spectrogram bins. If None, FFT bin center
            frequencies are used.
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        
    Returns:
        jnp.ndarray: Spectral centroid frequencies [shape=(..., 1, t)]
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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate  
        S: (optional) Pre-computed spectrogram magnitude
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
        jnp.ndarray: Frequency bandwidth for each frame [shape=(..., 1, t)]
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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude
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
        jnp.ndarray: Roll-off frequency for each frame [shape=(..., 1, t)]
    """
    if not 0.0 < roll_percent < 1.0:
        raise ValueError("roll_percent must lie in the range (0, 1)")
        
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
        y: Audio time series. Multichannel is supported.
        S: (optional) Pre-computed spectrogram magnitude
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length
        window: Window function
        center: If True, pad the signal
        pad_mode: Padding mode
        amin: Minimum threshold for S (added noise floor for numerical stability)
        power: Exponent for the magnitude spectrogram (e.g., 1 for energy, 2 for power)
        
    Returns:
        jnp.ndarray: Spectral flatness for each frame [shape=(..., 1, t)]
    """
    if amin <= 0:
        raise ValueError("amin must be strictly positive")

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
        y: (optional) Audio time series. Required if S is not input.
        S: (optional) Spectrogram magnitude. Required if y is not input.
        frame_length: Length of analysis frame (in samples) for energy calculation
        hop_length: Hop length for STFT
        center: If True and operating on time-domain input (y), pad the signal
            by frame_length//2 on either side. Has no effect on spectrogram input.
        pad_mode: Padding mode for centered analysis
        dtype: Data type of the output array
        
    Returns:
        jnp.ndarray: RMS value for each frame [shape=(..., 1, t)]
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
        assert S.shape[-2] == frame_length // 2 + 1, (
                f"Since S.shape[-2] is {S.shape[-2]}, "
                f"frame_length is expected to be {S.shape[-2] * 2 - 2} or {S.shape[-2] * 2 - 1}; "
                f"found {frame_length}")
            
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
        raise ValueError("Either y or S must be input.")
        
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
        y: Audio time series. Multichannel is supported.
        frame_length: Length of the frame over which to compute zero crossing rates
        hop_length: Number of samples to advance for each frame
        center: If True, frames are centered by padding the edges of y.
            Uses edge-value copies instead of zero-padding.
        **kwargs: Additional keyword arguments to pass to zero_crossings
        
    Returns:
        jnp.ndarray: Zero crossing rate for each frame [shape=(..., 1, t)]
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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude
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
        jnp.ndarray: Spectral contrast values [shape=(..., n_bands + 1, t)]
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
        raise ValueError(f"freq.shape mismatch: expected ({S.shape[-2]},)")

    if n_bands < 1 or not isinstance(n_bands, int):
        raise ValueError("n_bands must be a positive integer")

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in the range (0, 1)")

    if fmin <= 0:
        raise ValueError("fmin must be a positive number")
        
    # Create octave bands
    octa = jnp.zeros(n_bands + 2)
    octa = octa.at[1:].set(fmin * (2.0 ** jnp.arange(0, n_bands + 1)))

    # if jnp.any(octa[:-1] >= 0.5 * sr):
    #     raise ValueError("Frequency band exceeds Nyquist. Reduce either fmin or n_bands.")

    # Initialize output arrays
    shape = list(S.shape)
    shape[-2] = n_bands + 1
    
    valley_list = []
    peak_list = []
    
    # Process each frequency band
    for k in range(n_bands + 1):
        f_low = octa[k]
        if k < n_bands:
            f_high = octa[k + 1]
        else:
            # Last band - set a high upper limit
            f_high = sr / 2
            
        # Find bins in current band
        current_band = jnp.logical_and(freq >= f_low, freq <= f_high)
        
        # Find indices of current band
        idx = jnp.where(current_band)[0]
        
        # Adjust band boundaries
        if k > 0 and len(idx) > 0 and idx[0] > 0:
            # Include one bin below for all but first band
            current_band = current_band.at[idx[0] - 1].set(True)
            
        if k == n_bands:
            # Include all bins above for last band
            idx_last = jnp.where(current_band)[0]
            if len(idx_last) > 0 and idx_last[-1] < len(freq) - 1:
                current_band = current_band.at[idx_last[-1] + 1:].set(True)
                
        # Extract sub-band
        sub_band = S[..., current_band, :]
        
        if k < n_bands and sub_band.shape[-2] > 1:
            # Remove last frequency bin for all but the last band
            sub_band = sub_band[..., :-1, :]
            
        # Calculate number of bins for peaks/valleys
        n_bins = sub_band.shape[-2]
        
        # Always take at least one bin from each side
        n_idx = int(jnp.maximum(jnp.rint(quantile * n_bins), 1))
        
        # Sort sub-band along frequency axis
        sortedr = jnp.sort(sub_band, axis=-2)
        
        # Compute valley (low energy) and peak (high energy)
        valley_val = jnp.mean(sortedr[..., :n_idx, :], axis=-2)
        peak_val = jnp.mean(sortedr[..., -n_idx:, :], axis=-2)
        
        valley_list.append(valley_val)
        peak_list.append(peak_val)
        
    # Stack results
    valley = jnp.stack(valley_list, axis=-2)
    peak = jnp.stack(peak_list, axis=-2)
            
    # Compute contrast
    if linear:
        contrast = peak - valley
    else:
        contrast = power_to_db(peak) - power_to_db(valley)
        
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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate
        S: (optional) Pre-computed spectrogram magnitude
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
            See librosa.filters.mel for details.
        dtype: Data type of the output array
        
    Returns:
        jnp.ndarray: Mel spectrogram [shape=(..., n_mels, t)]
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
    
    # Build mel filter matrix
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
        dtype=np.float32,  # librosa uses numpy
    )
    
    # Convert to JAX array
    mel_basis = jnp.array(mel_basis, dtype=dtype)
    
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
        y: Audio time series. Multichannel is supported.
        sr: Audio sampling rate
        S: (optional) log-power mel spectrogram
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
        jnp.ndarray: MFCC sequence [shape=(..., n_mfcc, t)]
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
    
    # Apply DCT
    from jax.scipy.fft import dct as jax_dct
    mfccs = jax_dct(S_transposed, type=dct_type, norm=norm, axis=-1)
    
    # Keep only the first n_mfcc coefficients
    mfccs = mfccs[..., :n_mfcc]
    
    # Transpose back to (..., n_mfcc, t)
    mfccs = jnp.moveaxis(mfccs, -1, -2)
    
    # Apply liftering if requested
    if lifter > 0:
        # Create liftering coefficients
        n = jnp.arange(n_mfcc)
        lift = 1 + (lifter / 2) * jnp.sin(jnp.pi * (n + 1) / lifter)
        
        # Reshape for broadcasting
        ndim_diff = mfccs.ndim - 1
        lift_shape = [1] * ndim_diff + [n_mfcc] + [1]
        lift = lift.reshape(lift_shape)
        
        mfccs = mfccs * lift
    
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
    dtype: jnp.dtype = jnp.float32,
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
    D = jnp.arange(0, n_chroma, dtype=jnp.float32)[jnp.newaxis, :] - frqbins[:, jnp.newaxis]
    
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


def note_to_hz(note: str, **kwargs) -> float:
    """Convert note name to frequency in Hz.
    
    Args:
        note: Note name (e.g., 'C4', 'A#3', 'Bb5')
        **kwargs: Additional arguments passed to librosa.note_to_hz
        
    Returns:
        float: Frequency in Hz
    """
    # For now, delegate to librosa
    return float(librosa.note_to_hz(note, **kwargs))


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


def _cqt_filter_bank_fixed(
    sr: float,
    n_bins: int,
    bins_per_octave: int,
    fmin: float,
    filter_scale: float,
    norm: Optional[float],
    window: str = "hann",
    n_fft_fixed: int = 16384,  # Fixed FFT size for JIT
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create CQT filter bank using proper wavelet basis functions.
    
    Returns:
        filters: Filter coefficients for each CQT bin
        lengths: Length of each filter
    """
    # Get CQT frequencies
    freqs = cqt_frequencies(
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        tuning=0.0
    )
    
    # Q factor (constant for all bins)
    Q = float(filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1))
    
    # Calculate filter lengths
    lengths = (Q * sr / freqs).astype(jnp.int32)
    
    # Create filters using vectorized operations
    # For JIT compatibility, we use a fixed FFT size
    
    # Create time grid for all filters
    t_max = jnp.arange(n_fft_fixed) / sr
    
    # Broadcast to create time arrays for each filter
    # Shape: (n_bins, n_fft_fixed)
    t_grid = t_max[jnp.newaxis, :]
    
    # Create masks for valid samples in each filter
    length_mask = t_grid < (lengths[:, jnp.newaxis] / sr)
    
    # Complex exponentials for all filters
    # Shape: (n_bins, n_fft_fixed)
    kernels = jnp.exp(-2j * jnp.pi * freqs[:, jnp.newaxis] * t_grid)
    
    # Apply window function
    if window == "hann":
        # Hann window for each filter length
        win = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * t_grid * sr / lengths[:, jnp.newaxis])
        win = jnp.where(length_mask, win, 0.0)
    else:
        # Rectangular window
        win = jnp.where(length_mask, 1.0, 0.0)
    
    # Apply window to kernels
    kernels = kernels * win
    
    # Normalize each filter
    if norm == 1:
        norm_factors = jnp.sum(jnp.abs(kernels), axis=1, keepdims=True)
        kernels = kernels / jnp.where(norm_factors > 0, norm_factors, 1.0)
    elif norm == 2:
        norm_factors = jnp.sqrt(jnp.sum(jnp.abs(kernels) ** 2, axis=1, keepdims=True))
        kernels = kernels / jnp.where(norm_factors > 0, norm_factors, 1.0)
    elif norm == jnp.inf:
        norm_factors = jnp.max(jnp.abs(kernels), axis=1, keepdims=True)
        kernels = kernels / jnp.where(norm_factors > 0, norm_factors, 1.0)
    
    filters = kernels
    
    return filters, lengths


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
) -> jnp.ndarray:
    """Compute the constant-Q transform using recursive downsampling.
    
    This implementation uses a simplified approach with fixed FFT size for JIT compatibility.
    
    Note:
        For JAX JIT compilation, all arguments except ``y`` should be marked as static:
        ``sr``, ``hop_length``, ``fmin``, ``n_bins``, ``bins_per_octave``, ``tuning``,
        ``filter_scale``, ``norm``, ``sparsity``, ``window``, ``scale``, ``pad_mode``,
        ``res_type``, ``dtype``
    
    Args:
        y: Audio time series
        sr: Sampling rate
        hop_length: Number of samples between successive CQT columns
        fmin: Minimum frequency (default: C1)
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
        tuning: Tuning offset in fractions of a bin
        filter_scale: Filter scale factor
        norm: Normalization type
        sparsity: Sparsification factor (not implemented)
        window: Window function
        scale: If True, scale by filter length
        pad_mode: Padding mode
        res_type: Resampling type (not used in this simplified version)
        dtype: Complex data type
        
    Returns:
        CQT matrix [shape=(n_bins, t)]
    """
    if fmin is None:
        fmin = note_to_hz("C1")
        
    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    
    # Calculate number of octaves
    n_octaves = jnp.ceil(n_bins / bins_per_octave).astype(jnp.int32)
    
    # For now, let's implement a simpler version that's JIT-compatible
    # The full recursive CQT with jax.lax.scan requires more complex state management
    # Fall back to an improved pseudo-CQT with better filter design
    
    # Use fixed FFT size for JIT compatibility
    n_fft = 16384
    
    # Create proper CQT filter bank
    filters, lengths = _cqt_filter_bank_fixed(
        sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        filter_scale=filter_scale,
        norm=norm,
        window=window,
        n_fft_fixed=n_fft,
    )
    
    # Compute STFT with the fixed FFT size
    D = stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode=pad_mode,
    )
    
    # Apply filters in frequency domain
    filters_fft = jnp.fft.fft(filters, axis=1)[:, :D.shape[0]]
    
    # Apply filters: multiply in frequency domain
    # D shape: (n_freqs, n_frames)
    # filters_fft shape: (n_bins, n_freqs)
    C = jnp.einsum('bf,ft->bt', filters_fft.conj(), D)
    
    if scale:
        # Scale by sqrt of filter lengths
        # Simplified - would need proper per-bin scaling
        C = C / jnp.sqrt(sr / hop_length)
    
    return C


def pseudo_cqt(
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
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Compute the pseudo constant-Q transform of an audio signal.
    
    This is a simplified CQT that uses a single FFT size rather than
    the recursive sub-sampling of the full CQT. It's faster but less
    accurate for low frequencies.
    
    Args:
        y: Audio time series. Multichannel is supported.
        sr: Sampling rate
        hop_length: Number of samples between successive CQT columns
        fmin: Minimum frequency. Defaults to C1 ~= 32.70 Hz
        n_bins: Number of frequency bins, starting at fmin
        bins_per_octave: Number of bins per octave
        tuning: Tuning offset in fractions of a bin. If None, assumed 0.
        filter_scale: Filter scale factor. Small values (<1) use shorter windows
        norm: Type of norm to use for basis function normalization
        sparsity: Sparsification factor (not implemented yet)
        window: Window function
        scale: If True, scale by sqrt(n_fft)
        pad_mode: Padding mode for frame analysis
        dtype: Complex data type for calculations
        
    Returns:
        jnp.ndarray: Pseudo Constant-Q transform [shape=(..., n_bins, t)]
    """
    if fmin is None:
        # C1 by default
        fmin = note_to_hz("C1")
        
    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    
    # Get CQT frequencies
    freqs = cqt_frequencies(
        n_bins=n_bins,
        bins_per_octave=bins_per_octave, 
        fmin=fmin,
        tuning=0.0  # Already applied above
    )
    
    # For pseudo-CQT, we use a simple approach with fixed FFT size
    # This is less accurate than full CQT but much simpler to implement
    
    # Estimate required FFT size
    # We need enough frequency resolution for the lowest frequency
    # and enough time resolution for the highest frequency
    alpha = 1.0  # Bandwidth factor (simplified)
    
    # Compute filter lengths for each frequency
    # Using a simplified version - proper implementation would use
    # filters.wavelet_lengths from librosa
    Q = filter_scale / alpha
    lengths = Q * sr / freqs
    
    # Use jax.lax.cond to select appropriate FFT size based on requirements
    # This allows some flexibility while maintaining JIT compatibility
    
    # Calculate required FFT size based on lowest frequency
    Q = filter_scale / alpha
    max_len = Q * sr / jnp.min(freqs)
    min_required = jnp.maximum(max_len, 2 * hop_length)
    
    # For JIT compatibility, use a fixed FFT size
    # This is a limitation of the current implementation
    # In practice, 16384 is large enough for most use cases
    n_fft = 16384
    
    # Compute STFT
    D = stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode=pad_mode,
    )
    
    # Get magnitude spectrogram
    S = jnp.abs(D)
    
    # Build the CQT kernel (simplified version)
    # In a full implementation, this would use proper wavelet basis functions
    n_freqs = 1 + n_fft // 2
    fft_freqs = fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Create Gaussian-like filters centered at CQT frequencies
    # This is a simplified approximation
    # Width is proportional to the CQT frequency (constant-Q property)
    sigmas = freqs / (bins_per_octave * 2)  # Simplified bandwidth
    
    # Vectorized computation of all filters
    # fft_freqs shape: (n_freqs,)
    # freqs shape: (n_bins,)
    # Broadcasting to compute all filters at once
    diff = fft_freqs[jnp.newaxis, :] - freqs[:, jnp.newaxis]  # (n_bins, n_freqs)
    weights = jnp.exp(-0.5 * (diff / sigmas[:, jnp.newaxis]) ** 2)
    
    # Normalize if requested
    if norm == 1:
        kernel = weights / jnp.sum(weights, axis=1, keepdims=True)
    elif norm == 2:
        kernel = weights / jnp.sqrt(jnp.sum(weights ** 2, axis=1, keepdims=True))
    elif norm == jnp.inf:
        kernel = weights / jnp.max(weights, axis=1, keepdims=True)
    else:
        kernel = weights
    
    # Apply CQT kernel to magnitude spectrogram
    # S shape: (..., n_freqs, time)
    # kernel shape: (n_bins, n_freqs)
    # Result shape: (..., n_bins, time)
    C = jnp.einsum("...ft,bf->...bt", S, kernel)
    
    if scale:
        C = C / jnp.sqrt(n_fft)
        
    return C


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
    window: Optional[jnp.ndarray] = None,
    bins_per_octave: int = 36,
    cqt_mode: str = "full",
    **kwargs,
) -> jnp.ndarray:
    """Chromagram from a constant-Q transform.
    
    Args:
        y: Audio time series. Multichannel is supported.
        sr: Sampling rate  
        C: Pre-computed CQT spectrogram
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
        **kwargs: Additional parameters for pseudo_cqt
        
    Returns:
        jnp.ndarray: Normalized chroma [shape=(..., n_chroma, t)]
    """
    if fmin is None:
        fmin = note_to_hz("C1")
        
    if C is None:
        # Use pseudo_cqt for now (full CQT not implemented yet)
        C = jnp.abs(
            pseudo_cqt(
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
                tuning=tuning,
                **kwargs,
            )
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
        y: Audio time series. Multichannel is supported.
        sr: Sampling rate of y
        chroma: Normalized energy for each chroma bin at each frame.
            If None, a chroma_stft is computed.
        **kwargs: Additional keyword arguments to chroma_stft,
            if chroma is not pre-computed.
            
    Returns:
        jnp.ndarray: Tonal centroid features [shape=(..., 6, t)]
            Tonnetz dimensions:
            - 0: Fifth x-axis
            - 1: Fifth y-axis
            - 2: Minor x-axis
            - 3: Minor y-axis
            - 4: Major x-axis
            - 5: Major y-axis
    """
    if y is None and chroma is None:
        raise ValueError(
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
        y: Audio time series. Multichannel is supported.
        sr: Sampling rate
        S: Power spectrogram (optional if y is provided)
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
        jnp.ndarray: Chromagram [shape=(..., n_chroma, t)]
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
    
    # For now, we don't implement automatic tuning estimation
    # Users need to provide tuning explicitly or use default 0.0
    if tuning is None:
        # In a full implementation, we would estimate tuning here
        # For now, just use A440 standard tuning
        tuning = 0.0
    
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
