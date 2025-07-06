from typing import Any, Callable, Optional, Union

from jax import numpy as jnp
from jax.scipy import signal
import numpy as np
from scipy.signal import get_window
from ..util.exceptions import ParameterError


__all__ = [
    "stft",
    "istft",
    # "magphase",
    # "iirt",
    # "reassigned_spectrogram",
    # "phase_vocoder",
    # "perceptual_weighting",
    "power_to_db",
    "db_to_power",
    "amplitude_to_db",
    "db_to_amplitude",
    # "fmt",
    # "pcen",
    # "griffinlim",
]


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

    This function is a JAX implementation of ``librosax.power_to_db``.

    Args:
        x: Input power spectrogram.
        amin: Minimum threshold for input values. Default is 1e-10.
        top_db: Threshold the output at top_db below the peak. Default is 80.0.
        ref: Reference value for scaling. Default is 1.0.

    Returns:
        jnp.ndarray: dB-scaled spectrogram.

    Raises:
        librosax.util.exceptions.ParameterError: If ``top_db`` is negative.
    """
    log_spec = 10.0 * jnp.log10(jnp.maximum(amin, x))
    log_spec = log_spec - 10.0 * jnp.log10(jnp.maximum(amin, ref))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = jnp.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def db_to_power(S_db: jnp.ndarray, *, ref: float = 1.0) -> jnp.ndarray:
    """Convert dB-scale values to a power values.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power values

    Notes
    -----
    This function caches at level 30.
    """
    return ref * jnp.power(10.0, S_db * 0.1)


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


def db_to_amplitude(S_db: jnp.ndarray, *, ref: float = 1.0) -> jnp.ndarray:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Optional reference amplitude.

    Returns
    -------
    S : np.ndarray
        Linear magnitude values

    Notes
    -----
    This function caches at level 30.
    """
    return jnp.sqrt(db_to_power(S_db, ref=ref**2))


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


