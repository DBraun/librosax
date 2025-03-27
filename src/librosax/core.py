from typing import Any, Callable, Optional, Union

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
                reconstructed_signal, ((0, 0), (0, 0), (0, pad_width)), mode="constant"
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
