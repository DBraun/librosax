from typing import Any, Callable, Optional, Union

import jax
from jax import numpy as jnp
from jax.scipy import signal
from jax.scipy.fft import dct as jax_dct
import librosa.filters
import numpy as np
from scipy.fftpack import dct as scipy_dct
import warnings

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
    "_dct_flexible",
]


def _dct_flexible(x: jnp.ndarray, type: int = 2, norm: Optional[str] = None, axis: int = -1) -> jnp.ndarray:
    """DCT with fallback to scipy for types 1 and 3.

    JAX only supports DCT type 2. For types 1 and 3, we use scipy via
    jax.pure_callback to maintain compatibility with JAX transformations.

    Args:
        x: Input array.
        type: DCT type (1, 2, 3, or 4). Default is 2.
        norm: Normalization mode. Default is None.
        axis: Axis along which to compute DCT. Default is -1.

    Returns:
        DCT of the input array.
    """
    if type == 2:
        return jax_dct(x, type=type, norm=norm, axis=axis)
    else:
        def scipy_dct_wrapper(x_arr):
            return scipy_dct(np.asarray(x_arr), type=type, norm=norm, axis=axis).astype(x_arr.dtype)

        return jax.pure_callback(
            scipy_dct_wrapper,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            x
        )


_WindowSpec = Union[str, tuple, float, Callable, np.ndarray, jnp.ndarray]


def _analysis_window(window: _WindowSpec, win_length: int, n_fft: int) -> np.ndarray:
    """Build a zero-padded analysis window of length ``n_fft``.

    Args:
        window: Window specification. Accepts everything
            ``librosa.filters.get_window`` does — a name such as ``"hann"``, a
            ``(name, param)`` tuple such as ``("kaiser", 4.0)``, a float
            interpreted as a Kaiser beta, a callable invoked as
            ``window(win_length)``, or an explicit array of length
            ``win_length``. The extra name ``"sqrt_hann"`` gives the square
            root of a Hann window.
        win_length: Length of the window itself.
        n_fft: Length of the padded output; the window is centered within it.

    Returns:
        NumPy array of length ``n_fft``. The result is NumPy rather than JAX so
        that it stays a compile-time constant when the caller is under
        ``jax.jit``.

    Raises:
        librosa.util.exceptions.ParameterError: If an explicit window array
            does not have length ``win_length``.
    """
    if isinstance(window, jax.Array):
        # librosa's dispatcher recognizes lists and NumPy arrays, not JAX ones.
        window = np.asarray(window)

    if isinstance(window, str) and window == "sqrt_hann":
        win = np.sqrt(librosa.filters.get_window("hann", win_length, fftbins=True))
    else:
        win = np.asarray(
            librosa.filters.get_window(window, win_length, fftbins=True),
            dtype=float,
        )

    padded_win = np.zeros(n_fft)
    start = (n_fft - win_length) // 2
    padded_win[start : start + win_length] = win
    return padded_win


def stft(
    waveform: jnp.ndarray,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: str = "constant",
):
    """Compute the Short-Time Fourier Transform (STFT) of a waveform.

    This function computes the STFT of the given waveform using JAX's ``scipy.signal.stft`` implementation.

    Note:
        For JAX JIT compilation, the following arguments should be marked as static:
        ``n_fft``, ``hop_length``, ``win_length``, ``window``, ``center``, ``pad_mode``

    Args:
        waveform: Input signal waveform. The last axis must be time (samples).
            Supported shapes:

            - ``(T,)`` - single waveform with T samples
            - ``(B, T)`` - batch of B waveforms
            - ``(B, C, T)`` - batch of B waveforms with C channels
        n_fft: FFT size. Determines the number of frequency bins in the output:
            ``n_fft // 2 + 1``.
        hop_length: Number of samples between successive frames. Default is ``win_length // 4``.
        win_length: Window size. Default is ``n_fft``.
        window: Window specification. Accepts anything
            ``librosa.filters.get_window`` does: a name such as ``"hann"``, a
            ``(name, param)`` tuple such as ``("kaiser", 4.0)``, a float
            (Kaiser beta), a callable invoked as ``window(win_length)`` such as
            ``np.hanning``, or an explicit array of length ``win_length``.
            librosax additionally accepts ``"sqrt_hann"``. Default is
            ``"hann"``.

            Note that ``np.hanning`` is the *symmetric* Hann window, which
            differs from the periodic window ``"hann"`` selects.
        center: If ``True``, the waveform is padded so that frames are centered. Default is ``True``.
        pad_mode: Padding mode for the waveform. Must be one of ``["constant", "reflect"]``. Default is ``"constant"``.

    Returns:
        Complex STFT matrix. The second-to-last axis is frequency (``n_fft // 2 + 1`` bins)
        and the last axis is time frames. Output shapes correspond to inputs:

        - ``(T,)`` → ``(F, N)``
        - ``(B, T)`` → ``(B, F, N)``
        - ``(B, C, T)`` → ``(B, C, F, N)``

        where F = ``n_fft // 2 + 1`` and N = number of frames.

    Raises:
        AssertionError: If pad_mode is not one of ``["constant", "reflect"]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import librosax
        >>> # Single waveform: (n_samples,) -> (n_freq, n_frames)
        >>> y = jnp.zeros(22050)  # 1 second at 22050 Hz
        >>> S = librosax.stft(y, n_fft=2048, hop_length=512)
        >>> S.shape
        (1025, 44)
        >>> # Batched waveforms: (batch, n_samples) -> (batch, n_freq, n_frames)
        >>> y_batch = jnp.zeros((4, 22050))
        >>> S_batch = librosax.stft(y_batch, n_fft=2048, hop_length=512)
        >>> S_batch.shape
        (4, 1025, 44)
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

    win = _analysis_window(window, win_length, n_fft)

    _, _, Zxx = signal.stft(
        waveform,
        window=jnp.asarray(win),
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=boundary if center else None,
        padded=False,
        axis=-1,
    )
    Zxx = Zxx * win.sum()
    return Zxx


def istft(
    stft_matrix: jnp.ndarray,
    hop_length: int = None,
    win_length: int = None,
    n_fft: int = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    length: int = None,
):
    """Compute the Inverse Short-Time Fourier Transform (ISTFT).

    This function reconstructs a waveform from an STFT matrix using JAX's ``scipy.signal.istft`` implementation.

    Args:
        stft_matrix: Complex STFT matrix. The second-to-last axis must be frequency
            (``n_fft // 2 + 1`` bins) and the last axis must be time frames.
            Supported shapes:

            - ``(F, N)`` - single spectrogram
            - ``(B, F, N)`` - batch of B spectrograms
            - ``(B, C, F, N)`` - batch of B spectrograms with C channels

            where F = ``n_fft // 2 + 1`` and N = number of frames.
        hop_length: Number of samples between successive frames. Default is ``win_length // 4``.
        win_length: Window size. Default is ``n_fft``.
        n_fft: FFT size. Default is ``(stft_matrix.shape[-2] - 1) * 2``, inferred from
            the frequency axis of the input.
        window: Window specification, as in ``stft``: a name, a ``(name,
            param)`` tuple, a Kaiser beta, a callable such as ``np.hanning``,
            or an explicit array of length ``win_length``. Default is
            ``"hann"``.
        center: If ``True``, assumes the waveform was padded so that frames were centered. Default is ``True``.
        length: Target length for the reconstructed signal. If ``None``, the entire signal is returned.

    Returns:
        Reconstructed time-domain signal. The last axis is time (samples).
        Output shapes correspond to inputs:

        - ``(F, N)`` → ``(T,)``
        - ``(B, F, N)`` → ``(B, T)``
        - ``(B, C, F, N)`` → ``(B, C, T)``

        where T is the reconstructed signal length.

    Raises:
        AssertionError: If center is ``False`` because the function is only tested for ``center=True``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import librosax
        >>> # Single STFT: (n_freq, n_frames) -> (n_samples,)
        >>> S = jnp.zeros((1025, 44), dtype=jnp.complex64)
        >>> y = librosax.istft(S, hop_length=512)
        >>> y.shape
        (22016,)
        >>> # Batched STFT: (batch, n_freq, n_frames) -> (batch, n_samples)
        >>> S_batch = jnp.zeros((4, 1025, 44), dtype=jnp.complex64)
        >>> y_batch = librosax.istft(S_batch, hop_length=512)
        >>> y_batch.shape
        (4, 22016)
        >>> # With target length
        >>> y_trimmed = librosax.istft(S, hop_length=512, length=22050)
        >>> y_trimmed.shape
        (22050,)
    """
    # Only center=True is currently supported
    # center=False would require different padding/trimming logic
    if not center:
        raise NotImplementedError("istft currently only supports center=True")

    if n_fft is None:
        n_fft = (stft_matrix.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    win = _analysis_window(window, win_length, n_fft)

    _, reconstructed_signal = signal.istft(
        stft_matrix,
        window=jnp.asarray(win),
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=center,
    )

    reconstructed_signal = reconstructed_signal / win.sum()

    # Trim or pad the output signal to the desired length
    if length is not None:
        if length > reconstructed_signal.shape[-1]:
            # Pad the signal if it is shorter than the desired length
            pad_width = length - reconstructed_signal.shape[-1]
            reconstructed_signal = jnp.pad(
                reconstructed_signal,
                [(0, 0)] * (reconstructed_signal.ndim - 1) + [(0, pad_width)],
                mode="constant",
            )
        else:
            # Trim the signal if it is longer than the desired length
            reconstructed_signal = reconstructed_signal[..., :length]

    return reconstructed_signal


_AxesSpec = Union[str, int, tuple, None]


def _resolve_axes(x: jnp.ndarray, axes: _AxesSpec) -> Union[int, tuple, None]:
    """Resolve the ``axes="auto"`` sentinel against an array's rank.

    Args:
        x: Array whose rank determines the inferred axes.
        axes: Either the literal ``"auto"``, or an explicit axis specification
            that is returned unchanged.

    Returns:
        ``None`` for scalars, ``(-1,)`` for 1-d input and ``(-2, -1)`` for
        anything higher, when ``axes`` is ``"auto"``; otherwise ``axes``.
    """
    if axes != "auto":
        return axes
    if x.ndim >= 2:
        return (-2, -1)
    if x.ndim == 1:
        return (-1,)
    return None


def power_to_db(
    S: jnp.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
    axes: _AxesSpec = "auto",
) -> jnp.ndarray:
    """Convert a power spectrogram to decibel (dB) units.

    This computes the scaling ``10 * log10(S / ref)`` in a numerically stable
    way.

    Args:
        S: Input power spectrogram. Any shape; applied element-wise.
        ref: Reference value for scaling. If scalar, the power is scaled
            relative to ``ref``. If callable, the reference value is computed as
            ``ref(S, axis=axes, keepdims=True)``, so each slice along ``axes``
            gets its own reference. Default is 1.0.
        amin: Minimum threshold for input values. Default is 1e-10.
        top_db: Threshold the output at ``top_db`` below the peak of each slice
            along ``axes``. Default is 80.0.
        axes: Axis or axes over which ``ref`` and ``top_db`` are reduced. If
            ``"auto"`` (default), the axes are inferred from the rank of ``S``:
            ``None`` for a scalar, ``(-1,)`` for 1-d, and ``(-2, -1)``
            otherwise, so that batched inputs are scaled independently. Pass
            ``None`` to reduce over the whole array at once.

    Returns:
        dB-scaled spectrogram with same shape as input.

    Raises:
        librosax.util.exceptions.ParameterError: If ``amin`` is not strictly
            positive, if ``top_db`` is negative, or if a callable ``ref`` does
            not accept ``axis`` and ``keepdims``.
    """
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if jnp.issubdtype(S.dtype, jnp.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(jnp.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = jnp.abs(S)
    else:
        magnitude = S

    axes = _resolve_axes(magnitude, axes)

    if callable(ref):
        # The reference is computed per slice so that batched inputs do not
        # share a single global reference.
        try:
            ref_value = ref(magnitude, axis=axes, keepdims=True)
        except TypeError as exc:
            raise ParameterError(
                "The provided reference function must support 'axis' and "
                "'keepdims' arguments for proper multichannel processing."
            ) from exc
    else:
        ref_value = jnp.abs(ref)

    log_spec = 10.0 * jnp.log10(jnp.maximum(amin, magnitude))
    log_spec = log_spec - 10.0 * jnp.log10(jnp.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = jnp.maximum(
            log_spec, jnp.max(log_spec, axis=axes, keepdims=True) - top_db
        )

    return log_spec


def db_to_power(S_db: jnp.ndarray, *, ref: float = 1.0) -> jnp.ndarray:
    """Convert dB-scale values to power values.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    Args:
        S_db: dB-scaled values.
        ref: Reference power. Output will be scaled by this value.

    Returns:
        Power values.

    Note:
        This function caches at level 30.
    """
    return ref * jnp.power(10.0, S_db * 0.1)


def amplitude_to_db(
    S: jnp.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
    axes: _AxesSpec = "auto",
) -> Union[jnp.floating[Any], jnp.ndarray]:
    """Convert an amplitude spectrogram to decibel (dB) units.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2,
    top_db=top_db)``, but is provided for convenience.

    Args:
        S: Input amplitude spectrogram. Any shape; applied element-wise.
        ref: Reference value for scaling. If scalar, the amplitude ``abs(S)`` is
            scaled relative to ``ref``: ``20 * log10(S / ref)``. If callable,
            the reference value is computed as
            ``ref(abs(S), axis=axes, keepdims=True)``. Default is 1.0.
        amin: Minimum threshold for input values. Default is 1e-5.
        top_db: Threshold the output at ``top_db`` below the peak of each slice
            along ``axes``. Default is 80.0.
        axes: Axis or axes over which ``ref`` and ``top_db`` are reduced. See
            ``power_to_db``.

    Returns:
        dB-scaled spectrogram with same shape as input.

    See Also:
        power_to_db, db_to_amplitude
    """
    magnitude = jnp.abs(S)
    axes = _resolve_axes(magnitude, axes)

    if callable(ref):
        # User supplied a function to calculate reference power
        try:
            ref_value = ref(magnitude, axis=axes, keepdims=True)
        except TypeError as exc:
            raise ParameterError(
                "The provided reference function must support 'axis' and "
                "'keepdims' arguments for proper multichannel processing."
            ) from exc
    else:
        ref_value = jnp.abs(ref)

    power = jnp.square(magnitude)

    db: jnp.ndarray = power_to_db(
        power, ref=ref_value**2, amin=amin**2, top_db=top_db, axes=axes
    )
    return db


def db_to_amplitude(S_db: jnp.ndarray, *, ref: float = 1.0) -> jnp.ndarray:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Args:
        S_db: dB-scaled values.
        ref: Optional reference amplitude.

    Returns:
        Linear magnitude values.

    Note:
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


def tiny(x: jnp.ndarray) -> float:
    """Return the smallest positive normal value for an array's dtype.

    Non-floating dtypes have no such value, so float32's is used, matching
    ``librosa.util.tiny``.

    Args:
        x: Array whose dtype is inspected.

    Returns:
        The smallest positive normal representable in the inferred dtype.

    Examples:
        >>> librosax.core.spectrum.tiny(jnp.zeros(3))
        1.1754944e-38
    """
    dtype = jnp.asarray(x).dtype
    if not (
        jnp.issubdtype(dtype, jnp.floating)
        or jnp.issubdtype(dtype, jnp.complexfloating)
    ):
        dtype = jnp.dtype(jnp.float32)

    return jnp.finfo(dtype).tiny


def normalize(
    S: jnp.ndarray,
    *,
    norm: Optional[float] = jnp.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> jnp.ndarray:
    """Normalize an array along a chosen axis.

    Given a norm and a target axis, the input array is scaled so that
    ``norm(S, axis=axis) == 1``.

    Args:
        S: The array to normalize.
        norm: Normalization order.

            - ``jnp.inf``: maximum absolute value
            - ``-jnp.inf``: minimum absolute value
            - ``0``: number of non-zeros (the support)
            - float > 0: the corresponding l_p norm
            - ``None``: no normalization is performed

        axis: Axis along which to compute the norm. If ``None``, the norm is
            taken over the whole array.
        threshold: Only slices with norm of at least ``threshold`` are
            normalized. Defaults to the numerical precision of ``S.dtype``.
        fill: If ``None``, slices with norm below ``threshold`` are left as is.
            If ``False``, they are set to 0. If ``True``, they are filled
            uniformly such that their norm is 1.

    Returns:
        Normalized array with the same shape as ``S``.

    Raises:
        librosax.util.exceptions.ParameterError: If ``threshold`` is not
            strictly positive, if ``fill`` is not ``None`` or boolean, if
            ``norm`` is unsupported, or if ``norm=0`` is combined with
            ``fill=True``.

    Examples:
        >>> librosax.util.normalize(jnp.array([[1.0, 2.0], [3.0, 4.0]]), norm=2, axis=0)
        Array([[0.31622776, 0.4472136 ],
               [0.9486833 , 0.8944272 ]], dtype=float32)
    """
    if threshold is None:
        threshold = tiny(S)
    elif threshold <= 0:
        raise ParameterError(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise ParameterError(f"fill={fill} must be None or boolean")

    if norm is None:
        return S

    # All norms only depend on magnitude
    mag = jnp.abs(S).astype(float)

    # For max/min/support norms, filling with 1 gives a unit norm
    fill_norm = 1

    if norm == jnp.inf:
        length = jnp.max(mag, axis=axis, keepdims=True)
    elif norm == -jnp.inf:
        length = jnp.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")
        length = jnp.sum(mag > 0, axis=axis, keepdims=True).astype(mag.dtype)
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = jnp.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)
    else:
        raise ParameterError(f"Unsupported norm: {norm!r}")

    # Indices where the norm is below the threshold
    small_idx = length < threshold

    if fill is None:
        # Leave small indices un-normalized
        length = jnp.where(small_idx, 1.0, length)
        Snorm = S / length
    elif fill:
        # Divide the small slices to NaN, then substitute the uniform fill
        length = jnp.where(small_idx, jnp.nan, length)
        Snorm = S / length
        Snorm = jnp.where(jnp.isnan(Snorm), fill_norm, Snorm)
    else:
        # Divide the small slices to infinity, which zeroes them
        length = jnp.where(small_idx, jnp.inf, length)
        Snorm = S / length

    return Snorm


def _spectrogram(
    y: Optional[jnp.ndarray] = None,
    S: Optional[jnp.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
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
        window: Window specification, as in ``stft``: a name, a ``(name,
            param)`` tuple, a Kaiser beta, a callable such as ``np.hanning``,
            or an explicit array of length ``win_length``. Default is
            ``"hann"``.
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


