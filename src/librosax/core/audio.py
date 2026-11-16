#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio signal generation and processing utilities"""
from typing import Optional
from jax import numpy as jnp
import numpy as np
import librosa
from ..util.exceptions import ParameterError


__all__ = [
    "tone",
    "autocorrelate",
    "fix_length",
    "to_mono",
    "to_stereo",
    "to_multi",
]


def tone(
    frequency: float,
    *,
    sr: float = 22050,
    length: Optional[int] = None,
    duration: Optional[float] = None,
    phi: Optional[float] = None,
) -> jnp.ndarray:
    """Construct a pure tone (cosine) signal at a given frequency.

    Parameters
    ----------
    frequency : float > 0
        frequency
    sr : number > 0
        desired sampling rate of the output signal
    length : int > 0
        desired number of samples in the output signal.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    duration : float > 0
        desired duration in seconds.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    phi : float or None
        phase offset, in radians. If unspecified, defaults to ``-jnp.pi * 0.5``.

    Returns
    -------
    tone_signal : jnp.ndarray [shape=(length,), dtype=float64]
        Synthesized pure sine tone signal

    Raises
    ------
    ParameterError
        - If ``frequency`` is not provided.
        - If neither ``length`` nor ``duration`` are provided.

    Examples
    --------
    Generate a pure sine tone A4

    >>> tone = librosax.tone(440, duration=1, sr=22050)

    Or generate the same signal using `length`

    >>> tone = librosax.tone(440, sr=22050, length=22050)
    """
    if frequency is None:
        raise ParameterError('"frequency" must be provided')

    # Compute signal length
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
        length = int(duration * sr)

    if phi is None:
        phi = -jnp.pi * 0.5

    y: jnp.ndarray = jnp.cos(2 * jnp.pi * frequency * jnp.arange(length) / sr + phi)
    return y


def autocorrelate(
    y: jnp.ndarray, *, max_size: Optional[int] = None, axis: int = -1
) -> jnp.ndarray:
    """Bounded-lag auto-correlation.

    This function computes the autocorrelation of a signal using FFT for efficiency.
    The autocorrelation is bounded to a maximum lag specified by ``max_size``.

    Args:
        y: Array to autocorrelate.
        max_size: Maximum correlation lag. If None, defaults to ``y.shape[axis]`` (unbounded).
        axis: Axis along which to autocorrelate. Default is -1 (last axis).

    Returns:
        Autocorrelated array. If ``max_size`` is specified, the shape along ``axis``
        will be ``max_size``. Otherwise, it matches ``y.shape[axis]``.

    Examples:
        Compute full autocorrelation of a white noise signal

        >>> import numpy as np
        >>> y = np.random.randn(256)
        >>> z = librosax.autocorrelate(y)
        >>> z.shape
        (256,)

        Compute autocorrelation with a maximum lag of 32 samples

        >>> z = librosax.autocorrelate(y, max_size=32)
        >>> z.shape
        (32,)

        Autocorrelate along the time axis of a batch of signals

        >>> y = np.random.randn(10, 256)  # 10 signals of length 256
        >>> z = librosax.autocorrelate(y, axis=-1)
        >>> z.shape
        (10, 256)
    """
    y = jnp.asarray(y)

    n = y.shape[axis]
    if max_size is None:
        max_size = n
    max_size = int(min(max_size, n))

    # Zero-pad past 2n-1 so the circular correlation of the FFT does not wrap
    # onto the lags we keep.
    n_pad = 2 * n - 1

    if jnp.issubdtype(y.dtype, jnp.complexfloating):
        powspec = jnp.abs(jnp.fft.fft(y, n=n_pad, axis=axis)) ** 2
        autocorr = jnp.fft.ifft(powspec, n=n_pad, axis=axis)
    else:
        powspec = jnp.abs(jnp.fft.rfft(y, n=n_pad, axis=axis)) ** 2
        autocorr = jnp.fft.irfft(powspec, n=n_pad, axis=axis)

    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)
    return autocorr[tuple(subslice)]


def fix_length(data: jnp.ndarray, *, size: int, axis: int = -1, **kwargs) -> jnp.ndarray:
    """Fix the length of an array along an axis by trimming or zero-padding.

    Args:
        data: Input array.
        size: Target length along ``axis``.
        axis: Axis to resize.
        **kwargs: Additional arguments forwarded to ``jnp.pad`` when padding is
            needed. Defaults to zero-padding at the end.

    Returns:
        Array whose extent along ``axis`` is exactly ``size``.

    Examples:
        >>> librosax.fix_length(jnp.arange(7.0), size=10)
        Array([0., 1., 2., 3., 4., 5., 6., 0., 0., 0.], dtype=float32)
        >>> librosax.fix_length(jnp.arange(7.0), size=3)
        Array([0., 1., 2.], dtype=float32)
    """
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    if n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return jnp.pad(data, lengths, **kwargs)

    return data


def _validate_signals(signals: tuple) -> tuple:
    """Check a signal collection and report its length range and common dtype.

    Only the static parts of ``librosa.util.valid_audio`` are enforced: testing
    every sample for finiteness would force a device-to-host sync.

    Args:
        signals: One or more audio buffers, time along the last axis.

    Returns:
        ``(signals, n_min, n_max, dtype)`` where the signals have been coerced
        to JAX arrays, the lengths are taken over the last axis, and the dtype
        is promoted across all inputs. Coercion happens first so that the
        promoted dtype respects the ``jax_enable_x64`` setting; promoting raw
        NumPy float64 input would otherwise ask for an unavailable dtype.

    Raises:
        librosax.util.exceptions.ParameterError: If no signals are given, or if
            any signal is a scalar or not floating-point.
    """
    if not signals:
        raise ParameterError("At least one signal must be provided.")

    signals = tuple(jnp.asarray(y) for y in signals)

    n_min = n_max = signals[0].shape[-1]
    dtype = signals[0].dtype
    for y in signals:
        if y.ndim == 0:
            raise ParameterError(
                f"Audio data must be at least one-dimensional, given y.shape={y.shape}"
            )
        if not jnp.issubdtype(y.dtype, jnp.floating):
            raise ParameterError(
                f"Audio data must be of floating-point type, given y.dtype={y.dtype}"
            )
        n_min = min(n_min, y.shape[-1])
        n_max = max(n_max, y.shape[-1])
        dtype = jnp.promote_types(dtype, y.dtype)

    return signals, n_min, n_max, dtype


def to_mono(*signals: jnp.ndarray, pad: bool = True, norm: bool = True) -> jnp.ndarray:
    """Downmix one or more signals to a single mono channel.

    Each input is first collapsed across its channel dimensions, then the
    results are combined.

    Args:
        *signals: One or more audio buffers with shape ``(..., n)``. Signals may
            differ in both channel layout and length.
        pad: If ``True`` (default), shorter signals are zero-padded to the
            longest input. If ``False``, longer signals are trimmed to the
            shortest.
        norm: If ``True`` (default), channels and signals are combined by
            averaging. If ``False``, they are combined by summing.

    Returns:
        Mono signal with shape ``(n_out,)``.

    Raises:
        librosax.util.exceptions.ParameterError: If no signals are given, or if
            any signal is a scalar or not floating-point.

    See Also:
        to_stereo, to_multi, fix_length

    Examples:
        >>> y1 = librosax.tone(440.0, sr=22050, duration=1.0)
        >>> y2 = librosax.tone(550.0, sr=22050, duration=1.0)
        >>> y_mono = librosax.to_mono(y1, y2)
    """
    signals, n_min, n_max, dtype = _validate_signals(signals)
    size = n_max if pad else n_min

    combine = jnp.mean if norm else jnp.sum

    output = jnp.zeros(size, dtype=dtype)
    for y in signals:
        mono = combine(y, axis=tuple(range(y.ndim - 1))) if y.ndim > 1 else y
        output = output + fix_length(mono.astype(dtype), size=size, axis=-1)

    if norm:
        output = output / len(signals)

    return output


def to_stereo(
    *,
    left: Optional[jnp.ndarray],
    right: Optional[jnp.ndarray],
    downmix: bool = True,
    pad: bool = True,
    norm: bool = True,
) -> jnp.ndarray:
    """Combine two signals into a stereo signal.

    Args:
        left: Left channel signal with shape ``(..., n)``, or ``None`` for
            silence.
        right: Right channel signal with shape ``(..., m)``, or ``None`` for
            silence.
        downmix: If ``True`` (default), each side is downmixed to mono before
            being placed in its output row. If ``False``, the inputs must
            already be mono or stereo and are combined additively.
        pad: If ``True`` (default), the shorter channel is zero-padded to match
            the longer one. If ``False``, the longer one is trimmed.
        norm: If ``True`` (default), signals are combined by averaging. If
            ``False``, they are combined by summing.

    Returns:
        Stereo signal with shape ``(2, n_out)``, left channel first.

    Raises:
        librosax.util.exceptions.ParameterError: If both ``left`` and ``right``
            are ``None``, or if a channel has an unsupported shape when
            ``downmix`` is ``False``.

    See Also:
        to_mono, to_multi, fix_length

    Examples:
        >>> y1 = librosax.tone(440.0, sr=22050, duration=1.0)
        >>> y2 = librosax.tone(550.0, sr=22050, duration=1.0)
        >>> y_stereo = librosax.to_stereo(left=y1, right=y2)

        Upmix a mono signal into the left channel only

        >>> y_left = librosax.to_stereo(left=y1, right=None)
    """
    onesided = True
    if left is None and right is None:
        raise ParameterError("At least one of 'left' or 'right' must be provided")
    elif left is None:
        # Coerce first so the silent channel picks up a canonical JAX dtype.
        right = jnp.asarray(right)
        left = jnp.zeros_like(right)
    elif right is None:
        left = jnp.asarray(left)
        right = jnp.zeros_like(left)
    else:
        left, right = jnp.asarray(left), jnp.asarray(right)
        onesided = False

    if pad:
        size = max(left.shape[-1], right.shape[-1])
    else:
        size = min(left.shape[-1], right.shape[-1])

    left = fix_length(left, size=size, axis=-1)
    right = fix_length(right, size=size, axis=-1)

    dtype = jnp.promote_types(left.dtype, right.dtype)
    left = left.astype(dtype)
    right = right.astype(dtype)

    if downmix:
        return jnp.stack(
            [to_mono(left, norm=norm), to_mono(right, norm=norm)], axis=0
        )

    output = jnp.zeros((2, size), dtype=dtype)

    if left.ndim == 1:
        output = output.at[0].set(left)
    elif left.ndim == 2 and left.shape[0] == 2:
        output = left
    else:
        raise ParameterError(
            f"left input has unsupported shape {left.shape} for downmix=False"
        )

    if right.ndim == 1:
        output = output.at[1].add(right)
    elif right.ndim == 2 and right.shape[0] == 2:
        output = output + right
    else:
        raise ParameterError(
            f"right input has unsupported shape {right.shape} for downmix=False"
        )

    if norm and not onesided:
        output = output / 2

    return output


def to_multi(
    *signals: jnp.ndarray,
    downmix: bool = True,
    pad: bool = True,
    norm: bool = True,
) -> jnp.ndarray:
    """Combine several signals into one multi-channel signal.

    Args:
        *signals: One or more audio buffers with shape ``(..., n)``.
        downmix: If ``True`` (default), each input is downmixed to mono and
            becomes one channel of the output, giving shape
            ``(len(signals), n_out)``. If ``False``, all inputs must share a
            channel layout and are combined additively into it.
        pad: If ``True`` (default), shorter signals are zero-padded to the
            longest input. If ``False``, longer signals are trimmed.
        norm: If ``True`` (default), signals are combined by averaging. If
            ``False``, they are combined by summing.

    Returns:
        Multi-channel signal.

    Raises:
        librosax.util.exceptions.ParameterError: If no signals are given, or if
            ``downmix`` is ``False`` and the inputs disagree on channel layout.

    See Also:
        to_mono, to_stereo, fix_length

    Examples:
        >>> y1 = librosax.tone(440.0, sr=22050, duration=1.0)
        >>> y2 = librosax.tone(550.0, sr=22050, duration=1.0)
        >>> y_multi = librosax.to_multi(y1, y2)
        >>> y_multi.shape
        (2, 22050)
    """
    signals, n_min, n_max, dtype = _validate_signals(signals)

    if not downmix:
        channel_layout = tuple(signals[0].shape[:-1])
        for y in signals:
            if tuple(y.shape[:-1]) != channel_layout:
                raise ParameterError(
                    f"Cannot combine signals with different channel layouts "
                    f"{y.shape[:-1]} when downmix=False"
                )

    size = n_max if pad else n_min

    if downmix:
        return jnp.stack(
            [
                fix_length(to_mono(y, norm=norm).astype(dtype), size=size, axis=-1)
                for y in signals
            ],
            axis=0,
        )

    output = jnp.zeros(channel_layout + (size,), dtype=dtype)
    for y in signals:
        output = output + fix_length(y.astype(dtype), size=size, axis=-1)

    if norm:
        output = output / len(signals)

    return output
