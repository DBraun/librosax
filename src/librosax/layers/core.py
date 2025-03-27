from dataclasses import field
from typing import Optional

from einops import rearrange
from flax import linen as nn
import jax
from jax import numpy as jnp, random
from jax.scipy.fft import dct
import librosa

from librosax import stft, power_to_db


class DropStripes(nn.Module):
    """A module that randomly drops stripes (time or frequency bands) from a spectrogram.

    This module is used for data augmentation in audio tasks by randomly masking
    time frames or frequency bands in the spectrogram.

    Attributes:
        axis: Axis along which to drop stripes. 2 for time, 3 for frequency.
        drop_width: Maximum width of stripes to drop.
        stripes_num: Number of stripes to drop.
        deterministic: If ``True``, no dropping is performed. Default is ``None``.
    """

    axis: int
    drop_width: int
    stripes_num: int
    deterministic: Optional[bool] = None

    def __post_init__(self):
        assert self.axis in [2, 3]  # axis 2: time; axis 3: frequency
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic=None) -> jnp.ndarray:
        """Apply random stripe dropping to the input spectrogram.

        Args:
            inputs: Input tensor of shape ``(batch_size, channels, time_steps, freq_bins)``.
            deterministic: If ``True``, no dropping is performed. Overrides the module attribute
                if provided. Default is ``None``.

        Returns:
            jnp.ndarray: Transformed tensor with same shape as input.

        Raises:
            AssertionError: If the input does not have 4 dimensions.
        """
        assert inputs.ndim == 4

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        if deterministic:
            return inputs

        # Get shape information
        batch_size = inputs.shape[0]
        total_width = inputs.shape[self.axis]

        key = self.make_rng("dropout")

        # Create a copy of the input that we'll modify
        outputs = inputs

        for n in range(batch_size):
            key, subkey = random.split(key)
            outputs = outputs.at[n].set(
                self._transform_slice(outputs[n], total_width, subkey)
            )

        return outputs

    def _transform_slice(self, e: jnp.ndarray, total_width: int, key: jax.Array):
        """Transform a single slice by dropping random stripes.

        Args:
            e: Single slice tensor of shape ``(channels, time_steps, freq_bins)``.
            total_width: Total width of the dimension along which to drop stripes.
            key: JAX PRNG key.

        Returns:
            jnp.ndarray: Transformed slice with random stripes set to zero.
        """
        for _ in range(self.stripes_num):
            # Split the key for random operations
            key, distance_key, bgn_key = random.split(key, 3)

            # Generate random width and beginning position
            distance = random.randint(
                distance_key, shape=(), minval=0, maxval=self.drop_width
            )
            bgn = random.randint(
                bgn_key, shape=(), minval=0, maxval=total_width - distance
            )

            # Apply mask based on dimension
            if self.axis == 2:
                e = e.at[:, bgn : bgn + distance, :].set(0)
            elif self.axis == 3:
                e = e.at[:, :, bgn : bgn + distance].set(0)

        return e


class SpecAugmentation(nn.Module):
    """A module that applies SpecAugment data augmentation to spectrograms.

    SpecAugment is a data augmentation technique that applies both time
    and frequency masking to spectrograms for audio tasks.

    Attributes:
        time_drop_width: Maximum width of time stripes to drop.
        time_stripes_num: Number of time stripes to drop.
        freq_drop_width: Maximum width of frequency stripes to drop.
        freq_stripes_num: Number of frequency stripes to drop.
        deterministic: If ``True``, no augmentation is applied. Default is ``None``.
    """

    time_drop_width: int
    time_stripes_num: int
    freq_drop_width: int
    freq_stripes_num: int
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic=None) -> jnp.ndarray:
        """Apply SpecAugment to the input spectrogram.

        Args:
            x: Input tensor of shape ``(batch_size, channels, time_steps, freq_bins)``
               or ``(batch_size, time_steps, freq_bins)``.
            deterministic: If ``True``, no augmentation is applied. Overrides the module
                attribute if provided. Default is ``None``.

        Returns:
            jnp.ndarray: Augmented tensor with same shape as input.
        """
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        did_expand = x.ndim == 3
        if did_expand:
            x = jnp.expand_dims(x, axis=1)

        # Apply time dropping
        x = DropStripes(
            axis=2,
            drop_width=self.time_drop_width,
            stripes_num=self.time_stripes_num,
            deterministic=deterministic,
            name="time_dropper",
        )(x)

        # Apply frequency dropping
        x = DropStripes(
            axis=3,
            drop_width=self.freq_drop_width,
            stripes_num=self.freq_stripes_num,
            deterministic=deterministic,
            name="freq_dropper",
        )(x)

        if did_expand:
            x = jnp.squeeze(x, axis=1)

        return x


class Spectrogram(nn.Module):
    """A module that computes a spectrogram from a waveform using JAX.

    This module transforms audio time-domain signals into time-frequency representation.

    Attributes:
        n_fft: FFT size. Default is 2048.
        hop_length: Step between successive frames. Default is ``n_fft // 4``.
        win_length: Window size. Default is ``n_fft``.
        window: Window function type. Default is ``"hann"``.
        center: If ``True``, the waveform is padded so that frames are centered. Default is ``True``.
        pad_mode: Padding mode for the waveform. Default is ``"reflect"``.
        power: Exponent for the magnitude (2.0 means power spectrogram). Default is 2.0.
        freeze_parameters: If ``True``, parameters are not updated during training. Default is ``True``.
    """

    n_fft: int = 2048
    hop_length: int = None
    win_length: int = None
    window: str = "hann"
    center: Optional[bool] = field(default=True)
    pad_mode: str = "reflect"
    power: float = 2.0
    freeze_parameters: Optional[bool] = field(default=True)

    def __post_init__(self):
        assert (
            self.freeze_parameters
        ), "This has only been tested with `freeze_parameters==True`."

        if self.hop_length is None:
            self.hop_length = self.n_fft // 4
        super().__post_init__()

    @nn.compact
    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Compute a spectrogram from a signal using JAX.

        Args:
            waveform: A waveform whose last axis is time.
                The waveform can be 1D ``(T,)``, 2D ``(B, T)``, or 3D ``(B, C, T)``.

        Returns:
            jnp.ndarray: Spectrogram with appropriate shape based on input dimensions.
                - For 1D input: shape ``(time_steps, freq_bins)``
                - For 2D input: shape ``(batch_size, time_steps, freq_bins)``
                - For 3D input: shape ``(batch_size, channels, time_steps, freq_bins)``

        Raises:
            ValueError: If the waveform has invalid dimensions.
        """
        Zxx = stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
        )

        # Compute squared magnitude (or adjust if power != 2).
        S = jnp.abs(Zxx)
        if self.power != 2.0:
            S = jnp.pow(S, self.power)
        else:
            S = jnp.square(S)

        if S.ndim == 4:
            S = rearrange(S, "B C F T -> B C T F")
        elif S.ndim == 3:
            S = rearrange(S, "B F T -> B T F")
        elif S.ndim == 2:
            S = S.T
        else:
            raise ValueError(f"Invalid shape for power spectrogram.")
        return S


class LogmelFilterBank(nn.Module):
    """A module that converts spectrograms to (log) mel spectrograms.

    This module applies mel filterbank on spectrogram and optionally converts
    the result to log scale.

    Attributes:
        sr: Sample rate of the audio signal. Default is 22_050.
        n_fft: FFT size. Default is 2048.
        n_mels: Number of mel filterbanks. Default is 64.
        fmin: Minimum frequency for mel filterbank. Default is 0.0.
        fmax: Maximum frequency for mel filterbank. Default is ``sr // 2``.
        is_log: If ``True``, convert to log scale. Default is ``True``.
        ref: Reference value for log scaling. Default is 1.0.
        amin: Minimum value for log scaling. Default is 1e-10.
        top_db: Maximum dynamic range in dB. Default is 80.0.
        freeze_parameters: If ``True``, parameters are not updated during training. Default is ``True``.
    """

    sr: int = 22_050
    n_fft: int = 2048
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float = None
    is_log: Optional[bool] = field(default=True)
    ref: float = 1.0
    amin: float = 1e-10
    top_db: float | None = 80.0
    freeze_parameters: Optional[bool] = field(default=True)

    def __post_init__(self):
        if self.fmax is None:
            self.fmax = self.sr // 2
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate (log) mel spectrogram from spectrogram.

        Args:
            x: Spectrogram of shape ``(*, frames, n_fft//2+1)``.

        Returns:
            jnp.ndarray: (Log) mel spectrogram of shape ``(*, frames, mel_bins)``.
        """
        melW = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        ).T  # (n_fft // 2 + 1, mel_bins)

        if not self.freeze_parameters:
            melW = self.param("melW", lambda key: melW)

        # Mel spectrogram
        mel_spectrogram = jnp.matmul(x, melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = power_to_db(
                mel_spectrogram, ref=self.ref, amin=self.amin, top_db=self.top_db
            )
        else:
            output = mel_spectrogram

        return output


class MFCC(LogmelFilterBank):
    """A module that computes Mel-Frequency Cepstral Coefficients (MFCCs).

    This module extends LogmelFilterBank to compute MFCCs by applying
    a Discrete Cosine Transform (DCT) to the log-mel spectrogram.

    Attributes:
        n_mfcc: Number of MFCCs to return. Default is 20.
        dct_type: Type of DCT (1-4). Default is 2.
        norm: Normalization mode for DCT. Default is "ortho".
        lifter: Liftering coefficient. 0 means no liftering. Default is 0.
        is_log: If ``True``, convert to log scale (must be ``True`` for MFCCs). Default is ``True``.

        Inherits all attributes from LogmelFilterBank.
    """

    n_mfcc: int = 20  # Number of MFCCs to return
    dct_type: int = 2  # DCT type (2 is the most common for MFCCs)
    norm: str = "ortho"  # Normalization mode for DCT
    lifter: int = 0  # Apply liftering; 0 = no lifter
    is_log: Optional[bool] = field(default=True)

    def __post_init__(self):
        assert self.is_log, "MFCC requires log-mel spectrograms (is_log must be True)"
        # Validate DCT type (jax.scipy.fft.dct supports types 1-4)
        assert (
            1 <= self.dct_type <= 4
        ), f"DCT type must be 1, 2, 3, or 4, got {self.dct_type}"
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute MFCCs from a spectrogram.

        Args:
            x: Spectrogram of shape ``(*, frames, n_fft//2+1)``.

        Returns:
            jnp.ndarray: MFCCs with appropriate shape based on input dimensions.
                - For 2D input: shape ``(n_mfcc, time_steps)``
                - For 3D input: shape ``(batch_size, n_mfcc, time_steps)``

        Raises:
            ValueError: If the input has invalid dimensions.
        """
        # Get log-mel spectrogram from parent class
        mel_spec = super().__call__(x)

        mfccs = dct(mel_spec, type=self.dct_type, norm=self.norm)
        mfccs = mfccs[..., : self.n_mfcc]

        if mel_spec.ndim == 3:
            mfccs = rearrange(mfccs, "B T n_mfcc -> B n_mfcc T")
        elif mel_spec.ndim == 2:
            mfccs = mfccs.T
        else:
            raise ValueError(f"Unsupported log-mel spectrogram shape: {mel_spec.shape}")

        # Apply liftering if requested
        if self.lifter > 0:
            mfccs = self._lifter(mfccs, self.lifter)

        return mfccs

    @staticmethod
    def _lifter(mfccs: jnp.ndarray, lifter: float = 0):
        """Apply liftering to MFCCs to emphasize higher-order coefficients.

        Args:
            mfccs: MFCC features.
            lifter: Liftering coefficient. 0 means no liftering. Default is 0.

        Returns:
            jnp.ndarray: Liftered MFCCs with the same shape as the input.

        Raises:
            ValueError: If the input has invalid dimensions.
        """
        if lifter <= 0:
            return mfccs

        # Get the shape of MFCCs for proper broadcasting
        if mfccs.ndim == 3:
            # Batched input
            n_coeffs = mfccs.shape[-2]
            lift = 1 + (lifter / 2) * jnp.sin(
                jnp.pi * jnp.arange(1, 1 + n_coeffs) / lifter
            )
            return mfccs * jnp.expand_dims(lift, (0, -1))
        elif mfccs.ndim == 2:
            # Non-batched input
            n_coeffs = mfccs.shape[-1]
            lift = 1 + (lifter / 2) * jnp.sin(
                jnp.pi * jnp.arange(1, 1 + n_coeffs) / lifter
            )
            return mfccs * jnp.expand_dims(lift, 0)
        else:
            raise ValueError(f"Unsupported MFCC shape: {mfccs.shape}")
