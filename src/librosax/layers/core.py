from einops import rearrange
from flax import nnx
from flax.nnx import Module
from flax.nnx.module import first_from
from flax.nnx import rnglib
import jax
from jax import numpy as jnp, random
from jax.scipy.fft import dct
import librosa

from librosax import stft, power_to_db


class DropStripes(Module):
    """A module that randomly drops stripes (time or frequency bands) from a spectrogram.

    This module is used for data augmentation in audio tasks by randomly masking
    contiguous blocks along either the time or frequency dimension. Each stripe
    is a rectangular mask that spans the entire height (for time masking) or 
    width (for frequency masking) of the spectrogram.

    Attributes:
        axis: Axis along which to drop stripes. 
            - ``axis=2``: Drop vertical stripes (time masking) - masks entire frequency range for selected time frames
            - ``axis=3``: Drop horizontal stripes (frequency masking) - masks entire time range for selected frequency bins
        drop_width: Maximum width of each stripe to drop. Each stripe will have a 
            random width between 0 and this value. For time masking, this is the
            maximum number of consecutive time frames to mask. For frequency masking,
            this is the maximum number of consecutive frequency bins to mask.
        stripes_num: Number of stripes to drop. Each stripe is independently
            positioned and sized. Multiple stripes may overlap.
        deterministic: If ``True``, no dropping is performed. Default is ``False``.
        rng_collection: The rng collection name to use when requesting an rng key.
            Default is ``"dropout"``.
        rngs: Random number generator key for generating random masks.
    
    Implementation Details:
        - Each stripe has a random width sampled from [0, drop_width)
        - Each stripe has a random starting position that ensures it fits within bounds
        - Stripes are applied multiplicatively (regions set to 0)
        - Each item in a batch receives different random stripes
        - Overlapping stripes do not create additional masking effect
    """

    def __init__(
        self,
        axis: int,
        drop_width: int,
        stripes_num: int,
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ):
        self.axis = axis
        self.drop_width = drop_width
        self.stripes_num = stripes_num
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = nnx.data(None) if rngs is None else rngs.fork()

        if self.axis < 0:
            self.axis += 4
        assert self.axis in [2, 3]  # axis 2: time; axis 3: frequency

    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> jnp.ndarray:
        """Apply random stripe dropping to the input spectrogram.

        Args:
            inputs: Input tensor of shape ``(batch_size, channels, time_steps, freq_bins)``.
            deterministic: If ``True``, no dropping is performed. Overrides the module attribute
                if provided. Default is ``None``.
            rngs: an optional key, RngStream, or Rngs object used to generate the dropout mask.
                If given it will take precedence over the rngs passed into the constructor.

        Returns:
            jnp.ndarray: Transformed tensor with same shape as input.

        Raises:
            AssertionError: If the input does not have 4 dimensions.
        """
        assert inputs.ndim == 4

        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropStripes as either a __call__ argument or class attribute""",
        )

        if deterministic:
            return inputs

        rngs = first_from(  # type: ignore[assignment]
            rngs,
            self.rngs,
            error_msg="""`deterministic` is False, but no `rngs` argument was provided to DropStripes
              as either a __call__ argument or class attribute.""",
        )

        if isinstance(rngs, rnglib.Rngs):
            key = rngs[self.rng_collection]()
        elif isinstance(rngs, rnglib.RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or jax.Array, but got {type(rngs)}."
            )

        # Get shape information
        batch_size = inputs.shape[0]
        total_width = inputs.shape[self.axis]

        # Create a separate key for each element in the batch
        batch_keys = random.split(key, batch_size)

        # Vectorize the _transform_slice function over the batch dimension
        transformed = jax.vmap(lambda x, k: self._transform_slice(x, total_width, k))(
            inputs, batch_keys
        )

        return transformed

    def _transform_slice(self, e: jnp.ndarray, total_width: int, key: jax.Array):
        """Transform a single slice by dropping random stripes.

        Args:
            e: Single slice tensor of shape ``(channels, time_steps, freq_bins)``.
            total_width: Total width of the dimension along which to drop stripes.
            key: JAX PRNG key.

        Returns:
            jnp.ndarray: Transformed slice with random stripes set to zero.
        """
        # Create a mask of ones with the same shape as e
        mask = jnp.ones_like(e)

        # Define the body function for fori_loop
        def body_fn(i, carry):
            mask, key = carry
            key, distance_key, bgn_key = random.split(key, 3)

            # Generate random width and beginning position
            distance = random.randint(
                distance_key, shape=(), minval=0, maxval=self.drop_width
            )
            bgn = random.randint(
                bgn_key, shape=(), minval=0, maxval=total_width - distance
            )

            # Create and apply the stripe mask based on axis
            if self.axis == 2:
                # Create a mask for the time dimension
                time_indices = jnp.arange(e.shape[1])
                stripe_mask = (time_indices >= bgn) & (time_indices < bgn + distance)
                stripe_mask = jnp.reshape(stripe_mask, (1, -1, 1))
                mask = mask * (1.0 - stripe_mask.astype(mask.dtype))
            elif self.axis == 3:
                # Create a mask for the frequency dimension
                freq_indices = jnp.arange(e.shape[2])
                stripe_mask = (freq_indices >= bgn) & (freq_indices < bgn + distance)
                stripe_mask = jnp.reshape(stripe_mask, (1, 1, -1))
                mask = mask * (1.0 - stripe_mask.astype(mask.dtype))

            return (mask, key)

        # Use jax.lax.fori_loop to iterate through the stripes
        (mask, _) = jax.lax.fori_loop(0, self.stripes_num, body_fn, (mask, key))

        # Apply the mask to the input
        return e * mask


class SpecAugmentation(Module):
    """A module that applies SpecAugment data augmentation to spectrograms.

    SpecAugment is a data augmentation technique that applies both time
    and frequency masking to spectrograms for audio tasks. It randomly masks
    rectangular blocks along the time and frequency dimensions by setting
    values to zero.

    Attributes:
        time_drop_width: Maximum width (in time frames) of each time mask. 
            Each mask will have a random width between 0 and this value.
            For example, if ``time_drop_width=30`` and your spectrogram has 
            200 time frames, each mask can be 0-30 frames wide.
        time_stripes_num: Number of time masks to apply. Each mask is 
            independently positioned and sized.
        freq_drop_width: Maximum width (in frequency bins) of each frequency mask.
            Each mask will have a random width between 0 and this value.
            For example, if ``freq_drop_width=20`` and your spectrogram has 
            128 frequency bins, each mask can be 0-20 bins wide.
        freq_stripes_num: Number of frequency masks to apply. Each mask is
            independently positioned and sized.
        deterministic: If ``True``, no augmentation is applied. Default is ``False``.
        rng_collection: The rng collection name to use when requesting an rng key.
            Default is ``"dropout"``.
        rngs: Random number generator key for generating random masks.
    
    Example:
        >>> import jax
        >>> from flax import nnx
        >>> from librosax.layers import SpecAugmentation
        >>> 
        >>> # Create augmentation layer
        >>> spec_aug = SpecAugmentation(
        ...     time_drop_width=64,  # Max 64 time frames per mask
        ...     time_stripes_num=2,   # Apply 2 time masks
        ...     freq_drop_width=16,   # Max 16 freq bins per mask  
        ...     freq_stripes_num=2,   # Apply 2 frequency masks
        ...     rngs=nnx.Rngs(jax.random.key(0))
        ... )
        >>> 
        >>> # Apply to spectrogram (batch_size, channels, time, freq)
        >>> spec = jnp.ones((4, 1, 200, 128))
        >>> augmented = spec_aug(spec, deterministic=False)
    
    Note:
        The masks are applied multiplicatively (by setting regions to 0), so
        overlapping masks will not create additional effect. Each batch item
        receives different random masks for better augmentation diversity.
    """

    def __init__(
        self,
        time_drop_width: int,
        time_stripes_num: int,
        freq_drop_width: int,
        freq_stripes_num: int,
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ):
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num
        self.deterministic = deterministic
        self.rng_collection = rng_collection
        self.rngs = nnx.data(None) if rngs is None else rngs.fork()

        self.time_drop = DropStripes(
            axis=2,
            drop_width=time_drop_width,
            stripes_num=time_stripes_num,
            deterministic=deterministic,
            rng_collection=rng_collection,
            rngs=rngs,
        )
        self.freq_drop = DropStripes(
            axis=3,
            drop_width=freq_drop_width,
            stripes_num=freq_stripes_num,
            deterministic=deterministic,
            rng_collection=rng_collection,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """Apply SpecAugment to the input spectrogram.

        Args:
            x: Input tensor of shape ``(batch_size, channels, time_steps, freq_bins)``
               or ``(batch_size, time_steps, freq_bins)``.
            deterministic: If ``True``, no augmentation is applied. Overrides the module
                attribute if provided. Default is ``None``.

        Returns:
            jnp.ndarray: Augmented tensor with same shape as input.
        """

        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropStripes as either a __call__ argument or class attribute""",
        )

        if deterministic:
            return x

        did_expand = x.ndim == 3
        if did_expand:
            x = jnp.expand_dims(x, axis=1)

        x = self.time_drop(x)
        x = self.freq_drop(x)

        if did_expand:
            x = jnp.squeeze(x, axis=1)

        return x


class Spectrogram(Module):
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

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = None,
        win_length: int = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        freeze_parameters: bool = True,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.freeze_parameters = freeze_parameters

        assert (
            self.freeze_parameters
        ), "Spectrogram has only been tested with `freeze_parameters==True`."

        if self.hop_length is None:
            self.hop_length = self.n_fft // 4

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Compute a spectrogram from a signal using JAX.

        Note:
            Unlike ``librosax.stft``, this layer outputs with time before frequency
            (suitable for neural network processing).

        Args:
            waveform: Audio waveform. The last axis must be time.

                - ``(T,)`` - single waveform
                - ``(B, T)`` - batch of waveforms
                - ``(B, C, T)`` - batch with channels

        Returns:
            Power spectrogram. Note the axis order is ``(time, freq)``:

            - ``(T,)`` → ``(N, F)``
            - ``(B, T)`` → ``(B, N, F)``
            - ``(B, C, T)`` → ``(B, C, N, F)``

            where F = ``n_fft // 2 + 1`` and N = number of frames.

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


class LogMelFilterBank(Module):
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

    def __init__(
        self,
        sr: int = 22_050,
        n_fft: int = 2048,
        n_mels: int = 64,
        fmin: float = 0.0,
        fmax: float = None,
        is_log: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float | None = 80.0,
        freeze_parameters: bool = True,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.freeze_parameters = freeze_parameters

        if self.fmax is None:
            self.fmax = self.sr // 2

        if not self.freeze_parameters:
            melW = librosa.filters.mel(
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            ).T  # (n_fft // 2 + 1, mel_bins)

            self.melW = nnx.Param(melW)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate (log) mel spectrogram from spectrogram.

        Args:
            x: Power spectrogram with shape ``(..., N, F)`` where N is frames
                and F = ``n_fft // 2 + 1``. Typically output from ``Spectrogram``.

                - ``(N, F)`` - single spectrogram
                - ``(B, N, F)`` - batch of spectrograms
                - ``(B, C, N, F)`` - batch with channels

        Returns:
            (Log) mel spectrogram with shape ``(..., N, n_mels)``.

            - ``(N, F)`` → ``(N, n_mels)``
            - ``(B, N, F)`` → ``(B, N, n_mels)``
            - ``(B, C, N, F)`` → ``(B, C, N, n_mels)``
        """
        if self.freeze_parameters:
            melW = librosa.filters.mel(
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            ).T  # (n_fft // 2 + 1, mel_bins)
        else:
            melW = self.melW

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


class MFCC(LogMelFilterBank):
    """A module that computes Mel-Frequency Cepstral Coefficients (MFCCs).

    This module extends LogMelFilterBank to compute MFCCs by applying
    a Discrete Cosine Transform (DCT) to the log-mel spectrogram.

    Attributes:
        n_mfcc: Number of MFCCs to return. Default is 20.
        dct_type: Type of DCT (1-4). Default is 2.
        norm: Normalization mode for DCT. Default is "ortho".
        lifter: Liftering coefficient. 0 means no liftering. Default is 0.
        is_log: If ``True``, convert to log scale (must be ``True`` for MFCCs). Default is ``True``.

        Inherits all attributes from LogMelFilterBank.
    """

    def __init__(
        self,
        sr: int = 22_050,
        n_fft: int = 2048,
        n_mels: int = 64,
        fmin: float = 0.0,
        fmax: float = None,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float | None = 80.0,
        freeze_parameters: bool = True,
        n_mfcc: int = 20,  # Number of MFCCs to return
        dct_type: int = 2,  # DCT type (2 is the most common for MFCCs)
        norm: str = "ortho",  # Normalization mode for DCT
        lifter: int = 0,  # Apply liftering; 0 = no lifter
    ):
        super().__init__(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            is_log=True,  # MFCC requires log-mel spectrograms
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=freeze_parameters,
        )
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter

        # Validate DCT type (jax.scipy.fft.dct supports types 1-4)
        assert (
            1 <= self.dct_type <= 4
        ), f"DCT type must be 1, 2, 3, or 4, got {self.dct_type}"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute MFCCs from a spectrogram.

        Args:
            x: Power spectrogram with shape ``(..., N, F)`` where N is frames
                and F = ``n_fft // 2 + 1``. Typically output from ``Spectrogram``.

                - ``(N, F)`` - single spectrogram
                - ``(B, N, F)`` - batch of spectrograms
                - ``(B, C, N, F)`` - batch with channels

        Returns:
            MFCCs with shape ``(..., n_mfcc, N)``. Note the output transposes
            the last two axes compared to the input.

            - ``(N, F)`` → ``(n_mfcc, N)``
            - ``(B, N, F)`` → ``(B, n_mfcc, N)``
            - ``(B, C, N, F)`` → ``(B, C, n_mfcc, N)``

        Raises:
            ValueError: If the input has invalid dimensions.
        """
        # Get log-mel spectrogram from parent class
        mel_spec = super().__call__(x)

        mfccs = dct(mel_spec, type=self.dct_type, norm=self.norm)
        mfccs = mfccs[..., : self.n_mfcc]

        # (..., T, n_mfcc) -> (..., n_mfcc, T)
        mfccs = rearrange(mfccs, "... T n_mfcc -> ... n_mfcc T")

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

        n_coeffs = mfccs.shape[-2]
        lift = 1 + (lifter / 2) * jnp.sin(jnp.pi * jnp.arange(1, 1 + n_coeffs) / lifter)

        # Get the shape of MFCCs for proper broadcasting
        if mfccs.ndim == 4:
            return mfccs * jnp.expand_dims(lift, (0, 1, -1))
        elif mfccs.ndim == 3:
            return mfccs * jnp.expand_dims(lift, (0, -1))
        elif mfccs.ndim == 2:
            return mfccs * jnp.expand_dims(lift, -1)
        else:
            raise ValueError(f"Unsupported MFCC shape: {mfccs.shape}")
