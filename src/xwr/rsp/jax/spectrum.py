"""Calibrated Spectrum Processing."""

from typing import Generic, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Bool, Complex64, Float, Float32, Int16

from xwr.rsp import iq_from_iiqq

from .rsp import RSPJax

TRSP = TypeVar("TRSP", bound=RSPJax)


class CFAR:
    """Cell-averaging CFAR.

    Expects a 2d input, with the `guard` and `window` sizes corresponding to
    the respective input axes.

    ```
        ┌─────────────────┐ ▲ window[0]
        │    ┌───────┐    │ │
        │    │  ┌─┐  │    │ ▼
        │    │  └─┘  │    │ ▲ guard[0]
        │    └───────┘    │ ▼
        └─────────────────┘
    guard[1] ◄──► ◄───────► window[1]
    ```

    !!! note

        The user is responsible for applying the desired thresholding.
        For example, when using a gaussian model, the threshold should be
        calculated using an inverse normal CDF (e.g. `scipy.stats.norm.isf`):

        ```python
        cfar = CFAR(guard=(2, 2), window=(4, 4))
        thresholds = cfar(image)
        mask = (thresholds > scipy.stats.norm.isf(0.01))
        ```

    Args:
        guard: size of guard cells (excluded from noise estimation).
        window: total CFAR window size.
    """

    def __init__(
        self, guard: tuple[int, int] = (2, 2), window: tuple[int, int] = (4, 4)
    ) -> None:
        w0, w1 = window
        g0, g1 = guard

        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0 : w0 + g0 + 1, w1 - g1 : w1 + g1 + 1] = 0.0
        self.mask: Array = jnp.array(mask)

    def __call__(self, x: Float[Array, "d r ..."]) -> Float[Array, "d r"]:
        """Get CFAR thresholds.

        !!! note

            Boundary cells are zero-padded.

        Args:
            x: input. If more than 2 axes are present, the additional axes
                are averaged before running CFAR.

        Returns:
            CFAR threshold values for this input.
        """
        # Collapse additional axes if required
        if x.ndim > 2:
            x = jnp.mean(x.reshape(x.shape[0], x.shape[1], -1), -1)

        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support.
        valid = convolve2d(jnp.ones_like(x), self.mask, mode="same")
        mu = convolve2d(x, self.mask, mode="same") / valid
        second_moment = convolve2d(x**2, self.mask, mode="same") / valid
        sigma = jnp.sqrt(second_moment - mu**2)

        return (x - mu) / sigma


class CFARCASO:
    """CASO_CFAR: Cell Averaging Smallest of CFAR.

    Args:
        train_range: train window size for range dimension.
        guard_range: guard window size for range dimension.
        train_doppler: train window size for doppler dimension.
        guard_doppler: guard window size for doppler dimension.
        snr_range: signal to noise ratio threshold on range dimension.
        snr_doppler: signal to noise ratio threshold on doppler dimension.
        discard_range_close: discard low frequency noise bin.
        discard_range_far: discard high frequency noise bin.
    """

    def __init__(
        self,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        snr_range: float = 5.0,
        snr_doppler: float = 3.0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
    ):
        self.pad_r = train_range + guard_range
        self.pad_d = train_doppler + guard_doppler

        # discard detect object around DC
        self.discard_close, self.discard_far = (
            discard_range_close,
            discard_range_far,
        )

        # caso
        def make_caso_kernels(train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:train], ker_b[-train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            return jnp.asarray(ker_a), jnp.asarray(ker_b)

        self.r_ker_a, self.r_ker_b = make_caso_kernels(train_range, self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(
            train_doppler, self.pad_d
        )

        self.snr_r, self.snr_d = snr_range, snr_doppler

    def caso(
        self,
        signal: Float32[Array, "n"],
        ker_a: Float32[Array, "w"],
        ker_b: Float32[Array, "w"],
        snr: float,
        pad: int,
    ) -> tuple[Bool[Array, "m"], Float32[Array, "m"]]:
        """1D CFAR CASO.

        Args:
            signal: 1D frequency spectrum.
            ker_a: one side cfar kernel.
            ker_b: one side cfar kernel.
            snr: signal to noise ratio threshold.
            pad: padding number of the input signal.

        Returns:
            detection mask.
            noise level.
        """
        cor_a = jnp.correlate(signal, ker_a, mode="valid")
        cor_b = jnp.correlate(signal, ker_b, mode="valid")
        noise = jnp.minimum(cor_a, cor_b)
        detect = signal[pad:-pad] > snr * noise
        return detect, noise

    def __call__(
        self, signal_cube: Float32[Array, "doppler Rx Tx range"]
    ) -> tuple[
        Bool[Array, "range doppler"],
        Float32[Array, "range doppler"],
        Float32[Array, "range doppler"],
    ]:
        """Run 2D CFAR CASO.

        Args:
            signal_cube: post range doppler FFT radar cube in amplitude.

        Returns:
            cfar detected object mask.
            range doppler spectrum for detection.
            signal to noise ratio.
        """
        signal_cube = signal_cube.transpose(3, 0, 1, 2)
        s_r, s_d, _, _ = signal_cube.shape
        range_dopp = signal_cube.reshape(s_r, s_d, -1)

        # non-coherent signal combination along the antenna array
        signal = jnp.sum(range_dopp**2, axis=-1) + 1
        sig_discard = signal[self.discard_close : -self.discard_far]
        sig_pad_r = jnp.concat(
            (
                sig_discard[: self.pad_r],
                sig_discard,
                sig_discard[-self.pad_r :],
            ),
            axis=0,
        )
        sig_pad_d = jnp.pad(
            signal, ((0, 0), (self.pad_d, self.pad_d)), mode="wrap"
        )

        # detection
        detect_r, noise = jax.vmap(
            self.caso, in_axes=(1, None, None, None, None)
        )(sig_pad_r, self.r_ker_a, self.r_ker_b, self.snr_r, self.pad_r)
        detect_r, noise = detect_r.swapaxes(0, 1), noise.swapaxes(0, 1)
        detect_r = jnp.pad(
            detect_r, ((self.discard_close, self.discard_far), (0, 0))
        )
        noise = jnp.pad(
            noise,
            ((self.discard_close, self.discard_far), (0, 0)),
            constant_values=1,
        )
        detect_d, _ = jax.vmap(self.caso, in_axes=(0, None, None, None, None))(
            sig_pad_d, self.d_ker_a, self.d_ker_b, self.snr_d, self.pad_d
        )

        snr = signal / noise
        obj_mask = jnp.logical_and(detect_r, detect_d)

        return obj_mask, signal, snr


class CalibratedSpectrum(Generic[TRSP]):
    """Radar processing with zero-doppler calibration.

    !!! info "Zero Doppler Calibration"

        Due to the antenna geometry and radar returns from the data collection
        rig which is mounted rigidly to the radar, the radar spectrum has a
        substantial constant offset in the zero-doppler bins.

        - We assume that the range-Doppler plots are sparse, and take the
          median across a number of sample frames for the zero-doppler bin to
          estimate this offset.
        - If a hanning window is applied, we instead calculate the offset
          across doppler bins `[-1, 1]` to account for doppler bleed.
        - This calculated offset is subtracted from the calculated spectrum.

    Args:
        rsp: RSP pipeline to use.
    """

    def __init__(
        self,
        rsp: TRSP,
    ) -> None:
        self.rsp = rsp

    def calibration_patch(
        self,
        sample: Complex64[Array, "n slow tx rx fast"]
        | Int16[Array, "n slow tx rx fast2"],
        batch: int = 1,
    ) -> Float32[Array, "doppler el az range"]:
        """Create a calibration patch for zero-doppler correction.

        Args:
            sample: sample IQ data to use for calibration.
            batch: sample size for RSP processing. Uses batch size `1` by
                default; should evenly divide the number of samples.

        Returns:
            Patch of the doppler-range-azimuth image which should be subracted
                from the zero-doppler bins of the range-doppler-angle spectrum.
        """
        sample = iq_from_iiqq(sample)

        s0 = self.rsp(sample[:batch])
        shape = s0.shape[1:]

        zero = shape[0] // 2
        start, stop = zero, zero + 1
        if "doppler" in self.rsp.window:
            start -= 1
            stop += 1
        self.slice = (slice(None), slice(start, stop))

        @jax.jit
        def _calib(frames) -> Float32[Array, "batch slice az el range"]:
            return jnp.abs(self.rsp(frames))[self.slice]

        batched = sample.reshape(-1, batch, *sample.shape[1:])
        slices = [s0[self.slice]] + [_calib(batch) for batch in batched]
        return jnp.median(jnp.concatenate(slices, axis=0))

    def __call__(
        self,
        iq: Complex64[Array, "#batch doppler tx rx range"]
        | Int16[Array, "#batch doppler tx rx range2"],
        calib: Float32[Array, "doppler el az range"],
    ) -> Float32[Array, "batch doppler el az range"]:
        """Run radar spectrum processing pipeline.

        !!! note

            After subtracting the calibration patch, any negative values are
            clipped to zero.

        Args:
            iq: batch of IQ data to run.
            calib: calibration patch to apply.

        Returns:
            Doppler-elevation-azimuth-range real spectrum, with zero doppler
                correction applied.
        """
        raw = jnp.abs(self.rsp(iq))
        return raw.at[self.slice].set(jnp.maximum(raw[self.slice] - calib, 0.0))
