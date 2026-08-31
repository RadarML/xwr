"""Calibrated Spectrum Processing."""

from typing import Generic, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Bool, Complex64, Float, Float32, Int16

from xwr.rsp import iq_from_iiqq
from xwr.rsp import spectrum as base

from .rsp import RSPJax

TRSP = TypeVar("TRSP", bound=RSPJax)

jax.tree_util.register_dataclass(
    base.Detection, data_fields=["mask", "signal", "snr"], meta_fields=[])
"""Register [`Detection`][xwr.rsp.] as a pytree, so that it can be returned
from a `jax.jit`-ed function."""


class CFAR(base.CFAR[Array]):
    """Cell-averaging CFAR."""

    @staticmethod
    def _asarray(x: Float[np.ndarray, "..."]) -> Float[Array, "..."]:
        return jnp.asarray(x)

    def _noise(
        self, signal: Float[Array, "range doppler"]
    ) -> Float[Array, "range doppler"]:
        """Get the ring-averaged noise floor for a range-doppler image."""
        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support; the training cell count
        # is normalized out to compensate at the edges.
        valid = convolve2d(jnp.ones_like(signal), self.mask, mode="same")
        return convolve2d(signal, self.mask, mode="same") / valid

    def _cfar(
        self, signal_cube: Float[Array, "batch doppler channel range"]
    ) -> base.Detection[Array]:
        # Non-coherent integration over the channel axis, offset by 1 so
        # that an empty cell has unit power instead of dividing by zero.
        signal = jnp.sum(signal_cube**2, axis=2).transpose(0, 2, 1) + 1
        _, s_r, _ = signal.shape

        noise_r = jax.vmap(self._noise)(signal)

        near, far = self.discard_r[0], self.discard_r[1]
        # 1 outside the discarded band, so the reported SNR there is the raw
        # signal rather than a division by zero.
        noise = jnp.ones_like(signal).at[:, near : s_r - far].set(
            noise_r[:, near : s_r - far])

        snr = signal / noise
        obj_mask = jnp.zeros(signal.shape, dtype=bool).at[
            :, near : s_r - far].set(snr[:, near : s_r - far] > self.snr_thresh)

        return base.Detection(obj_mask, signal, snr)


class CFARCASO(base.CFARCASO[Array]):
    """Cell-averaging Smallest of CFAR."""

    @staticmethod
    def _caso(
        signal: Float[Array, "..."],
        axis: int,
        train: int,
        pad: int,
        snr: float,
    ) -> tuple[Bool[Array, "..."], Float[Array, "..."]]:
        """Run 1D CFAR CASO along `axis` of an arbitrarily batched array.

        Implementation notes:

        - The training cells are a contiguous box on each side of the cell
            under test, so the leading and trailing one-sided means are
            accumulated directly from shifted slices, rather than correlated
            against a mostly zero kernel.
        - `train` is a static Python int, so the sum unrolls at trace time.

        Args:
            signal: signal, already padded by `pad` on both ends of `axis`.
            axis: axis to run CFAR along.
            train: number of training cells on each side.
            pad: number of training plus guard cells on each side.
            snr: signal to noise ratio threshold, as a linear power ratio.

        Returns:
            detection mask and noise level, with `axis` trimmed by `2 * pad`
                back to the unpadded length.
        """
        size = signal.shape[axis] - 2 * pad

        def one_sided(start: int) -> Float[Array, "..."]:
            acc = jax.lax.slice_in_dim(signal, start, start + size, axis=axis)
            for i in range(1, train):
                acc = acc + jax.lax.slice_in_dim(
                    signal, start + i, start + i + size, axis=axis)
            return acc / train

        noise = jnp.minimum(one_sided(0), one_sided(2 * pad + 1 - train))
        cut = jax.lax.slice_in_dim(signal, pad, pad + size, axis=axis)
        return cut > snr * noise, noise

    def _cfar(
        self, signal_cube: Float[Array, "batch doppler channel range"]
    ) -> base.Detection[Array]:
        # Non-coherent integration over the channel axis, offset by 1 so
        # that an empty cell has unit power instead of dividing by zero.
        signal = jnp.sum(signal_cube**2, axis=2).transpose(0, 2, 1) + 1
        _, s_r, _ = signal.shape

        near, far = self.discard_r[0], self.discard_r[1]
        sig_discard = signal[:, near : s_r - far]
        sig_pad_r = jnp.concat(
            (
                sig_discard[:, : self.pad_r],
                sig_discard,
                sig_discard[:, -self.pad_r :],
            ),
            axis=1,
        )
        sig_pad_d = jnp.pad(
            signal, ((0, 0), (0, 0), (self.pad_d, self.pad_d)), mode="wrap"
        )

        # detection
        detect_r, noise_r = self._caso(
            sig_pad_r, 1, self.train_r, self.pad_r, self.snr_r)
        detect_r = jnp.pad(detect_r, ((0, 0), (near, far), (0, 0)))
        # 1 outside the discarded band, so the reported SNR there is the raw
        # signal rather than a division by zero.
        noise = jnp.pad(
            noise_r, ((0, 0), (near, far), (0, 0)), constant_values=1)
        detect_d, _ = self._caso(
            sig_pad_d, 2, self.train_d, self.pad_d, self.snr_d)

        snr = signal / noise
        obj_mask = jnp.logical_and(detect_r, detect_d)

        return base.Detection(obj_mask, signal, snr)


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
