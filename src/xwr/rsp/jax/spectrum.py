"""Calibrated Spectrum Processing."""

from collections.abc import Sequence
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
        thresholds = cfar(jnp.abs(range_doppler))
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

    def _cfar(
        self, x: Float[Array, "range doppler"]
    ) -> Float[Array, "range doppler"]:
        """Get CFAR scores for a single range-doppler image."""
        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support.
        valid = convolve2d(jnp.ones_like(x), self.mask, mode="same")
        mu = convolve2d(x, self.mask, mode="same") / valid
        second_moment = convolve2d(x**2, self.mask, mode="same") / valid
        sigma = jnp.sqrt(second_moment - mu**2)

        return (x - mu) / sigma

    def __call__(
        self, signal_cube: Float[Array, "batch doppler tx rx range"]
    ) -> Float[Array, "batch range doppler"]:
        """Get CFAR scores.

        Args:
            signal_cube: batch of post range doppler FFT radar cubes in
                amplitude.

        Returns:
            CFAR z-scores, i.e. the number of standard deviations each cell
                lies above its local mean. This is not a threshold level and
                not a boolean mask; see the note above on applying a
                threshold.
        """
        b, d, _, _, r = signal_cube.shape
        # Combine along the antenna array, and reorder to (range, doppler).
        range_dopp = signal_cube.transpose(0, 4, 1, 2, 3).reshape(b, r, d, -1)
        x = jnp.mean(range_dopp, axis=-1)

        return jax.vmap(self._cfar)(x)


class CFARCASO:
    """Cell-averaging Smallest of CFAR.

    Expects a batch of range-doppler cubes, with the `train_window` and
    `guard_window` sizes corresponding to the (range, doppler) axes.

    !!! info

        Instead of the 2D kernel used in Cell-averaging CFAR, CASO uses
        a separate 1D kernel for the range and doppler axes; detection occurs
        if the SNR exceeds the specified threshold on either axis.

    ```
               ┌─┐       ▲ window[0]
               │ │       │
               ├─┤       ▼
         ┌───┬─┼─┼─┬───┐
         └───┴─┼─┼─┴───┘ ▲ guard[0]
               ├─┤       ▼
               │ │
               └─┘
    guard[1] ◄─►   ◄───► window[1]
    ```

    Args:
        train_window: number of training cells on **each** side of the cell
            under test, for (range, doppler). Rather than averaging both
            sides, CASO ("smallest of") takes the **minimum** of the two
            one-sided means, so a strong target on one side cannot inflate
            the noise floor and mask a weaker target on the other.
        guard_window: number of guard cells on **each** side of the cell
            under test, for (range, doppler); these sit between the cell
            under test and the training cells and are excluded from the noise
            estimate. They absorb target energy spread by FFT sidelobes and
            windowing, which would otherwise raise the target's own noise
            floor. The asymmetric default reflects that range leakage is
            broad while doppler needs no guard.
        snr_thresh: signal to noise ratio threshold for (range, doppler), as
            a **linear power ratio** (not dB). A cell is detected on an axis
            when its integrated power exceeds `snr_thresh * noise`, and a
            detection is reported only where **both** axes fire. Raising it
            gives fewer false alarms and a lower probability of detection.
        discard_range: range bins (close, far) to discard around DC. The
            closest bins are dominated by TX to RX leakage and DC, and the
            furthest are past useful SNR; discarded bins are forced to
            non-detect and assigned unit noise.
    """

    def __init__(
        self,
        train_window: Sequence[int] = (8, 4),
        guard_window: Sequence[int] = (8, 0),
        snr_thresh: Sequence[float] = (5.0, 3.0),
        discard_range: Sequence[int] = (10, 20),
    ):
        if len(train_window) != 2:
            raise ValueError(f"Train window {train_window} must be length 2.")
        if len(guard_window) != 2:
            raise ValueError(f"Guard window {guard_window} must be length 2.")
        if len(discard_range) != 2:
            raise ValueError(
                f"Discard range {discard_range} must be length 2.")
        if len(snr_thresh) != 2:
            raise ValueError(f"SNR thresh {snr_thresh} must be length 2.")

        # discard detect object around DC
        self.discard_r = discard_range
        self.snr_r, self.snr_d = snr_thresh

        self.pad_r = train_window[0] + guard_window[0]
        self.pad_d = train_window[1] + guard_window[1]

        # caso
        def make_caso_kernels(train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:train], ker_b[-train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            return jnp.asarray(ker_a), jnp.asarray(ker_b)

        self.r_ker_a, self.r_ker_b = make_caso_kernels(
            train_window[0], self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(
            train_window[1], self.pad_d)

    @staticmethod
    def _caso(
        signal: Float[Array, "n"],
        ker_a: Float[Array, "w"],
        ker_b: Float[Array, "w"],
        snr: float,
        pad: int,
    ) -> tuple[Bool[Array, "m"], Float[Array, "m"]]:
        """Run 1D CFAR CASO, returning a detection mask and noise level."""
        cor_a = jnp.correlate(signal, ker_a, mode="valid")
        cor_b = jnp.correlate(signal, ker_b, mode="valid")
        noise = jnp.minimum(cor_a, cor_b)
        detect = signal[pad:-pad] > snr * noise
        return detect, noise

    def _cfar(
        self, signal_cube: Float[Array, "doppler tx rx range"]
    ) -> tuple[
        Bool[Array, "range doppler"],
        Float[Array, "range doppler"],
        Float[Array, "range doppler"],
    ]:
        """Run 2D CFAR CASO on a single radar cube."""
        signal_cube = signal_cube.transpose(3, 0, 1, 2)
        s_r, s_d, _, _ = signal_cube.shape
        range_dopp = signal_cube.reshape(s_r, s_d, -1)

        # non-coherent signal combination along the antenna array
        signal = jnp.sum(range_dopp**2, axis=-1) + 1
        sig_discard = signal[self.discard_r[0] : -self.discard_r[1]]
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
            self._caso, in_axes=(1, None, None, None, None)
        )(sig_pad_r, self.r_ker_a, self.r_ker_b, self.snr_r, self.pad_r)
        detect_r, noise = detect_r.swapaxes(0, 1), noise.swapaxes(0, 1)
        detect_r = jnp.pad(
            detect_r, ((self.discard_r[0], self.discard_r[1]), (0, 0))
        )
        noise = jnp.pad(
            noise,
            ((self.discard_r[0], self.discard_r[1]), (0, 0)),
            constant_values=1,
        )
        detect_d, _ = jax.vmap(self._caso, in_axes=(0, None, None, None, None))(
            sig_pad_d, self.d_ker_a, self.d_ker_b, self.snr_d, self.pad_d
        )

        snr = signal / noise
        obj_mask = jnp.logical_and(detect_r, detect_d)

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[Array, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
    ]:
        """Run 2D CFAR CASO.

        !!! note

            The transmit and receive antenna axes are combined
            non-coherently, so their relative order does not matter.

        Args:
            signal_cube: batch of post range doppler FFT radar cubes in
                amplitude.

        Returns:
            cfar detected object mask.
            non-coherently integrated power across the virtual array, i.e.
                the range-doppler spectrum used for detection.
            signal to noise ratio, as a linear power ratio.
        """
        return jax.vmap(self._cfar)(signal_cube)


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
