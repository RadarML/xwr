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

    Expects a batch of range-doppler cubes, with the `guard` and `train`
    sizes corresponding to the (range, doppler) axes.

    ```
        ┌─────────────────┐ ▲ guard[0]+train[0]
        │    ┌───────┐    │ │
        │    │  ┌─┐  │    │ ▼
        │    │  └─┘  │    │ ▲ guard[0]
        │    └───────┘    │ ▼
        └─────────────────┘
    guard[1] ◄──► ◄───────► guard[1]+train[1]
    ```

    !!! info

        The noise floor is the mean of the training cells in the 2D ring,
        which tests the cell under test once; contrast
        [`CFARCASO`][xwr.rsp.jax.CFARCASO], which tests the range and doppler
        axes separately and requires **both** to fire.

    Args:
        guard: number of guard cells on **each** side of the cell under test,
            for (range, doppler); these sit between the cell under test and
            the training cells and are excluded from the noise estimate.
        train: number of training cells on **each** side of the guard region,
            for (range, doppler); the window half-width on each axis is
            `guard + train`.
        snr_thresh: signal to noise ratio threshold, as a **linear power
            ratio** (not dB). A cell is detected when its integrated power
            exceeds `snr_thresh * noise`. Raising it gives fewer false alarms
            and a lower probability of detection.
        discard_range: range bins (close, far) to discard around DC. The
            closest bins are dominated by TX to RX leakage and DC, and the
            furthest are past useful SNR; discarded bins are forced to
            non-detect and assigned unit noise.
    """

    def __init__(
        self,
        guard: tuple[int, int] = (2, 2),
        train: tuple[int, int] = (2, 2),
        snr_thresh: float = 5.0,
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        self.snr_thresh = snr_thresh
        # discard detect object around DC
        self.discard_r = discard_range

        g0, g1 = guard
        w0, w1 = g0 + train[0], g1 + train[1]

        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0 : w0 + g0 + 1, w1 - g1 : w1 + g1 + 1] = 0.0
        if mask.sum() == 0:
            raise ValueError(
                f"CFAR mask is empty; check guard={guard} and train={train}.")
        self.mask: Array = jnp.array(mask)

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
        self, signal_cube: Float[Array, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
    ]:
        """Run 2D cell-averaging CFAR on a batch of radar cubes."""
        # non-coherent signal combination along the antenna array
        signal = jnp.sum(signal_cube**2, axis=(2, 3)).transpose(0, 2, 1) + 1
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

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[Array, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
        Float[Array, "batch range doppler"],
    ]:
        """Run 2D cell-averaging CFAR.

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
        return self._cfar(signal_cube)


class CFARCASO:
    """Cell-averaging Smallest of CFAR.

    Expects a batch of range-doppler cubes, with the `guard` and
    `train` sizes corresponding to the (range, doppler) axes.

    !!! info

        Instead of the 2D kernel used in Cell-averaging CFAR, CASO uses
        a separate 1D kernel for the range and doppler axes; detection occurs
        if the SNR exceeds the specified threshold on either axis.

    ```
               ┌─┐       ▲ train[0]
               │ │       │
               ├─┤       ▼
         ┌───┬─┼─┼─┬───┐
         └───┴─┼─┼─┴───┘ ▲ guard[0]
               ├─┤       ▼
               │ │
               └─┘
    guard[1] ◄─►   ◄───► train[1]
    ```

    Args:
        guard: number of guard cells on **each** side of the cell
            under test, for (range, doppler); these sit between the cell
            under test and the training cells and are excluded from the noise
            estimate. They absorb target energy spread by FFT sidelobes and
            windowing, which would otherwise raise the target's own noise
            floor. The asymmetric default reflects that range leakage is
            broad while doppler needs no guard.
        train: number of training cells on **each** side of the guard region,
            for (range, doppler). Rather than averaging both sides, CASO
            ("smallest of") takes the **minimum** of the two one-sided means,
            so a strong target on one side cannot inflate the noise floor
            and mask a weaker target on the other.
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
        guard: tuple[int, int] = (8, 0),
        train: tuple[int, int] = (8, 4),
        snr_thresh: tuple[float, float] = (5.0, 3.0),
        discard_range: tuple[int, int] = (10, 20),
    ):
        # discard detect object around DC
        self.discard_r = discard_range
        self.snr_r, self.snr_d = snr_thresh

        self.pad_r = train[0] + guard[0]
        self.pad_d = train[1] + guard[1]

        # caso
        def make_caso_kernels(n_train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:n_train], ker_b[-n_train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            return jnp.asarray(ker_a), jnp.asarray(ker_b)

        self.r_ker_a, self.r_ker_b = make_caso_kernels(
            train[0], self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(
            train[1], self.pad_d)

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
        sig_discard = signal[self.discard_r[0] : s_r - self.discard_r[1]]
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
