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

    Expects a batch of range-doppler cubes, with the `guard` and `window`
    sizes corresponding to the (range, doppler) axes.

    ```
        ┌─────────────────┐ ▲ window[0]
        │    ┌───────┐    │ │
        │    │  ┌─┐  │    │ ▼
        │    │  └─┘  │    │ ▲ guard[0]
        │    └───────┘    │ ▼
        └─────────────────┘
    guard[1] ◄──► ◄───────► window[1]
    ```

    !!! info

        The noise floor is the mean of the training cells in the 2D ring,
        which tests the cell under test once; contrast
        [`CFARCASO`][xwr.rsp.jax.CFARCASO], which tests the range and doppler
        axes separately and requires **both** to fire.

    Args:
        guard: size of guard cells (excluded from noise estimation).
        window: total CFAR window size.
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
        window: tuple[int, int] = (4, 4),
        snr_thresh: float = 5.0,
        discard_range: Sequence[int] = (10, 20),
    ) -> None:
        if len(discard_range) != 2:
            raise ValueError(
                f"Discard range {discard_range} must be length 2.")

        self.snr_thresh = snr_thresh
        # discard detect object around DC
        self.discard_r = discard_range

        w0, w1 = window
        g0, g1 = guard
        if g0 > w0 or g1 > w1:
            raise ValueError(
                f"Guard {guard} must be <= window {window} on each axis.")

        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0 : w0 + g0 + 1, w1 - g1 : w1 + g1 + 1] = 0.0
        if mask.sum() == 0:
            raise ValueError(
                f"CFAR mask is empty; check guard={guard} and window={window}.")
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
        if any(t < 1 for t in train_window):
            raise ValueError(
                f"Train window {train_window} must be >= 1 on each axis.")

        # discard detect object around DC
        self.discard_r = discard_range
        self.snr_r, self.snr_d = snr_thresh

        self.train_r, self.train_d = train_window
        self.pad_r = train_window[0] + guard_window[0]
        self.pad_d = train_window[1] + guard_window[1]

    @staticmethod
    def _caso(
        signal: Float[Array, "..."],
        axis: int,
        train: int,
        pad: int,
        snr: float,
    ) -> tuple[Bool[Array, "..."], Float[Array, "..."]]:
        """Run 1D CFAR CASO along `axis` of an arbitrarily batched array.

        The training cells are a contiguous box on each side of the cell under
        test, so the leading and trailing one-sided means are accumulated
        directly from shifted slices rather than correlated against a mostly
        zero kernel. `train` is a static Python int, so the sum unrolls at
        trace time.

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
        # non-coherent signal combination along the antenna array
        signal = jnp.sum(signal_cube**2, axis=(2, 3)).transpose(0, 2, 1) + 1
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
