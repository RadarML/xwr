"""Calibrated Spectrum Processing."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Float
from scipy.signal import convolve2d, correlate


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
        [`CFARCASO`][xwr.rsp.numpy.CFARCASO], which tests the range and
        doppler axes separately and requires **both** to fire.

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
        self.mask: np.ndarray = mask

    def _noise(
        self, signal: Float[np.ndarray, "range doppler"]
    ) -> Float[np.ndarray, "range doppler"]:
        """Get the ring-averaged noise floor for a range-doppler image."""
        valid = convolve2d(np.ones_like(signal), self.mask, mode="same")
        return convolve2d(signal, self.mask, mode="same") / valid

    def _cfar(
        self, signal_cube: Float[np.ndarray, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
    ]:
        """Run 2D cell-averaging CFAR on a batch of radar cubes."""
        # non-coherent signal combination along the antenna array
        signal = np.sum(signal_cube**2, axis=(2, 3)).transpose(0, 2, 1) + 1
        _, s_r, _ = signal.shape

        noise_r = np.stack([self._noise(s) for s in signal])

        near, far = self.discard_r[0], self.discard_r[1]
        # 1 outside the discarded band, so the reported SNR there is the raw
        # signal rather than a division by zero.
        noise = np.ones_like(signal)
        noise[:, near : s_r - far] = noise_r[:, near : s_r - far]

        snr = signal / noise
        obj_mask = np.zeros(signal.shape, dtype=bool)
        obj_mask[:, near : s_r - far] = snr[:, near : s_r - far] > self.snr_thresh

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[np.ndarray, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
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
            return ker_a, ker_b

        self.r_ker_a, self.r_ker_b = make_caso_kernels(
            train_window[0], self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(
            train_window[1], self.pad_d)

    @staticmethod
    def _caso(
        signal: Float[np.ndarray, "n"],
        ker_a: Float[np.ndarray, "w"],
        ker_b: Float[np.ndarray, "w"],
        snr: float,
        pad: int,
    ) -> tuple[Bool[np.ndarray, "m"], Float[np.ndarray, "m"]]:
        """Run 1D CFAR CASO, returning a detection mask and noise level."""
        cor_a = correlate(signal, ker_a, mode="valid")
        cor_b = correlate(signal, ker_b, mode="valid")
        noise = np.minimum(cor_a, cor_b)
        detect = signal[pad:-pad] > snr * noise
        return detect, noise

    def _cfar(
        self, signal_cube: Float[np.ndarray, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
    ]:
        """Run 2D CFAR CASO on a batch of radar cubes."""
        # non-coherent signal combination along the antenna array
        signal = np.sum(signal_cube**2, axis=(2, 3)).transpose(0, 2, 1) + 1
        batch, s_r, s_d = signal.shape

        near, far = self.discard_r[0], self.discard_r[1]
        sig_discard = signal[:, near : s_r - far]
        sig_pad_r = np.concatenate(
            (
                sig_discard[:, : self.pad_r],
                sig_discard,
                sig_discard[:, -self.pad_r :],
            ),
            axis=1,
        )
        sig_pad_d = np.pad(
            signal, ((0, 0), (0, 0), (self.pad_d, self.pad_d)), mode="wrap"
        )

        # detection: loop over (batch, doppler) rows for the range axis, and
        # over (batch, range) rows for the doppler axis.
        detect_r = np.zeros((batch, s_r - near - far, s_d), dtype=bool)
        noise_r = np.zeros((batch, s_r - near - far, s_d), dtype=np.float32)
        for b in range(batch):
            for d in range(s_d):
                detect_r[b, :, d], noise_r[b, :, d] = self._caso(
                    sig_pad_r[b, :, d], self.r_ker_a, self.r_ker_b,
                    self.snr_r, self.pad_r)

        detect = np.zeros(signal.shape, dtype=bool)
        detect[:, near : s_r - far] = detect_r
        noise = np.ones(signal.shape, dtype=np.float32)
        noise[:, near : s_r - far] = noise_r

        detect_d = np.zeros((batch, s_r, s_d), dtype=bool)
        for b in range(batch):
            for r in range(s_r):
                detect_d[b, r, :], _ = self._caso(
                    sig_pad_d[b, r, :], self.d_ker_a, self.d_ker_b,
                    self.snr_d, self.pad_d)

        snr = signal / noise
        obj_mask = np.logical_and(detect, detect_d)

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[np.ndarray, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
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
        return self._cfar(signal_cube)
