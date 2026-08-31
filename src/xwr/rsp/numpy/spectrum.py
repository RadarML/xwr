"""Calibrated Spectrum Processing."""

import numpy as np
from jaxtyping import Bool, Float
from scipy.signal import convolve2d, correlate

from xwr.rsp import spectrum as base


def _integrate(
    signal_cube: Float[np.ndarray, "batch doppler channel range"]
) -> Float[np.ndarray, "batch range doppler"]:
    """Combine the channel axis non-coherently into a range-doppler image.

    Args:
        signal_cube: batch of post range doppler FFT radar cubes in amplitude,
            flattened to a single channel axis.

    Returns:
        Integrated power, offset by 1 so that an empty cell has unit power
            instead of dividing by zero downstream.
    """
    return np.sum(signal_cube**2, axis=2).transpose(0, 2, 1) + 1


class CFAR(base.CFAR[np.ndarray]):
    """Cell-averaging CFAR."""

    @staticmethod
    def _asarray(x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        return x

    def _noise(
        self, signal: Float[np.ndarray, "range doppler"]
    ) -> Float[np.ndarray, "range doppler"]:
        """Get the ring-averaged noise floor for a range-doppler image."""
        valid = convolve2d(np.ones_like(signal), self.mask, mode="same")
        return convolve2d(signal, self.mask, mode="same") / valid

    def _cfar(
        self, signal_cube: Float[np.ndarray, "batch doppler channel range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
    ]:
        signal = _integrate(signal_cube)
        _, s_r, _ = signal.shape

        noise_r = np.stack([self._noise(s) for s in signal])

        near, far = self.discard_r[0], self.discard_r[1]
        # 1 outside the discarded band, so the reported SNR there is the raw
        # signal rather than a division by zero.
        noise = np.ones_like(signal)
        noise[:, near : s_r - far] = noise_r[:, near : s_r - far]

        snr = signal / noise
        obj_mask = np.zeros(signal.shape, dtype=bool)
        obj_mask[:, near : s_r - far] = (
            snr[:, near : s_r - far] > self.snr_thresh)

        return obj_mask, signal, snr


class CFARCASO(base.CFARCASO[np.ndarray]):
    """Cell-averaging Smallest of CFAR."""

    def __init__(
        self,
        guard: tuple[int, int] = (8, 0),
        train: tuple[int, int] = (8, 4),
        snr_thresh: tuple[float, float] = (5.0, 3.0),
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        super().__init__(guard, train, snr_thresh, discard_range)

        # Unlike the jax backend, which accumulates the one-sided means from
        # shifted slices, scipy correlates against an explicit kernel.
        def make_caso_kernels(n_train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:n_train], ker_b[-n_train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            return ker_a, ker_b

        self.r_ker_a, self.r_ker_b = make_caso_kernels(
            self.train_r, self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(
            self.train_d, self.pad_d)

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
        self, signal_cube: Float[np.ndarray, "batch doppler channel range"]
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
        Float[np.ndarray, "batch range doppler"],
    ]:
        signal = _integrate(signal_cube)
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
