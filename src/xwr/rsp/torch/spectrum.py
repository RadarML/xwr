"""Calibrated Spectrum Processing."""

from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import conv1d, conv2d


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
        [`CFARCASO`][xwr.rsp.torch.CFARCASO], which tests the range and doppler axes
        separately and requires **both** to fire.

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
        self.mask: Tensor = torch.from_numpy(mask)[None, None]

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move the ring kernel to `device`/`dtype`, if needed."""
        if self.mask.device != device or self.mask.dtype != dtype:
            self.mask = self.mask.to(device=device, dtype=dtype)

    def _cfar(
        self, signal_cube: Float[Tensor, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
    ]:
        """Run 2D cell-averaging CFAR on a batch of radar cubes."""
        # non-coherent signal combination along the antenna array
        signal = (signal_cube**2).sum(dim=(2, 3)).transpose(1, 2) + 1
        self._to(signal.device, signal.dtype)
        _, s_r, _ = signal.shape

        # Torch only supports zero padding here, matching jax's 'fill', so the
        # training cell count is normalized out to compensate at the edges.
        sig = signal[:, None]
        valid = conv2d(torch.ones_like(sig), self.mask, padding="same")
        noise_r = (conv2d(sig, self.mask, padding="same") / valid)[:, 0]

        near, far = self.discard_r[0], self.discard_r[1]
        # 1 outside the discarded band, so the reported SNR there is the raw
        # signal rather than a division by zero.
        noise = signal.new_ones(signal.shape)
        noise[:, near : s_r - far] = noise_r[:, near : s_r - far]

        snr = signal / noise
        obj_mask = signal.new_zeros(signal.shape, dtype=torch.bool)
        obj_mask[:, near : s_r - far] = (
            snr[:, near : s_r - far] > self.snr_thresh)

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[Tensor, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
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
        a separate 1D kernel for the range and doppler axes; a detection is
        reported only where the SNR exceeds the specified threshold on
        **both** axes.

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
            # Stacked as two output channels of a single `conv1d`.
            return torch.from_numpy(np.stack((ker_a, ker_b))[:, None])

        self.r_ker = make_caso_kernels(train_window[0], self.pad_r)
        self.d_ker = make_caso_kernels(train_window[1], self.pad_d)

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move the training kernels to `device`/`dtype`, if needed."""
        if self.r_ker.device != device or self.r_ker.dtype != dtype:
            self.r_ker = self.r_ker.to(device=device, dtype=dtype)
            self.d_ker = self.d_ker.to(device=device, dtype=dtype)

    @staticmethod
    def _caso(
        signal: Float[Tensor, "... n"],
        ker: Float[Tensor, "2 1 w"],
        snr: float,
        pad: int,
    ) -> tuple[Bool[Tensor, "... m"], Float[Tensor, "... m"]]:
        """Run 1D CFAR CASO, returning a detection mask and noise level.

        Operates on the last axis, over an arbitrary batch of leading axes;
        the two one-sided kernels are evaluated as the two output channels of
        a single `conv1d`, whose `valid` output is what jax gets from
        `jnp.correlate(..., mode="valid")`.
        """
        flat = signal.reshape(-1, 1, signal.shape[-1])
        noise = conv1d(flat, ker).min(dim=1).values.reshape(
            *signal.shape[:-1], -1)
        detect = signal[..., pad:-pad] > snr * noise
        return detect, noise

    def _cfar(
        self, signal_cube: Float[Tensor, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
    ]:
        """Run 2D CFAR CASO on a batch of radar cubes."""
        # non-coherent signal combination along the antenna array
        signal = (signal_cube**2).sum(dim=(2, 3)).transpose(1, 2) + 1
        self._to(signal.device, signal.dtype)
        _, s_r, s_d = signal.shape

        near, far = self.discard_r[0], self.discard_r[1]
        sig_discard = signal[:, near : s_r - far]
        sig_pad_r = torch.concat(
            (
                sig_discard[:, : self.pad_r],
                sig_discard,
                sig_discard[:, -self.pad_r :],
            ),
            dim=1,
        )
        sig_pad_d = torch.concat(
            (
                signal[:, :, s_d - self.pad_d :],
                signal,
                signal[:, :, : self.pad_d],
            ),
            dim=2,
        )

        # detection
        detect_r, noise_r = self._caso(
            sig_pad_r.transpose(1, 2), self.r_ker, self.snr_r, self.pad_r)
        detect_r, noise_r = detect_r.transpose(1, 2), noise_r.transpose(1, 2)
        detect = signal.new_zeros(signal.shape, dtype=torch.bool)
        detect[:, near : s_r - far] = detect_r
        noise = signal.new_ones(signal.shape)
        noise[:, near : s_r - far] = noise_r
        detect_d, _ = self._caso(
            sig_pad_d, self.d_ker, self.snr_d, self.pad_d)

        snr = signal / noise
        obj_mask = torch.logical_and(detect, detect_d)

        return obj_mask, signal, snr

    def __call__(
        self, signal_cube: Float[Tensor, "batch doppler tx rx range"]
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
        Float[Tensor, "batch range doppler"],
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
