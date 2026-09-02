"""Calibrated Spectrum Processing."""

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import conv1d, conv2d

from xwr.rsp import spectrum as base


class CACFAR(base.CACFAR[Tensor]):
    """Cell-averaging CFAR."""

    ker: Float[Tensor, "kr kd"] | None = None

    def _to(
        self, device: torch.device, dtype: torch.dtype
    ) -> Float[Tensor, "kr kd"]:
        """Get the ring kernel on `device`/`dtype`, converting if needed."""
        if (self.ker is None or self.ker.device != device
                or self.ker.dtype != dtype):
            self.ker = torch.from_numpy(self.mask).to(
                device=device, dtype=dtype)
        return self.ker

    def _cfar(
        self, signal_cube: Float[Tensor, "batch doppler channel range"]
    ) -> base.Detection[Tensor]:
        # Offset by 1 to prevent division by zero for SNR calculations.
        signal = (signal_cube**2).sum(dim=2).transpose(1, 2) + 1
        _, s_r, _ = signal.shape

        sig = signal[:, None]
        kernel = self._to(signal.device, signal.dtype)[None, None]
        valid = conv2d(torch.ones_like(sig), kernel, padding="same")
        noise_r = (conv2d(sig, kernel, padding="same") / valid)[:, 0]

        near, far = self.discard_r[0], self.discard_r[1]
        noise = signal.new_ones(signal.shape)
        noise[:, near : s_r - far] = noise_r[:, near : s_r - far]

        snr = signal / noise
        obj_mask = signal.new_zeros(signal.shape, dtype=torch.bool)
        obj_mask[:, near : s_r - far] = (
            snr[:, near : s_r - far] > self.snr_thresh)

        return base.Detection(obj_mask, signal, snr)


class CASOCFAR(base.CASOCFAR[Tensor]):
    """Cell-averaging Smallest of CFAR."""

    def __init__(
        self,
        guard: tuple[int, int] = (8, 0),
        train: tuple[int, int] = (8, 4),
        snr_thresh: tuple[float, float] = (5.0, 3.0),
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        super().__init__(guard, train, snr_thresh, discard_range)

        def make_caso_kernels(n_train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:n_train], ker_b[-n_train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            # Stacked as two output channels of a single `conv1d`.
            return torch.from_numpy(np.stack((ker_a, ker_b))[:, None])

        self.r_ker = make_caso_kernels(self.train_r, self.pad_r)
        self.d_ker = make_caso_kernels(self.train_d, self.pad_d)

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
        """Run 1D CFAR CASO, returning a detection mask and noise level."""
        flat = signal.reshape(-1, 1, signal.shape[-1])
        noise = conv1d(flat, ker).min(dim=1).values.reshape(
            *signal.shape[:-1], -1)
        detect = signal[..., pad:-pad] > snr * noise
        return detect, noise

    def _cfar(
        self, signal_cube: Float[Tensor, "batch doppler channel range"]
    ) -> base.Detection[Tensor]:
        # Offset by 1 to prevent division by zero for SNR calculations.
        signal = (signal_cube**2).sum(dim=2).transpose(1, 2) + 1
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

        return base.Detection(obj_mask, signal, snr)
