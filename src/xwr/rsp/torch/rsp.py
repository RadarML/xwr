"""Radar Signal Processing implementations."""

from abc import ABC

import numpy as np
import torch
from jaxtyping import Complex64, Shaped
from torch import Tensor

from xwr.rsp import RSP


class RSPTorch(RSP[Tensor], ABC):
    """Base Radar Signal Processing with common functionality.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def fft(
        self, array: Complex64[Tensor, "..."], axes: tuple[int, ...],
        shift: tuple[int, ...] | None = None
    ) -> Complex64[Tensor, "..."]:
        fftd = torch.fft.fftn(array, dim=axes)
        if shift is None:
            return fftd
        else:
            return torch.fft.fftshift(fftd, dim=shift)

    @staticmethod
    def pad(
        x: Shaped[Tensor, "..."], axis: int, size: int
    ) -> Shaped[Tensor, "..."]:
        if size <= x.shape[axis]:
            raise ValueError(
                f"Cannot zero-pad axis {axis} to target size {size}, which is "
                f"less than or equal the current size {x.shape[axis]}.")

        shape = list(x.shape)
        shape[axis] = size - x.shape[axis]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)

        return torch.concatenate([x, zeros], dim=axis)

    @staticmethod
    def hann(
        iq: Complex64[Tensor, "..."], axis: int
    ) -> Complex64[Tensor, "..."]:
        hann = np.hanning(iq.shape[axis] + 2).astype(np.float32)[1:-1]
        broadcast: list[None | slice] = [None] * iq.ndim
        broadcast[axis] = slice(None)
        window = torch.from_numpy(
            (hann / np.mean(hann))[tuple(broadcast)]).to(iq.device)
        return iq * window


class AWR1843AOP(RSPTorch):
    """Radar Signal Processing for AWR1843AOP.

    !!! info "Antenna Array"

        In the TI AWR1843AOP, the MIMO virtual array is arranged in a 2D grid:
            ```
            1-1 2-1 3-1   ^
            1-2 2-2 3-2   | Up
            1-3 2-3 3-3
            1-4 2-4 3-4 (TX-RX pairs)
            ```

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        _, _, tx, rx, _ = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        return torch.swapaxes(rd, 2, 3)


class AWR1843Boost(RSPTorch):
    """Radar Signal Processing for AWR1843Boost.

    !!! info "Antenna Array"

        In the TI AWR1843Boost, the MIMO virtual array has resolution 2x8, with
        a single 1/2-wavelength elevated middle antenna element:
        ```
        TX-RX:  2-1 2-2 2-3 2-4           ^
        1-1 1-2 1-3 1-4 3-1 3-2 3-3 3-4   | Up
        ```

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        mimo = torch.zeros(
            (batch, doppler, 2, 8, range),
            dtype=torch.complex64, device=rd.device)
        mimo[:, :, 0, 2:6, :] = rd[:, :, 1, :, :]
        mimo[:, :, 1, 0:4, :] = rd[:, :, 0, :, :]
        mimo[:, :, 1, 4:8, :] = rd[:, :, 2, :, :]
        return mimo



class AWR1642Boost(RSPTorch):
    """Radar Signal Processing for the AWR1642 or AWR1843 with TX2 disabled.

    !!! info "Antenna Array"

        The TI AWR1642Boost (or AWR1843Boost with TX2 disabled) has a
        1x8 linear MIMO array:
        ```
        1-1 1-2 1-3 1-4 2-1 2-2 2-3 2-4
        ```

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape

        # 1843Boost cast as 1642Boost
        if tx == 3:
            if rx != 4:
                raise ValueError(
                    f"Expected (tx, rx)=3x4 in 1843Boost -> 1642Boost "
                    f"emulation, got tx={tx} and rx={rx}.")
            rd = rd[:, :, [0, 2], :, :]
        else:
            if tx != 2 or rx != 4:
                raise ValueError(
                    f"Expected (tx, rx)=2x4, got tx={tx} and rx={rx}.")

        return rd.reshape(batch, doppler, 1, -1, range)
