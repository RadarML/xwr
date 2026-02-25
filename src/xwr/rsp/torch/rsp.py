"""Radar Signal Processing implementations."""

from abc import ABC

import numpy as np
import torch
from jaxtyping import Complex64, Float32, Shaped
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
        self, array: Complex64[Tensor, "..."] | Float32[Tensor, "..."],
        axes: tuple[int, ...],
        size: tuple[int, ...] | None = None,
        shift: tuple[int, ...] | None = None
    ) -> Complex64[Tensor, "..."]:
        if array.dtype == torch.float32:
            fftd = torch.fft.rfftn(array, s=size, dim=axes)
        else:
            fftd = torch.fft.fftn(array, s=size, dim=axes)
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
        x: Complex64[Tensor, "..."] | Float32[Tensor, "..."], axis: int
    ) -> Complex64[Tensor, "..."] | Float32[Tensor, "..."]:
        hann = np.hanning(x.shape[axis] + 2).astype(np.float32)[1:-1]
        broadcast: list[None | slice] = [None] * x.ndim
        broadcast[axis] = slice(None)
        window = torch.from_numpy(
            (hann / np.mean(hann))[tuple(broadcast)]).to(x.device)
        return x * window


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

        zeros = torch.zeros(
            (batch, doppler, 2, range), dtype=rd.dtype, device=rd.device)
        el_0 = torch.cat([zeros, rd[:, :, 1, :, :], zeros], dim=2)
        el_1 = torch.cat([rd[:, :, 0, :, :], rd[:, :, 2, :, :]], dim=2)
        return torch.stack([el_0, el_1], dim=2)


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


class AWR2944EVM(RSPTorch):
    """Radar Signal Processing for AWR2944EVM.

    !!! info "Antenna Array"

        The AWR2944EVM has a virtual array on a 2x12 grid:
        ```
                2-1 2-2 2-3 2-4
        1-1 1-2 1-3 1-4 3-1 3-2 3-3 3-4 4-1 4-2 4-3 4-4
        ```
        The horizontal spacing is 1/2 wavelength, and the vertical spacing is
        0.8 wavelength.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    SAMPLE_TYPE = "I"

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        zeros_2 = torch.zeros(
            (batch, doppler, 1, 2, range),
            dtype=torch.complex64, device=rd.device)
        zeros_6 = torch.zeros(
            (batch, doppler, 1, 6, range),
            dtype=torch.complex64, device=rd.device)
        row0 = torch.cat([zeros_2, rd[:, :, 1:2, :, :], zeros_6], dim=3)
        row1 = torch.cat(
            [rd[:, :, 0:1, :, :], rd[:, :, 2:3, :, :], rd[:, :, 3:4, :, :]],
            dim=3)
        return torch.cat([row0, row1], dim=2)

    def elevation_azimuth(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        """Calculate elevation-azimuth spectrum from range-doppler spectrum.

        !!! warning

            Special treatment is needed for the AWR2944EVM since the two
            rows of virtual elements are 0.8 wavelength apart instead of
            0.5. We compute the DTFT along the elevation axis with the
            steering matrix corresponding to the 0.8 lambda spacing.

        Args:
            rd: range-doppler spectrum.

        Returns:
            Computed elevation-azimuth spectrum, with windowing and padding if
                specified.
        """
        mimo = self.mimo_virtual_array(rd)

        if self.window.get("elevation", self._default_window):
            mimo = self.hann(mimo, 2)
        if self.window.get("azimuth", self._default_window):
            mimo = self.hann(mimo, 3)

        az_size = self.size.get("azimuth", mimo.shape[3])
        spectrum = self.fft(mimo, axes=(3,), shift=(3,), size=(az_size,))

        el_size = self.size.get("elevation", mimo.shape[2])
        sin_theta = torch.linspace(-1, 1, el_size, device=rd.device)
        el_elements = torch.arange(
            mimo.shape[2], dtype=torch.float32, device=rd.device)
        phases = (-2j * torch.pi * 0.8
                  * torch.outer(sin_theta, el_elements).to(torch.complex64))
        steering_matrix = torch.exp(phases)

        el_az_spectrum = torch.einsum(
            'bdear,ke->bdkar',
            spectrum, steering_matrix
        )

        return el_az_spectrum
