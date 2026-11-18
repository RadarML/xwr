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
            If an axis is not specified, it is not padded.
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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

    @staticmethod
    def doppler_correction(
        rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler tx rx range"]:
        _, n_doppler, n_tx, _, _ = rd.shape
        f = (torch.arange(n_doppler, device=rd.device, dtype=torch.float32)
             - n_doppler // 2) / n_doppler
        t = torch.arange(n_tx, device=rd.device, dtype=torch.float32) / n_tx
        angle = -2 * np.pi * torch.outer(f, t)
        phase = torch.polar(torch.ones_like(angle), angle)
        return rd * phase[None, :, :, None, None]


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
            If an axis is not specified, it is not padded.
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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
            If an axis is not specified, it is not padded.
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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
            If an axis is not specified, it is not padded.
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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


class AWRL6844EVM(RSPTorch):
    """Radar Signal Processing for AWRL6844.

    !!! info "Antenna Array"

        The AWRL6844 has a 4x4 MIMO virtual array with λ/2 spacing:
        ```
        2-1 2-4 1-1 1-4   ^
        2-2 2-3 1-2 1-3   | Up
        3-1 3-4 4-1 4-4
        3-2 3-3 4-2 4-3 (TX-RX pairs)
        ```

    !!! info "TX Phase Relationship"

        TX1 and TX3 are in phase with each other. TX2 and TX4 are also in
        phase with each other, but are 180° out of phase with TX1 and TX3.
        Their contributions to the virtual array are negated accordingly.

        Source: Table 3-1, *EVM User's Guide: AWRL6844EVM IWRL6844EVM*.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not specified, it is not padded.
    """

    SAMPLE_TYPE = "I"

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        _, _, tx, rx, _ = rd.shape
        if tx != 4 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=4x4, got tx={tx} and rx={rx}.")

        tx_idx = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 0, 0], [2, 2, 3, 3], [2, 2, 3, 3]],
            device=rd.device)
        rx_idx = torch.tensor(
            [[0, 3, 0, 3], [1, 2, 1, 2], [0, 3, 0, 3], [1, 2, 1, 2]],
            device=rd.device)
        phase = torch.tensor(
            [[-1, -1, 1, 1], [-1, -1, 1, 1], [1, 1, -1, -1], [1, 1, -1, -1]],
            dtype=torch.float32, device=rd.device)
        return rd[:, :, tx_idx, rx_idx, :] * phase[None, None, :, :, None]


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
            If an axis is not specified, it is not padded.
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


class AWR2E44PEVM(RSPTorch):
    """Radar Signal Processing for AWR2E44PEVM.

    !!! info "Antenna Array"

        The AWR2E44PEVM (launch-on-package antennas) has a sparse virtual
        array on a 2x19 grid with 1/2 wavelength spacing in both azimuth and
        elevation:
        ```
                                            4-1     4-2                 4-3     4-4   ^
        1-1     1-2 2-1     2-2 3-1 1-3 3-2 1-4 2-3     2-4 3-3     3-4               | Up
        ```
        Empty positions are filled with zeros, which raises the angular
        sidelobes compared to a filled array.

    !!! info "Channel Phase Calibration"

        The AWR2E44PEVM channels have large phase offsets, including a ~180°
        inversion of TX3 and RX4, so each TX-RX pair is multiplied by a
        unit-magnitude phase correction before forming the virtual array.
        The corrections are the phases of `antennaCalibParams` in the TI DDM
        demo profiles for the AWR2E44P
        (`mmwave_mcuplus_sdk_{version}/ti/demo/awr2x44P/mmw_ddm/profiles/awr2E44P`),
        which are given in `Im(Q) Re(I)` order; amplitudes are not corrected.

        Source: `antGeometryCfg` in the TI demo profiles for the AWR2E44P
        (`mmwave_mcuplus_sdk_{version}/ti/demo/awr2x44P/mmw_tdm/profiles/awr2E44P`),
        which use the TX1, TX2, TX3, TX4 chirp order.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not specified, it is not padded.
    """

    SAMPLE_TYPE = "I"

    # (elevation row, azimuth column) of each TX-RX pair; row 0 is up.
    _ROW = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    _COL = [[0, 2, 7, 9], [3, 5, 10, 12], [6, 8, 13, 15], [9, 11, 16, 18]]
    _SHAPE = (2, 19)
    # TI `antennaCalibParams` for the AWR2E44P, as (Im(Q), Re(I)) per TX-RX
    # pair in TX-major, RX-minor order.
    _CALIBRATION_QI = np.array([
        [0.6718, 0.3970], [0.5714, 0.5259], [-0.0907, 0.8665],
        [0.0989, -0.9754], [0.8083, 0.0366], [0.7769, 0.2025],
        [0.3331, 0.8275], [-0.3769, -0.9263], [-0.6290, -0.4124],
        [-0.5234, -0.5298], [0.1282, -0.8125], [-0.1418, 0.9100],
        [0.5869, 0.3367], [0.4966, 0.4459], [-0.0732, 0.7262],
        [0.0808, -0.8108]])
    # Unit-magnitude phase correction, shape (tx, rx).
    _CALIBRATION = np.exp(1j * np.arctan2(
        _CALIBRATION_QI[:, 0], _CALIBRATION_QI[:, 1])
    ).reshape(4, 4).astype(np.complex64)

    def mimo_virtual_array(
        self, rd: Complex64[Tensor, "#batch doppler tx rx range"]
    ) -> Complex64[Tensor, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        if tx != 4 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=4x4, got tx={tx} and rx={rx}.")

        mimo = torch.zeros(
            (batch, doppler, *self._SHAPE, range),
            dtype=torch.complex64, device=rd.device)
        row = torch.tensor(self._ROW, device=rd.device)
        col = torch.tensor(self._COL, device=rd.device)
        cal = torch.from_numpy(self._CALIBRATION).to(rd.device)
        mimo[:, :, row, col, :] = rd * cal[None, None, :, :, None]
        return mimo
