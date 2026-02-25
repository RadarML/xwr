"""Radar Signal Processing implementations."""

from abc import ABC
from collections.abc import Mapping
from typing import Literal

import numpy as np
from jaxtyping import Complex64, Float32, Shaped
from pyfftw import FFTW

from xwr.rsp import RSP


class RSPNumpy(RSP[np.ndarray], ABC):
    """Numpy Radar Signal Processing base class.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def __init__(
        self, window: bool | Mapping[
            Literal["range", "doppler", "azimuth", "elevation"], bool] = False,
        size: Mapping[
            Literal["range", "doppler", "azimuth", "elevation"], int] = {}
    ) -> None:
        super().__init__(window=window, size=size)
        self._fft_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], np.dtype], FFTW] = {}

    def fft(
        self, array: Complex64[np.ndarray, "..."] | Float32[np.ndarray, "..."],
        axes: tuple[int, ...],
        size: tuple[int, ...] | None = None,
        shift: tuple[int, ...] | None = None
    ) -> Complex64[np.ndarray, "..."]:
        if size is not None:
            for axis, s in zip(axes, size):
                array = self.pad(array, axis, s)

        key = (array.shape, axes, array.dtype)
        out_shape = array.shape
        if array.dtype == np.float32:
            out_shape = (*out_shape[:-1], out_shape[-1] // 2 + 1)

        if key not in self._fft_cache:
            self._fft_cache[key] = FFTW(
                np.copy(array),
                np.zeros(out_shape, dtype=np.complex64), axes=axes)

        fftd = self._fft_cache[key](array)
        return np.fft.fftshift(fftd, axes=shift) if shift else fftd

    @staticmethod
    def pad(
        x: Shaped[np.ndarray, "..."], axis: int, size: int
    ) -> Shaped[np.ndarray, "..."]:
        if size == x.shape[axis]:
            return x
        elif size < x.shape[axis]:
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(0, size)
            return x[tuple(slices)]
        else:
            shape = list(x.shape)
            shape[axis] = size - x.shape[axis]
            zeros = np.zeros(shape, dtype=x.dtype)
            return np.concatenate([x, zeros], axis=axis)

    @staticmethod
    def hann(
        x: Complex64[np.ndarray, "..."] | Float32[np.ndarray, "..."],
        axis: int
    ) -> Complex64[np.ndarray, "..."] | Float32[np.ndarray, "..."]:
        hann = np.hanning(x.shape[axis] + 2).astype(np.float32)[1:-1]
        broadcast: list[None | slice] = [None] * x.ndim
        broadcast[axis] = slice(None)
        return x * (hann / np.mean(hann))[tuple(broadcast)]


class AWR1843AOP(RSPNumpy):
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
    """

    def mimo_virtual_array(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        _, _, tx, rx, _ = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        return np.swapaxes(rd, 2, 3)


class AWR1843Boost(RSPNumpy):
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
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        mimo = np.zeros((batch, doppler, 2, 8, range), dtype=np.complex64)
        mimo[:, :, 0, 2:6, :] = rd[:, :, 1, :, :]
        mimo[:, :, 1, 0:4, :] = rd[:, :, 0, :, :]
        mimo[:, :, 1, 4:8, :] = rd[:, :, 2, :, :]
        return mimo


class AWR1642Boost(RSPNumpy):
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
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
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


class AWRL6844EVM(RSPNumpy):
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

    # TX and RX indices (0-based) for each (elevation, azimuth) position.
    _TX = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [2, 2, 3, 3], [2, 2, 3, 3]])
    _RX = np.array([[0, 3, 0, 3], [1, 2, 1, 2], [0, 3, 0, 3], [1, 2, 1, 2]])
    # TX2 and TX4 (1-indexed, i.e. 0-indexed TX1 and TX3) are 180° out of phase.
    _PHASE = np.array(
        [[-1, -1, 1, 1], [-1, -1, 1, 1], [1, 1, -1, -1], [1, 1, -1, -1]],
        dtype=np.float32)

    def mimo_virtual_array(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        _, _, tx, rx, _ = rd.shape
        if tx != 4 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=4x4, got tx={tx} and rx={rx}.")

        return rd[:, :, self._TX, self._RX, :] * self._PHASE[None, None, :, :, None]


class AWR2944EVM(RSPNumpy):
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
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        mimo = np.zeros((batch, doppler, 2, 12, range), dtype=np.complex64)
        mimo[:, :, 0, 2:6, :] = rd[:, :, 1, :, :]
        mimo[:, :, 1, 0:4, :] = rd[:, :, 0, :, :]
        mimo[:, :, 1, 4:8, :] = rd[:, :, 2, :, :]
        mimo[:, :, 1, 8:12, :] = rd[:, :, 3, :, :]

        return mimo

    def elevation_azimuth(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
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
        sin_theta = np.linspace(-1, 1, el_size)
        el_elements = np.arange(mimo.shape[2])
        phases = -2j * np.pi * 0.8 * np.outer(sin_theta, el_elements)
        steering_matrix = np.exp(phases).astype(np.complex64)

        el_az_spectrum = np.einsum(
            'bdear,ke->bdkar',
            spectrum, steering_matrix, optimize=True
        )

        return el_az_spectrum
