"""Radar Signal Processing implementations."""

from abc import ABC
from typing import Literal

import numpy as np
from jaxtyping import Complex64, Shaped
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
        self, window: bool | dict[
            Literal["range", "doppler", "azimuth", "elevation"], bool] = False,
        size: dict[
            Literal["range", "doppler", "azimuth", "elevation"], int] = {}
    ) -> None:
        super().__init__(window=window, size=size)
        self._fft_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], FFTW] = {}

    def fft(
        self, array: Complex64[np.ndarray, "..."], axes: tuple[int, ...],
        shift: tuple[int, ...] | None = None
    ) -> Complex64[np.ndarray, "..."]:
        key = (array.shape, axes)
        if key not in self._fft_cache:
            self._fft_cache[key] = FFTW(
                np.copy(array), np.zeros_like(array), axes=axes)

        fftd = self._fft_cache[key](array)
        return np.fft.fftshift(fftd, axes=shift) if shift else fftd

    @staticmethod
    def pad(
        x: Shaped[np.ndarray, "..."], axis: int, size: int
    ) -> Shaped[np.ndarray, "..."]:
        if size <= x.shape[axis]:
            raise ValueError(
                f"Cannot zero-pad axis {axis} to target size {size}, which is "
                f"less than or equal the current size {x.shape[axis]}.")

        shape = list(x.shape)
        shape[axis] = size - x.shape[axis]
        zeros = np.zeros(shape, dtype=x.dtype)

        return np.concatenate([x, zeros], axis=axis)

    @staticmethod
    def hann(
        iq: Complex64[np.ndarray, "..."], axis: int
    ) -> Complex64[np.ndarray, "..."]:
        hann = np.hanning(iq.shape[axis] + 2).astype(np.float32)[1:-1]
        broadcast: list[None | slice] = [None] * iq.ndim
        broadcast[axis] = slice(None)
        return iq * (hann / np.mean(hann))[tuple(broadcast)]


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
            If an axis is not spacified, it is not padded.
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
