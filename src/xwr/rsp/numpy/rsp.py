"""Radar Signal Processing implementations."""

from abc import ABC
from collections.abc import Mapping
from typing import Literal

import numpy as np
from jaxtyping import Complex64, Int16, Shaped
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
            tuple[tuple[int, ...], tuple[int, ...]], FFTW] = {}

    def fft(
        self, array: Complex64[np.ndarray, "..."], axes: tuple[int, ...],
        size: tuple[int, ...] | None = None,
        shift: tuple[int, ...] | None = None
    ) -> Complex64[np.ndarray, "..."]:
        if size is not None:
            for axis, s in zip(axes, size):
                array = self.pad(array, axis, s)

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


class AWR2944EVM(RSPNumpy):
    """Radar Signal Processing for AWR2944EVM.

    !!! info "Antenna Array"

        TODO

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def __call__(
        self, iq: Complex64[np.ndarray, "#batch doppler tx rx _range"]
            | Int16[np.ndarray, "#batch doppler tx rx _range"]
    ) -> Complex64[np.ndarray, "#batch doppler2 el az _range"]:
        """Process IQ data to compute elevation-azimuth spectrum.

        Args:
            iq: IQ data in complex or interleaved int16 IQ format.

        Returns:
            Computed doppler-elevation-azimuth-range spectrum.
        """
        if iq.dtype == np.int16:
            iq = iq.astype(np.complex64)

        dr = self.doppler_range(iq)
        drae = self.elevation_azimuth(dr)
        return drae

    def doppler_range(
        self, iq: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler2 tx rx range2"]:
        """Calculate range-doppler spectrum from IQ data.

        Args:
            iq: IQ data.

        Returns:
            Computed range-doppler spectrum, with windowing if specified.
        """
        nrange = iq.shape[4] // 2

        if self.window.get("range", self._default_window):
            iq = self.hann(iq, 4)
        if self.window.get("doppler", self._default_window):
            iq = self.hann(iq, 1)

        rng = self.fft(
            iq, axes=(4,), size=(self.size.get("range", nrange) * 2,)
        )[..., :nrange]
        return self.fft(
            rng, axes=(1,), shift=(1,),
            size=(self.size.get("doppler", iq.shape[1]),))

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
