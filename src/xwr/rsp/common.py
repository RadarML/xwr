"""Common RSP components."""

from abc import ABC, abstractmethod
from typing import Literal, overload

import numpy as np
from jaxtyping import Complex64, Int16, Shaped


@overload
def iq_from_iiqq(
    iiqq: Int16[np.ndarray, "... n"], complex: Literal[True] = True
) -> Complex64[np.ndarray, "... n/2"]: ...

@overload
def iq_from_iiqq(
    iiqq: Int16[np.ndarray, "... n"], complex: Literal[False] = False
) -> Int16[np.ndarray, "... n/2 2"]: ...

def iq_from_iiqq(
    iiqq: Int16[np.ndarray, "... n"], complex: bool = True
) -> Complex64[np.ndarray, "... n/2"] | Int16[np.ndarray, "... n/2 2"]:
    """Un-interleave IIQQ data.

    Args:
        iiqq: interleaved IIQQ data; see [`RadarFrame`][xwr.capture.types.].
        complex: if `True`, return a complex array. Otherwise, return an array
            with trailing I (in-phase) and Q (quadrature) channels, in that
            order.

    Returns:
        IQ data in a reasonable, uninterleaved format.
    """
    shape = (*iiqq.shape[:-1], iiqq.shape[-1] // 2)

    if complex:
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = iiqq[..., 0::4] + 1j * iiqq[..., 2::4]
        iq[..., 1::2] = iiqq[..., 1::4] + 1j * iiqq[..., 3::4]
        return iq
    else:
        iq = np.zeros((*shape, 2), dtype=np.int16)
        iq[..., 0::2, 0] = iiqq[..., 0::4]
        iq[..., 1::2, 0] = iiqq[..., 1::4]
        iq[..., 0::2, 1] = iiqq[..., 2::4]
        iq[..., 1::2, 1] = iiqq[..., 3::4]
        return iq


class BaseRSP(ABC):
    """Base Radar Signal Processing with common functionality.

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
        if isinstance(window, bool):
            self.window = {}
            self._default_window = self.window
        else:
            self.window = window
            self._default_window = False

        self.size = size

    @staticmethod
    def pad(
        x: Shaped[np.ndarray, "..."], axis: int, size: int
    ) -> Shaped[np.ndarray, "..."]:
        """Pad the specified axis with zeros to reach the target size.

        Args:
            x: input array.
            axis: Axis along which to pad.
            pad: Target size after padding.

        Returns:
            Input array with padding applied along the specified axis.
        """
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
        """Apply a Hann window to the specified axis of the IQ data.

        Args:
            iq: IQ data.
            axis: Axis along which to apply the Hann window.

        Returns:
            IQ data with the Hann window applied along the specified axis.
        """
        hann = np.hanning(iq.shape[axis])
        broadcast: list[None | slice] = [None] * iq.ndim
        broadcast[axis] = slice(None)
        return iq * (hann / np.mean(hann))[tuple(broadcast)]

    def _pad_and_window(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler tx rx range"]:
        if self.window.get("elevation", self._default_window):
            rd = self.hann(rd, 2)
        if self.window.get("azimuth", self._default_window):
            rd = self.hann(rd, 3)

        if self.size.get("elevation") is not None:
            rd = self.pad(rd, 2, self.size["elevation"])
        if self.size.get("azimuth") is not None:
            rd = self.pad(rd, 3, self.size["azimuth"])

        return rd

    def doppler_range(
        self, iq: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler tx rx range"]:
        """Calculate range-doppler spectrum from IQ data.

        Args:
            iq: IQ data.

        Returns:
            Computed range-doppler spectrum, with windowing if specified.
        """
        if self.window.get("range", self._default_window):
            iq = self.hann(iq, 4)
        if self.window.get("doppler", self._default_window):
            iq = self.hann(iq, 1)

        if self.size.get("range") is not None:
            iq = self.pad(iq, 4, self.size["range"])
        if self.size.get("doppler") is not None:
            iq = self.pad(iq, 1, self.size["doppler"])

        iq = np.fft.fftn(iq, axes=(1, 4))
        iq = np.fft.fftshift(iq, axes=(1,))
        return iq

    @abstractmethod
    def mimo_virtual_array(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler elevation azimuth range"]:
        """Set up MIMO virtual array from range-doppler spectrum.

        Args:
            rd: range-doppler spectrum.

        Returns:
            Computed MIMO virtual array, in elevation-azimuth order.
        """
        ...

    def elevation_azimuth(
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        """Calculate elevation-azimuth spectrum from range-doppler spectrum.

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

        if self.size.get("elevation") is not None:
            mimo = self.pad(mimo, 2, self.size["elevation"])
        if self.size.get("azimuth") is not None:
            mimo = self.pad(mimo, 3, self.size["azimuth"])

        aoa = np.fft.fftn(mimo, axes=(2, 3))
        aoa = np.fft.fftshift(aoa, axes=(2, 3))
        return aoa

    def process(
        self,
        iq: Complex64[np.ndarray, "#batch doppler tx rx range"]
        | Int16[np.ndarray, "#batch doppler tx rx range*2"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        """Process IQ data to compute elevation-azimuth spectrum.

        Args:
            iq: IQ data in complex or interleaved int16 IQ format.

        Returns:
            Computed doppler-elevation-azimuth-range spectrum.
        """
        if iq.dtype == np.int16:
            iq = iq_from_iiqq(iq, complex=True)

        dr = self.doppler_range(iq)
        drae = self.elevation_azimuth(dr)
        return drae
