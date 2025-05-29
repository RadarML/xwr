"""Radar Signal Processing implementations."""

import numpy as np
from jaxtyping import Complex64

from .common import BaseRSP


class AWR1843AOP(BaseRSP):
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


class AWR1843Boost(BaseRSP):
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



class AWR1642Boost(BaseRSP):
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
        if tx != 2 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=2x4, got tx={tx} and rx={rx}.")
        return rd.reshape(batch, doppler, -1, range)
