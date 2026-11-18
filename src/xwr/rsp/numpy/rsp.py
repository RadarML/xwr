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
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
    """

    def __init__(
        self, window: bool | Mapping[
            Literal["range", "doppler", "azimuth", "elevation"], bool] = False,
        size: Mapping[
            Literal["range", "doppler", "azimuth", "elevation"], int] = {},
        sample_swap: bool = False,
        doppler_correction: bool = False,
    ) -> None:
        super().__init__(
            window=window, size=size, sample_swap=sample_swap,
            doppler_correction=doppler_correction)
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

    @staticmethod
    def doppler_correction(
        rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler tx rx range"]:
        _, n_doppler, n_tx, _, _ = rd.shape
        f = (np.arange(n_doppler) - n_doppler // 2) / n_doppler
        t = np.arange(n_tx) / n_tx
        phase = np.exp(-2j * np.pi * np.outer(f, t)).astype(np.complex64)
        return rd * phase[None, :, :, None, None]


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
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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
        sample_swap: if `True`, swap the I and Q components when
            un-interleaving IIQQ data.
        doppler_correction: if `True`, apply
            [`doppler_correction`][xwr.rsp.RSP.doppler_correction] to the
            range-doppler spectrum in `__call__`, before the angle FFTs.
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


class AWR2E44PEVM(RSPNumpy):
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
        self, rd: Complex64[np.ndarray, "#batch doppler tx rx range"]
    ) -> Complex64[np.ndarray, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        if tx != 4 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=4x4, got tx={tx} and rx={rx}.")

        mimo = np.zeros(
            (batch, doppler, *self._SHAPE, range), dtype=np.complex64)
        mimo[:, :, self._ROW, self._COL, :] = (
            rd * self._CALIBRATION[None, None, :, :, None])
        return mimo
