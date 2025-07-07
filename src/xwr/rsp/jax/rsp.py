"""Radar Signal Processing implementations."""

from abc import ABC

from jax import numpy as jnp
from jaxtyping import Array, Complex64, Float32, Int, Int16, Shaped

from xwr.rsp import RSP, iq_from_iiqq


class RSPJax(RSP[Array], ABC):
    """Base Radar Signal Processing with common functionality.

    Args:
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def fft(
        self, array: Complex64[Array, "..."], axes: tuple[int, ...],
        shift: tuple[int, ...] | None = None
    ) -> Complex64[Array, "..."]:
        fftd = jnp.fft.fftn(array, axes=axes)
        if shift is None:
            return fftd
        else:
            return jnp.fft.fftshift(fftd, axes=shift)

    @staticmethod
    def pad(
        x: Shaped[Array, "..."], axis: int, size: int
    ) -> Shaped[Array, "..."]:
        if size <= x.shape[axis]:
            raise ValueError(
                f"Cannot zero-pad axis {axis} to target size {size}, which is "
                f"less than or equal the current size {x.shape[axis]}.")

        shape = list(x.shape)
        shape[axis] = size - x.shape[axis]
        zeros = jnp.zeros(shape, dtype=x.dtype)

        return jnp.concatenate([x, zeros], axis=axis)

    @staticmethod
    def hann(
        iq: Complex64[Array, "..."], axis: int
    ) -> Complex64[Array, "..."]:
        hann = jnp.hanning(iq.shape[axis] + 2)[1:-1]
        broadcast: list[None | slice] = [None] * iq.ndim
        broadcast[axis] = slice(None)
        return iq * (hann / jnp.mean(hann))[tuple(broadcast)]

    def azimuth_aoa(
        self, iq: Complex64[Array, "batch slow tx rx fast"]
        | Int16[Array, "batch slow tx rx fast*2"]
    ) -> Int[Array, "batch doppler range"]:
        """Estimate angle of arrival (AoA).

        !!! note

            The AOA bin resolution is determined by the number of bins this
            RSP instance is configured with.

        Args:
            iq: raw IQ data.

        Returns:
            Estimated angle of arrival (AoA) index for each range-Doppler bin.
        """
        spec: Complex64[Array, "batch doppler el az range"] = self(iq)
        az_spec: Float32[Array, "batch doppler az range"] = (
            jnp.mean(jnp.abs(spec), axis=2))
        return jnp.argmax(az_spec, axis=2)

class AWR1843AOP(RSPJax):
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
        self, rd: Complex64[Array, "#batch doppler tx rx range"]
    ) -> Complex64[Array, "#batch doppler el az range"]:
        _, _, tx, rx, _ = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        return jnp.swapaxes(rd, 2, 3)


class AWR1843Boost(RSPJax):
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
        self, rd: Complex64[Array, "#batch doppler tx rx range"]
    ) -> Complex64[Array, "#batch doppler el az range"]:
        batch, doppler, tx, rx, range = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(
                f"Expected (tx, rx)=3x4, got tx={tx} and rx={rx}.")

        mimo = jnp.zeros(
            (batch, doppler, 2, 8, range), dtype=jnp.complex64
        ).at[:, :, 0, 2:6, :].set(rd[:, :, 1, :, :]
        ).at[:, :, 1, 0:4, :].set(rd[:, :, 0, :, :]
        ).at[:, :, 1, 4:8, :].set(rd[:, :, 2, :, :])
        return mimo

    def elevation_aoa(
        self, iq: Complex64[Array, "batch slow tx rx fast"]
        | Int16[Array, "batch slow tx rx fast*2"]
    ) -> Float32[Array, "batch doppler range"]:
        """Estimate elevation angle of arrival (AoA).

        Args:
            iq: raw IQ data.

        Returns:
            Estimated elevation angle of arrival (AoA) in radians for each
                range-Doppler bin.
        """
        iq = iq_from_iiqq(iq)
        rd = self.doppler_range(iq)
        mimo = self.mimo_virtual_array(rd)[:, :, :, 2:-2]

        angle = jnp.angle(mimo)
        phase_diff: Float32[Array, "batch doppler range"] = jnp.median(
            angle[:, :, 0] - angle[:, :, 1], axis=3)
        el_angle = jnp.arcsin((phase_diff / jnp.pi + 1) % 2 - 1)
        return el_angle


class AWR1642Boost(RSPJax):
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
        self, rd: Complex64[Array, "#batch doppler tx rx range"]
    ) -> Complex64[Array, "#batch doppler el az range"]:
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
