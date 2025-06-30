"""Backend-agnostic components."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from jaxtyping import Complex64, Int16, Shaped


@runtime_checkable
class ArrayLike(Protocol):
    """Array with shape and dtype."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


@staticmethod
def _check_backend(x: ArrayLike) -> Literal["jax", "torch", "numpy"]:
    """Check the backend of the array-like object without importing.

    - We assume that `numpy` is always available since it's a required
        dependency, and relatively cheap to initialize.
    - For jax, we check for the presence `.at` attribute, which denotes
        jax-specific syntax for index updates.
    - For torch, we check for `.to`, which is a torch-specific method for
        device transfers.

    Args:
        x: Array-like object to check.

    Returns:
        Backend name as a string.
    """
    if isinstance(x, np.ndarray):
        return "numpy"
    elif hasattr(x, "at"):
        return "jax"
    elif hasattr(x, "to"):
        return "torch"
    else:
        raise TypeError(
            f"Unsupported array-like type: {type(x)}. "
            "Expected numpy.ndarray, jax.Array, or torch.Tensor.")


TArray = TypeVar("TArray", bound=ArrayLike)


def iqiq_from_iiqq(
    iiqq: Int16[TArray, "... n"]
) -> Int16[TArray, "... n/2 2"]:
    """Un-interleave IIQQ data.

    Type Parameters:
        - `TArray`: This function is multi-backend, and supports numpy
            `np.ndarray`, jax `jax.Array`, and torch `Tensor`.

    Args:
        iiqq: interleaved IIQQ data; see [`RadarFrame`][xwr.capture.types.].

    Returns:
        IQ data in an uninterleaved format with a trailing I/Q axis.
    """
    shape = (*iiqq.shape[:-1], iiqq.shape[-1] // 2)

    backend = _check_backend(iiqq)
    if backend == "numpy":
        assert isinstance(iiqq, np.ndarray)

        iq = np.zeros((*shape, 2), dtype=np.int16)
        iq[..., 0::2, 1] = iiqq[..., 0::4]
        iq[..., 1::2, 1] = iiqq[..., 1::4]
        iq[..., 0::2, 0] = iiqq[..., 2::4]
        iq[..., 1::2, 0] = iiqq[..., 3::4]
        return cast(Int16[TArray, "... n/2 2"], iq)

    elif backend == "jax":
        from jax import numpy as jnp
        assert isinstance(iiqq, jnp.ndarray)

        iq = jnp.zeros(
            (*shape, 2), dtype=jnp.int16
        ).at[..., 0::2, 1].set(iiqq[..., 0::4]
        ).at[..., 1::2, 1].set(iiqq[..., 1::4]
        ).at[..., 0::2, 0].set(iiqq[..., 2::4]
        ).at[..., 1::2, 0].set(iiqq[..., 3::4])
        return cast(Int16[TArray, "... n/2 2"], iq)

    else:  # backend == "torch"
        import torch
        assert isinstance(iiqq, torch.Tensor)

        iq = torch.zeros((*shape, 2), dtype=torch.int16, device=iiqq.device)
        iq[..., 0::2, 1] = iiqq[..., 0::4]
        iq[..., 1::2, 1] = iiqq[..., 1::4]
        iq[..., 0::2, 0] = iiqq[..., 2::4]
        iq[..., 1::2, 0] = iiqq[..., 3::4]
        return cast(Int16[TArray, "... n/2 2"], iq)


def iq_from_iiqq(
    iiqq: Int16[TArray, "... n"] | Complex64[TArray, "... _n"],
) -> Complex64[TArray, "... n2"]:
    """Un-interleave IIQQ data.

    Type Parameters:
        - `TArray`: This function is multi-backend, and supports numpy
            `np.ndarray`, jax `jax.Array`, and torch `Tensor`.

    Args:
        iiqq: interleaved IIQQ data; see [`RadarFrame`][xwr.capture.types.].
            If already complex, leave it as is.

    Returns:
        Complex IQ data.
    """
    shape = (*iiqq.shape[:-1], iiqq.shape[-1] // 2)

    backend = _check_backend(iiqq)
    if backend == "numpy":
        assert isinstance(iiqq, np.ndarray)

        if iiqq.dtype == np.complex64:
            return iiqq
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
        return cast(Complex64[TArray, "... n/2"], iq)

    elif backend == "jax":
        from jax import numpy as jnp
        assert isinstance(iiqq, jnp.ndarray)

        if iiqq.dtype == jnp.complex64:
            return iiqq
        iq = jnp.zeros(
            shape, dtype=jnp.complex64
        ).at[..., 0::2].set(1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        ).at[..., 1::2].set(1j * iiqq[..., 1::4] + iiqq[..., 3::4])
        return cast(Complex64[TArray, "... n/2"], iq)

    else: # backend == "torch"
        import torch
        assert isinstance(iiqq, torch.Tensor)

        if iiqq.dtype == torch.complex64:
            return iiqq
        iq = torch.zeros(shape, dtype=torch.complex64, device=iiqq.device)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
        return cast(Complex64[TArray, "... n/2"], iq)


class RSP(ABC, Generic[TArray]):
    """Abstract, backend-agnostic Radar Signal Processing base class.

    !!! info

        This class documents the public interface for all radar signal
        processing (RSP) classes, except where otherwise noted.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

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
        self.window: dict[
            Literal["range", "doppler", "azimuth", "elevation"], bool]
        self._default_window: bool | dict[
            Literal["range", "doppler", "azimuth", "elevation"], bool]

        if isinstance(window, bool):
            self.window = {}
            self._default_window = window
        else:
            self.window = window
            self._default_window = False

        self.size = size

    @abstractmethod
    def fft(
        self, array: Complex64[TArray, "..."], axes: tuple[int, ...],
        shift: tuple[int, ...] | None = None
    ) -> Complex64[TArray, "..."]:
        """Compute FFT on the specified axes of the array.

        Args:
            array: Input array.
            axes: Axes along which to compute the FFT.
            shift: Axes to shift after FFT, if any.

        Returns:
            FFT of the input array along the specified axes.
        """
        ...

    @staticmethod
    @abstractmethod
    def pad(
        x: Shaped[TArray, "..."], axis: int, size: int
    ) -> Shaped[TArray, "..."]:
        """Pad the specified axis with zeros to reach the target size.

        Args:
            x: input array.
            axis: Axis along which to pad.
            size: Target size after padding.

        Returns:
            Input array with padding applied along the specified axis.
        """
        ...

    @staticmethod
    @abstractmethod
    def hann(
        iq: Complex64[TArray, "..."], axis: int
    ) -> Complex64[TArray, "..."]:
        """Apply a Hann window to the specified axis of the IQ data.

        Args:
            iq: IQ data.
            axis: Axis along which to apply the Hann window.

        Returns:
            IQ data with the Hann window applied along the specified axis.
        """
        ...

    def doppler_range(
        self, iq: Complex64[TArray, "#batch doppler tx rx range"]
    ) -> Complex64[TArray, "#batch doppler2 tx rx range2"]:
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

        return self.fft(iq, axes=(1, 4), shift=(1,))

    @abstractmethod
    def mimo_virtual_array(
        self, rd: Complex64[TArray, "#batch doppler tx rx range"]
    ) -> Complex64[TArray, "#batch doppler elevation azimuth range"]:
        """Set up MIMO virtual array from range-doppler spectrum.

        Args:
            rd: range-doppler spectrum.

        Returns:
            Computed MIMO virtual array, in elevation-azimuth order.
        """
        ...

    def elevation_azimuth(
        self, rd: Complex64[TArray, "#batch doppler tx rx range"]
    ) -> Complex64[TArray, "#batch doppler el az range"]:
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

        return self.fft(mimo, axes=(2, 3), shift=(2, 3))

    def __call__(
        self, iq: Complex64[TArray, "#batch doppler tx rx _range"]
            | Int16[TArray, "#batch doppler tx rx _range"]
    ) -> Complex64[TArray, "#batch doppler2 el az _range"]:
        """Process IQ data to compute elevation-azimuth spectrum.

        Args:
            iq: IQ data in complex or interleaved int16 IQ format.

        Returns:
            Computed doppler-elevation-azimuth-range spectrum.
        """
        uninterleaved = iq_from_iiqq(iq)
        dr = self.doppler_range(uninterleaved)
        drae = self.elevation_azimuth(dr)
        return drae
