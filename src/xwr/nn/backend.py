"""Internal backend for pytorch / numpy interop."""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
import torch

TArray = TypeVar("TArray", bound = np.ndarray | torch.Tensor)


def concatenate(arrays: Sequence[TArray], axis: int) -> TArray:
    """Concatenate arrays."""
    if isinstance(arrays[0], np.ndarray):
        return cast(TArray, np.concatenate(
            cast(Sequence[np.ndarray], arrays), axis=axis))
    else:
        return cast(TArray, torch.concatenate(
            cast(tuple[torch.Tensor], tuple(arrays)), dim=axis))


def stack(arrays: Sequence[TArray], axis: int) -> TArray:
    """Stack arrays."""
    if isinstance(arrays[0], np.ndarray):
        return cast(TArray, np.stack(
            cast(Sequence[np.ndarray], arrays), axis=axis))
    else:
        return cast(TArray, torch.stack(
            cast(tuple[torch.Tensor], tuple(arrays)), dim=axis))


def flip(arr: TArray, axis: int = 0) -> TArray:
    """Flip array."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.flip(arr, axis=axis))
    else:
        return cast(TArray, torch.flip(arr, dims=(axis,)))


def log(arr: TArray) -> TArray:
    """Elementwise log."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.log(arr))
    else:
        return cast(TArray, torch.log(arr))


def sqrt(arr: TArray) -> TArray:
    """Elementwise square root."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.sqrt(arr))
    else:
        return cast(TArray, torch.sqrt(arr))


def abs(arr: TArray) -> TArray:
    """Elementwise absolute value."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.abs(arr))
    else:
        return cast(TArray, torch.abs(arr))


def angle(arr: TArray) -> TArray:
    """Elementwise angle (phase) of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.angle(arr))
    else:
        return cast(TArray, torch.angle(arr))


def real(arr: TArray) -> TArray:
    """Elementwise real part of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.real(arr))
    else:
        return cast(TArray, torch.real(arr))


def imag(arr: TArray) -> TArray:
    """Elementwise imaginary part of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.imag(arr))
    else:
        return cast(TArray, torch.imag(arr))


def exp(arr: TArray) -> TArray:
    """Elementwise exponential."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.exp(arr))
    else:
        return cast(TArray, torch.exp(arr))


def maximum(arr: TArray, value: float) -> TArray:
    """Elementwise maximum with a scalar."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.maximum(arr, value))
    else:
        return cast(TArray, torch.maximum(arr, torch.tensor(value)))
