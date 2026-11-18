"""Internal backend for pytorch / numpy interop."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch


def concatenate[TArray: np.ndarray | torch.Tensor](arrays: Sequence[TArray], axis: int) -> TArray:
    """Concatenate arrays."""
    if isinstance(arrays[0], np.ndarray):
        return cast(TArray, np.concatenate(
            cast(Sequence[np.ndarray], arrays), axis=axis))
    else:
        return cast(TArray, torch.concatenate(
            cast(tuple[torch.Tensor], tuple(arrays)), dim=axis))


def stack[TArray: np.ndarray | torch.Tensor](arrays: Sequence[TArray], axis: int) -> TArray:
    """Stack arrays."""
    if isinstance(arrays[0], np.ndarray):
        return cast(TArray, np.stack(
            cast(Sequence[np.ndarray], arrays), axis=axis))
    else:
        return cast(TArray, torch.stack(
            cast(tuple[torch.Tensor], tuple(arrays)), dim=axis))


def flip[TArray: np.ndarray | torch.Tensor](arr: TArray, axis: int = 0) -> TArray:
    """Flip array."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.flip(arr, axis=axis))
    else:
        return cast(TArray, torch.flip(arr, dims=(axis,)))


def log[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise log."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.log(arr))
    else:
        return cast(TArray, torch.log(arr))


def sqrt[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise square root."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.sqrt(arr))
    else:
        return cast(TArray, torch.sqrt(arr))


def abs[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise absolute value."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.abs(arr))
    else:
        return cast(TArray, torch.abs(arr))


def angle[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise angle (phase) of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.angle(arr))
    else:
        return cast(TArray, torch.angle(arr))


def real[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise real part of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.real(arr))
    else:
        return cast(TArray, torch.real(arr))


def imag[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise imaginary part of complex numbers."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.imag(arr))
    else:
        return cast(TArray, torch.imag(arr))


def exp[TArray: np.ndarray | torch.Tensor](arr: TArray) -> TArray:
    """Elementwise exponential."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.exp(arr))
    else:
        return cast(TArray, torch.exp(arr))


def maximum[TArray: np.ndarray | torch.Tensor](arr: TArray, value: float) -> TArray:
    """Elementwise maximum with a scalar."""
    if isinstance(arr, np.ndarray):
        return cast(TArray, np.maximum(arr, value))
    else:
        return cast(TArray, torch.maximum(arr, torch.tensor(value)))
