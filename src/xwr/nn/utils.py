"""Resize range-Doppler spectrum."""

from typing import cast

import numpy as np
import torch
from jaxtyping import Float, Float32
from torch import Tensor
from torchvision import transforms

from . import backend
from .backend import TArray


def msc_from_iq(
    iq: Float[Tensor, "... 2"], eps: float = 1e-6
) -> Float[Tensor, "... 3"]:
    """Convert `(real, imaginary)` to `(magnitude, sin, cosine)`.

    !!! warning

        If `iq` is close to zero, the sin and cosine (i.e., normalized
        real/imaginary parts) may not necessarily have a norm of `1.0`.

    Args:
        iq: input complex data, with a trailing `[real, complex]` channel.
        eps: small value to avoid division by zero.

    Returns:
        3-channel `(magnitude, sin, cosine)` representation.
    """
    magnitude = (iq[..., 0]**2 + iq[..., 1]**2)**0.5
    return torch.stack([
        magnitude,
        iq[..., 0] / torch.clamp(magnitude, min=eps),
        iq[..., 1] / torch.clamp(magnitude, min=eps)
    ], dim=-1)


def iq_from_msc(
    msc: Float[Tensor, "... 3"]
) -> Float[Tensor, "... 2"]:
    """Convert `(real, imaginary)` to `(magnitude, sin, cosine)`.

    !!! info

        The sin and cosine (i.e., normalized real/imaginary parts) are not
        required to have a norm of `1.0`.

    Args:
        msc: input 3-channel `(magnitude, sin, cosine)` representation.

    Returns:
        2-channel `(real, imaginary)` representation.
    """
    magnitude = msc[..., 0]
    norm = (msc[..., 1]**2 + msc[..., 2]**2)**0.5
    return torch.stack([
        magnitude * msc[..., 1] / norm,
        magnitude * msc[..., 2] / norm
    ], dim=-1)


def mp_from_iq(
    iq: Float[Tensor, "... 2"]
) -> Float[Tensor, "... 2"]:
    """Convert `(real, imaginary)` to `(magnitude, phase angle)`.

    Args:
        iq: input 2-channel `(real, imaginary)` representation.

    Returns:
        2-channel `(magnitude, phase angle)` representation.
    """
    magnitude = (iq[..., 0]**2 + iq[..., 1]**2)**0.5
    phase = torch.atan2(iq[..., 1], iq[..., 0])
    return torch.stack([magnitude, phase], dim=-1)


def iq_from_mp(
    mp: Float[Tensor, "... 2"]
) -> Float[Tensor, "... 2"]:
    """Convert `(magnitude, phase angle)` to `(real, imaginary)`.

    Args:
        mp: input 2-channel `(magnitude, phase angle)` representation.

    Returns:
        2-channel `(real, imaginary)` representation.
    """
    return torch.stack([
        mp[..., 0] * torch.cos(mp[..., 1]),
        mp[..., 0] * torch.sin(mp[..., 1])
    ], dim=-1)


def _wrap(
    x: Float32[TArray, "T D A ... R"], width: int
) -> Float32[TArray, "T D_crop A ... R"]:
    """Wrap doppler velocities."""
    i_left = x.shape[1] // 2 - width // 2
    i_right = x.shape[1] // 2 + width // 2

    left = x[:, :i_left]
    center = x[:, i_left:i_right]
    right = x[:, i_right:]

    center[:, :right.shape[1]] += right  # type: ignore
    center[:, -left.shape[1]:] += left  # type: ignore

    return cast(TArray, center)


def _resize(
    spectrum: Float32[TArray, "T D *A R"],
    nd: int, nr: int
) -> Float32[TArray, "T D2 *A R2"]:
    """Resize spectrum to the target range/doppler."""
    T, _, *A, _ = spectrum.shape

    # The leading T axis is transparently vectorized by Resize.
    # Note that we also have to do this reshape dance since Resize
    # only allows a maximum of 2 leading dimensions for some reason.
    spec_t: Float32[torch.Tensor, "T ... R D"]
    if isinstance(spectrum, torch.Tensor):
        spec_t = torch.moveaxis(spectrum, 1, -1)
    else:
        spec_t = torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(spectrum, 1, -1)))

    spec_flat_t: Float32[torch.Tensor, "X R D"]
    spec_flat_t = spec_t.reshape(-1, *spec_t.shape[-2:])

    resized_flat_t: Float32[torch.Tensor, "X R2 D2"] = transforms.Resize(
        (nr, nd),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True
    )(spec_flat_t)

    if isinstance(spectrum, torch.Tensor):
        return cast(TArray, torch.moveaxis(
            resized_flat_t.reshape(T, *A, nr, nd), -1, 1))
    else:
        return cast(TArray, np.moveaxis(
            resized_flat_t.reshape(T, *A, nr, nd).numpy(), -1, 1))


def _zeros(shape: tuple[int, ...], like: TArray) -> TArray:
    """Get zeros array with the correct backend, dtype, and device."""
    if isinstance(like, np.ndarray):
        return np.zeros(shape, dtype=like.dtype)
    else:
        return torch.zeros(shape, dtype=like.dtype, device=like.device)


def resize(
    spectrum: Float32[TArray, "T D *A R"],
    range_scale: float = 1.0, speed_scale: float = 1.0,
) -> Float32[TArray, "T D *A R"]:
    """Resize range-Doppler spectrum.

    !!! note

        We use `torchvision.transforms.Resize`, which requires a
        round-trip through a (cpu) `Tensor` for numpy arrays. From some limited
        testing, this appears to be the most performant image resizing which
        supports antialiasing, with `skimage.transform.resize` being
        particularly slow.

    !!! warning

        `resize` requires the following:

        - The doppler resolution is always even.
        - `speed_scale * n_doppler >= 2`.
        - `range_scale * n_range >= 1`.

    Type Parameters:
        - `TArray`: array type; `np.ndarray` or `torch.Tensor`.

    Args:
        spectrum: input spectrum as a real channel; should be output by one of
            the [`xwr.rsp.numpy`][xwr.rsp.numpy] classes.
        range_scale: scale factor for the range dimension; crops if greater
            than 1.0, and zero-pads if less than 1.0.
        speed_scale: scale factor for the Doppler dimension; wraps if greater
            than 1.0, and zero-pads if less than 1.0.
    """
    T, Nd, *A, Nr = spectrum.shape
    range_out_dim = int(range_scale * Nr)
    speed_out_dim = 2 * (int(speed_scale * Nd) // 2)

    assert Nd % 2 == 0, f"Doppler dim {Nd} must be even."
    assert int(speed_out_dim) >= 2, (
        f"Doppler dim {Nd} too small for scale {speed_scale}")
    assert int(range_out_dim) >= 1, (
        f"Range dim {Nr} too small for scale {range_scale}")

    if range_out_dim != Nr or speed_out_dim != Nd:
        resized = _resize(spectrum, nd=speed_out_dim, nr=range_out_dim)

        # Upsample -> crop
        if range_out_dim >= Nr:
            resized = resized[..., :Nr]
        # Downsample -> zero pad far ranges (high indices)
        else:
            pad = _zeros(
                (*resized.shape[:-1], Nr - range_out_dim), like=resized)
            resized = backend.concatenate([resized, pad], axis=-1)

        # Upsample -> wrap
        if speed_out_dim > spectrum.shape[1]:
            resized = _wrap(resized, Nd)
        # Downsample -> zero pad high velocities (low and high indices)
        else:
            pad = _zeros(
                (T, (Nd - speed_out_dim) // 2, *spectrum.shape[2:]),
                like=resized)
            resized = backend.concatenate([pad, resized, pad], axis=1)

        spectrum = cast(TArray, resized)

    return spectrum
