"""Resize range-Doppler spectrum."""

import torch
from jaxtyping import Float32
from torch import Tensor
from torchvision import transforms


def _wrap(
    x: Float32[Tensor, "T D A ... R"], width: int
) -> Float32[Tensor, "T D_crop A ... R"]:
    """Wrap doppler velocities."""
    i_left = x.shape[1] // 2 - width // 2
    i_right = x.shape[1] // 2 + width // 2

    left = x[:, :i_left]
    center = x[:, i_left:i_right]
    right = x[:, i_right:]

    center[:, :right.shape[1]] += right
    center[:, -left.shape[1]:] += left

    return center


def _resize(
    spectrum: Float32[Tensor, "T D *A R"],
    nd: int, nr: int
) -> Float32[Tensor, "T D2 *A R2"]:
    """Resize spectrum to the target range/doppler."""
    T, _, *A, _ = spectrum.shape

    # The leading T axis is transparently vectorized by Resize.
    # Note that we also have to do this reshape dance since Resize
    # only allows a maximum of 2 leading dimensions for some reason.
    spec_t: Float32[Tensor, "T ... R D"] = torch.moveaxis(spectrum, 1, -1)
    spec_flat_t: Float32[Tensor, "X R D"]
    spec_flat_t = spec_t.reshape(-1, *spec_t.shape[-2:])

    resized_flat_t: Float32[Tensor, "X R2 D2"] = transforms.Resize(
        (nr, nd),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True
    )(spec_flat_t)

    resized: Float32[Tensor, "T D2 *A R2"]
    resized = torch.moveaxis(resized_flat_t.reshape(T, *A, nr, nd), -1, 1)
    return resized


def resize(
    spectrum: Float32[Tensor, "T D *A R"],
    range_scale: float = 1.0, speed_scale: float = 1.0,
) -> Float32[Tensor, "T D *A R"]:
    """Resize range-Doppler spectrum.

    !!! note

        We use `torchvision.transforms.Resize`, which requires a
        round-trip through a (cpu) `Tensor`. From some limited testing,
        this appears to be the most performant image resizing which
        supports antialiasing, with `skimage.transform.resize` being
        particularly slow.

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

    if range_out_dim != Nr or speed_out_dim != Nd:
        resized = _resize(spectrum, nd=speed_out_dim, nr=range_out_dim)

        # Upsample -> crop
        if range_out_dim >= Nr:
            resized = resized[..., :Nr]
        # Downsample -> zero pad far ranges (high indices)
        else:
            pad = torch.zeros(
                (*spectrum.shape[:-1], Nr - range_out_dim),
                dtype=resized.dtype, device=resized.device)
            resized = torch.concatenate([resized, pad], dim=-1)

        # Upsample -> wrap
        if speed_out_dim > spectrum.shape[1]:
            resized = _wrap(resized, Nd)
        # Downsample -> zero pad high velocities (low and high indices)
        else:
            pad = torch.zeros(
                (T, (Nd - speed_out_dim) // 2, *spectrum.shape[2:]),
                dtype=resized.dtype, device=resized.device)
            resized = torch.concatenate((pad, resized, pad), dim=1)

        spectrum = resized

    return spectrum
