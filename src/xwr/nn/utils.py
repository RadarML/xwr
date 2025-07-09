"""Resize range-Doppler spectrum."""

import numpy as np
import torch
from jaxtyping import Float32
from torchvision import transforms


def _wrap(
    x: Float32[np.ndarray, "T D A ... R"], width: int
) -> Float32[np.ndarray, "T D_crop A ... R"]:
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
    spectrum: Float32[np.ndarray, "T D *A R"],
    nd: int, nr: int
) -> Float32[np.ndarray, "T D2 *A R2"]:
    """Resize spectrum to the target range/doppler."""
    T, _, *A, _ = spectrum.shape

    # The leading T axis is transparently vectorized by Resize.
    # Note that we also have to do this reshape dance since Resize
    # only allows a maximum of 2 leading dimensions for some reason.
    spec_t: Float32[torch.Tensor, "T ... R D"]
    spec_t = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(spectrum, 1, -1)))
    spec_flat_t: Float32[torch.Tensor, "X R D"]
    spec_flat_t = spec_t.reshape(-1, *spec_t.shape[-2:])

    resized_flat_t: Float32[torch.Tensor, "X R2 D2"] = transforms.Resize(
        (nr, nd),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True
    )(spec_flat_t)

    resized: Float32[np.ndarray, "T D2 *A R2"]
    resized = np.moveaxis(resized_flat_t.reshape(T, *A, nr, nd).numpy(), -1, 1)
    return resized


def resize(
    spectrum: Float32[np.ndarray, "T D *A R"],
    range_scale: float = 1.0, speed_scale: float = 1.0,
) -> Float32[np.ndarray, "T D *A R"]:
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
            pad = np.zeros(
                (*spectrum.shape[:-1], Nr - range_out_dim), dtype=np.float32)
            resized = np.concatenate([resized, pad], axis=-1)

        # Upsample -> wrap
        if speed_out_dim > spectrum.shape[1]:
            resized = _wrap(resized, Nd)
        # Downsample -> zero pad high velocities (low and high indices)
        else:
            pad = np.zeros(
                (T, (Nd - speed_out_dim) // 2, *spectrum.shape[2:]),
                dtype=np.float32)
            resized = np.concatenate((pad, resized, pad), axis=1)

        spectrum = resized

    return spectrum
