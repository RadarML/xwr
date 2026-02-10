"""NN Tests."""

import numpy as np
import pytest
import torch

from xwr.nn import representations, utils


def test_utils_conversions():
    """Test utils conversion functions."""
    # Test IQ <-> MSC
    iq = torch.randn(10, 2)
    msc = utils.msc_from_iq(iq)
    assert msc.shape == (10, 3)
    iq_recon = utils.iq_from_msc(msc)
    assert torch.allclose(iq, iq_recon, atol=1e-5)

    # Test IQ <-> MP
    iq = torch.randn(10, 2)
    mp = utils.mp_from_iq(iq)
    assert mp.shape == (10, 2)
    iq_recon = utils.iq_from_mp(mp)
    assert torch.allclose(iq, iq_recon, atol=1e-5)


def test_utils_resize():
    """Test utils resize function."""
    # Create a dummy spectrum: (Batch, Doppler, Azimuth, Range)
    # Smaller size for CI: (1, 8, 2, 16)
    spectrum = np.random.randn(1, 8, 2, 16).astype(np.float32)

    # Test no-op
    out = utils.resize(spectrum)
    assert out.shape == (1, 8, 2, 16)

    # Test downsample range
    out = utils.resize(spectrum, range_scale=0.5)
    assert out.shape == (1, 8, 2, 16)

    # Test upsample range
    out = utils.resize(spectrum, range_scale=2.0)
    assert out.shape == (1, 8, 2, 16)

    # Test downsample Doppler
    out = utils.resize(spectrum, speed_scale=0.5)
    assert out.shape == (1, 8, 2, 16)

    # Test upsample Doppler
    out = utils.resize(spectrum, speed_scale=2.0)
    assert out.shape == (1, 8, 2, 16)

    # Test torch input
    spectrum_t = torch.from_numpy(spectrum)
    out_t = utils.resize(spectrum_t, range_scale=0.5, speed_scale=0.5)
    assert isinstance(out_t, torch.Tensor)
    assert out_t.shape == (1, 8, 2, 16)


@pytest.mark.parametrize("RepClass, out_channels", [
    (representations.Magnitude, 1),
    (representations.PhaseAngle, 2),
    (representations.PhaseVec, 3),
    (representations.ComplexParts, 2),
])
def test_representations(RepClass, out_channels):
    """Test representation classes."""
    # Input spectrum: (Batch, Doppler, El, Az, Rng) -> Complex
    # Smaller shape for CI: (1, 4, 2, 2, 4)
    # Note: Doppler must be >= 4 for speed_scale=0.5 to result in >0 dim (due to 2*(N//2) logic)
    shape = (1, 4, 2, 2, 4)
    rng = np.random.default_rng()
    data_np = (
        rng.random(size=shape) + 1j * rng.random(size=shape)
    ).astype(np.complex64)
    data_torch = torch.from_numpy(data_np)

    rep = RepClass()

    # Test Numpy
    out_np = rep(data_np)
    # Expected output: (1, 4, 2, 2, 4, out_channels)
    expected_shape = shape + (out_channels,)
    assert out_np.shape == expected_shape
    assert isinstance(out_np, np.ndarray)

    # Test Torch
    out_torch = rep(data_torch)
    assert out_torch.shape == expected_shape
    assert isinstance(out_torch, torch.Tensor)

    # Test augmentation hooks (basic check that they don't crash)
    aug = {
        "azimuth_flip": True,
        "doppler_flip": True,
        "radar_scale": 1.1,
        "radar_phase": 0.1,
        "range_scale": 0.5,
        "speed_scale": 0.5
    }
    # Note: resize inside representation preserves shape (crops/pads)
    # Output: (1, 4, 2, 2, 4, out_channels)
    expected_aug_shape = (1, 4, 2, 2, 4, out_channels)
    
    out_aug = rep(data_np, aug=aug)
    assert out_aug.shape == expected_aug_shape
