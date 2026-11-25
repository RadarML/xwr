"""Tests for xwr.config."""

import pytest

from xwr.config import XWRConfig
from xwr.radar import AWR1642, AWR1843, AWR2944, AWR2944P, AWRL6844


def _config(device):
    return XWRConfig(
        device=device, frequency=77.0, idle_time=110.0, adc_start_time=4.0,
        ramp_end_time=56.0, tx_start_time=1.0, freq_slope=70.0,
        adc_samples=256, sample_rate=5_000, frame_length=64,
        frame_period=100.0)


@pytest.mark.parametrize("device", [AWR1642, AWR1843])
def test_max_range_iq(device):
    """Complex (IQ) samples: every range bin is unique."""
    cfg = _config(device)
    assert cfg.max_range == pytest.approx(cfg.range_resolution * 256)


@pytest.mark.parametrize("device", [AWR2944, AWR2944P, AWRL6844])
def test_max_range_real(device):
    """Real-only (I) samples: only half of the range bins are unique."""
    cfg = _config(device)
    assert cfg.max_range == pytest.approx(cfg.range_resolution * 128)


def test_max_range_matches_if_bandwidth():
    """max_range = c * f_max / (2 * slope); f_max = fs (IQ) or fs / 2 (I)."""
    c = 299792458
    slope = 70.0 * 1e12  # Hz/s
    fs = 5_000 * 1e3  # Hz
    assert _config(AWR1843).max_range == pytest.approx(c * fs / (2 * slope))
    assert _config(AWR2944).max_range == pytest.approx(
        c * (fs / 2) / (2 * slope))
