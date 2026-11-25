"""Tests for xwr.radar command generation (no hardware required)."""

import logging
from unittest.mock import MagicMock

import pytest

from xwr.radar import AWRL6888


def _mock_radar(cls):
    """Create a radar interface which records commands instead of sending."""
    radar = object.__new__(cls)
    radar.log = logging.getLogger("xwr/test")
    radar.port = MagicMock()
    radar._wait_for_response = MagicMock(return_value=bytearray())
    radar.commands = []
    radar.send = lambda cmd, timeout=10.0: radar.commands.append(cmd)
    return radar


def _command(commands: list[str], name: str) -> list[str]:
    matches = [c.split() for c in commands if c.split()[0] == name]
    assert len(matches) == 1, f"Expected one {name}, got {matches}"
    return matches[0][1:]


def test_awrl6888_setup():
    """Check the AWRL6888 setup command sequence."""
    radar = _mock_radar(AWRL6888)
    radar.setup(frame_length=32, idle_time=20.0, ramp_end_time=36.0)
    cmds = radar.commands

    assert _command(cmds, "channelCfg") == ["255", "255", "0"]
    # 200 MHz / 10 Msps = 20; 5 us * 10 Msps = 50 skip samples
    assert _command(cmds, "chirpComnCfg")[0] == "20"
    assert _command(cmds, "chirpTimingCfg")[1] == "50"
    # Single burst of 16 * 32 chirps; burst period = 512 * (20 + 36) us plus
    # the 135 us minimum burst idle time.
    assert _command(cmds, "frameCfg") == [
        "512", "0", "28807.0", "1", "100.0", "0"]
    # The firmware requires the full antenna geometry and 8x8 calibration.
    assert len(_command(cmds, "antGeometryTX")) == 16
    assert len(_command(cmds, "antGeometryRX")) == 16
    assert len(_command(cmds, "antGeometryDist")) == 2
    assert len(_command(cmds, "compRangeBiasAndRxChanPhase")) == 1 + 64 * 2
    assert _command(cmds, "adcLogging") == ["1"]
    # Radar is stopped before being configured.
    assert cmds[0] == "sensorStop 0"


def test_awrl6888_setup_invalid():
    """Check that unsupported AWRL6888 settings are rejected."""
    radar = _mock_radar(AWRL6888)
    # 200 MHz / 7 Msps is not a supported divisor.
    with pytest.raises(ValueError):
        radar.setup(sample_rate=7000)
    # 12 us * 10 Msps = 120 > 103 skip samples.
    with pytest.raises(ValueError):
        radar.setup(adc_start_time=12.0)
    assert radar.commands == []


def test_awrl6888_chunked_write():
    """Check that long AWRL6888 commands are written in chunks."""
    radar = _mock_radar(AWRL6888)
    radar._write(b"x" * 40)
    sizes = [len(c.args[0]) for c in radar.port.write.call_args_list]
    assert sizes == [16, 16, 8]
