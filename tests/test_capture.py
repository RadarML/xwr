"""Test radar capture."""

import os
import warnings

import yaml

import xwr


def _make_system():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    device = os.environ.get("XWR_DEVICE")
    if device is None:
        warnings.warn(
            "XWR_DEVICE is not set; all data capture tests will be skipped.")
        return None

    cfg['radar']['device'] = device
    return xwr.XWRSystem(**cfg)


def test_capture():
    """Test basic capture functionality."""
    system = _make_system()
    if system is None:
        return

    frames = []
    for frame in system.stream():
        frames.append(frame)

        if len(frames) >= 3:
            system.stop()
            break

    for frame in frames:
        assert len(frame.data) == system.config.frame_size


def test_stream():
    """Test data streaming (to queue and as numpy arrays)."""
    system = _make_system()
    if system is None:
        return

    q = system.qstream()

    frame = q.get(timeout=5.0)
    assert frame is not None
    assert len(frame.data) == system.config.frame_size
    system.stop()

    for frame in system.dstream(numpy=True):
        assert frame.shape == system.config.raw_shape
        system.stop()
        break
