"""Test radar capture."""

import os
import socket
import threading
import time
import warnings

import pytest
import yaml

import xwr
from xwr.capture import DCA1000EVM


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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_dca_close_releases_ports():
    """Closing the capture card releases its ports for a new instance."""
    config_port, data_port = _free_port(), _free_port()

    def make():
        return DCA1000EVM(
            sys_ip="127.0.0.1", config_port=config_port, data_port=data_port)

    with make():
        pass

    # Would raise `OSError: [Errno 98] Address already in use` if not closed.
    dca = make()
    dca.close()
    dca.close()


def _local_dca(timeout: float = 0.2) -> DCA1000EVM:
    return DCA1000EVM(
        sys_ip="127.0.0.1", config_port=_free_port(), data_port=_free_port(),
        timeout=timeout)


def test_dca_stream_releases_lock():
    """A stream that ends (no data within `timeout`) lets the next one run."""
    with _local_dca() as dca:
        assert list(dca.stream([4])) == []
        assert list(dca.stream([4])) == []


def test_dca_close_waits_for_stream():
    """`close()` waits for a stream running in another thread to exit."""
    dca = _local_dca()
    errors = []

    def reader():
        try:
            list(dca.stream([4]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)  # reader is now waiting for packets
    dca.close()
    assert not t.is_alive()
    assert errors == []


def test_dca_close_raises_if_stream_still_running():
    """`close()` refuses to close under an active reader, and can be retried."""
    dca = _local_dca()
    dca._lock.acquire()  # simulate a stream that never ends
    with pytest.raises(RuntimeError):
        dca.close()
    assert dca.data_socket.fileno() != -1  # nothing was closed

    dca._lock.release()  # the stream ends
    dca.close()
    assert dca.data_socket.fileno() == -1


def test_capture():
    """Test basic capture functionality."""
    system = _make_system()
    if system is None:
        return

    with system:
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

    with system:
        q = system.qstream()

        frame = q.get(timeout=5.0)
        assert frame is not None
        assert len(frame.data) == system.config.frame_size
        system.stop()

        for frame in system.dstream(numpy=True):
            assert frame.shape == system.config.raw_shape
            system.stop()
            break


def test_sequential_systems():
    """A new system can be created in the same process after closing."""
    for _ in range(2):
        system = _make_system()
        if system is None:
            return

        with system:
            q = system.qstream()
            frame = q.get(timeout=5.0)
            assert frame is not None
            assert len(frame.data) == system.config.frame_size


def test_new_stream_ends_previous():
    """Starting a new stream ends the previous one."""
    system = _make_system()
    if system is None:
        return

    with system:
        q1 = system.qstream()
        assert q1.get(timeout=5.0) is not None

        q2 = system.qstream()
        while q1.get(timeout=5.0) is not None:
            pass  # previous stream ended
        assert q2.get(timeout=5.0) is not None
