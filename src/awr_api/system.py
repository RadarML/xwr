"""Full radar capture system API."""

import logging
import threading
from queue import Queue

from beartype.typing import Iterator

from .awr_api import AWR1843
from .config import CaptureConfig, RadarConfig
from .dca_api import DCA1000EVM
from .dca_types import RadarFrame


class AWRSystem:
    """Radar capture system with a AWR1843Boost and DCA1000EVM.

    Args:
        radar: radar configuration; can be passed as a dict or a
            :py:class:`.RadarConfig`.
        capture: capture card configuration; can be passed as a dict or a
            :py:class:`.CaptureConfig`.
        name: friendly name for logging; can be default.
    """

    def __init__(
        self, *, radar: RadarConfig | dict, capture: CaptureConfig | dict,
        name: str = "RadarCapture"
    ) -> None:
        if isinstance(radar, dict):
            radar = RadarConfig(**radar)
        if isinstance(capture, dict):
            capture = CaptureConfig(**capture)

        self.log = logging.getLogger(name)
        self._statistics(radar, capture)

        self.dca = DCA1000EVM(
            sys_ip=capture.sys_ip, fpga_ip=capture.fpga_ip,
            data_port=capture.data_port, config_port=capture.config_port,
            timeout=capture.timeout, socket_buffer=capture.socket_buffer)
        self.dca.setup(delay=capture.delay)
        self.awr = AWR1843(port=radar.port)

        self.config = radar
        self.fps = 1000.0 / radar.frame_period

    def _statistics(self, radar: RadarConfig, capture: CaptureConfig) -> None:
        """Compute statistics, and warn if potentially invalid."""
        # Network utilization
        util = radar.throughput / capture.throughput
        self.log.info("Radar/Capture card: {} Mbps / {} Mbps ({:.1f}%)".format(
            int(radar.throughput / 1e6), int(capture.throughput / 1e6),
            util * 100))
        if radar.throughput > capture.throughput * 0.8:
            self.log.warning(
                "Network utilization > 80%: {:.1f}%".format(util * 100))

        # Buffer size
        ratio = capture.socket_buffer / radar.frame_size
        self.log.info("Recv buffer size: {:.2f} frames".format(ratio))
        if ratio < 2.0:
            self.log.warning("Recv buffer < 2 frames: {} (1 frame = {})".format(
                capture.socket_buffer, radar.frame_size))

        # Radar duty cycle
        duty_cycle = radar.frame_time / radar.frame_period
        self.log.info("Radar duty cycle: {:.1f}%".format(duty_cycle * 100))
        if duty_cycle > 0.95:
            self.log.warning(
                "Radar duty cycle > 95%: {:.1f}%".format(duty_cycle * 100))

        # Ramp timing
        excess = (
            radar.ramp_end_time - radar.adc_start_time - radar.sample_time)
        self.log.info("Excess ramp time: {:.1f}us".format(excess))
        if excess < 0:
            self.log.warning("Excess ramp time < 0: {:.1f}us".format(excess))

    def stream(self) -> Iterator[RadarFrame]:
        """Frame iterator.

        Note that `.stream()` does not internally terminate data collection;
        another worker must call :py:meth:`.AWRSystem.stop`.

        Yields:
            Read frames; the iterator terminates when the capture card stream
            times out.
        """
        # send a "stop" command in case the capture card is still running
        self.dca.stop()
        # reboot radar in case it is stuck
        self.dca.reset_ar_device()
        # clear buffer from possible previous data collection
        # (will mess up byte count indices if we don't)
        self.dca.flush()

        # start capture card & radar
        self.dca.start()
        self.awr.setup(**self.config.as_dict())
        self.awr.start()

        return self.dca.stream(self.config.raw_shape)

    def qstream(self) -> Queue[RadarFrame | None]:
        """Read into a queue from a threaded worker.

        The threaded worker is run with `daemon=True`. Like
        :py:meth:`.AWRSystem.stream`, `.qstream()` also relies on another
        worker to trigger :py:meth:`.AWRSystem.stop`.

        Returns:
            A queue of :py:class:`.RadarFrame` read by the capture card. When
            the stream terminates, `None` is written to the queue.
        """
        out: Queue[RadarFrame | None] = Queue()

        def worker():
            for frame in self.stream():
                out.put(frame)
            out.put(None)

        threading.Thread(target=worker, daemon=True).start()
        return out

    def stop(self):
        """Stop by halting the capture card and reboot the radar.

        In testing, we found that the radar may ignore commands if the frame
        timings are too tight, which prevents a soft-reset. We simply reboot
        the radar via the capture card instead.
        """
        self.dca.stop()
        self.dca.reset_ar_device()
