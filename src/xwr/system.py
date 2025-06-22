"""High level radar capture system API."""

import logging
import threading
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Generic, Literal, Type, TypeVar, cast, overload

import numpy as np

from . import radar as xwr_radar
from .capture import types
from .config import DCAConfig, XWRConfig

TRadar = TypeVar("TRadar", bound=xwr_radar.XWRBase)


class XWRSystem(Generic[TRadar]):
    """Radar capture system with a mmWave Radar and DCA1000EVM.

    !!! info "Known Constraints"

        The `XWRSystem` will check for certain known constraints, and warn if
        these are violated via a logger:

        - Radar data throughput is greater than 80% of the capture card
            theoretical network throughput.
        - Receive buffer size (in the linux networking stack) can hold less
            than 2 full frames.
        - The duty cycle (active frame time / frame period) of the radar is
            greater than 95%.
        - The ADC is still sampling when the ramp ends.

    Type Parameters:
        - `TRadar`: radar type (subclass of [`XWRBase`][xwr.radar.])

    Args:
        radar: radar configuration; if `dict`, the key/value pairs are passed
            to `XWRConfig`.
        capture: capture card configuration; if `dict`, the key/value pairs are
            passed to `DCAConfig`.
        type: radar type; if `str`, the class in [`xwr.radar`][xwr.radar] with
            the corresponding name is used.
        name: friendly name for logging; can be default.
    """

    def __init__(
        self, *, radar: XWRConfig | dict, capture: DCAConfig | dict,
        type: Type[TRadar] | str = "AWR1843",
        name: str = "RadarCapture"
    ) -> None:
        if isinstance(radar, dict):
            radar = XWRConfig(**radar)
        if isinstance(capture, dict):
            capture = DCAConfig(**capture)

        if isinstance(type, str):
            try:
                RadarType = cast(Type[TRadar], getattr(xwr_radar, type))
            except AttributeError:
                raise ValueError(f"Unknown radar type: {type}")
        else:
            RadarType = type

        self.log = logging.getLogger(name)
        self._statistics(radar, capture)

        self.dca = capture.create()
        self.xwr = RadarType(port=radar.port)

        self.config = radar
        self.fps = 1000.0 / radar.frame_period

    def _statistics(self, radar: XWRConfig, capture: DCAConfig) -> None:
        """Compute (and log) statistics, and warn if potentially invalid."""
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

    def stream(self) -> Iterator[types.RadarFrame]:
        """Iterator which yields successive frames.

        !!! note

            `.stream()` does not internally terminate data collection;
            another worker must call [`stop`][..].

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
        self.xwr.setup(**self.config.as_dict())
        self.xwr.start()

        return self.dca.stream(self.config.raw_shape)

    @overload
    def qstream(self, numpy: Literal[True]) -> Queue[np.ndarray | None]: ...

    @overload
    def qstream(self, numpy: Literal[False]) -> Queue[types.RadarFrame | None]:
        ...

    def qstream(
        self, numpy: bool = False
    ) -> Queue[types.RadarFrame | None] | Queue[np.ndarray | None]:
        """Read into a queue from a threaded worker.

        The threaded worker is run with `daemon=True`. Like [`stream`][..],
        `.qstream()` also relies on another worker to trigger [`stop`][..].

        !!! note

            If a `TimeoutError` is received (e.g. after `.stop()`), the
            error is caught, and the stream is halted.

        Args:
            numpy: yield a numpy array instead of a `RadarFrame`.

        Returns:
            A queue of `RadarFrame` (or np.ndarray) read by the capture card.
                When the stream terminates, `None` is written to the queue.
        """
        out: Queue[types.RadarFrame | None] | Queue[np.ndarray | None] = Queue()

        def worker():
            try:
                for frame in self.stream():
                    if numpy:
                        if frame is not None:
                            frame = np.frombuffer(
                                frame.data, dtype=np.int16
                            ).reshape(*self.config.raw_shape)
                        # Type inference can't figure out this overload check
                        cast(Queue[np.ndarray | None], out).put(frame)
                    else:
                        out.put(frame)
            except TimeoutError:
                pass
            out.put(None)

        threading.Thread(target=worker, daemon=True).start()
        return out

    @overload
    def dstream(self, numpy: Literal[True]) -> Iterator[np.ndarray]: ...

    @overload
    def dstream(self, numpy: Literal[False]) -> Iterator[types.RadarFrame]: ...

    def dstream(
        self, numpy: bool = False
    ) -> Iterator[types.RadarFrame | np.ndarray]:
        """Stream frames, dropping any frames if the consumer gets behind.

        Args:
            numpy: yield a numpy array instead of a `RadarFrame`.

        Yields:
            Read frames; the iterator terminates when the capture card stream
                times out.
        """
        def drop_frames(q):
            dropped = 0
            latest = q.get(block=True)

            while True:
                try:
                    latest = q.get_nowait()
                    dropped += 1
                except Empty:
                    return latest, dropped

        q = self.qstream(numpy=numpy)
        while True:
            frame, dropped = drop_frames(q)
            if dropped > 0:
                self.log.warning(f"Dropped {dropped} frames.")
            if frame is None:
                break
            else:
                yield frame

    def stop(self):
        """Stop by halting the capture card and reboot the radar.

        In testing, we found that the radar may ignore commands if the frame
        timings are too tight, which prevents a soft reset. We simply reboot
        the radar via the capture card instead.

        !!! warning

            If you fail to `.stop()` the system before exiting, the radar may
            become non-responsive, and require a power cycle.
        """
        self.dca.stop()
        self.dca.reset_ar_device()
