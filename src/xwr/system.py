"""High level radar capture system API."""

import logging
import threading
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Generic, Literal, TypeVar, cast, overload

import numpy as np

from .capture import DCA1000EVM, types
from .config import DCAConfig, XWRConfig
from .radar import XWRBase

TRadar = TypeVar("TRadar", bound=XWRBase)


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
            greater than 99%.
        - The ADC is still sampling when the ramp ends.
        - The range-Doppler frame size is greater than 2^14.
        - The number of samples per chirp (i.e., range resolution) or chirps
            per frame (i.e., doppler resolution) is not a power of 2.

    Type Parameters:
        - `TRadar`: radar type (subclass of [`XWRBase`][xwr.radar.])

    Args:
        radar: radar configuration; if `dict`, the key/value pairs are passed
            to `XWRConfig`.
        capture: capture card configuration; if `dict`, the key/value pairs are
            passed to `DCAConfig`.
        name: friendly name for logging; can be default.
        strict: if `True`, raise an error instead of logging a warning if the
            radar configuration contains potentially invalid values.
    """

    def __init__(
        self, *, radar: XWRConfig | dict, capture: DCAConfig | dict,
        name: str = "RadarCapture", strict: bool = False
    ) -> None:
        if isinstance(radar, dict):
            radar = XWRConfig(**radar)
        if isinstance(capture, dict):
            capture = DCAConfig(**capture)

        self.log: logging.Logger = logging.getLogger(name)
        self._check_config(radar, capture)

        self.dca: DCA1000EVM = capture.create()
        self.xwr: TRadar = cast(
            type[TRadar], radar.device_type)(port=radar.port)

        self.config = radar
        self.fps: float = 1000.0 / radar.frame_period
        self.strict = strict

    def _assert(self, cond: bool, desc: str) -> None:
        """Check a condition and log (or raise) a warning if it is not met."""
        if not cond:
            if self.strict:
                raise ValueError(f"Potentially invalid configuration: {desc}")
            self.log.warning(f"Invalid radar configuration: {desc}")
        else:
            self.log.debug(f"Passed check: {desc}")

    def _check_config(self, radar: XWRConfig, capture: DCAConfig) -> None:
        """Check config, and warn if potentially invalid."""
        util = 100 * radar.throughput / capture.throughput
        self.log.info(
            f"Radar/Capture card throughput: {int(radar.throughput / 1e6)} "
            f"Mbps / {int(capture.throughput / 1e6)} Mbps ({util:.1f}%)")
        self._assert(util < 80, f"Network utilization > 80%: {util:.1f}%")

        ratio = capture.socket_buffer / radar.frame_size
        self.log.info("Recv buffer size: {:.2f} frames".format(ratio))
        self._assert(ratio > 2.0,
            f"Recv buffer < 2 frames: {capture.socket_buffer} "
            f"(1 frame = {radar.frame_size})")

        duty_cycle = 100 * radar.frame_time / radar.frame_period
        self.log.info(f"Radar duty cycle: {duty_cycle:.1f}%")
        self._assert(duty_cycle < 99, f"Duty cycle > 99%: {duty_cycle:.1f}%")

        excess = radar.ramp_end_time - radar.adc_start_time - radar.sample_time
        self.log.info(f"Excess ramp time: {excess:.1f}us")
        self._assert(excess >= 0, f"Excess ramp time < 0: {excess:.1f}us")

        frame_size = radar.frame_length * radar.adc_samples
        self.log.info(
            f"Range-Doppler size: {radar.frame_length} x "
            f"{radar.adc_samples} = {frame_size}")
        self._assert(frame_size <= 2**14,
            f"Range-doppler frame size > 2^14: {frame_size}")
        self._assert(radar.frame_length & (radar.frame_length - 1) == 0,
            f"Frame length not a power of 2: {radar.frame_length}")
        self._assert(radar.adc_samples & (radar.adc_samples - 1) == 0,
            f"ADC samples not a power of 2: {radar.adc_samples}")

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
    def qstream(
        self, numpy: Literal[False] = False
    ) -> Queue[types.RadarFrame | None]: ...

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

    def stop(self) -> None:
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
