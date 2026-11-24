"""High level radar capture system API."""

import logging
import threading
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Generic, Literal, TypeVar, cast, overload

import numpy as np

from .capture import DCA1000EVM, DCAError, types
from .config import DCAConfig, XWRConfig
from .constraints import check_config
from .radar import XWRBase

TRadar = TypeVar("TRadar", bound=XWRBase)


class XWRSystem(Generic[TRadar]):
    """Radar capture system with a mmWave Radar and DCA1000EVM.

    This high-level API provides three different ways to read frames, each
    with different performance characteristics:

    [`stream`][.]: debugging, user-managed real-time systems; *no buffering*
    :   A simple iterator which yields successive frames. No buffering is done,
        which may lead to dropped packets if the consumer is too slow.

    [`qstream`][.]: data collection; *no dropped packets, but unbounded latency*
    :   A queue-based interface, where a separate thread reads frames into a
        queue. This prevents dropped packets, but may lead to unbounded latency
        if the consumer is too slow.

    [`dstream`][.]: real-time demos; *no dropped packets, but may skip frames*
    :   A "drop frame" iterator, which yields the most recent frame, skipping
        any frames received while the consumer is processing a frame.

    Starting a new stream (with any of these methods) ends the previous one.

    !!! info "Known Constraints"

        The `XWRSystem` will check for known constraints on initialization,
        and warn or raise if any are violated. See
        [`xwr.constraints`][xwr.constraints] for the full list.

    !!! warning "Releasing resources"

        Use the `XWRSystem` as a context manager, or call [`close`][.] when
        done; otherwise, the radar may become non-responsive and require a
        power cycle, and the capture card ports stay bound until the object is
        garbage collected.

        ```python
        with xwr.XWRSystem(**cfg) as awr:
            for frame in awr.dstream():
                ...
        ```

    Type Parameters:
        TRadar: radar type (subclass of [`XWRBase`][xwr.radar.])

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
        self.strict = strict
        self._check_config(radar, capture)

        self._closed = False

        self.dca: DCA1000EVM = capture.create()
        try:
            self.xwr: TRadar = cast(
                type[TRadar], radar.device_type)(port=radar.port)
        except BaseException:
            self.dca.close()
            raise

        self.config = radar
        self.fps: float = 1000.0 / radar.frame_period

    def _check_config(self, radar: XWRConfig, capture: DCAConfig) -> None:
        """Check config, and raise if strict mode is enabled and invalid."""
        for r in check_config(radar, capture):
            if r.passed is False and self.strict:
                raise ValueError(
                    f"Invalid configuration | {r.constraint.__name__}: {r.detail}")

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
        # (this also ends any previous stream)
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

        The radar is configured and started in the calling thread (so this
        call blocks until the radar has started); frames are then read by a
        worker thread run with `daemon=True`. Like [`stream`][..],
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
        frames = self.stream()
        out: Queue[types.RadarFrame | None] | Queue[np.ndarray | None] = Queue()

        def worker():
            try:
                for frame in frames:
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
            finally:
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

        The system can be restarted after `stop()` by calling any of the
        streaming methods again; use [`close`][..] to release it instead.
        """
        self.dca.stop()
        self.dca.reset_ar_device()

    def close(self) -> None:
        """Stop the system, and release the capture card and radar port.

        - Waits for any running stream to exit after stopping the capture card.
        - Unlike [`stop`][..], the system cannot be used after `close()`.
        - Calling `close()` again does nothing.

        Raises:
            RuntimeError: if a stream is still running; see
                [`DCA1000EVM.close`][xwr.capture.DCA1000EVM.close]. Nothing is
                released in this case, so `close()` can be retried.
        """
        if self._closed:
            return

        try:
            self.stop()
        except (TimeoutError, DCAError) as e:
            self.log.warning(f"Could not stop cleanly before closing: {e!r}")

        self.dca.close()
        self.xwr.close()
        self._closed = True

    def __enter__(self) -> "XWRSystem[TRadar]":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
