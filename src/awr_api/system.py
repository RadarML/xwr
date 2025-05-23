"""High level radar capture system API."""

import logging
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Iterator, Literal, cast, overload

import numpy as np

from .capture import DCA1000EVM, defines, types
from .radar import AWR1843

SPEED_OF_LIGHT = 299792458
"""Speed of light, in m/s."""

@dataclass
class AWRConfig:
    """Radar configuration.

    The [TI mmWave sensing estimator](
    https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.4.0/)
    may be helpful for creating a configuration.

    Attributes:
        port: Control serial port (usually `/dev/ttyACM0`).
        frequency: base frequency, in GHz.
        idle_time: radar timing parameters; in microseconds.
        adc_start_time: radar timing parameters; in microseconds.
        ramp_end_time: radar timing parameters; in microseconds.
        tx_start_time: radar timing parameters; in microseconds.
        freq_slope: chirp slope, in MHz/us.
        adc_samples: number of samples per chirp.
        sample_rate: ADC sampling rate, in KHz.
        frame_length: number of chirps per TX antenna per frame.
        frame_period: periodicity of frames, in ms.
        num_tx: number of TX antenna; 3 for the AWR1843.
        num_rx: number of RX antenna; 4 for the AWR1843.
    """

    frequency: float
    idle_time: float
    adc_start_time: float
    ramp_end_time: float
    tx_start_time: float
    freq_slope: float
    adc_samples: int
    sample_rate: int
    frame_length: int
    frame_period: float
    port: str = "/dev/ttyACM0"
    num_tx: int = 2
    num_rx: int = 4

    @property
    def shape(self) -> list[int]:
        """Radar data cube shape."""
        return [
            self.frame_length, self.num_tx, self.num_rx, self.adc_samples]

    @property
    def raw_shape(self) -> list[int]:
        """Radar IIQQ data shape."""
        return [
            self.frame_length, self.num_tx, self.num_rx, self.adc_samples * 2]

    @property
    def frame_size(self) -> int:
        """Radar data cube size, in bytes."""
        return (self.frame_length * self.num_tx * self.num_rx *
                self.adc_samples * 2 * 2)

    @property
    def chirp_time(self) -> float:
        """Per-TX antenna inter-chirp time T_c, in microseconds."""
        return (self.idle_time + self.ramp_end_time) * self.num_tx

    @property
    def frame_time(self) -> float:
        """Total radar frame time, in ms."""
        return self.chirp_time * self.frame_length / 1e3

    @property
    def sample_time(self) -> float:
        """Total sampling time T_s, in us."""
        return self.adc_samples / self.sample_rate * 1e3

    @property
    def bandwidth(self) -> float:
        """Effective bandwidth, in MHz."""
        return self.freq_slope * self.sample_time

    @property
    def range_resolution(self) -> float:
        """Range resolution, in m."""
        return SPEED_OF_LIGHT / (2 * self.bandwidth * 1e6)

    @property
    def max_range(self) -> float:
        """Maximum range, in m."""
        return self.range_resolution * self.adc_samples

    @property
    def wavelength(self) -> float:
        """Center wavelength, in m."""
        offset_time = self.adc_start_time + self.sample_time / 2
        return SPEED_OF_LIGHT / (
            self.frequency * 1e9 + self.freq_slope * (offset_time) * 1e6)

    @property
    def doppler_resolution(self) -> float:
        """Doppler resolution, in m/s."""
        return (
            self.wavelength / (2 * self.frame_length * self.chirp_time * 1e-6))

    @property
    def max_doppler(self) -> float:
        """Maximum doppler velocity, in m/s."""
        return self.wavelength / (4 * self.chirp_time * 1e-6)

    @property
    def throughput(self) -> float:
        """Average throughput, in bits/sec."""
        return self.frame_size * 8 / self.frame_period * 1e3

    def as_dict(self) -> dict:
        """Export as dictionary."""
        RADAR_PROPERTIES = [
            "frequency", "idle_time", "adc_start_time", "ramp_end_time",
            "tx_start_time", "freq_slope", "adc_samples", "sample_rate",
            "frame_length", "frame_period", "num_tx"]
        return {k: getattr(self, k) for k in RADAR_PROPERTIES}

    def as_intrinsics(self) -> dict:
        """Export as intrinsics dictionary."""
        RADAR_INTRINSICS = [
            "shape", "range_resolution", "doppler_resolution"]
        return {k: getattr(self, k) for k in RADAR_INTRINSICS}

    def check(self) -> None:
        """Check validity.

        - Duty cycle `< 1.`
        - Excess ramp time `> 0.`
        """
        duty_cycle = self.frame_time / self.frame_period
        print("Duty cycle (<1):", duty_cycle)

        excess = self.ramp_end_time - self.adc_start_time - self.sample_time
        print("Excess ramping time (>0):", excess)


@dataclass
class DCAConfig:
    """DCA1000EVM Capture card configuration.

    Attributes:
        sys_ip: system IP; should be manually configured with a subnet mask of
            `255.255.255.0`.
        fpga_ip: FPGA IP address; either hard-coded or configured.
        data_port: data network port number.
        config_port: configuration network port number.
        timeout: Socket read timeout, in seconds.
        socket_buffer: Network read buffer size; should be less than
            [`rmem_max`](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/net.html#rmem-max).
        delay: Packet delay for the capture card, in microseconds.
    """

    sys_ip: str = "192.168.33.30"
    fpga_ip: str = "192.168.33.180"
    data_port: int = 4098
    config_port: int = 4096
    timeout: float = 1.0
    socket_buffer: int = 2048000
    delay: float = 5.0

    @property
    def throughput(self):
        """Theoretical maximum data rate, in bits/sec."""
        packet_time = (
            defines.DCAConstants.DCA_PACKET_SIZE
            * 8 / defines.DCAConstants.DCA_BITRATE + self.delay / 1e6)
        return 1 / packet_time * defines.DCAConstants.DCA_PACKET_SIZE * 8

    def create(self) -> DCA1000EVM:
        """Initialize and setup capture card from this configuration."""
        dca = DCA1000EVM(
            sys_ip=self.sys_ip, fpga_ip=self.fpga_ip,
            data_port=self.data_port, config_port=self.config_port,
            timeout=self.timeout, socket_buffer=self.socket_buffer)
        dca.setup(delay=self.delay)
        return dca


class AWRSystem:
    """Radar capture system with a AWR1843Boost and DCA1000EVM.

    !!! info "Known Constraints"

        The `AWRSystem` will check for certain known constraints, and warn if
        these are violated via a logger:

        - Radar data throughput is greater than 80% of the capture card
            theoretical network throughput.
        - Receive buffer size (in the linux networking stack) can hold less
            than 2 full frames.
        - The duty cycle (active frame time / frame period) of the radar is
            greater than 95%.
        - The ADC is still sampling when the ramp ends.

    Args:
        radar: radar configuration; if `dict`, the key/value pairs are passed
            to `AWRConfig`.
        capture: capture card configuration; if `dict`, the key/value pairs are
            passed to `DCAConfig`.
        name: friendly name for logging; can be default.
    """

    def __init__(
        self, *, radar: AWRConfig | dict, capture: DCAConfig | dict,
        name: str = "RadarCapture"
    ) -> None:
        if isinstance(radar, dict):
            radar = AWRConfig(**radar)
        if isinstance(capture, dict):
            capture = DCAConfig(**capture)

        self.log = logging.getLogger(name)
        self._statistics(radar, capture)

        self.dca = capture.create()
        self.awr = AWR1843(port=radar.port)

        self.config = radar
        self.fps = 1000.0 / radar.frame_period

    def _statistics(self, radar: AWRConfig, capture: DCAConfig) -> None:
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
        self.awr.setup(**self.config.as_dict())
        self.awr.start()

        return self.dca.stream(self.config.raw_shape)

    @overload
    def qstream(
        self, numpy: Literal[True] = True
    ) -> Queue[np.ndarray | None]:
        ...

    @overload
    def qstream(
        self, numpy: Literal[False] = False
    ) -> Queue[types.RadarFrame | None]:
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

    def stop(self):
        """Stop by halting the capture card and reboot the radar.

        In testing, we found that the radar may ignore commands if the frame
        timings are too tight, which prevents a soft reset. We simply reboot
        the radar via the capture card instead.
        """
        self.dca.stop()
        self.dca.reset_ar_device()
