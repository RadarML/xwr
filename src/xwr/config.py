"""High level radar capture system API."""

from dataclasses import dataclass

from .capture import DCA1000EVM, defines

SPEED_OF_LIGHT = 299792458
"""Speed of light, in m/s."""

@dataclass
class XWRConfig:
    """Radar configuration.

    The [TI mmWave sensing estimator](
    https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.4.0/)
    may be helpful for creating a configuration.

    Attributes:
        port: Control serial port (usually `/dev/ttyACM0`). Use `None` to
            auto-detect; see [`XWRBase`][xwr.radar.XWRBase].
        frequency: base frequency, in GHz.
        idle_time: radar timing parameters; in microseconds.
        adc_start_time: radar timing parameters; in microseconds.
        ramp_end_time: radar timing parameters; in microseconds.
        tx_start_time: radar timing parameters; in microseconds.
        freq_slope: chirp slope, in MHz/us.
        adc_samples: number of samples per chirp.
        sample_rate: ADC sampling rate, in KHz.
        frame_length: number of chirps per TX antenna per frame. Must be a
            power of two.
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
    port: str | None = None
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
            "frame_length", "frame_period", "num_tx", "num_rx"]
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
