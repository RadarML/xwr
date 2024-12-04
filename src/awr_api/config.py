"""Radar configuration."""

from beartype.typing import NamedTuple

SPEED_OF_LIGHT = 299792458
"""Speed of light, in m/s."""


RADAR_PROPERTIES = [
    "frequency", "idle_time", "adc_start_time", "ramp_end_time",
    "tx_start_time", "freq_slope", "adc_samples", "sample_rate",
    "frame_length", "frame_period", "num_tx"]
"""Properties which define a radar configuration."""


RADAR_INTRINSICS = [
    "shape", "range_resolution", "doppler_resolution"]
"""Intrinsics needed for radar data processing."""


class RadarConfig(NamedTuple):
    """Radar configuration.

    [R8]_ may be helpful for creating a configuration.

    Attributes:
        port: Control serial port (usually `/dev/ttyACM0`).
        frequency: base frequency, in GHz.
        idle_time, adc_start_time, ramp_end_time, tx_start_time: radar timing
            parameters, in us.
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
        """Per-TX antenna inter-chirp time T_c, in us."""
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
        return {k: getattr(self, k) for k in RADAR_PROPERTIES}

    def as_intrinsics(self) -> dict:
        """Export as intrinsics dictionary."""
        return {k: getattr(self, k) for k in RADAR_INTRINSICS}

    def check(self) -> None:
        """Check validity.

        Currently checks the following:

        - Duty cycle `< 1.`
        - Excess ramp time `> 0.`
        """
        duty_cycle = self.frame_time / self.frame_period
        print("Duty cycle (<1):", duty_cycle)

        excess = self.ramp_end_time - self.adc_start_time - self.sample_time
        print("Excess ramping time (>0):", excess)


DCA_PACKET_SIZE = 1466
"""Typical radar packet size, hard-coded in the FPGA."""


DCA_BITRATE = 1e9
"""Gigabit ethernet speed."""


class CaptureConfig(NamedTuple):
    """Capture card configuration.

    Attributes:
        sys_ip: system IP; should be manually configured with a subnet mask of
            `255.255.255.0`.
        fpga_ip: FPGA IP address; either hard-coded or configured.
        data_port, config_port: data, configuration network ports.
        timeout: Socket read timeout, in seconds.
        socket_buffer: Network read buffer size; should be less than `rmem_max`.
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
        packet_time = DCA_PACKET_SIZE * 8 / DCA_BITRATE + self.delay / 1e6
        return 1 / packet_time * DCA_PACKET_SIZE * 8
