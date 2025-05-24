"""AWR1843Boost Radar & DCA1000EVM Capture Card API.

!!! usage

    To use the high-level API, create a [`AWRConfig`][.] and [`DCAConfig`][.];
    then pass these to the [`AWRSystem`][.]. Then, use [`stream`][.AWRSystem.]
    or [`qstream`][.AWRSystem.] to automatically configure, start, and stream
    spectrum data from the radar.

??? example "Example Configuration"

    Note that these configurations can be passed to `AWRSystem` by simply
    unpacking them as arguments (`system = AWRSystem(**config)`).

    === "256x64, 22m range x 8m/s Doppler"

        ```yaml
        radar:
            port: null
            frequency: 77.0
            idle_time: 6.0
            adc_start_time: 5.7
            ramp_end_time: 34.00
            tx_start_time: 1.0
            freq_slope: 67.012
            adc_samples: 256
            sample_rate: 10000
            frame_length: 64
            frame_period: 50.0
            num_tx: 3
        capture:
            sys_ip: 192.168.33.30
            fpga_ip: 192.168.33.180
            data_port: 4098
            config_port: 4096
            timeout: 1.0
            socket_buffer: 16777216
            delay: 60.0
        ```
    
    === "128x128, 5m range x 1.2m/s Doppler"

        ```yaml
        radar:
            port: null
            frequency: 77.0
            idle_time: 331.0
            adc_start_time: 5.7
            ramp_end_time: 59.00
            tx_start_time: 1.0
            freq_slope: 67.012
            adc_samples: 128
            sample_rate: 2500
            frame_length: 128
            frame_period: 100.0
            num_tx: 2
        capture:
            sys_ip: 192.168.33.30
            fpga_ip: 192.168.33.180
            data_port: 4098
            config_port: 4096
            timeout: 1.0
            socket_buffer: 4194304
            delay: 200.0
        ```
"""
from beartype.claw import beartype_this_package

beartype_this_package()

# ruff: noqa: E402
from . import capture, radar
from .system import AWRConfig, AWRSystem, DCAConfig

__all__ = [
    "capture", "radar", "AWRConfig", "AWRSystem", "DCAConfig"]
