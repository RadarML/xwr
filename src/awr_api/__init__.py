"""AWR1843Boost Radar & DCA1000EVM Capture Card API.

!!! usage

    To use the high-level API, create a [`AWRConfig`][.] and [`DCAConfig`][.];
    then pass these to the [`AWRSystem`][.]. Then, use [`stream`][.AWRSystem.]
    or [`qstream`][.AWRSystem.] to automatically configure, start, and stream
    spectrum data from the radar.

??? example "Example Configuration"

    ```yaml
    # This can be passed to `AWRSystem` by simply unpacking:
    # system = AWRSystem(**config)
    radar:  # -> AWRConfig
      port: /dev/ttyACM0
      frequency: 77.0
      idle_time: 200.0
      adc_start_time: 5.7
      ramp_end_time: 59.00
      tx_start_time: 1.0
      freq_slope: 67.012
      adc_samples: 256
      sample_rate: 5000
      frame_length: 64
      frame_period: 50.0
      num_tx: 3
    capture:  # -> DCAConfig
      sys_ip: 192.168.33.30
      fpga_ip: 192.168.33.180
      data_port: 4098
      config_port: 4096
      timeout: 1.0
      socket_buffer: 16777216
      delay: 60.0
    ```
"""
from beartype.claw import beartype_this_package

beartype_this_package()

# ruff: noqa: E402
from . import capture, radar
from .system import AWRConfig, AWRSystem, DCAConfig

__all__ = [
    "capture", "radar", "AWRConfig", "AWRSystem", "DCAConfig"]
