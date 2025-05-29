"""TI mmWave Radar + DCA1000EVM Capture Card Raw Data Capture API.

!!! usage

    To use the high-level API, create a [`XWRConfig`][.] and [`DCAConfig`][.];
    then pass these to the [`XWRSystem`][.]. Use [`stream`][.XWRSystem.]
    or [`qstream`][.XWRSystem.] to automatically configure, start, and stream
    spectrum data from the radar.

??? example "Example Configuration"

    Note that these configurations can be passed to `XWRSystem` by simply
    unpacking them as arguments (`system = XWRSystem(**config)`).

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

To use `xwr`, you will need to configure the following:

1. The network interface connected to the DCA1000EVM should be
    configured with a static IP address matching the provided `sys_ip`,
    e.g., `192.168.33.30` with a subnet mask of `255.255.255.0`.
    ```sh
    RADAR_IF=eth0  # your radar interface name, e.g., eth0, enp0s25, etc.
    RADAR_SYS_IP=192.168.33.30
    sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask 255.255.255.0
    ```

2. To reduce dropped packets, the receive socket buffer size should also
    be increased to at least 2 frames of data (even larger is fine):
    ```sh
    RECV_BUF_SIZE=16777216  # 16 MiB = 21 frames (~1 sec) @ 786k each.
    echo $(RECV_BUF_SIZE) | sudo tee /proc/sys/net/core/rmem_max
    ```

3. Provide read/write permissions for the serial ports:
    ```sh
    sudo chmod 777 /dev/ttyACM0  # or whatever port the radar is on.
    ```
"""

from beartype.claw import beartype_this_package

beartype_this_package()

# ruff: noqa: E402
from . import capture, radar, rsp
from .config import DCAConfig, XWRConfig
from .system import XWRSystem

__all__ = [
    "capture", "radar", "rsp", "XWRConfig", "XWRSystem", "DCAConfig"]
