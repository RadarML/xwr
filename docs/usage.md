# User Guide

## Hardware

If you have not done so already, set up your radar and capture card according to the [hardware setup guide](./setup.md). Also, ensure that:

- The micro-USB port on the radar is connected to the capture computer
- The ethernet port on the capture card is connected to the capture computer
- Both the radar and capture card are powered on (lights on both boards should be illuminated)

## Operating System

A few configuration options requiring elevated privileges must be set at the operating system level; since `xwr` operates only in user-space, the caller is responsible for making sure these are correctly configured.

### Network Interface

The network interface connected to the DCA1000EVM should be configured with a static IP address matching the provided `sys_ip`, e.g., `192.168.33.30` with a subnet mask of `255.255.255.0` (unless this IP has been changed; see [`configure_eeprom`][xwr.capture.DCA1000EVM.configure_eeprom]).
```sh
RADAR_IF=eth0  # your radar interface name, e.g., eth0, enp0s25, etc.
RADAR_SYS_IP=192.168.33.30
sudo ifconfig $RADAR_IF $RADAR_SYS_IP netmask 255.255.255.0
```

!!! warning

    In ubuntu desktop systems with a GUI installed, the GUI settings can override command line options. If this happens, it may be easier to set the IP and netmask using the settings app instead.

### Receive Socket Buffer

To reduce dropped packets, the receive socket buffer size should also be increased to at least 2 frames of data:
```sh
RECV_BUF_SIZE=6291456  # 6.3 MiB = 8 frames @ 786k each.
echo $RECV_BUF_SIZE | sudo tee /proc/sys/net/core/rmem_max
# or
sudo sysctl -w net.core.rmem_max=$RECV_BUF_SIZE
```

!!! note

    There is no problem with making this value arbitrarily large; if the receive buffer size is set to a value which exceeds the system limits (`cat /proc/sys/net/ipv4/tcp_rmem`), the actual value will be capped to the maximum allowed. This is usually more than enough (6291456 ~ 6MiB in our test system).

!!! tip

    You can also set `net.core.rmem_max=6291456` in `/etc/sysctl.conf` (then reload to apply settings with `sudo sysctl -p`)  to make this change persistent across reboots.

### Serial Port Permissions

Provide read/write permissions for the serial ports:
```sh
sudo chmod 777 /dev/ttyACM0  # or whatever port the radar is on.
```

!!! warning

    This step may need to be repeated each time the radar is reconnected or rebooted.

## Radar Configuration

The high level interface for `xwr` includes a capture card configuration ([`DCAConfig`][xwr.DCAConfig]) and a radar configuration ([`XWRConfig`][xwr.XWRConfig]).

While the capture card configuration can be [left as default][xwr.DCAConfig], the radar requires a valid modulation to be configured; see [`XWRConfig`][xwr.XWRConfig] for details.

!!! tip

    The [TI mmWave sensing estimator](https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.4.0/) may be helpful for creating a configuration.

As a general guideline:

1. Select a range/doppler resolution in bins; there appears to be an effective limitation on the total range-Doppler frame size of `2^14` (i.e., `128x128`, `64x256`, etc is the maximum size).
    - The number of range bins is the `adc_samples`.
    - The number of doppler bins is the `frame_length`.

2. Select a target doppler resolution/maximum doppler.
    - This informs the per-antenna chirp rate and `frame_period`.

3. Select a target range resolution/maximum range, keeping in mind the theoretical range resolution limit of `C / 2B` (the bandwidth `B` is 4GHz for a 77-81 GHz mmWave radar).
    - Based on the maximum chirp time, you will need to select a `freq_slope`, `sample_rate`, `adc_start_time`, `tx_start_time`, and `ramp_end_time` which keeps the total chirp time under the chirp period.
    - Set the `idle_time` to meet the target per-chirp time.

!!! example "Example Configurations"

    Note that these configurations can be passed to `XWRSystem` by simply
    unpacking them as arguments (`system = XWRSystem(**config)`).

    === "256x64, 22m range x 8m/s Doppler"

        ```yaml
        radar:
            device: AWR1843
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
        capture:
            sys_ip: 192.168.33.30
            fpga_ip: 192.168.33.180
            socket_buffer: 6291456
        ```

    === "128x128, 5m range x 1.2m/s Doppler"

        ```yaml
        radar:
            device: AWR1843
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
        capture:
            sys_ip: 192.168.33.30
            fpga_ip: 192.168.33.180
            socket_buffer: 6291456
        ```

## Capture Data

### Run the Demo

With [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/) and the XWR repository cloned:
```sh
uv run demo/demo.py --extra demo --device AWR1843 --rsp AWR1843
```

!!! info

    Replace the `--device` and `--rsp` with specifications according to your radar development board:

    | Radar                   | `--device` | `--rsp`         |
    |-------------------------|------------|-----------------|
    | AWR1843Boost            | AWR1843    | AWR1843Boost    |
    | AWR1843Boost, 1642 Mode | AWR1843L   | AWR1642Boost    |
    | AWR1843AOPEVM           | AWR1843    | AWR1843AOP      |
    | AWR1642Boost            | AWR1642    | AWR1642Boost    |

![Spectrum Demo](images/demo.png)

### Use the High Level API

Pass the configuration to [`xwr.XWRSystem`][xwr.XWRSystem]:
```python
import logging
import yaml
import xwr

logging.basicConfig(level=logging.DEBUG)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

awr = xwr.XWRSystem(**cfg)
for frame in awr.stream():
    break

awr.stop()
```

See the [high level API documentation](system.md) for additional details.

!!! warning

    Since data capture is performance sensitive, we provide three different ways to stream data:

    - [`XWRSystem.stream`][xwr.XWRSystem.stream]: stream data directly as an iterator in the main thread. Note that a high *worst case execution time* (e.g., due to file system writes) may lead to packet drops!
    - [`XWRSystem.qstream`][xwr.XWRSystem.qstream]: read data into a **(q)ueue** using a separate thread. This is much less performance sensitive, though the reader must still keep up with the radar's data rate.
    - [`XWRSystem.dstream`][xwr.XWRSystem.dstream]: also read data using a separate thread, but **(d)rop** older frames if they are not consumed fast enough to ensure that the reader always gets the latest frame.

## Troubleshooting

!!! info "Initialization Delay"

    While the radar is booting, you will not be able to open the serial port.
    ```
    [Errno 16] Device or resource busy: '/dev/ttyACM0'
    ```
    This is normal, and should go away after ~10-30 seconds.

**Radar start times out**: If the radar does not respond after issuing `Send: sensorStart`, this usually indicates an invalid radar configuration; the TI firmware does not provide any error messages or indications.

**Dead FPGA**: When powered on, the capture card error lights should all come on for ~1sec, then turn off again. If this does not occur, the FPGA may be dead.

**Device Times Out**: This can also be caused by a loose LVDS cable (the blue ribbon cable between the radar and capture card), if the pins corresponding to commands are loose.

**Frequent Dropped Packets**: If you receive dropped packet warnings (`Dropped packets: X bytes`); check to make sure that `rmem_max` was set correctly. In particular, look for another warning:
```
Receive buffer size X is smaller than 2 frames (Y); is net.core.rmem_max set?
```

**Other Hardware Faults**: The [TI mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) is a good way to validate radar hardware functionality, and uses the same demo firmware.

!!! note

    If an error is returned on the console in the Demo Visualizer: there may be a hardware fault. It should be raised with a line number in `mss_main.c`; the error case (e.g. `RL_RF_AE_CPUFAULT_SB`) should reveal what general type of fault it is.
