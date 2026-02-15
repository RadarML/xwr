# XWR Ros Node

A ROS 2 node wrapping the XWR data collection pipeline can be found in the [xwr_ros](https://github.com/radarML/xwr_ros) package.

## Setup

1. Follow the [hardware](setup.md) and [software setup](usage.md) instructions for your radar and capture card.

    - Don't forget to set the capture card's IP address to `192.168.33.30` as described in the software setup.
    - You should also create a [config file](usage.md#radar-configuration)

2. Install [uv](https://uv.pypa.io/en/stable/) and [colcon-uv](https://github.com/nzlz/colcon-uv):
    ```sh
    pip install uv colcon-uv
    ```

3. After cloning the repository, build the ROS node with colcon-uv:
    ```sh
    UV_VENV_CLEAR=1 UV_LINK_MODE=symlink colcon build
    ```
    This will create a uv venv and install necessary dependencies.

## Run the ROS node

Increase [socket receive buffer size](https://radarml.github.io/xwr/usage/#receive-socket-buffer) before starting streaming:
```sh
sudo sysctl -w net.core.rmem_max=6291456  # 6.3 MiB = 8 frames @ 786k each.
```

Then, launch the streaming node with the config file.
```sh
ros2 launch xwr_ros radar.launch.py config:=config
```

- This will publish radar signal topic `/xwr/iq`.

To run signal processing visualization:
```sh
ros2 run xwr_ros process
```

- This will publish range_doppler and range_azimuth visualization images to `/xwr/range_doppler` and  `/xwr/range_azimuth`.

## Messages

The ROS node publishes the following messages:

**ChirpInfo**:
```
std_msgs/Header header

float32 frequency       # GHz
float32 idle_time       # us
float32 adc_start_time  # us
float32 ramp_end_time   # us
float32 tx_start_time   # us
float32 freq_slope      # MHz/us
uint16 adc_samples      # number of samples per chirp
uint16 sample_rate      # KHz
uint16 frame_length     # number of chirp loop
```

**IQ**:
```
std_msgs/Header header

std_msgs/Int16MultiArray iq # raw data contain chirp layout
bool complete
```
