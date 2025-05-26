# `xwr`: Linux-Compatible Real-Time Raw Data Capture for TI mmWave Radars

## Setup

## Supported Devices

**AWR1843Boost/AWR1843AOPEMV**:

**AWR2544LOPEVM**:

## Troubleshooting

### Hardware Faults

The [TI mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) is a good way to validate radar hardware functionality, and uses the same demo firmware.

- If an error is returned on the console in the Demo Visualizer: there may be a hardware fault. It should be raised with a line number in `mss_main.c`; the error case (e.g. `RL_RF_AE_CPUFAULT_SB`) should reveal what general type of fault it is.

When powered on, the capture card error lights should all come on for ~1sec, then turn off again. If this does not occur, the FPGA may be dead.

### Initialization

While the radar is booting, you will not be able to open the serial port:
```
[Errno 16] Device or resource busy: '/dev/ttyACM0'
```
This is normal, and should go away after ~10-30 seconds.

### Device Times Out

This can also be caused by a loose LVDS cable (the blue ribbon cable between the radar and capture card), if the pins corresponding to commands are loose.
