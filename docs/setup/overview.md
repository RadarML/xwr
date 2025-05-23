# Radar & Capture Card Setup Guide

*Red Rover* is based on the TI AWR1843Boost mmWave radar ("red board") and the DCA1000EVM capture card ("green board"). The AWR1843Boost
has a LVDS (Low Voltage Differential Signaling) "debug" port which outputs data being sent from the radar back-end to an onboard DSP processor, which can be sent to the capture card; the capture card contains a FPGA, which buffers this data and translates it into ethernet packets.

The AWR1843Boost device firmware has three code sections:

1. MSS (Master Sub-System): high-level control of the radar, which runs on an onboard ARM Cortex R4F. We use the demo firmware provided with mmWave SDK, which can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin`.
2. DSS (DSP Sub-System): control code for the onboard DSP. This section is combined with the MSS code in the compiled code distributed with mmWave SDK.
3. RSS/BSS (Radar Sub-System / Backend Sub-System): low-level control of the radar. Uses TI proprietary code, which can be found in `firmware/radarss/xwr18xx_radarss_rprc.bin` in a mmWave SDK installation. Note that this is (probably) also included with the MSS code.

A new system needs to be configured with the following:

1. Flash firmware to the Radar, and set the DIP switch to "functional" mode.
2. Configure DIP switches on the Capture Card (DCA1000EVM - green board).

## Hardware Troubleshooting

The [TI mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) is a good way to validate hardware functionality, and uses the same demo firmware.

Possible faults:

- An error is returned on the console in the Demo Visualizer: there may be a hardware fault. It should be raised with a line number in `mss_main.c`; the error case (e.g. `RL_RF_AE_CPUFAULT_SB`) should reveal what general type of fault it is.
- When powered on, the capture card error lights should all come on for ~1sec, then turn off again. If this does not occur, the FPGA may be dead.
