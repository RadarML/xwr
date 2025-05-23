# Radar Setup Guide

## Flashing the AWR1843Boost

Flash the radar using [TI UniFlash](https://www.ti.com/tool/UNIFLASH); note that it seems to work most reliably on Windows. Also, obtain a copy of the `xwr18xx_mmw_demo.bin` firmware (e.g. through installing [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK)).

1. Set the radar to flash mode.
    - Find `SOP0:2` (DIP switches on the front of the radar).
    - Set the switches to `SOP0:2=101`, where 1 corresponds to the "on" position labeled on the PCB.
    - Find switch `S2` in the middle of the radar, and set it to `SPI` (lower position).
2. Flash using UniFlash.
    - Uniflash should automatically discover the radar.
    - Select the `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description "XDS110 Class Application/User UART (COM3)".
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".
    - If the SOP switches or `S2` are not in the correct position, flashing will fail with
        > Not able to connect to serial port. Recheck COM port selected and/or permissions
3. Set the radar to functional mode.
    - Set `SOP0:2=001`.
    - Note that mmWave studio expects the radar to be in *debug* mode (`SOP0:2=011`), so switching between Red Rover and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.
