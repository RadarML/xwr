# Radar Setup Guide


## DCA1000EVM Capture Card

!!! warning

    In our experience, the DCA1000EVM is particularly fragile; be careful with electrostatic discharge (ESD).

Ensure that the following DIP switches are set:

- SW2.5: `SW_CONFIG`
- SW2.6: `USER_SW1` (the marked right side), unless the EEPROM is messed up from a misconfigured [`configure_eeprom`][awr_api.capture.DCA1000EVM.configure_eeprom] call.

The `DC_JACK_5V_IN` (the large switch on the side) should also be set, depending on whether the FPGA will be powered via the DC jack or via the radar.

??? info "Hardware Configuration Switches (Optional)"

    The following are configured by [`configure_fpga`][awr_api.capture.DCA1000EVM.configure_fpga] under normal operation, but can be manually set in case that isn't working:

    - SW1: 16-bit mode (`16BIT_ON`, `14BIT_OFF`, `12BIT_OFF`).
    - SW2.1: `LVDS_CAPTURE`
    - SW2.2: `ETH_STREAM`
    - SW2.3: `AR1642_MODE` (2-lane LVDS)
    - SW2.4: `RAW_MODE`
    - SW2.5: `HW_CONFIG`


## AWR1843Boost

!!! info "Firmware"

    After installing the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK), the firmware can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin` in the install directory.

!!! tip

    Setting the large power switch on the DCA1000EVM to `RADAR_5V_IN`, a single power supply connected to the radar is sufficient to power the entire system.

1. Set the radar to flash mode.

    - Find `SOP0:2` (DIP switches on the front of the radar).
    - Set the switches to `SOP0:2=101`, where 1 corresponds to the "on" position labeled on the PCB.
    - Find switch `S2` in the middle of the radar, and set it to `SPI` (lower position).

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    !!! note

        UniFlash seems to work most reliably on windows.

    - Uniflash should automatically discover the radar. If not, select the `AWR1843Boost` device.
    - Select the `xwr18xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description "XDS110 Class Application/User UART".
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".
    - If the SOP switches or `S2` are not in the correct position, flashing will fail with
        ```
        Not able to connect to serial port.
        Recheck COM port selected and/or permissions.
        ```

3. Set the radar to functional mode.

    - Set `SOP0:2=001`.
    - Note that mmWave studio expects the radar to be in *debug* mode (`SOP0:2=011`), so switching between Red Rover and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.

## AWR1843AOP

!!! info "Firmware"

    After installing the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK), the firmware can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_aop_demo.bin` in the install directory.

!!! warning

    As per the AWR1843AOPEVM [user manual](https://www.ti.com/lit/pdf/spruix8), the SICP2105 drivers must be installed to access the UART port.
    
    If these drivers are not (properly) installed, the serial ports will appear as an "Enhanced Com Port" and "Standard Com Port" with a warning icon in the windows device manager.
    
    Download and install the drivers [here](https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers): after downloading, go to
    ```
    device manager > Standard / Enhanced Com Port > Update Drivers > Browse my computer for drivers
    ```
    then select the folder containing the drivers you downloaded. You may need to do this twice: once for the "Enhanced Com Port", and once for the "Standard Com Port".

1. Set the radar to flash mode.

    - Find `SOP0`, `SOP`, `SOP2`. `SOP2` is all the way on the left, while `SOP0` and `SOP1` are on the right-most block of 4 switches.
    - Set `SOP0:2=001`. In both cases, on (1) is up.

    ??? quote "Switch Positions"

        |    | 1   | 2   | 3   | 4   |
        | -- | --- | --- | --- | --- |
        | S1 | any | any | OFF | OFF |
        | S2 | any | any | any | any |
        | S3 | ON  |     |     |     |

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    !!! warning

        Make sure the DCA1000EVM capture card, if connected, is not powered on. If the capture card is powered, flashing will fail with `Received Unexpected Data`.


    - Select the `AWR1843` device.
    - Select the `xwr18xx_mmw_aop_demo.bin` image to flash.
    - Choose the serial port corresponding to the `Silicon Labs Dual CP2105 USB to UART Bridge: Enhanced`.

3. Set the radar to functional, DCA1000EVM mode.

    - DIPSwitch 2 (center), position 2 should be on (up). All other switches should be off (down).

    ??? quote "Switch Positions"

        |    | 1   | 2   | 3   | 4   |
        | -- | --- | --- | --- | --- |
        | S1 | OFF | any | OFF | OFF |
        | S2 | OFF | ON  | any | any |
        | S3 | OFF |     |     |     |

## AWR2544LOPEVM

!!! failure "Not yet working"

!!! info "Firmware"

    After installing the [MMWAVE-MCUPLUS-SDK](https://www.ti.com/tool/MMWAVE-MCUPLUS-SDK), two files are needed:
    
    - SBL image at `tools/awr2544/sbl_qspi.release.tiimage` in the `mmwave_mcuplus_sdk_<ver>/mmwave_mcuplus_sdk_<ver>` install directory.
    - AppImage at `ti/demo/awr2544/mmw/awr2544_mmw_demo.appimage`, again in the `mmwave_mcuplus_sdk_<ver>/mmwave_mcuplus_sdk_<ver>` install directory.

!!! warning

    Flashing the AWR2544LOPEVM requires two jumper caps or wires in order to physicall short the required pins. One of these jumpers must remain on the board to set it to functional mode.

!!! warning

    The AWR2544LOPEVM is one of TI's latest radar products; if this option is missing when flashing, please [update UniFlash](https://www.ti.com/tool/UNIFLASH).

1. Prepare for flashing.

    - Plug in a USB cable to the XDS port (on the right side).
    - Find SOP0-2. These are physical jumpers, which must be shorted using jumper caps or wires.
    - Short SOP0 and SOP2 (top and bottom).

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    - Uniflash should automatically discover the radar. If not, select the `AWR2544LOPEVM` device.
    - Select the SBL image as `sbl_qspi.release.tiimage`, and the AppImage as `awr2544_mmw_demo.appimage`.
