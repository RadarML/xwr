# Hardware Setup

!!! abstract "TL;DR"

    A mmWave radar I/Q data collection system consists of two parts:

    1. A radar with a Low Voltage Differential Signaling (LVDS) debug port which dumps out raw data, and
    2. A capture card which translates these LVDS signals into ethernet packets.

    To use these components with our [`radar`][xwr.radar] and [`capture`][xwr.capture] interfaces, the radar needs to be flashed with firmware with LVDS streaming enabled, while the capture cards need to be configured to read the output signals.

## DCA1000EVM Capture Card

!!! danger

    In our experience, the DCA1000EVM is particularly fragile; be careful with electrostatic discharge (ESD).

![DCA1000EVM](images/dca1000evm.jpg){: style="width: 50%"}

Ensure that the following DIP switches are set:

- SW2.5: `SW_CONFIG`
- SW2.6: `USER_SW1` (the marked right side), unless the EEPROM is messed up from a misconfigured [`configure_eeprom`][xwr.capture.DCA1000EVM.configure_eeprom] call.

The `DC_JACK_5V_IN` (the large switch on the side) should also be set, depending on whether the FPGA will be powered via the DC jack or via the radar.

??? info "Hardware Configuration Switches (Optional)"

    The following are configured by [`configure_fpga`][xwr.capture.DCA1000EVM.configure_fpga] under normal operation, but can be manually set in case that isn't working:

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

<div class="grid" markdown>

![AWR1843Boost](images/awr1843boost.jpg)

![AWR1843Boost Inset](images/awr1843boost-inset.jpg)

</div>

1. Prepare for flashing.

    - Connect a micro-USB cable to the port on the radar, and power the radar on by connecting its power supply.
    - Find `SOP2:0` (DIP switches on the front of the radar). Set the switches to `SOP2:0=101`, where 1 corresponds to the "on" position labeled on the PCB.
    - Find switch `S2` in the middle of the radar, and set it to `SPI` (lower position).

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    !!! note

        UniFlash seems to work most reliably on windows.

    - Uniflash should automatically discover the radar. If not, select the `AWR1843Boost` device.
    - Select the `xwr18xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

    ??? failure "`Not able to connect to serial port. Recheck COM port selected and/or permissions.`"

        If the SOP switches or `S2` are not in the correct position, flashing will fail with
        ```
        Not able to connect to serial port.
        Recheck COM port selected and/or permissions.
        ```

3. Set the radar to functional mode: `SOP2:0=001`.

    !!! note
    
        mmWave studio expects the radar to be in *debug* mode (`SOP2:0=011`), so switching between `xwr` and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.

## AWR1843AOP

!!! info "Firmware"

    After installing the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK), the firmware can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_aop_demo.bin` in the install directory.

??? warning "[SICP2105 drivers](https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers) must be installed to flash the radar."

    As per the AWR1843AOPEVM [user manual](https://www.ti.com/lit/pdf/spruix8), the SICP2105 drivers must be installed to access the UART port during the flashing process. 
    
    If these drivers are not (properly) installed, the serial ports will appear as an "Enhanced Com Port" and "Standard Com Port" with a warning icon in the windows device manager.
    
    Download and install the drivers [here](https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers): after downloading, go to
    ```
    device manager > Standard / Enhanced Com Port > Update Drivers > Browse my computer for drivers
    ```
    then select the folder containing the drivers you downloaded. You may need to do this twice: once for the "Enhanced Com Port", and once for the "Standard Com Port".

<div class="grid" markdown>

![AWR1843AOPEVM](images/awr1843aopevm.jpg)

![AWR1843AOPEVM Inset](images/awr1843aopevm-inset.jpg)

</div>

1. Prepare the radar for flashing.

    - Connect a micro-USB cable to the port on the radar. The additional power supply is not needed for this step.
    - Find `SOP0`, `SOP`, `SOP2`. `SOP2` is all the way on the left, while `SOP0` and `SOP1` are on the right-most block of 4 switches.
    - Set `SOP2:0=001`. In both cases, on (1) is up.

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

    - DIP Switch 2 (center), position 2 should be on (up). All other switches should be off (down).

    ??? quote "Switch Positions"

        |    | 1   | 2   | 3   | 4   |
        | -- | --- | --- | --- | --- |
        | S1 | OFF | any | OFF | OFF |
        | S2 | OFF | ON  | any | any |
        | S3 | OFF |     |     |     |

## AWR1642Boost

!!! info "Firmware"

    After installing the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK), the firmware can be found at `demo/xwr16xx/mmwave/xwr16xx_mmw_demo.bin` in the install directory.

!!! tip

    Setting the large power switch on the DCA1000EVM to `RADAR_5V_IN`, a single power supply connected to the radar is sufficient to power the entire system.


![AWR1642Boost](images/awr1642boost.jpg){: style="width: 50%"}

1. Prepare for flashing.

    - Connect a micro-USB cable to the port on the radar, and power the radar on by connecting its power supply.
    - Find `SOP2:0`, and short SOP0 and SOP2 (`SOP2:0=101`). These are physical jumpers, which must be shorted using jumper caps or wires.

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    !!! note

        UniFlash seems to work most reliably on windows.

    - Uniflash should automatically discover the radar. If not, select the `AWR1642Boost` device.
    - Select the `xwr16xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

    ??? failure "`Not able to connect to serial port. Recheck COM port selected and/or permissions.`"

        If the SOP switches or `S2` are not in the correct position, flashing will fail with
        ```
        Not able to connect to serial port.
        Recheck COM port selected and/or permissions.
        ```

3. Set the radar to functional mode: `SOP2:0=001`.

    !!! note
    
        mmWave studio expects the radar to be in *debug* mode (`SOP2:0=011`), so switching between `xwr` and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.

## :construction_site: AWR2544LOPEVM

!!! failure "Not yet working"

!!! warning

    Flashing the AWR2544LOPEVM requires two jumper caps or wires in order to physicall short the required pins. One of these jumpers must remain on the board to set it to functional mode.

1. Prepare for flashing.

    - Plug in a micro USB cable to the XDS port (on the right side).
    - Find `SOP2:0`, and short SOP0 and SOP2 (`SOP2:0=101`). These are physical jumpers, which must be shorted using jumper caps or wires.

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    !!! note

        UniFlash seems to work most reliably on windows.

    - Uniflash should automatically discover the radar. If not, select the `AWR1843Boost` device.
    - Select the `xwr18xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

    ??? failure "`Not able to connect to serial port. Recheck COM port selected and/or permissions.`"

        If the SOP switches or `S2` are not in the correct position, flashing will fail with
        ```
        Not able to connect to serial port.
        Recheck COM port selected and/or permissions.
        ```

3. Set the radar to functional mode: `SOP2:0=001`.

    !!! note
    
        mmWave studio expects the radar to be in *debug* mode (`SOP2:0=011`), so switching between `xwr` and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.

1. Prepare for flashing.

    - Plug in a USB cable to the XDS port (on the right side).
    - Find `SOP2:0`, and short SOP0 and SOP2 (`SOP2:0=101`). These are physical jumpers, which must be shorted using jumper caps or wires.

2. Obtain firmware with LVDS streaming enabled, and flash using the provided command line tools.

    !!! bug

        Flashing using TI Uniflash does not work; you must use the python script.

    ??? quote "Build requirements for firmware compilation on linux"

        1. Prerequisites. Note that the TI libraries require 32-bit compatibility mode for some reason.
        ```sh
        sudo dpkg --add-architecture i386
        sudo apt-get update
        sudo apt-get install libc6:i386 libstdc++6:i386

        sudo apt-get install build-essential
        sudo apt-get install mono-complete
        ```

        2. [MMWAVE MCUPLUS SDK](https://www.ti.com/tool/MMWAVE-MCUPLUS-SDK)
        ```sh
        wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-U4MY7aGNn5/04.07.00.01/mmwave_mcuplus_sdk_04_07_00_01-Linux-x86-Install.bin
        chmod +x mmwave_mcuplus_sdk_04_07_00_01-Linux-x86-Install.bin
        ./mmwave_mcuplus_sdk_04_07_00_01-Linux-x86-Install.bin
        ```
            - Accept all packages.
            - Keep all options default.

        3. [SYSCONFIG](https://www.ti.com/tool/download/SYSCONFIG/1.21.0.3721)
        ```sh
        wget https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-nsUM6f7Vvb/1.21.0.3721/sysconfig-1.21.0_3721-setup.run
        chmod +x sysconfig-1.21.0_3721-setup.run
        ./chmod +x sysconfig-1.21.0_3721-setup.run
        ```
            - Keep all options default, except `Create Desktop Shortcut` and `Launch TI System Configuration Tool`.

        4. [ARM-CGT-CLANG](https://www.ti.com/tool/download/ARM-CGT-CLANG/4.0.3.LTS)
        ```sh
        wget https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-ayxs93eZNN/4.0.3.LTS/ti_cgt_armllvm_4.0.3.LTS_linux-x64_installer.bin
        chmod +x ti_cgt_armllvm_4.0.3.LTS_linux-x64_installer.bin
        ./ti_cgt_armllvm_4.0.3.LTS_linux-x64_installer.bin
        ```
            - Recommended: set the destination directory to `/home/<username>/ti/cgt-armllvm_4.0.3.LTS`.

        5. Set up environment.

            Modify `~/ti/mmwave_mcuplus_sdk_04_07_00_01/mmwave_mcuplus_sdk_04_07_00_01/scripts/unix/setenv.sh`:
            ```sh
            ...
            # Change to /home/<username>/ti/ccs<version>
            export CCS_INSTALL_PATH=/opt/ti/ccs1281
            ...
            # Change to /home/<username>/ti/cgt-armllvm_4.0.3.LTS
            export R5F_CLANG_INSTALL_PATH=/opt/ti/sysconfig_1.21.0 
            ```

    ??? quote "Compile firmware with LVDS streaming"

        Edit `~/mmwave_mcuplus_sdk_04_07_00_01/mmwave_mcuplus_sdk_04_07_00_01/ti/common/`, and add `-DLVDS_STREAM` to `DEFINES`:
        ```makefile
        DEFINES = \
            -DSUBSYS_MSS \
            -D$(PLATFORM_DEFINE) \
            -D$(DEVICE_TYPE) \
            -D_LITTLE_ENDIAN \
            -DLVDS_STREAM \
        ```

        Then compile:
        ```sh
        cd ~/ti/mmwave_mcuplus_sdk_04_07_00_01/mmwave_mcuplus_sdk_04_07_00_01/scripts/unix
        source setenv.sh

        cd ~/ti/mmwave_mcuplus_sdk_04_07_00_01/mmwave_mcuplus_sdk_04_07_00_01/ti/demo/awr2544/mmw
        make clean
        make mmwDemoBuild
        ```

    - Make sure you have `python` installed, along with `pyserial`, `xmodem`, `tqdm`:
        ```sh
        pip install pyserial xmodem tqdm
        ``` 

    - Find the COM port corresponding to the `XDS110 Class Application/User UART` port.
    - Then, in `C:\ti\mmwave_mcuplus_sdk_04_07_00_01\mmwave_mcuplus_sdk_04_07_00_01\tools\awr2544`, run the following (replacing `COM5` with the radar board's COM port):
        ```
        python C:\ti\mmwave_mcuplus_sdk_04_07_00_01\mcu_plus_sdk_awr2544_10_00_00_07\tools\boot\uart_uniflash.py -p COM5 --cfg default.cfg
        ```

3. Switch the radar to functional mode.

    - Remove the jumper on SOP2, so only a jumper on SOP0 remains (`SOP2:0=001`).
