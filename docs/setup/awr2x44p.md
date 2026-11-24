# AWR2X44(P)

After installing the [mmWave MCU Plus SDK](https://www.ti.com/tool/MMWAVE-MCUPLUS-SDK), you will need two firmware files, all relative to the inner `mmwave_mcuplus_sdk_{version}` directory of the install (e.g. `~/ti/mmwave_mcuplus_sdk_{version}/mmwave_mcuplus_sdk_{version}`):

| Radar | Application Image | Bootloader Image |
| ----- | ----------------- | ---------------- |
| AWR2944EVM | `ti/demo/awr294x/mmw/awr2944_mmw_demoTDM.appimage` | `tools/awr294x/sbl_qspi.release.tiimage` |
| AWR2944PEVM | `ti/demo/awr2x44P/mmw_tdm/awr2x44P_mmw_demoTDM.appimage` | `tools/awr2x44P/sbl_qspi.release.tiimage` |
| AWR2E44PEVM | `ti/demo/awr2x44P/mmw_tdm/awr2x44P_mmw_demoTDM.appimage` | `tools/awr2x44P/sbl_qspi.release.tiimage` |

The pre-built AWR2x44P images are built for the AWR2E44P package by default, and the stock TDM demo supports raw ADC streaming over LVDS.

!!! note

    Flashing these radars requires two jumper caps or wires in order to physically short the required pins. One of these jumpers must remain on the board to set it to functional mode.

!!! warning "Requires External 12V Power"

    These radars require external 12V power (2.1mm barrel jack, center positive, at least 2.5A) to operate, while the DCA1000EVM requires external 5V power.

    ??? danger "Both power supplies have the same barrel jack size, and mixing them up will immediately destroy the DCA1000EVM."

        Plugging the 12V radar supply into the DCA1000EVM kills it instantly; there is no protection and no second chance. Don't ask me how I know this!

    We recommend labeling the power supplies at the barrel jack end, as well as connecting the radar power supply first (so that if the radar does not power on, you know that you've mixed them up).

1. Prepare for flashing.

    - Plug a USB cable into the XDS port (on the right side).
    - Find `SOP2:0` (SOP2 is `J17`, SOP1 is `J18`, SOP0 is `J20`), and short SOP0 and SOP2 (`SOP2:0=101`). These are physical jumpers, which must be shorted using jumper caps or wires.
    - Power on the radar using external 12V power.

    !!! info

        The SOP pins are only read when the radar boots. If you change the jumpers while the radar is powered, press `NRST` or power cycle the radar.

2. Flash using [TI UniFlash](#flashing-with-uniflash) (AWR2944EVM) or [`uart_uniflash.py`](#flashing-with-uart_uniflashpy) (AWR2944PEVM, AWR2E44PEVM).

3. Switch the radar to functional mode.

    - Remove the jumper on SOP2, so only a jumper on SOP0 remains (`SOP2:0=001`).
    - Press `NRST` (or power cycle the radar) to boot the newly flashed firmware.

## Flashing with UniFlash

!!! note

    UniFlash seems to work most reliably on Windows.

- UniFlash should automatically discover the radar. If not, select the `AWR2944EVM` device.
- Select the `awr2944_mmw_demoTDM.appimage` and `sbl_qspi.release.tiimage` images to flash.
- Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
- Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

??? failure "`Not able to connect to serial port. Recheck COM port selected and/or permissions.`"

    If the SOP jumpers are not in the correct position, flashing will fail with
    ```
    Not able to connect to serial port.
    Recheck COM port selected and/or permissions.
    ```

## Flashing with `uart_uniflash.py`

!!! bug

    Flashing the AWR2944PEVM using TI UniFlash does not work, so we use TI's `uart_uniflash.py` script instead (see the [mmWave MCU Plus SDK user guide](https://dr-download-cdn.ti.com/software-development/software-development-kit-sdk/MD-U4MY7aGNn5/04.07.01.03/mmwave_mcuplus_sdk_user_guide.pdf) for details).

In addition to the application and bootloader images, you will need the temporary flash-writer image `tools/awr2x44P/sbl_uart_uniflash.release.tiimage`, which the flashing script loads onto the radar to perform the flash.

=== "Linux"

    - Set up the flashing environment from the root of the `xwr` repository. The `flash/` project contains the script's dependencies (`pyserial`, `xmodem`, `tqdm`):
        ```sh
        uv sync --project flash
        ```
    - Set paths to the SDK and to the `xwr` repository (adjust the versions to match your install):
        ```sh
        export XWR=$(pwd)
        export SDK=~/ti/mmwave_mcuplus_sdk_04_07_02_01/mmwave_mcuplus_sdk_04_07_02_01
        export MCU_PLUS=~/ti/mmwave_mcuplus_sdk_04_07_02_01/mcu_plus_sdk_awr2x44p_10_02_00_04
        ```
    - Find the serial port. The XDS110 shows up as two serial ports (e.g. `/dev/ttyACM0` and `/dev/ttyACM1`); use the lower-numbered one, which is the `XDS110 Class Application/User UART`.
        ```sh
        ls -l /dev/serial/by-id/
        ```

        ??? failure "Permission denied"

            Your user needs to be in the `dialout` group to access serial ports:
            ```sh
            sudo usermod -aG dialout $USER
            ```
            Then log out and back in.

    - The SDK's `default.cfg` flashes a debug image instead of the demo. Make a copy which flashes the TDM demo instead:
        ```sh
        cd $SDK/tools/awr2x44P
        sed 's#ti/utils/ccsdebug/awr2x44P_ccsdebug.appimage#ti/demo/awr2x44P/mmw_tdm/awr2x44P_mmw_demoTDM.appimage#' default.cfg > tdm.cfg
        ```
        This should result in the following change:
        ```diff
        - --file=../../ti/utils/ccsdebug/awr2x44P_ccsdebug.appimage --operation=flash --flash-offset=0xA0000
        + --file=../../ti/demo/awr2x44P/mmw_tdm/awr2x44P_mmw_demoTDM.appimage --operation=flash --flash-offset=0xA0000
        ```
    - Then, flash the radar (replacing `/dev/ttyACM0` with the radar's serial port). This must be run from `$SDK/tools/awr2x44P`, since the paths in the config file are relative to it.
        ```sh
        cd $SDK/tools/awr2x44P
        uv run --project $XWR/flash python $MCU_PLUS/tools/boot/uart_uniflash.py -p /dev/ttyACM0 --cfg=tdm.cfg
        ```
        Flashing takes around 1 minute; each of the three steps should end with `[STATUS] SUCCESS !!!`.

=== "Windows"

    - Make sure you have `python` installed, along with `pyserial`, `xmodem`, `tqdm`:
        ```sh
        pip install pyserial xmodem tqdm
        ```
    - Set the correct device in `C:\ti\mmwave_mcuplus_sdk_{version}\mmwave_mcuplus_sdk_{version}\scripts\windows\setenv.bat` and ensure that all software versions are correct:
        ```
        set MMWAVE_SDK_DEVICE=awr2x44P
        ```
    - Set the environment variables by executing the script:
        ```
        cd C:\ti\mmwave_mcuplus_sdk_{version}\mmwave_mcuplus_sdk_{version}\scripts\windows
        setenv.bat
        ```
    - Find the COM port corresponding to the `XDS110 Class Application/User UART` port.
    - The SDK's `default.cfg` flashes a debug image instead of the demo. Update the config in `C:\ti\mmwave_mcuplus_sdk_{version}\mmwave_mcuplus_sdk_{version}\tools\awr2x44P\default.cfg` to flash the TDM demo instead:
        ```diff
        - --file=../../ti/utils/ccsdebug/awr2x44P_ccsdebug.appimage --operation=flash --flash-offset=0xA0000
        + --file=../../ti/demo/awr2x44P/mmw_tdm/awr2x44P_mmw_demoTDM.appimage --operation=flash --flash-offset=0xA0000
        ```
    - Then, run the following (replacing `{COM Port}` with the radar board's COM port):
        ```
        cd %MMWAVE_SDK_INSTALL_PATH%\tools\awr2x44P
        python %MCU_PLUS_AWR2X44P_INSTALL_PATH%\tools\boot\uart_uniflash.py -p {COM Port} --cfg=default.cfg
        ```

??? failure "`send error: expected NAK, CRC, EOT or CAN; got b'd'`"

    When it boots, the radar prints a few hundred characters of status output (`d0003010000...`) before it starts sending `C` every 3 seconds to signal that it is ready to receive. If this output is still sitting in the serial buffer, the flashing script reads it first, counts 10 errors, and aborts with `XMODEM send failed`.

    To fix this, drain the buffer and check that the radar is sending `C`, then immediately run the flashing command again. Using the same Python environment as the flashing script (on Linux, `uv run --project $XWR/flash python`), and replacing `/dev/ttyACM0` with the radar's serial port:
    ```sh
    python -c "
    import serial
    s = serial.Serial('/dev/ttyACM0', 115200, timeout=4)
    s.reset_input_buffer(); print(s.read(1024))"
    ```
    This should print `b'C'` (possibly preceded by leftover `0`s). If nothing is printed, check the SOP jumpers and press `NRST`.
