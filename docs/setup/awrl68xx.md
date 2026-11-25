# AWRL68XX

After installing the [MMWAVE-L-SDK](https://www.ti.com/tool/MMWAVE-L-SDK), the demo firmware can be found at the following paths in the install directory:

| Radar | Firmware |
| ----- | -------- |
| AWRL6844EVM | `examples/mmw_demo/mmwave_demo/prebuilt_binaries/xwrL684x-evm/mmwave_demo.release.appimage` |
| AWRL6888EVM | `examples/mmw_demo/mmwave_demo_L688x/prebuilt_binaries/xwrL688x-evm/mmwave_demo.release.appimage` |

!!! info

    AWRL6888 support was added in MMWAVE-L-SDK `06.01.03.04`; earlier versions do not include the `mmwave_demo_L688x` firmware.

Both radars have the same switch layout and use the same switch settings. For the full switch tables, see TI's [AWRL6844EVM flashing guide (SWRU636)](https://www.ti.com/lit/ug/swru636/swru636.pdf?ts=1772012346542) or [AWRL6888EVM user's guide (SWRU647)](https://www.ti.com/lit/pdf/SWRU647), section 2.3.

!!! warning "The S1 switch is installed upside down"

    S1 is ON when the switch is down and OFF when it is up.

1. Prepare for flashing.

    - Plug a USB cable into the XDS port (J5, on the right side of the board).
    - Set the SOP switches (S8 = SOP1, S7 = SOP0; labeled "flashing mode" and "functional mode" on the PCB) to 00 (flashing mode).
    - Set the undocumented switches (S5) to OFF.
    - Set the S1 and S4 DIP switches to the following positions:

        |    | 1   | 2   | 3   | 4   | 5   | 6   |
        | -- | --- | --- | --- | --- | --- | --- |
        | S1 | any | OFF | any | ON  | ON  | any |
        | S4 | any | any | any | OFF | OFF | OFF |

    - Press the reset button (S2), or power cycle the radar. The SOP switches are only read when the radar boots.

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH) on Windows.

    !!! bug "Flash from Windows"

        As of November 2026, UniFlash (and its `dslite` command line tool) hangs at the bootloader handshake ("Checking fileType appropriateness for this device") on Linux, even with the correct switch settings. Flashing with UniFlash on Windows works.

    - Select `AWRL68XX` as the device. This will not be detected automatically.
    - Select the firmware image listed above to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

3. Switch the radar to functional mode.

    - Set the SOP switches to 01 (functional mode).
    - Leave S1 and S5 as is, and set S4.6 to ON:

        |    | 1   | 2   | 3   | 4   | 5   | 6   |
        | -- | --- | --- | --- | --- | --- | --- |
        | S1 | any | OFF | any | ON  | ON  | any |
        | S4 | any | any | any | OFF | OFF | ON  |

    - Press the reset button (S2), or power cycle the radar. The radar CLI uses UART-A via the XDS110 (usually `/dev/ttyACM0`), which `xwr` detects automatically.
