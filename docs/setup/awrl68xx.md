# AWRL68XX

After installing the [MMWAVE-L-SDK](https://www.ti.com/tool/MMWAVE-L-SDK), the demo firmware can be found at the following paths in the install directory:

| Radar | Firmware |
| ----- | -------- |
| AWRL6844EVM | `examples/mmw_demo/mmwave_demo/prebuilt_binaries/xwrL684x-evm/mmwave_demo.release.appimage` |

The switch positions below follow [this flashing guide](https://www.ti.com/lit/ug/swru636/swru636.pdf?ts=1772012346542).

1. Prepare for flashing.

    - Plug in a USB cable to the XDS port (on the right side).
    - Set the large "flashing mode" SOP switches to 00 (labeled "flashing mode" on the PCB).
    - Set the undocumented switches (S5) to OFF.
    - Set all S1 switches to off except S1.2, which should be on.
    - Set all S4 switches to off.

2. Flash using [TI UniFlash](https://www.ti.com/tool/UNIFLASH).

    - Select `AWRL68XX` as the device. This will not be detected automatically.
    - Select the `mmwave_demo.release.appimage` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description `XDS110 Class Application/User UART`.
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".

3. Switch the radar to functional mode.

    - Set the large "flashing mode" SOP switches to 01 (labeled "functional mode" on the PCB).
    - Set the undocumented switches (S5) to OFF.
    - Set all S1 switches to off except S1.2, which should be on.
    - Set all S4 switches to off except S4.6, which should be on.
