# DCA1000EVM Capture Card Setup

!!! warning

    In our experience, the DCA1000EVM is particularly fragile; be careful with electrostatic discharge (ESD).

Ensure that the following DIP switches are set:

- SW2.5: `SW_CONFIG`
- SW2.6: `USER_SW1` (the marked right side), unless the EEPROM is messed up from a misconfigured `configure_eeprom(...)` call.

The following are configured by `configure_fpga(...)` under normal operation, but can be manually set in case that isn't working:

- SW1: 16-bit mode (`16BIT_ON`, `14BIT_OFF`, `12BIT_OFF`).
- SW2.1: `LVDS_CAPTURE`
- SW2.2: `ETH_STREAM`
- SW2.3: `AR1642_MODE` (2-lane LVDS)
- SW2.4: `RAW_MODE`
- SW2.5: `HW_CONFIG`
