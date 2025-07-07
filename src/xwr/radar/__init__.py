"""Interface implementation for generic TI demo MSS firmware.

!!! abstract "Usage"

    After selecting the appropriate `XWRXXXX` interface:

    1. Initialization parameters can be defaults. The `port` may need to
        be changed if multiple radars are being used, or another device
        uses the `/dev/ttyACM0` default name. The baudrate should not be
        changed.
    2. Setup with `.setup(...)` with the desired radar configuration.
    3. Start the radar with `.start()`.
    4. Stop the radar with `.stop()`.

!!! info

    You may need to provide read/write permissions to access the serial port:
    ```sh
    sudo chmod 777 /dev/ttyACM0  # or whatever port the radar is on.
    ```

!!! danger

    If the configuration is invalid, `.start()` may return an error, or
    cause the radar to freeze. This may require the radar to be rebooted
    via manually disconnecting the power supply.
"""

from . import defines
from .api import AWR1642, AWR1843, AWR1843L  # , AWR2544
from .base import XWRBase, XWRError

__all__ = [
    "defines", "AWR1642", "AWR1843", "AWR1843L",   # "AWR2544",
    "XWRError", "XWRBase"]
