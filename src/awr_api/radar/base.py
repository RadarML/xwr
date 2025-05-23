"""TI radar demo firmware API base class."""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803

import logging
import time

import serial

from .raw import APIMixins, BoilerplateMixins


class AWRException(Exception):
    """Error raised by the Radar (via non-normal return message)."""

    pass


class AWRBase(APIMixins, BoilerplateMixins):
    """Generic AWR Interface for the TI demo MSS firmware.

    The interface is based on a UART ASCII CLI, and is documented by the
    following sources:

    - The `packages/ti/demo/xwr18xx/mmw` folder in the mmWave SDK install.
    - [mmWave SDK user guide, Table 1 (Page 19)](
        https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf)
    - [mmWave Studio](https://www.ti.com/tool/MMWAVE-STUDIO)
    - [AWR1843 Data Sheet](
        https://www.ti.com/lit/ds/symlink/awr1843.pdf?ts=1708800208074)

    !!! warning

        We only implement a partial API. Non-mandatory calls which do not
        affect the LVDS raw I/Q stream are not implemented.

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _CMD_PROMPT = "\rmmwDemo:/>"

    def __init__(
        self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        self.log = logging.getLogger(name=name)
        self.port = serial.Serial(port, baudrate, timeout=None)
        self.port.set_low_latency_mode(True)

    def setup_from_config(self, path: str) -> None:
        """Run raw setup from a config file."""
        with open(path) as f:
            cmds = f.readlines()
        for c in cmds:
            self.send(c.rstrip('\n'))

    def setup(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def send(self, cmd: str, timeout: float = 10.0) -> None:
        """Send message, and wait for a response.

        Args:
            cmd: command to send.
            timeout: raises `TimeoutError` if the expected response is not
                received by this time.
        """
        self.log.info("Send: {}".format(cmd))
        self.port.write((cmd + '\n').encode('ascii'))

        # Read until we get "...\rmmwDemo:/>"
        rx_buf = bytearray()
        prompt = self._CMD_PROMPT.encode('utf-8')
        start = time.time()
        while not rx_buf.endswith(prompt):
            rx_buf.extend(self.port.read(self.port.in_waiting))
            if time.time() - start > timeout:
                self.log.error("Timed out while waiting for response.")
                raise TimeoutError()

        # Remove all the cruft
        resp = (
            rx_buf.decode('utf-8', errors='replace')
            .replace(self._CMD_PROMPT, '').replace(cmd, '')
            .rstrip(' \n\t').lstrip(' \n\t')
            .replace('\n', '; ').replace('\r', ''))
        self.log.debug("Response: {}".format(resp))

        # Check for non-normal response
        if resp != 'Done':
            if resp.startswith("Ignored"):
                self.log.warning(resp)
            elif resp.startswith("Debug") or resp.startswith("Skipped"):
                if "Error" in resp:
                    self.log.error(resp)
            elif '*****' in resp:
                pass  # header
            else:
                self.log.error(resp)
                raise AWRException(resp)

    def start(self, reconfigure: bool = True) -> None:
        """Start radar.

        Args:
            reconfigure: Whether the radar needs to be configured.
        """
        if reconfigure:
            self.send("sensorStart")
        else:
            self.send("sensorStart 0")

    def stop(self) -> None:
        """Stop radar.

        !!! warning

            The radar may be non-responsive to commands in some conditions, for
            example if the specified timings are fairly tight (or invalid).
        """
        self.send("sensorStop")
