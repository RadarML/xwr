"""TI radar demo firmware API base class."""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803

import logging
import re
import time

import serial
from serial.tools import list_ports

from .raw import APIMixins, BoilerplateMixins


class XWRError(Exception):
    """Error raised by the Radar (via non-normal return message)."""

    pass


class XWRBase(APIMixins, BoilerplateMixins):
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

    !!! info

        If the radar serial port is not provided, we auto-detect the port by
        fetching the lowest-numbered one which contains "XDS110" in the USB
        device description (Or "CP2105 ... Enhanced" in the case of the
        AWR1843AOPEVM), which corresponds to the [TI XDS110 JTAG debugger](
        https://www.ti.com/tool/TMDSEMU110-U) embedded in each radar dev board.

    Args:
        port: radar control serial port; typically the lower numbered one. If
            not provided (`None`), we attempt to auto-detect the port.
        baudrate: baudrate of control port.
        name: human-readable name.

    Attributes:
        NUM_TX: number of TX antennas.
        NUM_RX: number of RX antennas.
    """

    _CMD_PROMPT = "mmwDemo:/>"
    _PORT_NAME = r"XDS110"

    NUM_TX: int = 3
    NUM_RX: int = 4

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        self.log: logging.Logger = logging.getLogger(name=name)

        if port is None:
            port = self.__detect_port()
            self.log.info(f"Auto-detected port: {port}")

        self.port: serial.Serial = serial.Serial(port, baudrate, timeout=None)

        # Only linux supports low latency mode.
        if hasattr(self.port, 'set_low_latency_mode'):
            self.port.set_low_latency_mode(True)
        else:
            self.log.warning(
                "Low latency mode is only supported on linux. This may cause "
                "initialization to take longer than expected.")

        self.port.reset_input_buffer()
        self.port.reset_output_buffer()

    def __detect_port(self) -> str:
        sorted_ports = sorted(list_ports.comports(), key=lambda x: x.device)
        for port in sorted_ports:
            if port.description is not None:
                if re.match(self._PORT_NAME, port.description, re.IGNORECASE):
                    return port.device

        self.log.error("Failed to auto-detect radar port.")
        raise XWRError(
            "Auto-detecting the radar port (`port=None`) failed: none of the "
            f"available ports contain '{self._PORT_NAME}' in the "
            "USB description. "
            f"Available ports: {[p.device for p in list_ports.comports()]}")

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
            timeout: timeout, in seconds.

        Raises:
            TimeoutError: if no response is received by the timeout.
        """
        self.log.debug("Send: {}".format(cmd))
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
        decoded = rx_buf.decode('utf-8', errors='replace')
        resp = (
            decoded
            .replace(self._CMD_PROMPT, '').replace(cmd, '')
            .rstrip(' ;\r\n\t').lstrip(' \n\t'))
        self.log.log(5, "Response: {}".format(resp))

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
                self.log.info(f"Raw buffer for this error was: {decoded}")
                raise XWRError(resp)

    def start(self, reconfigure: bool = True) -> None:
        """Start radar.

        Args:
            reconfigure: Whether the radar needs to be configured.
        """
        if reconfigure:
            self.send("sensorStart")
        else:
            self.send("sensorStart 0")

        self.log.info("Radar Started.")

    def stop(self) -> None:
        """Stop radar.

        !!! warning

            The radar may be non-responsive to commands in some conditions, for
            example if the specified timings are fairly tight (or invalid).
        """
        self.send("sensorStop")
        self.log.info("Radar Stopped.")

    def _configure_channels(self, rx: int = 0b1111, tx: int = 0b111) -> None:
        """Configure RX and TX channels with sequential chirps.

        Args:
            rx: RX channel mask, e.g. `0b1111` for 4 RX antennas.
            tx: TX channel mask, e.g. `0b111` for 3 TX antennas.
        """
        self.channelCfg(rxChannelEn=rx, txChannelEn=tx)
        chirp = 0
        i_tx = 0
        while tx > 0:
            if tx & 1:
                self.chirpCfg(chirpIdx=chirp, txEnable=i_tx)
                chirp += 1
            tx = tx >> 1
            i_tx += 1
