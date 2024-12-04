"""DCA1000EVM API."""

import logging
import socket
import struct
import threading
import time

import numpy as np
from beartype.typing import Iterator, Optional

from . import dca_types as types


class DCA1000EVM:
    """DCA1000EVM Interface.

    Documented by [R1]_; based on a (little-endian) UDP protocol (Sec 5).
    Included C++ source code exerpts from the mmWave API are used as a
    secondary reference [R2]_.

    Args:
        sys_ip: IP of this computer associated with the desired ethernet
            interface.
        fpga_ip: Static IP of the DCA1000EVM FPGA.
        data_port: Port used for recording data.
        config_port: Port used for configuration.
        timeout: Config socket read timeout.
        socket_buffer: Receive socket buffer size. Ensure that `socket_buffer`
            is less than `/proc/sys/net/core/rmem_max`.
        name: Human-readable name.

    Raises:
        TimeoutError: request timed out (is the device connected?).
        DCAException: exception raised by the FPGA.

    Usage:
        (1) Initialization parameters can be defaults.
        (2) Setup with `.setup(...)` with the appropriate `LVDS.TWO_LANE`
            (1843, 1642) or `LVDS.FOUR_LANE` (1243, 1443).
        (3) Start recording with `.start(...)`.
        (4) Start the radar.
            NOTE: the radar should be started afer `DCA1000EVM.start` to ensure
            that the data are correctly aligned with respect to the byte count.
        (5) Stop recording with `.stop()`.
        (6) Stop the radar.
    """

    _MAX_PACKET_SIZE = 2048

    def _create_socket(
        self, addr: tuple[str, int], timeout: float
    ) -> socket.socket:
        """Create socket."""
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.bind(addr)
        sock.settimeout(timeout)
        self.log.info("Connected to {}:{}".format(*addr))
        return sock

    def __init__(
        self, sys_ip: str = "192.168.33.30", fpga_ip: str = "192.168.33.180",
        data_port: int = 4098, config_port: int = 4096, timeout: float = 1.0,
        socket_buffer: int = 2048000, name: str = "DCA1000EVM"
    ) -> None:
        self.log = logging.getLogger(name=name)

        self.sys_ip = sys_ip
        self.fpga_ip = fpga_ip
        self.config_port = config_port
        self.data_port = data_port
        self.recording = False
        self.thread: Optional[threading.Thread] = None

        self.config_socket = self._create_socket(
            (sys_ip, config_port), timeout)
        self.data_socket = self._create_socket((sys_ip, data_port), 0.0)
        self.data_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, socket_buffer)

        self.timeout = timeout

        self._warn_ooo_counter = 0

    def flush(self) -> None:
        """Clear data receive buffers."""
        self.data_socket.settimeout(0.0)
        try:
            while True:
                data = self.data_socket.recv(self._MAX_PACKET_SIZE)
                if not data:
                    break
        except BlockingIOError:
            pass

        self.data_socket.settimeout(self.timeout)

    def setup(
        self, delay: float = 5.0, lvds=types.LVDS.TWO_LANE
    ) -> None:
        """Configure DCA1000EVM capture card.

        Args:
            delay: packet delay in microseconds.
            lvds: `FOUR_LANE` or `TWO_LANE`; select based on radar.
        """
        self.system_aliveness()
        self.read_fpga_version()
        self.configure_record(delay=delay)
        self.configure_fpga(
            log=types.Log.RAW_MODE, lvds=lvds,
            transfer=types.DataTransfer.CAPTURE,
            format=types.DataFormat.BIT16,
            capture=types.DataCapture.ETH_STREAM)

    def _recv(self) -> Optional[types.DataPacket]:
        """Receive data.

        NOTE: due to high packet rates (up to 200KHz), we only busy-wait, and
        manually track the timeout using `perf_counter`.
        """
        timeout = time.perf_counter() + self.timeout
        while True:
            try:
                raw, _ = self.data_socket.recvfrom(self._MAX_PACKET_SIZE)
                return types.DataPacket.from_bytes(raw)
            except BlockingIOError:
                if time.perf_counter() > timeout:
                    return None

    def _warn_ooo(self, missing: int) -> None:
        """Out of order packet warning."""
        self._warn_ooo_counter += 1
        if self._warn_ooo_counter < 10:
            self.log.error("Out of order packet: {} bytes".format(missing))
        if self._warn_ooo_counter == 10:
            self.log.error("Suppressing 'out of order' on the 10th trigger.")

    def stream(
        self, shape: list[int] = []
    ) -> Iterator[types.RadarFrame]:
        """Get a python iterator corresponding to the data stream.

        NOTE: `shape` should have twice as many samples on the last axis
        to account for two IQ uint16s per sample. Note that these samples are
        also in IIQQ order, not IQ order.
        """
        size = int(np.prod(shape)) * np.dtype(np.uint16).itemsize

        offset = 0
        timestamp = 0.0
        buf = bytearray()
        while True:
            packet = self._recv()
            if packet is None:
                return

            if offset == 0:
                offset = packet.byte_count - (packet.byte_count % size)
                timestamp = time.time()

            complete = True
            missing = packet.byte_count - offset
            if missing < 0:
                self._warn_ooo(missing)
            else:
                if missing > 0:
                    self.log.warn("Dropped packets: {} bytes".format(missing))
                    buf.extend(b'\x00' * missing)
                    offset = packet.byte_count
                    complete = False

                buf.extend(packet.data)
                offset += len(packet.data)

            # Write out if the buffer contains complete
            while len(buf) >= size:
                yield types.RadarFrame(
                    timestamp=timestamp, data=buf[:size], complete=complete)
                buf[:size] = b''

                # Update timestamp for remainder
                if len(buf) < size:
                    timestamp = time.time()

    def system_aliveness(self) -> None:
        """Simple ping to query system status."""
        cmd = types.Request(types.Command.SYSTEM_ALIVENESS, bytes())
        self._config_request(cmd, desc="Verify FPGA connectivity")

    def reset_ar_device(self) -> None:
        """Reset (i.e. reboot) Radar (AR - Automotive Radar) device."""
        cmd = types.Request(types.Command.RESET_AR_DEV, bytes())
        self._config_request(cmd, "Reset AR Device")

    def configure_fpga(
        self, lvds=types.LVDS.TWO_LANE, log=types.Log.RAW_MODE,
        transfer=types.DataTransfer.CAPTURE, format=types.DataFormat.BIT16,
        capture=types.DataCapture.ETH_STREAM
    ) -> None:
        """Configure FPGA.

        NOTE: This seems to cause the FPGA to ignore requests for a short time
        after. Sending `system_aliveness` pings until it responds seems to be
        the best way to check when it's ready again.

        Args:
            lvds: `FOUR_LANE` or `TWO_LANE`; select based on radar.
            log: raw or data separated mode; we assume `RAW_MODE`.
            transfer: capture or playback; we assume `CAPTURE`.
            format: ADC bit depth; we assume `BIT16`.
            capture: data capture mode; we assume `ETH_STREAM`.
        """
        self.log.info("Configuring FPGA: {}, {}, {}, {}, {}".format(
            log, lvds, transfer, capture, format))
        cfg = struct.pack(
            "BBBBBB", log.value, lvds.value, transfer.value,
            capture.value, format.value, types.FPGA_CONFIG_DEFAULT_TIMER)
        cmd = types.Request(types.Command.CONFIG_FPGA, cfg)
        self._config_request(cmd, desc="Configure FPGA")

        self.log.info("Testing/waiting for FPGA to respond to new requests.")
        for _ in range(30):
            try:
                return self.system_aliveness()
            except TimeoutError:
                pass
        else:
            msg = "FPGA stopped responding to requests after configuring."
            self.log.error(msg)
            raise TimeoutError(msg)

    def configure_eeprom(
        self, sys_ip: str = "192.168.33.30", fpga_ip: str = "192.168.33.180",
        fpga_mac: str = "12:34:56:78:90:12",
        config_port: int = 4096, data_port: int = 4098
    ) -> None:
        """Configure EEPROM; contains IP, MAC, port information.

        NOTE: Use with extreme caution. This should never be used in normal
        operation. May require delay before use depending on the previous cmd.

        If this operation messes up the system IP and FPGA IP, the radar needs
        to be switched to hard-coded IP mode (user switch 1; see [1]) and
        the EEPROM correctly reprogrammed.
        """
        cfg = struct.pack(
            "B" * (4 + 4 + 6) + "HH",
            *types.ipv4_to_int(sys_ip), *types.ipv4_to_int(fpga_ip),
            *types.mac_to_int(fpga_mac), config_port, data_port)

        cmd = types.Request(types.Command.CONFIG_EEPROM, cfg)
        self._config_request(cmd, desc="Configure EEPROM")

    def start(self) -> None:
        """Start recording data."""
        cmd = types.Request(types.Command.START_RECORD, bytes())
        self._config_request(cmd, desc="Start recording")

    def stop(self) -> None:
        """Stop recording data."""
        cmd = types.Request(types.Command.STOP_RECORD, bytes())
        self._config_request(cmd, desc="Stop recording")

    def read_fpga_version(self) -> tuple[int, int, bool]:
        """Get current FPGA version."""
        cmd = types.Request(types.Command.READ_FPGA_VERSION, bytes())
        resp = self._config_request(cmd)
        if resp.status < 0:
            self.log.error(
                "Unable to read FPGA version: {}".format(resp.status))
            return (0, 0, False)
        else:
            major = resp.status & 0x7F
            minor = 0x7f & (resp.status >> 7)
            playback = resp.status & 0x4000
            self.log.info(
                "FPGA Version: {}.{} [mode={}]".format(
                    major, minor, "playback" if playback else "record"))
            return (major, minor, playback != 0)

    def configure_record(self, delay: float = 25.0) -> None:
        """Configure data packets (with a packet delay in us).

        The packet delay must be between 5 and 500 us (Table 19, [1]). This
        sets the theoretical maximum throughput to between 193 and 706 Mbps.
        """
        assert 5.0 <= delay
        assert delay <= 500.0

        converted = int(
            delay * types.FPGA_CLK_CONVERSION_FACTOR
            / types.FPGA_CLK_PERIOD_IN_NANO_SEC)
        cfg = struct.pack("HHH", types.MAX_BYTES_PER_PACKET, converted, 0)
        cmd = types.Request(types.Command.CONFIG_RECORD, cfg)
        self._config_request(cmd, desc="Configure recording")

    def _config_request(
        self, cmd: types.Request, desc: Optional[str] = None
    ) -> types.Response:
        """Send config command."""
        payload = cmd.to_bytes()
        self.config_socket.sendto(payload, (self.fpga_ip, self.config_port))
        self.log.debug("Sent: {}".format(cmd))

        raw, _ = self.config_socket.recvfrom(self._MAX_PACKET_SIZE)
        response = types.Response.from_bytes(raw)
        self.log.debug("Received: {}".format(response))

        if desc is not None:
            if response.status == 0:
                self.log.info("Success: {}".format(desc))
            else:
                msg = "Failure: {} (status={})".format(desc, response.status)
                self.log.error(msg)
                raise types.DCAException(msg)
        return response
