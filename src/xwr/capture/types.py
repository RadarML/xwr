"""DCA1000EVM capture card API data types."""

import struct
from dataclasses import dataclass
from typing import cast

from .defines import Command


def ipv4_to_int(ipv4: str) -> tuple[int, int, int, int]:
    """Parse ipv4 string as a tuple of 4 integers."""
    addr = tuple(reversed(list(int(x) for x in ipv4.split('.'))))
    return cast(tuple[int, int, int, int], addr)


def mac_to_int(mac: str) -> tuple[int, int, int, int, int, int]:
    """Parse MAC address string as a tuple of 6 integers."""
    addr = tuple(reversed(list(int(x, 16) for x in mac.split(':'))))
    return cast(tuple[int, int, int, int, int, int], addr)


@dataclass
class Request:
    """Command request protocol."""

    cmd: Command
    data: bytes | bytearray

    def to_bytes(self) -> bytes:
        """Form into a single packet.

        Data format: `<HHH{}sH`.

        - `<` : assumed to be little endian. Not documented anywhere, but
          implied since mmWave API uses native linux/x86 structs, which are
          little endian.
        - `H` : Header is always `0xA55A` (Table 13[^1]).
        - `H` : Command code (Table 12[^1]).
        - `H` : Data size; must be between 0 and 504 (Section 5.1[^1]).
        - `{}s` : Payload; can be empty.
        - `H` : Footer is always `0xEEAA` (Table 13[^1]).

        [^1]: [DCA1000EVM Data Capture Card User's Guide (Rev A)](
        https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf?ts=1709104212742).
        """
        assert len(self.data) < 504
        return struct.pack(
            "<HHH{}sH".format(len(self.data)),
            0xa55a, self.cmd.value, len(self.data), self.data, 0xeeaa)


@dataclass
class Response:
    """Command response protocol."""

    cmd: int
    status: int

    @classmethod
    def from_bytes(cls, packet: bytes) -> "Response":
        """Read packet."""
        header, command_code, status, footer = struct.unpack("HHHH", packet)
        assert header == 0xa55a
        assert footer == 0xeeaa
        return cls(cmd=command_code, status=status)


@dataclass
class DataPacket:
    """Data packet protocol."""

    sequence_number: int
    byte_count: int
    data: bytes | bytearray

    @classmethod
    def from_bytes(cls, packet: bytes) -> "DataPacket":
        """Read packet.

        Packet format (Sec. 5.2[^1]):

        - `<` : assumed to be little endian.
        - `L` : 4-byte sequence number (packet number).
        - `Q` : 6-byte byte count index; appended with x0000 to make a uint64.

        [^1]: [DCA1000EVM Data Capture Card User's Guide (Rev A)](
            https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf?ts=1709104212742).
        """
        sn, bc = struct.unpack('<LQ', packet[:10] + b'\x00\x00')
        return cls(sequence_number=sn, byte_count=bc, data=packet[10:])


@dataclass
class RadarFrame:
    """Radar frame, in IIQQ format (Fig 11[^1]).

    [^1]: [MMwave Radar Device ADC Raw Capture Data](
    https://www.ti.com/lit/an/swra581b/swra581b.pdf?ts=1609161628089)

    !!! warning

        Assuming the radar/capture card are configured for 16-bit capture and
        [`SampleSwap.MSB_LSB_IQ`][xwr.radar.defines.SampleSwap] order,
        the output data use an interleaved Complex32 format consisting of real
        (I: in-phase) and complex (Q: quadrature) `i16` parts.

        Since the output is little-endian, `MSB_LSB_IQ` indicates that `I`
        is in the MSB, i.e., comes last, and the `Q` in the LSB comes first.

        For example, if there are two LVDS lanes (e.g., AWR1843, AWR2544), each
        lane takes the following structure:

        ```
        Lane 0  | Q[0] | I[0] | Q[2] | I[2] | ...
        Lane 1  | Q[1] | I[1] | Q[3] | I[3] | ...
        ```

        These lanes are then interleaved by the capture card:

        ```
        Output  | Q[0] | Q[1] | I[0] | I[1] | Q[2] | Q[3] | I[2] | I[3] | ...
        ```

    ??? example "Code snippet: interpreting the `data`"

        ```python
        shape = [64, 4, 2, 128]  # shape: (chirps, tx, rx, samples)
        iiqq = np.frombuffer(
            frame.data, dtype=np.int16
        ).reshape([*shape[:-1], shape[-1] * 2])
        iq = np.zeros(shape, dtype=np.complex64)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
        ```

    Attributes:
        timestamp: system timestamp of the first packet received for this frame.
        data: radar frame data.
        complete: whether the frame is "complete"; if `False`, this frame
            includes zero-filled data.
    """

    timestamp: float
    data: bytes | bytearray
    complete: bool
