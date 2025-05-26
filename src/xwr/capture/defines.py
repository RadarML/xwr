"""DCA1000EVM capture card constants, enums, and other definitions."""

from enum import Enum


class Command(Enum):
    """Command request codes; see `rf_api.h:CMD_CODE_*`."""

    RESET_FPGA = 0x01
    RESET_AR_DEV = 0x02
    CONFIG_FPGA = 0x03
    CONFIG_EEPROM = 0x04
    START_RECORD = 0x05
    STOP_RECORD = 0x06
    START_PLAYBACK = 0x07
    STOP_PLAYBACK = 0x08
    SYSTEM_ALIVENESS = 0x09
    ASYNC_STATUS = 0x0A
    CONFIG_RECORD = 0x0B
    CONFIG_AR_DEV = 0x0C
    INIT_FPGA_PLAYBACK = 0x0D
    READ_FPGA_VERSION = 0x0E


class Log(Enum):
    """Data log mode; see `rf_api.h:enum CONFIG_LOG_MODE`."""

    RAW_MODE = 1
    MULTI_MODE = 2


class LVDS(Enum):
    """LVDS mode (number of lanes); see `rf_api.h:enum CONFIG_LVDS_MODE`.

    Attributes:
        FOUR_LANE: 4-lane mode, e.g., AR1243.
        TWO_LANE: 2-lane mode; much more common, e.g. AR1642, AWR1843, AWR2544.
    """

    FOUR_LANE = 1
    TWO_LANE = 2


class DataTransfer(Enum):
    """Data transfer mode; see `rf_api.h:enum CONFIG_TRANSFER_MODE`.

    Attributes:
        CAPTURE: capture mode (the normal mode used).
        PLAYBACK: play back data from an onboard SD card/SSD.
    """

    CAPTURE = 1
    PLAYBACK = 2


class DataFormat(Enum):
    """Data format (bit depth); see `rf_api.h:enum CONFIG_FORMAT_MODE`.

    Attributes:
        BIT12: 12-bit mode.
        BIT14: 14-bit mode.
        BIT16: 16-bit mode (used by default).
    """

    BIT12 = 1
    BIT14 = 2
    BIT16 = 3


class DataCapture(Enum):
    """Data capture mode; see `rf_api.h:enum CONFIG_CAPTURE_MODE`.

    Attributes:
        SD_STORAGE: store to SD card.
        ETH_STREAM: stream to ethernet interface (use this normally).
    """

    SD_STORAGE = 1
    ETH_STREAM = 2


class Status:
    """Status codes.

    Attributes:
        SUCCESS: success.
        FAILURE: failure.
    """

    SUCCESS = 0
    FAILURE = 1


class DCAConstants:
    """DCA1000EVM capture card API constants.

    Collected from the [DCA1000EVM User's Guide](
    https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf?ts=1709104212742) and the
    reference API at `ReferenceCode/DCA1000/SourceCode` in a [mmWave Studio
    install](https://www.ti.com/tool/MMWAVE-STUDIO).
    """

    FPGA_CLK_CONVERSION_FACTOR = 1000
    """Record packet delay clock conversion factor."""

    FPGA_CLK_PERIOD_IN_NANO_SEC = 8
    """Record packet delay clock period in ns."""

    FPGA_CONFIG_DEFAULT_TIMER = 30
    """LVDS timeout is always 30 (units not documented / unknown)."""

    MAX_BYTES_PER_PACKET = 1470
    """Maximum number of bytes in a single FPGA data packet."""

    DCA_PACKET_SIZE = 1466
    """Radar packet size; hard-coded in the FPGA."""

    DCA_BITRATE = 1e9
    """DCA1000EVM interface speed, in bits per seconds (Gigabit Ethernet)."""
