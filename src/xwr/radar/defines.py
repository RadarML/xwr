"""TI Radar demo firmware common API defines.

See the [mmWave SDK user guide](
https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf),
Table 1 (Page 19).
"""

from enum import Enum


class LVDSFormat(Enum):
    """LVDS data format.

    Attributes:
        DISABLED: LVDS disabled.
        ADC: ADC data; we use this mode to get spectrum data.
        CP_ADC_CQ: CP ADC and CQ data.
    """

    DISABLED = 0
    ADC = 1
    _RESERVED2 = 2
    _RESERVED3 = 3
    CP_ADC_CQ = 4


class DFEMode(Enum):
    """Frame type; note that continuous chirping is not supported.

    Attributes:
        LEGACY: legacy mode (we only support this for now).
        ADVANCED: advanced mode.
    """

    LEGACY = 1
    CONTINUOUS_UNSUPPORTED = 2
    ADVANCED = 3


class ADCDepth(Enum):
    """ADC bit depth.

    Attributes:
        BIT12: 12-bit mode.
        BIT14: 14-bit mode.
        BIT16: 16-bit mode (used by default).
    """

    BIT12 = 0
    BIT14 = 1
    BIT16 = 2


class ADCFormat(Enum):
    """ADC output format.

    COMPLEX_1X has the image band filtered out, while COMPLEX_2X does not.

    Attributes:
        REAL: real data.
        COMPLEX_1X: complex data, with image band filtered out (default).
        COMPLEX_2X: raw complex data.
    """

    REAL = 0
    COMPLEX_1X = 1
    COMPLEX_2X = 2


class SampleSwap(Enum):
    """ADC I/Q bit order.

    !!! bug

        MSB_LSB_QI doesn't seem to work, so we've labeled it as
        `_NONFUNCTIONAL`.

    Attributes:
        MSB_LSB_IQ: I is in the MSB, Q is in the LSB (default).
    """

    MSB_LSB_QI_NONFUNCTIONAL = 0
    MSB_LSB_IQ = 1


class HPFCornerFreq1(Enum):
    """High pass filter 1 corner frequency.

    Attributes:
        KHZ175: 175 kHz (default).
        KHZ235: 235 kHz.
        KHZ350: 350 kHz.
        KHZ700: 700 kHz.
    """

    KHZ175 = 0
    KHZ235 = 1
    KHZ350 = 2
    KHZ700 = 3


class HPFCornerFreq2(Enum):
    """High pass filter 2 corner frequency.

    Attributes:
        KHZ350: 350 kHz (default).
        KHZ700: 700 kHz.
        MHZ1_4: 1.4 MHz.
        MHZ2_8: 2.8 MHz.
    """

    KHZ350 = 0
    KHZ700 = 1
    MHZ1_4 = 2
    MHZ2_8 = 3
