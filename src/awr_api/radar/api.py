"""Radar sensor APIs."""

from . import defines
from .base import AWRBase


class AWR1843(AWRBase):
    """Interface implementation for the TI AWR1843 family.

    Supported devices:

    - AWR1843Boost
    - AWR1843AOP

    !!! note

        Provides a 2-TX mode, which supports the use case of skipping the
        middle TX antenna on the AWR1843Boost, which is 1/2-wavelength above
        the other two.

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    def __init__(
        self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

    def setup(
        self, num_tx: int = 2,
        frequency: float = 77.0, idle_time: float = 110.0,
        adc_start_time: float = 4.0, ramp_end_time: float = 56.0,
        tx_start_time: float = 1.0, freq_slope: float = 70.006,
        adc_samples: int = 256, sample_rate: int = 5000,
        frame_length: int = 64, frame_period: float = 100.0
    ) -> None:
        """Configure radar.

        Args:
            num_tx: TX antenna config (2 or 3).
            frequency: frequency band, in GHz; 77.0 or 76.0.
            idle_time: see TI chirp timing documentation; in us.
            adc_start_time: see TI chirp timing documentation; in us.
            ramp_end_time: see TI chirp timing documentation; in us.
            tx_start_time: see TI chirp timing documentation; in us.
            freq_slope: chirp frequency slope; in MHz/us.
            adc_samples: number of samples per chirp.
            sample_rate: ADC sampling rate; in ksps.
            frame_length: chirps per frame per TX antenna. Must be a power of 2.
            frame_period: time between the start of each frame; in ms.
        """
        assert frame_length & (frame_length - 1) == 0
        assert num_tx in {2, 3}

        self.stop()
        self.flushCfg()
        self.dfeDataOutputMode(defines.DFEMode.LEGACY)
        self.adcCfg(adcOutputFmt=defines.ADCFormat.COMPLEX_1X)
        self.adcbufCfg(adcOutputFmt=defines.ADCFormat.COMPLEX_1X)
        self.profileCfg(
            startFreq=frequency, idleTime=idle_time,
            adcStartTime=adc_start_time, rampEndTime=ramp_end_time,
            txStartTime=tx_start_time, freqSlopeConst=freq_slope,
            numAdcSamples=adc_samples, digOutSampleRate=sample_rate)

        if num_tx == 2:
            self.channelCfg(rxChannelEn=0b1111, txChannelEn=0b101)
            self.chirpCfg(chirpIdx=0, txEnable=0)
            self.chirpCfg(chirpIdx=1, txEnable=2)
        else:
            self.channelCfg(rxChannelEn=0b1111, txChannelEn=0b111)
            self.chirpCfg(chirpIdx=0, txEnable=0)
            self.chirpCfg(chirpIdx=1, txEnable=1)
            self.chirpCfg(chirpIdx=2, txEnable=2)

        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=num_tx - 1,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase()
        self.lvdsStreamCfg()

        self.boilerplate_setup()


class AWR2544(AWRBase):
    """Interface implementation for the TI AWR2544 family.

    Supported devices:

    - AWR2544LOPEVM

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    def __init__(
        self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

    def setup(
        self, frequency: float = 77.0, idle_time: float = 110.0,
        adc_start_time: float = 4.0, ramp_end_time: float = 56.0,
        tx_start_time: float = 1.0, freq_slope: float = 70.006,
        adc_samples: int = 256, sample_rate: int = 5000,
        frame_length: int = 64, frame_period: float = 100.0
    ) -> None:
        """Configure radar.

        Args:
            frequency: frequency band, in GHz; 77.0 or 76.0.
            idle_time: see TI chirp timing documentation; in us.
            adc_start_time: see TI chirp timing documentation; in us.
            ramp_end_time: see TI chirp timing documentation; in us.
            tx_start_time: see TI chirp timing documentation; in us.
            freq_slope: chirp frequency slope; in MHz/us.
            adc_samples: number of samples per chirp.
            sample_rate: ADC sampling rate; in ksps.
            frame_length: chirps per frame per TX antenna. Must be a power of 2.
            frame_period: time between the start of each frame; in ms.
        """
        assert frame_length & (frame_length - 1) == 0

        self.stop()
        self.flushCfg()
        self.dfeDataOutputMode(defines.DFEMode.LEGACY)
        self.adcCfg(adcOutputFmt=defines.ADCFormat.COMPLEX_1X)
        self.adcbufCfg(adcOutputFmt=defines.ADCFormat.COMPLEX_1X)
        self.profileCfg(
            startFreq=frequency, idleTime=idle_time,
            adcStartTime=adc_start_time, rampEndTime=ramp_end_time,
            txStartTime=tx_start_time, freqSlopeConst=freq_slope,
            numAdcSamples=adc_samples, digOutSampleRate=sample_rate)

        self.channelCfg(rxChannelEn=0b1111, txChannelEn=0b1111)
        self.chirpCfg(chirpIdx=0, txEnable=0)
        self.chirpCfg(chirpIdx=1, txEnable=1)
        self.chirpCfg(chirpIdx=2, txEnable=2)
        self.chirpCfg(chirpIdx=3, txEnable=3)

        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=3,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase()
        self.lvdsStreamCfg()

        self.boilerplate_setup()
