"""Radar sensor APIs."""

from typing import Literal

from . import defines
from .base import XWRBase

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803


class AWR1843(XWRBase):
    """Interface implementation for the TI AWR1843 family.

    !!! info "Supported devices"

        - AWR1843Boost
        - AWR1843AOPEVM

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
        self, port: str | None = None, baudrate: int = 115200,
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


class AWR2544(XWRBase):
    """Interface implementation for the TI AWR2544 family.

    !!! info "Supported devices"

        - AWR2544LOPEVM

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

    def setup(
        self, num_tx: Literal[4] = 4,
        frequency: float = 77.0, idle_time: float = 110.0,
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

        return self.setup_from_config("test.cfg")

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
            framePeriodicity=frame_period, numAdcSamples=adc_samples)
        # self.lvdsStreamCfg()

        # self.boilerplate_setup()
        self.lowPower()
        self.CQRxSatMonitor()
        self.CQSigImgMonitor()
        self.analogMonitor()
        self.calibData()

    def channelCfg(
        self, rxChannelEn: int = 0b1111, txChannelEn: int = 0b101,
        cascading: int = 0, ethOscClkEn: Literal[0, 1] = 0,
        driveStrength: int = 0
    ) -> None:
        """Channel configuration for the radar subsystem.

        Args:
            rxChannelEn: bit-masked rx channels to enable.
            txChannelEn: bit-masked tx channels to enable.
            cascading: must always be set to 0.
            ethOscClkEn: enable 25MHz ethernet oscillator clock supply from the
                chip; not used (`0`) by this library.
            driveStrength: ethernet oscillator clock drive strength.
        """
        cmd = "channelCfg {} {} {} {} {}".format(
            rxChannelEn, txChannelEn, cascading, ethOscClkEn, driveStrength)
        self.send(cmd)

    def frameCfg(
        self, chirpStartIdx: int = 0, chirpEndIdx: int = 1, numLoops: int = 16,
        numFrames: int = 0, numAdcSamples: int = 256,
        framePeriodicity: float = 100.0,
        triggerSelect: int = 1, frameTriggerDelay: float = 0.0
    ) -> None:
        """Radar frame configuration.

        !!! warning

            The frame should not have more than a 50% duty cycle according to
            the mmWave SDK documentation.

        Args:
            chirpStartIdx: chirps to use in the frame.
            chirpEndIdx: chirps to use in the frame.
            numLoops: number of chirps per frame; must be >= 16 based on
                trial/error.
            numFrames: how many frames to run before stopping; infinite if 0.
            numAdcSamples: number of samples per chirp; must match the
                `numAdcSamples` provided to `profileCfg`.
            framePeriodicity: period between frames, in ms.
            triggerSelect: only software trigger (1) is supported.
            frameTriggerDelay: does not appear to be documented.
        """
        cmd = "frameCfg {} {} {} {} {} {} {} {}".format(
            chirpStartIdx, chirpEndIdx, numLoops, numFrames, numAdcSamples,
            framePeriodicity, triggerSelect, frameTriggerDelay)
        self.send(cmd)

    def analogMonitor(
        self, rxSaturation: int = 0, sigImgBand: int = 0,
        apllLdoSCMonEn: int = 0
    ) -> None:
        """Enable/disable monitoring."""
        cmd = "analogMonitor {} {} {}".format(
            rxSaturation, sigImgBand, apllLdoSCMonEn)
        self.send(cmd)
