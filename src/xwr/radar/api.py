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

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'(?=.*CP2105)(?=.*Enhanced)|XDS110'
    _TX_MASK = 0b111
    NUM_TX = 3
    NUM_RX = 4

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
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

        self._configure_channels(rx=0b1111, tx=self._TX_MASK)
        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=self.NUM_TX - 1,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase(rx_phase = [(0, 1)] * 4 * 3)
        self.lvdsStreamCfg()

        self.boilerplate_setup()
        self.log.info("Radar setup complete.")


class AWR1843L(AWR1843):
    """TI AWR1843Boost with its middle antenna disabled.

    !!! info "Supported devices"

        - AWR1843Boost, with the middle TX antenna which is 1/2-wavelength
          above the other two disabled.

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _TX_MASK = 0b101
    NUM_TX = 2
    NUM_RX = 4


class AWR1642(XWRBase):
    """Interface implementation for the TI AWR1642 family.

    !!! info "Supported devices"

        - AWR1642Boost

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'XDS110'
    NUM_TX = 2
    NUM_RX = 4

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR1642"
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

        self._configure_channels(rx=0b1111, tx=0b011)
        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=self.NUM_TX - 1,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase(rx_phase = [(0, 1)] * 4 * 2)
        self.send("bpmCfg -1 0 0 1")
        self.lvdsStreamCfg()

        self.boilerplate_setup()
        self.log.info("Radar setup complete.")

    def lowPower(self, dontCare: int = 0, adcMode: int = 1) -> None:
        """Low power mode config.

        !!! warning

            For some reason, the AWR1642 requires `adcMode=1`. Not sure what
            this does.
        """
        cmd = "lowPower {} {}".format(dontCare, adcMode)
        self.send(cmd)


class AWR2544(XWRBase):
    """Interface implementation for the TI AWR2544 family.

    !!! info "Supported devices"

        - AWR2544LOPEVM

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'XDS110'
    NUM_TX = 4
    NUM_RX = 4

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
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

        self._configure_channels(rx=0b1111, tx=0b1111)
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

    def frameCfg(  # type: ignore
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
