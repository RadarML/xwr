"""Radar sensor APIs."""

from . import common, defines
from .base import XWRBase

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803


class AWR1843(XWRBase, common.APIMixins):
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
    _START_COMMAND = "sensorStart"
    _TX_MASK = 0b111
    _RX_MASK = 0b1111
    NUM_TX = 3
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2 * 2

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
        self.send("flushCfg")
        self.send("dfeDataOutputMode {modeType.value}".format(
            modeType=defines.DFEMode.LEGACY))
        self.send(common.configure_adc(adc_fmt=defines.ADCFormat.COMPLEX_1X))
        self.profileCfg(
            startFreq=frequency, idleTime=idle_time,
            adcStartTime=adc_start_time, rampEndTime=ramp_end_time,
            txStartTime=tx_start_time, freqSlopeConst=freq_slope,
            numAdcSamples=adc_samples, digOutSampleRate=sample_rate)

        self.send(common.configure_channels(rx=self._RX_MASK, tx=self._TX_MASK))
        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=self.NUM_TX - 1,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase(rx_phase = [(0, 1)] * 4 * 3)
        self.lvdsStreamCfg()

        self.send("lowPower 0 0")
        self.send(common.get_boilerplate())
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

    _PORT_NAME = r"XDS110"
    _START_COMMAND = "sensorStart"
    _TX_MASK = 0b101
    _RX_MASK = 0b1111
    NUM_TX = 2
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2 * 2


class AWR1642(XWRBase, common.APIMixins):
    """Interface implementation for the TI AWR1642 family.

    !!! info "Supported devices"

        - AWR1642Boost

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'XDS110'
    _START_COMMAND = "sensorStart"
    _TX_MASK = 0b011
    _RX_MASK = 0b1111
    NUM_TX = 2
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2 * 2

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
        self.send("flushCfg")
        self.send("dfeDataOutputMode {modeType.value}".format(
            modeType=defines.DFEMode.LEGACY))
        self.send(common.configure_adc(adc_fmt=defines.ADCFormat.COMPLEX_1X))
        self.profileCfg(
            startFreq=frequency, idleTime=idle_time,
            adcStartTime=adc_start_time, rampEndTime=ramp_end_time,
            txStartTime=tx_start_time, freqSlopeConst=freq_slope,
            numAdcSamples=adc_samples, digOutSampleRate=sample_rate)

        self.send(common.configure_channels(rx=self._RX_MASK, tx=self._TX_MASK))
        self.frameCfg(
            numLoops=frame_length, chirpEndIdx=self.NUM_TX - 1,
            framePeriodicity=frame_period)
        self.compRangeBiasAndRxChanPhase(rx_phase = [(0, 1)] * 4 * 2)
        self.send("bpmCfg -1 0 0 1")
        self.lvdsStreamCfg()

        # For some reason, the AWR1642 requires adcMode=1.
        # Not sure what this does.
        self.send("lowPower 0 1")

        self.send(common.get_boilerplate())
        self.log.info("Radar setup complete.")


class AWR2944(XWRBase, common.APIMixins):
    """Interface implementation for the TI AWR2944.

    !!! info "Supported devices"

        - AWR2944EVM

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'XDS110'
    _START_COMMAND = "sensorStart"
    _TX_MASK = 0b1111
    _RX_MASK = 0b1111
    NUM_TX = 4
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR2944"
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

        self.port.write('\n'.encode('ascii'))
        self._wait_for_response()

        self.stop()
        self.send("flushCfg")

        self.send("dfeDataOutputMode {modeType.value}".format(
            modeType=defines.DFEMode.LEGACY))

        self.send(common.configure_adc(adc_fmt=defines.ADCFormat.REAL))
        self.profileCfg(
            startFreq=frequency, idleTime=idle_time,
            adcStartTime=adc_start_time, rampEndTime=ramp_end_time,
            txStartTime=tx_start_time, freqSlopeConst=freq_slope,
            numAdcSamples=adc_samples, digOutSampleRate=sample_rate)
        self.send((
            "frameCfg {chirpStartIdx} {chirpEndIdx} {numLoops} {numFrames} "
            "{numAdcSamples} {framePeriodicity} "
            "{triggerSelect} "      # 1 = software trigger
            "{frameTriggerDelay}"   # Undocumented
        ).format(
            chirpStartIdx=0, chirpEndIdx=3, numLoops=frame_length,
            numFrames=0, numAdcSamples=256, framePeriodicity=frame_period,
            triggerSelect=1, frameTriggerDelay=0.0))

        self.send(common.configure_channels(rx=self._RX_MASK, tx=self._TX_MASK))
        self.compRangeBiasAndRxChanPhase(rx_phase = [(1, 0)] * 4 * 4)
        self.lvdsStreamCfg()

        # Can't be bothered to figure out what this does
        self.send(
            "antGeometryCfg 1 0 1 1 1 2 1 3 0 2 0 3 0 4 0 5 1 4 1 5 1 6 1 7 "
            "1 8 1 9 1 10 1 11 0.5 0.8")

        self.send("lowPower 0 0")
        self.send(common.get_boilerplate())

        self.log.info("Radar setup complete.")


class AWRL6844(XWRBase):
    """Interface implementation for the TI AWRL6844.

    !!! info "Supported devices"

        - AWRL6844EVM

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _PORT_NAME = r'XDS110'
    _START_COMMAND = "sensorStart 0 0 0 0"

    NUM_TX = 4
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWRL6844"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

    def stop(self) -> None:
        self.send("sensorStop 0")
        self.log.info("Radar Stopped.")

    def setup(
        self, frequency: float = 60.0, idle_time: float = 110.0,
        adc_start_time: float = 4.0, ramp_end_time: float = 56.0,
        tx_start_time: float = 1.0, freq_slope: float = 70.006,
        adc_samples: int = 256, sample_rate: int = 5000,
        frame_length: int = 64, frame_period: float = 100.0
    ) -> None:
        """Configure radar.

        Args:
            frequency: chirp start frequency, in GHz.
            idle_time: chirp idle time; in us.
            adc_start_time: ADC start offset, in us.
            ramp_end_time: chirp ramp end time; in us.
            tx_start_time: TX start time offset; in us.
            freq_slope: chirp frequency slope; in MHz/us.
            adc_samples: number of ADC samples per chirp.
            sample_rate: ADC sampling rate; in ksps.
            frame_length: bursts per frame (= chirps per TX antenna per frame).
                Must be a power of 2.
            frame_period: time between the start of each frame; in ms.
        """
        assert frame_length & (frame_length - 1) == 0

        self.port.write('\n'.encode('ascii'))
        self._wait_for_response()

        self.stop()

        self.send("channelCfg 153 255 0")
        self.send("apllFreqShiftEn 0")

        # digOutputSampRate: clock divider for the 400 MHz APLL.
        # ADC rate (Msps) = 400 / (2 * digOutputSampRate), so:
        # digOutputSampRate = 400_000 / (2 * sample_rate_ksps)
        #                   = 200_000 // sample_rate
        dig_output_samp_rate = 200_000 // sample_rate

        # chirpAdcStartTime takes skip samples (int), not µs.
        adc_start_skip = round(adc_start_time * sample_rate / 1000)

        # burstPeriodus: repetition interval for one burst of NUM_TX chirps.
        # Minimum = NUM_TX * (idle_time + ramp_end_time).
        burst_period = self.NUM_TX * (idle_time + ramp_end_time)

        # chirpTxMimoPatSel=4 is MIMO_TDM_PATTERN (one TX per chirp, cycling).
        self.send(
            f"chirpComnCfg {dig_output_samp_rate} 0 0 "
            f"{adc_samples} 1 {ramp_end_time} 0")
        self.send(
            f"chirpTimingCfg {idle_time} {adc_start_skip} "
            f"{tx_start_time} {freq_slope} {frequency}")

        self.send("adcDataDitherCfg  1")

        # numOfChirpsInBurst=NUM_TX cycles through all TX antennas per burst.
        # numOfBurstsInFrame=frame_length gives frame_length chirps per TX.
        # Constraint: (NUM_TX * frame_length) % NUM_TX == 0 always holds.
        self.send(
            f"frameCfg {frame_length * 4} 0 {burst_period} "
            f"1 {frame_period} 0")

        self.send("gpAdcMeasConfig 0 0")
        self.send("guiMonitor 1 1 0 0 0 1")
        self.send("cfarProcCfg 0 2 8 4 3 0 9.0 0")
        self.send("cfarProcCfg 1 2 4 2 2 1 9.0 0")
        self.send("cfarFovCfg 0 0.25 9.0")
        self.send("cfarFovCfg 1 -20.16 20.16")
        self.send("aoaProcCfg 64 64")
        self.send("aoaFovCfg -60 60 -60 60")
        self.send("clutterRemoval 0")

        self.send("runtimeCalibCfg 1")
        self.send("antGeometryBoard xWRL6844EVM")
        self.send("adcLogging 1")
        self.send("lowPowerCfg 0")

        self.log.info("Radar setup complete.")
