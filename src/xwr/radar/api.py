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
    # `(ethOscClkEn, driveStrength)` for devices whose `channelCfg` takes
    # these; see [`configure_channels`][xwr.radar.common.configure_channels].
    _ETH_OSC_CLK: tuple[int, int] | None = None
    # `antGeometryCfg` arguments: (row, column) of each virtual antenna in
    # TX-major, RX-minor order, followed by the azimuth and elevation spacing
    # (in wavelengths). Only used by the on-chip processing chain, but is a
    # mandatory command for the TI demo firmware.
    _ANT_GEOMETRY = (
        "1 0 1 1 1 2 1 3 0 2 0 3 0 4 0 5 1 4 1 5 1 6 1 7 "
        "1 8 1 9 1 10 1 11 0.5 0.8")
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
            numFrames=0, numAdcSamples=adc_samples,
            framePeriodicity=frame_period,
            triggerSelect=1, frameTriggerDelay=0.0))

        self.send(common.configure_channels(
            rx=self._RX_MASK, tx=self._TX_MASK,
            eth_osc_clk=self._ETH_OSC_CLK))
        self.compRangeBiasAndRxChanPhase(rx_phase = [(1, 0)] * 4 * 4)
        self.lvdsStreamCfg()

        self.send(f"antGeometryCfg {self._ANT_GEOMETRY}")

        self.send("lowPower 0 0")
        self.send(common.get_boilerplate())

        self.log.info("Radar setup complete.")


class AWR2944P(AWR2944):
    """Interface implementation for the TI AWR2944P.

    !!! info "Supported devices"

        - AWR2944PEVM

    The AWR2944P shares the RF front end, antenna layout, and LVDS interface
    of the AWR2944; the only difference is that the AWR2x44P has a 25MHz
    ethernet oscillator clock output, so its `channelCfg` takes two additional
    arguments `<ethOscClkEn> <driveStrength>` (see the mmWave MCU+ SDK user
    guide), which we leave disabled.

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _ETH_OSC_CLK = (0, 0)

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR2944P"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)


class AWR2E44P(AWR2944P):
    """Interface implementation for the TI AWR2E44P.

    !!! info "Supported devices"

        - AWR2E44PEVM

    The AWR2E44P is the launch-on-package (LOP) variant of the AWR2x44P,
    i.e., the same die as the AWR2944P with the antennas moved onto the
    package. It uses the same firmware (the pre-built mmWave MCU Plus SDK
    AWR2x44P demos are built for the AWR2E44P by default), RF front end, and
    LVDS interface; only the antenna layout differs.

    The antenna geometry is taken from the TI demo profiles for the AWR2E44P
    (`ti/demo/awr2x44P/mmw_tdm/profiles/awr2E44P`), which use the same
    TX1, TX2, TX3, TX4 chirp order as
    [`configure_channels`][xwr.radar.common.configure_channels].

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _ANT_GEOMETRY = (
        "1 0 1 2 1 7 1 9 1 3 1 5 1 10 1 12 1 6 1 8 1 13 1 15 "
        "0 9 0 11 0 16 0 18 0.5 0.5")

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWR2E44P"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

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


class AWRL6888(AWRL6844):
    """Interface implementation for the TI AWRL6888.

    !!! info "Supported devices"

        - AWRL6888EVM

    The AWRL6888 has 8 TX and 8 RX antennas, but only 4 ADC chains. Each
    chain is switched between two RX antennas (bank A and bank B), so a
    single chirp only captures 4 of the 8 RX antennas. To cover all of them
    in TDM mode, each TX fires twice per chirp loop, once for each RX bank,
    which gives 16 chirps per loop.

    We report each of these chirps as its own "TX", so the data has
    `NUM_TX = 16` and `NUM_RX = 4`, in the order the chirps are streamed.
    Chirp `s` on chain `k` corresponds to TX slot `s // 2` and RX index
    `(s % 2) * 4 + k` in the 8T8R virtual array. This is the same TX-major,
    RX-minor order that the TI demo uses for `antGeometryTX`,
    `antGeometryRX`, and `compRangeBiasAndRxChanPhase`.

    Note that the TX slots follow the order the antennas fire in, which is
    different from the TX bit order in `channelCfg`. The firing order is TX0A,
    TX0B, TX1B, TX1A, TX2A, TX2B, TX3B, TX3A (see
    `MMWave_configureChirpMimoPat` in the SDK).

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    NUM_TX = 16
    NUM_RX = 4
    BYTES_PER_SAMPLE = 2

    # The demo firmware drops characters from long commands (e.g., the ~830
    # character `compRangeBiasAndRxChanPhase`) sent at full speed.
    _WRITE_CHUNK_SIZE = 16
    _WRITE_CHUNK_DELAY = 0.002

    # Antenna geometry and phase calibration, from the TI demo profile
    # (`mmwave_l_sdk_{version}/examples/mmw_demo/mmwave_demo_L688x/profiles/
    # profile_8T8R_tdm.cfg`). These only affect the on-chip processing chain,
    # but the demo firmware refuses to start without them.
    _ANT_GEOMETRY_TX = "1 0 0 1 1 2 0 3 1 4 0 5 1 6 0 7"
    _ANT_GEOMETRY_RX = "0 0 2 0 4 0 6 0 1 0 3 0 5 0 7 0"
    _ANT_GEOMETRY_DIST = "5.08 5.08"
    _COMP_RANGE_BIAS_RX_PHASE = (
        "-0.08620 0.12087 0.19660 -0.10342 0.21609 -0.42792 -0.29734 -0.53109 "
        "-0.09694 -0.07475 -0.27053 0.13933 -0.21205 0.39096 0.42558 0.80347 "
        "0.43634 -0.09095 0.25783 -0.28356 0.06454 0.08816 -0.53468 -0.14678 "
        "-0.55148 0.20000 -0.27487 0.30910 -0.02338 -0.21630 0.55871 -0.08312 "
        "0.82233 0.21498 -0.11059 0.21324 0.12333 -0.55400 0.29647 -0.35827 "
        "0.49100 -0.28427 0.05688 -0.20338 -0.16217 0.62439 -0.18612 0.82026 "
        "-0.38472 0.21541 0.14833 -0.01147 0.26646 -0.59729 -0.34488 -0.66099 "
        "-0.04423 -0.19447 -0.24246 0.05524 -0.27635 0.51272 0.47706 0.84307 "
        "0.53781 0.12349 0.25251 -0.14238 0.24885 -0.14851 -0.47495 -0.37736 "
        "-0.36972 -0.06873 -0.33379 0.18454 -0.22966 0.08840 0.52451 0.25120 "
        "0.59857 -0.10961 0.22560 -0.25738 0.02878 0.19583 -0.49201 -0.04146 "
        "-0.53768 0.20457 -0.22605 0.26833 0.00783 -0.29264 0.48573 -0.20415 "
        "0.69670 0.24238 -0.09038 0.21091 0.15542 -0.41290 0.27007 -0.25526 "
        "0.42491 -0.31011 0.02178 -0.19037 -0.19111 0.48299 -0.22925 0.49817 "
        "-0.44428 0.22775 0.10133 0.04338 0.25819 -0.35094 -0.17731 -0.41329 "
        "0.00960 -0.23911 -0.19779 -0.00323 -0.26883 0.36584 0.25458 0.49123 "
        "0.14393")

    # Supported `digOutputSampRate` decimation factors; the ADC sample rate
    # is 200 MHz / `digOutputSampRate`. Source: *Parameter Tuning and
    # Customization Guide for the xWRL6888 mmWave Demo*, `chirpComnCfg`.
    _SAMPLE_RATE_DIVISORS = (8, 10, 13, 15, 20, 25, 40, 50, 60, 80, 100)
    # Maximum `chirpAdcStartTime` (ADC skip samples) accepted by the SDK.
    _MAX_ADC_SKIP_SAMPLES = 103
    # Minimum idle time after each burst, in us; the radar silently fails to
    # chirp if this is violated. Source: *Parameter Tuning and Customization
    # Guide for the xWRL6888 mmWave Demo*, `frameCfg` (burst constraints).
    _MIN_BURST_IDLE_TIME = 135.0

    def __init__(
        self, port: str | None = None, baudrate: int = 115200,
        name: str = "AWRL6888"
    ) -> None:
        super().__init__(port=port, baudrate=baudrate, name=name)

    def setup(
        self, frequency: float = 60.0, idle_time: float = 20.0,
        adc_start_time: float = 5.0, ramp_end_time: float = 36.0,
        tx_start_time: float = 1.5, freq_slope: float = 100.0,
        adc_samples: int = 256, sample_rate: int = 10000,
        frame_length: int = 32, frame_period: float = 100.0
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
            sample_rate: ADC sampling rate; in ksps. Must be 200 MHz divided
                by one of the supported divisors (8, 10, 13, 15, 20, 25, 40,
                50, 60, 80, 100).
            frame_length: chirp loops per frame (= chirps per TX slot per
                frame). Must be a power of 2.
            frame_period: time between the start of each frame; in ms.

        Raises:
            ValueError: if `sample_rate` or `adc_start_time` is not supported
                by the firmware.
        """
        dig_output_samp_rate = 200_000 // sample_rate
        if dig_output_samp_rate not in self._SAMPLE_RATE_DIVISORS:
            raise ValueError(
                f"Unsupported sample_rate={sample_rate} ksps: the AWRL6888 "
                f"sample rate must be 200,000 ksps / N for N in "
                f"{self._SAMPLE_RATE_DIVISORS}.")

        # chirpAdcStartTime takes skip samples (int), not us.
        adc_start_skip = round(adc_start_time * sample_rate / 1000)
        if adc_start_skip > self._MAX_ADC_SKIP_SAMPLES:
            raise ValueError(
                f"adc_start_time={adc_start_time} us is {adc_start_skip} "
                f"samples at {sample_rate} ksps, which exceeds the maximum of "
                f"{self._MAX_ADC_SKIP_SAMPLES} ADC skip samples.")

        self.port.write('\n'.encode('ascii'))
        self._wait_for_response()

        self.stop()

        self.send("channelCfg 255 255 0")
        self.send("apllFreqShiftEn 0")

        # digOutputBitsSel=5: full 16-bit output (recommended by TI when
        # chirp accumulation is disabled). chirpTxMimoPatSel=1: TDM.
        self.send(
            f"chirpComnCfg {dig_output_samp_rate} 5 0 "
            f"{adc_samples} 1 {ramp_end_time} 1")
        self.send(
            f"chirpTimingCfg {idle_time} {adc_start_skip} "
            f"{tx_start_time} {freq_slope} {frequency}")
        self.send("adcDataDitherCfg 0")

        # Send the whole frame as a single burst so that chirp loops are
        # contiguous (i.e., `XWRConfig.chirp_time` is the true loop period);
        # each burst must be followed by at least `_MIN_BURST_IDLE_TIME`.
        num_chirps = self.NUM_TX * frame_length
        burst_period = (
            num_chirps * (idle_time + ramp_end_time)
            + self._MIN_BURST_IDLE_TIME)
        self.send(
            f"frameCfg {num_chirps} 0 {burst_period} 1 {frame_period} 0")

        # On-chip processing settings, from `profile_8T8R_tdm.cfg`.
        self.send("gpAdcMeasConfig 0 0")
        self.send("guiMonitor 1 0 0 0 0 1")
        self.send("cfarProcCfg 0 0 8 4 4 0 11.75 1")
        self.send("cfarProcCfg 1 0 8 4 4 1 11.75 1")
        self.send("cfarProcCfg 2 0 4 4 3 1 5.88 1")
        self.send("cfarProcCfg 3 0 4 4 3 1 5.88 1")
        self.send("aoaSidelobeRejCfg 1 0.8")
        self.send("fftWindowCfg 1 1")
        self.send("fftBflyScaling 3 7 3 3")
        self.send("cfarFovCfg 0 0.25 10.0")
        self.send("cfarFovCfg 1 -20.16 20.16")
        self.send("aoaProcCfg 32 32")
        self.send("aoaFovCfg -60 60 -60 60")
        self.send("clutterRemoval 0")

        self.send("runtimeCalibCfg 1")
        self.send(f"antGeometryTX {self._ANT_GEOMETRY_TX}")
        self.send(f"antGeometryRX {self._ANT_GEOMETRY_RX}")
        self.send(f"antGeometryDist {self._ANT_GEOMETRY_DIST}")
        self.send(
            f"compRangeBiasAndRxChanPhase {self._COMP_RANGE_BIAS_RX_PHASE}")
        self.send("adcLogging 1")
        self.send("lowPowerCfg 0")

        self.log.info("Radar setup complete.")
