"""AWR1843 TI Demo API."""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803

import logging
import time

import serial

from . import awr_types as types
from .awr_boilerplate import AWR1843Mixins


class AWR1843(AWR1843Mixins):
    """AWR1843 Interface for the TI `demo/xwr18xx` MSS firmware.

    Documented by [R3]_, [R4]_, [R5]_, [R6]_; based on a UART ASCII CLI.

    NOTE: only a partial API is implemented. Non-mandatory calls which do not
    affect the LVDS raw I/Q stream are not implemented.

    Usage:
        (1) Initialization parameters can be defaults. The `port` may need to
            be changed if multiple radars are being used, or another device
            uses the `/dev/ttyACM0` default name. The baudrate should not be
            changed.
        (2) Setup with `.setup(...)` with the desired radar configuration.
        (3) Start the radar with `.start()`.
            NOTE: if the configuration is invalid, `.start()` may return an
            error, or cause the radar to freeze. This may require the radar to
            be rebooted via manually disconnecting the power supply.
        (4) Stop the radar with `.stop()`.

    Args:
        port: radar control serial port; typically the lower numbered one.
        baudrate: baudrate of control port.
        name: human-readable name.
    """

    _CMD_PROMPT = "\rmmwDemo:/>"

    def __init__(
        self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
        name: str = "AWR1843"
    ) -> None:
        self.log = logging.getLogger(name=name)
        self.port = serial.Serial(port, baudrate, timeout=None)
        self.port.set_low_latency_mode(True)

    def setup_from_config(self, path: str) -> None:
        """Run raw setup from a config file."""
        with open(path) as f:
            cmds = f.readlines()
        for c in cmds:
            self.send(c.rstrip('\n'))

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
            idle_time, adc_start_time, ramp_end_time, tx_start_time: see TI
                chirp timing documentation; in us.
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
        self.dfeDataOutputMode(types.DFEMode.LEGACY)
        self.adcCfg(adcOutputFmt=types.ADCFormat.COMPLEX_1X)
        self.adcbufCfg(adcOutputFmt=types.ADCFormat.COMPLEX_1X)
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

    def send(self, cmd: str, timeout: float = 10.0) -> None:
        """Send message, and wait for a response.

        Args:
            cmd: command to send.
            timeout: raises `TimeoutError` if the expected response is not
                received by this time.
        """
        self.log.info("Send: {}".format(cmd))
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
        resp = (
            rx_buf.decode('utf-8', errors='replace')
            .replace(self._CMD_PROMPT, '').replace(cmd, '')
            .rstrip(' \n\t').lstrip(' \n\t')
            .replace('\n', '; ').replace('\r', ''))
        self.log.debug("Response: {}".format(resp))

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
                raise types.AWRException(resp)

    def start(self, reconfigure: bool = True) -> None:
        """Start radar.

        Args:
            reconfigure: Whether the radar needs to be configured.
        """
        if reconfigure:
            self.send("sensorStart")
        else:
            self.send("sensorStart 0")

    def stop(self) -> None:
        """Stop radar."""
        self.send("sensorStop")

    def flushCfg(self) -> None:
        """Clear existing (possibly partial) configuration."""
        self.send("flushCfg")

    def dfeDataOutputMode(
        self, modeType: types.DFEMode = types.DFEMode.LEGACY
    ) -> None:
        """Set frame data output mode."""
        cmd = "dfeDataOutputMode {}".format(modeType.value)
        self.send(cmd)

    def channelCfg(
        self, rxChannelEn: int = 0b1111, txChannelEn: int = 0b101,
        cascading: int = 0
    ) -> None:
        """Channel configuration for the radar subsystem.

        Args:
            rxChannelEn, txChannelEn: bit-masked rx/tx channels to enable.
            cascading: must always be set to 0.
        """
        cmd = "channelCfg {} {} {}".format(rxChannelEn, txChannelEn, cascading)
        self.send(cmd)

    def adcCfg(
        self, numADCBits: types.ADCDepth = types.ADCDepth.BIT16,
        adcOutputFmt: types.ADCFormat = types.ADCFormat.COMPLEX_1X
    ) -> None:
        """Configure radar subsystem ADC.

        Args:
            numADCBits: ADC bit depth
            adcOutputFmt: real, complex, and whether to filter the image band.
        """
        cmd = "adcCfg {} {}".format(numADCBits.value, adcOutputFmt.value)
        self.send(cmd)

    def adcbufCfg(
        self, subFrameIdx: int = -1,
        adcOutputFmt: types.ADCFormat = types.ADCFormat.COMPLEX_1X,
        sampleSwap: types.SampleSwap = types.SampleSwap.MSB_LSB_IQ,
        chanInterleave: int = 1, chirpThreshold: int = 1
    ) -> None:
        """ADC Buffer hardware configuration.

        Args:
            subFrameIdx: subframe to apply to; if `-1`, applies to all.
            adcOutputFmt: real/complex ADC format.
            sampleSwap: write samples in IQ or QI order. We assume `MSB_LSB_IQ`.
                NOTE: the output is an interleaved complex-int-32 format; see
                `RadarFrame` for details. `MSB_LSB_QI` doesn't seem to work.
            chanInterleave: only non-interleaved (1) is supported.
            chirpThreshold: some kind of "ping-pong" demo parameter.
        """
        cmd = "adcbufCfg {} {} {} {} {}".format(
            subFrameIdx, 1 if adcOutputFmt == types.ADCFormat.REAL else 0,
            sampleSwap.value, chanInterleave, chirpThreshold)
        self.send(cmd)

    def profileCfg(
        self, profileId: int = 0, startFreq: float = 77.0,
        idleTime: float = 267.0, adcStartTime: float = 7.0,
        rampEndTime: float = 57.14, txStartTime: float = 1.0,
        txOutPower: int = 0, txPhaseShifter: int = 0,
        freqSlopeConst: float = 70.0,
        numAdcSamples: int = 256, digOutSampleRate: int = 5209,
        hpfCornerFreq1: types.HPFCornerFreq1 = types.HPFCornerFreq1.KHZ175,
        hpfCornerFreq2: types.HPFCornerFreq2 = types.HPFCornerFreq2.KHZ350,
        rxGain: int = 30
    ) -> None:
        """Configure chirp profile(s).

        Args:
            profileId: profile to configure. Can only have one in
                `DFEMode.LEGACY`.
            startFreq: chirp start frequency, in GHz. Can be 76 or 77 [R6]_.
            idleTime, adcStartTime, rampEndTime, txStartTime: chirp timing; see
                the "RampTimingCalculator" [R5]_.
            txOutPower, txPhaseShifter: not entirely clear what this does. The
                demo claims that only '0' is tested / should be used.
            freqSlopeConst: frequency slope ("ramp rate") in MHz/us; <100MHz/us.
            numAdcSamples: Number of ADC samples per chirp.
            digOutSampleRate: ADC sample rate in ksps (<12500); see
                Table 8-4 [R6_].
            hpfCornerFreq1, hpfCornerFreq2: high pass filter corner frequencies.
            rxGain: RX gain in dB. The meaning of this value is not clear.
        """
        assert startFreq in {76.0, 77.0}
        assert freqSlopeConst < 100.0
        assert digOutSampleRate < 12500

        cmd = "profileCfg {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
            profileId, startFreq, idleTime, adcStartTime, rampEndTime,
            txOutPower, txPhaseShifter, freqSlopeConst, txStartTime,
            numAdcSamples, digOutSampleRate, hpfCornerFreq1.value,
            hpfCornerFreq2.value, rxGain)
        self.send(cmd)

    def chirpCfg(
        self, chirpIdx: int = 0, profileId: int = 0,
        startFreqVar: float = 0.0, freqSlopeVar: float = 0.0,
        idleTimeVar: float = 0.0, adcStartTimeVar: float = 0.0,
        txEnable: int = 0
    ) -> None:
        """Radar chirp configuration.

        Args:
            chirpIdx: Antenna index. Sets `chirpStartIdx`, `chirpEndIdx` [R4]_
                to `chirpIdx`, and `txEnable` (antenna bitmask) to
                `1 << chirpIdx`.
            profileId: chirp profile to use.
            startFreqVar, freqSlopeVar, idleTimeVar, adcStartTimeVar: allowed
                freq/slope/time tolerances; documentation states only 0 is
                tested.
            txEnable: antenna to enable; is converted to a bit mask.
        """
        cmd = "chirpCfg {} {} {} {} {} {} {} {}".format(
            chirpIdx, chirpIdx, profileId, startFreqVar, freqSlopeVar,
            idleTimeVar, adcStartTimeVar, 1 << txEnable)
        self.send(cmd)

    def frameCfg(
        self, chirpStartIdx: int = 0, chirpEndIdx: int = 1, numLoops: int = 16,
        numFrames: int = 0, framePeriodicity: float = 100.0,
        triggerSelect: int = 1, frameTriggerDelay: float = 0.0
    ) -> None:
        """Radar frame configuration.

        NOTE: the frame should not have more than a 50% duty cycle according to
        the mmWave SDK documentation [R4]_.

        Args:
            chirpStartIdx, chirpEndIdx: chirps to use in the frame.
            numLoops: number of chirps per frame; must be >= 16 based on
                trial/error.
            numFrames: how many frames to run before stopping; infinite if 0.
            framePeriodicity: period between frames, in ms.
            triggerSelect: only software trigger (1) is supported.
            frameTriggerDelay: does not appear to be documented.
        """
        cmd = "frameCfg {} {} {} {} {} {} {}".format(
            chirpStartIdx, chirpEndIdx, numLoops, numFrames, framePeriodicity,
            triggerSelect, frameTriggerDelay)
        self.send(cmd)

    def compRangeBiasAndRxChanPhase(
        self, rangeBias: float = 0.0,
        rx_phase: list[tuple[int, int]] = [(0, 1)] * 12
    ) -> None:
        """Set range bias, channel phase compensation.

        NOTE: rx_phase must have one term per TX-RX pair.
        """
        args = ' '.join("{} {}".format(re, im) for re, im in rx_phase)
        cmd = "compRangeBiasAndRxChanPhase {} {}".format(rangeBias, args)
        self.send(cmd)

    def lvdsStreamCfg(
        self, subFrameIdx: int = -1, enableHeader: bool = False,
        dataFmt: types.LVDSFormat = types.LVDSFormat.ADC,
        enableSW: bool = False
    ) -> None:
        """Configure LVDS stream (to the DCA1000EVM); `LvdsStreamCfg`.

        Args:
            subframe: subframe to apply to. If `-1`, applies to all subframes.
            enableHeader: HSI (High speed interface; refers to LVDS) Header
                enabled/disabled flag; disabled for raw mode.
            dataFmt: LVDS format; we assume `LVDSFormat.ADC`.
            enableSW: Use software (SW) instead of hardware streaming; causes
                chirps to be streamed during the inter-frame time after
                processing. We assume HW streaming.
        """
        cmd = "lvdsStreamCfg {} {} {} {}".format(
            subFrameIdx, 1 if enableHeader else 0, dataFmt.value,
            1 if enableSW else 0)
        self.send(cmd)
