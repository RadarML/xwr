"""Raw, unabstracted commands.

These commands are commonly issued by dumping the text file output of the TI
demo visualizer onto a serial port. However, this is not great for debugging,
and the individual parameters are not documented. We instead split each command
into a documented and individually callable function.
"""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803

from . import defines


class APIMixins:
    """Mixins capturing the raw TI Demo API."""

    def send(self, cmd: str, timeout: float = 10.0) -> None:
        raise NotImplementedError()

    def flushCfg(self) -> None:
        """Clear existing (possibly partial) configuration."""
        self.send("flushCfg")

    def dfeDataOutputMode(
        self, modeType: defines.DFEMode = defines.DFEMode.LEGACY
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
            rxChannelEn: bit-masked rx channels to enable.
            txChannelEn: bit-masked tx channels to enable.
            cascading: must always be set to 0.
        """
        cmd = "channelCfg {} {} {}".format(rxChannelEn, txChannelEn, cascading)
        self.send(cmd)

    def adcCfg(
        self, numADCBits: defines.ADCDepth = defines.ADCDepth.BIT16,
        adcOutputFmt: defines.ADCFormat = defines.ADCFormat.COMPLEX_1X
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
        adcOutputFmt: defines.ADCFormat = defines.ADCFormat.COMPLEX_1X,
        sampleSwap: defines.SampleSwap = defines.SampleSwap.MSB_LSB_IQ,
        chanInterleave: int = 1, chirpThreshold: int = 1
    ) -> None:
        """ADC Buffer hardware configuration.

        Args:
            subFrameIdx: subframe to apply to; if `-1`, applies to all.
            adcOutputFmt: real/complex ADC format.
            sampleSwap: write samples in IQ or QI order. We assume `MSB_LSB_IQ`.
            chanInterleave: only non-interleaved (1) is supported.
            chirpThreshold: some kind of "ping-pong" demo parameter.
        """
        cmd = "adcbufCfg {} {} {} {} {}".format(
            subFrameIdx, 1 if adcOutputFmt == defines.ADCFormat.REAL else 0,
            sampleSwap.value, chanInterleave, chirpThreshold)
        self.send(cmd)

    def profileCfg(
        self, profileId: int = 0, startFreq: float = 77.0,
        idleTime: float = 267.0, adcStartTime: float = 7.0,
        rampEndTime: float = 57.14, txStartTime: float = 1.0,
        txOutPower: int = 0, txPhaseShifter: int = 0,
        freqSlopeConst: float = 70.0,
        numAdcSamples: int = 256, digOutSampleRate: int = 5209,
        hpfCornerFreq1: defines.HPFCornerFreq1 = defines.HPFCornerFreq1.KHZ175,
        hpfCornerFreq2: defines.HPFCornerFreq2 = defines.HPFCornerFreq2.KHZ350,
        rxGain: int = 30
    ) -> None:
        """Configure chirp profile(s).

        See the ramp timing calculator in [mmWave Studio](
        https://www.ti.com/tool/MMWAVE-STUDIO) for chirp timing details, and
        the [AWR1843 Datasheet](
        https://www.ti.com/lit/ds/symlink/awr1843.pdf?ts=1708800208074) for
        frequency and sample rate constraints.

        Args:
            profileId: profile to configure. Can only have one in
                `DFEMode.LEGACY`.
            startFreq: chirp start frequency, in GHz. Can be 76 or 77.
            idleTime: chirp timing; see the "RampTimingCalculator".
            adcStartTime: chirp timing; see the "RampTimingCalculator".
            rampEndTime: chirp timing; see the "RampTimingCalculator".
            txStartTime: chirp timing; see the "RampTimingCalculator".
            txOutPower: not entirely clear what this does. The
                demo claims that only '0' is tested / should be used.
            txPhaseShifter: not entirely clear what this does. The
                demo claims that only '0' is tested / should be used.
            freqSlopeConst: frequency slope ("ramp rate") in MHz/us; <100MHz/us.
            numAdcSamples: Number of ADC samples per chirp.
            digOutSampleRate: ADC sample rate in ksps (<12500); see
                Table 8-4 in the AWR1843 Datasheet.
            hpfCornerFreq1: high pass filter corner frequencies.
            hpfCornerFreq2: high pass filter corner frequencies.
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

        See the [mmWave SDK user guide](
        https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf),
        Table 1 (Page 19).

        Args:
            chirpIdx: Antenna index. Sets `chirpStartIdx`, `chirpEndIdx`
                to `chirpIdx`, and `txEnable` (antenna bitmask) to
                `1 << chirpIdx`.
            profileId: chirp profile to use.
            startFreqVar: allowed frequency tolerance; documentation states
                only 0 is tested.
            freqSlopeVar: allowed frequency tolerance; documentation states
                only 0 is tested.
            idleTimeVar: allowed time tolerance; documentation states
                only 0 is tested.
            adcStartTimeVar: allowed time tolerance; documentation states
                only 0 is tested.
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

        !!! warning

            The frame should not have more than a 50% duty cycle according to
            the mmWave SDK documentation.

        Args:
            chirpStartIdx: chirps to use in the frame.
            chirpEndIdx: chirps to use in the frame.
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

        !!! note

            rx_phase must have one term per TX-RX pair.
        """
        args = ' '.join("{} {}".format(re, im) for re, im in rx_phase)
        cmd = "compRangeBiasAndRxChanPhase {} {}".format(rangeBias, args)
        self.send(cmd)

    def lvdsStreamCfg(
        self, subFrameIdx: int = -1, enableHeader: bool = False,
        dataFmt: defines.LVDSFormat = defines.LVDSFormat.ADC,
        enableSW: bool = False
    ) -> None:
        """Configure LVDS stream (to the DCA1000EVM); `LvdsStreamCfg`.

        Args:
            subFrameIdx: subframe to apply to. If `-1`, applies to all
                subframes.
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


class BoilerplateMixins:
    """Mixins capturing non-relevant parts of the TI AWR Demo API.

    These configuration class are required for the software to not cause an
    error, but are not actually relevant to the output.

    !!! note

        The arguments used here are generally not fully documented. If there
        are any use cases for these commands, we can properly document them and
        move them to the main API.
    """

    def send(self, cmd: str, timeout: float = 10.0) -> None:
        raise NotImplementedError()

    def boilerplate_setup(self) -> None:
        """Call mandatory but irrelevant commands."""
        self.lowPower()
        self.guiMonitor()
        self.cfarCfg(procDirection=0)
        self.cfarCfg(procDirection=1)
        self.multiObjBeamForming()
        self.calibDcRangeSig()
        self.clutterRemoval()
        self.aoaFovCfg()
        self.cfarFovCfg(procDirection=0)
        self.cfarFovCfg(procDirection=1)
        self.measureRangeBiasAndRxChanPhase()
        self.extendedMaxVelocity()
        self.CQRxSatMonitor()
        self.CQSigImgMonitor()
        self.analogMonitor()
        self.calibData()

    def lowPower(self, dontCare: int = 0, adcMode: int = 0) -> None:
        """Low power mode config."""
        cmd = "lowPower {} {}".format(dontCare, adcMode)
        self.send(cmd)

    def guiMonitor(
        self, subFrameIdx: int = -1, detectedObjects: int = 0,
        logMagRange: int = 0, noiseProfile: int = 0,
        rangeAzimuthHeatMap: int = 0, rangeDopplerHeatMap: int = 0,
        statsInfo: int = 0
    ) -> None:
        """Set GUI exports.

        !!! note

            We disable everything to minimize the chances of interference.
        """
        cmd = "guiMonitor {} {} {} {} {} {} {}".format(
            subFrameIdx, detectedObjects, logMagRange, noiseProfile,
            rangeAzimuthHeatMap, rangeDopplerHeatMap, statsInfo)
        self.send(cmd)

    def cfarCfg(
        self, subFrameIdx: int = -1, procDirection: int = 1,
        averageMode: int = 0, winLen: int = 4, guardLen: int = 2,
        noiseDivShift: int = 3, cyclicMode: int = 1, threshold: float = 15.0,
        peakGroupingEn: int = 1
    ) -> None:
        """Configure CFAR.

        !!! note

            This command must be called twice for `procDirection=0, 1`.
        """
        cmd = "cfarCfg {} {} {} {} {} {} {} {} {}".format(
            subFrameIdx, procDirection, averageMode, winLen, guardLen,
            noiseDivShift, cyclicMode, threshold, peakGroupingEn)
        self.send(cmd)

    def multiObjBeamForming(
        self, subFrameIdx: int = -1, enabled: int = 0, threshold: float = 0.5
    ) -> None:
        """Configure multi-object beamforming."""
        cmd = "multiObjBeamForming {} {} {}".format(
            subFrameIdx, enabled, threshold)
        self.send(cmd)

    def calibDcRangeSig(
        self, subFrameIdx: int = -1, enabled: int = 0,
        negativeBinIdx: int = -5, positiveBinIdx: int = 8,
        numAvgFrames: int = 256
    ) -> None:
        """DC range calibration at radar start.

        !!! quote "TI's note"
            Antenna coupling signature dominates the range bins close to
            the radar. These are the bins in the range FFT output located
            around DC.

            When this feature is enabled, the signature is estimated during
            the first N chirps, and then it is subtracted during the
            subsequent chirps

        !!! info

            Rover performs this step during offline data processing.
        """
        cmd = "calibDcRangeSig {} {} {} {} {}".format(
            subFrameIdx, enabled, negativeBinIdx, positiveBinIdx, numAvgFrames)
        self.send(cmd)

    def clutterRemoval(self, subFrameIdx: int = -1, enabled: int = 0) -> None:
        """Static clutter removal."""
        cmd = "clutterRemoval {} {}".format(subFrameIdx, enabled)
        self.send(cmd)


    def aoaFovCfg(
        self, subFrameIdx: int = -1, minAzimuthDeg: int = -90,
        maxAzimuthDeg: int = 90, minElevationDeg: int = -90,
        maxElevationDeg: int = 90
    ) -> None:
        """FOV limits for CFAR."""
        cmd = "aoaFovCfg {} {} {} {} {}".format(
            subFrameIdx, minAzimuthDeg, maxAzimuthDeg,
            minElevationDeg, maxElevationDeg)
        self.send(cmd)

    def cfarFovCfg(
        self, subFrameIdx: int = -1, procDirection: int = 0,
        min_meters_or_mps: float = 0, max_meters_or_mps: float = 0
    ) -> None:
        """Range/doppler limits for CFAR.

        !!! note

            Must be called twice for `procDirection=0, 1`.
        """
        cmd = "cfarFovCfg {} {} {} {}".format(
            subFrameIdx, procDirection, min_meters_or_mps, max_meters_or_mps)
        self.send(cmd)

    def measureRangeBiasAndRxChanPhase(
        self, enabled: int = 0, targetDistance: float = 1.5,
        searchWin: float = 0.2
    ) -> None:
        """Only used in a specific calibration procedure."""
        cmd = "measureRangeBiasAndRxChanPhase {} {} {}".format(
            enabled, targetDistance, searchWin)
        self.send(cmd)

    def extendedMaxVelocity(
        self, subFrameIdx: int = -1, enabled: int = 0
    ) -> None:
        """Velocity disambiguation feature."""
        cmd = "extendedMaxVelocity {} {}".format(subFrameIdx, enabled)
        self.send(cmd)

    def CQRxSatMonitor(
        self, profile: int = 0, satMonSel: int = 3, priSliceDuration: int = 5,
        numSlices: int = 121, rxChanMask: int = 0
    ) -> None:
        """Saturation monitoring."""
        cmd = "CQRxSatMonitor {} {} {} {} {}".format(
            profile, satMonSel, priSliceDuration, numSlices, rxChanMask)
        self.send(cmd)

    def CQSigImgMonitor(
        self, profile: int = 0, numSlices: int = 127,
        numSamplePerSlice: int = 4
    ) -> None:
        """Signal/image band energy monitoring."""
        cmd = "CQSigImgMonitor {} {} {}".format(
            profile, numSlices, numSamplePerSlice)
        self.send(cmd)

    def analogMonitor(
        self, rxSaturation: int = 0, sigImgBand: int = 0
    ) -> None:
        """Enable/disable monitoring."""
        cmd = "analogMonitor {} {}".format(rxSaturation, sigImgBand)
        self.send(cmd)

    def calibData(
        self, save_enable: int = 0, restore_enable: int = 0,
        Flash_offset: int = 0
    ) -> None:
        """Save/restore RF calibration data."""
        cmd = "calibData {} {} {}".format(
            save_enable, restore_enable, Flash_offset)
        self.send(cmd)
