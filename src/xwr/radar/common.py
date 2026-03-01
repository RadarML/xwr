"""Raw, unabstracted commands.

These commands are commonly issued by dumping the text file output of the TI
demo visualizer onto a serial port. However, this is not great for debugging,
and the individual parameters are not documented. We instead split commands
into documented and individually callable functions.
"""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803, E501

from . import defines


def configure_adc(
    adc_fmt: defines.ADCFormat = defines.ADCFormat.COMPLEX_1X,
) -> str:
    """Configure radar ADC.

    Args:
        adc_fmt: ADC output format (real or complex).

    Returns:
        A string containing multiple lines of commands to send.
    """
    return (
        "adcCfg {numADCBits.value} {adcOutputFmt.value}\n"
        "adcbufCfg {subFrameIdx} {adcbufFmt} "
        "{sampleSwap.value} {chanInterleave} {chirpThreshold}"
    ).format(
        numADCBits=defines.ADCDepth.BIT16,  # only 16-bit is used
        subFrameIdx=-1,                     # all subframes
        adcOutputFmt=adc_fmt,               # 0=complex, 1=real
        adcbufFmt=int(adc_fmt == defines.ADCFormat.REAL),
        sampleSwap=defines.SampleSwap.MSB_LSB_IQ,   # we assume IQ order
        chanInterleave=1,               # only non-interleaved (1) is supported
        chirpThreshold=1                # unknown ping-pong demo parameter
    )


def configure_channels(rx: int = 0b1111, tx: int = 0b111) -> str:
    """Configure channels and chirps for time-division multiplexing.

    Assigns one sequential chirp per enabled TX antenna (LSB-first).

    Used By: AWR1843, AWR1843L, AWR1642, AWR2944.

    Args:
        rx: RX channel bitmask, e.g. ``0b1111`` for 4 RX antennas.
        tx: TX channel bitmask, e.g. ``0b111`` for 3 TX antennas.

    Returns:
        A string containing multiple lines of commands to send.
    """
    a = {
        "cascading": 0,        # must always be 0
        "profileId": 0,
        "startFreqVar": 0.0,   # frequency tolerance; only 0 is tested
        "freqSlopeVar": 0.0,   # frequency tolerance; only 0 is tested
        "idleTimeVar": 0.0,    # time tolerance; only 0 is tested
        "adcStartTimeVar": 0.0,  # time tolerance; only 0 is tested
    }
    lines = [f"channelCfg {rx} {tx} {a['cascading']}"]
    chirp = 0
    i_tx = 0
    tx_remaining = tx
    # Add a chirp for each tx
    while tx_remaining > 0:
        if tx_remaining & 1:
            lines.append(
                f"chirpCfg {chirp} {chirp} {a['profileId']} {a['startFreqVar']} "
                f"{a['freqSlopeVar']} {a['idleTimeVar']} {a['adcStartTimeVar']} "
                f"{1 << i_tx}")  # txEnable bitmask for this TX antenna
            chirp += 1
        tx_remaining >>= 1
        i_tx += 1
    return "\n".join(lines)


def get_boilerplate() -> str:
    """Execute mandatory-but-irrelevant TI demo firmware commands.

    Used by: AWR1843, AWR1843L, AWR1642, AWR2944.

    Returns:
        A string containing multiple lines of commands to send.
    """
    a = {
        "subFrameIdx": -1,
        "detectedObjects": 0, "logMagRange": 0, "noiseProfile": 0,
        "rangeAzimuthHeatMap": 0, "rangeDopplerHeatMap": 0, "statsInfo": 0,
        "averageMode": 0, "winLen": 4, "guardLen": 2,
        "noiseDivShift": 3, "cyclicMode": 1, "threshold": 15.0,
        "peakGroupingEn": 1,
        "mobeEnabled": 0, "mobeThreshold": 0.5,
        "calibEnabled": 0, "negativeBinIdx": -5, "positiveBinIdx": 8,
        "numAvgFrames": 256,
        "clutterEnabled": 0,
        "minAzimuthDeg": -90, "maxAzimuthDeg": 90, "minElevationDeg": -90,
        "maxElevationDeg": 90,
        "min_meters_or_mps": 0, "max_meters_or_mps": 0,
        "measureEnabled": 0, "targetDistance": 1.5, "searchWin": 0.2,
        "extVelEnabled": 0,
        "profile": 0, "satMonSel": 3, "priSliceDuration": 5,
        "cqNumSlices": 121, "rxChanMask": 0,
        "sigImgNumSlices": 127, "numSamplePerSlice": 4,
        "rxSaturation": 0, "sigImgBand": 0,
        "save_enable": 0, "restore_enable": 0, "flash_offset": 0,
    }
    return (
        '# Disabled to minimize chances of interference.\n'
        f'guiMonitor {a["subFrameIdx"]} {a["detectedObjects"]} {a["logMagRange"]}'
        f' {a["noiseProfile"]} {a["rangeAzimuthHeatMap"]} {a["rangeDopplerHeatMap"]} {a["statsInfo"]}\n'
        '# Called twice for procDirection=0, 1.\n'
        f'cfarCfg {a["subFrameIdx"]} 0 {a["averageMode"]} {a["winLen"]} {a["guardLen"]}'
        f' {a["noiseDivShift"]} {a["cyclicMode"]} {a["threshold"]} {a["peakGroupingEn"]}\n'
        f'cfarCfg {a["subFrameIdx"]} 1 {a["averageMode"]} {a["winLen"]} {a["guardLen"]}'
        f' {a["noiseDivShift"]} {a["cyclicMode"]} {a["threshold"]} {a["peakGroupingEn"]}\n'
        f'multiObjBeamForming {a["subFrameIdx"]} {a["mobeEnabled"]} {a["mobeThreshold"]}\n'
        '# Perform this step during offline data processing.\n'
        f'calibDcRangeSig {a["subFrameIdx"]} {a["calibEnabled"]}'
        f' {a["negativeBinIdx"]} {a["positiveBinIdx"]} {a["numAvgFrames"]}\n'
        f'clutterRemoval {a["subFrameIdx"]} {a["clutterEnabled"]}\n'
        f'aoaFovCfg {a["subFrameIdx"]} {a["minAzimuthDeg"]} {a["maxAzimuthDeg"]}'
        f' {a["minElevationDeg"]} {a["maxElevationDeg"]}\n'
        '# Called twice for procDirection=0, 1.\n'
        f'cfarFovCfg {a["subFrameIdx"]} 0 {a["min_meters_or_mps"]} {a["max_meters_or_mps"]}\n'
        f'cfarFovCfg {a["subFrameIdx"]} 1 {a["min_meters_or_mps"]} {a["max_meters_or_mps"]}\n'
        '# Only used in a specific calibration procedure.\n'
        f'measureRangeBiasAndRxChanPhase {a["measureEnabled"]} {a["targetDistance"]} {a["searchWin"]}\n'
        f'extendedMaxVelocity {a["subFrameIdx"]} {a["extVelEnabled"]}\n'
        f'CQRxSatMonitor {a["profile"]} {a["satMonSel"]} {a["priSliceDuration"]}'
        f' {a["cqNumSlices"]} {a["rxChanMask"]}\n'
        f'CQSigImgMonitor {a["profile"]} {a["sigImgNumSlices"]} {a["numSamplePerSlice"]}\n'
        f'analogMonitor {a["rxSaturation"]} {a["sigImgBand"]}\n'
        f'calibData {a["save_enable"]} {a["restore_enable"]} {a["flash_offset"]}'
    )


class APIMixins:
    """Mixins capturing the raw TI Demo API."""

    def send(self, cmd: str, timeout: float = 10.0) -> None:
        raise NotImplementedError()

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
                Figure 7-1 (sec. 7.7) in the AWR1843 Datasheet.
            hpfCornerFreq1: high pass filter corner frequencies.
            hpfCornerFreq2: high pass filter corner frequencies.
            rxGain: RX gain in dB. The meaning of this value is not clear.
        """
        assert startFreq in {76.0, 77.0}
        # TODO: check these by radar
        # AWR1843 is 100, 12500
        # AWR2944 is 250, 45000 (sec 7.8, RF Specifications)
        # assert freqSlopeConst < 100.0
        # assert digOutSampleRate < 12500

        self.send(
            f"profileCfg {profileId} {startFreq} {idleTime} {adcStartTime} "
            f"{rampEndTime} {txOutPower} {txPhaseShifter} {freqSlopeConst} "
            f"{txStartTime} {numAdcSamples} {digOutSampleRate} "
            f"{hpfCornerFreq1.value} {hpfCornerFreq2.value} {rxGain}")

    def frameCfg(
        self, chirpStartIdx: int = 0, chirpEndIdx: int = 1, numLoops: int = 16,
        numFrames: int = 0, framePeriodicity: float = 100.0,
        triggerSelect: int = 1, frameTriggerDelay: float = 0.0
    ) -> None:
        """Radar frame configuration.

        !!! warning

            The frame should not have more than a 50% RF duty cycle according
            to the mmWave SDK documentation.

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
        self.send(
            f"frameCfg {chirpStartIdx} {chirpEndIdx} {numLoops} {numFrames} "
            f"{framePeriodicity} {triggerSelect} {frameTriggerDelay}")

    def compRangeBiasAndRxChanPhase(
        self, rangeBias: float = 0.0,
        rx_phase: list[tuple[int, int]] = [(0, 1)] * 12
    ) -> None:
        """Set range bias, channel phase compensation.

        !!! note

            rx_phase must have one term per TX-RX pair.
        """
        args = ' '.join(f"{re} {im}" for re, im in rx_phase)
        self.send(f"compRangeBiasAndRxChanPhase {rangeBias} {args}")

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
        self.send(
            f"lvdsStreamCfg {subFrameIdx} {1 if enableHeader else 0} "
            f"{dataFmt.value} {1 if enableSW else 0}")
