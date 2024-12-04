"""Irrelevant, but required, AWR initialization steps."""

# NOTE: We ignore a few naming rules to maintain consistency with TI's naming.
# ruff: noqa: N802, N803


class AWR1843Mixins:
    """Mixins capturing non-relevant parts of the AWR1843 Demo API.

    These configuration class are required for the software to not cause an
    error, but are not actually relevant to the output.
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

        NOTE: We disable everything to minimize the chances of interference.
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

        NOTE: this command must be called twice for `procDirection=0, 1`.
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

        TI's note [R4]_:

            Antenna coupling signature dominates the range bins close to
            the radar. These are the bins in the range FFT output located
            around DC.

            When this feature is enabled, the signature is estimated during
            the first N chirps, and then it is subtracted during the
            subsequent chirps

        NOTE: Rover performs this step during offline data processing.
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

        NOTE: must be called twice for `procDirection=0, 1`.
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
