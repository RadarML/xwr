"""Simple range-doppler and range-azimuth visualization demo."""

import logging
import os
from collections import deque
from queue import Empty

import numpy as np
import tyro
import yaml
from rich.logging import RichHandler
from viz import FrameRateLogger, RadarPlot, VisualizationConfig

import xwr
from xwr.rsp import numpy as xwr_rsp


def _run_dstream(awr, rsp_inst, plot: RadarPlot, framerate: FrameRateLogger):
    """Process frames as they arrive, dropping any if we fall behind."""
    for frame in awr.dstream(numpy=True):
        # batch doppler tx rx range
        dear = np.abs(rsp_inst(frame[None, ...]))
        framerate.tick()
        plot.update(dear, framerate.fps)


def _run_continuous(
    awr, rsp_inst, plot: RadarPlot, framerate: FrameRateLogger, doppler: int
):
    """Accumulate individual chirps, processing the most recent `doppler`.

    Each read is a single chirp (`frame_length == 1`); doppler processing is
    handled here in software by accumulating chirps into a rolling window.
    """
    chirps: deque[np.ndarray] = deque(maxlen=doppler)
    q = awr.qstream(numpy=True)
    while True:
        # Block for the next chirp, then drain any others already queued,
        # so we always process the most recently accumulated chirps,
        # discarding the rest.
        chirp = q.get(block=True)
        if chirp is None:
            break
        chirps.append(chirp)

        stopped = False
        while True:
            try:
                chirp = q.get_nowait()
            except Empty:
                break
            if chirp is None:
                stopped = True
                break
            chirps.append(chirp)
        if stopped:
            break

        if len(chirps) < doppler:
            continue

        # batch doppler tx rx range
        frame = np.concatenate(chirps, axis=0)
        dear = np.abs(rsp_inst(frame[None, ...]))
        framerate.tick()
        plot.update(dear, framerate.fps)


def cli_main(
    config: str | None = None,
    rsp: str = "AWR1843Boost",
    device: str | None = None,
    vis: VisualizationConfig = VisualizationConfig(),
    verbose: int = 20,
):
    """Range-doppler visualization demo.

    Using the default configuration, you will need to set the `device` field
    and `rsp` field to match your radar; see the xwr documentation
    (https://radarml.github.io/xwr/) for more details.

    Note: configurations specifying `frame_length: 1` are treated as continuous
    chirping, where slow-time frame accumulation is handled in software.

    Args:
        config: path to configuration file. If not provided, defaults to the
            included `config.yaml`.
        rsp: radar signal processing class to use; is responsible for handling
            the virtual antenna array for angular spectrum computation.
        device: optional device name; if provided, overrides (or sets) the
            radar `device` field in the configuration file.
        vis: spectrum visualization settings.
        verbose: logging verbosity level (10-debug; 20-info; 30-warning;
            40-error).
    """
    logging.basicConfig(
        level=verbose, format="%(name)-12s  %(message)s", datefmt="[%H:%M:%S]",
        handlers=[RichHandler()])
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    log = logging.getLogger("xwr/demo")

    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    if device is not None:
        cfg["radar"]["device"] = device

    continuous = cfg["radar"]["frame_length"] == 1

    awr = xwr.XWRSystem(**cfg)

    rsp_inst = getattr(xwr_rsp, rsp)(
        window=False, size={"azimuth": vis.azimuth})

    if rsp_inst.SAMPLE_TYPE == "I":
        Nr = cfg["radar"]["adc_samples"] // 2 + 1
    else:
        Nr = cfg["radar"]["adc_samples"]

    n_doppler = vis.doppler if continuous else cfg["radar"]["frame_length"]
    plot = RadarPlot(
        n_range=Nr, n_doppler=n_doppler, n_azimuth=vis.azimuth,
        max_range=awr.config.max_range, max_doppler=awr.config.max_doppler,
        range_resolution=awr.config.range_resolution,
        pclip=vis.pclip, power=vis.power)

    framerate = FrameRateLogger(log)
    try:
        if continuous:
            _run_continuous(awr, rsp_inst, plot, framerate, vis.doppler)
        else:
            _run_dstream(awr, rsp_inst, plot, framerate)

    except KeyboardInterrupt:
        log.warning("Demo interrupted by user.")
        awr.stop()

if __name__ == "__main__":
    tyro.cli(cli_main)
