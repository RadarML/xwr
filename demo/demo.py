"""Simple range-doppler, range-azimuth, and elevation visualization demo."""

import logging
import os
import re
from collections import deque
from queue import Empty

import numpy as np
import tyro
import yaml
from rich.logging import RichHandler
from viz import FrameRateLogger, RadarPlot, VisualizationConfig

import xwr
from xwr.rsp import numpy as xwr_rsp

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

DEFAULT_RSP = {
    "AWR1642": "AWR1642Boost",
    "AWR1843": "AWR1843Boost",
    "AWR1843L": "AWR1642Boost",
    "AWR2944": "AWR2944EVM",
    "AWR2944P": "AWR2944EVM",
    "AWR2E44P": "AWR2E44PEVM",
    "AWRL6844": "AWRL6844EVM",
    "AWRL6888": "AWRL6888EVM",
}
"""Default `--rsp` for each `--device`."""


def resolve_device(name: str) -> str:
    """Resolve a (possibly partial) device name to an `xwr.radar` class name.

    Matching is case-insensitive, and the `AWR` / `AWRL` prefix may be
    omitted: `1843`, `l6888`, and `2e44p` resolve to `AWR1843`, `AWRL6888`,
    and `AWR2E44P`. An exact match always wins; otherwise, the name must be
    the end of exactly one device name.

    Raises:
        ValueError: if no device or more than one device matches.
    """
    devices = [
        k for k in xwr.radar.__all__
        if isinstance(v := getattr(xwr.radar, k), type)
        and issubclass(v, xwr.radar.XWRBase) and v is not xwr.radar.XWRBase]
    query = name.lower()

    exact = [d for d in devices if d.lower() == query]
    if exact:
        return exact[0]

    partial = [d for d in devices if d.lower().endswith(query)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(
            f"Device '{name}' is ambiguous: could be {', '.join(partial)}.")
    raise ValueError(
        f"Unknown device '{name}'; expected one of {', '.join(devices)}.")


def _pattern_matches(pattern: str, name: str) -> bool:
    """Check if a config name matches a device, with `x` as a wildcard."""
    return len(pattern) == len(name) and all(
        p == "x" or p == c for p, c in zip(pattern, name))


def find_config(device: str, config_dir: str = CONFIG_DIR) -> str:
    """Find the demo config for a device class name.

    Config files are named after the device they apply to, in lowercase,
    with `x` matching any single character (e.g., `awr2x44.yaml` matches
    `AWR2944` and `AWR2E44P`). If the device name has a suffix (e.g. the `P`
    in `AWR2944P`), configs for the full name are tried first, then configs
    for the name without the suffix. Among matches, the most specific
    (fewest `x`) config wins.

    Raises:
        ValueError: if no config matches, or the best matches are tied.
    """
    configs = {
        os.path.splitext(f)[0].lower(): os.path.join(config_dir, f)
        for f in sorted(os.listdir(config_dir)) if f.endswith(".yaml")}

    name = device.lower()
    candidates = [name]
    base = re.sub(r"(?<=\d)[a-z]+$", "", name)
    if base != name:
        candidates.append(base)

    for candidate in candidates:
        matches = [c for c in configs if _pattern_matches(c, candidate)]
        if not matches:
            continue
        fewest = min(c.count("x") for c in matches)
        best = [c for c in matches if c.count("x") == fewest]
        if len(best) > 1:
            raise ValueError(
                f"Multiple configs match {device}: "
                f"{', '.join(configs[c] for c in best)}.")
        return configs[best[0]]

    raise ValueError(
        f"No config for {device} in {config_dir}; pass one with --config.")


def load_config(
    device: str | None, config: str | None, rsp: str | None
) -> tuple[dict, str, str]:
    """Load the demo config and pick the signal processing class.

    Returns:
        The config, the config path, and the `xwr.rsp` class name.

    Raises:
        ValueError: if the device, config, or rsp cannot be determined.
    """
    if config is None:
        if device is None:
            raise ValueError("Specify the radar with --device (or --config).")
        device = resolve_device(device)
        config = find_config(device)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = cfg["radar"].get("device")
        if device is None:
            raise ValueError(
                f"{config} does not specify `radar.device`; pass --device.")
    cfg["radar"]["device"] = device = resolve_device(device)

    if rsp is None:
        rsp = DEFAULT_RSP.get(device)
        if rsp is None:
            raise ValueError(f"No default --rsp for {device}; pass --rsp.")

    return cfg, config, rsp


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
    device: str | None = None,
    config: str | None = None,
    rsp: str | None = None,
    vis: VisualizationConfig = VisualizationConfig(),
    verbose: int = 20,
):
    """Range-doppler visualization demo.

    Select your radar with `--device`; the demo then picks the matching config
    from `config/` and a default `--rsp`. See the xwr documentation
    (https://radarml.github.io/xwr/) for more details.

    Note: configurations specifying `frame_length: 1` are treated as continuous
    chirping, where slow-time frame accumulation is handled in software.

    Args:
        device: radar device, i.e. a class in `xwr.radar`. Case-insensitive,
            and the `AWR` / `AWRL` prefix may be omitted (e.g. `1843`, `6888`).
            Overrides the `radar.device` field of `--config`.
        config: path to a configuration file. If not provided, the config in
            `config/` matching `--device` is used.
        rsp: radar signal processing class to use; is responsible for handling
            the virtual antenna array for angular spectrum computation. If
            not provided, defaults to the usual board for `--device` (e.g.
            `AWR1843Boost` for the `AWR1843`).
        vis: spectrum visualization settings.
        verbose: logging verbosity level (10-debug; 20-info; 30-warning;
            40-error).
    """
    logging.basicConfig(
        level=verbose, format="%(name)-12s  %(message)s", datefmt="[%H:%M:%S]",
        handlers=[RichHandler()])
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    log = logging.getLogger("xwr/demo")

    try:
        cfg, config, rsp = load_config(device, config, rsp)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None
    log.info(
        f"Device {cfg['radar']['device']}, config {config}, rsp {rsp}.")

    continuous = cfg["radar"]["frame_length"] == 1

    # Only zero-pad elevation if the virtual array has more than one row.
    rsp_cls = getattr(xwr_rsp, rsp)
    device_cls = getattr(xwr.radar, cfg["radar"]["device"])
    n_elevation = rsp_cls().mimo_virtual_array(np.zeros(
        (1, 1, device_cls.NUM_TX, device_cls.NUM_RX, 1), dtype=np.complex64)
    ).shape[2]
    size = {"azimuth": vis.azimuth}
    if n_elevation > 1 and vis.elevation is not None:
        size["elevation"] = n_elevation = vis.elevation
    rsp_inst = rsp_cls(
        window=False, size=size, doppler_correction=vis.doppler_correction)

    if rsp_inst.SAMPLE_TYPE == "I":
        Nr = cfg["radar"]["adc_samples"] // 2 + 1
    else:
        Nr = cfg["radar"]["adc_samples"]

    with xwr.XWRSystem(**cfg) as awr:
        n_doppler = vis.doppler if continuous else cfg["radar"]["frame_length"]
        plot = RadarPlot(
            n_range=Nr, n_doppler=n_doppler, n_azimuth=vis.azimuth,
            n_elevation=n_elevation, min_range=vis.min_range,
            max_range=awr.config.max_range,
            max_doppler=awr.config.max_doppler,
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

if __name__ == "__main__":
    tyro.cli(cli_main)
