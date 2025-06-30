"""Simple range-doppler and range-azimuth visualization demo."""

import logging
import os

import numpy as np
import tyro
import yaml
from matplotlib import pyplot as plt
from rich.logging import RichHandler

import xwr
from xwr.rsp import numpy as xwr_rsp


def cli_main(
    config: str | None = None,
    rsp: str = "AWR1843Boost",
    device: str | None = None,
    verbose: int = 20,
):
    """Range-doppler visualization demo.

    Using the default configuration, you will need to set the `device` field
    and `rsp` field to match your radar. For example:

    - AWR1843Boost: --device AWR1843 --rsp AWR1843Boost
    - AWR1843AOPEVM: --device AWR1843 --rsp AWR1843AOP
    - AWR1642Boost: --device AWR1642 --rsp AWR1642Boost

    Args:
        config: path to configuration file. If not provided, defaults to the
            included `config.yaml`.
        rsp: radar signal processing class to use; is responsible for handling
            the virtual antenna array for angular spectrum computation.
        device: optional device name; if provided, overrides (or sets) the
            radar `device` field in the configuration file.
        verbose: logging verbosity level (10-debug; 20-info; 30-warning;
            40-error).
    """
    logging.basicConfig(
        level=verbose, format="%(name)-12s  %(message)s", datefmt="[%H:%M:%S]",
        handlers=[RichHandler()])
    log = logging.getLogger("XWRDemo")

    if config is None:
        config = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    if device is not None:
        cfg["radar"]["device"] = device

    # Create a figure
    plt.ion()  # Enable interactive mode
    fig, axs = plt.subplots(1, 2)

    im1 = axs[0].imshow(
        np.zeros((128, 128), dtype=np.float32),
        cmap="viridis", aspect='auto', origin='lower')
    im2 = axs[1].imshow(
        np.zeros((128, 128), dtype=np.float32),
        cmap="viridis", aspect='auto', origin='lower')

    axs[0].set_xlabel("Doppler")
    axs[0].set_ylabel("Range")
    axs[1].set_xlabel("Azimuth")
    axs[1].set_ylabel("Range")
    fig.tight_layout()

    awr = xwr.XWRSystem(**cfg)
    rsp_inst = getattr(xwr_rsp, rsp)(window=False, size={"azimuth": 128})

    try:
        for frame in awr.dstream(numpy=True):
            # batch doppler elevation azimuth range
            dear = np.abs(rsp_inst(frame[None, ...]))
            rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)
            ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)

            im1.set_data(rd)
            im1.set_clim(vmin=np.min(rd), vmax=np.max(rd))
            im2.set_data(ra)
            im2.set_clim(vmin=np.min(ra), vmax=np.max(ra))

            # Needed to update the figure
            plt.pause(0.001)

    except KeyboardInterrupt:
        log.warning("Demo interrupted by user.")
        awr.stop()

if __name__ == "__main__":
    tyro.cli(cli_main)
