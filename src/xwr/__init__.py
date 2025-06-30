"""TI mmWave Radar + DCA1000EVM Capture Card Raw Data Capture API.

!!! abstract "Usage"

    To use the high-level API, create a [`XWRConfig`][.] and [`DCAConfig`][.];
    then pass these to the [`XWRSystem`][.]. Use [`stream`][.XWRSystem.]
    or [`qstream`][.XWRSystem.] to automatically configure, start, and stream
    spectrum data from the radar.
"""

from beartype.claw import beartype_this_package

beartype_this_package()

# ruff: noqa: E402
from . import capture, radar, rsp
from .config import DCAConfig, XWRConfig
from .system import XWRSystem

__all__ = [
    "capture", "radar", "rsp", "XWRConfig", "XWRSystem", "DCAConfig"]
