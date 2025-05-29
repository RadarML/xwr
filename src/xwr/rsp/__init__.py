"""Radar Signal Processing in numpy for batched 4D spectrum.

!!! warning "Image Axis Order"

    Elevation and azimuth axes are in "image order": increasing index is
    down and to the right, respectively.
"""
from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp", "beartype.beartype"):
    from .common import BaseRSP, iq_from_iiqq
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost


__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "BaseRSP",
    "iq_from_iiqq",
]
