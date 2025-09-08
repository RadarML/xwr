"""Radar Signal Processing in Jax.

!!! info

    In addition to mirroring the functionality of
    [`xwr.rsp.numpy`][xwr.rsp.numpy], this module also provides a range of
    point cloud processing algorithms.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from xwr.rsp import jax as rsp
    ```

    Since jax is not declared as a required dependency, you will also need
    to install `jax` yourself (or install the `jax` extra with
    `pip install xwr[jax]`).
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp.jax", "beartype.beartype"):
    from .aoa import PointCloud
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost, RSPJax
    from .spectrum import CFAR, CFARCASO, CalibratedSpectrum

__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "RSPJax",
    "CFAR",
    "CFARCASO",
    "CalibratedSpectrum",
    "PointCloud"
]
