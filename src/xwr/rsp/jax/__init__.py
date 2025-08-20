"""Radar Signal Processing in Jax.

!!! info

    This module mirrors the functionality of [`xwr.rsp.numpy`][xwr.rsp.numpy].

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
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost, RSPJax
    from .spectrum import CFAR, CalibratedSpectrum, CFAR_CASO
    from .aoa import PointCloud

__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "RSPJax",
    "CFAR",
    "CFAR_CASO",
    "CalibratedSpectrum",
    "PointCloud"
]
