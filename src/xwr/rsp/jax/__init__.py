"""Radar Signal Processing in jax.

!!! info

    This module mirrors the functionality of [`xwr.rsp`][xwr.rsp].

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from xwr.rsp import jax as xwr_jax
    ```

    Since jax is not declared as a required dependency, you will also need
    to install `jax` yourself (or install the `jax` extra with
    `pip install xwr[jax]`).
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp.jax", "beartype.beartype"):
    from .common import BaseRSP, iq_from_iiqq
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost


__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "BaseRSP",
    "iq_from_iiqq",
]
