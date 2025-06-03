"""Radar Signal Processing in Pytorch.

!!! info

    This module mirrors the functionality of [`xwr.rsp`][xwr.rsp].

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from xwr.rsp import torch as xwr_torch
    ```

    Since pytorch is not declared as a required dependency, you will also need
    to install `torch` yourself (or install the `torch` extra with
    `pip install xwr[torch]`).
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp.torch", "beartype.beartype"):
    from .common import BaseRSP, iq_from_iiqq
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost


__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "BaseRSP",
    "iq_from_iiqq",
]
