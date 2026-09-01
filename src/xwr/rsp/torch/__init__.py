"""Radar Signal Processing in Pytorch.

!!! info

    In addition to mirroring the functionality of
    [`xwr.rsp.numpy`][xwr.rsp.numpy] and [`xwr.rsp.jax`][xwr.rsp.jax], this
    module also provides a range of point cloud processing algorithms,
    mirroring both.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from xwr.rsp import torch as xwr_torch
    ```

    Since pytorch is not declared as a required dependency, you will also need
    to install `torch` yourself (or install the `torch` extra with
    `pip install xwr[torch]`).

!!! tip
    The RSP implementations in this submodule support automatic differentiation
    in pytorch.
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp.torch", "beartype.beartype"):
    from .aoa import PointCloud  # noqa: I001
    from .rsp import (
        AWR1843AOP,
        AWR2944EVM,
        AWRL6844EVM,
        AWR1642Boost,
        AWR1843Boost,
        RSPTorch,
    )
    from .spectrum import CACFAR, CASOCFAR

__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "AWR2944EVM",
    "AWRL6844EVM",
    "RSPTorch",
    "CACFAR",
    "CASOCFAR",
    "PointCloud"
]
