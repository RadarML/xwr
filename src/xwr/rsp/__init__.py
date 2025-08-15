"""Radar Signal Processing in numpy for batched 4D spectrum.

!!! warning "Image Axis Order"

    Elevation and azimuth axes are in "image order": increasing index is
    down and to the right, respectively.

To use the RSP:

1. Pick your backend. Currently, we support numpy, jax, and pytorch.

    !!! note

        Each RSP backend is not imported by default; you must explicitly import
        the backend you want to use, and make sure its dependencies are
        installed (pytorch, jax, etc).
        ```python
        from xwr.rsp import numpy as rsp
        # or
        from xwr.rsp import jax as rsp
        ```

2. Select the appropriate radar model.
3. Import the RSP class matching your backend and radar:
    ```python
    from xwr.rsp import RSP
    from xwr.rsp.torch import AWR1843AOP

    rsp: RSP = AWR1843AOP()
    ```

    !!! tip

        Use [`xwr.rsp.RSP`][xwr.rsp.RSP] as the type for a generic RSP, and
        `RSP[np.ndarray]`, `RSP[jax.Array]`, `RSP[torch.Tensor]`, etc for a
        RSP with a specific backend.
"""
from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp", "beartype.beartype"):
    from .generic import RSP, iq_from_iiqq, iqiq_from_iiqq


__all__ = [
    "RSP", "iq_from_iiqq", "iqiq_from_iiqq"
]
