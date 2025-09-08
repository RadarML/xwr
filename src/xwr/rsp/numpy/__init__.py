"""Radar Signal Processing in Numpy.

!!! tip

    We use [pyfftw](https://pyfftw.readthedocs.io/en/latest/index.html)
    to perform FFTs, which wraps [FFTW](https://www.fftw.org/). In our
    testing for computing `64x3x4x256` range-Doppler frames, this
    provides a ~5x speedup over `np.fft.fftn` for single frames and a
    ~10x speedup for batches of 8 frames.

    FFTW plans are also cached for efficiency: the first time a particular
    shape and axes are requested from `RSPNumpy.fft`, a copy of the array
    is provided to fftw to create a plan, which is saved by `RSPNumpy`.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from xwr.rsp import numpy as rsp
    ```
"""

from jaxtyping import install_import_hook

with install_import_hook("xwr.rsp.numpy", "beartype.beartype"):
    from .rsp import AWR1843AOP, AWR1642Boost, AWR1843Boost, RSPNumpy


__all__ = [
    "AWR1642Boost",
    "AWR1843AOP",
    "AWR1843Boost",
    "RSPNumpy",
]
