"""Backend-agnostic Angle of Arrival estimation and point cloud."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from jaxtyping import Bool, Float32, Int

from .generic import TArray


class PointCloud(ABC, Generic[TArray]):
    """Get radar point cloud from post FFT cube.

    To convert azimuth-elevation bin indices to azimuth-elevation angles,
    we use the property that the azimuth bin indices correspond to the sin of
    the angle
    ```
    angles = np.arcsin(
        np.clip(np.linspace(-1.0, 1.0, bin_size) / (2 * antenna_spacing),
                -1.0, 1.0)
    )
    ```
    where the *corrected* antenna spacing is calculated by
    ```
    0.5 * chirp_center_frequency / antenna_design_frequency
    ```

    !!! info

        The antenna design frequency here refers to the grid alignment of the
        antenna array, which are typically 0.5 wavelengths apart at some
        nominal design frequency. Thus, you must correct by a corresponding
        scale factor when the chirp center frequency differs.

    Implementation notes:

    - Range bins are mapped by `bin * range_res`, so bin `0` is zero range.
        Doppler bins are mapped to a *signed* value by
        `(bin - doppler // 2) * doppler_res`, so the middle bin is zero
        velocity; this assumes the doppler axis has already been `fftshift`ed,
        which [`doppler_range`][xwr.rsp.RSP.] does. With the default
        resolutions of `1.0`, the point cloud is in range/doppler bins instead
        of meters and meters/second.
    - `antenna_spacing` sets the sin-space to angle mapping and must be
        positive; a non-positive value raises `ValueError`, while a wrong
        (but positive) value gives systematically wrong angles rather than an
        error.
    - The bin-to-angle lookup tables are derived from the cube's own angle
        axes on each call, so no angle size has to be declared up front and
        one instance handles cubes of differing angle sizes.
    - `angle_fov` is applied as a symmetric `±fov` bound, and points falling
        outside it are excluded from the returned mask. This rejects
        estimates near the edge of the array's sin-space, where a sparse MIMO
        array has little real resolving power and grating lobes appear.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Args:
        range_res: range resolution, i.e. meters per range bin; see
            [`XWRConfig.range_resolution`][xwr.config.XWRConfig]. Defaults
            to `1.0`, which leaves the range axis in bins.
        doppler_res: doppler resolution, i.e. meters/second per doppler bin;
            see [`XWRConfig.doppler_resolution`][xwr.config.XWRConfig].
            Defaults to `1.0`, which leaves the doppler axis in bins.
        angle_fov: angle field of view **in degrees**, for
            (elevation, azimuth).
        antenna_spacing: antenna spacing in terms of wavelength (default 0.5
            for a half-wavelength grid).
    """

    def __init__(
        self,
        range_res: float = 1.0,
        doppler_res: float = 1.0,
        angle_fov: tuple[float, float] = (20.0, 80.0),
        antenna_spacing: float = 0.5,
    ) -> None:
        self.range_res = range_res
        self.doppler_res = doppler_res

        if antenna_spacing <= 0:
            raise ValueError("antenna_spacing must be > 0")
        self.antenna_spacing = antenna_spacing
        self.el_fov = float(np.deg2rad(angle_fov[0]))
        self.az_fov = float(np.deg2rad(angle_fov[1]))

    def _angle_table(self, n: int) -> Float32[np.ndarray, " n"]:
        """Bin-to-angle lookup for an angle axis of length `n`.

        Args:
            n: length of the angle axis, i.e. the angle fft size.

        Returns:
            Angle in radians for each bin, ascending.
        """
        sin = np.clip(
            np.linspace(-1.0, 1.0, n) / (2 * self.antenna_spacing), -1.0, 1.0)
        return np.arcsin(sin).astype(np.float32)

    @abstractmethod
    def aoa(
        self, cube: Float32[TArray, "batch range doppler el az"]
    ) -> Int[TArray, "batch range doppler 2"]:
        """Angle of arrival estimation.

        Takes the argmax over the (elevation, azimuth) angle spectrum of each
        range-doppler bin, yielding **bin indices** into the angle axes
        rather than angles.

        !!! warning

            This assumes at most one scatterer per range-doppler cell; if two
            targets share a range and velocity, only the stronger is reported.

        Args:
            cube: batch of post fft spectrum amplitudes, with the angle axes
                trailing.

        Returns:
            ang: detect angle index for every range doppler bin, as
                `(elevation, azimuth)` along the trailing axis.
        """
        ...

    @abstractmethod
    def __call__(
        self,
        cube: Float32[TArray, "batch doppler el az range"],
        mask: Bool[TArray, "batch range doppler"],
    ) -> tuple[
        Bool[TArray, "batch range doppler"],
        Float32[TArray, "batch range doppler 4"],
    ]:
        """Get point cloud from radar cube and detection mask.

        !!! note

            The returned point cloud is **dense**: every range-doppler bin
            yields a point, and the caller is expected to gather the valid
            ones using the returned mask (e.g. `pc[pc_mask]`).

        Args:
            cube: batch of post fft spectrum amplitudes.
            mask: CFAR detection mask.

        Returns:
            mask of valid points, i.e. the CFAR detection mask combined with
                the angular bounds set by `angle_fov`.
            all possible radar points, where the trailing axis holds
                `(x, y, z, v)`: position in meters and signed radial velocity
                in meters/second (in range/doppler bins if `range_res` and
                `doppler_res` are left at their default of `1.0`). Position is
                computed as
                `x = r cos(-az) cos(el)`, `y = r sin(-az) cos(el)`, and
                `z = r sin(el)`; note that the azimuth angle is **negated**,
                and that `x` is the boresight direction at zero angle.
        """
        ...
