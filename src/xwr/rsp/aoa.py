"""Backend-agnostic Angle of Arrival estimation and point cloud."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import numpy as np
from jaxtyping import Bool, Float32, Int

from .generic import TArray


@dataclass
class DensePoints(Generic[TArray]):
    """A dense radar point cloud, and the points in it which are valid.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Attributes:
        mask: mask of valid points, i.e. the CFAR detection mask combined with
            the angular bounds set by `angle_fov`.
        points: all possible radar points, where the trailing axis holds
            `(x, y, z, v)`.
    """

    mask: Bool[TArray, "batch range doppler"]
    points: Float32[TArray, "batch range doppler 4"]


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

    -  With the default range and dopplerresolutions of `1.0`, the point cloud
        is in range/doppler bins instead of meters and meters/second.
    - Radars have little resolving power close to the array plane in angle,
        and suffer from high noise at the edge of the main lobe. Reject these
        points with `angle_fov`.

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
        angle_fov: (elevation, azimuth) field of view **in degrees**, where points
            outside +/-elevation and +/-azimuth are rejected.
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
    ) -> DensePoints[TArray]:
        """Get point cloud from radar cube and detection mask.

        !!! note

            The returned point cloud is **dense**: every range-doppler bin
            yields a point, and the caller is expected to gather the valid
            ones using the returned mask (e.g. `pc[pc_mask]`).

        Implementation notes:

        - Return points are multiplied by `range_res` and `doppler_res` to convert
            from bins to meters and meters/second, respectively.
        - Points position compute as `x = r cos(-az) cos(el)`,
            `y = r sin(-az) cos(el)`, and `z = r sin(el)`

        Args:
            cube: batch of post fft spectrum amplitudes.
            mask: CFAR detection mask.

        Returns:
            The dense point cloud, and the mask of points in it which are
                valid; see [`DensePoints`][xwr.rsp.].
        """
        ...
