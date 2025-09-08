"""Angle of Arrival Estimation and Point Cloud Module using JAX."""

from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int


class PointCloud:
    """Get radar point cloud from post FFT cube.

    To convert azimuth-elevation bin indices to azimuth-elevation angles,
    we use the property that the azimuth bin indices correspond to the sin of
    the angle
    ```
    angles = jnp.arcsin(
        jnp.linspace(-jnp.pi, jnp.pi, bin_size)
        / (2 * jnp.pi * antenna_spacing)
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

    Args:
        range_resolution: range fft resolution
        doppler_resolution: doppler fft resolution
        angle_fov: angle field of view in degrees for (elevation, azimuth).
        angle_size: angle fft size for (elevation, azimuth).
        antenna_spacing: antenna spacing in terms of wavelength (default 0.5).
    """

    def __init__(
        self,
        range_resolution: float,
        doppler_resolution: float,
        angle_fov: Sequence[float] = (20.0, 80.0),
        angle_size: Sequence[int] = (128, 128),
        antenna_spacing: float = 0.5,
    ) -> None:
        self.range_res = range_resolution
        self.doppler_res = doppler_resolution

        assert len(angle_fov) == 2 and len(angle_size) == 2, (
            "angle_fov and angle_size must be a sequence of length 2."
        )
        self.ele_fov = jnp.deg2rad(angle_fov[0])
        self.azi_fov = jnp.deg2rad(angle_fov[1])
        self.ele_angles = jnp.arcsin(
            jnp.linspace(-jnp.pi, jnp.pi, angle_size[0])
            / (2 * jnp.pi * antenna_spacing)
        )
        self.azi_angles = jnp.arcsin(
            jnp.linspace(-jnp.pi, jnp.pi, angle_size[1])
            / (2 * jnp.pi * antenna_spacing)
        )

    @staticmethod
    def _argmax_aoa(ang_sptr: Float32[Array, "ele azi"]) -> tuple[Array, ...]:
        """Argmax for angle of arrival estimation.

        Args:
            ang_sptr: post fft angle spectrum amplitude in 2D.

        Returns:
            detected angle index (elevation, azimuth).
        """
        idx = jnp.argmax(ang_sptr)
        idx2d = jnp.unravel_index(idx, ang_sptr.shape)
        return idx2d

    def aoa(
        self, cube: Float32[Array, "range doppler ele azi"]
    ) -> Int[Array, "range doppler 2"]:
        """Angle of arrival estimation.

        Args:
            cube: post fft spectrum amplitude.

        Returns:
            ang: detect angle index for every range doppler bin.
        """
        idxs = jax.vmap(jax.vmap(self._argmax_aoa))(cube)
        ang = jnp.stack((idxs), axis=-1)
        return ang

    def __call__(
        self,
        cube: Float32[Array, "doppler ele azi range"],
        mask: Bool[Array, "range doppler"],
    ) -> tuple[Bool[Array, "range doppler"], Float32[Array, "range doppler 4"]]:
        """Get point cloud from radar cube and detection mask.

        Args:
            cube: post fft spectrum amplitude.
            mask: CFAR detection mask.

        Returns:
            mask of valid points (given the specified angular bounds)
            all possible radar points
        """
        r_size, d_size = mask.shape
        range_v = jnp.arange(r_size) * self.range_res
        doppler_v = (jnp.arange(d_size) - d_size // 2) * self.doppler_res
        r_grid, d_grid = jnp.meshgrid(range_v, doppler_v, indexing="ij")

        angle_idx = self.aoa(cube.transpose(3, 0, 1, 2))
        ang_e = self.ele_angles[angle_idx[:, :, 0]]
        ang_a = self.azi_angles[angle_idx[:, :, 1]]
        mask_e = jnp.logical_and(ang_e < self.ele_fov, ang_e > -self.ele_fov)
        mask_a = jnp.logical_and(ang_a < self.azi_fov, ang_a > -self.azi_fov)
        mask_ang = jnp.logical_and(mask_a, mask_e)

        x = r_grid * jnp.cos(-ang_a) * jnp.cos(ang_e)
        y = r_grid * jnp.sin(-ang_a) * jnp.cos(ang_e)
        z = r_grid * jnp.sin(-ang_e)
        v = d_grid

        pc_mask = jnp.logical_and(mask, mask_ang)
        pc = jnp.stack((x, y, z, v), axis=-1)

        return pc_mask, pc
