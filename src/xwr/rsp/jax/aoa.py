"""Angle of Arrival Estimation and Point Cloud Module using JAX."""

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int


class PointCloud:
    """Get radar point cloud from post FFT cube.

    !!! note
        The angle bin distribution is in arcsin space.
        ```
        jnp.arcsin(jnp.linspace(-jnp.pi, jnp.pi, bin_size) / (2 * jnp.pi))
        ```

    Args:
        range_resolution: range fft resolution
        doppler_resolution: doppler fft resolution
        elevation_fov: elevation threshold in degree
        azimuth_fov: azimuth threshold in degree
    """

    def __init__(
        self,
        range_resolution: float,
        doppler_resolution: float,
        elevation_fov: float = 20.0,
        azimuth_fov: float = 80.0,
        ele_size: int = 128,
        azi_size: int = 128,
    ) -> None:
        self.range_res = range_resolution
        self.doppler_res = doppler_resolution
        self.ele_fov = jnp.deg2rad(elevation_fov)
        self.azi_fov = jnp.deg2rad(azimuth_fov)
        self.ele_angles = jnp.arcsin(
            jnp.linspace(-jnp.pi, jnp.pi, ele_size) / (2 * jnp.pi)
        )
        self.azi_angles = jnp.arcsin(
            jnp.linspace(-jnp.pi, jnp.pi, azi_size) / (2 * jnp.pi)
        )

    def argmax_aoa(
        self, ang_sptr: Float32[Array, "ele azi"]
    ) -> tuple[Array, ...]:
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
        idxs = jax.vmap(jax.vmap(self.argmax_aoa))(cube)
        ang = jnp.stack((idxs), axis=-1)
        return ang

    def __call__(
        self,
        cube: Float32[Array, "doppler ele azi range"],
        range_doppler_mask: Bool[Array, "range doppler"],
    ) -> tuple[Bool[Array, "range doppler"], Float32[Array, "range doppler 4"]]:
        """Get point cloud from radar cube and detection mask.

        Args:
            cube: post fft spectrum amplitude.
            range_doppler_mask: CFAR detection mask.

        Returns:
            pc_mask valid point mask
            pc all possible radar points
        """
        r_size, d_size = range_doppler_mask.shape
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

        pc_mask = jnp.logical_and(range_doppler_mask, mask_ang)
        pc = jnp.stack((x, y, z, v), axis=-1)

        return pc_mask, pc
