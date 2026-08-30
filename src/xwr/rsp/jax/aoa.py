"""Angle of Arrival Estimation and Point Cloud Module using JAX."""

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Float32, Int

from xwr.rsp import aoa as base


class PointCloud(base.PointCloud[Array]):
    """Get radar point cloud from post FFT cube."""

    @staticmethod
    def _asarray(x: Float[np.ndarray, "..."]) -> Float[Array, "..."]:
        return jnp.asarray(x)

    @staticmethod
    def _argmax_aoa(ang_sptr: Float32[Array, "... el az"]) -> Int[
        Array, "... 2"
    ]:
        el, az = ang_sptr.shape[-2:]
        idx = jnp.argmax(ang_sptr.reshape(*ang_sptr.shape[:-2], el * az), -1)
        return jnp.stack((idx // az, idx % az), axis=-1)

    def _point_cloud(
        self,
        cube: Float32[Array, "batch doppler el az range"],
        mask: Bool[Array, "batch range doppler"],
    ) -> tuple[
        Bool[Array, "batch range doppler"],
        Float32[Array, "batch range doppler 4"],
    ]:
        _, r_size, d_size = mask.shape
        range_v = jnp.arange(r_size) * self.range_res
        doppler_v = (jnp.arange(d_size) - d_size // 2) * self.doppler_res
        r_grid, d_grid = jnp.meshgrid(range_v, doppler_v, indexing="ij")

        angle_idx = self.aoa(cube.transpose(0, 4, 1, 2, 3))
        ang_e = self.el_angles[angle_idx[..., 0]]
        ang_a = self.az_angles[angle_idx[..., 1]]
        mask_e = jnp.logical_and(ang_e < self.el_fov, ang_e > -self.el_fov)
        mask_a = jnp.logical_and(ang_a < self.az_fov, ang_a > -self.az_fov)
        mask_ang = jnp.logical_and(mask_a, mask_e)

        x = r_grid * jnp.cos(-ang_a) * jnp.cos(ang_e)
        y = r_grid * jnp.sin(-ang_a) * jnp.cos(ang_e)
        z = r_grid * jnp.sin(ang_e)
        v = jnp.broadcast_to(d_grid, x.shape)

        pc_mask = jnp.logical_and(mask, mask_ang)
        pc = jnp.stack((x, y, z, v), axis=-1)

        return pc_mask, pc
