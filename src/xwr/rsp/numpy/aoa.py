"""Angle of Arrival Estimation and Point Cloud Module using Numpy."""

import numpy as np
from jaxtyping import Bool, Float, Float32, Int

from xwr.rsp import aoa as base


class PointCloud(base.PointCloud[np.ndarray]):
    """Get radar point cloud from post FFT cube."""

    @staticmethod
    def _asarray(x: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        return x

    @staticmethod
    def _argmax_aoa(ang_sptr: Float32[np.ndarray, "... el az"]) -> Int[
        np.ndarray, "... 2"
    ]:
        el, az = ang_sptr.shape[-2:]
        idx = np.argmax(ang_sptr.reshape(*ang_sptr.shape[:-2], el * az), -1)
        return np.stack((idx // az, idx % az), axis=-1)

    def _point_cloud(
        self,
        cube: Float32[np.ndarray, "batch doppler el az range"],
        mask: Bool[np.ndarray, "batch range doppler"],
    ) -> tuple[
        Bool[np.ndarray, "batch range doppler"],
        Float32[np.ndarray, "batch range doppler 4"],
    ]:
        _, r_size, d_size = mask.shape
        range_v = np.arange(r_size) * self.range_res
        doppler_v = (np.arange(d_size) - d_size // 2) * self.doppler_res
        r_grid, d_grid = np.meshgrid(range_v, doppler_v, indexing="ij")

        angle_idx = self.aoa(cube.transpose(0, 4, 1, 2, 3))
        ang_e = self.el_angles[angle_idx[..., 0]]
        ang_a = self.az_angles[angle_idx[..., 1]]
        mask_e = np.logical_and(ang_e < self.el_fov, ang_e > -self.el_fov)
        mask_a = np.logical_and(ang_a < self.az_fov, ang_a > -self.az_fov)
        mask_ang = np.logical_and(mask_a, mask_e)

        x = r_grid * np.cos(-ang_a) * np.cos(ang_e)
        y = r_grid * np.sin(-ang_a) * np.cos(ang_e)
        z = r_grid * np.sin(ang_e)
        v = np.broadcast_to(d_grid, x.shape)

        pc_mask = np.logical_and(mask, mask_ang)
        pc = np.stack((x, y, z, v), axis=-1).astype(np.float32)

        return pc_mask, pc
