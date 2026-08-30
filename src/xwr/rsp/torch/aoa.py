"""Angle of Arrival Estimation and Point Cloud Module using Pytorch."""

import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, Int
from torch import Tensor

from xwr.rsp import aoa as base


class PointCloud(base.PointCloud[Tensor]):
    """Get radar point cloud from post FFT cube."""

    def _asarray(
        self, x: Float[np.ndarray, "..."], like: Float32[Tensor, "..."]
    ) -> Float[Tensor, "..."]:
        return torch.from_numpy(x).to(like.device)

    @staticmethod
    def _argmax_aoa(ang_sptr: Float32[Tensor, "... el az"]) -> Int[
        Tensor, "... 2"
    ]:
        el, az = ang_sptr.shape[-2:]
        idx = torch.argmax(ang_sptr.reshape(*ang_sptr.shape[:-2], el * az), -1)
        return torch.stack((idx // az, idx % az), dim=-1)

    def _point_cloud(
        self,
        cube: Float32[Tensor, "batch doppler el az range"],
        mask: Bool[Tensor, "batch range doppler"],
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float32[Tensor, "batch range doppler 4"],
    ]:
        el_angles = self._asarray(self.angle_table(cube.shape[2]), cube)
        az_angles = self._asarray(self.angle_table(cube.shape[3]), cube)

        _, r_size, d_size = mask.shape
        range_v = torch.arange(
            r_size, device=cube.device, dtype=cube.dtype) * self.range_res
        doppler_v = (
            torch.arange(d_size, device=cube.device, dtype=cube.dtype)
            - d_size // 2
        ) * self.doppler_res
        r_grid, d_grid = torch.meshgrid(range_v, doppler_v, indexing="ij")

        angle_idx = self.aoa(cube.permute(0, 4, 1, 2, 3))
        ang_e = el_angles[angle_idx[..., 0]]
        ang_a = az_angles[angle_idx[..., 1]]
        mask_e = torch.logical_and(ang_e < self.el_fov, ang_e > -self.el_fov)
        mask_a = torch.logical_and(ang_a < self.az_fov, ang_a > -self.az_fov)
        mask_ang = torch.logical_and(mask_a, mask_e)

        x = r_grid * torch.cos(-ang_a) * torch.cos(ang_e)
        y = r_grid * torch.sin(-ang_a) * torch.cos(ang_e)
        z = r_grid * torch.sin(ang_e)
        v = d_grid.broadcast_to(x.shape)

        pc_mask = torch.logical_and(mask, mask_ang)
        pc = torch.stack((x, y, z, v), dim=-1)

        return pc_mask, pc
