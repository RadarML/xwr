"""Angle of Arrival Estimation and Point Cloud Module using Pytorch."""

from collections.abc import Sequence

import torch
from jaxtyping import Bool, Float32, Int
from torch import Tensor

from xwr.config import XWRConfig


class PointCloud:
    """Get radar point cloud from post FFT cube.

    To convert azimuth-elevation bin indices to azimuth-elevation angles,
    we use the property that the azimuth bin indices correspond to the sin of
    the angle
    ```
    angles = torch.arcsin(
        torch.linspace(-torch.pi, torch.pi, bin_size)
        / (2 * torch.pi * antenna_spacing)
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
        config: radar configuration; see [`XWRConfig`][xwr.config.]. Only the
            derived `range_resolution` (meters per range bin) and
            `doppler_resolution` (meters/second per doppler bin) are used.

            - Range bins are mapped to meters by `bin * range_resolution`, so
              bin `0` is zero range.
            - Doppler bins are mapped to *signed* radial velocity by
              `(bin - doppler // 2) * doppler_resolution`, so the middle bin
              is zero velocity. This assumes the doppler axis has already
              been `fftshift`ed, which [`doppler_range`][xwr.rsp.RSP.] does.
        angle_fov: angle field of view **in degrees** for (elevation,
            azimuth), applied as a symmetric `±fov` bound. Points whose
            estimated angle falls outside the bound are excluded from the
            returned mask. This rejects estimates near the edge of the array's
            sin-space, where a sparse MIMO array has little real resolving
            power and grating lobes appear.
        angle_size: angle fft size for (elevation, azimuth). **Must match the
            `size={"elevation": ..., "azimuth": ...}` the
            [`RSP`][xwr.rsp.] was constructed with**: the bin-to-angle lookup
            tables are built at this length, then indexed by the argmax over
            the cube's angle axes. A mismatch silently yields wrong angles
            rather than an error.
        antenna_spacing: antenna spacing in terms of wavelength (default 0.5
            for a half-wavelength grid). Sets the sin-space to angle mapping;
            when the chirp center frequency differs from the antenna design
            frequency, it must be corrected using the formula above. A wrong
            value gives systematically wrong angles, not an error.
    """

    def __init__(
        self,
        config: XWRConfig,
        angle_fov: Sequence[float] = (20.0, 80.0),
        angle_size: Sequence[int] = (128, 128),
        antenna_spacing: float = 0.5,
    ) -> None:
        self.range_res = config.range_resolution
        self.doppler_res = config.doppler_resolution

        assert len(angle_fov) == 2 and len(angle_size) == 2, (
            "angle_fov and angle_size must be a sequence of length 2."
        )
        if antenna_spacing <= 0:
            raise ValueError("antenna_spacing must be > 0")
        self.el_fov = torch.deg2rad(torch.tensor(angle_fov[0]))
        self.az_fov = torch.deg2rad(torch.tensor(angle_fov[1]))
        el_sin = (
            torch.linspace(-1.0, 1.0, angle_size[0]) / (2 * antenna_spacing)
        ).clamp(-1.0, 1.0)
        az_sin = (
            torch.linspace(-1.0, 1.0, angle_size[1]) / (2 * antenna_spacing)
        ).clamp(-1.0, 1.0)
        self.el_angles = torch.arcsin(el_sin)
        self.az_angles = torch.arcsin(az_sin)

    def _to(self, device: torch.device) -> None:
        """Move the bin-to-angle lookup tables to `device`, if needed."""
        if self.el_angles.device != device:
            self.el_fov = self.el_fov.to(device)
            self.az_fov = self.az_fov.to(device)
            self.el_angles = self.el_angles.to(device)
            self.az_angles = self.az_angles.to(device)

    @staticmethod
    def _argmax_aoa(ang_sptr: Float32[Tensor, "... el az"]) -> Int[
        Tensor, "... 2"
    ]:
        """Get the (elevation, azimuth) index of the spectrum peak."""
        el, az = ang_sptr.shape[-2:]
        idx = torch.argmax(ang_sptr.reshape(*ang_sptr.shape[:-2], el * az), -1)
        return torch.stack((idx // az, idx % az), dim=-1)

    def aoa(
        self, cube: Float32[Tensor, "batch range doppler el az"]
    ) -> Int[Tensor, "batch range doppler 2"]:
        """Angle of arrival estimation.

        Takes the argmax over the (elevation, azimuth) angle spectrum of each
        range-doppler bin, yielding **bin indices** into the `el_angles` and
        `az_angles` lookup tables rather than angles.

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
        return self._argmax_aoa(cube)

    def __call__(
        self,
        cube: Float32[Tensor, "batch doppler el az range"],
        mask: Bool[Tensor, "batch range doppler"],
    ) -> tuple[
        Bool[Tensor, "batch range doppler"],
        Float32[Tensor, "batch range doppler 4"],
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
                in meters/second. Position is computed as
                `x = r cos(-az) cos(el)`, `y = r sin(-az) cos(el)`, and
                `z = r sin(el)`; note that the azimuth angle is **negated**,
                and that `x` is the boresight direction at zero angle.
        """
        self._to(cube.device)

        _, r_size, d_size = mask.shape
        range_v = torch.arange(
            r_size, device=cube.device, dtype=cube.dtype) * self.range_res
        doppler_v = (
            torch.arange(d_size, device=cube.device, dtype=cube.dtype)
            - d_size // 2
        ) * self.doppler_res
        r_grid, d_grid = torch.meshgrid(range_v, doppler_v, indexing="ij")

        angle_idx = self.aoa(cube.permute(0, 4, 1, 2, 3))
        ang_e = self.el_angles[angle_idx[..., 0]]
        ang_a = self.az_angles[angle_idx[..., 1]]
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
