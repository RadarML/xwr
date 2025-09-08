"""Visualize Point Cloud from Dataset using Rerun."""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import tyro
from abstract_dataloader import generic
from PIL import Image
from roverd import Dataset, sensors
from roverd.sensors.radar import RadarMetadata
from tqdm import tqdm

from xwr.rsp import RSP, iq_from_iiqq
from xwr.rsp.jax import CFARCASO, AWR1843Boost, PointCloud


def main(
    path: str, /,
    trace: str = "bike/bloomfield.back",
    gain: float = 5e-6,
    azimuth_size: int = 128,
    elevation_size: int = 128,
    grpc_port: int = 4195,
    save: str | None = None,
):
    """Visualize Point Cloud from Dataset.

    Args:
        path: base folder of the dataset.
        trace: trace name to visualize.
        gain: a fixed value to normalize radar spectrum for visualization
        azimuth_size: azimuth fft size.
        elevation_size: elevation fft size.
        grpc_port: rerun grpc port number.
        save: rerun log file name.

    """
    traces = [os.path.join(path, trace)]
    dataset = Dataset.from_config(
        traces,
        sync=generic.Nearest("radar"),
        sensors={"radar": sensors.XWRRadar, "camera": sensors.Camera},
    )
    radar_cfg: RadarMetadata = dataset.traces[0].sensors["radar"].metadata

    rsp: RSP = AWR1843Boost(
        size={"azimuth": azimuth_size, "elevation": elevation_size},
        window={"range": True, "doppler": True},
    )
    cfar = CFARCASO()
    radar_pc = PointCloud(
        radar_cfg.range_resolution[0].item(),
        radar_cfg.doppler_resolution[0].item(),
    )

    @jax.jit
    def sig_process(iq):
        cube_rd = rsp.doppler_range(iq)
        cube = rsp.elevation_azimuth(cube_rd)

        rd_mask, sig, snr = cfar.__call__(jnp.abs(cube_rd.squeeze()))
        pc_mask, pc = radar_pc.__call__(jnp.abs(cube.squeeze()), rd_mask)

        return rd_mask, cube_rd.squeeze(), pc_mask, pc

    cmap = plt.get_cmap("hot")
    rr.init("radar_vis")
    rr.serve_grpc(grpc_port=grpc_port, server_memory_limit="50%")

    if save is not None:
        rr.save(save)

    pbar = tqdm(dataset)  # type: ignore
    for data in pbar:
        iq = iq_from_iiqq(data["radar"].iq.squeeze(1))
        img = data["camera"].image.squeeze()
        t_radar = data["radar"].timestamps[0, 0]
        t_cam = data["camera"].timestamps[0, 0]

        iq = jnp.asarray(iq)

        rd_mask, rd, pc_mask, pc = sig_process(iq)
        pc = np.asarray(pc)[pc_mask]

        D, nrx, ntd, R = rd.shape
        rd = np.transpose(rd, (3, 0, 1, 2))
        rd = np.flip(
            np.clip(np.mean(np.abs(rd).reshape(R, D, -1), -1) * gain, 0, 1), 0
        )
        rd_img = Image.fromarray((cmap(rd)[..., :3] * 255).astype(np.uint8))
        obj_mask = np.flip(rd_mask, 0)
        cfar_img = cmap(rd)[..., :3]
        cfar_img[obj_mask] = [0, 0.99, 0]
        cfar_img = Image.fromarray((cfar_img * 255).astype(np.uint8))

        rr.set_time("time", timestamp=t_radar)
        rr.log("range_doppler", rr.Image(rd_img))
        rr.log("cfar", rr.Image(cfar_img))
        rr.log("pc", rr.Points3D(np.asarray(pc[:, :3])))

        rr.set_time("time", timestamp=t_cam)
        cam = Image.fromarray(img).resize(
            (img.shape[1] // 2, img.shape[0] // 2)
        )
        rr.log("camera", rr.Image(cam))


if __name__ == "__main__":
    tyro.cli(main)
