"""Visualize Point Cloud from Dataset using Rerun."""

import os
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import tyro
import yaml
from abstract_dataloader import generic
from PIL import Image
from roverd import Dataset, sensors
from tqdm import tqdm

from xwr.config import XWRConfig
from xwr.rsp import RSP, iq_from_iiqq

Backend = Literal["torch", "jax", "numpy"]


def _torch_backend(
    rsp_args: dict, radar_cfg: XWRConfig, device: str | None
) -> tuple[Any, Any, Any, str]:
    """Set up the torch RSP pipeline.

    Returns:
        The per-frame signal processing function, a raw IIQQ to device-array
            converter, an array to numpy converter, and the device name.
    """
    import torch

    from xwr.rsp.torch import CFARCASO, AWR1843Boost, PointCloud

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    rsp: RSP = AWR1843Boost(**rsp_args)
    cfar = CFARCASO()
    radar_pc = PointCloud(
        radar_cfg.range_resolution, radar_cfg.doppler_resolution)

    @torch.no_grad()
    def sig_process(iq):
        cube_rd = rsp.doppler_range(iq)
        cube = rsp.elevation_azimuth(cube_rd)

        detection = cfar(torch.abs(cube_rd))
        points = radar_pc(torch.abs(cube), detection.mask)

        return detection, cube_rd, points

    def to_input(iiqq):
        # Move to the device before un-interleaving, so the int16 -> complex64
        # conversion happens there as well.
        return iq_from_iiqq(torch.from_numpy(iiqq).to(dev))

    return sig_process, to_input, lambda x: x.cpu().numpy(), str(dev)


def _jax_backend(
    rsp_args: dict, radar_cfg: XWRConfig, device: str | None
) -> tuple[Any, Any, Any, str]:
    """Set up the jax RSP pipeline; `device` is ignored.

    Returns:
        The per-frame signal processing function, a raw IIQQ to device-array
            converter, an array to numpy converter, and the device name.
    """
    import jax
    import jax.numpy as jnp

    from xwr.rsp.jax import CFARCASO, AWR1843Boost, PointCloud

    rsp: RSP = AWR1843Boost(**rsp_args)
    cfar = CFARCASO()
    radar_pc = PointCloud(
        radar_cfg.range_resolution, radar_cfg.doppler_resolution)

    @jax.jit
    def sig_process(iq):
        cube_rd = rsp.doppler_range(iq)
        cube = rsp.elevation_azimuth(cube_rd)

        detection = cfar(jnp.abs(cube_rd))
        points = radar_pc(jnp.abs(cube), detection.mask)

        return detection, cube_rd, points

    def to_input(iiqq):
        return jnp.asarray(iq_from_iiqq(iiqq))

    return sig_process, to_input, np.asarray, str(jax.devices()[0])


def _numpy_backend(
    rsp_args: dict, radar_cfg: XWRConfig, device: str | None
) -> tuple[Any, Any, Any, str]:
    """Set up the numpy RSP pipeline; `device` is ignored.

    Returns:
        The per-frame signal processing function, a raw IIQQ to device-array
            converter, an array to numpy converter, and the device name.
    """
    from xwr.rsp.numpy import CFARCASO, AWR1843Boost, PointCloud

    rsp: RSP = AWR1843Boost(**rsp_args)
    cfar = CFARCASO()
    radar_pc = PointCloud(
        radar_cfg.range_resolution, radar_cfg.doppler_resolution)

    def sig_process(iq):
        cube_rd = rsp.doppler_range(iq)
        cube = rsp.elevation_azimuth(cube_rd)

        detection = cfar(np.abs(cube_rd))
        points = radar_pc(np.abs(cube), detection.mask)

        return detection, cube_rd, points

    def to_input(iiqq):
        return iq_from_iiqq(iiqq)

    return sig_process, to_input, lambda x: x, "cpu"


def main(
    path: str, /,
    trace: str = "bike/bloomfield.back",
    backend: Backend = "torch",
    gain: float = 5e-6,
    azimuth_size: int = 128,
    elevation_size: int = 128,
    save: str | None = None,
    show: bool = False,
    config: str | None = None,
    frames: int | None = 6000,
    start: int = 0,
    device: str | None = None,
):
    """Visualize Point Cloud from Dataset.

    Args:
        path: base folder of the dataset.
        trace: trace name to visualize.
        backend: RSP backend to run the pipeline on. All three produce the
            same point cloud; see [`xwr.rsp.torch`][xwr.rsp.torch],
            [`xwr.rsp.jax`][xwr.rsp.jax], and [`xwr.rsp.numpy`][xwr.rsp.numpy].
        gain: a fixed value to normalize radar spectrum for visualization
        azimuth_size: azimuth fft size.
        elevation_size: elevation fft size.
        save: rerun `.rrd` log file to write; defaults to the trace name
            with `/` replaced by `_` and the backend appended, in the current
            directory.
        show: also stream to a rerun viewer window, in addition to writing
            the log file.
        config: path to the radar configuration file; defaults to the
            `config.yaml` in the trace directory.
        frames: number of radar frames to render; `None` renders the whole
            trace. The default of `6000` is ~5 minutes at 20Hz.
        start: index of the first radar frame to render.
        device: `torch` device to run the RSP on; defaults to `cuda` when
            available, and `cpu` otherwise. Ignored by the `jax` backend,
            which uses whichever device jax selects, and by the `numpy`
            backend, which always runs on `cpu`.

    """
    traces = [os.path.join(path, trace)]
    dataset = Dataset.from_config(
        traces,
        sync=generic.Nearest("radar"),
        sensors={"radar": sensors.XWRRadar, "camera": sensors.Camera},
    )

    if config is None:
        config = os.path.join(path, trace, "config.yaml")
    with open(config) as f:
        radar_args = yaml.safe_load(f)["radar"]["args"]["radar"]
    # `num_tx` is derived from `device` in XWRConfig, so it is dropped here.
    radar_args.pop("num_tx", None)
    radar_cfg = XWRConfig(device="AWR1843", **radar_args)

    rsp_args = {
        "size": {"azimuth": azimuth_size, "elevation": elevation_size},
        "window": {"range": True, "doppler": True},
    }
    setup = {
        "torch": _torch_backend, "jax": _jax_backend, "numpy": _numpy_backend,
    }[backend]
    sig_process, to_input, to_numpy, dev_name = setup(
        rsp_args, radar_cfg, device)
    print(f"Running the {backend} RSP on {dev_name}.")

    if save is None:
        save = f"{trace.replace('/', '_')}_{backend}.rrd"

    cmap = plt.get_cmap("hot")
    rr.init("radar_vis")
    if show:
        # Launch a viewer, and attach it alongside the file sink.
        rr.spawn(connect=False)
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(save))
    else:
        rr.save(save)

    n = len(dataset)  # type: ignore
    stop = n if frames is None else min(start + frames, n)
    for i in tqdm(range(start, stop)):
        data = dataset[i]
        img = data["camera"].image.squeeze()
        t_radar = data["radar"].timestamps[0, 0]
        t_cam = data["camera"].timestamps[0, 0]

        iq = to_input(data["radar"].iq.squeeze(1))

        detection, rd, points = sig_process(iq)
        pc = to_numpy(points.points[points.mask])
        rd_mask = to_numpy(detection.mask)[0]
        rd = to_numpy(rd)[0]

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
        rr.log("camera", rr.Image(cam).compress(jpeg_quality=75))


if __name__ == "__main__":
    tyro.cli(main)
