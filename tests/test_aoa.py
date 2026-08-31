"""Angle of arrival / point cloud geometry tests."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from xwr import rsp
from xwr.config import XWRConfig
from xwr.radar import AWR1843
from xwr.rsp import jax as rspj
from xwr.rsp import numpy as rspn
from xwr.rsp import torch as rspt

BACKENDS = ["jax", "torch", "numpy"]

# Odd angle sizes, so `linspace(-1, 1, n)` has an exact 0 at the center bin
# and boresight is exactly representable.
EL, AZ = 9, 9
CENTER = 4
BATCH, DOPPLER, RANGE = 1, 4, 5


@pytest.fixture
def config():
    """A valid AWR1843 configuration."""
    return XWRConfig(
        device=AWR1843,
        frequency=77.0,
        idle_time=200.0,
        adc_start_time=5.7,
        ramp_end_time=59.0,
        tx_start_time=1.0,
        freq_slope=67.012,
        adc_samples=128,
        sample_rate=2500,
        frame_length=128,
        frame_period=125.0,
    )


def _res(config: XWRConfig) -> tuple[float, float]:
    """`PointCloud` range/doppler resolution args for a configuration."""
    return config.range_resolution, config.doppler_resolution


def _peak_cube(el: int, az: int) -> np.ndarray:
    """Cube whose angle spectrum peaks at `(el, az)` for every bin."""
    cube = np.zeros((BATCH, DOPPLER, EL, AZ, RANGE), dtype=np.float32)
    cube[:, :, el, az, :] = 1.0
    return cube


def _point_cloud(backend, config, cube, mask=None, **kwargs):
    """Run `PointCloud` in the given backend, returning numpy arrays."""
    if mask is None:
        mask = np.ones((BATCH, RANGE, DOPPLER), dtype=bool)

    if backend == "jax":
        pc = rspj.PointCloud(*_res(config), **kwargs)
        pc_mask, points = pc(jnp.array(cube), jnp.array(mask))
        return np.asarray(pc_mask), np.asarray(points)
    elif backend == "torch":
        pc = rspt.PointCloud(*_res(config), **kwargs)
        pc_mask, points = pc(torch.from_numpy(cube), torch.from_numpy(mask))
        return pc_mask.numpy(), points.numpy()

    pc = rspn.PointCloud(*_res(config), **kwargs)
    return pc(cube, mask)


def _angles(backend, config, **kwargs):
    """Get the (elevation, azimuth) lookup tables as numpy arrays."""
    module = {"jax": rspj, "torch": rspt, "numpy": rspn}[backend]
    pc = module.PointCloud(*_res(config), **kwargs)
    return pc._angle_table(EL), pc._angle_table(AZ)



# ---------------------------------------------------------------------------
# Geometry conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_boresight(backend, config):
    """A boresight peak lies on the +x axis: y and z are both zero."""
    _, points = _point_cloud(
        backend, config, _peak_cube(CENTER, CENTER),
        angle_fov=(90.0, 90.0))

    x, y, z, _ = np.moveaxis(points, -1, 0)
    assert np.allclose(y, 0.0, atol=1e-6)
    assert np.allclose(z, 0.0, atol=1e-6)
    # x is the range itself, broadcast over doppler.
    expected = np.arange(RANGE) * config.range_resolution
    assert np.allclose(x[0], expected[:, None], atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_azimuth_is_negated(backend, config):
    """A positive azimuth bin maps to negative y; the angle is negated."""
    el_angles, az_angles = _angles(backend, config, angle_fov=(90.0, 90.0))
    assert az_angles[CENTER + 2] > 0

    _, points = _point_cloud(
        backend, config, _peak_cube(CENTER, CENTER + 2),
        angle_fov=(90.0, 90.0))

    x, y, z, _ = np.moveaxis(points, -1, 0)
    # Nonzero range bins only; bin 0 sits at the origin.
    assert (y[0, 1:] < 0).all()
    assert (x[0, 1:] > 0).all()
    assert np.allclose(z, 0.0, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_elevation_is_positive_z(backend, config):
    """A positive elevation bin maps to positive z."""
    _, points = _point_cloud(
        backend, config, _peak_cube(CENTER + 2, CENTER),
        angle_fov=(90.0, 90.0))

    x, y, z, _ = np.moveaxis(points, -1, 0)
    assert (z[0, 1:] > 0).all()
    assert (x[0, 1:] > 0).all()
    assert np.allclose(y, 0.0, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("el,az", [(CENTER, CENTER), (CENTER, CENTER + 2),
                                   (CENTER + 2, CENTER), (CENTER - 3, 1)])
def test_range_and_velocity(backend, config, el, az):
    """Position norm is the range, and the last channel is signed velocity."""
    _, points = _point_cloud(
        backend, config, _peak_cube(el, az), angle_fov=(90.0, 90.0))

    x, y, z, v = np.moveaxis(points, -1, 0)

    expected_range = np.arange(RANGE) * config.range_resolution
    assert np.allclose(
        np.sqrt(x**2 + y**2 + z**2)[0], expected_range[:, None], atol=1e-6)

    expected_v = (
        np.arange(DOPPLER) - DOPPLER // 2) * config.doppler_resolution
    assert np.allclose(v[0], expected_v[None, :], atol=1e-6)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_angle_fov(backend, config):
    """Peaks outside the field of view are excluded from the mask."""
    fov = {"angle_fov": (5.0, 5.0)}

    kept, _ = _point_cloud(backend, config, _peak_cube(CENTER, CENTER), **fov)
    assert kept.all()

    for cube in (_peak_cube(CENTER, CENTER + 2),
                 _peak_cube(CENTER + 2, CENTER)):
        rejected, _ = _point_cloud(backend, config, cube, **fov)
        assert not rejected.any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_detection_mask_is_combined(backend, config):
    """The output mask is the detection mask AND the angular bounds."""
    rng = np.random.default_rng(0)
    mask = rng.random((BATCH, RANGE, DOPPLER)) > 0.5

    # In-bounds angle: the detection mask passes through unchanged.
    kept, _ = _point_cloud(
        backend, config, _peak_cube(CENTER, CENTER), mask=mask,
        angle_fov=(90.0, 90.0))
    assert np.array_equal(kept, mask)

    # Out-of-bounds angle: nothing survives, whatever the detection mask.
    rejected, _ = _point_cloud(
        backend, config, _peak_cube(CENTER, CENTER + 2), mask=mask,
        angle_fov=(5.0, 5.0))
    assert not rejected.any()


# ---------------------------------------------------------------------------
# Antenna spacing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spacing", [0.5, 0.4, 0.25])
def test_antenna_spacing_is_clamped(backend, config, spacing):
    """Sub-half-wavelength spacing clamps to ±90° instead of yielding nan."""
    el_angles, az_angles = _angles(
        backend, config, antenna_spacing=spacing)

    for angles in (el_angles, az_angles):
        assert not np.isnan(angles).any()
        assert np.isclose(angles[0], -np.pi / 2)
        assert np.isclose(angles[-1], np.pi / 2)
        # Still monotonic, so bin order is preserved.
        assert (np.diff(angles) >= 0).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_antenna_spacing_must_be_positive(backend, config):
    """A non-positive antenna spacing is rejected at construction."""
    module = {"jax": rspj, "torch": rspt, "numpy": rspn}[backend]
    for spacing in (0.0, -0.5):
        with pytest.raises(ValueError):
            module.PointCloud(*_res(config), antenna_spacing=spacing)


# ---------------------------------------------------------------------------
# Varying angle size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("el,az", [(EL, AZ), (EL, AZ + 8), (EL + 8, AZ)])
def test_varying_angle_size(backend, config, el, az):
    """One instance handles cubes of differing angle sizes.

    The bin-to-angle tables come from the cube's own angle axes, so nothing
    has to be declared up front and nothing can disagree with the cube.
    """
    cube = np.zeros((BATCH, DOPPLER, el, az, RANGE), dtype=np.float32)
    cube[:, :, el // 2, az // 2, :] = 1.0

    pc_mask, points = _point_cloud(backend, config, cube)

    assert points.shape == (BATCH, RANGE, DOPPLER, 4)
    assert pc_mask.shape == (BATCH, RANGE, DOPPLER)


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spacing", [0.5, 0.4])
def test_angle_table_parity(config, spacing):
    """All backends build identical bin-to-angle lookup tables."""
    angles = {b: _angles(b, config, antenna_spacing=spacing) for b in BACKENDS}
    el0, az0 = angles[BACKENDS[0]]
    for backend in BACKENDS[1:]:
        el, az = angles[backend]
        assert np.allclose(el0, el, atol=1e-6)
        assert np.allclose(az0, az, atol=1e-6)


def test_point_cloud_parity(config):
    """All backends produce the same point cloud from the same cube."""
    rng = np.random.default_rng(0)
    cube = rng.random(
        (BATCH, DOPPLER, EL, AZ, RANGE)).astype(np.float32)
    mask = rng.random((BATCH, RANGE, DOPPLER)) > 0.5

    results = {
        b: _point_cloud(b, config, cube, mask=mask) for b in BACKENDS}
    mask0, points0 = results[BACKENDS[0]]
    for backend in BACKENDS[1:]:
        mask_b, points_b = results[backend]
        assert np.array_equal(mask0, mask_b)
        assert np.allclose(points0, points_b, atol=1e-4)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detector", ["CFAR", "CFARCASO"])
def test_end_to_end_parity(config, detector):
    """RSP -> detector -> point cloud agrees across backends."""
    rng = np.random.default_rng(1)
    shape = (2, 16, 3, 4, 32)
    iq = (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex64)

    kwargs = (
        {"guard": (2, 2), "train": (2, 2), "snr_thresh": 1.2,
         "discard_range": (2, 2)}
        if detector == "CFAR" else
        {"train": (4, 2), "guard": (2, 0),
         "snr_thresh": (1.2, 1.1), "discard_range": (2, 2)})

    cube_j = jnp.abs(rspj.AWR1843Boost(window=False, size={})(jnp.array(iq)))
    cube_t = torch.abs(
        rspt.AWR1843Boost(window=False, size={})(torch.from_numpy(iq)))
    cube_n = np.abs(rspn.AWR1843Boost(window=False, size={})(iq))
    _, _, el, az, _ = cube_j.shape

    mask_j = getattr(rspj, detector)(**kwargs)(cube_j).mask
    mask_t = getattr(rspt, detector)(**kwargs)(cube_t).mask
    mask_n = getattr(rspn, detector)(**kwargs)(cube_n).mask
    assert np.array_equal(np.asarray(mask_j), mask_t.numpy())
    assert np.array_equal(np.asarray(mask_j), mask_n)
    assert np.asarray(mask_j).any(), "no detections; test would be vacuous"

    pc_mask_j, points_j = rspj.PointCloud(
        *_res(config))(cube_j, mask_j)
    pc_mask_t, points_t = rspt.PointCloud(
        *_res(config))(cube_t, mask_t)
    pc_mask_n, points_n = rspn.PointCloud(
        *_res(config))(cube_n, mask_n)

    assert np.array_equal(np.asarray(pc_mask_j), pc_mask_t.numpy())
    assert np.array_equal(np.asarray(pc_mask_j), pc_mask_n)
    assert np.allclose(np.asarray(points_j), points_t.numpy(), atol=1e-4)
    assert np.allclose(np.asarray(points_j), points_n, atol=1e-4)


# ---------------------------------------------------------------------------
# Shared base classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_share_base_classes(backend):
    """Every backend implements the backend-agnostic interfaces.

    This is the invariant that keeps the three backends from drifting: the
    docstrings, constructor arguments, and validation live once in
    `xwr.rsp`, and a backend which stopped subclassing them could silently
    grow a different signature again.
    """
    module = {"jax": rspj, "torch": rspt, "numpy": rspn}[backend]

    assert issubclass(module.PointCloud, rsp.PointCloud)
    assert issubclass(module.CFAR, rsp.CFAR)
    assert issubclass(module.CFARCASO, rsp.CFARCASO)
    assert issubclass(module.CFAR, rsp.Detector)
    assert issubclass(module.CFARCASO, rsp.Detector)
