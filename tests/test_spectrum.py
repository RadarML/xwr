"""Detector (CFAR / CFAR CASO) tests."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from xwr.rsp import jax as rspj
from xwr.rsp import torch as rspt

# (batch, doppler, tx, rx, range) of the synthetic cubes used here.
SHAPE = (1, 16, 3, 4, 64)
TARGET = (30, 5)
"""Injected target, as a (range, doppler) bin index."""


def _target_cube(amplitude: float = 50.0, seed: int = 0) -> np.ndarray:
    """Low-amplitude noise floor with a single strong target."""
    rng = np.random.default_rng(seed)
    cube = rng.random(SHAPE).astype(np.float32) * 0.1 + 0.1
    cube[:, TARGET[1], :, :, TARGET[0]] = amplitude
    return cube


def _signal(cube: np.ndarray) -> np.ndarray:
    """Reference non-coherent integration: (batch, range, doppler)."""
    return (cube.astype(np.float64) ** 2).sum(axis=(2, 3)).transpose(0, 2, 1) + 1


def _ring_noise(signal: np.ndarray, guard, window) -> np.ndarray:
    """Reference ring-averaged noise floor, by explicit summation.

    Zero-padded at the edges and normalized by the number of in-bounds
    training cells, matching both backends' `valid` normalization.
    """
    (g0, g1), (w0, w1) = guard, window
    mask = np.ones((2 * w0 + 1, 2 * w1 + 1))
    mask[w0 - g0: w0 + g0 + 1, w1 - g1: w1 + g1 + 1] = 0.0

    s_r, s_d = signal.shape
    noise = np.zeros_like(signal)
    for i in range(s_r):
        for j in range(s_d):
            total, count = 0.0, 0.0
            for di in range(-w0, w0 + 1):
                for dj in range(-w1, w1 + 1):
                    if mask[di + w0, dj + w1] == 0:
                        continue
                    if 0 <= i + di < s_r and 0 <= j + dj < s_d:
                        total += signal[i + di, j + dj]
                        count += 1
            noise[i, j] = total / count
    return noise


def _numpy(*arrays):
    """Convert jax arrays or torch tensors to numpy."""
    return tuple(
        x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        for x in arrays)


def _run(backend, detector, cube: np.ndarray, **kwargs):
    """Run a detector on `cube` in the given backend, returning numpy."""
    module = {"jax": rspj, "torch": rspt}[backend]
    data = jnp.array(cube) if backend == "jax" else torch.from_numpy(cube)
    return _numpy(*getattr(module, detector)(**kwargs)(data))


BACKENDS = ["jax", "torch"]

CFAR_PARAMS = [
    {"guard": (2, 2), "window": (4, 4), "snr_thresh": 5.0,
     "discard_range": (10, 20)},
    {"guard": (1, 1), "window": (3, 2), "snr_thresh": 2.0,
     "discard_range": (4, 6)},
    # far=0: `signal[near:-far]` would slice to empty; regression for the
    # jax/torch divergence.
    {"guard": (2, 2), "window": (4, 4), "snr_thresh": 5.0,
     "discard_range": (4, 0)},
    {"guard": (0, 0), "window": (2, 2), "snr_thresh": 3.0,
     "discard_range": (0, 0)},
]

CASO_PARAMS = [
    {"train_window": (8, 4), "guard_window": (8, 0), "snr_thresh": (5.0, 3.0),
     "discard_range": (10, 20)},
    {"train_window": (4, 2), "guard_window": (2, 0), "snr_thresh": (2.0, 1.5),
     "discard_range": (4, 6)},
    {"train_window": (4, 2), "guard_window": (2, 0), "snr_thresh": (5.0, 3.0),
     "discard_range": (4, 0)},
    {"train_window": (4, 2), "guard_window": (2, 0), "snr_thresh": (5.0, 3.0),
     "discard_range": (0, 0)},
]


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", CFAR_PARAMS)
def test_cfar_parity(kwargs):
    """Jax and torch CFAR must agree; they mirror the same algorithm."""
    cube = _target_cube()
    mask_j, signal_j, snr_j = _run("jax", "CFAR", cube, **kwargs)
    mask_t, signal_t, snr_t = _run("torch", "CFAR", cube, **kwargs)

    assert np.array_equal(mask_j, mask_t)
    assert np.allclose(signal_j, signal_t, atol=1e-4)
    assert np.allclose(snr_j, snr_t, rtol=1e-4)


@pytest.mark.parametrize("kwargs", CASO_PARAMS)
def test_caso_parity(kwargs):
    """Jax and torch CFAR CASO must agree."""
    cube = _target_cube()
    mask_j, signal_j, snr_j = _run("jax", "CFARCASO", cube, **kwargs)
    mask_t, signal_t, snr_t = _run("torch", "CFARCASO", cube, **kwargs)

    assert np.array_equal(mask_j, mask_t)
    assert np.allclose(signal_j, signal_t, atol=1e-4)
    assert np.allclose(snr_j, snr_t, rtol=1e-4)


# ---------------------------------------------------------------------------
# Detection correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("detector", ["CFAR", "CFARCASO"])
def test_detects_only_the_target(backend, detector):
    """An isolated strong target is the one and only detection."""
    cube = _target_cube()
    kwargs = (
        {"guard": (2, 2), "window": (4, 4), "snr_thresh": 5.0,
         "discard_range": (10, 20)}
        if detector == "CFAR" else
        {"train_window": (4, 2), "guard_window": (2, 0),
         "snr_thresh": (5.0, 3.0), "discard_range": (10, 20)})

    mask, _, snr = _run(backend, detector, cube, **kwargs)

    assert mask[0, TARGET[0], TARGET[1]]
    # One detection per batch element, and nowhere else.
    assert np.array_equal(
        np.argwhere(mask), [[0, TARGET[0], TARGET[1]]])
    assert snr[0, TARGET[0], TARGET[1]] > 5.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("detector", ["CFAR", "CFARCASO"])
def test_integrated_signal(backend, detector):
    """`signal` is the non-coherent sum over the virtual array, plus one."""
    cube = _target_cube()
    kwargs = (
        {"discard_range": (10, 20)} if detector == "CFAR" else
        {"train_window": (4, 2), "guard_window": (2, 0),
         "discard_range": (10, 20)})

    _, signal, _ = _run(backend, detector, cube, **kwargs)

    assert signal.shape == (SHAPE[0], SHAPE[4], SHAPE[1])
    assert np.allclose(signal, _signal(cube), rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("detector", ["CFAR", "CFARCASO"])
def test_discard_band(backend, detector):
    """Discarded range bins never detect, and are assigned unit noise."""
    near, far = 10, 20
    cube = _target_cube()
    # Put a second target inside the discarded band; it must not fire.
    cube[:, TARGET[1], :, :, near - 1] = 50.0
    kwargs = (
        {"discard_range": (near, far)} if detector == "CFAR" else
        {"train_window": (4, 2), "guard_window": (2, 0),
         "discard_range": (near, far)})

    mask, signal, snr = _run(backend, detector, cube, **kwargs)
    s_r = signal.shape[1]

    assert not mask[:, :near].any()
    assert not mask[:, s_r - far:].any()
    # noise == 1 outside the band, so snr is the raw integrated signal.
    assert np.allclose(snr[:, :near], signal[:, :near], rtol=1e-5)
    assert np.allclose(snr[:, s_r - far:], signal[:, s_r - far:], rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cfar_noise_floor(backend):
    """The CFAR ring noise floor matches an explicit numpy reference."""
    guard, window = (1, 1), (2, 2)
    cube = _target_cube()
    # No discarded band, so every bin exercises the ring average.
    _, signal, snr = _run(
        backend, "CFAR", cube, guard=guard, window=window,
        discard_range=(0, 0))

    expected = _ring_noise(_signal(cube)[0], guard, window)
    assert np.allclose(signal[0] / snr[0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs", [
    {"guard": (4, 4), "window": (2, 2)},        # guard exceeds window
    {"guard": (2, 2), "window": (2, 2)},        # no training cells left
    {"discard_range": (1, 2, 3)},               # wrong length
])
def test_cfar_invalid(backend, kwargs):
    """Bad CFAR parameters are rejected at construction."""
    module = {"jax": rspj, "torch": rspt}[backend]
    with pytest.raises(ValueError):
        module.CFAR(**kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs", [
    {"train_window": (8, 4, 2)},
    {"guard_window": (8, 0, 1)},
    {"discard_range": (1, 2, 3)},
    {"snr_thresh": (5.0, 3.0, 1.0)},
])
def test_caso_invalid(backend, kwargs):
    """Bad CFAR CASO parameters are rejected at construction."""
    module = {"jax": rspj, "torch": rspt}[backend]
    with pytest.raises(ValueError):
        module.CFARCASO(**kwargs)
