"""RSP Tests."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from xwr import rsp
from xwr.rsp import jax as rspj
from xwr.rsp import numpy as rspn
from xwr.rsp import torch as rspt


def _iq_complex(shape):
    rng = np.random.default_rng()
    return (
        rng.random(size=shape) + 1j * rng.random(size=shape)
    ).astype(np.complex64)


def _iiqq_int16(shape):
    rng = np.random.default_rng()
    return rng.integers(-32768, 32767, shape, dtype=np.int16)


SHAPE = {
    "AWR1843AOP": (2, 4, 3, 4, 8),
    "AWR1843Boost": (2, 4, 3, 4, 8),
    "AWR1642Boost": (2, 4, 2, 4, 8)
}

PARAMS = [
    (False, {}, "AWR1843AOP"),
    (False, {}, "AWR1843Boost"),
    (False, {}, "AWR1642Boost"),
    (True, {}, "AWR1843Boost"),
    (True, {"azimuth": 16}, "AWR1843Boost"),
    (True, {"doppler": 8}, "AWR1843Boost"),
    (True, {"range": 16}, "AWR1843Boost"),
    (True, {"elevation": 4}, "AWR1843Boost"),
    ({"range": True, "doppler": True}, {}, "AWR1843Boost")
]


def test_numpy_basic():
    """Basic tests for numpy backend."""
    data = _iq_complex((2, 4, 3, 4, 8))
    assert np.allclose(rsp.iq_from_iiqq(data), data)

    data = _iiqq_int16((2, 4, 3, 4, 8 * 2))
    assert rsp.iqiq_from_iiqq(data).shape == (2, 4, 3, 4, 8, 2)
    assert rsp.iq_from_iiqq(data).shape == (2, 4, 3, 4, 8)

    data = _iq_complex((2, 4, 3, 4, 8))

    awr1843boost = rspn.AWR1843Boost(window=False, size={})
    assert awr1843boost(data).shape == (2, 4, 2, 8, 8)

    with pytest.raises(ValueError):
        _ = rspn.AWR1843Boost(window=False, size={"azimuth": 4})(data)

    awr1843aop = rspn.AWR1843AOP(window=False, size={})
    assert awr1843aop(data).shape == (2, 4, 4, 3, 8)

    awr1642 = rspn.AWR1642Boost(window=False, size={})
    assert awr1642(data[:, :, :2]).shape == (2, 4, 1, 8, 8)


@pytest.mark.parametrize("window,size,radar", PARAMS)
def test_jax(window, size, radar):
    """Test jax vs numpy RSP."""
    shape = SHAPE[radar]
    rsp_numpy = getattr(rspn, radar)(window=window, size=size)
    rsp_jax = getattr(rspj, radar)(window=window, size=size)

    data = _iq_complex(shape)
    numpy_result = rsp_numpy(data)
    jax_result = rsp_jax(jnp.array(data))

    assert np.allclose(numpy_result, np.array(jax_result), atol=1e-4)

    with pytest.raises(ValueError):
        rsp_numpy(data[:, :, :-1, :, :])
    with pytest.raises(ValueError):
        rsp_jax(jnp.array(data[:, :, :-1, :, :]))
    with pytest.raises(ValueError):
        rsp_numpy(data[:, :, :, :-1, :])
    with pytest.raises(ValueError):
        rsp_jax(jnp.array(data[:, :, :, :-1, :]))


@pytest.mark.parametrize("window,size,radar", PARAMS)
def test_torch(window, size, radar):
    """Test jax vs pytorch RSP."""
    shape = SHAPE[radar]

    rsp_numpy = getattr(rspn, radar)(window=window, size=size)
    rsp_torch = getattr(rspt, radar)(window=window, size=size)

    data = _iq_complex(shape)
    numpy_result = rsp_numpy(data)
    torch_result = rsp_torch(torch.from_numpy(data))

    assert np.allclose(numpy_result, torch_result.numpy(), atol=1e-4)

    with pytest.raises(ValueError):
        rsp_torch(torch.from_numpy(data)[:, :, :-1, :, :])
    with pytest.raises(ValueError):
        rsp_torch(torch.from_numpy(data)[:, :, :, :-1, :])
