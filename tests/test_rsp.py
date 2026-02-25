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
    "AWR1642Boost": (2, 4, 2, 4, 8),
    "AWR2944EVM": (2, 4, 4, 4, 16),
    "AWRL6844EVM": (2, 4, 4, 4, 16),
}


def _i_float32(shape):
    rng = np.random.default_rng()
    return rng.random(size=shape).astype(np.float32)


AWR2944_PARAMS = [
    (False, {}),
    (True, {}),
    (True, {"azimuth": 24}),
    (True, {"doppler": 8}),
    (True, {"range": 16}),
    (True, {"elevation": 4}),
    ({"range": True, "doppler": True}, {}),
]

AWRL6844_PARAMS = [
    (False, {}),
    (True, {}),
    (True, {"azimuth": 8}),
    (True, {"doppler": 8}),
    (True, {"range": 16}),
    (True, {"elevation": 8}),
    ({"range": True, "doppler": True}, {}),
]

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

    awr1843aop = rspn.AWR1843AOP(window=False, size={})
    assert awr1843aop(data).shape == (2, 4, 4, 3, 8)

    awr1642 = rspn.AWR1642Boost(window=False, size={})
    assert awr1642(data[:, :, :2]).shape == (2, 4, 1, 8, 8)

    # AWR2944EVM: I-only input; range bins = input_range // 2 + 1 = 9
    data = _i_float32(SHAPE["AWR2944EVM"])
    awr2944 = rspn.AWR2944EVM(window=False, size={})
    assert awr2944(data).shape == (2, 4, 2, 12, 9)

    # AWRL6844EVM: I-only input, 4x4 virtual array; range bins = 16//2+1 = 9
    data = _i_float32(SHAPE["AWRL6844EVM"])
    awrl6844 = rspn.AWRL6844EVM(window=False, size={})
    assert awrl6844(data).shape == (2, 4, 4, 4, 9)


@pytest.mark.parametrize("window,size", AWR2944_PARAMS)
def test_jax_awr2944(window, size):
    """Test jax vs numpy RSP for AWR2944EVM (I-only data)."""
    rsp_numpy = rspn.AWR2944EVM(window=window, size=size)
    rsp_jax = rspj.AWR2944EVM(window=window, size=size)

    data = _i_float32(SHAPE["AWR2944EVM"])
    numpy_result = rsp_numpy(data)
    jax_result = rsp_jax(jnp.array(data))

    assert np.allclose(numpy_result, np.array(jax_result), atol=1e-4)


@pytest.mark.parametrize("window,size", AWR2944_PARAMS)
def test_torch_awr2944(window, size):
    """Test torch vs numpy RSP for AWR2944EVM (I-only data)."""
    rsp_numpy = rspn.AWR2944EVM(window=window, size=size)
    rsp_torch = rspt.AWR2944EVM(window=window, size=size)

    data = _i_float32(SHAPE["AWR2944EVM"])
    numpy_result = rsp_numpy(data)
    torch_result = rsp_torch(torch.from_numpy(data))

    assert np.allclose(numpy_result, torch_result.numpy(), atol=1e-4)


def test_torch_awr2944_backward():
    """Test that gradients flow through AWR2944EVM in torch."""
    rng = np.random.default_rng()
    data = torch.from_numpy(
        rng.random(SHAPE["AWR2944EVM"]).astype(np.float32)
    ).requires_grad_(True)

    result = rspt.AWR2944EVM()(data)
    result.abs().sum().backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape


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


@pytest.mark.parametrize("window,size", AWRL6844_PARAMS)
def test_jax_awrl6844evm(window, size):
    """Test jax vs numpy RSP for AWRL6844EVM (I-only data)."""
    rsp_numpy = rspn.AWRL6844EVM(window=window, size=size)
    rsp_jax = rspj.AWRL6844EVM(window=window, size=size)

    data = _i_float32(SHAPE["AWRL6844EVM"])
    numpy_result = rsp_numpy(data)
    jax_result = rsp_jax(jnp.array(data))

    assert np.allclose(numpy_result, np.array(jax_result), atol=1e-4)


@pytest.mark.parametrize("window,size", AWRL6844_PARAMS)
def test_torch_awrl6844evm(window, size):
    """Test torch vs numpy RSP for AWRL6844EVM (I-only data)."""
    rsp_numpy = rspn.AWRL6844EVM(window=window, size=size)
    rsp_torch = rspt.AWRL6844EVM(window=window, size=size)

    data = _i_float32(SHAPE["AWRL6844EVM"])
    numpy_result = rsp_numpy(data)
    torch_result = rsp_torch(torch.from_numpy(data))

    assert np.allclose(numpy_result, torch_result.numpy(), atol=1e-4)

    with pytest.raises(ValueError):
        rsp_torch(torch.from_numpy(data)[:, :, :-1, :, :])
    with pytest.raises(ValueError):
        rsp_torch(torch.from_numpy(data)[:, :, :, :-1, :])


def test_torch_awrl6844evm_backward():
    """Test that gradients flow through AWRL6844EVM in torch."""
    rng = np.random.default_rng()
    data = torch.from_numpy(
        rng.random(SHAPE["AWRL6844EVM"]).astype(np.float32)
    ).requires_grad_(True)

    result = rspt.AWRL6844EVM()(data)
    result.abs().sum().backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape
