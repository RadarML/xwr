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
    "AWR2E44PEVM": (2, 4, 4, 4, 16),
    "AWRL6844EVM": (2, 4, 4, 4, 16),
    "AWRL6888EVM": (2, 4, 16, 4, 16),
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

AWR2E44P_PARAMS = [
    (False, {}),
    (True, {}),
    (True, {"azimuth": 32}),
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

    # AWR2E44PEVM: I-only input, sparse 2x19 virtual array
    data = _i_float32(SHAPE["AWR2E44PEVM"])
    awr2e44p = rspn.AWR2E44PEVM(window=False, size={})
    assert awr2e44p(data).shape == (2, 4, 2, 19, 9)

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


@pytest.mark.parametrize("radar", list(SHAPE.keys()))
def test_doppler_correction_backends(radar):
    """Doppler correction matches across numpy, jax, and torch."""
    shape = SHAPE[radar]
    rn = getattr(rspn, radar)(window=True, size={"doppler": 8})
    rj = getattr(rspj, radar)(window=True, size={"doppler": 8})
    rt = getattr(rspt, radar)(window=True, size={"doppler": 8})

    data = (_i_float32(shape) if rn.SAMPLE_TYPE == "I"
            else _iq_complex(shape))
    if rn.SAMPLE_TYPE == "IQ":
        data_n = rsp.iq_from_iiqq(data)
    else:
        data_n = data

    out_n = rn.doppler_correction(rn.doppler_range(data_n))
    out_j = rj.doppler_correction(rj.doppler_range(jnp.array(data_n)))
    out_t = rt.doppler_correction(rt.doppler_range(torch.from_numpy(data_n)))

    assert out_n.dtype == np.complex64
    assert np.allclose(out_n, np.array(out_j), atol=1e-4)
    assert np.allclose(out_n, out_t.numpy(), atol=1e-4)

    # Full pipeline with correction also agrees.
    spec_n = rn.elevation_azimuth(out_n)
    spec_t = rt.elevation_azimuth(out_t)
    assert np.allclose(spec_n, spec_t.numpy(), atol=1e-3)


@pytest.mark.parametrize("n_doppler", [16, 15])
@pytest.mark.parametrize("k0", [3, -5, 0])
def test_doppler_correction_physics(n_doppler, k0):
    """A moving target has equal phase across TXs after correction."""
    n_tx, n_rx, n_range, r0 = 3, 4, 8, 2
    f0 = k0 / n_doppler
    n = np.arange(n_doppler)[:, None]
    t = np.arange(n_tx)[None, :]
    slow = np.exp(2j * np.pi * f0 * (n + t / n_tx))  # doppler x tx
    fast = np.exp(2j * np.pi * r0 * np.arange(n_range) / n_range)
    x = (slow[None, :, :, None, None] * fast[None, None, None, None, :]
         * np.ones((1, 1, 1, n_rx, 1))).astype(np.complex64)

    for r in [rspn.AWR1843Boost(), rspj.AWR1843Boost(),
              rspt.AWR1843Boost()]:
        if isinstance(r, rspj.AWR1843Boost):
            x_in = jnp.array(x)
        elif isinstance(r, rspt.AWR1843Boost):
            x_in = torch.from_numpy(x)
        else:
            x_in = x
        rd = r.doppler_range(x_in)
        rd_c = r.doppler_correction(rd)
        peak = n_doppler // 2 + k0
        raw = np.angle(np.asarray(rd)[0, peak, :, 0, r0])
        cor = np.angle(np.asarray(rd_c)[0, peak, :, 0, r0])

        expected = 2 * np.pi * f0 * np.arange(n_tx) / n_tx
        assert np.allclose(
            np.exp(1j * (raw - raw[0])), np.exp(1j * expected), atol=1e-4)
        assert np.allclose(np.exp(1j * (cor - cor[0])), 1, atol=1e-4)


def test_doppler_correction_identity_at_zero():
    """Correction leaves the zero-Doppler bin unchanged."""
    rd = _iq_complex((2, 8, 3, 4, 8))
    out = rspn.AWR1843Boost().doppler_correction(rd)
    assert np.allclose(out[:, 4], rd[:, 4])


def test_torch_doppler_correction_backward():
    """Gradients flow through torch doppler correction."""
    data = torch.from_numpy(
        _iq_complex(SHAPE["AWR1843Boost"])).requires_grad_(True)
    r = rspt.AWR1843Boost()
    out = r.elevation_azimuth(r.doppler_correction(r.doppler_range(data)))
    out.abs().sum().backward()
    assert data.grad is not None


def test_awr2e44p_virtual_array():
    """Check AWR2E44PEVM TX-RX positions and calibration against TI's."""
    # antGeometryCfg from TI's AWR2E44P demo profiles (TX-major, RX-minor).
    geometry = [
        (1, 0), (1, 2), (1, 7), (1, 9),
        (1, 3), (1, 5), (1, 10), (1, 12),
        (1, 6), (1, 8), (1, 13), (1, 15),
        (0, 9), (0, 11), (0, 16), (0, 18)]

    # antennaCalibParams from TI's AWR2E44P DDM demo profiles: Im(Q) Re(I).
    calib = np.array([float(v) for v in (
        "0.6718 0.3970 0.5714 0.5259 -0.0907 0.8665 0.0989 -0.9754 "
        "0.8083 0.0366 0.7769 0.2025 0.3331 0.8275 -0.3769 -0.9263 "
        "-0.6290 -0.4124 -0.5234 -0.5298 0.1282 -0.8125 -0.1418 0.9100 "
        "0.5869 0.3367 0.4966 0.4459 -0.0732 0.7262 0.0808 -0.8108"
    ).split()]).reshape(16, 2)
    calib = calib[:, 1] + 1j * calib[:, 0]
    calib = calib / np.abs(calib)

    # Label each TX-RX pair with a unique nonzero value.
    rd = (np.arange(16, dtype=np.float32) + 1).reshape(1, 1, 4, 4, 1)
    rd = rd.astype(np.complex64)
    expected = np.zeros((2, 19), dtype=np.complex64)
    for i, (row, col) in enumerate(geometry):
        expected[row, col] = (i + 1) * calib[i]

    numpy_mimo = rspn.AWR2E44PEVM().mimo_virtual_array(rd)[0, 0, :, :, 0]
    assert np.allclose(numpy_mimo, expected, atol=1e-5)
    jax_mimo = rspj.AWR2E44PEVM().mimo_virtual_array(jnp.array(rd))
    assert np.allclose(np.array(jax_mimo)[0, 0, :, :, 0], expected, atol=1e-5)
    torch_mimo = rspt.AWR2E44PEVM().mimo_virtual_array(torch.from_numpy(rd))
    assert np.allclose(torch_mimo.numpy()[0, 0, :, :, 0], expected, atol=1e-5)
    # Positions with no TX-RX pair stay empty.
    assert np.count_nonzero(numpy_mimo) == 16

    with pytest.raises(ValueError):
        rspn.AWR2E44PEVM().mimo_virtual_array(rd[:, :, :3])
    with pytest.raises(ValueError):
        rspj.AWR2E44PEVM().mimo_virtual_array(jnp.array(rd[:, :, :3]))
    with pytest.raises(ValueError):
        rspt.AWR2E44PEVM().mimo_virtual_array(torch.from_numpy(rd[:, :, :3]))


@pytest.mark.parametrize("window,size", AWR2E44P_PARAMS)
def test_jax_awr2e44p(window, size):
    """Test jax vs numpy RSP for AWR2E44PEVM (I-only data)."""
    rsp_numpy = rspn.AWR2E44PEVM(window=window, size=size)
    rsp_jax = rspj.AWR2E44PEVM(window=window, size=size)

    data = _i_float32(SHAPE["AWR2E44PEVM"])
    numpy_result = rsp_numpy(data)
    jax_result = rsp_jax(jnp.array(data))

    assert np.allclose(numpy_result, np.array(jax_result), atol=1e-4)


@pytest.mark.parametrize("window,size", AWR2E44P_PARAMS)
def test_torch_awr2e44p(window, size):
    """Test torch vs numpy RSP for AWR2E44PEVM (I-only data)."""
    rsp_numpy = rspn.AWR2E44PEVM(window=window, size=size)
    rsp_torch = rspt.AWR2E44PEVM(window=window, size=size)

    data = _i_float32(SHAPE["AWR2E44PEVM"])
    numpy_result = rsp_numpy(data)
    torch_result = rsp_torch(torch.from_numpy(data))

    assert np.allclose(numpy_result, torch_result.numpy(), atol=1e-4)


def test_torch_awr2e44p_backward():
    """Test that gradients flow through AWR2E44PEVM in torch."""
    rng = np.random.default_rng()
    data = torch.from_numpy(
        rng.random(SHAPE["AWR2E44PEVM"]).astype(np.float32)
    ).requires_grad_(True)

    result = rspt.AWR2E44PEVM()(data)
    result.abs().sum().backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape


AWRL6888_PARAMS = [
    (False, {}),
    (True, {}),
    (True, {"azimuth": 32}),
    (True, {"doppler": 8}),
    (True, {"range": 16}),
    (True, {"elevation": 16}),
    ({"range": True, "doppler": True}, {}),
]


def test_awrl6888_virtual_array():
    """Check AWRL6888EVM virtual array and calibration against TI's."""
    # antGeometryTX/RX from TI's profile_8T8R_tdm.cfg; (row, col) pairs where
    # the row is azimuth and the column is elevation.
    tx = np.array([int(v) for v in "1 0 0 1 1 2 0 3 1 4 0 5 1 6 0 7".split()])
    rx = np.array([int(v) for v in "0 0 2 0 4 0 6 0 1 0 3 0 5 0 7 0".split()])
    tx, rx = tx.reshape(8, 2), rx.reshape(8, 2)

    # compRangeBiasAndRxChanPhase from TI's profile_8T8R_tdm.cfg
    from xwr.radar import AWRL6888
    calib = np.array([
        float(v) for v in AWRL6888._COMP_RANGE_BIAS_RX_PHASE.split()[1:]])
    calib = calib.reshape(64, 2)
    calib = calib[:, 0] + 1j * calib[:, 1]
    calib = calib / np.abs(calib)

    # Label each chirp slot / RX chain with a unique nonzero value.
    rd = (np.arange(64, dtype=np.float32) + 1).reshape(1, 1, 16, 4, 1)
    rd = rd.astype(np.complex64)
    expected = np.zeros((8, 9), dtype=np.complex64)
    for s in range(16):
        for k in range(4):
            # TI virtual antenna: TX t, RX m (TX-major, RX-minor).
            t, m = s // 2, (s % 2) * 4 + k
            # Elevation is reversed relative to TI's geometry (image order).
            az = tx[t, 0] + rx[m, 0]
            el = 7 - (tx[t, 1] + rx[m, 1])
            expected[el, az] = (s * 4 + k + 1) * calib[t * 8 + m]

    numpy_mimo = rspn.AWRL6888EVM().mimo_virtual_array(rd)[0, 0, :, :, 0]
    assert np.allclose(numpy_mimo, expected, atol=1e-5)
    jax_mimo = rspj.AWRL6888EVM().mimo_virtual_array(jnp.array(rd))
    assert np.allclose(np.array(jax_mimo)[0, 0, :, :, 0], expected, atol=1e-5)
    torch_mimo = rspt.AWRL6888EVM().mimo_virtual_array(torch.from_numpy(rd))
    assert np.allclose(torch_mimo.numpy()[0, 0, :, :, 0], expected, atol=1e-5)
    # 64 of the 72 positions are filled.
    assert np.count_nonzero(numpy_mimo) == 64

    # 8x8 (physical TX x RX) input is rejected.
    rd8 = np.zeros((1, 1, 8, 8, 1), dtype=np.complex64)
    with pytest.raises(ValueError):
        rspn.AWRL6888EVM().mimo_virtual_array(rd8)
    with pytest.raises(ValueError):
        rspj.AWRL6888EVM().mimo_virtual_array(jnp.array(rd8))
    with pytest.raises(ValueError):
        rspt.AWRL6888EVM().mimo_virtual_array(torch.from_numpy(rd8))


def test_numpy_awrl6888_shape():
    """Check AWRL6888EVM output shape: 8x9 virtual array, range = 16//2+1."""
    data = _i_float32(SHAPE["AWRL6888EVM"])
    out = rspn.AWRL6888EVM(window=False, size={})(data)
    assert out.shape == (2, 4, 8, 9, 9)


@pytest.mark.parametrize("window,size", AWRL6888_PARAMS)
@pytest.mark.parametrize("doppler_correction", [False, True])
def test_jax_awrl6888(window, size, doppler_correction):
    """Test jax vs numpy RSP for AWRL6888EVM (I-only data)."""
    kwargs = dict(
        window=window, size=size, doppler_correction=doppler_correction)
    data = _i_float32(SHAPE["AWRL6888EVM"])
    numpy_result = rspn.AWRL6888EVM(**kwargs)(data)
    jax_result = rspj.AWRL6888EVM(**kwargs)(jnp.array(data))
    # float32 FFT error scales with the output's peak (the DC bin, ~1e3 here
    # with 16 chirp slots), so a fixed atol=1e-4 is flaky.
    assert np.allclose(
        numpy_result, np.array(jax_result),
        atol=1e-6 * np.abs(numpy_result).max())


@pytest.mark.parametrize("window,size", AWRL6888_PARAMS)
@pytest.mark.parametrize("doppler_correction", [False, True])
def test_torch_awrl6888(window, size, doppler_correction):
    """Test torch vs numpy RSP for AWRL6888EVM (I-only data)."""
    kwargs = dict(
        window=window, size=size, doppler_correction=doppler_correction)
    data = _i_float32(SHAPE["AWRL6888EVM"])
    numpy_result = rspn.AWRL6888EVM(**kwargs)(data)
    torch_result = rspt.AWRL6888EVM(**kwargs)(torch.from_numpy(data))
    assert np.allclose(
        numpy_result, torch_result.numpy(),
        atol=1e-6 * np.abs(numpy_result).max())


def test_torch_awrl6888_backward():
    """Test that gradients flow through AWRL6888EVM in torch."""
    rng = np.random.default_rng()
    data = torch.from_numpy(
        rng.random(SHAPE["AWRL6888EVM"]).astype(np.float32)
    ).requires_grad_(True)

    result = rspt.AWRL6888EVM()(data)
    result.abs().sum().backward()

    assert data.grad is not None
    assert data.grad.shape == data.shape
