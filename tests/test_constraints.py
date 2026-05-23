"""Tests for xwr.constraints."""

import dataclasses

import pytest

from xwr.config import DCAConfig, XWRConfig
from xwr.constraints import (
    AdcSamplesPowerOfTwo,
    ConstraintCheck,
    CubeSizeLimit,
    ExcessRampTime,
    FrameDutyCycle,
    FrameLengthPowerOfTwo,
    FrequencyRange,
    MaxBandwidth,
    MaxSampleRate,
    MinSampleRate,
    NetworkUtilization,
    RFDutyCycle,
    ReceiveBuffer,
    check_config,
)
from xwr.radar import AWR1642, AWR1843, AWR1843L, AWR2944, AWRL6844

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def radar():
    """A valid AWR1843 config that passes all constraints."""
    return XWRConfig(
        device=AWR1843,
        frequency=77.0,
        idle_time=110.0,
        adc_start_time=4.0,
        ramp_end_time=56.0,   # excess = 56 - 4 - 51.2 = 0.8us
        tx_start_time=1.0,
        freq_slope=70.0,
        adc_samples=256,       # power of two; sample_time = 51.2us
        sample_rate=5_000,     # 5 Ksps, within [2000, 25000]
        frame_length=64,       # power of two; duty cycle ~32%
        frame_period=100.0,
    )


@pytest.fixture
def capture():
    """A valid DCAConfig that passes all cross-config constraints."""
    return DCAConfig()


def replace(cfg, **kwargs):
    return dataclasses.replace(cfg, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result_for(constraint, radar, capture=None):
    return constraint.check(radar, capture)


def assert_passed(r: ConstraintCheck):
    assert r.passed is True, f"Expected pass, got: {r}"


def assert_failed(r: ConstraintCheck):
    assert r.passed is False, f"Expected fail, got: {r}"


def assert_skipped(r: ConstraintCheck):
    assert r.passed is None, f"Expected skip, got: {r}"


# ---------------------------------------------------------------------------
# FrameDutyCycle
# ---------------------------------------------------------------------------

def test_frame_duty_cycle_pass(radar):
    assert_passed(FrameDutyCycle.check(radar))


def test_frame_duty_cycle_fail(radar):
    # frame_period barely less than frame_time → duty cycle > 99%
    assert_failed(FrameDutyCycle.check(replace(radar, frame_period=30.0)))


# ---------------------------------------------------------------------------
# RFDutyCycle
# ---------------------------------------------------------------------------

def test_rf_duty_cycle_pass(radar):
    # RF duty = 56 * 3 * 64 / (100 * 1000) = 10.75%
    assert_passed(RFDutyCycle.check(radar))


def test_rf_duty_cycle_fail(radar):
    # ramp_end_time=300 → RF duty = 300 * 3 * 64 / (100 * 1000) = 57.6%
    assert_failed(RFDutyCycle.check(replace(radar, ramp_end_time=300.0)))


def test_rf_duty_cycle_at_boundary(radar):
    # RF duty exactly 50% should fail (< 50% required)
    # ramp_end_time = 0.5 * frame_period * 1000 / (num_tx * frame_length)
    # = 0.5 * 100000 / (3 * 64) = 260.4167us → RF = 50.0%
    assert_failed(RFDutyCycle.check(replace(radar, ramp_end_time=260.42)))


# ---------------------------------------------------------------------------
# ExcessRampTime
# ---------------------------------------------------------------------------

def test_excess_ramp_time_pass(radar):
    assert_passed(ExcessRampTime.check(radar))


def test_excess_ramp_time_fail(radar):
    # ramp_end_time < adc_start_time + sample_time
    assert_failed(ExcessRampTime.check(replace(radar, ramp_end_time=10.0)))


# ---------------------------------------------------------------------------
# CubeSizeLimit
# ---------------------------------------------------------------------------

def test_cube_size_limit_pass(radar):
    # AWR1843: frame_size = 64*3*4*256*4 = 786432 < 1MiB
    assert_passed(CubeSizeLimit.check(radar))


def test_cube_size_limit_fail(radar):
    # Push frame_length up so frame_size > 1 MiB (AWR1843 limit)
    # frame_size = 128 * 3 * 4 * 256 * 4 = 1572864 > 1048576
    assert_failed(CubeSizeLimit.check(replace(radar, frame_length=128)))


def test_cube_size_limit_awr2944_larger_buffer(radar):
    # Same frame_length=128 passes on AWR2944 (2.5 MiB limit, 2-byte samples)
    # frame_size = 128 * 4 * 4 * 256 * 2 = 1048576 < 2.5 MiB
    cfg = replace(radar, device=AWR2944, frame_length=128)
    assert_passed(CubeSizeLimit.check(cfg))


def test_cube_size_limit_skip_unknown_device(radar):
    class CustomRadar(AWR1843):
        pass
    assert_skipped(CubeSizeLimit.check(replace(radar, device=CustomRadar)))


# ---------------------------------------------------------------------------
# FrameLengthPowerOfTwo
# ---------------------------------------------------------------------------

def test_frame_length_power_of_two_pass(radar):
    assert_passed(FrameLengthPowerOfTwo.check(radar))


def test_frame_length_power_of_two_fail(radar):
    assert_failed(FrameLengthPowerOfTwo.check(replace(radar, frame_length=48)))


# ---------------------------------------------------------------------------
# AdcSamplesPowerOfTwo
# ---------------------------------------------------------------------------

def test_adc_samples_power_of_two_pass(radar):
    assert_passed(AdcSamplesPowerOfTwo.check(radar))


def test_adc_samples_power_of_two_fail(radar):
    assert_failed(AdcSamplesPowerOfTwo.check(replace(radar, adc_samples=300)))


# ---------------------------------------------------------------------------
# MaxSampleRate
# ---------------------------------------------------------------------------

def test_max_sample_rate_pass(radar):
    assert_passed(MaxSampleRate.check(radar))


def test_max_sample_rate_fail_awr1642(radar):
    cfg = replace(radar, device=AWR1642, sample_rate=20_000)
    assert_failed(MaxSampleRate.check(cfg))


def test_max_sample_rate_at_limit(radar):
    assert_passed(MaxSampleRate.check(replace(radar, sample_rate=25_000)))


def test_max_sample_rate_skip_unknown(radar):
    class CustomRadar(AWR1843):
        pass
    assert_skipped(MaxSampleRate.check(replace(radar, device=CustomRadar)))


# ---------------------------------------------------------------------------
# MinSampleRate
# ---------------------------------------------------------------------------

def test_min_sample_rate_pass(radar):
    assert_passed(MinSampleRate.check(radar))


def test_min_sample_rate_fail(radar):
    assert_failed(MinSampleRate.check(replace(radar, sample_rate=1_000)))


def test_min_sample_rate_skip_awr1642(radar):
    # AWR1642 has no minimum sample rate
    cfg = replace(radar, device=AWR1642, sample_rate=1_000)
    assert_skipped(MinSampleRate.check(cfg))


def test_min_sample_rate_applies_to_awr1843l(radar):
    cfg = replace(radar, device=AWR1843L, sample_rate=1_000)
    assert_failed(MinSampleRate.check(cfg))


# ---------------------------------------------------------------------------
# FrequencyRange
# ---------------------------------------------------------------------------

def test_frequency_range_pass(radar):
    # start=77.0, end=77.0 + 3584/1000 = 80.584 GHz, within 76-81
    assert_passed(FrequencyRange.check(radar))


def test_frequency_range_fail_start_too_low(radar):
    assert_failed(FrequencyRange.check(replace(radar, frequency=75.0)))


def test_frequency_range_fail_end_too_high(radar):
    # freq_slope=80 → bandwidth = 80 * 51.2 = 4096 MHz → end = 77 + 4.096 = 81.096 GHz
    assert_failed(FrequencyRange.check(replace(radar, freq_slope=80.0)))


def test_frequency_range_skip_unknown_device(radar):
    class CustomRadar(AWR1843):
        pass
    assert_skipped(FrequencyRange.check(replace(radar, device=CustomRadar)))


def test_frequency_range_awrl6844_pass(radar):
    # AWRL6844 band: 57-64 GHz; end = 60.0 + 3.584 = 63.584 GHz
    cfg = replace(radar, device=AWRL6844, frequency=60.0)
    assert_passed(FrequencyRange.check(cfg))


def test_frequency_range_awrl6844_fail_start_too_low(radar):
    cfg = replace(radar, device=AWRL6844, frequency=56.0)
    assert_failed(FrequencyRange.check(cfg))


def test_frequency_range_awrl6844_fail_end_too_high(radar):
    # frequency=62.0, freq_slope=80 → end = 62.0 + 4.096 = 66.096 GHz > 64
    cfg = replace(radar, device=AWRL6844, frequency=62.0, freq_slope=80.0)
    assert_failed(FrequencyRange.check(cfg))


def test_frequency_range_all_76ghz_devices(radar):
    # AWR1642, AWR1843, AWR2944 all use the 76-81 GHz band
    for device in [AWR1642, AWR1843, AWR1843L, AWR2944]:
        assert_passed(FrequencyRange.check(replace(radar, device=device)))


# ---------------------------------------------------------------------------
# MaxBandwidth
# ---------------------------------------------------------------------------

def test_max_bandwidth_pass(radar):
    # bandwidth = 70 * 51.2 = 3584 MHz < 4000 MHz
    assert_passed(MaxBandwidth.check(radar))


def test_max_bandwidth_fail(radar):
    # freq_slope=80 → bandwidth = 80 * 51.2 = 4096 MHz > 4000 MHz
    assert_failed(MaxBandwidth.check(replace(radar, freq_slope=80.0)))


def test_max_bandwidth_at_limit(radar):
    # freq_slope = 4000 / 51.2 = 78.125 → bandwidth exactly 4000 MHz (pass)
    assert_passed(MaxBandwidth.check(replace(radar, freq_slope=78.125)))


def test_max_bandwidth_skip_awrl6844(radar):
    # AWRL6844 has no MaxBandwidth entry; FrequencyRange enforces its limits
    assert_skipped(MaxBandwidth.check(replace(radar, device=AWRL6844)))


def test_max_bandwidth_applies_to_awr1642(radar):
    cfg = replace(radar, device=AWR1642, freq_slope=80.0)
    assert_failed(MaxBandwidth.check(cfg))


# ---------------------------------------------------------------------------
# NetworkUtilization
# ---------------------------------------------------------------------------

def test_network_utilization_pass(radar, capture):
    assert_passed(NetworkUtilization.check(radar, capture))


def test_network_utilization_fail(radar, capture):
    # Increase frame rate so radar throughput exceeds 80% of capture card
    assert_failed(NetworkUtilization.check(replace(radar, frame_period=1.0), capture))


def test_network_utilization_skip_no_capture(radar):
    assert_skipped(NetworkUtilization.check(radar, None))


# ---------------------------------------------------------------------------
# ReceiveBuffer
# ---------------------------------------------------------------------------

def test_receive_buffer_pass(radar, capture):
    assert_passed(ReceiveBuffer.check(radar, capture))


def test_receive_buffer_fail(radar, capture):
    # socket_buffer smaller than 2 frames
    small_capture = replace(capture, socket_buffer=radar.frame_size)
    assert_failed(ReceiveBuffer.check(radar, small_capture))


def test_receive_buffer_skip_no_capture(radar):
    assert_skipped(ReceiveBuffer.check(radar, None))


# ---------------------------------------------------------------------------
# check_config integration
# ---------------------------------------------------------------------------

def test_check_config_all_pass(radar, capture):
    results = check_config(radar, capture, log=False)
    failures = [r for r in results if r.passed is False]
    assert not failures, f"Unexpected failures: {failures}"


def test_check_config_returns_all_constraints(radar, capture):
    results = check_config(radar, capture, log=False)
    assert len(results) == 12  # one per constraint class
    assert all(isinstance(r, ConstraintCheck) for r in results)


def test_check_config_no_capture_skips_cross_config(radar):
    results = check_config(radar, capture=None, log=False)
    cross_config = {NetworkUtilization, ReceiveBuffer}
    for r in results:
        if r.constraint in cross_config:
            assert r.passed is None


def test_check_config_no_log(radar, capture, caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="xwr/constraints"):
        check_config(radar, capture, log=False)
    assert not caplog.records


def test_check_config_log(radar, capture, caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="xwr/constraints"):
        check_config(radar, capture, log=True)
    assert caplog.records
