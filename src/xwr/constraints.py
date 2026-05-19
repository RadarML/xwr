"""Radar configuration constraints.

This module documents and enforces known constraints on radar configurations.

!!! warning

    These constraints are not exhaustive or guaranteed to be correct. If you
    find a missing constraint, or run into an undocumented or incorrectly
    implemented check, please open an issue!
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .config import DCAConfig, XWRConfig


class Constraint(ABC):
    """Base class for a radar configuration constraint."""

    @staticmethod
    @abstractmethod
    def check(
        radar: XWRConfig, capture: DCAConfig | None = None
    ) -> "ConstraintCheck":
        """Return a [`ConstraintCheck`][xwr.constraints.ConstraintCheck] result."""


@dataclass(frozen=True)
class ConstraintCheck:
    """Result of a single constraint check.

    Attributes:
        constraint: type of the constraint being checked.
        passed: `True` if the constraint is satisfied, `False` if violated, or
            `None` if not applicable.
        detail: computed value (on pass/skip) or violation description
            (on fail).
    """

    constraint: type[Constraint]
    passed: bool | None
    detail: str


class FrameDutyCycle(Constraint):
    """Active frame time must not exceed the frame period.

    The fraction of each period spent actively transmitting must stay
    below 99%:

        frame_time / frame_period < 99%

    where `frame_time = chirp_time × frame_length / 1000` ms (see
    [`XWRConfig.frame_time`][xwr.config.XWRConfig]).
    """

    @staticmethod
    def check(radar, capture=None):
        duty_cycle = 100 * radar.frame_time / radar.frame_period
        passed = duty_cycle < 99
        detail = f"frame duty cycle = {duty_cycle:.1f}%"
        if not passed:
            detail += " (must be < 99%)"
        return ConstraintCheck(FrameDutyCycle, passed, detail)


class RFDutyCycle(Constraint):
    """RF transmitter on-time must not exceed 50% of the frame period.

    Counts only the time the RF ramp is active (`ramp_end_time` per TX
    per chirp), excluding idle time and the inter-frame gap:

        ramp_end_time × num_tx × frame_length / (frame_period × 1000) < 50%
    """

    @staticmethod
    def check(radar, capture=None):
        rf_on_us = radar.ramp_end_time * radar.num_tx * radar.frame_length
        period_us = radar.frame_period * 1e3
        duty_cycle = 100 * rf_on_us / period_us
        passed = duty_cycle < 50
        detail = f"RF duty cycle = {duty_cycle:.1f}%"
        if not passed:
            detail += " (should be < 50%)"
        return ConstraintCheck(RFDutyCycle, passed, detail)


class ExcessRampTime(Constraint):
    """ADC sampling must complete before the frequency ramp ends.

    The ADC window must fit within the chirp ramp:

        ramp_end_time - adc_start_time ≥ T_s

    where `T_s = adc_samples / sample_rate × 1000` μs (see
    [`XWRConfig.sample_time`][xwr.config.XWRConfig]).
    """

    @staticmethod
    def check(radar, capture=None):
        excess = radar.ramp_end_time - radar.adc_start_time - radar.sample_time
        passed = excess >= 0
        detail = f"excess ramp time = {excess:.1f}us"
        if not passed:
            detail += " (must be ≥ 0)"
        return ConstraintCheck(ExcessRampTime, passed, detail)


class CubeSizeLimit(Constraint):
    """Radar data cube must fit within the device L3 radar buffer.

    Checks `frame_size` (total bytes for one frame) against the device
    hardware L3 memory limit:

        frame_size ≤ _LIMITS[device_name]

    where `frame_size = frame_length × num_tx × num_rx × adc_samples
    × BYTES_PER_SAMPLE` (see [`XWRConfig.frame_size`][xwr.config.XWRConfig]).

    | Device   | L3 size  |
    |----------|----------|
    | AWR1642  | 768 KiB  |
    | AWR1843  | 1 MiB    |
    | AWR2944  | 2.5 MiB  |
    | AWRL6844 | 896 KiB  |
    """

    _LIMITS: ClassVar[dict[str, int]] = {
        "AWR1642":  768 * 1024,
        "AWR1843":  1024 * 1024,
        "AWR1843L": 1024 * 1024,
        "AWR2944":  int(2.5 * 1024 * 1024),
        "AWRL6844": 896 * 1024,
    }

    @staticmethod
    def check(radar, capture=None):
        limit = CubeSizeLimit._LIMITS.get(radar.device_name)
        if limit is None:
            return ConstraintCheck(
                CubeSizeLimit, None,
                f"not checked for {radar.device_name}")
        passed = radar.frame_size <= limit
        detail = (
            f"frame_size = {radar.frame_size} bytes, "
            f"L3 limit = {limit} bytes ({limit // 1024} KiB)")
        if not passed:
            detail = (
                f"frame_size = {radar.frame_size} bytes "
                f"> device L3 limit {limit} bytes ({limit // 1024} KiB)")
        return ConstraintCheck(CubeSizeLimit, passed, detail)


class FrameLengthPowerOfTwo(Constraint):
    """Frame length must be a power of two.

    The number of chirps per TX antenna per frame must be a power of two
    for the range-Doppler FFT:

        frame_length & (frame_length - 1) == 0
    """

    @staticmethod
    def check(radar, capture=None):
        fl = radar.frame_length
        passed = fl & (fl - 1) == 0
        detail = f"frame_length = {fl}"
        if not passed:
            detail += " (not a power of two)"
        return ConstraintCheck(FrameLengthPowerOfTwo, passed, detail)


class AdcSamplesPowerOfTwo(Constraint):
    """ADC samples per chirp must be a power of two.

    The number of samples per chirp must be a power of two for the
    range FFT:

        adc_samples & (adc_samples - 1) == 0
    """

    @staticmethod
    def check(radar, capture=None):
        n = radar.adc_samples
        passed = n & (n - 1) == 0
        detail = f"adc_samples = {n}"
        if not passed:
            detail += " (not a power of two)"
        return ConstraintCheck(AdcSamplesPowerOfTwo, passed, detail)


class MaxSampleRate(Constraint):
    """ADC sampling rate must not exceed the device maximum.

    | Device   | Maximum     |
    |----------|-------------|
    | AWR1642  | 12,500 Ksps |
    | AWR1843  | 25,000 Ksps |
    | AWR2944  | 37,500 Ksps |
    | AWRL6844 | 25,000 Ksps |
    """

    _LIMITS: ClassVar[dict[str, int]] = {
        "AWR1642":  12_500,
        "AWR1843":  25_000,
        "AWR1843L": 25_000,
        "AWR2944":  37_500,
        "AWRL6844": 25_000,
    }

    @staticmethod
    def check(radar, capture=None):
        limit = MaxSampleRate._LIMITS.get(radar.device_name)
        if limit is None:
            return ConstraintCheck(
                MaxSampleRate, None,
                f"not checked for {radar.device_name}")
        passed = radar.sample_rate <= limit
        detail = f"sample_rate = {radar.sample_rate} Ksps, maximum = {limit} Ksps"
        if not passed:
            detail = (
                f"sample_rate = {radar.sample_rate} Ksps "
                f"> device maximum {limit} Ksps")
        return ConstraintCheck(MaxSampleRate, passed, detail)


class MinSampleRate(Constraint):
    """ADC sampling rate must meet the device minimum.

    | Device   | Minimum    |
    |----------|------------|
    | AWR1843  | 2,000 Ksps |
    """

    _LIMITS: ClassVar[dict[str, int]] = {
        "AWR1843":  2_000,
        "AWR1843L": 2_000,
    }

    @staticmethod
    def check(radar, capture=None):
        limit = MinSampleRate._LIMITS.get(radar.device_name)
        if limit is None:
            return ConstraintCheck(
                MinSampleRate, None,
                f"not checked for {radar.device_name}")
        passed = radar.sample_rate >= limit
        detail = f"sample_rate = {radar.sample_rate} Ksps, minimum = {limit} Ksps"
        if not passed:
            detail = (
                f"sample_rate = {radar.sample_rate} Ksps "
                f"< device minimum {limit} Ksps")
        return ConstraintCheck(MinSampleRate, passed, detail)


class FrequencyRange(Constraint):
    """Start and end frequencies must lie within the device RF band.

    Both the start frequency and the end frequency
    (`start_freq + bandwidth / 1000`) are checked against per-device limits:

    | Device   | Min (GHz) | Max (GHz) |
    |----------|-----------|-----------|
    | AWR1642  | 76        | 81        |
    | AWR1843  | 76        | 81        |
    | AWR2944  | 76        | 81        |
    | AWRL6844 | 57        | 64        |
    """

    _LIMITS: ClassVar[dict[str, tuple[float, float]]] = {
        "AWR1642":  (76.0, 81.0),
        "AWR1843":  (76.0, 81.0),
        "AWR1843L": (76.0, 81.0),
        "AWR2944":  (76.0, 81.0),
        "AWRL6844": (57.0, 64.0),
    }

    @staticmethod
    def check(radar, capture=None):
        limits = FrequencyRange._LIMITS.get(radar.device_name)
        if limits is None:
            return ConstraintCheck(
                FrequencyRange, None,
                f"not checked for {radar.device_name}")
        min_freq, max_freq = limits
        start = radar.frequency
        end = radar.frequency + radar.bandwidth / 1000
        if start < min_freq:
            return ConstraintCheck(
                FrequencyRange, False,
                f"start frequency {start:.3f} GHz < device minimum {min_freq} GHz")
        if end > max_freq:
            return ConstraintCheck(
                FrequencyRange, False,
                f"end frequency {end:.3f} GHz > device maximum {max_freq} GHz")
        return ConstraintCheck(
            FrequencyRange, True,
            f"frequency range {start:.3f}–{end:.3f} GHz "
            f"(device band {min_freq}–{max_freq} GHz)")


class MaxBandwidth(Constraint):
    """Effective chirp bandwidth must not exceed the device RF limit.

    Bandwidth is computed as `freq_slope × T_s` (see
    [`XWRConfig.bandwidth`][xwr.config.XWRConfig]).

    | Device   | Maximum  |
    |----------|----------|
    | AWR1642  | 4000 MHz |
    | AWR1843  | 4000 MHz |
    | AWR2944  | 4000 MHz |
    """

    _LIMITS: ClassVar[dict[str, float]] = {
        "AWR1642":  4000.0,
        "AWR1843":  4000.0,
        "AWR1843L": 4000.0,
        "AWR2944":  4000.0,
    }

    @staticmethod
    def check(radar, capture=None):
        limit = MaxBandwidth._LIMITS.get(radar.device_name)
        if limit is None:
            return ConstraintCheck(
                MaxBandwidth, None,
                f"not checked for {radar.device_name}")
        passed = radar.bandwidth <= limit
        detail = f"bandwidth = {radar.bandwidth:.1f} MHz, maximum = {limit:.0f} MHz"
        if not passed:
            detail = (
                f"bandwidth = {radar.bandwidth:.1f} MHz "
                f"> device maximum {limit:.0f} MHz")
        return ConstraintCheck(MaxBandwidth, passed, detail)


class NetworkUtilization(Constraint):
    """Radar data throughput must not exceed 80% of capture card capacity.

    High utilization risks packet loss in the networking:

        radar.throughput / capture.throughput < 80%

    Skipped if no [`DCAConfig`][xwr.config.] is provided.
    """

    @staticmethod
    def check(radar, capture=None):
        if capture is None:
            return ConstraintCheck(
                NetworkUtilization, None, "no capture config provided")
        util = 100 * radar.throughput / capture.throughput
        passed = util < 80
        detail = (
            f"network utilization = {util:.1f}% "
            f"(radar {int(radar.throughput / 1e6)} Mbps "
            f"/ capture {int(capture.throughput / 1e6)} Mbps)")
        if not passed:
            detail += " (must be < 80%)"
        return ConstraintCheck(NetworkUtilization, passed, detail)


class ReceiveBuffer(Constraint):
    """OS receive buffer must hold at least two full radar frames.

    Since radar frames are transmitted by the capture card in consecutive
    bursts of packets, a buffer smaller than two frames risks dropping packets
    when the consumer falls momentarily behind:

        socket_buffer / frame_size ≥ 2

    Skipped if no [`DCAConfig`][xwr.config.] is provided.
    """

    @staticmethod
    def check(radar, capture=None):
        if capture is None:
            return ConstraintCheck(
                ReceiveBuffer, None, "no capture config provided")
        ratio = capture.socket_buffer / radar.frame_size
        passed = ratio > 2.0
        detail = (
            f"recv buffer = {capture.socket_buffer} bytes "
            f"= {ratio:.2f} frames (1 frame = {radar.frame_size} bytes)"
            + (" (must be > 2)" if not passed else ""))
        return ConstraintCheck(ReceiveBuffer, passed, detail)


def check_config(
    radar: XWRConfig,
    capture: DCAConfig | None = None,
    log: bool = True,
) -> list[ConstraintCheck]:
    """Run all constraints against a configuration.

    Args:
        radar: radar configuration.
        capture: capture card configuration; cross-config constraints are
            skipped if not provided.
        log: if `True`, log each result at INFO level (pass/skip) or WARNING
            level (fail) using the `xwr/constraints` logger.

    Returns:
        All constraint results, including passed and skipped checks.
    """
    CONSTRAINTS: list[type[Constraint]] = [
        FrameDutyCycle,
        RFDutyCycle,
        ExcessRampTime,
        CubeSizeLimit,
        FrameLengthPowerOfTwo,
        AdcSamplesPowerOfTwo,
        MaxSampleRate,
        MinSampleRate,
        FrequencyRange,
        MaxBandwidth,
        NetworkUtilization,
        ReceiveBuffer,
    ]
    results = [C.check(radar, capture) for C in CONSTRAINTS]
    if log:
        logger = logging.getLogger("xwr/constraints")
        for r in results:
            name = r.constraint.__name__
            if r.passed is False:
                logger.warning(f"Possibly invalid - {name}: {r.detail}")
            else:
                logger.info(
                    f"{'skipped' if r.passed is None else 'pass'}"
                    f" | {name}: {r.detail}")
    return results
