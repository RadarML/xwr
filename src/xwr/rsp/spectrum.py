"""Backend-agnostic detector (CFAR) base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, cast

import numpy as np
from jaxtyping import Bool, Float

from .generic import TArray

SignalCube = (
    Float[TArray, "batch doppler tx rx range"]
    | Float[TArray, "batch doppler range"]
    | Float[TArray, "batch doppler el az range"])
"""Accepted detector input. A virtual array cube, an angle spectrum, or
    an already-combined range-doppler image.
"""


@dataclass
class Detection(Generic[TArray]):
    """Detected objects, and the statistics they were detected from.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Attributes:
        mask: cfar detected object mask.
        signal: non-coherently integrated power across the channel axes, i.e.
            the range-doppler spectrum used for detection.
        snr: signal to noise ratio, as a linear power ratio.
    """

    mask: Bool[TArray, "batch range doppler"]
    signal: Float[TArray, "batch range doppler"]
    snr: Float[TArray, "batch range doppler"]


class Detector(ABC, Generic[TArray]):
    """Abstract, backend-agnostic constant false alarm rate detector.

    This class documents the interface shared by every CFAR variant: given a
    batch of post range-doppler FFT radar cubes, report which range-doppler
    cells hold a target. Concrete detectors differ in how they estimate the
    noise floor.

    The following implementations are available:

    | Detector | Noise estimate |
    |----------|----------------|
    | [`CFAR`][xwr.rsp.] | 2D cell-averaging ring |
    | [`CFARCASO`][xwr.rsp.] | Separate "smallest of" tests on the range and doppler axes |

    Implementation notes:

    - Every variant takes a batch of range-doppler cubes and is parameterized
        the same way, with the `guard` and `train` sizes corresponding to the
        (range, doppler) axes and counted on **each** side of the cell under
        test. `guard`, `train`, and `discard_range` must all be `>= 0`; a
        negative value raises `ValueError`.
    - The input may be a virtual array cube
        (`batch doppler tx rx range`), an angle spectrum
        (`batch doppler el az range`), or an already-combined range-doppler
        image (`batch doppler range`). Whatever sits between the doppler and
        range axes is flattened into a single channel axis and integrated
        non-coherently, so a range-doppler image is simply treated as having
        one channel.
    - Guard cells sit between the cell under test and the training cells and
        are excluded from the noise estimate, absorbing target energy spread
        by FFT sidelobes and windowing which would otherwise raise the
        target's own noise floor. Training cells sit outside them and form
        the estimate.
    - The closest range bins are dominated by TX to RX leakage and DC, and
        the furthest are past useful SNR. Bins inside `discard_range` are
        forced to non-detect and assigned unit noise.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Args:
        guard: guard cells on each side of the cell under test, for
            (range, doppler).
        train: training cells on each side of the guard region, for
            (range, doppler).
        discard_range: range bins (close, far) to discard around DC.
    """

    def __init__(
        self,
        guard: tuple[int, int],
        train: tuple[int, int],
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        if any(g < 0 for g in guard):
            raise ValueError(f"Guard {guard} must be >= 0 on each axis.")
        if any(t < 0 for t in train):
            raise ValueError(f"Train {train} must be >= 0 on each axis.")
        if any(d < 0 for d in discard_range):
            raise ValueError(
                f"Discard range {discard_range} must be >= 0 on each side.")

        self.guard = guard
        self.train = train
        self.discard_r = discard_range

    @abstractmethod
    def _cfar(
        self, signal_cube: Float[TArray, "batch doppler channel range"]
    ) -> Detection[TArray]:
        """Run this detector on a batch of radar cubes.

        Args:
            signal_cube: batch of post range doppler FFT radar cubes in
                amplitude, with the channel axes already flattened into a
                single axis.

        Returns:
            The same detections as [`__call__`][xwr.rsp.Detector.__call__].
        """
        ...

    def _flatten(
        self, signal_cube: SignalCube
    ) -> Float[TArray, "batch doppler channel range"]:
        """Collapse any axes between doppler and range into a channel axis.

        Args:
            signal_cube: batch of post range doppler FFT radar cubes in
                amplitude.

        Returns:
            The same values, with a single channel axis; a range-doppler
                image gains a channel axis of length 1.
        """
        batch, doppler, rng = (
            signal_cube.shape[0], signal_cube.shape[1], signal_cube.shape[-1])
        return cast(Any, signal_cube).reshape((batch, doppler, -1, rng))

    def __call__(self, signal_cube: SignalCube) -> Detection[TArray]:
        """Run 2D CFAR detection.

        !!! note

            The channel axes (the transmit and receive antennas of the
            virtual array, or the elevation and azimuth bins of an angle
            spectrum) are combined non-coherently, so their relative order
            does not matter.

        Args:
            signal_cube: batch of post range doppler FFT radar cubes in
                amplitude, or a range-doppler image which is already combined
                across the virtual array.

        Returns:
            The detection mask, the range-doppler spectrum it was computed
                from, and the signal to noise ratio; see
                [`Detection`][xwr.rsp.].
        """
        return self._cfar(self._flatten(signal_cube))


class CFAR(Detector[TArray], ABC):
    """Cell-averaging CFAR.

    ```
        ┌─────────────────┐ ▲ guard[0]+train[0]
        │    ┌───────┐    │ │
        │    │  ┌─┐  │    │ ▼
        │    │  └─┘  │    │ ▲ guard[0]
        │    └───────┘    │ ▼
        └─────────────────┘
    guard[1] ◄──► ◄───────► guard[1]+train[1]
    ```

    Implementation notes:

    - The noise floor is the mean of the training cells in the 2D ring, so
        the cell under test is tested once. The window half-width on each
        axis is `guard + train`; `train` may be `0` on one axis, training on
        the other axis only, but `0` on both leaves an empty ring and raises
        `ValueError`.
    - A cell is detected when its integrated power exceeds
        `snr_thresh * noise`. Raising the threshold gives fewer false alarms
        and a lower probability of detection.

    See [`Detector`][xwr.rsp.] for the shared `guard`, `train`, and
    `discard_range` semantics.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Args:
        guard: guard cells on each side of the cell under test, for
            (range, doppler).
        train: training cells on each side of the guard region, for
            (range, doppler).
        snr_thresh: detection threshold, as a **linear power ratio** (not dB).
        discard_range: range bins (close, far) to discard around DC.
    """

    def __init__(
        self,
        guard: tuple[int, int] = (2, 2),
        train: tuple[int, int] = (2, 2),
        snr_thresh: float = 5.0,
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        super().__init__(guard, train, discard_range)
        self.snr_thresh = snr_thresh

        g0, g1 = guard
        w0, w1 = g0 + train[0], g1 + train[1]

        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0 : w0 + g0 + 1, w1 - g1 : w1 + g1 + 1] = 0.0
        if mask.sum() == 0:
            raise ValueError(
                f"CFAR mask is empty; check guard={guard} and train={train}.")
        self.mask: Float[TArray, "kr kd"] = self._asarray(mask)

    @staticmethod
    @abstractmethod
    def _asarray(x: Float[np.ndarray, "..."]) -> Float[TArray, "..."]:
        """Convert a numpy array to this backend's array type.

        Args:
            x: numpy array to convert.

        Returns:
            The same values, as a backend array.
        """
        ...


class CFARCASO(Detector[TArray], ABC):
    """Cell-averaging Smallest of CFAR.

    !!! info

        Instead of the 2D kernel used in [`CFAR`][xwr.rsp.], CASO uses a
        separate 1D kernel for the range and doppler axes, and reports a
        detection only where **both** axes fire.

    ```
               ┌─┐       ▲ train[0]
               │ │       │
               ├─┤       ▼
         ┌───┬─┼─┼─┬───┐
         └───┴─┼─┼─┴───┘ ▲ guard[0]
               ├─┤       ▼
               │ │
               └─┘
    guard[1] ◄─►   ◄───► train[1]
    ```

    Implementation notes:

    - Rather than averaging both sides of the cell under test, CASO
        ("smallest of") takes the **minimum** of the two one-sided means, so
        a strong target on one side cannot inflate the noise floor and mask a
        weaker target on the other. `train` must be `>= 1` on both axes,
        since each one-sided mean divides by it, and the asymmetric default
        `guard` reflects that range leakage is broad while doppler needs no
        guard.
    - An axis fires where its integrated power exceeds that axis's
        `snr_thresh * noise`. Raising a threshold gives fewer false alarms
        and a lower probability of detection.

    See [`Detector`][xwr.rsp.] for the shared `guard`, `train`, and
    `discard_range` semantics.

    Type Parameters:
        - `TArray`: Generic backend, e.g., `np.ndarray`, jax `jax.Array`, or
            torch `Tensor`.

    Args:
        guard: guard cells on each side of the cell under test, for
            (range, doppler).
        train: training cells on each side of the guard region, for
            (range, doppler).
        snr_thresh: detection threshold for (range, doppler), as a **linear
            power ratio** (not dB).
        discard_range: range bins (close, far) to discard around DC.
    """

    def __init__(
        self,
        guard: tuple[int, int] = (8, 0),
        train: tuple[int, int] = (8, 4),
        snr_thresh: tuple[float, float] = (5.0, 3.0),
        discard_range: tuple[int, int] = (10, 20),
    ) -> None:
        super().__init__(guard, train, discard_range)
        if any(t < 1 for t in train):
            raise ValueError(f"Train {train} must be >= 1 on each axis.")

        self.snr_r, self.snr_d = snr_thresh

        self.train_r, self.train_d = train
        self.pad_r = train[0] + guard[0]
        self.pad_d = train[1] + guard[1]
