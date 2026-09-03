"""Shared live-plotting helpers for xwr demos."""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class VisualizationConfig:
    """Range-doppler spectrum visualization settings.

    Attributes:
        pclip: percentile to clip the range-doppler spectrum to, i.e. the
            color scale spans the `(pclip, 100 - pclip)` percentile range
            instead of the min/max.
        power: power to raise the range-doppler spectrum to before plotting.
        azimuth: azimuth FFT resolution, i.e. number of azimuth bins.
        doppler: number of chirps to use for signal processing (if not already
            specified by the modulation, i.e., for continuous chirping).
    """

    pclip: float = 0.1
    power: float = 0.5
    azimuth: int = 128
    doppler: int = 128


class BEVProjector:
    """Resamples a (range, azimuth) polar spectrum onto a bird's-eye grid.

    To convert azimuth bin indices to angles, we use the property that the
    azimuth bin indices correspond to the sin of the angle; antenna spacing
    is left uncorrected (i.e. assumed to be a half-wavelength array).

    Args:
        n_range: number of range bins.
        n_azimuth: number of azimuth bins (FFT size).
        max_range: maximum range, in m.
        range_resolution: range resolution, in m/bin.
    """

    def __init__(
        self, n_range: int, n_azimuth: int, max_range: float,
        range_resolution: float
    ) -> None:
        self.n_y = 2 * n_range
        self.n_x = 4 * n_range

        angles = np.arcsin(np.clip(
            np.linspace(-1.0, 1.0, n_azimuth) / (2 * 0.5), -1.0, 1.0))

        # x is lateral, y is forward (boresight) range.
        x = np.linspace(-max_range, max_range, self.n_x)
        y = np.linspace(0, max_range, self.n_y)
        xx, yy = np.meshgrid(x, y)

        r_grid = np.sqrt(xx**2 + yy**2)
        theta_grid = np.arctan2(xx, yy)

        self.range_idx = np.clip(
            np.round(r_grid / range_resolution).astype(int), 0, n_range - 1)
        self.azimuth_idx = np.clip(
            np.round(np.interp(theta_grid, angles, np.arange(n_azimuth)))
            .astype(int),
            0, n_azimuth - 1)
        self.mask = r_grid <= max_range

    def project(self, ra: np.ndarray) -> np.ndarray:
        """Resample a (range, azimuth) spectrum onto the BEV (x, y) grid."""
        return np.where(
            self.mask, ra[self.range_idx, self.azimuth_idx], np.nan)


class RadarPlot:
    """Live range-doppler / bird's-eye-view range-azimuth radar plot.

    Args:
        n_range: number of range bins.
        n_doppler: number of doppler bins.
        n_azimuth: number of azimuth bins (FFT size).
        max_range: maximum range, in m.
        max_doppler: maximum doppler velocity, in m/s.
        range_resolution: range resolution, in m/bin.
        pclip: percentile to clip the range-doppler spectrum to, i.e. the
            color scale spans the `(pclip, 100 - pclip)` percentile range
            instead of the min/max.
        power: power to raise the range-doppler spectrum to before plotting.
    """

    def __init__(
        self, n_range: int, n_doppler: int, n_azimuth: int, max_range: float,
        max_doppler: float, range_resolution: float,
        pclip: float = 1.0, power: float = 1.0
    ) -> None:
        self.pclip = pclip
        self.power = power
        self.projector = BEVProjector(
            n_range, n_azimuth, max_range, range_resolution)

        plt.ion()
        self.fig, axs = plt.subplots(1, 2)

        # `animated=True` excludes these from normal full draws; they are
        # only ever rendered explicitly, via `draw_artist` in `_on_draw` and
        # `update` below (see the blitting notes there).
        self.im_rd = axs[0].imshow(
            np.zeros((n_range, n_doppler), dtype=np.float32),
            cmap="viridis", aspect='auto', origin='lower',
            extent=(-max_doppler, max_doppler, 0, max_range),
            animated=True)
        self.im_bev = axs[1].imshow(
            np.zeros((self.projector.n_y, self.projector.n_x),
                      dtype=np.float32),
            cmap="viridis", aspect='equal', origin='lower',
            extent=(-max_range, max_range, 0, max_range),
            animated=True)

        axs[0].set_xlabel("Doppler (m/s)")
        axs[0].set_ylabel("Range (m)")
        axs[1].set_xlabel("x (m)")
        axs[1].set_ylabel("y (m)")

        self.fps_text = self.fig.text(
            0.99, 0.01, "", ha="right", va="bottom", fontsize=12,
            color="white", bbox=dict(
                boxstyle="round", facecolor="black", alpha=0.5,
                linewidth=0),
            animated=True)
        self.fps_text.set_in_layout(False)

        self.fig.tight_layout()
        self.fig.canvas.mpl_connect(
            'resize_event', lambda event: self.fig.tight_layout())

        self._animated = (self.im_rd, self.im_bev, self.fps_text)
        self._bg = None
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        self.fig.canvas.draw()

    def _on_draw(self, event=None) -> None:
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for artist in self._animated:
            self.fig.draw_artist(artist)

    def update(self, dear: np.ndarray, fps: float | None = None) -> None:
        """Update the plot from a doppler-elevation-azimuth-range spectrum."""
        rd = np.sqrt(
            np.swapaxes(np.mean(dear**2, axis=(0, 2, 3)), 0, 1)
        ) ** self.power
        ra = np.sqrt(np.swapaxes(np.mean(dear**2, axis=(0, 1, 2)), 0, 1))
        bev = self.projector.project(ra)

        self.im_rd.set_data(rd)
        vmin, vmax = np.percentile(rd, [self.pclip, 100 - self.pclip])
        self.im_rd.set_clim(vmin=vmin, vmax=vmax)

        self.im_bev.set_data(bev)
        self.im_bev.set_clim(vmin=np.nanmin(bev), vmax=np.nanmax(bev))

        if fps is not None:
            self.fps_text.set_text(f"{fps:.2f} fps")

        if self._bg is None:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.restore_region(self._bg)
            for artist in self._animated:
                self.fig.draw_artist(artist)
            self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


class FrameRateLogger:
    """Tracks framerate as a moving average over the past `n` frames.

    Args:
        log: logger to log the framerate to, at `INFO` level.
        n: number of frames to average the framerate over.
        interval: minimum interval between log messages, in seconds.
    """

    def __init__(self, log, n: int = 10, interval: float = 5.0) -> None:
        self.log = log
        self.interval = interval
        self.times: deque[float] = deque(maxlen=n)
        self.last_log = time.perf_counter()

    @property
    def fps(self) -> float | None:
        """Current moving-average framerate, or `None` if not yet known."""
        if len(self.times) < 2:
            return None
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])

    def tick(self) -> None:
        """Record a frame, logging the framerate if `interval` has elapsed."""
        self.times.append(time.perf_counter())
        if self.times[-1] - self.last_log >= self.interval:
            fps = self.fps
            if fps is not None:
                self.log.info(f"Demo framerate: {fps:.2f}fps")
            self.last_log = self.times[-1]
