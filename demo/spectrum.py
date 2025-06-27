"""Simple range-doppler visualization demo."""

import logging

import numpy as np
import yaml
from matplotlib import pyplot as plt

import xwr
from xwr.rsp import numpy as xwr_rsp

logging.basicConfig(level=logging.INFO)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

awr = xwr.XWRSystem(**cfg, module="AWR1843")
rsp = xwr_rsp.AWR1843AOP(window=False)

# Create a figure
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((128, 128)), cmap="viridis", aspect='auto')
ax.set_xlabel("Doppler")
ax.set_ylabel("Range")

for frame in awr.dstream(numpy=True):
    rd_cplx = rsp.doppler_range(xwr.rsp.iq_from_iiqq(frame)[None, ...])
    rd_real = np.mean(np.abs(rd_cplx), axis=(0, 2, 3))
    rd_real = np.swapaxes(rd_real, 0, 1)

    im.set_data(rd_real)
    im.set_clim(vmin=np.min(rd_real), vmax=np.max(rd_real))
    plt.pause(0.001)

awr.stop()
