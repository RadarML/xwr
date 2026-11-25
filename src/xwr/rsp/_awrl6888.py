"""Backend-independent AWRL6888EVM antenna geometry and calibration.

See [`xwr.rsp.numpy.AWRL6888EVM`][xwr.rsp.numpy.AWRL6888EVM] for details.
"""

import numpy as np

# Azimuth offset of each TX; RX bank A/B are at even/odd azimuth offsets.
_TX_AZ = np.array([1, 0, 1, 0, 1, 0, 1, 0])
_SLOT = np.arange(16)[:, None]

# Elevation and azimuth index of each (chirp slot, RX chain); shape (16, 4).
# These follow the TI `antGeometryTX` / `antGeometryRX` layout, except that
# elevation is reversed so that the output is in image order (elevation index
# 0 is up); the directions were verified on hardware.
EL = np.broadcast_to(7 - _SLOT // 2, (16, 4)).copy()
AZ = _TX_AZ[_SLOT // 2] + 2 * np.arange(4)[None, :] + _SLOT % 2
# Virtual array shape (elevation, azimuth).
SHAPE = (8, 9)

# TI `compRangeBiasAndRxChanPhase` for the AWRL6888EVM, excluding the
# leading range bias, as (Re, Im) per virtual antenna in TX-major, RX-minor
# order (= chirp slot-major, RX chain-minor order). Source:
# `mmwave_l_sdk_{version}/examples/mmw_demo/mmwave_demo_L688x/profiles/
# profile_8T8R_tdm.cfg`.
_CALIBRATION_RE_IM = np.array([float(v) for v in (
    "0.12087 0.19660 -0.10342 0.21609 -0.42792 -0.29734 -0.53109 "
    "-0.09694 -0.07475 -0.27053 0.13933 -0.21205 0.39096 0.42558 0.80347 "
    "0.43634 -0.09095 0.25783 -0.28356 0.06454 0.08816 -0.53468 -0.14678 "
    "-0.55148 0.20000 -0.27487 0.30910 -0.02338 -0.21630 0.55871 -0.08312 "
    "0.82233 0.21498 -0.11059 0.21324 0.12333 -0.55400 0.29647 -0.35827 "
    "0.49100 -0.28427 0.05688 -0.20338 -0.16217 0.62439 -0.18612 0.82026 "
    "-0.38472 0.21541 0.14833 -0.01147 0.26646 -0.59729 -0.34488 -0.66099 "
    "-0.04423 -0.19447 -0.24246 0.05524 -0.27635 0.51272 0.47706 0.84307 "
    "0.53781 0.12349 0.25251 -0.14238 0.24885 -0.14851 -0.47495 -0.37736 "
    "-0.36972 -0.06873 -0.33379 0.18454 -0.22966 0.08840 0.52451 0.25120 "
    "0.59857 -0.10961 0.22560 -0.25738 0.02878 0.19583 -0.49201 -0.04146 "
    "-0.53768 0.20457 -0.22605 0.26833 0.00783 -0.29264 0.48573 -0.20415 "
    "0.69670 0.24238 -0.09038 0.21091 0.15542 -0.41290 0.27007 -0.25526 "
    "0.42491 -0.31011 0.02178 -0.19037 -0.19111 0.48299 -0.22925 0.49817 "
    "-0.44428 0.22775 0.10133 0.04338 0.25819 -0.35094 -0.17731 -0.41329 "
    "0.00960 -0.23911 -0.19779 -0.00323 -0.26883 0.36584 0.25458 0.49123 "
    "0.14393").split()]).reshape(64, 2)

# Unit-magnitude phase correction, shape (chirp slot, RX chain).
CALIBRATION = np.exp(1j * np.arctan2(
    _CALIBRATION_RE_IM[:, 1], _CALIBRATION_RE_IM[:, 0])
).reshape(16, 4).astype(np.complex64)
