r"""AWR1843Boost Radar & DCA1000EVM Capture Card API.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    TI AWR1843Boost/DCA1000EVM Raw I/Q API


.. [R1] DCA1000EVM Data Capture Card User's Guide (Rev A)
    https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf?ts=1709104212742
.. [R2] `ReferenceCode/DCA1000/SourceCode` folder in the mmWave Studio install.
.. [R3] `packages/ti/demo/xwr18xx/mmw` folder in the mmWave SDK install.
.. [R4] mmWave SDK user guide, Table 1 (Page 19)
    https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf
.. [R5] mmWave Studio
    https://www.ti.com/tool/MMWAVE-STUDIO
.. [R6] AWR1843 Data Sheet
    https://www.ti.com/lit/ds/symlink/awr1843.pdf?ts=1708800208074
.. [R7] MMwave Radar Device ADC Raw Capture Data
    https://www.ti.com/lit/an/swra581b/swra581b.pdf?ts=1609161628089
.. [R8] TI mmWave Sensing Estimator
    https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.4.0/
"""  # noqa: D205

from . import awr_types, dca_types
from .awr_api import AWR1843
from .config import CaptureConfig, RadarConfig
from .dca_api import DCA1000EVM
from .system import AWRSystem

__all__ = [
    "AWRSystem", "RadarConfig", "CaptureConfig",
    "awr_types", "dca_types", "AWR1843", "DCA1000EVM"]
