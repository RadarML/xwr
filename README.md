# `xwr`: Linux-Compatible Real-Time Raw Data Capture for TI mmWave Radars

[![pypi version](https://img.shields.io/pypi/v/xwr.svg)](https://pypi.org/project/xwr/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xwr)
![License - MIT](https://img.shields.io/badge/license-MIT-green)
![PyPI - Types](https://img.shields.io/pypi/types/xwr)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

`xwr` is a pure-python, linux-based real time raw data capture system for TI mmWave radars, and includes four key components:

- [`xwr`](https://radarml.github.io/xwr/system/): a high-level data capture interface
- [`xwr.rsp`](https://radarml.github.io/xwr/rsp/rsp/): a radar signal processing library with Numpy, Pytorch, and Jax support
- [`xwr.radar`](https://radarml.github.io/xwr/radar/api/): a parameterized python interface for the default radar firmware
- [`xwr.capture`](https://radarml.github.io/xwr/dca/api/): a pure-python, real-time interface for the DCA1000EVM

See our [documentation site](https://radarml.github.io/xwr/) for more details, setup guides, the included demo, and more!

## Requirements

`xwr` assumes a linux-based system and radar hardware which consists of the DCA1000EVM and a supported TI mmWave Radar (XWR) development board.

> [!IMPORTANT] 
> Supported Devices:
>    - AWR1843 Family: AWR1843Boost, AWR1843AOPEVM
>    - AWR1642
>
> WIP:
>    - AWR2544LOPEVM

## Install

The `xwr` library can be installed from pypi or github:

```sh
pip install xwr
# or
pip install git+ssh://github.com/RadarML/xwr.git
```

> [!WARNING]
> `xwr` does not include a copy of `torch` or `jax` by default! You must specify your own dependency and/or use the `xwr[torch]` and `xwr[jax]` extras if you intend to use these backends for the radar signal processing (`xwr.rsp`) submodule.
