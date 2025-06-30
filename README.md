# `xwr`: Linux-Compatible Real-Time Raw Data Capture for TI mmWave Radars

![License - MIT](https://img.shields.io/badge/license-MIT-green)
![Typed](https://img.shields.io/badge/types-typed-brightgreen)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

`xwr` is a pure-python, linux-based real time raw data capture system for TI mmWave radars, and includes four key components:

- [`xwr`](https://wiselabcmu.github.io/xwr/system/): a high-level data capture interface
- [`xwr.rsp`](https://wiselabcmu.github.io/xwr/rsp/rsp/): a radar signal processing library with Numpy, Pytorch, and Jax support
- [`xwr.radar`](https://wiselabcmu.github.io/xwr/radar/api/): a parameterized python interface for the default radar firmware
- [`xwr.capture`](https://wiselabcmu.github.io/xwr/dca/api/): a pure-python, real-time interface for the DCA1000EVM

See our [documentation site](https://wiselabcmu.github.io/xwr/) for more details, setup guides, the included demo, and more!

## Requirements

`xwr` assumes a linux-based system and radar hardware which consists of the DCA1000EVM and a supported TI mmWave Radar (XWR) development board.

> [!IMPORTANT] 
> Supported Devices:
>    - AWR1843 Family: AWR1843Boost, AWR1843AOPEVM
>    - AWR1642
> WIP:
>    - AWR2544LOPEVM

## Install

The `xwr` library is currently only distributed via github:

```sh
# Install directly from github
pip install git+ssh://github.com/WiseLabCMU/xwr.git
```
