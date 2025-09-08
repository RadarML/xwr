# `xwr`: Linux-Based Real-Time Raw Data Capture for TI mmWave Radars

![AWR1642Boost](images/awr1642boost.jpg){: style="width: 32%"}
![DCA1000EVM](images/dca1000evm.jpg){: style="width: 32%"}
![AWR1843Boost](images/awr1843aopevm-inset.jpg){: style="width: 32%"}

[![pypi version](https://img.shields.io/pypi/v/xwr.svg)](https://pypi.org/project/xwr/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xwr)
![License - MIT](https://img.shields.io/badge/license-MIT-green)
![PyPI - Types](https://img.shields.io/pypi/types/xwr)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

`xwr` is a pure-python, linux-based real time raw data capture system for TI mmWave radars, and includes four key components:

<div class="grid cards" markdown>

- [`xwr`](system.md): a high-level data capture interface
- [`xwr.rsp`](rsp/index.md): a radar signal processing library with [Numpy](rsp/numpy.md), [Pytorch](rsp/torch.md), and [Jax](rsp/jax.md) support
- [`xwr.radar`](radar/api.md): a parameterized python interface for the default radar firmware
- [`xwr.capture`](dca/api.md): a pure-python, real-time interface for the DCA1000EVM

</div>

!!! tip "Fully Typed"

    `xwr` is type-annotated where possible[^1], and has runtime type checking enabled via [beartype](https://beartype.readthedocs.io).

[^1]: You can check the type-completeness of `xwr` with `pyright ./src --verifytypes xwr --ignoreexternal`; the vast majority of remaining untyped (partially typed) code comes from numerical arrays, which currently cannot be statically type checked, beyond verifying their backend (e.g., `np.ndarray`, `torch.Tensor`, or `jax.Array`). As of time of writing, `xwr` nevertheless has a 90.6% type completeness score!

## Requirements

`xwr` assumes a linux-based system and radar hardware which consists of the DCA1000EVM and a supported TI mmWave Radar (XWR) development board.

!!! info "Supported Devices"

    <div class="grid cards" markdown>

    - [:material-arrow-right: DCA1000EVM Capture Card](https://www.ti.com/tool/DCA1000EVM)
    - [:material-arrow-right: AWR1843Boost](https://www.ti.com/tool/AWR1843BOOST)
    - [:material-arrow-right: AWR1843AOPEVM](https://www.ti.com/tool/AWR1843AOPEVM)
    - [:material-arrow-right: AWR1642Boost](https://www.ti.com/tool/AWR1642BOOST)
    - :construction_site: WIP: AWR2544LOPEVM

    </div>

!!! tip

    This list of supported radars is expanding, and we may add support for additional radars in the future! Feel free to leave an issue if you have a specific request.

## Install

The `xwr` library can be installed from [pypi](https://pypi.org/project/xwr/) or github:

=== "Direct Install"

    ```sh
    pip install xwr
    # or, for the bleeding-edge version:
    pip install git+ssh://github.com/RadarML/xwr.git
    ```

=== "Integrated Development"

    ```sh
    git clone git@github.com:RadarML/xwr.git
    pip install -e ./xwr
    ```

=== "Standalone Development"

    ```sh
    git clone git@github.com:RadarML/xwr.git
    cd xwr; uv sync --all-extras --frozen
    ```

=== "Using `uv` and `pyproject.toml`"

    ```toml
    [project]
    dependencies = ["xwr"]

    [tool.uv.sources]
    xwr = { git = "ssh://git@github.com/RadarML/xwr.git" }
    ```

See the [user guide](usage.md) and [hardware setup](setup.md) for instructions on how to configure and use `xwr`.

!!! warning

    `xwr` does not include a copy of `torch` or `jax` by default! You must specify your own dependency and/or use the `xwr[torch]` and `xwr[jax]` extras if you intend to use these backends for the radar signal processing (`xwr.rsp`) submodule.


## See Also

<div class="grid cards" markdown>

- :material-cube-outline: [`abstract_dataloader`](https://radarml.github.io/abstract-dataloader/)

    ---

    abstract interface for composable dataloaders and preprocessing pipelines

- :dart: [`dart`](https://wiselabcmu.github.io/dart/)

    ---

    *our prior work, DART: Implicit Doppler Tomography for Radar Novel View Synthesis*

- :material-video-wireless-outline: [`rover`](https://github.com/wiselabcmu/rover)

    ---

    *our previous data collection platform for radar time signal*

</div>

