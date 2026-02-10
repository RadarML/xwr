# Custom Firmware

While `xwr` generally works with the default TI Demo firmware, we are working on custom firmware versions which remove unnecessary bloat and disable software limitations.

This firmware code is based on the TI mmWave demo, and is distributed via a [separate repository](github.com/RadarML/firmware). We currently have the following versions:

| Version | Devices | Description |
| ------- | ------- |  ---------- |
| `xwr18xx` | AWR1843, AWR1843AOP | Demo with most point cloud processing features removed. The relevant CLI calls are stubbed out. Note that this also enables continuous chirping, which is not possible with the stock firmware. |
| :construction_site: `xwr18xx-custom` | AWR1843, AWR1843AOP | Work-in-progress. The CLI is completely removed and replaced with a custom interface. |

## Setup

Prerequisites:

- [Mono](https://www.mono-project.com/): run .NET executable files on linux
    ```sh
    sudo apt-get install mono-complete -y
    ```
- 32-bit libraries (required for mmWave SDK):
    ```sh
    sudo dpkg --add-architecture i386
    sudo apt-get update
    sudo apt-get install libc6:i386 libstdc++6:i386 -y
    ```

Install the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK#downloads).

```sh
wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.02.00-LTS/mmwave_sdk_03_06_02_00-LTS-Linux-x86-Install.bin
chmod +x mmwave_sdk_03_06_02_00-LTS-Linux-x86-Install.bin
./mmwave_sdk_03_06_02_00-LTS-Linux-x86-Install.bin
```

!!! info

    This guide was developed and tested with version `03_06_02_00`; other versions should also work using the same instructions, but are untested.

## Compile

Each time you compile, you will need to set the mmWave SDK environment variables:
```sh
cd ~/ti/mmwave_sdk_03_06_02_00-LTS/packages/scripts/unix
source ./setenv.sh
```

- This should give you `$MMW_SDK_INSTALL_PATH`

!!! bug

    You must set the working directory to run `source setenv.sh` since the TI script does not properly handle relative paths. `source`-ing the script directly from another directory will not work!


You can then build the firmware:

!!! info

    These defines are documented in this comment in `$(MMWAVE_SDK_INSTALL_PATH)/ti/common/mmwave_sdk.mak`:
    ```
    # Platform specific definitions:
    # Legend:
    # MMWAVE_SDK_DEVICE     : awr14xx, awr16xx, awr18xx, awr68xx, iwr14xx, iwr16xx, iwr18xx, iwr68xx
    # MMWAVE_SDK_DEVICE_TYPE: xwr14xx (common for awr14xx, iwr14xx),
    #                         xwr16xx (common for awr16xx, iwr16xx),
    #                         xwr18xx (common for awr18xx, iwr18xx),
    #                         xwr68xx (common for awr68xx, iwr68xx)
    # PLATFORM_DEFINE       : SOC_XWR14XX, SOC_XWR16XX, SOC_XWR18XX, SOC_XWR68XX
    # BOARD_DEFINE		: ISK, ODS, AOP, DEFAULT_ANT_DESIGN, NO_ANTENNA_CORRECTION
    # XDC_PLATFORM_DEFINE   : AWR14XX, AWR16XX, AWR18XX, AWR68XX, IWR14XX, IWR16XX, IWR18XX, IWR68XX
    ```

```sh
export MMWAVE_SDK_DEVICE=awr18xx;
export MMWAVE_SDK_DEVICE_TYPE=xwr18xx;
export PLATFORM_DEFINE=SOC_XWR18XX;
export BOARD_DEVICE=AOP;
export XDC_PLATFORM_DEFINE=AWR18XX
make all
```

After building, flash `xwr...demo.bin` to your radar; see the [hardware setup instructions](../setup.md).
