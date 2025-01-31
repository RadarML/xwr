# AWR API

Standalone linux-based implementation for real time raw data capture from the TI AWR1843 & DCA1000EVM, using minimal dependencies (numpy + pyserial + type checking).

To set up:

1. Purchase the [TI AWR1843Boost](https://www.ti.com/tool/AWR1843BOOST) radar development board and the [TI DCA1000EVM](https://www.ti.com/tool/DCA1000EVM) capture card.

2. Configure the radar and capture card using the [setup instructions](https://wiselabcmu.github.io/RadarML/awr_api/setup.html).

3. Install the API:

    ```
    pip install git+ssh://github.com/WiseLabCMU/awr-api.git
    # or, if you are developing:
    git clone git@github.com:WiseLabCMU/awr-api.git
    pip install -e ./awr-api
    ```

For install, setup, and usage information, see the CMU RadarML [documentation site](https://wiselabcmu.github.io/RadarML/awr_api/index.html).
