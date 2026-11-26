# NOAA DART records

These annual records come from the [historical DART data](https://www.ndbc.noaa.gov/historical_data.shtml) maintained by the National Oceanic and Atmospheric Administration's National Data Buoy Center. DART stands for Deep-ocean Assessment and Reporting of Tsunamis.

| File | Station | Year |
| --- | --- | --- |
| [23461t2022.txt](23461t2022.txt) | 23461 | 2022 |
| [44402t2016.txt](44402t2016.txt) | 44402 | 2016 |
| [46407t2011.txt](46407t2011.txt) | 46407 | 2011 |
| [52403t2020.txt](52403t2020.txt) | 52403 | 2020 |

HEIGHT expresses bottom pressure as equivalent water column height in meters. Data preparation forms windows of 192 consecutive routine T=1 observations, each 15 minutes apart. Each window stays within an uninterrupted segment of the time series. The normalization mean and standard deviation are estimated from the training data.

NOAA's [DART description](https://www.ndbc.noaa.gov/dart/dart.shtml) and [measurement descriptions](https://www.ndbc.noaa.gov/measdes.shtml) explain the instruments and data fields. These records do not include event labels for evaluating tsunami detection or a warning system.

## Station 21415 follow-up

[21415t2018.txt](21415t2018.txt) is the decompressed NDBC annual file downloaded on November 26, 2026. The record begins January 1, 2018 UTC. Preparation uses the interval ending January 1, 2019, exclusive. [NOAA annual download](https://www.ndbc.noaa.gov/data/historical/dart/21415t2018.txt.gz)

| File | SHA-256 |
| --- | --- |
| `21415t2018.txt` | `bb888a1856aab261f9da184cfe5552a253808ab6124f37a6b5a68d5105127678` |
| [dart_21415_2018.npz](dart_21415_2018.npz) | `b1a5f4f8aa0fd6188e493cf4298c53ac5ae321ed70a337b5c6a4e0d5adb1945e` |

The NPZ preserves the arrays used for training and scoring. To check them against the raw observations, call `prepare_dart_record("data/21415t2018.txt", "2018-01-01", "2019-01-01")` in [data_preparation.py](../src/data_preparation.py). The public routine returns the windows, normalization, weights, and starts. The archive also carries three reassignment fields from the earlier study. The optimizer experiment reads only the training and test windows and their weights. NumPy 2.4.4 reproduced all 15 returned arrays exactly on the verification host. NumPy 1.26.4 can change the last bits of the normalization and window values. Run the experiment with the supplied NPZ.
