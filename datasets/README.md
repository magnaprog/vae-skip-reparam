# NOAA DART records

The National Oceanic and Atmospheric Administration's National Data Buoy Center provides these annual records through its [historical DART data](https://www.ndbc.noaa.gov/historical_data.shtml) service. DART stands for Deep-ocean Assessment and Reporting of Tsunamis.

| File | Station | Year |
| --- | --- | --- |
| `23461t2022.txt` | 23461 | 2022 |
| `44402t2016.txt` | 44402 | 2016 |
| `46407t2011.txt` | 46407 | 2011 |
| `52403t2020.txt` | 52403 | 2020 |

HEIGHT expresses bottom pressure as equivalent water column height in meters. Data preparation uses routine T=1 observations 15 minutes apart and windows of 192 consecutive observations. It preserves interruptions in the time series and estimates the normalization mean and standard deviation from the training data.

See NOAA's [DART description](https://www.ndbc.noaa.gov/dart/dart.shtml) and [measurement descriptions](https://www.ndbc.noaa.gov/measdes.shtml) for the instruments and data fields. These records provide observations without event labels for evaluating tsunami detection or a warning system.
