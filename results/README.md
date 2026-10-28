# DART energy score results

The JSON and CSV files contain energy score and covariance comparisons for 84 DART fits: seven annual records, each with four model conditions and three training seeds per condition.

The energy score is the mean distance from model samples to observations minus half the mean distance between independent model samples. Both distance terms are needed to obtain a proper score for distributions with finite first moments. Lower scores indicate better distributional agreement. The name comes from energy statistics and does not denote physical ocean energy. See Gneiting and Raftery, 2007, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).

| Files | Contents |
| --- | --- |
| [energy_scores.json](energy_scores.json) | Results across records and fits |
| `energy_terms_*.csv` | Two distance terms and their combined score |
| `variance_changes_*.csv` | Score changes after replacing observation variance with an estimate from training data |
| `skip_comparisons_*.csv` | Skip comparisons at the same observation variance |
| `ppca_*.csv` | Probabilistic principal component analysis (PPCA) scores and comparisons with neural models |
| [covariance_structure.csv](covariance_structure.csv) | Contributions from the first sixteen principal components of the training windows |
| [comparisons.json](../figures/comparisons.json) | Absolute neural and baseline scores, PPCA differences, and interval scores with coverage |

## Reading the values

Scores use dimensionless coordinates obtained by subtracting the training mean from HEIGHT and dividing by the training standard deviation. Multiplying a score by that standard deviation converts it to meters. In the CSV columns, `observation_distance` is the weighted mean distance from model samples to observations, and `half_sample_distance` is half the mean distance between distinct model samples. The `energy_score` column is `observation_distance - half_sample_distance`.

The model conditions are `no_skip_base`, `skip_base`, `no_skip_high` and `skip_high`, in that order. Base and high denote observation variance, as defined in [Model training](../TRAINING.md#gaussian-and-dart-experiments). Skip comparisons subtract `no_skip_base` from `skip_base` and `no_skip_high` from `skip_high`.

The `full_window` evaluation scores the complete vector of observations in each window. The `without_window_mean` evaluation subtracts each generated or observed window's own mean before scoring, and `time_differences` uses time differences across several lags. Score changes for `variance_refit` subtract the original score from the score after replacing observation variance, using the same encoder and decoder weights.

Skip and variance changes are computed within each sampling stream, then averaged within each training seed. Subtracting the medians of the two distance terms need not give the median energy score. PPCA and neural models use separate sampling streams, so their comparisons subtract the means across streams.

Covariance fractions measure how much variance in each evaluation's coordinates comes from the first sixteen principal components of the original weighted training covariance. The same components are transformed for each evaluation.

Columns ending in `stream_min` and `stream_max` span the three sampling streams within one fit. Columns ending in `seed_min` and `seed_max` span the three training fits after averaging their sampling streams. In `energy_scores.json`, each summary gives its number of fits in `count`; its range spans those fits. These ranges describe variation in the available calculations. Population confidence intervals require a separate uncertainty analysis.

Repeating model evaluation requires prepared window arrays and generated samples that are outside this release.
