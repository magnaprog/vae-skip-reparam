# DART energy score results

The JSON and CSV files contain energy score and covariance comparisons for 84 DART fits: seven annual records, four model conditions and three training seeds per record.

The energy score is the mean distance from model samples to observations minus half the mean distance between independent model samples. Lower scores indicate better distributional agreement. The name comes from energy statistics and does not denote physical ocean energy. See Gneiting and Raftery, 2007, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).

| Files | Contents |
| --- | --- |
| `energy_analysis_results.json` | Results across records and fits |
| `reductions/energy_terms_*.csv` | Two distance terms and their combined score |
| `reductions/variance_changes_*.csv` | Changes after estimating observation variance |
| `reductions/skip_comparisons_*.csv` | Skip comparisons at the same observation variance |
| `reductions/ppca_*.csv` | Probabilistic principal component analysis (PPCA) scores and comparisons with neural models |
| `reductions/covariance_structure.csv` | Fractions of training covariance in the leading sixteen directions |
| `figures/table_figure_data.json` | Absolute neural and baseline scores, PPCA differences, and interval scores with coverage |

## Reading the values

Scores use dimensionless coordinates obtained by subtracting the training mean from HEIGHT and dividing by the training standard deviation. Multiplying a score by that standard deviation converts it to meters. In the CSV columns, `A` is the mean distance from model samples to observations, and `B` is half the mean distance between distinct model samples. These distance terms are separate from the model condition codes. Their difference is the energy score.

The `joint` evaluation uses full windows, `centered` subtracts each window's own mean, and `increment` uses time differences across several lags. The label `reset` denotes replacement of observation variance while keeping encoder and decoder weights the same.

Changes are computed within each sampling stream and then averaged within each training seed. Medians of the two distance terms need not subtract to the median energy score. PPCA and neural models use separate sampling streams, so their comparisons subtract stream means. Covariance fractions use the leading sixteen directions of the training covariance, with the window weights for each record.

Subtracting the second distance term from the first gives a proper score for distributions with finite first moments. Each distance term alone is insufficient for that assessment. Ranges across three training seeds describe variation among these fits. They are not population confidence intervals. Repeating model evaluation requires the prepared window arrays and generated samples, which are not included here.
