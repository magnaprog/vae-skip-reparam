# DART energy-score results

The JSON summary and CSV files contain the energy-score and covariance comparisons for all 84 DART fits.

The energy score is the mean distance from model samples to observations minus half the mean distance between independent model samples. Lower scores indicate better distributional agreement. The name comes from energy statistics; it does not denote physical ocean energy. See Gneiting and Raftery, 2007, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).

| Files | Contents |
| --- | --- |
| `energy_analysis_results.json` | Results across records and fits |
| `reductions/energy_terms_*.csv` | Two distance terms and their combined score |
| `reductions/variance_changes_*.csv` | Changes after estimating observation variance |
| `reductions/skip_comparisons_*.csv` | Skip comparisons at the same observation variance |
| `reductions/ppca_*.csv` | Probabilistic principal component analysis (PPCA) scores and comparisons with neural models |
| `reductions/covariance_structure.csv` | Fractions of training covariance in the leading sixteen directions |
| `figures/table_figure_data.json` | Absolute neural and baseline scores, PPCA differences, and interval scores with coverage |

In the CSV columns, `A` is the mean model-to-data distance and `B` is half the mean distance between distinct model samples. These distance-term names are separate from model-condition codes. `joint` denotes full windows, `centered` denotes windows after subtraction of their own means, and `increment` denotes time differences. `reset` refers to replacing observation variance while keeping encoder and decoder weights the same.

Changes are computed within each sampling stream and then averaged within each training seed. Medians of the two terms need not subtract to the median energy score. PPCA and neural models use separate sampling streams, so comparisons use their stream means. Covariance fractions use the leading sixteen directions of the training covariance with the stated window weights.

The two distance terms help explain score changes, but neither term alone is a proper score. Variation across three training seeds describes these fits; it is not a population confidence interval. Recomputing the results requires the original prepared arrays and generated samples.
