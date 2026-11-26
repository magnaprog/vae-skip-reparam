# Figure inputs

Render all ten figures from the supplied results and observations. Run this command from the repository root:

```bash
python figures/render.py --output-dir runs/figures
```

Each PDF is 5.5 inches wide, with DejaVu Serif text and STIX math. Labels are 9 pt; ticks and legends are 8.5 pt. Use the versions in [requirements.txt](../requirements.txt) to reproduce the paper's fonts and plotting behavior.

The four-condition legends use blue upward triangles and pink squares for the models without and with a skip connection at base variance. At high variance, the model without a skip connection uses orange downward triangles and the model with a skip connection uses teal circles. All four colors come from Paul Tol’s Vibrant scheme. [Tol’s color schemes](https://sronpersonalpages.nl/~pault/#sec:qualitative)

## Values and units

| Input | Meaning |
| --- | --- |
| [data.npz](data.npz) | Feature values, weights and spectral powers |
| [data_manifest.json](data_manifest.json) | Array shapes, file and array hashes, and indexing by fit and sampling stream |
| [summary.json](summary.json) | Score differences, variance comparisons and record information |
| [comparisons.json](comparisons.json) | Comparisons with PPCA, absolute scores and scores within time intervals |
| [NOAA records](../data/README.md) | Annual observations for the context figure |

The feature arrays contain three quantities in meters: window mean minus training mean, standard deviation within a window, and the norm of time increments. The increment feature is the Euclidean norm of the concatenated differences at lags 1, 4, 16, 48 and 96 observations, with each lag's differences divided by $\sqrt{192-\ell}$ for lag $\ell$. Spectral arrays contain power in 96 frequency bins, in square meters, at positive frequencies from 0.5 to 48 cycles per day. Energy scores use dimensionless coordinates, as described in the [results guide](../results/README.md).

The four original station-years use equal window weights. Figure 2 shows routine observations (T=1) in blue. The gap label at station 52403 gives the complete UTC dates without a routine observation.

The three 2018 records give equal total weight to each of six Coordinated Universal Time (UTC) start hours. Training and test weights are computed separately. Generated windows have equal weights within each sample set.

## Curves and comparisons

In the skip comparison, each training seed contributes one score difference, averaged over three common sampling streams. The symbol marks the median of those three differences. The capped line directly below it runs from the smallest to the largest seed difference.

Curves average three sampling streams within each fit. Neural curves show the median and full range across three training seeds at each feature value or frequency. The probabilistic principal component analysis (PPCA) curves average streams from one fit whose noise variance is estimated from the training covariance.

Empirical distribution functions retain ties and the zero left limit. In the window standard deviation column, the horizontal line at probability 0.5 locates the median. On the ECDFs and spectra, shading spans the lowest and highest values across training seeds. Symbols identify the four model conditions on their median curves.

The spectra compare neural samples with observation noise (left) and decoder means at prior latent samples (right). Each column shows all four neural conditions and repeats the same training, test, and PPCA spectra. The two columns for each station share frequency and power limits. Powers are averaged in linear units before plotting on logarithmic axes.

The PPCA comparison shows all three neural fits at each skip and variance setting. Each horizontal bar runs from zero to the center of a model symbol. Its length gives the neural mean score across streams minus the PPCA mean at the same observation variance. PPCA and neural models use separate sampling streams. Absolute scores in [comparisons.json](comparisons.json) cover every neural fit and all five baseline settings, including diagonal Gaussian baselines and PPCA with variance estimated from the training covariance.

In the variance replacement figure, symbols at left give the replacement variance divided by the original variance. In each Change panel, a vertical tick marks one fit’s mean score change across three common sampling streams. The horizontal segment runs from the smallest to the largest change. In the adjacent Range panel, each fit has the same symbol as in the variance ratio panel. Its position gives the largest change minus the smallest on a logarithmic scale. Both panels use units of $10^{-3}$. The tables give the mean and endpoints for each fit and the changes in each stream. [Fit summaries](../results/variance_changes_fits.csv) · [Stream values](../results/variance_changes_streams.csv)

The interval figures show energy score differences for full windows, separately for each training seed, along with window counts and each interval's share of total test weight. At each date, marker pairs run left to right in seed order (42, 1729, and 31415), with base variance filled and high variance open. Intervals of 14 and 28 days are aligned to the start of the first eligible training window. Their endpoints are stored as Unix seconds. Each window belongs to the interval containing its start. Weighting the interval differences by those shares and summing recovers the difference over the full test partition. Windows can overlap, and both interval lengths use the same observations.

To repeat model evaluation, you need complete generated windows and network weights, which are outside this release.
