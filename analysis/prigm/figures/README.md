# Figure inputs

The renderer draws ten figures from the supplied arrays, score summaries and four annual DART records. Run it from the repository root:

```bash
python analysis/prigm/figures/render_figures.py --output-dir runs/figures
```

The PDFs use DejaVu Serif and STIX math at a width of 5.5 inches. Labels are 9 pt; ticks and legends are 8.5 pt. The dependency versions in `requirements.txt` preserve the fonts and plotting behavior used for the paper.

Conditions appear in this order: no skip at base variance, skip at base variance, no skip at high variance, and skip at high variance. This puts each skip comparison beside its baseline at the same observation variance.

## Values and units

| Input | Meaning |
| --- | --- |
| `plot_data.npz` | Feature values, weights and spectral powers |
| `plot_data_manifest.json` | Array shapes, file identities and fit/stream indexing |
| `plot_summary.json` | Score differences, variance comparisons and record information |
| `table_figure_data.json` | Comparisons with PPCA, absolute scores and scores within time intervals |
| `datasets/*.txt` | NOAA annual observations for the context figure |

Each feature array contains window mean minus training mean, standard deviation within a window, and the norm of time increments, all in meters. Spectral arrays contain 96 bin powers in square meters at positive frequencies from 0.5 to 48 cycles per day. The arrays are already in physical units.

The original records use equal window weights. The 2018 records use weights for Coordinated Universal Time (UTC) start hours. Generated windows have equal weights within each sample set.

For each fit, curves average three sampling streams. Neural curves then show the median and full range across three training seeds. Probabilistic principal component analysis curves average streams from one fit. Empirical distribution functions retain ties and the zero left limit. Spectral powers are averaged before plotting on logarithmic axes. Dashed neural spectra show decoder means; solid curves include observation noise.

The plotting arrays support these figures. They do not contain the complete generated windows or network weights needed to repeat model evaluation.

The PPCA comparison shows all three neural fits at each skip and variance setting. Each point subtracts the PPCA stream mean at the same observation variance from one neural fit's stream mean. The PPCA and neural streams are separate. Absolute scores for every neural fit and all five baseline settings remain in `table_figure_data.json`, including the diagonal Gaussian and PPCA with variance estimated from the training covariance.

The interval figures show score differences for full windows within each training seed, together with window counts and total test weight. Intervals of 14 and 28 days begin at the start of the first eligible training window. Their endpoints are stored as Unix seconds. A window belongs to the interval containing its start. Multiplying each interval difference by its weight and summing recovers the difference over the full test partition. Windows can overlap, and the two interval lengths reuse the same observations.
