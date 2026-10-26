# Figure inputs and rendering

The bundled cache reproduces the seven paper figures on CPU. It contains retained feature values, weights and spectral powers; it omits complete generated windows and checkpoints. Rendering uses the saved plotting arrays, numerical summaries and annual records.

`plot_data.npz` supplies ECDF and spectral arrays. `plot_data_manifest.json` records their shapes, hashes and fit/stream index. The comparison and variance figures read numerical summaries from [the scientific manifest](../../../paper/prigm/paper_evidence_manifest.json). The context figure reads the four annual records in [datasets](../../../datasets/).

## Render

Install the root [CPU dependencies](../../../requirements-release.txt), then run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python \
  analysis/prigm/figures/render_figures.py \
  --output-dir runs/figures-preview
```

This writes seven PDFs to the preview directory. The renderer uses Matplotlib 3.10.9 with its bundled DejaVu Serif 2.35 and STIX fonts and checks the regular DejaVu font's hash. Text is 9 pt and ticks/legends are 8.5 pt on canvases 5.5 inches wide. The paper includes them at native width; tight bounding-box cropping would change the printed scale.

## Array meanings

| Suffix | Values and units |
| --- | --- |
| `features` | Window mean minus the training mean, population standard deviation within a window, and norm of the vector of time increments, in meters |
| `weights` | Nonnegative masses normalized within a partition or sample set |
| `power` | 96 DFT bin powers at positive frequencies in square meters, from 0.5 to 48 cycles/day |

Values are already in physical units; do not apply the training scale again. Original observed windows use uniform mass. The 2018 partitions use their saved weights for UTC start hours. Generated samples use equal mass.

ECDFs retain ties and the zero left limit. Neural curves average streams within each fit before taking the median and range across training seeds at each feature value or frequency. PPCA curves average streams from one fit. Compact PDF paths remove only redundant vertices on constant segments. Spectral powers are averaged in linear units before plotting on logarithmic axes. Dashed neural spectra show decoder means; solid curves include observation noise. Shading gives the full range across training seeds.

## Compare numerical drawing inputs

The independent comparison uses the same cache with the retained original plotting functions and the current renderer:

```bash
mkdir -p runs
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python \
  analysis/prigm/figures/compare_plot_arrays.py \
  --repo . --output runs/figure-equivalence.json
```

Use a new output filename. The comparison expands compact ECDF paths onto the full support, compares other numerical drawing arguments directly, and checks that deliberate perturbations are rejected. It does not recover curves from historical raster images or certify visual layout. The saved [equivalence receipt](equivalence.json) records the inputs used for that comparison.

## Re-extract the cache

`prepare_figure_data.py` needs private arrays and metadata plus a mapping of the exact input names in `plot_data_manifest.json["files"]` to their local paths. Those arrays are not part of this release. The extractor checks all consumed hashes and thirteen current scientific source files; use a fresh output directory and compare numerical results before adopting a new cache.

[The presentation manifest](../../../paper/prigm/presentation_manifest.json) binds the cache, rendering/comparison code and seven retained PDFs. Replacing those inputs requires renewing that manifest and the paper's submission seal. A preview render does not replace the release figures.
