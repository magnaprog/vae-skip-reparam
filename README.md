# VAE skip reparameterization

Code and figure inputs for [*An Input Skip Can Reparameterize a VAE Without Improving the Generator*](paper/prigm/main.pdf), by Kevin Lee, PriGM 2026.

A constant input skip can reduce a VAE's reconstruction error while preserving its generative distribution. The paper derives this equivalence for a Gaussian conditional density and examines separately trained models on a synthetic Gaussian dataset and DART pressure records.

| Task | Included inputs | Requirements |
| --- | --- | --- |
| Render all seven figures | Plotting arrays, summaries and four annual records | CPU Python environment |
| Compile the paper | LaTeX, bibliography, tables and figures | LaTeX |
| Inspect the energy analysis | Summary and complete CSV reductions | JSON/CSV reader |
| Train a new model from the original experiment | Model code, Gaussian generator and four annual records | Separate TensorFlow environment and an available GPU |

The full reported study also uses retained checkpoints, generated samples, extension preparation records and historical launchers that are not included. The plotting cache supports figure reproduction; it does not contain those complete experiment inputs.

## CPU setup

Run commands from the repository root. The direct dependency snapshot was checked on Linux with Python 3.12.3; it is not a complete transitive lock.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-release.txt
```

## Render the figures

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python \
  analysis/prigm/figures/render_figures.py \
  --output-dir runs/figures-preview
```

The seven PDFs are written to `runs/figures-preview`, leaving the retained paper figures intact. Rendering needs no GPU. The [figure guide](analysis/prigm/figures/README.md) explains the arrays, units and comparison procedure.

## Compile the paper

Install `latexmk`, pdfLaTeX and BibTeX with the manuscript's LaTeX packages, then run:

```bash
latexmk -g -cd -pdf -recorder -interaction=nonstopmode -halt-on-error \
  paper/prigm/main.tex
```

The output is `paper/prigm/main.pdf`. The [paper guide](paper/README.md#verify-a-build) gives the verification command and explains the saved submission manifest.

## Tests

A focused CPU check covers the numerical metrics and figure data:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  -q -rs -p no:cacheprovider \
  tests/test_prigm_metrics.py tests/test_prigm_v2_metrics.py \
  tests/test_prigm_figure_data.py tests/test_prigm_figure_steps.py
```

Use the same command with `tests` in place of the four filenames for the full public suite. TensorFlow/HDF5 checks need optional dependencies; checks of unbundled acquisition files skip when those files are absent. Some builder fixtures use substantial temporary space. `-rs` reports skips.

## Training and results

The [training instructions](paper/README.md#train-a-new-model-from-the-original-experiment) use a separate Python 3.11 / TensorFlow 2.15 environment and include a short smoke run. The full original core contains 60 fits. The additional 36 DART fits need separate extension inputs and preparation.

`datasets/` contains four NOAA DART annual records. HEIGHT expresses bottom pressure as equivalent water column height. The retained windows and model scores do not establish event labels, cleaning accuracy or warning performance. The [bibliography](paper/prigm/references.bib) identifies the NOAA data sources.

The [energy analysis](analysis/prigm/README.md) contains the saved comparisons and explains the inputs needed to recompute them. The [scientific manifest](paper/prigm/paper_evidence_manifest.json) records evidence identities and numerical results.

## Citation and licensing

Cite Kevin Lee, *An Input Skip Can Reparameterize a VAE Without Improving the Generator*, PriGM 2026.

Software and associated documentation use the [MIT license](LICENSE). NOAA data and the NeurIPS template retain their source attribution. The repository does not declare a separate manuscript license; see the [license boundaries](paper/README.md#license-boundaries).
