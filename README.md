# VAE skip reparameterization

Code, data and numerical results for *An Input Skip Can Reparameterize a VAE Without Improving the Generator*, Kevin Lee, PriGM 2026.

A constant input skip can reduce reconstruction error without changing the distribution a Gaussian variational autoencoder generates. The experiments compare reconstruction and generation on synthetic Gaussian data and NOAA DART records, while varying the skip coefficient and observation variance separately.

This repository contains the model and evaluation routines, four annual DART records, numerical results and the inputs needed to draw all seven figures. The manuscript and development records are maintained separately.

## Draw the figures

From the repository root, create a CPU environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/prigm/figures/render_figures.py --output-dir runs/figures
```

The command writes seven PDFs. It uses the supplied plotting arrays and annual records; it runs on CPU without LaTeX. See the [figure guide](analysis/prigm/figures/README.md) for units and averaging conventions.

## Models and results

The [training guide](TRAINING.md) describes a separate TensorFlow environment and commands for a new Gaussian or DART experiment. The original experiment has 60 fits. Additional model, data and scoring routines are included for the subsequent comparisons.

The [energy-score results](analysis/prigm/README.md) contain comparisons for all 84 DART fits. Complete retraining of the full study also needs the preparation inputs and launchers for the additional 2018 records; the plotting arrays contain the information needed to reproduce the figures.

See [data sources](datasets/README.md) for the NOAA records. HEIGHT is bottom pressure expressed as equivalent water column height in meters.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  -q -rs -p no:cacheprovider tests
```

The CPU environment runs the data, metric and plotting checks; pytest lists any skipped checks. For model tests, use the separate installation and command in the [training guide](TRAINING.md#test-the-models).

## Citation and license

Kevin Lee. *An Input Skip Can Reparameterize a VAE Without Improving the Generator*. 2nd Workshop on Principles of Generative Modeling, NeurIPS 2026.

Software and documentation use the [MIT license](LICENSE). NOAA records retain their source attribution.
