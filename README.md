# VAE skip reparameterization

Code, data and numerical results for *An Input Skip Can Reparameterize a VAE Without Improving the Generator*, Kevin Lee, NeurIPS PriGM 2026.

A constant input skip can reduce reconstruction error without changing the distribution a Gaussian variational autoencoder generates. The experiments compare reconstruction and generation on synthetic Gaussian data and NOAA DART records, while varying the skip coefficient and observation variance separately.

## Reproduce the figures

From the repository root, create a CPU environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/prigm/figures/render_figures.py --output-dir runs/figures
```

The script saves all ten figures as PDFs in `runs/figures` and runs on CPU without LaTeX. See the [figure guide](analysis/prigm/figures/README.md) for units and averaging conventions.

## Models and results

The [training setup](TRAINING.md) runs 60 fits: one Gaussian dataset and four DART records, each with four skip and variance conditions and three training seeds per condition.

Numerical [results](analysis/prigm/README.md) cover 84 DART fits across seven annual records. Retraining the three records from 2018 requires preparation inputs and launchers that are outside this release.

See [data sources](datasets/README.md) for the NOAA records. HEIGHT is bottom pressure expressed as equivalent water column height in meters.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  -q -rs -p no:cacheprovider tests
```

Run the data, metric and plotting tests in the CPU environment. For model tests, use the separate environment in the [training guide](TRAINING.md#model-tests).

## Citation

```bibtex
@inproceedings{lee2026inputskip,
  author    = {Lee, Kevin},
  title     = {An Input Skip Can Reparameterize a {VAE} Without Improving the Generator},
  booktitle = {2nd Workshop on Principles of Generative Modeling at NeurIPS 2026},
  year      = {2026}
}
```

## License

Software and documentation use the [MIT license](LICENSE). NOAA records retain their source attribution.
