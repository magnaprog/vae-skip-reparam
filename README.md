# VAE skip reparameterization

Code, data and numerical results for *An Input Skip Can Reparameterize a VAE Without Improving the Generator*, Kevin Lee, PriGM 2026.

A constant input skip can reduce reconstruction error without changing the distribution a Gaussian variational autoencoder generates. The experiments compare reconstruction and generation on synthetic Gaussian data and NOAA DART records, while varying the skip coefficient and observation variance separately.

The repository contains model and evaluation code, four annual DART records, numerical results and the inputs for all ten figures in the paper.

## Draw the figures

From the repository root, create a CPU environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/prigm/figures/render_figures.py --output-dir runs/figures
```

The command writes ten PDFs. It uses the supplied plotting arrays and annual records; it runs on CPU without LaTeX. See the [figure guide](analysis/prigm/figures/README.md) for units and averaging conventions.

## Models and results

The [training guide](TRAINING.md) describes the TensorFlow environment and commands for 60 fits across one Gaussian dataset and four DART records. Each dataset has four combinations of input skip and observation variance, with three training seeds per combination.

The [results guide](analysis/prigm/README.md) covers all 84 DART fits, including the three records from 2018. The supplied arrays reproduce the figures for all seven records. Training on the 2018 records also requires preparation inputs and launchers that are outside this release.

See [data sources](datasets/README.md) for the NOAA records. HEIGHT is bottom pressure expressed as equivalent water column height in meters.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  -q -rs -p no:cacheprovider tests
```

The CPU environment runs the data, metric and plotting checks; pytest lists any skipped checks. For model tests, use the separate installation and command in the [training guide](TRAINING.md#test-the-models).

## Cite the paper

Kevin Lee. *An Input Skip Can Reparameterize a VAE Without Improving the Generator*. 2nd Workshop on Principles of Generative Modeling, NeurIPS 2026.

```bibtex
@inproceedings{lee2026inputskip,
  author    = {Lee, Kevin},
  title     = {An Input Skip Can Reparameterize a {VAE} Without Improving the Generator},
  booktitle = {2nd Workshop on Principles of Generative Modeling at NeurIPS 2026},
  year      = {2026}
}
```

GitHub's citation menu uses the same paper metadata from [CITATION.cff](CITATION.cff).

## License

Software and documentation use the [MIT license](LICENSE). NOAA records retain their source attribution.
