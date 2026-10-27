# VAE skip reparameterization

Code, data and numerical results for *An Input Skip Can Reparameterize a VAE Without Improving the Generator*, Kevin Lee, NeurIPS PriGM 2026.

A constant input skip can reduce reconstruction error without changing the distribution a Gaussian variational autoencoder generates. The experiments compare reconstruction and generation on synthetic Gaussian data and NOAA DART records, while varying the skip coefficient and observation variance separately.

## Results and reproduction

The [numerical results](results/README.md) cover 84 DART fits across seven annual records. The guide explains the energy scores, comparisons with probabilistic principal component analysis, and effects of changing observation variance.

To repeat model training and evaluation, follow the [training guide](TRAINING.md). Its launcher runs 60 training jobs: one Gaussian dataset and four DART records, each with four skip and variance conditions and three training seeds per condition. Retraining the three records from 2018 requires preparation inputs and launchers that are outside this release.

See [data sources](data/README.md) for the NOAA records. HEIGHT is bottom pressure expressed as equivalent water column height in meters.

### Figures

The figure script recreates the paper's plots from the supplied results and observations.

From the repository root, create a CPU environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python figures/render.py --output-dir runs/figures
```

The script saves all ten figures as PDFs in `runs/figures` and runs on CPU without LaTeX. See the [figure guide](figures/README.md) for units and averaging conventions.

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
  booktitle = {{NeurIPS} 2026 Workshop on Principles of Generative Modeling ({PriGM})},
  year      = {2026}
}
```

## License

Software and documentation use the [MIT license](LICENSE). NOAA records retain their source attribution.
