# Model training

Run commands from the repository root. Training uses a separate environment from figure rendering. The experiments used Python 3.11.8 with these direct dependencies:

```bash
python3.11 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install tensorflow==2.15.0 numpy==1.26.4 scipy==1.14.1 \
  pandas==2.2.3 h5py==3.12.1
```

Use a compatible NVIDIA driver and CUDA environment for TensorFlow 2.15.

## Short training run

Prepare a short run on CPU, then train one Gaussian model on an available GPU:

```bash
python src/prigm_experiment.py prepare --run-root runs/smoke --purpose smoke
CUDA_VISIBLE_DEVICES='GPU-REPLACE-ME' python src/prigm_experiment.py train --run-root runs/smoke \
  --dataset gaussian --condition no_skip_base --seed 42 --gpu-uuid 'GPU-REPLACE-ME'
```

Replace both UUID placeholders with the same GPU assigned to this run. The short run checks execution with fewer training steps and samples than the experiments in the paper.

## Gaussian and DART experiments

The launcher runs 60 fits: one Gaussian dataset and four annual DART records, with four conditions and three training seeds per dataset. The DART records are listed in [Data sources](datasets/README.md). Prepare a new directory, then supply the UUIDs of four available GPUs:

```bash
python src/prigm_experiment.py prepare --run-root runs/core
python src/prigm_experiment.py campaign --run-root runs/core \
  --gpu-uuids 'GPU-UUID-0' 'GPU-UUID-1' 'GPU-UUID-2' 'GPU-UUID-3'
```

| Condition | Input skip | Observation variance | Residual variance |
| --- | --- | --- | --- |
| `no_skip_base` | None | Base | Base |
| `skip_base` | 0.8 | Base | 0.04 times base |
| `no_skip_high` | None | 25 times base | 25 times base |
| `skip_high` | 0.8 | 25 times base | Base |

The `--condition` option selects one of the conditions above. Base observation variance comes from probabilistic principal component analysis (PPCA) of the training data.

The names `base` and `high` refer to observation variance. With skip coefficient `alpha` and `b = 1 - alpha`, the residual is `b*x - f(z)`, and `observation_variance = residual_variance / b**2`. A skip of 0.8 gives `b = 0.2`, so residual variance is 0.04 times observation variance. These are conditional noise variances; the generator's marginal variance also includes variation in its decoder mean.

The run directory saves the preparation settings, copies of the training code, training histories, network weights and evaluations. An interrupted fit restarts in a new attempt directory, without resuming the optimizer and random number state.

The paper also evaluates 36 fits on three DART records from 2018. Their scores and figure inputs are included, but their preparation inputs and training launchers are not part of this release.

When calling the numerical helpers with new data, use finite arrays, nonnegative weights with positive total mass, and finite positive observation variances.

## Model tests

In the training environment, install pytest and run the two model test modules:

```bash
python -m pip install pytest==9.0.3
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  -q -rs -p no:cacheprovider tests/test_prigm_model.py tests/test_prigm_decoder_coordinates.py
```

This command runs model checks on CPU. The figure and numerical tests use the separate environment described in the README.
