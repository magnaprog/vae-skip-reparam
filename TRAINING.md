# Model training

Run commands from the repository root in a separate environment from figure rendering. The experiments used Python 3.11.8 with these direct dependencies:

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
python src/experiment.py prepare --run-root runs/smoke --purpose smoke
CUDA_VISIBLE_DEVICES='GPU-REPLACE-ME' python src/experiment.py train --run-root runs/smoke \
  --dataset gaussian --condition no_skip_base --seed 42 --gpu-uuid 'GPU-REPLACE-ME'
```

Replace both UUID placeholders with the UUID of the GPU assigned to this run. This checks execution with fewer training steps and samples than the experiments in the paper.

## Gaussian and DART experiments

The launcher runs 60 training jobs across one Gaussian dataset and four annual DART records. Each dataset has four conditions and three training seeds per condition. The [data guide](data/README.md) lists the DART records. Prepare a fresh run directory, then supply the UUIDs of four available GPUs:

```bash
python src/experiment.py prepare --run-root runs/core
python src/experiment.py campaign --run-root runs/core \
  --gpu-uuids 'GPU-UUID-0' 'GPU-UUID-1' 'GPU-UUID-2' 'GPU-UUID-3'
```

| Condition | Input skip | Observation variance | Residual variance |
| --- | --- | --- | --- |
| `no_skip_base` | None | Base | Base |
| `skip_base` | 0.8 | Base | 0.04 times base |
| `no_skip_high` | None | 25 times base | 25 times base |
| `skip_high` | 0.8 | 25 times base | Base |

Select a condition with `--condition`. The base observation variance is estimated by fitting probabilistic principal component analysis (PPCA) to the training data.

The names `base` and `high` refer to observation variance. For skip coefficient `alpha` and `b = 1 - alpha`, the residual is `b*x - f(z)`, and `observation_variance = residual_variance / b**2`. A skip coefficient of 0.8 gives `b = 0.2`, so residual variance is 0.04 times observation variance. These are conditional noise variances; the generator's marginal variance also includes variation in its decoder mean.

The run directory contains preparation settings, copies of the training code, training histories, network weights and evaluations. Rerunning an interrupted fit creates a new attempt directory and starts training with reinitialized optimizer and random number state.

The paper evaluates another 36 fits on three DART records from 2018, bringing the DART total to 84. This release includes their scores and figure inputs. Retraining these models requires preparation inputs and training launchers that are outside this release.

Numerical helpers require finite arrays, nonnegative weights with positive total weight, and finite positive observation variances.

## Model tests

In the training environment, install pytest and run the model and integration tests:

```bash
python -m pip install pytest==9.0.3
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  -q -rs -p no:cacheprovider tests/test_model.py tests/test_decoder_coordinates.py \
  tests/test_experiment.py
```

These tests run on CPU. Run the figure and numerical tests in the separate environment described in the README.
