# Train a model

Run commands from the repository root. Training uses a separate environment from figure rendering. The experiments used Python 3.11.8 with these direct dependencies:

```bash
python3.11 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install tensorflow==2.15.0 numpy==1.26.4 scipy==1.14.1 \
  pandas==2.2.3 h5py==3.12.1
```

Use a compatible NVIDIA driver and CUDA environment for TensorFlow 2.15. The package list records the experiment's Python dependencies; driver installation depends on the host.

## Check execution

Prepare a short run on CPU, then train one Gaussian model on an available GPU:

```bash
python src/prigm_experiment.py prepare --run-root runs/smoke --purpose smoke
python src/prigm_experiment.py train --run-root runs/smoke \
  --dataset gaussian --arm A --seed 42 --gpu-uuid 'GPU-REPLACE-ME'
```

Replace the UUID with that of a GPU assigned to this run. The short run checks execution with fewer training steps and samples than the experiments in the paper.

## Run the original experiment

The original experiment uses the Gaussian dataset and four annual DART records, each with four conditions and three training seeds. Prepare a new directory, then supply the UUIDs of four available GPUs:

```bash
python src/prigm_experiment.py prepare --run-root runs/core
python src/prigm_experiment.py campaign --run-root runs/core \
  --gpu-uuids 'GPU-UUID-0' 'GPU-UUID-1' 'GPU-UUID-2' 'GPU-UUID-3'
```

| Code | Input skip | Observation variance |
| --- | --- | --- |
| A | None | Base |
| C | 0.8 | Base |
| D | None | 25 times base |
| B | 0.8 | 25 times base |

The CLI option `--arm` selects a condition. Its spelling and the result keys remain consistent with the saved experiments.

The run directory contains the preparation settings, source identities, training histories, network weights and evaluations. Interrupted fits start a new attempt; they do not resume the optimizer and random-number state exactly. Keep each completed run directory for subsequent analysis.

The `prigm_v2_data`, `prigm_v2_model` and `prigm_v2_metrics` modules contain the additional data, model and scoring routines. The 36 fits for the 2018 DART records require their separate preparation inputs and launchers. The supplied annual records support the original 60-fit experiment.

The numerical helpers expect finite arrays, nonnegative weights with positive total mass, and finite positive observation variances. Check these conditions when calling the helpers directly with new data.

## Test the models

In the training environment, install pytest and run the two model test modules:

```bash
python -m pip install pytest==9.0.3
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  -q -rs -p no:cacheprovider tests/test_prigm_model.py tests/test_prigm_v2_model.py
```

This command runs model checks on CPU. The figure and numerical tests use the separate environment described in the README.
