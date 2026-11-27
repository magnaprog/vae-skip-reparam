# Adam updates and gradient clipping at DART station 21415

This follow-up revisits the 2018 record at station 21415 from the seven-record study. At each of two observation variances (0.0001 and 0.0025), six training procedures start from the same observation means and receive the same minibatches and posterior noise for each seed (42, 1729, and 31415). The original runs used one RTX 3080 Ti per seed, TensorFlow 2.15.0 with TF32 enabled, NumPy 1.26.4, and 10,000 Adam steps. All scores use final weights.

| Mode | Stored decoder output parameters | Output Adam learning rate, epsilon | Coordinates of the clipping norm |
| --- | --- | --- | --- |
| `no_skip` | Observation, skip 0 | 0.001, 1e-7 | Observation |
| `native_residual` | Residual, skip 0.8 | 0.001, 1e-7 | Residual |
| `matched_obs` | Observation, skip 0.8 | 0.001, 1e-7 | Observation |
| `matched_res` | Observation, skip 0.8 | 0.001, 1e-7 | Residual |
| `original_obs` | Observation, skip 0.8 | 0.005, 2e-8 | Observation |
| `original_res` | Observation, skip 0.8 | 0.005, 2e-8 | Residual |

All other parameters use Adam at 0.001 and epsilon 1e-7. With a skip of 0.8 the residual output parameters are 0.2 times their observation-coordinate equivalents. Expressing their gradients in residual coordinates multiplies them by five before measuring the global norm. The global norm threshold is 10, and the same clipping factor applies to every layer. The two Adam configurations for the output layer differ in both learning rate and epsilon.

## Data and results

The supplied [prepared arrays](../../data/dart_21415_2018.npz) are the exact training and evaluation input. Chronological partitions contain 1,227 training, 364 validation, and 215 test windows of 192 consecutive observations each. The six UTC start times are four hours apart and each receives equal total weight within each partition. The archive includes normalization, window starts, and test weights. The raw source, SHA-256 digests, and preparation details are in the [data guide](../../data/README.md#station-21415-follow-up).

[The score table](scores.csv) gives three geometries per final fit and binds each fit to the SHA-256 of its weights. To generate each of its 1,024 windows, a stream draws a latent vector from the prior, decodes it, and adds Gaussian observation noise. The energy score compares full windows (`joint`), windows after mean subtraction (`centered`), or time increments (`increment`). [The comparison table](comparisons.csv) subtracts weighted test scores within each variance, training seed, geometry, and prior stream, then averages the three differences. The interaction is the matched-minus-original Adam difference under observation-norm clipping minus that difference under residual-norm clipping.

On this record the larger output learning rate with smaller epsilon lowered full-window energy score at both clipping norms and variances for all three seeds. The clipping comparison changed direction across fits. The matched observation-coordinate model and the no-skip model produced equal scores. Finite-precision training of the native residual model and its observation-coordinate equivalent reached different final scores.

## Reproduction

From the repository root, the following command regenerates all 234 comparison rows using the two public score sources. The earlier fits come from the final checkpoints in [energy_terms_streams.csv](../energy_terms_streams.csv). The released comparisons used their per-window reductions, so historical contrasts can differ in the last decimal places.

```sh
mkdir -p runs/optimizer
python src/optimizer_comparisons.py --output runs/optimizer/comparisons.csv
```

To train, use the Python 3.11 environment in the [training guide](../../TRAINING.md). Run the example on a GPU reserved for your use, replacing the UUID with its actual value. Each fit needs a fresh output directory. Supply `--seed` to run twelve fits on one device, or omit it to run all 36 sequentially on that device. The summarizer checks the mode, observation variance, seed, and step count in each fit’s completion record and verifies the SHA-256 hash of the final weights. Before writing scores, comparisons, and per-window differences, it compares the recorded hashes of the minibatch indices and posterior noise across fits with the same seed.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python src/optimizer_coordinates.py --gpu-uuid GPU-REPLACE-ME train --fits runs/optimizer/fits
python src/optimizer_coordinates.py --gpu-uuid GPU-REPLACE-ME summarize \
    --fits runs/optimizer/fits --output runs/optimizer/evaluation
```

Omitting `--gpu-uuid` runs on CPU. For a quick check, train two steps and evaluate that checkpoint with the same scoring rule. The complete-grid summary accepts only fits trained for 10,000 steps. Each fit writes its final HDF5 weights, per-step clipping and update measurements, and a completion record. A retraining run may follow a different numerical path across hardware or software versions.

```sh
python src/optimizer_coordinates.py train --fits runs/optimizer/smoke \
    --variance base --seed 42 --mode native_residual --steps 2
python src/optimizer_coordinates.py score --variance base --seed 42 --mode native_residual \
    --weights runs/optimizer/smoke/base-42-native_residual/final.weights.h5 \
    --output runs/optimizer/smoke-scores.csv
```

## Retrospective interval check

For each of the 12 comparisons, [output_update_windows.csv](output_update_windows.csv) gives the difference in full-window energy score on each of 215 test windows. The sign is `matched` minus `original` at the same clipping norm, variance, and training seed. Each row averages the differences across prior streams 71001, 71002, and 71003. The reducer matches each row to its test window by UTC start time in Unix seconds, using `window_start_unix` in the CSV and `starts_test` in the prepared archive.

[output_update_intervals.csv](output_update_intervals.csv) groups windows by start time in the existing intervals of 14 days measured from the archive’s `reassignment_block_origin`. Each row gives the test weight in that interval and the mean score difference within it, outside it, and across the full test set. Each mean renormalizes `eval_weights` over the windows it includes. Omitting any interval preserved all 12 positive full-test differences. Three differences reversed sign within the final interval, which contained one window. These overlapping summaries describe one station-year.

Run the reducer from the repository root with Python and NumPy to reproduce all 60 rows.

```sh
python src/optimizer_intervals.py --output intervals.csv
```

To use scores from a new run, pass `--windows runs/optimizer/evaluation/output_update_windows.csv` after running `summarize` above. Both commands use the same prepared test windows and weights.

## Verification

CPU evaluation of the 36 original final checkpoints with this implementation reproduced the three GPU stream scores for each fit and geometry to within 7 × 10⁻⁹ and the paired stream contrasts to within 1.4 × 10⁻⁸ in absolute value. Rebuilding comparisons directly from the supplied CSVs differed by at most 5.4 × 10⁻¹⁵. A separate CPU retraining of the base-variance native residual model at seed 42 reproduced both random streams through all 10,000 updates. Its mean full-window energy score was 11.47597, compared with 11.47913 for the GPU fit.

Run `python -m pytest -q tests/test_optimizer_coordinates.py` in the training environment to check raw and prepared inputs, the first sampled batch, coordinate updates, and the released comparisons and intervals.
