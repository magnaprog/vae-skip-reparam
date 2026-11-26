"""Regression checks against the released 21415/2018 inputs and GPU reductions."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_preparation import prepare_dart_record
from optimizer_comparisons import (MODES, RESULTS, ROOT, STREAMS, comparison_rows,
                                   read_csv)
from optimizer_coordinates import PREPARED, PREPARED_SHA, batches, build_update


def test_archived_windows_and_first_training_batch():
    raw = ROOT / "data" / "21415t2018.txt"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == (
        "bb888a1856aab261f9da184cfe5552a253808ab6124f37a6b5a68d5105127678")
    assert hashlib.sha256(PREPARED.read_bytes()).hexdigest() == PREPARED_SHA
    prepared = prepare_dart_record(raw, "2018-01-01", "2019-01-01")
    with np.load(PREPARED, allow_pickle=False) as archive:
        for key, value in prepared.items():
            if value.dtype.kind == "f":
                np.testing.assert_allclose(value, archive[key], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(value, archive[key])
        indices, noise = next(batches(42, archive["train_weights"], 1))
    assert hashlib.sha256(indices.tobytes()).hexdigest() == (
        "ed143d353a885e71d7a6bf36d0b7d27be12211c621dcae65e1e62ff4e258776a")
    assert hashlib.sha256(noise.tobytes()).hexdigest() == (
        "503e6f0079bf7b3f0b8420c6db0be420eaf63530bb6db61921170c18bd30712f")


def test_all_paired_streams_against_released_results():
    history = ROOT / "results" / "energy_terms_streams.csv"
    fits = read_csv(RESULTS / "scores.csv")
    expected = read_csv(RESULTS / "comparisons.csv")
    actual = comparison_rows(fits[::-1], history)

    def key(row):
        return row["variance_label"], row["geometry"], int(row["seed"]), row["contrast"]

    reference = {key(row): row for row in expected}
    assert len(reference) == len(actual) == 234
    assert set(reference) == {key(row) for row in actual}
    for row in actual:
        for field in [f"stream_{s}" for s in STREAMS] + ["stream_mean"]:
            assert row[field] == pytest.approx(float(reference[key(row)][field]), rel=0, abs=1e-13)
    with pytest.raises(ValueError, match="36 final fits"):
        comparison_rows(fits[1:], history)


def test_coordinate_updates_at_a_common_start():
    tf = pytest.importorskip("tensorflow")
    from decoder_coordinates import BypassVAE

    rng = np.random.default_rng(31415)
    x = tf.constant(rng.standard_normal((64, 12)), dtype=tf.float32)
    noise = tf.constant(rng.standard_normal((64, 3)), dtype=tf.float32)
    models = {}
    for mode, (condition, transformed, clipping) in MODES.items():
        model = BypassVAE(input_dim=12, latent_dim=3, condition=condition,
                          observation_variance=0.0001, seed=42)
        variables = model.trainable_variables
        heads = {v.ref() for v in model.output_head.trainable_variables}
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model.loss_terms(x, noise)["negative_elbo"])
        gradients = tape.gradient(loss, variables)
        norm_grads = []
        for gradient, var in zip(gradients, variables):
            value = gradient.numpy().astype(np.float64)
            if var.ref() in heads:
                value *= model.head_scale
                if clipping == "residual":
                    value /= model.b
            norm_grads.append(value.ravel())
        scale = min(1.0, 10.0 / np.linalg.norm(np.concatenate(norm_grads)))
        update, optimizer = build_update(model, mode, x)
        measured_scale = float(update(tf.range(64), noise)[4])
        assert measured_scale == pytest.approx(scale, rel=2e-6)
        head = optimizer._optimizers["head"]
        assert float(head.learning_rate) == pytest.approx(0.005 if transformed else 0.001)
        assert head.epsilon == pytest.approx(2e-8 if transformed else 1e-7)
        models[mode] = model
    for a, b in zip(models["no_skip"].weights, models["matched_obs"].weights):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    native = models["native_residual"]
    heads = {v.ref() for v in native.output_head.trainable_variables}
    for w, native_var in zip(models["original_res"].weights, native.weights):
        equivalent = native_var.numpy() / (native.b if native_var.ref() in heads else 1.0)
        np.testing.assert_allclose(w.numpy(), equivalent, rtol=2e-5, atol=5e-6)
