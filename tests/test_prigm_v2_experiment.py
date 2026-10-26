"""Runner closure and disposable-child cancellation checks, without GPU training.

Genuine HDF5 fixtures are function-level optional; GPU training remains a host gate.
"""
import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import prigm_v2_experiment as v2


def test_stage_definitions_cover_matrix():
    assert set(v2.STAGES) == {"numerical_null"}
    assert v2.STAGES["numerical_null"] == [
        ("gaussian_v2", condition, 42, variance)
        for variance in ("low", "high") for condition in ("base", "canonical")]
    assert sum(map(len, v2.STAGES.values())) == 4


def test_probe_schedule_matches_plan():
    steps = v2.SPEC["probe_steps"]
    assert steps[0] == 1 and 250 in steps and 1000 in steps and steps[-1] == 10000
    assert all(s in steps for s in (1, 2, 249, 250))
    assert 251 not in steps and 500 not in steps
    assert len([s for s in steps if s > 250]) == 10  # every 1000 only


def test_spec_frozen_fields():
    assert v2.SPEC["max_steps"] == 10000 and v2.SPEC["batch_size"] == 64
    assert v2.SPEC["learning_rate"] == 0.001
    assert v2.SPEC["adam"] == {"beta_1": 0.9, "beta_2": 0.999,
                               "epsilon": 1e-7, "amsgrad": False}
    assert v2.SPEC["gradient_global_norm"] == 10.0
    assert v2.SPEC["synthetic_observation_variances"] == [0.0625, 1.5625]
    assert v2.SPEC["evaluation_prior_samples"] == 1024
    assert v2.SPEC["evaluation_streams"] == 3
    assert v2.SPEC["iw_samples"] == 256
    assert v2.SPEC["linear_control_k"] == [64, 256, 1024]
    assert v2.SPEC["validation_every"] == 1000
    assert v2.SPEC["validation_prior_samples"] == 512
    assert v2.SPEC["model_storage_contract"] == 3
    assert v2.SPEC["purpose"] == "numerical_validation"
    assert v2.SPEC["gradient_clipping_enabled"] is True
    assert v2.SPEC["exact_resume_supported"] is False


def test_job_id_rejects_unknown():
    with pytest.raises(ValueError):
        v2.job_id("gaussian_v2", "bogus", 42, "low")
    with pytest.raises(ValueError):
        v2.job_id("unknown", "base", 42, "low")
    assert v2.job_id("gaussian_v2", "raw", 42, "low") == "gaussian_v2_raw_42_low"


def test_build_and_verify_stage_roundtrip(tmp_path):
    prepared = tmp_path / "prepared-src"
    prepared.mkdir()
    rng = np.random.default_rng(0)
    for name in ("gaussian_v2",):
        np.savez_compressed(prepared / f"{name}.npz",
                            train=rng.standard_normal((128, 192)),
                            validation=rng.standard_normal((32, 192)),
                            test=rng.standard_normal((32, 192)))
    root = tmp_path / "stage"
    v2.build_stage(root, "numerical_null", str(prepared))
    manifest = v2.verify_stage(root)
    assert manifest["stage"] == "numerical_null"
    assert len(manifest["jobs"]) == 4
    assert manifest["schema_version"] == 3
    assert manifest["purpose"] == "numerical_validation"
    assert manifest["job_parameters"] == v2.job_parameters("numerical_null")
    # Tampering with prepared inputs must fail verification.
    with (root / "prepared" / "gaussian_v2.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError):
        v2.verify_stage(root)


def test_stage_rejects_existing_root(tmp_path):
    prepared = tmp_path / "p"
    prepared.mkdir()
    np.savez_compressed(prepared / "gaussian_v2.nz", train=np.zeros((4, 192)))
    np.savez_compressed(prepared / "gaussian_v2.npz", train=np.zeros((4, 192)))
    root = tmp_path / "stage"
    root.mkdir()
    with pytest.raises(FileExistsError):
        v2.build_stage(root, "numerical_null", str(prepared))


class _ConstantDecoder:
    latent_dim, input_dim = 16, 192
    alpha, b, head_scale = 0.0, 1.0, 1.0
    residual_variance = observation_variance = 0.0625

    def encode(self, x):
        return np.zeros((len(x), self.latent_dim)), np.zeros((len(x), self.latent_dim))

    def decode_mean(self, z):
        return np.zeros((len(z), self.input_dim))


@pytest.mark.parametrize("payload", ['{"step": 1, "step": 2}', '{"score": NaN}', '{"score": Infinity}'])
def test_json_identity_rejects_duplicate_keys_and_nonfinite_constants(tmp_path, payload):
    path = tmp_path / "identity.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        v2.read_json(path)
    with pytest.raises(ValueError):
        v2._rows(path)


def _hdf5_weights(path, configuration=None):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as store:
        store.attrs.update(storage_contract=3, head_coordinates="w", head_scale=1.0,
                           observation_variance=0.0625)
        if configuration is not None:
            for key in ("head_coordinates", "head_scale", "observation_variance"):
                store.attrs[key] = configuration[key]
        store.create_group("vars")
        for name, shape in (("layers/dense", (192, 128)), ("layers/dense_1", (128, 64)),
                            ("layers/dense_2", (16, 64)), ("layers/dense_3", (64, 128)),
                            ("mean_head", (64, 16)), ("logvar_head", (64, 16)),
                            ("output_head", (128, 192))):
            group = store.create_group(f"{name}/vars")
            group.create_dataset("0", data=np.zeros(shape, dtype=np.float32))
            group.create_dataset("1", data=np.zeros(shape[1], dtype=np.float32))


def _write_attempt(root, identifier="gaussian_v2_base_42_low", selected_step=10000,
                   genuine_weights=False):
    """Full synthetic audit fixture; fake bytes only for rejection tests."""
    manifest = v2.verify_stage(root)
    attempt = root / "jobs" / identifier / ("attempt-" + "a" * 32)
    attempt.mkdir(parents=True)
    configuration = v2._configuration_identity(root, manifest, identifier, attempt.name)
    runtime = {"gpu_uuid": "fixture-GPU", "pid": 123, "placement": "/device:GPU:0",
               "process_inventory": ["123, fixture-GPU, fixture, 0 MiB"],
               "tensorflow": "2.15.0-fixture", "python": sys.version, "numpy": np.__version__,
               "deterministic_ops_requested": True}
    v2.atomic_json(attempt / "configuration.json", dict(configuration, runtime=runtime))
    rows = []
    best = 0
    for step in range(1, 10001):
        rows.append({"step": step, "pre_update_loss": 1.0, "pre_update_minibatch_kl": 0.0,
                     "gradient_norm": 20.0, "clipping_enabled": True,
                     "clipping_scale": 0.5, "clipping_active": True})
        if step % 1000 == 0:
            best = min(step, selected_step)
            rows.append({"step": step, "validation_energy": float(max(1, (selected_step-step)//1000+1)),
                         "best_step": best})
    _write_rows(attempt / "history.jsonl", rows)
    _write_rows(attempt / "probes.jsonl", [
        {"step": step, "full_decoder_rms": 0.0, "head_only_rms": 0.0, "encoder_symmetric_kl": 0.0}
        for step in v2.SPEC["probe_steps"]])
    v2.atomic_json(attempt / "progress.json", {"step": 10000, "best_step": best,
                                              "best_score": 1.0, "elapsed_seconds": 10.0})
    for label in ("final", "selected"):
        path = attempt / f"{label}.weights.h5"
        if genuine_weights:
            _hdf5_weights(path, configuration)
        else:
            path.write_bytes(b"not HDF5")
    with np.load(root / "prepared" / f'{configuration["dataset"]}.npz') as store:
        data = dict(store)
    decoder = _ConstantDecoder()
    decoder.alpha, decoder.b = configuration["alpha"], configuration["b"]
    decoder.head_scale = configuration["head_scale"]
    decoder.observation_variance = configuration["observation_variance"]
    decoder.residual_variance = configuration["residual_variance"]
    summary = {"identifier": identifier, "attempt": attempt.name, "run_id": manifest["run_id"],
               "selected_step": selected_step, "best_score": 1.0, "selected": None}
    for label, step in (("final", 10000), ("selected", selected_step)):
        if label == "selected" and selected_step == 10000:
            continue
        result = v2._evaluate(attempt, decoder, data, manifest["specification"], identifier,
                              configuration["seed"], checkpoint_label=label)
        result.update(step=step, weights_sha256=v2.sha256(attempt / f"{label}.weights.h5"),
                      evaluation_sha256=v2.sha256(attempt / f"evaluation-{label}.npz"))
        summary[label] = result
    v2.atomic_json(attempt / "summary.json", summary)
    return attempt


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rehash_marker(root, attempt):
    """Deliberately self-consistent hashes do not make fabricated evidence valid."""
    manifest = v2.read_json(root / "stage_manifest.json")
    payload = {"schema_version": 3, "identifier": attempt.parent.name, "run_id": manifest["run_id"],
               "attempt": attempt.name, "stage_manifest_sha256": v2.sha256(root / "stage_manifest.json"),
               "artifact_sha256": {p.name: v2.sha256(p) for p in attempt.iterdir()
                                   if p.name != "complete.json"}}
    v2.atomic_json(attempt.parent / "complete.json", payload)
    return payload


def _stage_with_prepared(tmp_path):
    prepared = tmp_path / "prepared-src"
    prepared.mkdir()
    rng = np.random.default_rng(0)
    np.savez_compressed(prepared / "gaussian_v2.npz",
                        train=rng.standard_normal((8, 192)),
                        validation=rng.standard_normal((4, 192)),
                        test=rng.standard_normal((4, 192)))
    root = tmp_path / "stage"
    v2.build_stage(root, "numerical_null", str(prepared))
    return root


def test_verified_completion_rejects_marker_only(tmp_path):
    root = _stage_with_prepared(tmp_path)
    identifier = "gaussian_v2_base_42_low"
    job = root / "jobs" / identifier
    job.mkdir(parents=True)
    v2.atomic_json(job / "complete.json", {"attempt": "attempt-missing"})
    with pytest.raises(ValueError):
        v2.verified_completion(root, identifier)


@pytest.mark.parametrize("selected_step", [3000, 10000])
def test_verified_completion_accepts_genuine_closed_attempt(tmp_path, selected_step):
    pytest.importorskip("h5py")
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root, selected_step=selected_step, genuine_weights=True)
    assert not (attempt.parent / "complete.json").exists()
    completion = v2.check_attempt(root, attempt.parent.name, attempt)
    assert not (attempt.parent / "complete.json").exists()
    v2.atomic_json(attempt / "complete.json", completion)
    v2.atomic_json(attempt.parent / "complete.json", completion)
    assert v2.verified_completion(root, attempt.parent.name) == attempt


def _historical_attempt(tmp_path, monkeypatch, selected_step=10000, dart=False):
    # Freeze a genuine, completed producer fixture BEFORE changing current bytes.
    # These synthetic pins are independent of later resealed archive declarations.
    pytest.importorskip("h5py")
    root = _dart_stage(tmp_path) if dart else _stage_with_prepared(tmp_path)
    identifier = "dart_44402_2018_raw_42_low" if dart else "gaussian_v2_base_42_low"
    attempt = _write_attempt(root, identifier, selected_step, genuine_weights=True)
    v2._publish_completion(root, identifier, attempt)
    identity = {"manifest_sha256": v2.sha256(root / "stage_manifest.json"),
                "code_sha256": v2.read_json(root / "stage_manifest.json")["code_sha256"]}
    current = tmp_path / "current"
    shutil.copytree(root / "source", current)
    (current / "src/prigm_metrics.py").write_text("# changed current source identity\n")
    monkeypatch.setattr(v2, "ROOT", current)
    return root, attempt, identity


@pytest.mark.parametrize("selected_step,dart", [(3000, False), (10000, False), (10000, True)])
def test_historical_completion_reads_real_hdf5_but_default_writers_stay_strict(
        tmp_path, monkeypatch, selected_step, dart):
    root, attempt, identity = _historical_attempt(tmp_path, monkeypatch, selected_step, dart)
    identifier = attempt.parent.name
    for verify in (lambda: v2.verify_stage(root),
                   lambda: v2.check_attempt(root, identifier, attempt),
                   lambda: v2.verified_completion(root, identifier),
                   lambda: v2._publish_completion(root, identifier, attempt),
                   lambda: v2.train(root, "gaussian_v2", "base", 42, .0625, "low", "unused"),
                   lambda: v2.campaign(root, ["unused-0", "unused-1", "unused-2", "unused-3"])):
        with pytest.raises(ValueError, match="source changed after stage build"):
            verify()
    assert v2.verify_stage(root, historical_identity=identity)["code_sha256"] == identity["code_sha256"]
    completion = v2.check_attempt(root, identifier, attempt, historical_identity=identity)
    assert completion == v2.read_json(attempt / "complete.json")
    assert v2.verified_completion(root, identifier, historical_identity=identity) == attempt
    with pytest.raises(ValueError, match="approved pins"):
        v2.verify_stage(root, check_code=False, historical_identity=identity)


@pytest.mark.parametrize("mutation,error", [
    ("source", "Archived v2 source"), ("resealed_source", "approved pins"),
    ("prepared", "Prepared input"), ("marker", "completion marker"),
    ("configuration", "configuration identity"), ("tensor", "weight tensor dtype or value"),
    ("tensor_layout", "Keras weight group inventory"), ("evaluation", "evaluation"),
])
def test_historical_completion_rejects_tampering_after_resealing(tmp_path, monkeypatch, mutation, error):
    h5py = pytest.importorskip("h5py")

    root, attempt, identity = _historical_attempt(tmp_path, monkeypatch)
    marker = v2.read_json(attempt / "complete.json")
    if mutation in ("source", "resealed_source"):
        source = root / "source/src/prigm_metrics.py"
        source.write_text("# replaced producer\n")
        if mutation == "resealed_source":
            manifest = v2.read_json(root / "stage_manifest.json")
            manifest["code_sha256"]["src/prigm_metrics.py"] = v2.sha256(source)
            v2.atomic_json(root / "stage_manifest.json", manifest)
    elif mutation == "prepared":
        with (root / "prepared/gaussian_v2.npz").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "marker":
        marker["extra"] = "unapproved"
        v2.atomic_json(attempt / "complete.json", marker)
    else:
        if mutation == "configuration":
            config = v2.read_json(attempt / "configuration.json")
            config["head_scale"] = 9.
            v2.atomic_json(attempt / "configuration.json", config)
        elif mutation in ("tensor", "tensor_layout"):
            with h5py.File(attempt / "final.weights.h5", "r+") as store:
                if mutation == "tensor":
                    store["output_head/vars/0"][0, 0] = np.nan
                else:
                    store.move("mean_head", "renamed_mean_head")
            summary = v2.read_json(attempt / "summary.json")
            summary["final"]["weights_sha256"] = v2.sha256(attempt / "final.weights.h5")
            v2.atomic_json(attempt / "summary.json", summary)
        else:
            with np.load(attempt / "evaluation-final.npz") as store:
                arrays = dict(store)
            arrays["prior_samples"][0, 0] = np.nan
            np.savez_compressed(attempt / "evaluation-final.npz", **arrays)
            summary = v2.read_json(attempt / "summary.json")
            summary["final"]["evaluation_sha256"] = v2.sha256(attempt / "evaluation-final.npz")
            v2.atomic_json(attempt / "summary.json", summary)
        marker["artifact_sha256"] = {name: v2.sha256(attempt / name) for name in marker["artifact_sha256"]}
        for path in (attempt / "complete.json", attempt.parent / "complete.json"):
            v2.atomic_json(path, marker)
    with pytest.raises(ValueError, match=error):
        v2.verified_completion(root, attempt.parent.name, historical_identity=identity)


def test_verified_completion_rejects_fake_weight_bytes(tmp_path):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    _rehash_marker(root, attempt)
    with pytest.raises(ValueError, match="genuine HDF5"):
        v2.verified_completion(root, attempt.parent.name)


def test_verified_completion_requires_selected_eval_when_best_is_earlier(tmp_path):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root, selected_step=3000)
    (attempt / "evaluation-selected.npz").unlink()
    _rehash_marker(root, attempt)
    with pytest.raises(ValueError, match="Missing selected evaluation"):
        v2.verified_completion(root, attempt.parent.name)


def test_history_and_probe_logging_contract(tmp_path):
    path = tmp_path / "history.jsonl"
    v2.append_jsonl(path, {"step": 1, "loss": 1.0})
    v2.append_jsonl(path, {"step": 2, "loss": 0.5})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}]


def test_symmetric_kl_zero_for_identical():
    mu = np.array([[0.1, -0.2]])
    lv = np.array([[0.3, -0.1]])
    np.testing.assert_allclose(v2._symmetric_kl(mu, lv, mu, lv), 0.0, atol=1e-12)


def test_unweighted_training_indices_have_valid_shape_dtype_and_bounds():
    rng = np.random.default_rng(7)
    indices = v2.training_indices(rng, n=10, batch_size=20)
    assert indices.dtype == np.int32
    assert indices.shape == (20,)
    assert indices.min() >= 0 and indices.max() < 10


def test_training_indices_select_one_hot_weight_index():
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    rng = np.random.default_rng(11)
    indices = v2.training_indices(rng, n=5, batch_size=64, weights=weights)
    np.testing.assert_array_equal(indices, np.full(64, 2, dtype=np.int32))


def test_training_indices_reject_bad_weights():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        v2.training_indices(rng, n=3, batch_size=2, weights=np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        v2.training_indices(rng, n=3, batch_size=2, weights=np.array([1.0, 1.0]))


def test_prior_draws_deterministic_per_stream():
    class M:
        latent_dim, input_dim = 16, 192
    latent_a, noise_a = v2._prior_draws(M(), v2.SPEC, 0, 8)
    latent_b, noise_b = v2._prior_draws(M(), v2.SPEC, 0, 8)
    latent_c, _ = v2._prior_draws(M(), v2.SPEC, 1, 8)
    np.testing.assert_array_equal(latent_a, latent_b)
    np.testing.assert_array_equal(noise_a, noise_b)
    assert not np.array_equal(latent_a, latent_c)


@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 7.0]])
@pytest.mark.parametrize("reassignment_available", [True, False])
def test_evaluation_iw_summary_uses_window_weights(tmp_path, monkeypatch, weights, reassignment_available):
    class ConstantDecoder:
        latent_dim, input_dim = 16, 192
        b = residual_variance = observation_variance = 1.0

        def encode(self, x):
            return np.zeros((len(x), self.latent_dim)), np.zeros((len(x), self.latent_dim))

        def decode_mean(self, z):
            return np.zeros((len(z), self.input_dim))

        decode_residual = decode_mean

    rng = np.random.default_rng(5)
    data = {"train": rng.standard_normal((32, 192)),
            "test": np.repeat(np.array([[0.0], [1.0], [2.0]]), 192, axis=1)}
    if weights is not None:
        data["eval_weights"] = np.asarray(weights)
    if not reassignment_available:
        data.update(reassignment_available=np.array(False),
                    reassignment_donors=np.empty(0, dtype=np.int64))
    spec = dict(v2.SPEC, evaluation_prior_samples=8, iw_samples=8,
                linear_control_k=[64])
    monkeypatch.setattr(v2, "SPEC", spec)
    result = v2._evaluate(tmp_path, ConstantDecoder(), data, spec, "cpu-fixture", 5)
    expected_per_window = -0.5 * (np.square(data["test"]).sum(axis=1)
                                  + 192 * np.log(2 * np.pi))
    assert result["iw_mean_log_score"] == pytest.approx(
        np.average(expected_per_window, weights=weights), abs=1e-10)
    with np.load(tmp_path / "evaluation-final.npz") as saved:
        np.testing.assert_allclose(saved["iw_log_scores"],
                                   np.broadcast_to(expected_per_window, (3, 3)),
                                   rtol=0, atol=1e-10)
        measure = np.ones(3) if weights is None else np.asarray(weights)
        measure = measure / measure.sum()
        np.testing.assert_allclose(saved["evaluation_weights"], measure)
        for prefix in ("energy", "increment_energy"):
            reduced = saved[f"{prefix}_per_window"] @ measure
            np.testing.assert_allclose(saved[f"{prefix}_streams"], reduced)
            np.testing.assert_allclose(result[f"{prefix}_streams"], reduced)
        for summary_key, array_key in {
                "posterior_kl_nats_per_window": "kl",
                "posterior_expected_nll_nats_per_window": "posterior_expected_nll",
                "negative_elbo_nats_per_window": "negative_elbo_analytic_kl",
                "posterior_mean_mse": "posterior_mean_mse",
                "posterior_expected_mse": "posterior_expected_mse",
                "sample_jensen_gap_nats_per_window": "jensen_gap"}.items():
            assert result[summary_key] == pytest.approx(np.mean(saved[array_key] @ measure))
        assert saved["prior_samples"].shape == saved["prior_means"].shape == (3, 8, 192)
        np.testing.assert_allclose(saved["log_weights"],
                                   saved["sample_log_likelihood"] - saved["sample_logq_minus_logprior"])
        np.testing.assert_allclose(saved["kl"], saved["kl_by_latent"].sum(axis=-1))
        np.testing.assert_allclose(saved["negative_elbo_analytic_kl"],
                                   saved["posterior_expected_nll"] + saved["kl"])
        np.testing.assert_allclose(saved["linear_k64_error"],
                                   saved["linear_k64_iw_log_score"] - saved["linear_k64_exact_log_marginal"])
        assert result["reassignment_available"] is reassignment_available
        if not reassignment_available:
            assert result["reassignment_gain_weighted_mean"] is None
            assert saved["reassignment_gain"].shape == saved["reassignment_donors"].shape == (0,)
    v2._check_evaluation(tmp_path / "evaluation-final.npz", result, data, 1.0)
    if not reassignment_available:
        with pytest.raises(ValueError, match="cannot report a numerical gain"):
            v2._check_evaluation(tmp_path / "evaluation-final.npz",
                                 dict(result, reassignment_gain_weighted_mean=0.0), data, 1.0)


@pytest.mark.parametrize("mutation", [
    "schema", "purpose", "spec", "job_missing", "job_extra", "job_duplicate", "job_order",
    "parameters_missing", "variance", "clipping", "datasets_missing", "datasets_extra",
    "code_missing", "code_extra", "source_extra", "prepared_extra", "source_hash",
])
def test_stage_is_closed_identity(tmp_path, mutation):
    root = _stage_with_prepared(tmp_path)
    manifest = v2.read_json(root / "stage_manifest.json")
    if mutation == "schema":
        manifest["schema_version"] = 2
    elif mutation == "purpose":
        manifest["purpose"] = "research"
    elif mutation == "spec":
        manifest["specification"]["gradient_clipping_enabled"] = 1
    elif mutation == "job_missing":
        manifest["jobs"].pop()
    elif mutation == "job_extra":
        manifest["jobs"].append(["mixture_v2", "raw", 42, "low"])
    elif mutation == "job_duplicate":
        manifest["jobs"][-1] = manifest["jobs"][0]
    elif mutation == "job_order":
        manifest["jobs"].reverse()
    elif mutation == "parameters_missing":
        manifest["job_parameters"].pop("gaussian_v2_base_42_low")
    elif mutation in ("variance", "clipping"):
        key, value = (("observation_variance", 1.0) if mutation == "variance"
                      else ("gradient_clipping_enabled", False))
        manifest["job_parameters"]["gaussian_v2_base_42_low"][key] = value
    elif mutation == "datasets_missing":
        manifest["prepared_sha256"].clear()
    elif mutation == "datasets_extra":
        manifest["prepared_sha256"]["../outside"] = "0" * 64
    elif mutation == "code_missing":
        manifest["code_sha256"].pop(v2.DEPS_V1[0])
    elif mutation == "code_extra":
        manifest["code_sha256"]["../../outside.py"] = "0" * 64
    elif mutation == "source_extra":
        (root / "source" / "src" / "extra.py").write_text("pass\n")
    elif mutation == "prepared_extra":
        (root / "prepared" / "extra.npz").write_bytes(b"not a dataset")
    elif mutation == "source_hash":
        (root / "source" / "src" / "prigm_metrics.py").write_text("pass\n")
    v2.atomic_json(root / "stage_manifest.json", manifest)
    with pytest.raises(ValueError):
        v2.verify_stage(root)


def test_build_rejects_research_purpose_without_creating_root(tmp_path):
    root = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="numerical_validation"):
        v2.build_stage(root, "numerical_null", tmp_path, purpose="core")
    assert not root.exists()


@pytest.mark.parametrize("condition,seed,label,variance", [
    ("raw", 42, "low", 0.0625), ("base", 7, "low", 0.0625),
    ("base", 42, "v1", 0.0625), ("base", 42, "high", 0.0625),
    ("base", 42, "low", float("nan")), ("base", 42, "low", float("inf")),
])
def test_train_rejects_cli_before_job_directory_or_gpu(tmp_path, monkeypatch, condition, seed, label, variance):
    root = _stage_with_prepared(tmp_path)
    monkeypatch.setattr(v2, "gpu_runtime", lambda *_: pytest.fail("GPU must not be touched"))
    with pytest.raises(ValueError):
        v2.train(root, "gaussian_v2", condition, seed, variance, label, "fixture-GPU")
    assert list((root / "jobs").iterdir()) == []


@pytest.mark.parametrize("via_cli", [False, True])
def test_override_variance_must_match_label(tmp_path, monkeypatch, via_cli):
    root = _stage_with_prepared(tmp_path)
    if via_cli:
        monkeypatch.setattr(sys, "argv", ["runner", "train", "--stage-root", str(root),
                                         "--dataset", "gaussian_v2", "--condition", "base", "--seed", "42",
                                         "--variance-label", "low", "--observation-variance", "0.1",
                                         "--gpu-uuid", "fixture-GPU"])
        call = v2.main
    else:
        call = lambda: v2.train(root, "gaussian_v2", "base", 42, 0.1, "low", "fixture-GPU")
    with pytest.raises(ValueError, match="variance"):
        call()
    assert list((root / "jobs").iterdir()) == []


def test_existing_job_lock_is_exclusive_and_failed_attempts_are_preserved(tmp_path, monkeypatch):
    root = _stage_with_prepared(tmp_path)
    job = root / "jobs" / "gaussian_v2_base_42_low"
    job.mkdir()
    def unavailable(_):
        raise RuntimeError("fixture runtime unavailable")
    monkeypatch.setattr(v2, "gpu_runtime", unavailable)
    with (job / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            v2.train(root, "gaussian_v2", "base", 42, 0.0625, "low", "fixture-GPU")
        assert list(job.glob("attempt-*")) == []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fixture runtime"):
            v2.train(root, "gaussian_v2", "base", 42, 0.0625, "low", "fixture-GPU")
    attempts = list(job.glob("attempt-*"))
    assert len(attempts) == 2
    assert all((attempt / "failure.json").is_file() for attempt in attempts)
    assert not (job / "complete.json").exists()


@pytest.mark.parametrize("field,value", [
    ("attempt", "attempt-" + "b" * 32), ("identifier", "another-job"), ("run_id", "another-run"),
    ("condition", "canonical"), ("seed", 7), ("observation_variance", 1.0),
    ("model_storage_contract", 2), ("head_scale", 0.2), ("head_coordinates", "r"),
    ("body_learning_rate", 0.001), ("head_learning_rate", 0.005), ("epsilon", 1e-6),
    ("epsilon_float32", 1e-7), ("gradient_clipping_enabled", False), ("model_dtype", "float64"),
    ("stage_manifest_sha256", "0" * 64), ("code_sha256", {}), ("prepared_sha256", "0" * 64),
])
def test_completion_rejects_rehashed_configuration_before_backend(tmp_path, monkeypatch, field, value):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    configuration = v2.read_json(attempt / "configuration.json")
    configuration[field] = value
    v2.atomic_json(attempt / "configuration.json", configuration)
    _rehash_marker(root, attempt)
    monkeypatch.setattr(v2, "_check_weights", lambda *_: pytest.fail("Backend must not be reached"))
    with pytest.raises(ValueError, match="configuration"):
        v2.verified_completion(root, attempt.parent.name)


@pytest.mark.parametrize("mutation", [
    "one_row", "missing", "duplicate", "order", "old_fields", "nonfinite", "validation_missing",
    "validation_extra", "validation_step", "later_tie", "clip_flag", "clip_scale", "clip_active",
    "probe_missing", "probe_extra", "probe_nonfinite", "progress", "score", "selected_zero",
    "selected_extra", "summary_id", "checkpoint_id", "checkpoint_step", "checkpoint_hash", "inventory",
])
def test_completion_rejects_rehashed_history_and_identity(tmp_path, monkeypatch, mutation):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root, selected_step=3000)
    rows = v2._rows(attempt / "history.jsonl")
    probes = v2._rows(attempt / "probes.jsonl")
    summary = v2.read_json(attempt / "summary.json")
    if mutation == "one_row":
        rows = rows[:1]
    elif mutation == "missing":
        rows.pop(2)
    elif mutation == "duplicate":
        rows.insert(2, rows[1])
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "old_fields":
        rows[0]["loss"] = rows[0].pop("pre_update_loss")
    elif mutation == "nonfinite":
        rows[0]["pre_update_minibatch_kl"] = float("nan")
    elif mutation == "validation_missing":
        rows = [row for row in rows if not ("validation_energy" in row and row["step"] == 4000)]
    elif mutation == "validation_extra":
        rows.append(rows[-1])
    elif mutation == "validation_step":
        rows[1000]["step"] = 999
    elif mutation == "later_tie":
        next(row for row in rows if row.get("validation_energy") == 1.0 and row["step"] == 4000)["best_step"] = 4000
    elif mutation == "clip_flag":
        rows[0]["clipping_enabled"] = False
    elif mutation == "clip_scale":
        rows[0]["clipping_scale"] = 0.6
    elif mutation == "clip_active":
        rows[0]["clipping_active"] = False
    elif mutation == "probe_missing":
        probes.pop()
    elif mutation == "probe_extra":
        probes.append(probes[-1])
    elif mutation == "probe_nonfinite":
        probes[0]["head_only_rms"] = float("inf")
    elif mutation == "progress":
        progress = v2.read_json(attempt / "progress.json")
        progress["step"] = 9999
        v2.atomic_json(attempt / "progress.json", progress)
    elif mutation == "score":
        summary["best_score"] = 1.01
    elif mutation == "selected_zero":
        summary["selected_step"] = 0
    elif mutation == "selected_extra":
        (attempt / "unexpected.npz").write_bytes(b"unbound")
    elif mutation == "summary_id":
        summary["identifier"] = "other"
    elif mutation == "checkpoint_id":
        summary["final"]["checkpoint"] = "selected"
    elif mutation == "checkpoint_step":
        summary["final"]["step"] = 9000
    elif mutation == "checkpoint_hash":
        summary["final"]["weights_sha256"] = "0" * 64
    elif mutation == "inventory":
        (attempt / "failure.json").write_text("{}")
    _write_rows(attempt / "history.jsonl", rows)
    _write_rows(attempt / "probes.jsonl", probes)
    v2.atomic_json(attempt / "summary.json", summary)
    _rehash_marker(root, attempt)
    monkeypatch.setattr(v2, "_check_weights", lambda *_: pytest.fail("Backend must not be reached"))
    with pytest.raises(ValueError):
        v2.verified_completion(root, attempt.parent.name)


@pytest.mark.parametrize("mutation", ["array_shape", "array_missing", "nonfinite", "weights", "energy", "iw", "kl", "linear", "summary"])
def test_completion_rejects_rehashed_saved_component_disagreement(tmp_path, monkeypatch, mutation):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    path = attempt / "evaluation-final.npz"
    with np.load(path) as store:
        arrays = dict(store)
    summary = v2.read_json(attempt / "summary.json")
    if mutation == "array_shape":
        arrays["prior_means"] = arrays["prior_means"][:, :-1]
    elif mutation == "array_missing":
        arrays.pop("sample_logq_minus_logprior")
    elif mutation == "nonfinite":
        arrays["posterior_mean_mse"][0, 0] = np.inf
    elif mutation == "summary":
        summary["final"]["posterior_expected_nll_nats_per_window"] += 1.0
    else:
        key = {"weights": "evaluation_weights", "energy": "energy_streams", "iw": "iw_log_scores",
               "kl": "kl", "linear": "linear_k64_exact_log_marginal"}[mutation]
        arrays[key] = arrays[key] + 0.1
    np.savez_compressed(path, **arrays)
    summary["final"]["evaluation_sha256"] = v2.sha256(path)
    v2.atomic_json(attempt / "summary.json", summary)
    _rehash_marker(root, attempt)
    monkeypatch.setattr(v2, "_check_weights", lambda *_: pytest.fail("Backend must not be reached"))
    with pytest.raises(ValueError):
        v2.verified_completion(root, attempt.parent.name)


def test_final_selection_forbids_extra_selected_evaluation(tmp_path, monkeypatch):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    (attempt / "evaluation-selected.npz").write_bytes((attempt / "evaluation-final.npz").read_bytes())
    _rehash_marker(root, attempt)
    monkeypatch.setattr(v2, "_check_weights", lambda *_: pytest.fail("Backend must not be reached"))
    with pytest.raises(ValueError, match="inventory"):
        v2.verified_completion(root, attempt.parent.name)


@pytest.mark.parametrize("mutation", ["schema", "run", "attempt", "hash", "extra", "runtime"])
def test_completion_marker_identity_rejected_before_backend(tmp_path, monkeypatch, mutation):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    marker = _rehash_marker(root, attempt)
    if mutation == "schema":
        marker["schema_version"] = 2
    elif mutation == "run":
        marker["run_id"] = "wrong"
    elif mutation == "attempt":
        marker["attempt"] = "../outside"
    elif mutation == "hash":
        marker["artifact_sha256"]["history.jsonl"] = "0" * 64
    elif mutation == "extra":
        marker["artifact_sha256"]["../outside"] = "0" * 64
    elif mutation == "runtime":
        configuration = v2.read_json(attempt / "configuration.json")
        configuration["runtime"] = {}
        v2.atomic_json(attempt / "configuration.json", configuration)
        marker = _rehash_marker(root, attempt)
    v2.atomic_json(attempt.parent / "complete.json", marker)
    monkeypatch.setattr(v2, "_check_weights", lambda *_: pytest.fail("Backend must not be reached"))
    with pytest.raises(ValueError):
        v2.verified_completion(root, attempt.parent.name)


def test_failed_validation_does_not_publish_either_marker(tmp_path):
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root)
    with pytest.raises(ValueError, match="genuine HDF5"):
        v2._publish_completion(root, attempt.parent.name, attempt)
    assert not (attempt / "complete.json").exists()
    assert not (attempt.parent / "complete.json").exists()


@pytest.mark.parametrize("mutation", ["attribute", "nonfinite", "shape", "missing", "extra", "dtype", "link", "renamed_group"])
def test_hdf5_contract_requires_finite_correct_tensors(tmp_path, mutation):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "weights.weights.h5"
    _hdf5_weights(path)
    with h5py.File(path, "r+") as store:
        if mutation == "attribute":
            store.attrs["storage_contract"] = 2
        elif mutation == "nonfinite":
            store["layers/dense/vars/0"][0, 0] = np.inf
        elif mutation in ("shape", "dtype"):
            del store["layers/dense/vars/0"]
            store.create_dataset("layers/dense/vars/0", data=np.zeros(
                (191, 128) if mutation == "shape" else (192, 128),
                dtype=np.float32 if mutation == "shape" else np.float64))
        elif mutation == "missing":
            del store["layers/dense/vars/1"]
        elif mutation == "extra":
            store.create_dataset("extra", data=np.zeros(1, dtype=np.float32))
        elif mutation == "link":
            store["external"] = h5py.ExternalLink("outside.h5", "/")
        elif mutation == "renamed_group":
            store.move("mean_head", "renamed_mean_head")
    config = {"model_storage_contract": 3, "head_coordinates": "w", "head_scale": 1.0,
              "observation_variance": 0.0625}
    with pytest.raises(ValueError):
        v2._check_weights(path, config)


def test_hdf5_reader_accepts_actual_tensorflow_model_save(tmp_path):
    pytest.importorskip("h5py")
    tf = pytest.importorskip("tensorflow")
    from prigm_v2_model import BypassVAE
    with tf.device("/CPU:0"):
        model = BypassVAE(condition="canonical", observation_variance=0.0625)
        path = tmp_path / "actual.weights.h5"
        model.save_weights(path)
    v2._check_weights(path, {"model_storage_contract": model.storage_contract,
                            "head_coordinates": model.head_coordinates,
                            "head_scale": model.head_scale, "observation_variance": model.observation_variance})


def _dart_inputs(tmp_path, names=("dart_44402_2018",)):
    """Analytical fixtures only; not evidence of raw-record qualification."""
    from prigm_v2_data import same_phase_donors
    directory = tmp_path / "dart-inputs"
    directory.mkdir()
    records = {}
    origin = 1514764800
    rng = np.random.default_rng(71)
    for index, name in enumerate(names):
        records[name] = {
            "station": name[5:10], "interval_start": "2018-01-01", "interval_end": "2019-01-01",
            "raw_sha256": "a" * 64, "raw_filename": f"{name[5:10]}t2018.txt",
            "deployment": {"fixture": True}, "qualification": {"fixture": True},
            "selection_status": "retrospectively_selected_fixture",
        }
        data = {"dataset_name": np.array(name), "mean": np.float64(4000), "scale": np.float64(0.5)}
        for split, offset, count in (("train", 0, 60), ("validation", 221, 24), ("test", 294, 24)):
            starts = origin + offset * 86400 + np.arange(count, dtype=np.int64) * (52 * 3600)
            if split == "train":
                starts = np.sort(np.append(starts, origin + 4 * 3600))
            bins = starts % 86400 // 14400
            counts = np.bincount(bins, minlength=6)
            data[split] = rng.normal(scale=0.02 * (index + 1), size=(len(starts), 192))
            data[f"starts_{split}"] = starts
            data[f"source_starts_{split}"] = (starts - origin) // 900
            data["eval_weights" if split == "test" else f"{split}_weights"] = 1 / (6 * counts[bins])
        donors = same_phase_donors(data["starts_test"], v2.SPEC["reassignment_seed"],
                                  data["source_starts_test"], origin)
        data.update(reassignment_block_origin=np.int64(origin),
                    reassignment_available=np.bool_(donors is not None),
                    reassignment_donors=donors if donors is not None else np.empty(0, dtype=np.int64))
        np.savez_compressed(directory / f"{name}.npz", **data)
    v2.atomic_json(directory / "records.json", {"schema_version": 1, "records": records})
    return directory


def _dart_stage(tmp_path, names=("dart_44402_2018",)):
    directory = _dart_inputs(tmp_path, names)
    root = tmp_path / "dart-stage"
    v2.build_dart_stage(root, directory)
    return root


def test_dart_build_cli_grid_and_raw_identity(tmp_path, monkeypatch):
    from prigm_v2_metrics import select_noise_variance
    before = json.dumps(v2.SPEC, sort_keys=True)
    names = ("dart_44402_2018", "dart_55555_2018", "dart_55556_2018")
    directory = _dart_inputs(tmp_path, names)
    root = tmp_path / "dart-stage"
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setattr(sys, "argv", ["runner", "build-dart", "--run-root", str(root),
                                     "--dataset-path", str(directory)])
    v2.main()
    manifest = v2.verify_stage(root)
    assert manifest["schema_version"] == 4 and manifest["purpose"] == v2.DART_PURPOSE
    assert manifest["stage"] == "dart_record_extension"
    assert len(manifest["jobs"]) == 36
    assert set(v2.STAGES) == {"numerical_null"}
    assert (root / "prepared" / "records.json").read_bytes() == (directory / "records.json").read_bytes()
    assert manifest["specification"] == dict(v2.SPEC, purpose=v2.DART_PURPOSE)
    for name in names:
        with np.load(directory / f"{name}.npz") as store:
            selection = select_noise_variance(store["train"], store["train_weights"],
                                               store["validation"], store["validation_weights"],
                                               grid=(1e-4, 5e-4, 2.5e-3, 1.25e-2, 6.25e-2), rank=16)
        assert manifest["variance_selection"][name] == {
            key: selection[key] for key in ("selection", "grid_results", "boundary_selection")}
        for seed in (42, 1729, 31415):
            for label, multiplier in (("low", 1), ("high", 25)):
                for condition in ("base", "raw"):
                    identifier = v2.job_id(name, condition, seed, label)
                    config = v2._configuration_identity(root, manifest, identifier, "attempt-fixture")
                    assert config["schema_version"] == 4 and config["model_storage_contract"] == 3
                    assert config["purpose"] == v2.DART_PURPOSE
                    assert config["head_coordinates"] == ("w" if condition == "base" else "r")
                    assert config["head_scale"] == (1.0 if condition == "base" else 1.0 - 0.8)
                    assert config["observation_variance"] == selection["selection"]["variance"] * multiplier
                    assert config["head_learning_rate"] == config["body_learning_rate"] == float(np.float32(0.001))
                    assert config["gradient_clipping_enabled"] is True
    isolated = v2._resolved_spec(v2.DART_STAGE)
    isolated["adam"]["epsilon"] = 0
    assert json.dumps(v2.SPEC, sort_keys=True) == before


@pytest.mark.parametrize("mutation", ["primary", "too_many", "old_site", "wrong_year", "station",
                                      "interval", "schema", "extra", "missing", "hash", "deployment",
                                      "input_extra", "input_symlink"])
def test_dart_rejects_metadata_before_build(tmp_path, mutation):
    directory = _dart_inputs(tmp_path)
    metadata = v2.read_json(directory / "records.json")
    record = metadata["records"]["dart_44402_2018"]
    if mutation in ("primary", "wrong_year"):
        metadata["records"]["dart_44402_2017"] = metadata["records"].pop("dart_44402_2018")
    elif mutation == "too_many":
        for station in ("55555", "55556", "55557"):
            metadata["records"][f"dart_{station}_2018"] = dict(record, station=station)
    elif mutation == "old_site":
        metadata["records"]["dart_23461_2018"] = dict(record, station="23461")
    elif mutation == "station":
        record["station"] = "55555"
    elif mutation == "interval":
        record["interval_end"] = "2018-12-31"
    elif mutation == "schema":
        metadata["schema_version"] = True
    elif mutation == "extra":
        record["eligible"] = True
    elif mutation == "missing":
        record.pop("selection_status")
    elif mutation == "hash":
        record["raw_sha256"] = "not a hash"
    elif mutation == "deployment":
        record["deployment"] = "qualified"
    elif mutation == "input_extra":
        (directory / "unbound.txt").write_text("extra")
    else:
        (directory / "unbound.npz").symlink_to(directory / "dart_44402_2018.npz")
    v2.atomic_json(directory / "records.json", metadata)
    root = tmp_path / "rejected"
    with pytest.raises(ValueError):
        v2.build_dart_stage(root, directory)
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["width", "nonfinite", "dataset", "scale", "mean", "weights",
                                      "mass", "starts", "source_order", "source_overlap", "time_overlap",
                                      "grid", "packs", "counts", "donors"])
def test_dart_rejects_rehashed_prepared_arrays(tmp_path, mutation):
    root = _dart_stage(tmp_path)
    path = root / "prepared" / "dart_44402_2018.npz"
    with np.load(path) as store:
        data = dict(store)
    if mutation == "width":
        data["train"] = data["train"][:, :-1]
    elif mutation == "nonfinite":
        data["validation"][0, 0] = np.inf
    elif mutation == "dataset":
        data["dataset_name"] = np.array("dart_55555_2018")
    elif mutation == "scale":
        data["scale"] = np.array(0.0)
    elif mutation == "mean":
        data["mean"] = np.array(np.nan)
    elif mutation == "weights":
        data["train_weights"][:] = 1 / len(data["train_weights"])
    elif mutation == "mass":
        data["eval_weights"] *= 2
    elif mutation == "starts":
        data["starts_train"][1] = data["starts_train"][0]
    elif mutation == "source_order":
        data["source_starts_train"][1] = 0
    elif mutation == "source_overlap":
        data["source_starts_validation"] -= data["source_starts_validation"][0] - data["source_starts_train"][-1] - 100
    elif mutation == "time_overlap":
        data["starts_validation"] -= 150 * 86400
    elif mutation == "grid":
        data["starts_train"] += 900
    elif mutation == "packs":
        data["starts_train"] = 1514764800 + np.arange(len(data["train"])) * 14400
        bins = data["starts_train"] % 86400 // 14400
        data["train_weights"] = 1 / (6 * np.bincount(bins, minlength=6)[bins])
    elif mutation == "counts":
        data["train"] = data["train"][:-1]
    else:
        data["reassignment_donors"][0] = 0
    np.savez_compressed(path, **data)
    manifest = v2.read_json(root / "stage_manifest.json")
    manifest["prepared_sha256"]["dart_44402_2018"] = v2.sha256(path)
    v2.atomic_json(root / "stage_manifest.json", manifest)
    with pytest.raises(ValueError):
        v2.verify_stage(root)


@pytest.mark.parametrize("mutation", ["metadata_hash", "prepared_hash", "grid_score", "grid_missing",
                                      "selection", "boundary", "variance", "matrix", "counts", "schema",
                                      "purpose", "old_stage", "metadata_extra", "prepared_extra"])
def test_dart_manifest_tamper_rejected(tmp_path, mutation):
    root = _dart_stage(tmp_path)
    manifest = v2.read_json(root / "stage_manifest.json")
    name = "dart_44402_2018"
    selection = manifest["variance_selection"][name]
    if mutation == "metadata_hash":
        with (root / "prepared" / "records.json").open("a") as stream:
            stream.write(" ")
    elif mutation == "prepared_hash":
        with (root / "prepared" / f"{name}.npz").open("ab") as stream:
            stream.write(b"tamper")
    elif mutation == "grid_score":
        selection["grid_results"][0]["weighted_validation_log_density"] += 0.1
    elif mutation == "grid_missing":
        selection["grid_results"].pop()
    elif mutation == "selection":
        selection["selection"] = selection["grid_results"][-1]
    elif mutation == "boundary":
        selection["boundary_selection"] = not selection["boundary_selection"]
    elif mutation == "variance":
        manifest["job_parameters"][f"{name}_raw_42_low"]["observation_variance"] *= 2
    elif mutation == "matrix":
        manifest["jobs"][0][0] = "dart_55555_2018"
    elif mutation == "counts":
        manifest["record_summaries"][name]["train"]["windows"] += 1
    elif mutation == "schema":
        manifest["schema_version"] = 3
    elif mutation == "purpose":
        manifest["purpose"] = "numerical_validation"
    elif mutation == "old_stage":
        manifest["stage"] = "core"
    elif mutation == "metadata_extra":
        metadata = v2.read_json(root / "prepared" / "records.json")
        metadata["extra"] = "unbound"
        v2.atomic_json(root / "prepared" / "records.json", metadata)
        manifest["records_sha256"] = v2.sha256(root / "prepared" / "records.json")
    else:
        (root / "prepared" / "extra.npz").write_bytes(b"extra")
    v2.atomic_json(root / "stage_manifest.json", manifest)
    with pytest.raises(ValueError):
        v2.verify_stage(root)


def test_dart_entrypoints_enforce_membership_and_variance(tmp_path, monkeypatch):
    root = _dart_stage(tmp_path)
    monkeypatch.setattr(v2, "gpu_runtime", lambda *_: pytest.fail("GPU must not be touched"))
    manifest = v2.verify_stage(root)
    for dataset, condition, seed, variance in (("dart_55555_2018", "base", 42, 0.1),
                                              ("dart_44402_2018", "canonical", 42, 0.1),
                                              ("dart_44402_2018", "base", 7, 0.1),
                                              ("dart_44402_2018", "raw", 42, 0.1)):
        with pytest.raises(ValueError):
            v2.train(root, dataset, condition, seed, variance, "low", "fixture")
    with pytest.raises(ValueError):
        v2.verified_completion(root, "dart_55555_2018_base_42_low")
    with pytest.raises(ValueError):
        v2.check_attempt(root, "dart_55555_2018_base_42_low", root / "outside")
    assert not list((root / "jobs").iterdir())
    seen = []
    monkeypatch.setattr(v2, "train", lambda *args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["runner", "train", "--stage-root", str(root), "--dataset",
                                     "dart_44402_2018", "--condition", "raw", "--seed", "42",
                                     "--variance-label", "low", "--gpu-uuid", "fixture"])
    v2.main()
    assert seen[0][4] == manifest["job_parameters"]["dart_44402_2018_raw_42_low"]["observation_variance"]
    with pytest.raises(FileExistsError):
        v2.build_dart_stage(root, tmp_path / "dart-inputs")
    for old_stage in ("core", "scientific", "dart_record_extension"):
        with pytest.raises(ValueError):
            v2.build_stage(tmp_path / "old", old_stage, tmp_path / "dart-inputs")


def test_dart_completion_schema_without_tensor_backend(tmp_path, monkeypatch):
    """Exercise real history/evaluation closure; tensor inspection has a separate HDF5 test."""
    root = _dart_stage(tmp_path)
    attempt = _write_attempt(root, "dart_44402_2018_raw_1729_high")
    inspected = []
    monkeypatch.setattr(v2, "_check_weights", lambda path, config: inspected.append(config))
    v2._publish_completion(root, attempt.parent.name, attempt)
    assert v2.verified_completion(root, attempt.parent.name) == attempt
    assert v2.read_json(attempt.parent / "complete.json")["schema_version"] == 4
    assert inspected and all(config["head_coordinates"] == "r" for config in inspected)
    marker = v2.read_json(attempt.parent / "complete.json")
    v2.atomic_json(attempt.parent / "complete.json", dict(marker, schema_version=3))
    with pytest.raises(ValueError, match="completion identity"):
        v2.verified_completion(root, attempt.parent.name)


def test_history_and_evaluation_accept_explicit_spec_without_global_mutation(tmp_path):
    before = json.dumps(v2.SPEC, sort_keys=True)
    spec = dict(v2._resolved_spec(v2.DART_STAGE), max_steps=1, validation_every=1,
                probe_steps=[1], evaluation_prior_samples=8, iw_samples=8, linear_control_k=[64])
    _write_rows(tmp_path / "history.jsonl", [
        {"step": 1, "pre_update_loss": 1.0, "pre_update_minibatch_kl": 0.0,
         "gradient_norm": 0.0, "clipping_enabled": True, "clipping_scale": 1.0, "clipping_active": False},
        {"step": 1, "validation_energy": 1.0, "best_step": 1}])
    _write_rows(tmp_path / "probes.jsonl", [
        {"step": 1, "full_decoder_rms": 0.0, "head_only_rms": 0.0, "encoder_symmetric_kl": 0.0}])
    v2.atomic_json(tmp_path / "progress.json", {"step": 1, "best_step": 1, "best_score": 1.0,
                                               "elapsed_seconds": 1.0})
    assert v2._check_history(tmp_path, spec) == (1, 1.0)
    rng = np.random.default_rng(3)
    data = {"dataset_name": np.array("dart_44402_2018"),
            "train": rng.normal(size=(32, 192)), "test": rng.normal(size=(4, 192))}
    summary = v2._evaluate(tmp_path, _ConstantDecoder(), data, spec, "fixture", 42)
    v2._check_evaluation(tmp_path / "evaluation-final.npz", summary, data, 0.0625, spec)
    assert json.dumps(v2.SPEC, sort_keys=True) == before


def test_dart_provenance_filename_is_never_opened(tmp_path):
    directory = _dart_inputs(tmp_path)
    metadata = v2.read_json(directory / "records.json")
    metadata["records"]["dart_44402_2018"]["raw_filename"] = "../../not-an-input.txt"
    v2.atomic_json(directory / "records.json", metadata)
    root = tmp_path / "dart-stage"
    v2.build_dart_stage(root, directory)
    assert len(v2.verify_stage(root)["jobs"]) == 12


def test_dart_raw_completion_and_old_weights(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = _dart_stage(tmp_path)
    attempt = _write_attempt(root, "dart_44402_2018_raw_42_low", genuine_weights=True)
    completion = v2.check_attempt(root, attempt.parent.name, attempt)
    assert completion["schema_version"] == 4
    v2.atomic_json(attempt / "complete.json", completion)
    v2.atomic_json(attempt.parent / "complete.json", completion)
    assert v2.verified_completion(root, attempt.parent.name) == attempt
    old_marker = dict(completion, schema_version=3)
    v2.atomic_json(attempt.parent / "complete.json", old_marker)
    with pytest.raises(ValueError, match="completion identity"):
        v2.verified_completion(root, attempt.parent.name)
    v2.atomic_json(attempt.parent / "complete.json", completion)
    with h5py.File(attempt / "final.weights.h5", "r+") as store:
        store.attrs["storage_contract"] = 2
    configuration = v2.read_json(attempt / "configuration.json")
    with pytest.raises(ValueError, match="storage contract"):
        v2._check_weights(attempt / "final.weights.h5", configuration)


@pytest.fixture
def disposable_campaign(tmp_path):
    """Only this fixture substitutes the worker entrypoint; production stays closed."""
    root = _stage_with_prepared(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text('''import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
folder = Path(sys.argv[1])
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
(folder / (gpu + ".pid")).write_text(str(os.getpid()))
if sys.argv[2] == "failure" and gpu == "fixture-0":
    deadline = time.monotonic() + 10
    while len(list(folder.glob("*.pid"))) != 4:
        if time.monotonic() > deadline:
            sys.exit(8)
        time.sleep(0.01)
    sys.exit(7)
while True:
    time.sleep(1)
''')
    driver = tmp_path / "campaign_driver.py"
    driver.write_text('''import os, signal, subprocess, sys
sys.path.insert(0, sys.argv[1])
import prigm_v2_experiment as runner
original = subprocess.Popen
def fixture_worker(command, **kwargs):
    return original([sys.executable, sys.argv[3], sys.argv[4], sys.argv[5]], **kwargs)
runner.subprocess.Popen = fixture_worker
if sys.argv[5] == "signal_under_lock":
    original_lock, original_wait = runner.threading.Lock, runner.wait
    captured = []
    def capture_lock():
        lock = original_lock()
        if not captured:
            captured.append(lock)
        return lock
    def interrupt_while_main_owns_guard(*args, **kwargs):
        with captured[0]:
            signal.raise_signal(signal.SIGINT)
        return original_wait(*args, **kwargs)
    runner.threading.Lock = capture_lock
    runner.wait = interrupt_while_main_owns_guard
runner.campaign(sys.argv[2], ["fixture-" + str(i) for i in range(4)])
''')
    processes = []
    def launch(mode="hang"):
        folder = tmp_path / ("children-" + str(len(processes)))
        folder.mkdir()
        process = subprocess.Popen(
            [sys.executable, str(driver), str(SRC), str(root), str(worker), str(folder), mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1"))
        processes.append(process)
        return process, folder
    yield root, launch
    for process in processes:
        # This session contains only the disposable driver and its fixture children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)


def _wait_for_children(process, folder):
    deadline = time.monotonic() + 10
    while len(list(folder.glob("*.pid"))) != 4:
        if process.poll() is not None or time.monotonic() > deadline:
            pytest.fail("Disposable campaign did not start four children")
        time.sleep(0.01)
    return [int(path.read_text()) for path in folder.glob("*.pid")]


def _assert_reaped(pids):
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _assert_campaign_lock_free(root):
    with (root / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_campaign_failure_reaps_other_workers_promptly(disposable_campaign):
    root, launch = disposable_campaign
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        process, folder = launch("failure")
        pids = _wait_for_children(process, folder)
        started = time.monotonic()
        _, stderr = process.communicate(timeout=8)
        assert process.returncode != 0 and "Campaign stopped" in stderr
        assert time.monotonic() - started < 6
        _assert_reaped(pids)
        assert unrelated.poll() is None
        _assert_campaign_lock_free(root)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_campaign_interrupt_reaps_children_and_releases_lock(disposable_campaign, signum):
    root, launch = disposable_campaign
    process, folder = launch()
    pids = _wait_for_children(process, folder)
    process.send_signal(signum)
    _, stderr = process.communicate(timeout=8)
    assert process.returncode != 0 and "Campaign stopped" in stderr
    _assert_reaped(pids)
    _assert_campaign_lock_free(root)
    assert v2.read_json(root / "campaign_status.json")["interrupted"] is True


def test_campaign_signal_while_main_owns_guard_does_not_deadlock(disposable_campaign):
    root, launch = disposable_campaign
    process, folder = launch("signal_under_lock")
    _, stderr = process.communicate(timeout=8)
    assert process.returncode != 0 and "Campaign stopped" in stderr
    _assert_reaped([int(path.read_text()) for path in folder.glob("*.pid")])
    _assert_campaign_lock_free(root)


def test_duplicate_campaign_is_rejected_without_spawning_or_killing(disposable_campaign):
    root, launch = disposable_campaign
    first, folder = launch()
    pids = _wait_for_children(first, folder)
    duplicate, duplicate_folder = launch()
    _, stderr = duplicate.communicate(timeout=5)
    assert duplicate.returncode != 0 and "BlockingIOError" in stderr
    assert list(duplicate_folder.iterdir()) == []
    assert first.poll() is None
    for pid in pids:
        os.kill(pid, 0)
    first.send_signal(signal.SIGINT)
    first.communicate(timeout=8)
    _assert_reaped(pids)
    _assert_campaign_lock_free(root)


@pytest.mark.parametrize("donor_mse,accepted", [
    pytest.param(-2e-8, False, id="below-negative-tolerance"),
    pytest.param([0.0, 0.125, 0.0, 0.125], True, id="valid-negative-gain-and-zero"),
    pytest.param(-5e-9, True, id="roundoff-within-tolerance"),
    pytest.param(None, False, id="finite-components-overflow"),
])
def test_completion_checks_implied_reassignment_donor_mse(tmp_path, donor_mse, accepted):
    """Gain may be negative; own MSE plus gain must obey the MSE tolerance."""
    root = _stage_with_prepared(tmp_path)
    attempt = _write_attempt(root, genuine_weights=True)
    identifier = attempt.parent.name
    v2._publish_completion(root, identifier, attempt)
    assert v2.verified_completion(root, identifier) == attempt

    path = attempt / "evaluation-final.npz"
    with np.load(path, allow_pickle=False) as store:
        arrays = dict(store)
    if donor_mse is None:
        arrays["reassignment_own_mse"][:] = 1e308
        arrays["reassignment_gain"][:] = 1e308
        assert np.isfinite(arrays["reassignment_own_mse"]).all()
        assert np.isfinite(arrays["reassignment_gain"]).all()
        with np.errstate(over="ignore"):
            assert np.isinf(arrays["reassignment_own_mse"] + arrays["reassignment_gain"]).all()
    else:
        arrays["reassignment_gain"] = donor_mse - arrays["reassignment_own_mse"]
        assert np.all(arrays["reassignment_gain"] < 0)
        # Test safely away from -1e-8: subtracting and adding the own MSE
        # can round the reconstructed donor MSE at the tolerance boundary.
        np.testing.assert_allclose(
            arrays["reassignment_own_mse"] + arrays["reassignment_gain"],
            donor_mse, rtol=0, atol=1e-15)
    np.savez_compressed(path, **arrays)
    summary = v2.read_json(attempt / "summary.json")
    summary["final"]["reassignment_gain_weighted_mean"] = float(
        arrays["evaluation_weights"] @ arrays["reassignment_gain"])
    assert np.isfinite(summary["final"]["reassignment_gain_weighted_mean"])
    summary["final"]["evaluation_sha256"] = v2.sha256(path)
    v2.atomic_json(attempt / "summary.json", summary)
    completion = _rehash_marker(root, attempt)
    v2.atomic_json(attempt / "complete.json", completion)

    if accepted:
        assert v2.verified_completion(root, identifier) == attempt
    else:
        with np.errstate(over="ignore"), pytest.raises(ValueError):
            v2.verified_completion(root, identifier)
