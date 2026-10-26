"""Bounded CPU tests; no TensorFlow, remote access, training or frozen writes."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))
import prigm_dart_paper_evidence as evidence


@pytest.fixture(scope="module")
def metrics():
    return evidence._imports()[0]


def component_arrays(variance=0.5, n=2):
    mse = np.broadcast_to(np.linspace(.2, .4, n), (3, n)).copy()
    likelihood = np.broadcast_to((-96 * (np.log(2 * np.pi * variance) + mse / variance))[:, :, None], (3, n, 256)).copy()
    nll = -likelihood.mean(axis=2)
    kl_by_latent = np.full((3, n, 16), .125)
    kl = kl_by_latent.sum(axis=2)
    return {"log_score": likelihood[:, :, 0].copy(), "ess": np.full((3, n), 256.),
            "log_weights": likelihood.copy(), "sample_log_likelihood": likelihood,
            "sample_logq_minus_logprior": np.zeros_like(likelihood),
            "posterior_expected_nll": nll, "posterior_expected_mse": mse,
            "posterior_mean_mse": mse / 2, "kl": kl, "kl_by_latent": kl_by_latent,
            "negative_elbo_analytic_kl": nll + kl, "sample_negative_elbo": nll.copy(),
            "jensen_gap": np.zeros((3, n))}


def test_components_reuse_normalized_importance_and_weighted_reductions(metrics):
    arrays = component_arrays()
    result = evidence._components(arrays, 2, .5, np.array([.25, .75]), metrics)
    assert result["weighted"]["posterior_expected_mse"]["mean"] == pytest.approx(.35)
    assert result["uniform"]["posterior_expected_mse"]["mean"] == pytest.approx(.3)
    assert result["median_ess"] == 256
    assert result["weighted"]["kl_per_coordinate"]["mean"] == pytest.approx(2 / 192)


@pytest.mark.parametrize("key", evidence.COMPONENTS)
def test_component_identity_tamper_rejected(metrics, key):
    arrays = component_arrays()
    arrays[key].flat[0] += .1
    with pytest.raises(ValueError):
        evidence._components(arrays, 2, .5, np.array([.5, .5]), metrics)


@pytest.mark.parametrize("variance", [True, 0, -1, float("inf"), float("nan"), [1.]])
def test_invalid_variance_scalar(metrics, variance):
    with pytest.raises(ValueError):
        evidence._components(component_arrays(), 2, variance, np.full(2, .5), metrics)


def test_six_bin_weights_not_uniform_and_missing_bin_rejected():
    starts = np.array([0, 86400, 14400, 28800, 43200, 57600, 72000])
    weights = np.array([1/12, 1/12, 1/6, 1/6, 1/6, 1/6, 1/6])
    data = {"train": np.zeros((7, 192)), "starts_train": starts, "train_weights": weights}
    np.testing.assert_allclose(evidence._weights(data, "train", True), weights)
    np.testing.assert_array_equal(evidence._weights(data, "train", False), np.full(7, 1/7))
    data["train_weights"] = np.full(7, 1/7)
    with pytest.raises(ValueError, match="UTC target"):
        evidence._weights(data, "train", True)
    data["starts_train"] = np.zeros(7, dtype=int)
    with pytest.raises(ValueError, match="Missing UTC"):
        evidence._weights(data, "train", True)


def variance_arrays():
    mse = np.array([[1., 2.], [2., 3.], [3., 4.]])
    w = np.array([.25, .75])
    estimates = mse @ w
    return {"variance_refit_training_mse": mse, "variance_refit_training_weights": w,
            "variance_refit_stream_variances": estimates,
            "variance_refit_observation_variance": np.asarray(estimates.mean())}


def test_training_refit_pooling_and_resealed_npz_tamper(tmp_path):
    arrays = variance_arrays()
    assert evidence._variance_fit(arrays, arrays["variance_refit_training_weights"])["pooled"] == 2.75
    path = tmp_path / "diagnostic.npz"
    np.savez(path, **arrays)
    before = evidence._sha(path)
    arrays["variance_refit_training_mse"][0, 0] += 1
    np.savez(path, **arrays)  # A new outer checksum cannot repair the scientific identity.
    assert evidence._sha(path) != before
    with pytest.raises(ValueError, match="training variance streams"):
        evidence._variance_fit(evidence._load(path), arrays["variance_refit_training_weights"])


@pytest.mark.parametrize("shape", [(1,), (1, 1), (0,)])
def test_scalar_npz_actual_shape_not_self_declaration(shape):
    arrays = variance_arrays()
    arrays["variance_refit_observation_variance"] = np.full(shape, 2.75)
    with pytest.raises(ValueError, match="pooled training variance"):
        evidence._variance_fit(arrays, np.array([.25, .75]))


@pytest.mark.parametrize("step,label", [(6000, "selected"), (10000, "final")])
def test_selected_endpoint_alias_uses_archived_arrays(metrics, monkeypatch, tmp_path, step, label):
    data = {"test": np.zeros((6, 192)), "starts_test": np.arange(6) * 14400, "eval_weights": np.full(6, 1/6)}
    components = component_arrays(.5, n=6)
    arrays = {{"log_score": "iw_log_scores", "ess": "iw_ess"}.get(k, k): v for k, v in components.items()}
    arrays["prior_means"] = np.zeros((3, 4, 192))
    arrays["prior_samples"] = np.stack([evidence._observation_draw(means, .5, seed) for means, seed in
                                      zip(arrays["prior_means"], [62001, 62002, 62003])])
    for name, function in (("energy", metrics.energy_score), ("increment_energy", metrics.increment_energy_score)):
        arrays[name + "_per_window"] = np.stack([function(draws, data["test"]) for draws in arrays["prior_samples"]])
    opened = []
    def load(path):
        opened.append(path.name)
        return arrays
    monkeypatch.setattr(evidence, "_load", load)
    monkeypatch.setattr(evidence, "_sha", lambda path: "a" * 64)
    result = evidence._selected(tmp_path, {"selected_step": step}, data, .5,
                                {"14": np.zeros(6, dtype=int), "28": np.zeros(6, dtype=int)}, metrics)
    assert opened == [f"evaluation-{label}.npz"]
    assert result["aliases_final"] == (step == 10000)
    assert result["checkpoint"] == "selected" and result["raw_self_mse"] is None


@pytest.fixture(params=[False, True], ids=["old", "fresh"])
def replay_fixture(request, tmp_path):
    fresh = request.param
    components = component_arrays(n=6)
    data = {"test": np.zeros((6, 192)), "starts_test": np.arange(6) * 14400,
            "eval_weights": np.full(6, 1/6)}
    prefix = "stage_replay__" if fresh else ""
    arrays = {prefix + key: value.copy() for key, value in components.items()}
    for variant in ("original__", "variance_refit__"):
        arrays.update({variant + key: value.copy() for key, value in components.items()})
    if fresh:
        np.savez(tmp_path / "evaluation-final.npz", **{
            {"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key): value
            for key, value in components.items()})
        record = {"stage_evaluation_file": "evaluation-final.npz", "archived_component_replay": {
            "samples": 256, "stream_seeds": [65001, 65002, 65003],
            "max_abs_differences": dict.fromkeys(evidence.COMPONENTS, 0.),
            "reproduced_exactly": True, "rtol": 1e-5, "atol": 1e-6}}
    else:
        mse = components["posterior_mean_mse"][0]
        np.savez(tmp_path / "evaluation.npz", log_score=components["log_score"][0],
                 ess=components["ess"][0], corrected_mse=mse, self_mse=mse,
                 posterior_mean=np.broadcast_to(np.sqrt(mse)[:, None], (6, 192)))
        record = {"arm": "A", "archived_selected_replay_differences": {
            key + "_max_abs": 0. for key in ("log_score", "ess", "posterior_mean_mse")}}
    return arrays, record, tmp_path, data, .5, fresh


@pytest.mark.parametrize("key", ["posterior_mean_mse", "kl", "kl_by_latent"])
def test_replay_binds_both_diagnostic_variants(replay_fixture, metrics, key):
    arrays, *args = replay_fixture
    evidence._replay(arrays, *args, metrics)
    for prefix in ("original__", "variance_refit__"):
        arrays[prefix + key] += 1
    # Call replay directly: diagnostic KL algebra must not mask the missing link.
    with pytest.raises(ValueError, match="diagnostic deterministic replay " + key):
        evidence._replay(arrays, *args, metrics)


@pytest.mark.parametrize("key", ["posterior_mean_mse", "kl", "kl_by_latent"])
def test_replay_tolerates_float_arithmetic_and_different_mc(replay_fixture, metrics, key):
    arrays, *args = replay_fixture
    for prefix in ("original__", "variance_refit__"):
        arrays[prefix + key] += 5e-7
        arrays[prefix + "posterior_expected_mse"] += 99
    evidence._replay(arrays, *args, metrics)


def test_secondary_streams_are_archived_not_new_diagnostic_streams():
    streams = evidence.METHODS["selected_secondary_streams"]
    assert streams["prior_seeds"] == [62001, 62002, 62003]
    assert streams["iw_seeds"] == [65001, 65002, 65003]
    assert streams["prior_seeds"] != evidence.DART_DIAGNOSTICS["prior_seeds"]
    assert streams["iw_seeds"] != evidence.DART_DIAGNOSTICS["iw_seeds"]


def test_public_components_reject_resealed_inconsistent_likelihood(metrics):
    result = evidence._components(component_arrays(), 2, .5, np.full(2, .5), metrics)
    result["weighted"]["nll_per_coordinate"] = evidence._streams([2., 2., 2.])
    with pytest.raises(ValueError, match="normalized Gaussian likelihood"):
        evidence._validate_components(result, .5)


def test_prior_noise_rng_order_and_variance_only_change():
    means = np.arange(4 * 192, dtype=float).reshape(4, 192) / 100
    original = evidence._observation_draw(means, .25, 71001)
    refit = evidence._observation_draw(means, 4., 71001)
    np.testing.assert_allclose((original - means) / .5, (refit - means) / 2, atol=1e-14)
    rng = np.random.default_rng(71001)
    rng.standard_normal((4, 16))
    np.testing.assert_array_equal(original, means + .5 * rng.standard_normal((4, 192)))
    wrong = means + .5 * np.random.default_rng(71001).standard_normal((4, 192))
    with pytest.raises(ValueError, match="seeded noise"):
        evidence._close(wrong, original, "seeded noise")


@pytest.mark.parametrize("key", ["prior_means", "posterior_mean_mse", "posterior_expected_mse", "kl", "kl_by_latent", "sample_logq_minus_logprior"])
def test_fixed_function_arrays_cannot_change_with_variance(key):
    arrays = {prefix + field: np.ones((3, 2)) for prefix in ("original__", "variance_refit__")
              for field in ("prior_means", "posterior_mean_mse", "posterior_expected_mse", "kl", "kl_by_latent", "sample_logq_minus_logprior")}
    evidence._fixed_functions(arrays)
    arrays["variance_refit__" + key][0, 0] += .1
    with pytest.raises(ValueError, match="identity changed"):
        evidence._fixed_functions(arrays)


def test_reference_variance_floor_and_diagonal_max(metrics):
    data = {"train": np.zeros((4, 192))}
    fitted, models = evidence._reference_models(data, np.full(4, .25), 5e-4, metrics)
    assert fitted == 1e-4
    np.testing.assert_array_equal(models["ppca_fitted"]["covariance"], np.eye(192) * 1e-4)
    np.testing.assert_array_equal(models["diagonal_matched"]["covariance"], np.eye(192) * 5e-4)
    assert models["ppca_matched"]["effective_rank"] == 0


def test_json_duplicates_nonfinite_and_symlink_paths(tmp_path):
    path = tmp_path / "input.json"
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            evidence._read(path)
    path.write_text('{}')
    link = tmp_path / "alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        evidence._read(link / "input.json")


def test_duplicate_npz_and_nonfinite_shape(tmp_path):
    path = tmp_path / "arrays.npz"
    stream = io.BytesIO()
    np.save(stream, np.array(1.), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", stream.getvalue())
        with pytest.warns(UserWarning):
            archive.writestr("x.npy", stream.getvalue())
    with pytest.raises(ValueError, match="Duplicate NPZ"):
        evidence._load(path)
    np.savez(path, x=np.array(float("nan")))
    with pytest.raises(ValueError, match="finite array"):
        evidence._load(path, {"x": ()})


def admission():
    return {"schema_version": 1, "purpose": "dart_optional_execution_admission", "admitted_records": list(evidence.OPTIONAL),
            "stage_manifest_sha256": "a" * 64, "fits_per_record": 12, "total_new_fit_ceiling": 36,
            "optional_neural_results_exist": False, "allowance_per_optional_block_seconds": 100.,
            "remaining_seconds_before_reserve": 300., "basis": "measured costs only", "created_utc": "2026-09-08T14:26:13+00:00",
            "verified_counts": {"diagnostics-core-pilot": 12, "diagnostics-core-remaining": 36, "diagnostics-primary": 12}}


def test_primary_frozen_metadata_has_no_optional_calendar_days_field():
    record = {"qualification": {"eligible": True, "coverage": .99},
              "deployment": {"catalog_entry": "44402_20160423to20190509",
                             "tcon_exact_deployment_fields": {"name": "44402_20160423to20190509",
                                                              "data_start": "2016-04-24T02:03:15Z",
                                                              "data_end": "2019-05-09T22:23:30Z"}}}
    evidence._metadata_coverage(record, evidence.PRIMARY)


def test_processed_coverage_need_not_equal_catalogue_deployment_dates():
    record = {"qualification": {"eligible": True, "calendar_days": 365, "coverage": .9},
              "deployment": {"catalogue_deployment": {"id": "21418_20170706to20190830"},
                             "full_2018_inside_catalogue_and_processed_dates": True, "recovered_qc_not_input": True,
                             "processed_catalog_filename_dates": {"start": "20170706", "end": "20190829"}}}
    evidence._metadata_coverage(record, "dart_21418_2018")
    record["deployment"]["processed_catalog_filename_dates"]["end"] = "20181231"
    with pytest.raises(ValueError, match="outside recorded processed"):
        evidence._metadata_coverage(record, "dart_21418_2018")


def test_completed_optional_block_cannot_be_hidden_by_omitting_batch(tmp_path):
    optional = evidence.OPTIONAL[0]
    jobs = [[optional, condition, seed, label] for seed in evidence.SEEDS
            for label in ("low", "high") for condition in ("base", "raw")]
    manifest = {"prepared_sha256": {evidence.PRIMARY: "a" * 64, optional: "b" * 64}, "jobs": jobs}
    evidence._publication_blocks(tmp_path, manifest, {evidence.PRIMARY})
    for index, job in enumerate(jobs):
        directory = tmp_path / "jobs" / "_".join(map(str, job))
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text('{}')
        if index < 11:
            evidence._publication_blocks(tmp_path, manifest, {evidence.PRIMARY})
    with pytest.raises(ValueError, match="Completed optional block omitted"):
        evidence._publication_blocks(tmp_path, manifest, {evidence.PRIMARY})


def test_admission_never_equates_registration_with_completion():
    assert evidence._admission(None, "a" * 64, {evidence.PRIMARY})["published_fits"] == 12
    assert evidence._admission(admission(), "a" * 64, {evidence.PRIMARY})["published_fits"] == 12
    assert evidence._admission(admission(), "a" * 64, {evidence.PRIMARY, *evidence.OPTIONAL})["published_fits"] == 36
    for names in ({evidence.PRIMARY, evidence.OPTIONAL[0]}, set(evidence.OPTIONAL), set()):
        with pytest.raises(ValueError, match="exactly 12 or 36"):
            evidence._admission(admission(), "a" * 64, names)
    with pytest.raises(ValueError, match="require admission"):
        evidence._admission(None, "a" * 64, {evidence.PRIMARY, *evidence.OPTIONAL})


@pytest.mark.parametrize("key,value", [("optional_neural_results_exist", True), ("fits_per_record", True),
                                       ("remaining_seconds_before_reserve", 199.), ("total_new_fit_ceiling", 48),
                                       ("stage_manifest_sha256", "b" * 64), ("created_utc", "2026-09-08")])
def test_admission_tamper(key, value):
    receipt = admission()
    receipt[key] = value
    with pytest.raises(ValueError):
        evidence._admission(receipt, "a" * 64, {evidence.PRIMARY, *evidence.OPTIONAL})


def test_source_pin_checked_before_import(monkeypatch):
    calls = []
    monkeypatch.setattr(evidence, "_sha", lambda path: "0" * 64)
    monkeypatch.setattr(evidence.importlib, "import_module", lambda name: calls.append(name))
    with pytest.raises(ValueError, match="Unreviewed current source"):
        evidence._imports()
    assert calls == []


def test_historical_producer_maps_remain_independent_of_reviewed_current_sources():
    public = json.loads((evidence.ROOT / "paper/prigm/paper_evidence_manifest.json").read_text())
    assert {"src/" + name: digest for name, digest in evidence.CORE_SOURCES.items()} == public["core"]["code_sha256"]
    files = public["dart_extension"]["input_index"]["files"]
    for name, digest in evidence.STAGE_SOURCES.items():
        assert files["stage/source/src/" + name] == digest
    assert evidence.REVIEWED_SOURCES["prigm_model.py"] != evidence.CORE_SOURCES["prigm_model.py"]
    assert evidence.REVIEWED_SOURCES["prigm_metrics.py"] != evidence.STAGE_SOURCES["prigm_metrics.py"]
    assert evidence.REVIEWED_SOURCES["prigm_v2_model.py"] != evidence.STAGE_SOURCES["prigm_v2_model.py"]
    evidence._imports()  # Actual maintained bytes, including the corrected model docstring.


@pytest.mark.parametrize("source", list(evidence.REVIEWED_SOURCES))
def test_actual_current_source_tampering_is_rejected_before_import(tmp_path, monkeypatch, source):
    (tmp_path / "src").mkdir()
    for name in evidence.REVIEWED_SOURCES:
        (tmp_path / "src" / name).write_bytes((evidence.ROOT / "src" / name).read_bytes())
        monkeypatch.delitem(sys.modules, name[:-3], raising=False)
    changed = tmp_path / "src" / source
    changed.write_bytes(changed.read_bytes() + b"\n# unreviewed current bytes\n")
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence.importlib, "import_module", lambda name: pytest.fail("Imported before source authentication"))
    with pytest.raises(ValueError, match="Unreviewed current source"):
        evidence._imports()


def test_cached_project_module_from_other_tree_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "prigm_data", SimpleNamespace(__file__=str(tmp_path / "prigm_data.py")))
    with pytest.raises(ValueError, match="another source tree"):
        evidence._imports()


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    jobs = [[evidence.OLD[0], a, s, "selected"] for s in evidence.SEEDS for a in evidence.ARMS]
    source_hashes = {"eval_gaussian_final_checkpoints.py": evidence.EVALUATOR_REVIEWED_SHA256,
                     "prigm_v2_metrics.py": evidence.STAGE_SOURCES["prigm_v2_metrics.py"]}
    for worker in range(4):
        directory = root / f"worker-{worker}"
        directory.mkdir()
        (directory / "source").mkdir()
        for name in source_hashes:
            (directory / "source" / name).write_bytes((evidence.ROOT / "src" / name).read_bytes())
        protocol = {"schema_version": 2, "purpose": "dart_fixed_checkpoint_diagnostics", "core_manifest_sha256": evidence.CORE_SHA256,
                    "evaluator_source_sha256": source_hashes, "worker_index": worker, "worker_count": 4,
                    "jobs": jobs[worker::4], "all_jobs": jobs, "samples": 256, "stream_seeds": evidence.DART_DIAGNOSTICS["iw_seeds"],
                    "density_coordinates": "residual", "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32,
                    "runtime": {"private": "/home/operator/GPU-private"}, "dart_diagnostics": evidence.DART_DIAGNOSTICS,
                    "training_measure": "original_uniform_training_windows", "evaluation_measure": "original_uniform_test_windows"}
        (directory / "protocol.json").write_text(json.dumps(protocol))
        protocol_sha = evidence._sha(directory / "protocol.json")
        ref_name = f"references-{evidence.OLD[0]}-{'f' * 16}.npz"
        (directory / ref_name).write_bytes(b"private reference fixture")
        ref_sha = evidence._sha(directory / ref_name)
        artifacts = {ref_name: ref_sha}
        for job in jobs[worker::4]:
            stem = "_".join(map(str, job[:-1])) + "-selected"
            (directory / (stem + ".npz")).write_bytes(b"private model fixture")
            digest = evidence._sha(directory / (stem + ".npz"))
            record = {"protocol_sha256": protocol_sha, "array_sha256": digest,
                      "references": {"file": ref_name, "sha256": ref_sha}}
            (directory / (stem + ".json")).write_text(json.dumps(record))
            artifacts[stem + ".npz"] = digest
            artifacts[stem + ".json"] = evidence._sha(directory / (stem + ".json"))
        complete = {"schema_version": 2, "purpose": protocol["purpose"], "protocol_sha256": protocol_sha,
                    "core_manifest_sha256": evidence.CORE_SHA256, "evaluator_source_sha256": source_hashes, "artifact_sha256": artifacts}
        (directory / "complete.json").write_text(json.dumps(complete))
    return root


def check_batch(roots):
    def bind(logical, path, expected=None):
        digest = evidence._sha(path)
        if expected is not None:
            assert expected == digest
        assert "/home/" not in logical and "GPU-" not in logical
        return digest
    return evidence._batch_records(roots, False, {"_sha": evidence.CORE_SHA256}, bind)


def test_batch_exact_inventory_missing_and_duplicate_blocks(batch):
    assert len(check_batch([batch])) == 12
    with pytest.raises(ValueError, match="Duplicate batch"):
        check_batch([batch, batch])
    path = batch / "worker-0" / "unexpected.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="artifact inventory"):
        check_batch([batch])
    path.unlink()
    (batch / "worker-1" / "protocol.json").unlink()
    with pytest.raises(ValueError, match="regular input"):
        check_batch([batch])


def test_resealed_self_declared_producer_source_is_not_a_pin(batch):
    path = batch / "worker-0" / "protocol.json"
    protocol = evidence._read(path)
    protocol["evaluator_source_sha256"]["eval_gaussian_final_checkpoints.py"] = "0" * 64
    path.write_text(json.dumps(protocol))
    marker = batch / "worker-0" / "complete.json"
    complete = evidence._read(marker)
    complete["protocol_sha256"] = evidence._sha(path)
    complete["evaluator_source_sha256"] = protocol["evaluator_source_sha256"]
    marker.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="evaluator_source_sha256"):
        check_batch([batch])


@pytest.fixture(scope="module")
def public_fixture(metrics):
    fake = "a" * 64
    files = {"core/run_manifest.json": evidence.CORE_SHA256, "stage/stage_manifest.json": fake,
             "reviewed/prigm_dart_paper_evidence.py": fake,
             **{"reviewed/" + k: v for k, v in evidence.REVIEWED_SOURCES.items()}}
    public = {"schema_version": 1, "purpose": "dart_window_diagnostics", "methods": deepcopy(evidence.METHODS),
              "verification": deepcopy(evidence.VERIFICATION), "admission": evidence._admission(None, fake, {evidence.PRIMARY}),
              "input_index": {"core_manifest_sha256": evidence.CORE_SHA256, "stage_manifest_sha256": fake,
                              "helper_sha256": fake, "reviewed_source_sha256": dict(evidence.REVIEWED_SOURCES), "files": files}, "records": {}}
    scores = {name: evidence._score_summary(np.ones((3, 2)), np.full(2, .5), {"14": np.array([0, 0]), "28": np.array([0, 0])}) for name in evidence.SCORES}
    for name in (*evidence.OLD, evidence.PRIMARY):
        fresh = name == evidence.PRIMARY
        kind = "stage" if fresh else "core"
        files[f"{kind}/prepared/{name}.npz"] = fake
        files[f"{kind}/prepared/" + ("records.json" if fresh else "metadata.json")] = fake
        record = {"station": name.split("_")[1] if fresh else name[:5], "year": 2018 if fresh else int(name[-4:]),
                  "target": "six_UTC_bins_equal" if fresh else "original_uniform_windows", "primary_checkpoint": "final" if fresh else "selected",
                  "splits": {split: {"windows": 2, "first_start_unix": 0, "last_start_unix": 1, "packs": 2 if fresh else None} for split in ("train", "validation", "test")},
                  "mean": 2500., "scale": 1.5, "normalization": "unique_covered_training_source_rows" if fresh else "original_valid_training_source_rows",
                  "training_measure": "six_UTC_bins_equal" if fresh else "original_uniform_windows",
                  "window_measures": {split: {"within_window_variance": .5, "between_window_mean_variance": .5, "pooled_entry_variance": 1.} for split in ("train", "validation", "test")},
                  "metadata_sha256": fake, "prepared_sha256": fake,
                  "metadata_scope": "catalogued_annual_coverage_hardware_unestablished" if fresh else "original_record_protocol_retained",
                  "selection_role": "retrospective_additional_year_same_station" if fresh else "original_case_study",
                  "models": {}, "references": {}, "optimizer_summaries": {}, "contrasts": {}}
        for arm in evidence.ARMS:
            variance = .5 if arm in ("A", "C") else 12.5
            for seed in evidence.SEEDS:
                condition = "base" if arm in ("A", "D") else "raw"
                label = "low" if arm in ("A", "C") else "high"
                identifier = f"{name}_{condition}_{seed}_{label}" if fresh else f"{name}_{arm}_{seed}"
                files[f"{kind}/jobs/{identifier}/" + ("final.weights.h5" if fresh else "best.weights.h5")] = fake
                worker = f"{kind}-batch-000/worker-{evidence.ARMS.index(arm)}"
                stem = identifier + ("-final" if fresh else "-selected")
                for suffix in (".json", ".npz"):
                    files[worker + "/" + stem + suffix] = fake
                for receipt in ("protocol.json", "complete.json", f"references-{name}-{'f' * 16}.npz"):
                    files[worker + "/" + receipt] = fake
                for source in ("eval_gaussian_final_checkpoints.py", "prigm_v2_metrics.py", *(["prigm_metrics.py"] if fresh else [])):
                    files[worker + "/source/" + source] = (evidence.EVALUATOR_REVIEWED_SHA256
                        if source == "eval_gaussian_final_checkpoints.py" else evidence.STAGE_SOURCES[source])
                if fresh:
                    files[f"{kind}/jobs/{identifier}/selected.weights.h5"] = fake
                    files[f"{kind}/jobs/{identifier}/evaluation-final.npz"] = fake
                for variant in ("original", "variance_refit"):
                    v = variance if variant == "original" else .3
                    components = evidence._components(component_arrays(v), 2, v, np.full(2, .5), metrics)
                    model = {"observation_variance": v, "variance_fit": {"streams": evidence._streams([.3, .3, .3]), "pooled": .3},
                             "scores": deepcopy(scores), "components": components, "condition": arm, "seed": seed,
                             "original_variance_label": "low" if arm in ("A", "C") else "high", "checkpoint": record["primary_checkpoint"],
                             "step": 10000 if fresh else 1000, "checkpoint_sha256": fake, "original_observation_variance": variance,
                             "raw_self_mse": None if fresh else .1, "raw_self_mse_source": "not_saved_by_fresh_runner" if fresh else "original_evaluation_self_mse",
                             "selected_secondary": None}
                    if fresh and variant == "original":
                        model["selected_secondary"] = {"checkpoint": "selected", "step": 10000, "aliases_final": True,
                                                       "weights_sha256": fake, "evaluation_sha256": fake, "scores": deepcopy(scores),
                                                       "components": deepcopy(components), "raw_self_mse": None}
                    record["models"][f"{arm}/{seed}/{variant}"] = model
        for label, variance in (("low", .5), ("high", 12.5)):
            record["references"][label] = {ref: {"noise_variance": 1e-4 if ref == "ppca_fitted" else variance,
                                                "scores": deepcopy(scores), "weighted_log_density_per_coordinate": -1.,
                                                "uniform_log_density_per_coordinate": -1.} for ref in ("ppca_matched", "ppca_fitted", "diagonal_matched")}
        record["optimizer_summaries"], record["contrasts"] = evidence._aggregate(record["models"])
        public["records"][name] = record
    return public


def test_optional_publication_is_a_complete_36_fit_extension(public_fixture):
    public = deepcopy(public_fixture)
    files = public["input_index"]["files"]
    for dataset in evidence.OPTIONAL:
        record = deepcopy(public["records"][evidence.PRIMARY])
        record["station"] = dataset.split("_")[1]
        record["selection_role"] = "data_only_qualified_optional_site_resource_admitted"
        public["records"][dataset] = record
        for path, digest in list(files.items()):
            if evidence.PRIMARY in path:
                files[path.replace(evidence.PRIMARY, dataset)] = digest
    public["admission"] = {"optional_admitted": True, "published_fits": 36, "admission_sha256": "a" * 64}
    files["admission.json"] = "a" * 64
    evidence.validate_dart_extension(public)
    public["admission"]["optional_admitted"] = False
    with pytest.raises(ValueError, match="Unadmitted optional block"):
        evidence.validate_dart_extension(public)


def test_closed_public_serialization_endpoint_roles_and_mc_not_optimizer(public_fixture):
    public = deepcopy(public_fixture)
    assert evidence.validate_dart_extension(public) is public
    serialized = json.dumps(public, allow_nan=False)
    assert evidence.validate_dart_extension(json.loads(serialized), input_index=public["input_index"])
    old = public["records"][evidence.OLD[0]]
    fresh = public["records"][evidence.PRIMARY]
    assert old["primary_checkpoint"] == "selected" and fresh["primary_checkpoint"] == "final"
    assert fresh["models"]["A/42/original"]["selected_secondary"]["aliases_final"]
    assert old["contrasts"]["C-A/original"]["matched_variance"] is True
    assert old["contrasts"]["C-A/variance_refit"]["matched_variance"] is False
    assert "median" in old["optimizer_summaries"]["A/original"]["scores"]["joint"]
    assert "mean" in old["models"]["A/42/original"]["scores"]["joint"]["weighted"]
    assert "GPU-" not in serialized and "/home/" not in serialized


@pytest.mark.parametrize("mutation", ["private", "nan", "missing_record", "missing_model", "refit_match", "endpoint", "alias", "floor", "source", "logical_path"])
def test_public_rejects_invented_or_private_summary(public_fixture, mutation):
    public = deepcopy(public_fixture)
    record = public["records"][evidence.PRIMARY]
    if mutation == "private":
        record["runtime"] = {"pid": 123, "host": "private"}
    elif mutation == "nan":
        record["scale"] = float("nan")
    elif mutation == "missing_record":
        del public["records"][evidence.OLD[0]]
    elif mutation == "missing_model":
        del record["models"]["A/42/original"]
    elif mutation == "refit_match":
        record["contrasts"]["C-A/variance_refit"]["matched_variance"] = True
    elif mutation == "endpoint":
        record["primary_checkpoint"] = "selected"
    elif mutation == "alias":
        record["models"]["A/42/original"]["selected_secondary"]["aliases_final"] = False
    elif mutation == "floor":
        record["references"]["low"]["ppca_fitted"]["noise_variance"] = 1e-5
    elif mutation == "source":
        public["input_index"]["reviewed_source_sha256"]["prigm_v2_metrics.py"] = "f" * 64
    else:
        public["input_index"]["files"]["/home/private/results.json"] = "f" * 64
    with pytest.raises(ValueError):
        evidence.validate_dart_extension(public)


def test_public_input_index_needs_all_distinct_array_blocks(public_fixture):
    public = deepcopy(public_fixture)
    files = public["input_index"]["files"]
    name = next(name for name in files if "-batch-" in name and name.endswith("-selected.npz"))
    digest = files.pop(name)
    with pytest.raises(ValueError, match="indexed diagnostic arrays"):
        evidence.validate_dart_extension(public)
    files[name] = digest
    duplicate = name.replace("worker-0", "worker-1")
    assert duplicate != name
    files[duplicate] = digest
    files[duplicate[:-4] + ".json"] = files[name[:-4] + ".json"]
    with pytest.raises(ValueError, match="indexed diagnostic arrays"):
        evidence.validate_dart_extension(public)


@pytest.mark.parametrize("field", evidence.REDUCTIONS)
def test_uniform_target_requires_every_component_stream(metrics, field):
    components = evidence._components(component_arrays(), 2, .5, np.full(2, .5), metrics)
    uniform = components["uniform"]
    uniform[field] = evidence._streams(np.asarray(uniform[field]["values"]) + (-.1 if field == "mean_ess" else .1))
    # Keep the existing Gaussian/ELBO equations valid to isolate target equality.
    if field == "nll_per_coordinate":
        uniform["posterior_expected_mse"] = evidence._streams(
            np.asarray(uniform[field]["values"]) - .5 * np.log(np.pi))
    elif field == "posterior_expected_mse":
        uniform["nll_per_coordinate"] = evidence._streams(
            .5 * np.log(np.pi) + np.asarray(uniform[field]["values"]))
    if field == "negative_elbo_per_coordinate":
        uniform["kl_per_coordinate"] = evidence._streams(
            np.asarray(uniform[field]["values"]) - uniform["nll_per_coordinate"]["values"])
    else:
        uniform["negative_elbo_per_coordinate"] = evidence._streams(
            np.asarray(uniform["nll_per_coordinate"]["values"]) + uniform["kl_per_coordinate"]["values"])
    evidence._validate_components(components, .5)
    with pytest.raises(ValueError, match="uniform target component"):
        evidence._validate_components(components, .5, uniform_target=True)


@pytest.mark.parametrize("mutation", ["model_score", "component", "reference_score", "reference_density", "block_mass"])
@pytest.mark.parametrize("fresh", [False, True], ids=["old_rejected", "fresh_allowed"])
def test_public_uniform_target_guards(public_fixture, mutation, fresh):
    public = deepcopy(public_fixture)
    record = public["records"][evidence.PRIMARY if fresh else evidence.OLD[0]]
    model = record["models"]["A/42/original"]
    ref = record["references"]["low"]["ppca_matched"]
    if mutation == "model_score":
        model["scores"]["joint"]["uniform"] = evidence._streams([999., 999., 999.])
    elif mutation == "component":
        for variant in ("original", "variance_refit"):
            record["models"]["A/42/" + variant]["components"]["uniform"]["posterior_mean_mse"] = evidence._streams([99., 99., 99.])
        if fresh:
            model["selected_secondary"]["components"]["uniform"]["posterior_mean_mse"] = evidence._streams([99., 99., 99.])
    elif mutation == "reference_score":
        ref["scores"]["joint"]["uniform"] = evidence._streams([999., 999., 999.])
    elif mutation == "reference_density":
        ref["uniform_log_density_per_coordinate"] = 999.
    else:
        blocks = model["scores"]["joint"]["blocks"]
        for days in blocks:
            row = blocks[days][0]
            blocks[days] = [dict(deepcopy(row), block=i, windows=1, mass=mass)
                            for i, mass in enumerate((.25, .75))]
    if fresh:
        evidence.validate_dart_extension(public)
    else:
        with pytest.raises(ValueError, match="uniform target"):
            evidence.validate_dart_extension(public)


@pytest.mark.parametrize("field", ["posterior_mean_mse", "kl_per_coordinate"])
@pytest.mark.parametrize("measure", ["weighted", "uniform"])
def test_public_selected_alias_binds_deterministic_components(public_fixture, field, measure):
    public = deepcopy(public_fixture)
    model = public["records"][evidence.PRIMARY]["models"]["A/42/original"]
    secondary = model["selected_secondary"]
    components = secondary["components"][measure]
    components[field] = evidence._streams([99., 99., 99.])
    if field == "kl_per_coordinate":
        components["negative_elbo_per_coordinate"] = evidence._streams(
            np.asarray(components["nll_per_coordinate"]["values"]) + 99.)
    with pytest.raises(ValueError, match="selected final deterministic alias " + field):
        evidence.validate_dart_extension(public)
    secondary.update(step=6000, aliases_final=False)
    files = public["input_index"]["files"]
    files[f"stage/jobs/{evidence.PRIMARY}_base_42_low/evaluation-selected.npz"] = secondary["evaluation_sha256"]
    evidence.validate_dart_extension(public)


def test_public_selected_alias_allows_mc_and_serialization_differences(public_fixture, metrics):
    public = deepcopy(public_fixture)
    model = public["records"][evidence.PRIMARY]["models"]["A/42/original"]
    secondary = model["selected_secondary"]
    secondary["components"] = evidence._components(component_arrays(), 2, .5, np.array([.25, .75]), metrics)
    for measure in ("weighted", "uniform"):
        values = secondary["components"][measure]
        for field in ("posterior_mean_mse", "kl_per_coordinate"):
            values[field] = evidence._streams(np.asarray(model["components"][measure][field]["values"]) + 5e-7)
        values["posterior_expected_mse"] = evidence._streams([.9, .8, .7])
        values["nll_per_coordinate"] = evidence._streams(.5 * np.log(np.pi) + np.array([.9, .8, .7]))
        values["negative_elbo_per_coordinate"] = evidence._streams(
            np.asarray(values["nll_per_coordinate"]["values"]) + values["kl_per_coordinate"]["values"])
        values["negative_iw_per_coordinate"] = evidence._streams([2., 3., 4.])
        values["jensen_gap_per_coordinate"] = evidence._streams([1., 2., 3.])
    secondary["scores"] = {score: evidence._score_summary(
        np.array([[1., 3.], [2., 4.], [3., 5.]]), np.array([.25, .75]),
        {"14": np.array([0, 1]), "28": np.array([0, 1])}) for score in evidence.SCORES}
    secondary["weights_sha256"] = "b" * 64
    public["input_index"]["files"][f"stage/jobs/{evidence.PRIMARY}_base_42_low/selected.weights.h5"] = "b" * 64
    evidence.validate_dart_extension(public)


def test_public_missing_duplicate_descriptive_blocks(public_fixture):
    public = deepcopy(public_fixture)
    blocks = public["records"][evidence.OLD[0]]["models"]["A/42/original"]["scores"]["joint"]["blocks"]
    blocks["14"].append(deepcopy(blocks["14"][0]))
    with pytest.raises(ValueError, match="Duplicate/invalid block"):
        evidence.validate_dart_extension(public)
    blocks["14"] = []
    with pytest.raises(ValueError, match="Missing descriptive blocks"):
        evidence.validate_dart_extension(public)
