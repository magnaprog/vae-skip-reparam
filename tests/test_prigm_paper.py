"""CPU evidence-builder tests; all campaigns here are synthetic fixtures, not results."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_prigm_paper as paper
import prigm_experiment as experiment


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def read_json(path):
    return json.loads(path.read_text())


def evaluation_fixture(arm="A", score=2.0):
    """Tiny test split, with the real frozen evaluation dimensions and keys."""
    spec = experiment.specification("core")
    d = spec["window"]
    x = np.broadcast_to(np.array([0.25, 1.0, 2.0])[:, None], (3, d)).copy()
    corrected = np.array([0.0625, 1.0, 4.0])
    factor = 0.04 if arm in ("B", "C") else 1.0
    logs = -d * (score + np.array([-0.2, 0.0, 0.2]))
    draws = np.zeros((spec["prior_samples"], d))
    draws[0] = np.arange(d) / d + experiment.ARMS.index(arm)
    draws[1] = 12345  # Selection must never switch to this distinctive draw.
    arrays = {
        "log_score": logs, "ess": np.array([16., 32., 64.]),
        "normalized_ess": np.array([16., 32., 64.]) / spec["evaluation_samples"],
        "self_mse": corrected * factor, "corrected_mse": corrected,
        "kl": np.array([1., 3., 5.]), "permutation_delta": np.array([0., 0.5, 1.]),
        "energy": np.array([2., 3., 7.]),
        "stability_log_score": logs + d * 0.1, "stability_ess": np.full(3, 128.),
        "replicate_log_score": logs - d * 0.05, "stability_primary_log_score": logs.copy(),
        "prior_samples": draws, "posterior_mean": np.zeros_like(x),
        "posterior_kl_per_dimension": np.broadcast_to(np.array([1., 3., 5.])[:, None] / 16, (3, 16)).copy(),
    }
    return arrays, {"test": x, "scale": np.array(2.)}, spec


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A small hash-bound 60-job fixture using the real verifiers and baselines."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "prepared").mkdir()
    (root / "source" / "src").mkdir(parents=True)
    (root / "jobs").mkdir()
    source_root = tmp_path / "public"
    (source_root / "datasets").mkdir(parents=True)
    monkeypatch.setattr(paper, "ROOT", source_root)
    for name in experiment.CODE_FILES:
        (root / "source" / name).write_bytes((ROOT / name).read_bytes())
    spec = experiment.specification("core")
    metadata = {}
    for dataset in paper.DATASETS:
        _, data, _ = evaluation_fixture()
        scale, mean = (1., 0.) if dataset == "gaussian" else (2., 4000.25)
        packed = {s: data["test"] for s in ("train", "validation", "test")}
        packed.update({f"starts_{s}": np.arange(3) for s in ("train", "validation", "test")})
        packed.update(mean=np.array(mean), scale=np.array(scale), base_variance=np.array(0.025),
                      ppca_mean=np.zeros(spec["window"]), ppca_covariance=np.eye(spec["window"]),
                      diagonal_covariance=1.5 * np.eye(spec["window"]))
        if dataset == "gaussian":
            packed["oracle_covariance"] = 0.8 * np.eye(spec["window"])
        np.savez_compressed(root / "prepared" / f"{dataset}.npz", **packed)
        metadata[dataset] = {
            "dataset": dataset, "mean": mean, "scale": scale, "std": scale, "base_variance": 0.025,
            "windows": {s: 3 for s in ("train", "validation", "test")},
            "counts": {s: {"windows": 3} for s in ("train", "validation", "test")},
            "private_unused_metadata": "/home/private-user/not-for-paper",
        }
        if dataset != "gaussian":
            year = dataset.split("t")[1]
            path = source_root / "datasets" / f"{dataset}.txt"
            # Isolated rows, a duplicate, and a year-end row are all context.
            path.write_text("header\nunits\n" + "\n".join([
                f"{year} 1 1 0 0 0 1 4000.125",
                f"{year} 1 1 0 0 0 1 4000.375",
                f"{year} 6 3 0 0 0 2 4999.5",
                f"{year} 7 1 0 0 0 1 9999.0",
                f"{year} 12 31 12 0 0 1 4001.625",
            ]) + "\n")
            metadata[dataset].update(source_sha256=experiment.sha256(path), observed_t1_rows=3)
    write_json(root / "prepared" / "metadata.json", metadata)
    manifest = {
        "schema_version": 1, "purpose": "core", "datasets": list(paper.DATASETS),
        "specification": spec, "run_id": "PRIVATE-RUN-UUID",
        "code_sha256": {name: experiment.sha256(ROOT / name) for name in experiment.CODE_FILES},
        "prepared_sha256": {p.name: experiment.sha256(p) for p in (root / "prepared").iterdir()},
    }
    write_json(root / "run_manifest.json", manifest)
    pinned = experiment.sha256(root / "run_manifest.json")
    summaries = []
    for dataset in paper.DATASETS:
        for arm in paper.ARMS:
            for seed in paper.SEEDS:
                identifier = experiment.job_id(dataset, arm, seed)
                attempt = root / "jobs" / identifier / "attempt-PRIVATE-UUID"
                attempt.mkdir(parents=True)
                for name in experiment.JOB_ARTIFACTS:
                    (attempt / name).write_bytes(b"fixture checkpoint or log, never loaded as a model")
                configuration = {"dataset": dataset, "arm": arm, "seed": seed,
                                 "specification": spec, "run_manifest_sha256": pinned,
                                 "runtime": {"gpu_uuid": "PRIVATE-GPU-UUID", "username": "private-user",
                                             "path": "/home/private-user/machine"}}
                write_json(attempt / "configuration.json", configuration)
                score = 2. + paper.ARMS.index(arm) + paper.SEEDS.index(seed) / 10
                arrays, _, _ = evaluation_fixture(arm, score)
                np.savez_compressed(attempt / "evaluation.npz", **arrays)
                scale = metadata[dataset]["scale"]
                # Independent hand-computed expected scalar values.
                summary = {
                    "dataset": dataset, "arm": arm, "seed": seed, "best_step": 250, "steps_run": 500,
                    "best_validation_score": 123., "elapsed_seconds": 1., "test_windows": 3,
                    paper.SCORE: score,
                    "negative_log_score_meter_coordinates_per_dimension": score + float(np.log(scale)),
                    "median_effective_samples": 32., "median_normalized_ess": 0.125,
                    "self_mse": 1.6875 * (0.04 if arm in ("B", "C") else 1),
                    "corrected_mse": 1.6875, "kl_nats_per_window": 3., "energy_score": 4.,
                    "energy_score_meters": 4. * scale, "permutation_corrected_mse_increase": 0.5,
                    "stability_mean_log_score_change_per_dimension": 0.1,
                    "replicate_mean_log_score_change_per_dimension": -0.05,
                    "mapped_control": {"target_arm": {"A": "C", "C": "A", "B": "D", "D": "B"}[arm],
                                       "max_absolute_log_density_difference": 0.001,
                                       "max_absolute_sample_mean_difference": 0.000001,
                                       "relative_log_density_tolerance": 3e-5,
                                       "absolute_log_density_tolerance": 0.01},
                }
                write_json(attempt / "summary.json", summary)
                summaries.append(summary)
                write_json(attempt.parent / "complete.json", {
                    "attempt": attempt.name, "run_manifest_sha256": pinned,
                    "artifact_sha256": {name: experiment.sha256(attempt / name) for name in experiment.JOB_ARTIFACTS},
                })
    baselines = experiment.baseline_results(root, manifest)
    write_json(root / "results.json", {
        "schema_version": 1, "purpose": "core", "run_manifest_sha256": pinned,
        "specification": spec, "dataset_metadata": metadata, "results": summaries,
        "baselines": baselines, "unused_private_field": "/home/private-user/runtime",
    })
    files = [p for p in root.rglob("*") if p.is_file()]
    write_json(root / "archive_manifest.json", {
        "schema_version": 1,
        "artifact_sha256": {p.relative_to(root).as_posix(): experiment.sha256(p) for p in files},
    })
    return root, pinned


def reseal(root):
    """Update fixture hashes to test semantic checks beyond byte tampering."""
    for path in (root / "jobs").glob("*/complete.json"):
        completion = read_json(path)
        attempt = path.parent / completion["attempt"]
        completion["artifact_sha256"] = {name: experiment.sha256(attempt / name) for name in experiment.JOB_ARTIFACTS}
        write_json(path, completion)
    archive = read_json(root / "archive_manifest.json")
    archive["artifact_sha256"] = {name: experiment.sha256(root / name) for name in archive["artifact_sha256"]}
    write_json(root / "archive_manifest.json", archive)


def test_full_build_verifies_runner_interfaces_and_sanitizes(campaign, tmp_path, monkeypatch):
    root, pinned = campaign
    calls = []
    verify_run, complete = experiment.verify_run, experiment.verified_completion
    monkeypatch.setattr(experiment, "verify_run", lambda root: (calls.append("run"), verify_run(root))[1])
    monkeypatch.setattr(experiment, "verified_completion", lambda root, identifier: (
        calls.append(identifier), complete(root, identifier))[1])
    monkeypatch.setattr(experiment, "train", lambda *a, **k: pytest.fail("Builder must never train"))
    destination = tmp_path / "paper"
    result = paper.build_paper(root, destination, pinned)
    assert len(calls) == 61 and calls.count("run") == 1
    assert len(set(calls)) == 61
    assert result["run_manifest_sha256"] == pinned
    assert result["baselines"]["gaussian"]["ppca"][paper.SCORE] == pytest.approx(
        0.5 * (np.log(2 * np.pi) + 1.6875))
    assert result["builder_sha256"] == experiment.sha256(Path(paper.__file__))
    assert len(result["jobs"]) == 60
    assert len(result["public_source_sha256"]) == 4
    assert set(result["output_sha256"]) == set(paper.OUTPUTS)
    assert result == read_json(destination / paper.EVIDENCE_NAME)
    for name, digest in result["output_sha256"].items():
        path = destination / name
        assert experiment.sha256(path) == digest
        if path.suffix == ".png":
            with Image.open(path) as image:
                text = json.dumps(image.info)
                assert "PriGM CPU evidence builder" in text
                assert image.width > 500 and image.height > 500
        else:
            text = path.read_text(encoding="ascii")
        assert "/home/" not in text and "PRIVATE" not in text and "private-user" not in text
    text = (destination / paper.EVIDENCE_NAME).read_text()
    assert "/home/" not in text and "PRIVATE" not in text and "runtime" not in text
    assert all(set(job["artifact_sha256"]) == set(experiment.JOB_ARTIFACTS) for job in result["jobs"].values())


def test_expected_manifest_and_core_purpose_are_mandatory(campaign, tmp_path):
    root, pinned = campaign
    with pytest.raises(ValueError, match="Expected root manifest"):
        paper.build_paper(root, tmp_path / "no-output", "0" * 64)
    assert not (tmp_path / "no-output").exists()
    manifest = read_json(root / "run_manifest.json")
    manifest["purpose"] = "smoke"
    write_json(root / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="purpose core"):
        paper.verify_evidence(root, experiment.sha256(root / "run_manifest.json"))


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "jobs/../outside", "./results.json",
                                   "jobs//complete.json", "jobs\\outside", "C:/outside"])
def test_unsafe_relative_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="relative artifact path"):
        paper._verify_archive(tmp_path, {name: "0" * 64})


def test_symlinked_parent_and_artifact_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact").write_text("not an archive artifact")
    root = tmp_path / "run"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "file").symlink_to(outside / "artifact")
    for name in ("link/artifact", "file"):
        with pytest.raises(ValueError, match="Symlink"):
            paper._verify_archive(root, {name: "0" * 64})


@pytest.mark.parametrize("mutation", ["missing", "tampered", "extra"])
def test_archive_requires_exact_critical_artifacts(campaign, mutation):
    root, pinned = campaign
    archive = read_json(root / "archive_manifest.json")
    if mutation == "missing":
        del archive["artifact_sha256"]["results.json"]
    elif mutation == "tampered":
        archive["artifact_sha256"]["results.json"] = "0" * 64
    else:
        (root / "extra").write_text("not part of runner archive")
        archive["artifact_sha256"]["extra"] = experiment.sha256(root / "extra")
    write_json(root / "archive_manifest.json", archive)
    with pytest.raises(ValueError, match="critical artifacts|artifact changed"):
        paper.verify_evidence(root, pinned)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "different_run", "root_summary", "forged_means", "baselines"])
def test_root_results_cannot_substitute_for_evaluation(campaign, tmp_path, mutation):
    root, pinned = campaign
    results = read_json(root / "results.json")
    if mutation == "duplicate":
        results["results"][-1] = results["results"][0]
    elif mutation == "missing":
        results["results"].pop()
    elif mutation == "different_run":
        results["run_manifest_sha256"] = "0" * 64
    elif mutation in ("root_summary", "forged_means"):
        results["results"][0][paper.SCORE] = -999.
        if mutation == "forged_means":
            record = results["results"][0]
            identifier = experiment.job_id(record["dataset"], record["arm"], record["seed"])
            complete = read_json(root / "jobs" / identifier / "complete.json")
            write_json(root / "jobs" / identifier / complete["attempt"] / "summary.json", record)
    else:
        results["baselines"]["gaussian"]["oracle"][paper.SCORE] = -999.
    write_json(root / "results.json", results)
    reseal(root)
    with pytest.raises(ValueError, match="Duplicate job|Missing or unexpected|different run|summaries differ|Recomputed"):
        paper.build_paper(root, tmp_path / "no-output", pinned)
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize("mutation", ["different_run", "configuration", "identity", "attempt_path", "checkpoint"])
def test_completion_rejects_cross_run_identity_paths_and_checkpoint_tampering(campaign, mutation):
    root, pinned = campaign
    path = root / "jobs" / "gaussian_A_42" / "complete.json"
    completion = read_json(path)
    attempt = path.parent / completion["attempt"]
    if mutation == "different_run":
        completion["run_manifest_sha256"] = "0" * 64
    elif mutation == "attempt_path":
        completion["attempt"] = "../gaussian_B_42/attempt-PRIVATE-UUID"
    elif mutation in ("configuration", "identity"):
        configuration = read_json(attempt / "configuration.json")
        configuration["run_manifest_sha256" if mutation == "configuration" else "arm"] = "0" * 64 if mutation == "configuration" else "B"
        write_json(attempt / "configuration.json", configuration)
    else:
        (attempt / "best.weights.h5").write_bytes(b"tampered")
    write_json(path, completion)
    if mutation in ("configuration", "identity"):
        reseal(root)
    else:
        archive = read_json(root / "archive_manifest.json")
        archive["artifact_sha256"][path.relative_to(root).as_posix()] = experiment.sha256(path)
        write_json(root / "archive_manifest.json", archive)
    with pytest.raises(ValueError, match="different run|attempt path|matrix slot|artifact changed"):
        paper.verify_evidence(root, pinned)


def test_truncated_evaluation_is_rejected_even_with_updated_hashes(campaign):
    root, pinned = campaign
    complete = read_json(root / "jobs" / "gaussian_A_42" / "complete.json")
    path = root / "jobs" / "gaussian_A_42" / complete["attempt"] / "evaluation.npz"
    arrays = paper._load_npz(path)
    arrays["log_score"] = arrays["log_score"][:-1]
    np.savez_compressed(path, **arrays)
    reseal(root)
    with pytest.raises(ValueError, match="truncated array: log_score"):
        paper.verify_evidence(root, pinned)


def test_truncated_zip_and_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, values=np.arange(10))
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        paper._load_npz(path)
    path = tmp_path / "duplicate.json"
    path.write_text('{"jobs": 60, "jobs": 59}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        paper._read_json(path)
    for number in ("NaN", "Infinity", "1e999"):
        path.write_text('{"value": ' + number + '}')
        with pytest.raises(ValueError, match="Nonfinite JSON"):
            paper._read_json(path)


def test_evaluation_means_are_reconstructed_independently():
    arrays, data, spec = evaluation_fixture("B")
    stats = paper.evaluation_statistics(arrays, data, spec)
    assert stats[paper.SCORE] == pytest.approx(2.)
    assert stats["self_mse"] == pytest.approx(0.0675)
    assert stats["corrected_mse"] == pytest.approx(1.6875)
    assert stats["kl_nats_per_window"] == 3.
    assert stats["energy_score"] == 4.
    assert stats["energy_score_meters"] == 8.
    assert stats["median_effective_samples"] == 32.
    assert stats["negative_log_score_meter_coordinates_per_dimension"] == pytest.approx(2 + np.log(2))
    assert stats["stability_mean_log_score_change_per_dimension"] == pytest.approx(0.1)
    assert stats["replicate_mean_log_score_change_per_dimension"] == pytest.approx(-0.05)


@pytest.mark.parametrize("key", ["self_mse", "prior_samples", "stability_log_score", "posterior_mean"])
def test_every_evaluation_shape_is_checked(key):
    arrays, data, spec = evaluation_fixture()
    arrays[key] = arrays[key][:-1]
    with pytest.raises(ValueError, match="truncated array"):
        paper.evaluation_statistics(arrays, data, spec)


def test_nonfinite_empty_and_inconsistent_evaluation_are_rejected():
    arrays, data, spec = evaluation_fixture()
    arrays["energy"][0] = np.nan
    with pytest.raises(ValueError, match="truncated array"):
        paper.evaluation_statistics(arrays, data, spec)
    arrays, data, spec = evaluation_fixture()
    arrays["normalized_ess"] *= 2
    with pytest.raises(ValueError, match="normalized ESS"):
        paper.evaluation_statistics(arrays, data, spec)
    data["test"] = data["test"][:0]
    with pytest.raises(ValueError, match="Empty"):
        paper.evaluation_statistics(arrays, data, spec)


def test_tables_pair_seeds_before_aggregation_and_end_every_row(campaign):
    root, pinned = campaign
    evidence = paper.verify_evidence(root, pinned)
    jobs = evidence["jobs"]
    for arm, scores in {"A": [0, 100, 101], "B": [50, 51, 102],
                        "C": [5, 103, 100], "D": [51, 53, 105]}.items():
        for seed, score in zip(paper.SEEDS, scores):
            jobs[experiment.job_id("gaussian", arm, seed)][paper.SCORE] = score
    paired = paper.paired_statistics(jobs)["gaussian"]["negative_iw_score_difference"]
    assert paired["B-A"] == {"median": 1., "min": -49., "max": 50.,
                              "by_seed": {"42": 50., "1729": -49., "31415": 1.}}
    assert paired["C-A"]["median"] == 3.
    assert paired["B-D"]["median"] == -2.
    fragments = paper.table_fragments(evidence)
    expected = {"results_rows.tex": (5, 4), "all_runs_rows.tex": (60, 10),
                "data_rows.tex": (5, 7), "baseline_rows.tex": (11, 4)}
    for filename, (count, columns) in expected.items():
        text = fragments["generated/" + filename]
        assert len(text.splitlines()) == count
        assert text.endswith("\n") and not text.rstrip().endswith("\\\\")
        assert all(line.endswith("\\\\") for line in text.splitlines()[:-1])
        assert all(len(line.split(" & ")) == columns for line in text.splitlines())
    assert r"\shortstack{$1$\\$[-49,\,50]$}" in fragments["generated/results_rows.tex"]
    assert r"\newcommand{\PrigmJobCount}{60}" in fragments["generated/summary_values.tex"]
    assert r"\newcommand{\PrigmCoreSHA}{" + pinned + "}" in fragments["generated/summary_values.tex"]
    # Ratio-of-paired-means, not a pooled ratio or difference of medians.
    for seed, a, b in zip(paper.SEEDS, [1, 10, 100], [2, 40, 100]):
        jobs[experiment.job_id("gaussian", "A", seed)]["self_mse"] = a
        jobs[experiment.job_id("gaussian", "B", seed)]["self_mse"] = b
    ratio = paper.paired_statistics(jobs)["gaussian"]["self_mse_ratio"]["B/A"]
    assert (ratio["median"], ratio["min"], ratio["max"]) == (2, 1, 4)


def _complete_refinement(root, pinned, output, monkeypatch):
    import prigm_refine_evaluation as refinement

    monkeypatch.setattr(refinement, "CORE_MANIFEST_SHA256", pinned)
    refinement.prepare(root, output)
    manifest = refinement.verify_refinement(root, output)
    runtime = {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000", "pid": 1,
               "placement": "/device:GPU:0", "deterministic_ops_requested": True,
               "elapsed_seconds": 0.1, "private_extra": "/home/private-user/secret"}
    for job in refinement.canonical_jobs():
        identifier = experiment.job_id(*job)
        core_attempt = experiment.verified_completion(root, identifier)
        data = experiment.load_prepared(root, job[0])
        arrays = refinement.original_arrays(core_attempt, data)
        for budget in refinement.SAMPLE_COUNTS:
            arrays[f"log_score_{budget}"] = arrays["log_score_256"] + 0.25
            arrays[f"ess_{budget}"] = arrays["ess_256"].copy()
        attempt = output / "jobs" / identifier / ("attempt-" + "a" * 32)
        attempt.mkdir(parents=True)
        np.savez_compressed(attempt / "evaluation.npz", **arrays)
        refinement.publish_json(attempt / "summary.json", refinement.job_summary(
            output, manifest, identifier, arrays, runtime))
        refinement.publish_json(attempt.parent / "complete.json", {
            "schema_version": 1, "core_job_id": identifier, "attempt": attempt.name,
            "refinement_manifest_sha256": experiment.sha256(output / refinement.MANIFEST),
            "artifact_sha256": {name: experiment.sha256(attempt / name) for name in refinement.ARTIFACTS},
        })
    refinement.summarize(root, output)


def test_refinement_evidence_recomputes_pairs_and_excludes_private_fields(campaign, tmp_path, monkeypatch):
    root, pinned = campaign
    output = tmp_path / "refinement"
    _complete_refinement(root, pinned, output, monkeypatch)
    result_path = output / "results.json"
    results = read_json(result_path)
    results["private_extra"] = "/home/private-user/secret"
    write_json(result_path, results)
    archive = read_json(output / "archive_manifest.json")
    archive["artifact_sha256"]["results.json"] = experiment.sha256(result_path)
    write_json(output / "archive_manifest.json", archive)
    evidence = paper.verify_refinement_evidence(root, output)
    assert len(evidence["results"]["jobs"]) == 60
    assert "private-user" not in json.dumps(evidence)
    assert "GPU-" not in json.dumps(evidence)
    results["paired_comparisons"]["gaussian"]["B-A"]["16384"]["median"] += 10
    write_json(result_path, results)
    archive["artifact_sha256"]["results.json"] = experiment.sha256(result_path)
    write_json(output / "archive_manifest.json", archive)
    with pytest.raises(ValueError, match="results differ from verified arrays"):
        paper.verify_refinement_evidence(root, output)


@pytest.fixture
def historical_campaign(campaign, tmp_path, monkeypatch):
    import prigm_dart_paper_evidence as dart

    root, _ = campaign
    source = root / "source/src/prigm_metrics.py"
    source.write_bytes(source.read_bytes() + b"\n# independently frozen fixture producer\n")
    manifest = read_json(root / "run_manifest.json")
    manifest["code_sha256"]["src/prigm_metrics.py"] = experiment.sha256(source)
    write_json(root / "run_manifest.json", manifest)
    pin = experiment.sha256(root / "run_manifest.json")
    for marker in (root / "jobs").glob("*/complete.json"):
        completion = read_json(marker)
        config_path = marker.parent / completion["attempt"] / "configuration.json"
        config = read_json(config_path)
        config["run_manifest_sha256"] = pin
        write_json(config_path, config)
        completion["run_manifest_sha256"] = pin
        write_json(marker, completion)
    results = read_json(root / "results.json")
    results["run_manifest_sha256"] = pin
    write_json(root / "results.json", results)
    reseal(root)
    output = tmp_path / "historical-refinement"
    # Original refinement production precedes maintained-source admission.
    with monkeypatch.context() as producer:
        producer.setattr(experiment, "ROOT", root / "source")
        _complete_refinement(root, pin, output, monkeypatch)
    monkeypatch.setattr(dart, "CORE_SHA256", pin)
    monkeypatch.setattr(dart, "CORE_SOURCES", {Path(k).name: v for k, v in manifest["code_sha256"].items()})
    monkeypatch.setattr(paper, "HISTORICAL_REFINEMENT_SHA256", experiment.sha256(output / "refinement_manifest.json"))
    return root, output, pin, manifest["code_sha256"]


def test_historical_core_and_refinement_preserve_producers_and_nested_build(historical_campaign, tmp_path, monkeypatch):
    import prigm_refine_evaluation as refinement

    root, output, pin, producer_sources = historical_campaign
    for verify in (lambda: experiment.verify_run(root),
                   lambda: paper.verify_evidence(root, pin),
                   lambda: refinement.verify_refinement(root, output),
                   lambda: paper.verify_refinement_evidence(root, output)):
        with pytest.raises(ValueError, match="Frozen experiment source changed"):
            verify()
    core = paper.verify_evidence(root, pin, historical=True)
    refined = paper.verify_refinement_evidence(root, output, historical=True)
    assert core["code_sha256"] == producer_sources
    assert len(refined["results"]["jobs"]) == 60
    assert refined["code_sha256"] == {refinement.SOURCE_NAME: experiment.sha256(refinement.SOURCE_FILE)}
    def figures(directory, *args):
        for name in paper.OUTPUTS:
            if name.startswith("figures/"):
                Image.new("RGB", (2, 2)).save(directory / name)
    monkeypatch.setattr(paper, "_figures", figures)  # Rendering only; all evidence gates stay real.
    public = paper.build_paper(root, tmp_path / "historical-paper", pin, output, historical=True)
    assert public["code_sha256"] == producer_sources
    assert public["refinement"] == refined


@pytest.mark.parametrize("mutation", ["core_source", "resealed_core_source", "resealed_checkpoint",
                                      "refiner_source", "resealed_refiner_manifest", "current_refiner"])
def test_historical_refinement_rejects_source_and_checkpoint_tampering(historical_campaign, tmp_path, monkeypatch, mutation):
    import prigm_refine_evaluation as refinement

    root, output, pin, producer_sources = historical_campaign
    if mutation in ("core_source", "resealed_core_source"):
        source = root / "source/src/prigm_metrics.py"
        source.write_bytes(source.read_bytes() + b"\n# tampered producer\n")
        if mutation == "resealed_core_source":
            manifest = read_json(root / "run_manifest.json")
            manifest["code_sha256"]["src/prigm_metrics.py"] = experiment.sha256(source)
            write_json(root / "run_manifest.json", manifest)
        reseal(root)
    elif mutation == "resealed_checkpoint":
        attempt = experiment.verified_completion(root, "gaussian_A_42")
        (attempt / "best.weights.h5").write_bytes(b"changed checkpoint, including resealed core declarations")
        reseal(root)
    elif mutation == "refiner_source":
        (output / "source" / refinement.SOURCE_NAME).write_text("# changed archived refiner\n")
    elif mutation == "resealed_refiner_manifest":
        path = output / refinement.MANIFEST
        manifest = read_json(path)
        manifest["code_sha256"][refinement.SOURCE_NAME] = "0" * 64
        write_json(path, manifest)
        archive = read_json(output / "archive_manifest.json")
        archive["artifact_sha256"][refinement.MANIFEST] = experiment.sha256(path)
        write_json(output / "archive_manifest.json", archive)
    else:
        path = tmp_path / "changed-refiner.py"
        path.write_text("# changed current refiner\n")
        monkeypatch.setattr(refinement, "SOURCE_FILE", path)
    with pytest.raises(ValueError):
        paper.verify_refinement_evidence(root, output, historical=True)


def test_dart_private_admission_authenticates_historical_sources_before_batch_reads(historical_campaign, tmp_path, monkeypatch):
    import prigm_dart_paper_evidence as dart
    import prigm_v2_experiment as runner
    from test_prigm_v2_experiment import _dart_stage

    core, _, _, _ = historical_campaign
    stage = _dart_stage(tmp_path)
    # Freeze a distinct producer identity; neither archived source is executed.
    source = stage / "source/src/prigm_metrics.py"
    source.write_bytes((core / "source/src/prigm_metrics.py").read_bytes())
    manifest = runner.read_json(stage / "stage_manifest.json")
    manifest["code_sha256"]["src/prigm_metrics.py"] = runner.sha256(source)
    runner.atomic_json(stage / "stage_manifest.json", manifest)
    stage_pin = runner.sha256(stage / "stage_manifest.json")
    monkeypatch.setattr(dart, "STAGE_SOURCES", {Path(k).name: v for k, v in manifest["code_sha256"].items()})
    with pytest.raises(ValueError, match="source changed after stage build"):
        runner.verify_stage(stage)
    def batches(*args):
        raise RuntimeError("authenticated historical stage reached batch reading")
    monkeypatch.setattr(dart, "_batch_records", batches)
    with pytest.raises(RuntimeError, match="authenticated historical stage"):
        dart.verify_dart_extension(core, stage, [tmp_path], [tmp_path], stage_pin, None)
    source.write_bytes(source.read_bytes() + b"# tamper after independent pin\n")
    with pytest.raises(ValueError, match="Input SHA256 differs"):
        dart.verify_dart_extension(core, stage, [tmp_path], [tmp_path], stage_pin, None)


def test_context_keeps_isolated_duplicate_and_year_end_rows_and_hashes_source(campaign):
    root, pinned = campaign
    evidence = paper.verify_evidence(root, pinned)
    context = evidence["contexts"]["23461t2022"]
    np.testing.assert_array_equal(context["heights"], [4000.125, 4000.375, 4001.625])
    np.testing.assert_array_equal(context["days"], [1., 1., 365.5])
    path = paper.ROOT / "datasets" / "23461t2022.txt"
    path.write_bytes(path.read_bytes() + b"2022 12 31 13 0 0 1 4002.875\n")
    with pytest.raises(ValueError, match="Public source hash differs"):
        paper.verify_evidence(root, pinned)


def test_prior_selection_is_fixed_index_zero_seed_42_in_meters(campaign):
    root, pinned = campaign
    evidence = paper.verify_evidence(root, pinned)
    assert len(evidence["prior"]) == 16
    for dataset in paper.DATASETS[1:]:
        for arm in paper.ARMS:
            identifier = experiment.job_id(dataset, arm, 42)
            arrays, _, _ = evaluation_fixture(arm)
            np.testing.assert_array_equal(evidence["prior"][identifier], 4000.25 + 2 * arrays["prior_samples"][0])


def test_prepared_schema_requires_oracle_and_nonempty_splits(campaign):
    root, _ = campaign
    data = paper._load_npz(root / "prepared" / "gaussian.npz")
    metadata = read_json(root / "prepared" / "metadata.json")["gaussian"]
    spec = experiment.specification("core")
    oracle = data.pop("oracle_covariance")
    with pytest.raises(ValueError, match="prepared array schema"):
        paper._prepared_statistics(data, metadata, "gaussian", spec)
    data["oracle_covariance"] = oracle
    metadata["windows"]["test"] = 0
    data["test"] = data["test"][:0]
    with pytest.raises(ValueError, match="Empty or invalid prepared"):
        paper._prepared_statistics(data, metadata, "gaussian", spec)


def test_extra_completed_identity_and_missing_completion_are_rejected(campaign):
    root, pinned = campaign
    extra = root / "jobs" / "gaussian_A_duplicate"
    extra.mkdir()
    path = extra / "complete.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unexpected completed job identity"):
        paper.verify_evidence(root, pinned)
    path.unlink()
    (root / "jobs" / "gaussian_A_42" / "complete.json").unlink()
    with pytest.raises(ValueError, match="regular artifact"):
        paper.verify_evidence(root, pinned)


def test_figures_use_all_ranges_scatter_only_context_and_fixed_prior(campaign, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    root, pinned = campaign
    evidence = paper.verify_evidence(root, pinned)
    paired = paper.paired_statistics(evidence["jobs"])
    figures = {}
    def capture(figure, path, **kwargs):
        figures[Path(path).name] = figure
    monkeypatch.setattr(Figure, "savefig", capture)
    paper._figures(tmp_path, evidence, paired)
    comparison = figures["reconstruction_generation.png"]
    assert len(comparison.axes) == 2
    assert comparison.axes[0].get_yscale() == "log"
    assert comparison.axes[1].get_yscale() == "linear"
    for axis, field, suffix in zip(comparison.axes, ("self_mse_ratio", "negative_iw_score_difference"), ("/A", "-A")):
        assert len(axis.containers) == 3
        for container, arm in zip(axis.containers, ("B", "C", "D")):
            values = [paired[d][field][arm + suffix] for d in paper.DATASETS]
            np.testing.assert_allclose(container.lines[0].get_ydata(), [v["median"] for v in values])
            segments = container.lines[2][0].get_segments()
            assert len(segments) == 5
            for segment, value in zip(segments, values):
                np.testing.assert_allclose(segment[:, 1], [value["min"], value["max"]])
    context = figures["full_record_context.png"]
    assert len(context.axes) == 4
    for axis, dataset in zip(context.axes, paper.DATASETS[1:]):
        assert len(axis.lines) == 0 and len(axis.collections) == 1
        scatter = axis.collections[0]
        assert np.all(scatter.get_sizes() < 1)
        np.testing.assert_array_equal(scatter.get_offsets()[:, 1], evidence["contexts"][dataset]["heights"])
        assert axis.get_xlim()[1] >= evidence["contexts"][dataset]["days"].max()
    prior = figures["prior_samples.png"]
    assert len(prior.axes) == 4
    for axis, dataset in zip(prior.axes, paper.DATASETS[1:]):
        assert len(axis.lines) == 4
        for line, arm in zip(axis.lines, paper.ARMS):
            np.testing.assert_array_equal(line.get_ydata(), evidence["prior"][experiment.job_id(dataset, arm, 42)])


def test_cli_requires_pin_and_returns_only_artifact_names(tmp_path, monkeypatch, capsys):
    arguments = ["--run-root", str(tmp_path / "run"), "--paper-dir", str(tmp_path / "paper")]
    with pytest.raises(SystemExit) as error:
        paper.main(arguments)
    assert error.value.code == 2
    calls = []
    def build(root, destination, digest, refinement_root):
        calls.append((root, destination, digest, refinement_root))
        return {"run_manifest_sha256": digest, "output_sha256": dict.fromkeys(paper.OUTPUTS, "0" * 64)}
    monkeypatch.setattr(paper, "build_paper", build)
    paper.main([*arguments, "--expected-manifest-sha256", "0" * 64])
    assert calls == [(tmp_path / "run", tmp_path / "paper", "0" * 64, None)]
    result = json.loads(capsys.readouterr().out)
    assert result == {"run_manifest_sha256": "0" * 64, "artifacts": [*paper.OUTPUTS, paper.EVIDENCE_NAME]}


def test_renderer_failure_publishes_no_assets(campaign, tmp_path, monkeypatch):
    root, pinned = campaign
    destination = tmp_path / "paper"
    def fail(*args):
        raise RuntimeError("render failure")
    monkeypatch.setattr(paper, "_figures", fail)
    with pytest.raises(RuntimeError, match="render failure"):
        paper.build_paper(root, destination, pinned)
    assert list(destination.iterdir()) == []
