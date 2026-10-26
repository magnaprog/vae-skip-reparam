"""Build CPU-only PriGM paper evidence from a complete, pinned core archive.

No training, checkpoint loading, or manuscript editing occurs here. Table fragments
contain data rows only; the caller terminates the final row after input. Column contracts:
results: dataset, paired seed median[min,max] of negative IW score B-A, C-A, B-D;
all_runs: dataset, arm, seed, best step, self MSE, corrected MSE, negative IW
score, KL/window, energy, median ESS; data: dataset, train/validation/test window
counts, normalization mean/std, base variance; baselines: dataset, model,
negative log score/dimension, energy. Gaussian normalization is known, not fitted.
"""
from __future__ import annotations

import argparse
import calendar
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import numpy as np
import pandas as pd

import prigm_experiment as experiment

ROOT = Path(__file__).resolve().parents[1]
DATASETS, ARMS, SEEDS = experiment.DATASETS, experiment.ARMS, experiment.SEEDS
LABELS = {"gaussian": "Gaussian", **{name: name.replace("t", "/") for name in DATASETS[1:]}}
SCORE = "negative_log_score_nats_per_dimension"
OUTPUTS = (
    "generated/results_rows.tex", "generated/all_runs_rows.tex",
    "generated/data_rows.tex", "generated/baseline_rows.tex",
    "generated/summary_values.tex", "figures/reconstruction_generation.png",
    "figures/full_record_context.png", "figures/prior_samples.png",
)
EVIDENCE_NAME = "paper_evidence_manifest.json"
# Immutable original refinement input; its producer is not the current builder.
HISTORICAL_REFINEMENT_SHA256 = "e7d4bc227457ff3e4dacee1a2e5dbdfe1cdd15ec049a1e310ba2a84511bfb54f"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "Expected a lowercase SHA256 digest")
    return value


def _safe_path(root, name):
    """Reject noncanonical relative names and symlinks at every component."""
    _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
             "Unsafe relative artifact path")
    parts = name.split("/")
    _require(not PurePosixPath(name).is_absolute()
             and all(part not in ("", ".", "..") and ":" not in part for part in parts),
             "Unsafe relative artifact path")
    path = Path(root)
    for part in parts:
        path = path / part
        _require(not path.is_symlink(), "Symlink artifact path is not allowed")
    return path


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        _require(np.isfinite(value), "Nonfinite JSON number")
        return value

    def invalid_constant(text):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_float=finite_float, parse_constant=invalid_constant)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ member")
        return {name: archive[name] for name in archive.files}


def _array(value, shape, label):
    value = np.asarray(value)
    _require(value.shape == shape and value.dtype.kind in "fiu"
             and np.isfinite(value).all(), f"Invalid or truncated array: {label}")
    return value


def _close(actual, expected, label):
    _require(np.asarray(actual).dtype.kind in "fiu"
             and np.shape(actual) == np.shape(expected)
             and np.allclose(actual, expected, rtol=1e-10, atol=1e-10),
             f"Recomputed value differs: {label}")


def _index_jobs(records):
    _require(isinstance(records, list), "Results must be a job list")
    indexed = {}
    for record in records:
        _require(isinstance(record, dict) and type(record.get("seed")) is int,
                 "Invalid job identity")
        identifier = experiment.job_id(record["dataset"], record["arm"], record["seed"])
        _require(identifier not in indexed, "Duplicate job identity")
        indexed[identifier] = record
    expected = {experiment.job_id(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
    _require(set(indexed) == expected, "Missing or unexpected jobs in core matrix")
    return indexed


def _verify_archive(root, artifacts):
    _require(isinstance(artifacts, dict) and artifacts, "Empty archive artifact identity")
    # Validate ALL names before reading any named artifact.
    paths = {name: _safe_path(root, name) for name in artifacts}
    for name, path in paths.items():
        _require(experiment.sha256(path) == _digest(artifacts[name]),
                 f"Archive artifact changed: {name}")


def _prepared_statistics(data, metadata, name, spec):
    _require(metadata["dataset"] == name, "Prepared dataset identity differs")
    splits = ("train", "validation", "test")
    expected = {*splits, *(f"starts_{s}" for s in splits), "mean", "scale", "base_variance",
                "ppca_mean", "ppca_covariance", "diagonal_covariance"}
    if name == "gaussian":
        expected.add("oracle_covariance")
    _require(set(data) == expected, "Incomplete prepared array schema")
    dimension = spec["window"]
    _array(data["ppca_mean"], (dimension,), "PPCA mean")
    for key in ("ppca_covariance", "diagonal_covariance", "oracle_covariance"):
        if key in data:
            _array(data[key], (dimension, dimension), key)
    counts = {}
    for split in ("train", "validation", "test"):
        n = metadata["windows"][split]
        _require(type(n) is int and n > 0, "Empty or invalid prepared split")
        _array(data[split], (n, spec["window"]), f"{name}/{split}")
        _array(data[f"starts_{split}"], (n,), f"{name}/starts_{split}")
        _require(metadata["counts"][split]["windows"] == n, "Window counts differ")
        counts[split] = n
    for key in ("mean", "scale", "base_variance"):
        _array(data[key], (), key)
        _close(metadata[key], float(data[key]), key)
    _close(metadata["std"], float(data["scale"]), "std")
    _require(float(data["scale"]) > 0 and float(data["base_variance"]) > 0,
             "Invalid normalization or base variance")
    return {"windows": counts, "mean": float(data["mean"]),
            "std": float(data["scale"]), "base_variance": float(data["base_variance"])}


def evaluation_statistics(arrays, data, spec):
    """Reconstruct every scalar emitted by evaluate_job from its NPZ arrays."""
    n, dimension = data["test"].shape
    _require(n > 0 and dimension == spec["window"], "Empty or invalid test data")
    count = min(n, spec["stability_windows"])
    shapes = {key: (n,) for key in ("log_score", "ess", "normalized_ess", "self_mse",
                                    "corrected_mse", "kl", "permutation_delta", "energy")}
    shapes.update({key: (count,) for key in ("stability_log_score", "stability_ess",
                                            "replicate_log_score", "stability_primary_log_score")})
    shapes.update(prior_samples=(spec["prior_samples"], dimension),
                  posterior_mean=(n, dimension),
                  posterior_kl_per_dimension=(n, spec["latent_dim"]))
    _require(set(arrays) == set(shapes), "Incomplete evaluation schema")
    for key, shape in shapes.items():
        _array(arrays[key], shape, key)
    for key, samples in (("ess", spec["evaluation_samples"]),
                         ("stability_ess", spec["stability_samples"])):
        _require(np.all(arrays[key] >= 1 - 1e-7)
                 and np.all(arrays[key] <= samples * (1 + 1e-7)), "Invalid ESS")
    for key in ("self_mse", "corrected_mse"):
        _require(np.all(arrays[key] >= 0), "Negative MSE")
    _close(arrays["normalized_ess"], arrays["ess"] / spec["evaluation_samples"], "normalized ESS")
    _close(arrays["stability_primary_log_score"], arrays["log_score"][:count], "stability subset")
    _close(arrays["corrected_mse"], np.square(data["test"] - arrays["posterior_mean"]).mean(axis=1),
           "corrected MSE from posterior mean")
    scale = float(data["scale"])
    mean = lambda key: float(arrays[key].mean())
    score = -mean("log_score") / dimension
    return {
        "test_windows": n, SCORE: score,
        "negative_log_score_meter_coordinates_per_dimension": score + float(np.log(scale)),
        "median_effective_samples": float(np.median(arrays["ess"])),
        "median_normalized_ess": float(np.median(arrays["normalized_ess"])),
        "self_mse": mean("self_mse"), "corrected_mse": mean("corrected_mse"),
        "kl_nats_per_window": mean("kl"), "energy_score": mean("energy"),
        "energy_score_meters": mean("energy") * scale,
        "permutation_corrected_mse_increase": mean("permutation_delta"),
        "stability_mean_log_score_change_per_dimension": float(
            (arrays["stability_log_score"] - arrays["log_score"][:count]).mean() / dimension),
        "replicate_mean_log_score_change_per_dimension": float(
            (arrays["replicate_log_score"] - arrays["log_score"][:count]).mean() / dimension),
    }


def _mapped_statistics(record, arm):
    _require(record["target_arm"] == {"A": "C", "B": "D", "C": "A", "D": "B"}[arm],
             "Mapped control identity differs")
    result = {"target_arm": record["target_arm"]}
    for key in ("max_absolute_log_density_difference", "max_absolute_sample_mean_difference",
                "relative_log_density_tolerance", "absolute_log_density_tolerance"):
        value = record[key]
        _require(type(value) in (int, float) and np.isfinite(value) and value >= 0,
                 "Invalid mapped control statistic")
        result[key] = value
    _close(result["relative_log_density_tolerance"], 3e-5, "mapped relative tolerance")
    _close(result["absolute_log_density_tolerance"], 0.01, "mapped absolute tolerance")
    return result


def read_observed_context(path, expected_sha256):
    """Use all valid observed T=1 rows, not only window-eligible segments."""
    source = Path(path).read_bytes()
    _require(sha256(source).hexdigest() == _digest(expected_sha256), "Public source hash differs")
    frame = pd.read_csv(BytesIO(source), sep=r"\s+", header=None, skiprows=2,
                        dtype=np.float64, float_precision="round_trip", skip_blank_lines=False)
    _require(frame.shape[1] == 8, "DART context requires eight columns")
    dates = pd.to_datetime(dict(zip(("year", "month", "day", "hour", "minute", "second"),
                                    (frame.iloc[:, i] for i in range(6)))), utc=True, errors="coerce")
    heights = frame.iloc[:, 7].to_numpy(dtype=np.float64)
    keep = (frame.iloc[:, 6].to_numpy() == 1) & np.isfinite(heights) & (heights <= 9000) & dates.notna().to_numpy()
    _require(keep.any(), "No observed context data")
    dates = dates[keep]
    year = int(Path(path).stem.split("t")[1])
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    days = 1 + (dates - start).dt.total_seconds().to_numpy() / 86400
    return {"days": days, "heights": heights[keep], "year": year}


def _historical_core_identity():
    """Authenticate maintained imports before admitting the original DART core."""
    from prigm_dart_paper_evidence import CORE_SHA256, CORE_SOURCES, _imports

    _imports()
    return {"manifest_sha256": CORE_SHA256,
            "code_sha256": {"src/" + name: digest for name, digest in CORE_SOURCES.items()}}


def verify_evidence(run_root, expected_manifest_sha256, *, historical=False):
    """Verify the runner's exact schema and return only allowlisted evidence."""
    root = Path(run_root).resolve(strict=True)
    pinned = _digest(expected_manifest_sha256)
    identity = _historical_core_identity() if historical else None
    if historical:
        _require(pinned == identity["manifest_sha256"], "Unapproved historical core manifest")
    manifest_path = _safe_path(root, "run_manifest.json")
    _require(experiment.sha256(manifest_path) == pinned, "Expected root manifest SHA256 differs")
    manifest = _read_json(manifest_path)
    _require(manifest.get("purpose") == "core", "Paper requires purpose core")
    _require(manifest.get("schema_version") == 1 and manifest["datasets"] == list(DATASETS)
             and manifest["specification"] == experiment.specification("core"), "Core protocol differs")
    prepared_names = {"metadata.json", *(name + ".npz" for name in DATASETS)}
    _require(set(manifest["prepared_sha256"]) == prepared_names
             and set(manifest["code_sha256"]) == set(experiment.CODE_FILES), "Incomplete run identity")
    critical = {"run_manifest.json", "results.json"}
    critical.update("prepared/" + name for name in prepared_names)
    critical.update("source/" + name for name in experiment.CODE_FILES)
    for name in critical:
        _safe_path(root, name)
    archive_path = _safe_path(root, "archive_manifest.json")
    archive_sha = experiment.sha256(archive_path)
    archive = _read_json(archive_path)
    _require(archive.get("schema_version") == 1, "Unsupported archive schema")
    artifacts = archive["artifact_sha256"]
    _verify_archive(root, artifacts)
    _require(critical <= set(artifacts), "Archive missing critical artifacts")
    verified = (experiment.verify_run(root, historical_identity=identity) if historical
                else experiment.verify_run(root))
    _require(verified == manifest, "Verified run differs")
    results = _read_json(_safe_path(root, "results.json"))
    _require(results.get("schema_version") == 1 and results.get("purpose") == "core"
             and results.get("run_manifest_sha256") == pinned
             and results.get("specification") == manifest["specification"], "Results belong to a different run")
    summaries = _index_jobs(results["results"])
    metadata = _read_json(_safe_path(root, "prepared/metadata.json"))
    _require(set(metadata) == set(DATASETS) and results["dataset_metadata"] == metadata,
             "Results metadata differs from hashed preparation")
    spec = manifest["specification"]
    prepared, dataset_statistics, contexts, sources = {}, {}, {}, {}
    for name in DATASETS:
        prepared[name] = _load_npz(_safe_path(root, f"prepared/{name}.npz"))
        dataset_statistics[name] = _prepared_statistics(prepared[name], metadata[name], name, spec)
        if name != "gaussian":
            sources[f"datasets/{name}.txt"] = _digest(metadata[name]["source_sha256"])
            context = read_observed_context(_safe_path(ROOT, f"datasets/{name}.txt"), metadata[name]["source_sha256"])
            _require(len(context["days"]) == metadata[name]["observed_t1_rows"], "Observed source count differs")
            contexts[name] = context
            dataset_statistics[name]["observed_t1_rows"] = len(context["days"])

    jobs, prior = {}, {}
    job_root = _safe_path(root, "jobs")
    for path in job_root.iterdir():
        _require(not path.is_symlink(), "Symlink job directory")
        _require(path.name in summaries or not (path / "complete.json").exists(),
                 "Unexpected completed job identity")
    for identifier, summary in summaries.items():
        completion_name = f"jobs/{identifier}/complete.json"
        completion = _read_json(_safe_path(root, completion_name))
        _require(completion["run_manifest_sha256"] == pinned, "Completion belongs to a different run")
        attempt_name = completion["attempt"]
        _require(isinstance(attempt_name, str) and re.fullmatch(r"attempt-[A-Za-z0-9-]+", attempt_name),
                 "Unsafe job attempt path")
        attempt_relative = f"jobs/{identifier}/{attempt_name}"
        attempt = _safe_path(root, attempt_relative)
        _require(set(completion["artifact_sha256"]) == set(experiment.JOB_ARTIFACTS),
                 "Incomplete job artifact identity")
        critical.add(completion_name)
        for filename, digest in completion["artifact_sha256"].items():
            relative = f"{attempt_relative}/{filename}"
            _safe_path(root, relative)
            critical.add(relative)
            _require(artifacts.get(relative) == _digest(digest), "Archive/completion artifact hashes differ")
        _require(experiment.verified_completion(root, identifier) == attempt, "Missing verified completion")
        configuration = _read_json(attempt / "configuration.json")
        _require(configuration["run_manifest_sha256"] == pinned
                 and configuration["specification"] == spec, "Configuration belongs to a different run")
        _require(_read_json(attempt / "summary.json") == summary, "Root/job summaries differ")
        arrays = _load_npz(attempt / "evaluation.npz")
        stats = evaluation_statistics(arrays, prepared[summary["dataset"]], spec)
        for key, value in stats.items():
            _close(summary[key], value, f"{identifier}/{key}")
        _require(type(summary["best_step"]) is int and type(summary["steps_run"]) is int
                 and 0 < summary["best_step"] <= summary["steps_run"] <= spec["max_steps"]
                 and summary["best_step"] % spec["validation_every"] == 0,
                 "Invalid checkpoint selection steps")
        jobs[identifier] = {key: summary[key] for key in ("dataset", "arm", "seed", "best_step", "steps_run")}
        jobs[identifier].update(stats)
        jobs[identifier]["mapped_control"] = _mapped_statistics(summary["mapped_control"], summary["arm"])
        jobs[identifier]["artifact_sha256"] = dict(completion["artifact_sha256"])
        if summary["seed"] == 42 and summary["dataset"] != "gaussian":
            data = prepared[summary["dataset"]]
            sample = float(data["mean"]) + float(data["scale"]) * arrays["prior_samples"][0]
            prior[identifier] = _array(sample, (spec["window"],), "prior sample in meters")
    _require(set(artifacts) == critical, "Archive has missing or unexpected critical artifacts")
    # Reuse the frozen CPU baseline definition, not the root's scalar claims.
    baselines = experiment.baseline_results(root, manifest)
    _require(set(results["baselines"]) == set(DATASETS), "Incomplete baseline datasets")
    for name, records in baselines.items():
        _require(set(results["baselines"][name]) == set(records), "Baseline identities differ")
        for label, values in records.items():
            _require(set(results["baselines"][name][label]) == set(values), "Baseline schema differs")
            for key, value in values.items():
                _close(results["baselines"][name][label][key], value, f"{name}/{label}/{key}")
    # Reject files changed during verification; no output directory exists yet.
    _verify_archive(root, artifacts)
    _require(experiment.sha256(archive_path) == archive_sha, "Archive manifest changed during verification")
    return {
        "schema_version": 1, "purpose": "core", "run_manifest_sha256": pinned,
        "archive_manifest_sha256": archive_sha, "code_sha256": dict(manifest["code_sha256"]),
        "prepared_sha256": dict(manifest["prepared_sha256"]), "public_source_sha256": sources,
        "specification": spec, "dataset_statistics": dataset_statistics, "jobs": jobs,
        "baselines": baselines, "contexts": contexts, "prior": prior,
    }


def _range(values):
    values = np.asarray(values, dtype=float)
    _require(values.shape == (len(SEEDS),) and np.isfinite(values).all(), "Invalid paired seed statistics")
    return {"median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()), "by_seed": dict(zip(map(str, SEEDS), values.tolist()))}


def verify_refinement_evidence(core_root, refinement_root, *, historical=False):
    import prigm_refine_evaluation as refinement

    root = Path(refinement_root).resolve(strict=True)
    if historical:
        # The pinned original manifest binds all sixty core checkpoint identities.
        # Reuse verify_job below rather than copying the refiner's job verifier.
        verify_evidence(core_root, refinement.CORE_MANIFEST_SHA256, historical=True)
        path = _safe_path(root, refinement.MANIFEST)
        _require(experiment.sha256(path) == HISTORICAL_REFINEMENT_SHA256,
                 "Unapproved historical refinement manifest")
        manifest = _read_json(path)
        _require(manifest["core_manifest_sha256"] == refinement.CORE_MANIFEST_SHA256
                 and manifest["specification"] == refinement.specification()
                 and set(manifest["code_sha256"]) == {refinement.SOURCE_NAME}
                 and set(manifest["jobs"]) == {experiment.job_id(*job) for job in refinement.canonical_jobs()},
                 "Historical refinement protocol differs")
        source_sha = manifest["code_sha256"][refinement.SOURCE_NAME]
        _require(experiment.sha256(refinement.SOURCE_FILE) == source_sha
                 and experiment.sha256(_safe_path(root, "source/" + refinement.SOURCE_NAME)) == source_sha,
                 "Historical refinement source differs")
    else:
        manifest = refinement.verify_refinement(core_root, root)
    archive = _read_json(_safe_path(root, "archive_manifest.json"))
    artifacts = archive["artifact_sha256"]
    _verify_archive(root, artifacts)
    _require(archive["core_manifest_sha256"] == refinement.CORE_MANIFEST_SHA256,
             "Refinement archive belongs to a different core")
    expected = {refinement.MANIFEST, "source/" + refinement.SOURCE_NAME, "results.json"}
    summaries, records = [], {}
    for job in refinement.canonical_jobs():
        identifier = experiment.job_id(*job)
        verified = refinement.verify_job(core_root, root, manifest, identifier)
        _require(verified is not None, "Missing refinement job")
        attempt, summary, arrays = verified
        summaries.append({**manifest["jobs"][identifier], "metrics": summary["metrics"]})
        records[identifier] = arrays
        expected.add(str((attempt.parent / "complete.json").relative_to(root)))
        expected.update(str((attempt / name).relative_to(root)) for name in refinement.ARTIFACTS)
    _require(set(artifacts) == expected, "Incomplete refinement archive closure")
    results = _read_json(_safe_path(root, "results.json"))
    _require(results["core_manifest_sha256"] == refinement.CORE_MANIFEST_SHA256
             and results["refinement_manifest_sha256"] == experiment.sha256(root / refinement.MANIFEST)
             and results["specification"] == refinement.specification()
             and results["coordinates"] == "prepared_standardized"
             and results["jobs"] == summaries
             and results["paired_comparisons"] == refinement.paired_comparisons(records),
             "Refinement results differ from verified arrays")
    _verify_archive(root, artifacts)
    if historical:
        _require(experiment.sha256(root / refinement.MANIFEST) == HISTORICAL_REFINEMENT_SHA256,
                 "Historical refinement manifest changed")
    return {"manifest_sha256": experiment.sha256(root / refinement.MANIFEST),
            "archive_manifest_sha256": experiment.sha256(root / "archive_manifest.json"),
            "code_sha256": manifest["code_sha256"],
            "results": {"specification": refinement.specification(), "jobs": summaries,
                        "paired_comparisons": refinement.paired_comparisons(records)}}


def paired_statistics(jobs):
    """Pair optimization seeds before taking medians and full seed ranges."""
    _index_jobs(list(jobs.values()))
    paired = {}
    for dataset in DATASETS:
        records = {(a, s): jobs[experiment.job_id(dataset, a, s)] for a in ARMS for s in SEEDS}
        differences, ratios = {}, {}
        for left, right in (("B", "A"), ("C", "A"), ("D", "A"), ("B", "D")):
            differences[f"{left}-{right}"] = _range([
                records[left, seed][SCORE] - records[right, seed][SCORE] for seed in SEEDS])
        for arm in ("B", "C", "D"):
            _require(all(records["A", seed]["self_mse"] > 0 and records[arm, seed]["self_mse"] > 0
                         for seed in SEEDS), "Log-ratio figure requires positive self MSE")
            ratios[f"{arm}/A"] = _range([
                records[arm, seed]["self_mse"] / records["A", seed]["self_mse"] for seed in SEEDS])
        paired[dataset] = {"negative_iw_score_difference": differences, "self_mse_ratio": ratios}
    return paired


def _number(value):
    _require(np.isfinite(value), "Cannot format nonfinite statistic")
    text = format(float(value), ".4g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"
    return text


def _row(cells):
    return " & ".join(map(str, cells))


def table_fragments(evidence):
    """Return fragments whose final row is terminated by the including table."""
    jobs = evidence["jobs"]
    paired = paired_statistics(jobs)
    results, runs, data_rows, baselines = [], [], [], []
    for dataset in DATASETS:
        values = paired[dataset]["negative_iw_score_difference"]
        cells = [LABELS[dataset]]
        for contrast in ("B-A", "C-A", "B-D"):
            item = values[contrast]
            cells.append(rf"\shortstack{{${_number(item['median'])}$\\$[{_number(item['min'])},\,{_number(item['max'])}]$}}")
        results.append(_row(cells))
        for arm in ARMS:
            for seed in SEEDS:
                job = jobs[experiment.job_id(dataset, arm, seed)]
                runs.append(_row([LABELS[dataset], arm, seed, job["best_step"], *[
                    _number(job[key]) for key in ("self_mse", "corrected_mse", SCORE,
                                                  "kl_nats_per_window", "energy_score", "median_effective_samples")]]))
        stats = evidence["dataset_statistics"][dataset]
        data_rows.append(_row([LABELS[dataset], *[stats["windows"][s] for s in ("train", "validation", "test")],
                               format(stats["mean"], ".8g"),
                               *[_number(stats[k]) for k in ("std", "base_variance")]]))
        for key, label in (("oracle", "True Gaussian"), ("ppca", "PPCA"), ("diagonal_gaussian", "Diagonal Gaussian")):
            if key in evidence["baselines"][dataset]:
                record = evidence["baselines"][dataset][key]
                baselines.append(_row([LABELS[dataset], label, _number(record[SCORE]), _number(record["energy_score"])]))
    mapped = [job["mapped_control"] for job in jobs.values()]
    macros = {
        "PrigmJobCount": str(len(jobs)), "PrigmDatasetCount": str(len(DATASETS)),
        "PrigmSeedCount": str(len(SEEDS)), "PrigmCoreSHA": evidence["run_manifest_sha256"],
        "PrigmWindowLength": str(evidence["specification"]["window"]),
        "PrigmEvaluationSamples": str(evidence["specification"]["evaluation_samples"]),
        "PrigmPriorSampleCount": str(evidence["specification"]["prior_samples"]),
        "PrigmMaxMappedLogDensityError": _number(max(x["max_absolute_log_density_difference"] for x in mapped)),
        "PrigmMaxMappedSampleMeanError": _number(max(x["max_absolute_sample_mean_difference"] for x in mapped)),
        "PrigmMinMedianESS": _number(min(x["median_effective_samples"] for x in jobs.values())),
        "PrigmMaxAbsStabilityScoreChange": _number(max(abs(x["stability_mean_log_score_change_per_dimension"]) for x in jobs.values())),
        "PrigmMaxAbsReplicateScoreChange": _number(max(abs(x["replicate_mean_log_score_change_per_dimension"]) for x in jobs.values())),
    }
    return {
        "generated/results_rows.tex": " \\\\\n".join(results) + "\n",
        "generated/all_runs_rows.tex": " \\\\\n".join(runs) + "\n",
        "generated/data_rows.tex": " \\\\\n".join(data_rows) + "\n",
        "generated/baseline_rows.tex": " \\\\\n".join(baselines) + "\n",
        "generated/summary_values.tex": "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items()),
    }


def _figures(directory, evidence, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name, description):
        fig.tight_layout()
        fig.savefig(directory / "figures" / name, dpi=200,
                    metadata={"Software": "PriGM CPU evidence builder", "Description": description})
        plt.close(fig)

    colors = {"A": "#333333", "B": "#0072B2", "C": "#D55E00", "D": "#009E73"}
    with plt.rc_context({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif",
                         "font.size": 10, "axes.formatter.useoffset": False, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
        for axis, field, suffix in zip(axes, ("self_mse_ratio", "negative_iw_score_difference"), ("/A", "-A")):
            for offset, arm in zip((-0.22, 0, 0.22), ("B", "C", "D")):
                stats = [paired[d][field][arm + suffix] for d in DATASETS]
                middle = np.array([s["median"] for s in stats])
                error = np.array([[s["median"] - s["min"] for s in stats],
                                  [s["max"] - s["median"] for s in stats]])
                axis.errorbar(np.arange(len(DATASETS)) + offset, middle, yerr=error,
                              fmt="o", linestyle="none", capsize=3, markersize=4,
                              color=colors[arm], label=arm + suffix)
            axis.set_xticks(np.arange(len(DATASETS)),
                            ["Gaussian" if d == "gaussian" else d.replace("t", "\n") for d in DATASETS],
                            fontsize=9.5, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=9.5)
        axes[0].set_yscale("log")
        axes[0].axhline(1, color="0.5", linewidth=0.8)
        axes[0].set_ylabel("Raw self-MSE ratio to A (log scale)")
        axes[0].set_title("Self-reconstruction")
        axes[1].axhline(0, color="0.5", linewidth=0.8)
        axes[1].set_ylabel("Change in negative IW score\n(nats per coordinate)")
        axes[1].set_title("Importance-weighted score")
        save(fig, "reconstruction_generation.png", "All five datasets, all three paired optimization seeds; medians and full seed ranges.")

        fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
        for axis, dataset in zip(axes.flat, DATASETS[1:]):
            context = evidence["contexts"][dataset]
            axis.scatter(context["days"], context["heights"], s=0.35, linewidths=0, color=colors["A"])
            axis.set_xlim(min(1, context["days"].min()),
                          max(366 + calendar.isleap(context["year"]), context["days"].max()))
            axis.set_title(LABELS[dataset])
            axis.set_xlabel("Day of year")
            axis.set_ylabel("Observed T=1 HEIGHT (m)")
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        save(fig, "full_record_context.png", "All finite observed T=1 HEIGHT reports at or below 9000 m with valid timestamps; no window eligibility filtering or connecting lines.")

        fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
        hours = np.arange(evidence["specification"]["window"]) * 0.25
        for axis, dataset in zip(axes.flat, DATASETS[1:]):
            for arm in ARMS:
                axis.plot(hours, evidence["prior"][experiment.job_id(dataset, arm, 42)],
                          linewidth=0.9, color=colors[arm], label=arm)
            axis.set_title(LABELS[dataset])
            axis.set_xlabel("Elapsed time (h)")
            axis.set_ylabel("Generated HEIGHT (m)")
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.legend(ncol=2, fontsize=9.5, handlelength=1.2, columnspacing=0.7, borderpad=0.25)
        fig.suptitle("Unconditional prior draw index 0; optimization seed 42", fontsize=10)
        save(fig, "prior_samples.png", "Actual saved unconditional prior sample index 0, optimization seed 42, all four arms; converted using training normalization. No test conditioning or sample selection.")


def build_paper(run_root, paper_dir, expected_manifest_sha256, refinement_root=None, *, historical=False):
    """Publish verified assets; return the sanitized evidence manifest dictionary.

    output_sha256 lists every generated payload relative to paper_dir. The
    evidence manifest is not recursively self-hashed. It is replaced
    last; rendering failures publish nothing. Existing manuscripts are untouched.
    """
    policy = {"historical": True} if historical else {}
    evidence = verify_evidence(run_root, expected_manifest_sha256, **policy)
    paired = paired_statistics(evidence["jobs"])
    fragments = table_fragments(evidence)
    if refinement_root is not None:
        evidence["refinement"] = verify_refinement_evidence(run_root, refinement_root, **policy)
        rows = []
        for dataset in DATASETS:
            for contrast in ("B-A", "C-A", "B-D"):
                record = evidence["refinement"]["results"]["paired_comparisons"][dataset][contrast]
                cells = [LABELS[dataset], f"${contrast}$"]
                for budget in ("256", "4096", "16384"):
                    item = record[budget]
                    cells.append(rf"\shortstack{{${_number(item['median'])}$\\$[{_number(item['range'][0])},\,{_number(item['range'][1])}]$}}")
                rows.append(_row(cells))
        fragments["generated/refinement_rows.tex"] = " \\\\\n".join(rows) + "\n"
    outputs = (*fragments, *(name for name in OUTPUTS if name.startswith("figures/")))
    destination = Path(paper_dir).absolute()
    _require(not destination.is_symlink(), "Symlink paper directory")
    for name in (*outputs, EVIDENCE_NAME):
        target = _safe_path(destination, name)
        _require(not target.exists() or target.is_file(), "Output target is not a file")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prigm-evidence-", dir=destination) as temporary:
        stage = Path(temporary)
        (stage / "generated").mkdir()
        (stage / "figures").mkdir()
        for name, text in fragments.items():
            (stage / name).write_text(text, encoding="ascii")
        _figures(stage, evidence, paired)
        # These are private in-memory plotting inputs, never serialized.
        public = {key: value for key, value in evidence.items() if key not in ("contexts", "prior")}
        public.update(builder_sha256=experiment.sha256(Path(__file__)),
                      summary_statistics={"paired": paired, "job_count": len(evidence["jobs"]),
                                          "prior_figure_optimization_seed": 42, "prior_figure_sample_index": 0},
                      output_sha256={name: experiment.sha256(stage / name) for name in outputs})
        (stage / EVIDENCE_NAME).write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
        for name in (*outputs, EVIDENCE_NAME):
            target = _safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--refinement-root", type=Path)
    args = parser.parse_args(argv)
    evidence = build_paper(args.run_root, args.paper_dir, args.expected_manifest_sha256, args.refinement_root)
    print(json.dumps({"run_manifest_sha256": evidence["run_manifest_sha256"],
                      "artifacts": [*evidence["output_sha256"], EVIDENCE_NAME]}, indent=2))


if __name__ == "__main__":
    main()
