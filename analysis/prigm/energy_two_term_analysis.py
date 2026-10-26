#!/usr/bin/env python3
"""CPU-only retrospective accounting of frozen DART VAE diagnostic arrays.

No project modules, TensorFlow, neural execution, model fitting, or new samples
enter the scores. RNG use only checks the archived observation noise. The
manifest hash below pins the accepted evidence. Output must be a new directory.
Requires Python 3.11+, NumPy and SciPy. See --help for archive-root arguments.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist

MANIFEST_SHA = "db618b2d01b7cd7263058c4e554f957e61fd0580d7663a608e787f45b261a42c"
RECORDS = ("23461t2022", "44402t2016", "46407t2011", "52403t2020",
           "dart_44402_2018", "dart_21415_2018", "dart_21418_2018")
SEEDS = (42, 1729, 31415)
STREAMS = (71001, 71002, 71003)
GEOMETRIES = ("joint", "centered", "increment")
SCORE_KEYS = ("energy", "centered_energy", "increment_energy")
VARIANTS = ("original", "variance_refit")
LAGS = (1, 4, 16, 48, 96)
ARM_MAP = {("base", "low"): "A", ("raw", "low"): "C",
           ("base", "high"): "D", ("raw", "high"): "B"}
TOLERANCE = 1e-10


def require(ok, description):
    if not ok:
        raise ValueError(description)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(bool(rows), f"Empty table: {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Checks:
    def __init__(self):
        self.groups = {}

    def close(self, actual, expected, label, tolerance=TOLERANCE):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape, f"{label}: shape mismatch")
        require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{label}: nonfinite")
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
        group = self.groups.setdefault(label, {"comparisons": 0, "elements": 0, "max_abs_error": 0.})
        group["comparisons"] += 1
        group["elements"] += int(actual.size)
        group["max_abs_error"] = max(group["max_abs_error"], error)
        require(np.allclose(actual, expected, rtol=tolerance, atol=tolerance),
                f"{label}: tolerance {tolerance} exceeded, max absolute error {error}")


def transform(rows, geometry):
    rows = np.asarray(rows, dtype=np.float64)
    if geometry == "joint":
        return rows
    if geometry == "centered":
        return rows - rows.mean(axis=1, keepdims=True)
    return np.concatenate([(rows[:, lag:] - rows[:, :-lag]) / np.sqrt(192 - lag)
                           for lag in LAGS], axis=1)


def terms(draws, observations):
    """A per observation, B shared by observations; E=A-B (loss orientation)."""
    a = cdist(observations, draws).mean(axis=1)
    b = float(pdist(draws).mean() / 2)
    return a, b, a - b


def fit_means(rows, key_fields, value_fields):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in key_fields)
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in sorted(groups.items()):
        require(len(items) == 3, "Each fit must retain exactly three streams")
        row = dict(zip(key_fields, key))
        for field in value_fields:
            values = np.array([item[field] for item in items])
            row[field] = float(values.mean())
            row[field + "_stream_min"] = float(values.min())
            row[field + "_stream_max"] = float(values.max())
        result.append(row)
    return result


def seed_summaries(rows, key_fields):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in sorted(groups.items()):
        require({item["seed"] for item in items} == set(SEEDS), "Missing optimizer seed")
        row = dict(zip(key_fields, key))
        for field in ("delta_A", "delta_B", "delta_ES"):
            values = [item[field] for item in items]
            row[field + "_median"] = float(np.median(values))
            row[field + "_seed_min"] = float(min(values))
            row[field + "_seed_max"] = float(max(values))
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True, help="Original prigm-core-20260905 root")
    parser.add_argument("--dart-work", type=Path, required=True, help="Root containing four diagnostic bundles and stage")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths refused")
    args = parser.parse_args()
    require(digest(args.manifest) == MANIFEST_SHA, "Frozen evidence manifest hash differs")
    manifest = read_json(args.manifest)
    records = manifest["dart_extension"]["records"]
    index = manifest["dart_extension"]["input_index"]["files"]
    require(set(records) == set(RECORDS), "Expected exactly seven records")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "raw"
    raw.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    checks = Checks()
    receipts = {}

    def bind(logical, path, expected=None):
        path = path.absolute()
        expected = index[logical] if expected is None else expected
        require(path.is_file() and not path.is_symlink(), f"Missing/linked input: {path}")
        if logical not in receipts:
            actual = digest(path)
            require(actual == expected, f"Hash mismatch: {logical}")
            receipts[logical] = {"path": str(path), "expected_sha256": expected,
                                 "actual_sha256": actual, "bytes": path.stat().st_size}
        else:
            require(receipts[logical]["path"] == str(path)
                    and receipts[logical]["actual_sha256"] == expected, "Inconsistent input binding")
        return path

    bind("frozen-evidence", args.manifest, MANIFEST_SHA)
    data = {}
    for name in RECORDS:
        fresh = name.startswith("dart_")
        kind = "stage" if fresh else "core"
        root = args.dart_work / "dart-extension-stage-20260908" if fresh else args.core
        logical = f"{kind}/prepared/{name}.npz"
        path = bind(logical, root / "prepared" / (name + ".npz"))
        require(index[logical] == records[name]["prepared_sha256"], "Prepared binding mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            keys = ["train", "test", "starts_test"] + (["train_weights", "eval_weights"] if fresh else [])
            item = {key: arrays[key] for key in keys}
        for split in ("train", "test"):
            x = item[split]
            require(x.ndim == 2 and x.shape[1] == 192 and np.isfinite(x).all(), "Bad prepared windows")
            key = "train_weights" if split == "train" else "eval_weights"
            if not fresh:
                item[key] = np.full(len(x), 1 / len(x))
            weights = item[key]
            require(weights.shape == (len(x),) and np.isfinite(weights).all()
                    and np.all(weights >= 0), "Bad window weights")
            checks.close(weights.sum(), 1., "weight_sum")
            item[key] = weights / weights.sum()
        require(np.all(np.diff(item["starts_test"]) > 0), "Test window order differs")
        data[name] = item

    # Enumerate authenticated worker protocols, never select files from outcome-based globs.
    batches = {"core-batch-000": "diagnostics-core-pilot",
               "core-batch-001": "diagnostics-core-remaining",
               "stage-batch-000": "diagnostics-primary",
               "stage-batch-001": "diagnostics-optional"}
    jobs, seen = [], set()
    for batch, directory in batches.items():
        for worker in range(4):
            logical = f"{batch}/worker-{worker}"
            root = args.dart_work / directory / f"worker-{worker}"
            proto = read_json(bind(logical + "/protocol.json", root / "protocol.json"))
            completion = read_json(bind(logical + "/complete.json", root / "complete.json"))
            require(completion["protocol_sha256"] == index[logical + "/protocol.json"], "Worker protocol binding")
            require(proto["jobs"] == proto["all_jobs"][worker::4], "Worker ownership differs")
            for job in proto["jobs"]:
                name, condition, seed = job[:3]
                arm = ARM_MAP[(condition, job[3])] if batch.startswith("stage") else condition
                identity = (name, arm, seed)
                require(identity not in seen, "Duplicate fit")
                seen.add(identity)
                stem = "_".join(map(str, job[:-1])) + "-" + job[-1]
                metadata = read_json(bind(logical + "/" + stem + ".json", root / (stem + ".json")))
                archive = bind(logical + "/" + stem + ".npz", root / (stem + ".npz"))
                require(metadata["array_sha256"] == index[logical + "/" + stem + ".npz"], "Diagnostic array binding")
                require(metadata["protocol_sha256"] == index[logical + "/protocol.json"], "Diagnostic protocol binding")
                public = records[name]["models"][f"{arm}/{seed}/original"]
                require(metadata["dataset"] == name and metadata["seed"] == seed
                        and metadata["checkpoint"] == public["checkpoint"]
                        and metadata["checkpoint_sha256"] == public["checkpoint_sha256"]
                        and metadata["prepared_sha256"] == records[name]["prepared_sha256"], "Fit identity differs")
                require(metadata["checkpoint"] == ("final" if name.startswith("dart_") else "selected"), "Checkpoint rule")
                require(metadata["dimension"] == 192 and metadata["test_windows"] == len(data[name]["test"]), "Data dimensions")
                ref = metadata["references"]
                reference_path = bind(logical + "/" + ref["file"], root / ref["file"])
                require(ref["sha256"] == index[logical + "/" + ref["file"]], "Reference hash binding")
                for filename in (stem + ".json", stem + ".npz", ref["file"]):
                    require(completion["artifact_sha256"][filename] == index[logical + "/" + filename], "Completion artifact binding")
                jobs.append((name, arm, seed, archive, metadata, reference_path))
    require(seen == set(itertools.product(RECORDS, "ABCD", SEEDS)), "Incomplete 84-fit coverage")
    write_json(args.output / "protocol.json", {
        "started_utc": started, "manifest_sha256": MANIFEST_SHA,
        "script_sha256": digest(__file__), "arguments": {k: str(v) for k, v in vars(args).items()},
        "records": RECORDS, "arms": list("ABCD"), "training_seeds": SEEDS,
        "prior_streams": STREAMS, "variants": VARIANTS, "geometries": GEOMETRIES,
        "primary": "All 84 reset-minus-original comparisons",
        "secondary": "All 42 C-A and B-D comparisons at original observation variance",
        "atol": TOLERANCE, "rtol": TOLERANCE, "CPU_only": True,
        "new_fitting": False, "new_score_samples": False,
        "selection": "Complete coverage specified before this calculation; retrospective analysis",
        "reference_stream_pairing": "None with neural draws; compare stream means only",
        "checkpoint_scope": "Authenticated saved-array and metadata identities; no neural replay or checkpoint loading",
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "platform": platform.platform(), "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"]}})
    write_json(args.output / "input_receipts.json", receipts)
    print(f"Authenticated {len(receipts)} inputs; complete 84-fit coverage", flush=True)

    # Covariance calculation is a reduction of the existing training measure.
    covariance_rows, covariances = [], {}
    identity = np.eye(192)
    transforms = {geometry: transform(identity, geometry).T for geometry in GEOMETRIES}
    for name in RECORDS:
        x, w = data[name]["train"], data[name]["train_weights"]
        mean = w @ x
        covariance = ((x - mean) * w[:, None]).T @ (x - mean)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
        checks.close(vectors.T @ vectors, identity, "eigenvector_orthogonality")
        checks.close((vectors * eigenvalues) @ vectors.T, covariance, "spectral_reconstruction")
        fitted = max(float(eigenvalues[16:].mean()), 1e-4)
        checks.close(fitted, records[name]["references"]["low"]["ppca_fitted"]["noise_variance"], "fitted_variance", 1e-12)
        item = {"mean": mean, "training_covariance": covariance,
                "eigenvalues": eigenvalues, "eigenvectors": vectors}
        for label, v in (("low", 1e-4), ("high", .0025), ("fitted", fitted)):
            loadings = vectors[:, :16] * np.sqrt(np.maximum(eigenvalues[:16] - v, 0))
            item[label + "_covariance"] = loadings @ loadings.T + v * identity
        posterior_logvar = np.log(fitted / np.maximum(eigenvalues[:16], fitted))
        for geometry, matrix in transforms.items():
            power = eigenvalues * np.square(matrix @ vectors).sum(axis=0)
            trace = float(np.trace(matrix @ covariance @ matrix.T))
            checks.close(power.sum(), trace, "transformed_trace")
            row = {"record": name, "geometry": geometry, "rank": 16,
                   "covariance_trace": trace, "leading_trace": float(power[:16].sum()),
                   "leading_fraction": float(power[:16].sum() / trace), "fitted_variance": fitted,
                   "min_fitted_posterior_logvar": float(posterior_logvar.min()),
                   "fitted_posterior_directions_below_minus12": int((posterior_logvar < -12).sum())}
            for label, v in (("low", 1e-4), ("high", .0025), ("fitted", fitted)):
                model_trace = float(np.trace(matrix @ item[label + "_covariance"] @ matrix.T))
                exact = float(v * np.square(matrix).sum() + np.dot(
                    np.maximum(eigenvalues[:16] - v, 0), np.square(matrix @ vectors[:, :16]).sum(axis=0)))
                checks.close(model_trace, exact, "PPCA_transformed_trace")
                row[label + "_PPCA_trace"] = model_trace
            covariance_rows.append(row)
        np.savez_compressed(raw / (name + "_covariance.npz"), **item)
        covariances[name] = item
    write_csv(args.output / "covariance_structure.csv", covariance_rows)

    rows, reference_rows, reference_cache = [], [], {}
    for ordinal, (name, arm, seed, path, metadata, reference_path) in enumerate(jobs, 1):
        observations, weights = data[name]["test"], data[name]["eval_weights"]
        n = len(observations)
        variance_label = "low" if arm in "AC" else "high"
        with np.load(path, allow_pickle=False) as arrays:
            checks.close(arrays["diagnostic_test_weights"], weights, "declared_test_weights")
            require(np.array_equal(arrays["original__prior_means"], arrays["variance_refit__prior_means"]), "Reset changed means")
            a_windows = np.empty((2, 3, 3, n))
            b_terms = np.empty((2, 3, 3))
            e_windows = np.empty_like(a_windows)
            for variant_index, variant in enumerate(VARIANTS):
                prefix = variant + "__"
                variance = float(arrays[prefix + "observation_variance"])
                require(variance > 0 and np.isfinite(variance), "Invalid observation variance")
                public = records[name]["models"][f"{arm}/{seed}/{variant}"]
                checks.close(variance, public["observation_variance"], "archived_variance")
                require(int(arrays[prefix + "samples"]) == 1024
                        and np.array_equal(arrays[prefix + "seeds"], STREAMS), "Prior stream specification")
                means, draws = arrays[prefix + "prior_means"], arrays[prefix + "prior_samples"]
                require(means.shape == draws.shape == (3, 1024, 192)
                        and np.isfinite(means).all() and np.isfinite(draws).all(), "Invalid prior arrays")
                for stream_index, stream in enumerate(STREAMS):
                    rng = np.random.default_rng(stream)
                    rng.standard_normal((1024, 16)).astype(np.float32)
                    noise = rng.standard_normal((1024, 192))
                    checks.close(draws[stream_index], means[stream_index] + np.sqrt(variance) * noise,
                                 "shared_seeded_observation_noise", 1e-12)
                    for geometry_index, (geometry, saved_key) in enumerate(zip(GEOMETRIES, SCORE_KEYS)):
                        a, b, energy = terms(transform(draws[stream_index], geometry), transform(observations, geometry))
                        a_windows[variant_index, stream_index, geometry_index] = a
                        b_terms[variant_index, stream_index, geometry_index] = b
                        e_windows[variant_index, stream_index, geometry_index] = energy
                        checks.close(energy, arrays[prefix + saved_key + "_per_window"][stream_index], "neural_per_window_score")
                        a_mean, e_mean = float(weights @ a), float(weights @ energy)
                        checks.close(e_mean, arrays[prefix + saved_key + "_streams"][stream_index], "neural_weighted_stream_score")
                        checks.close(float(energy.mean()), arrays[prefix + saved_key + "_uniform_streams"][stream_index], "neural_uniform_stream_score")
                        checks.close(a_mean - b, e_mean, "weighted_term_accounting")
                        checks.close(e_mean, public["scores"][geometry]["weighted"]["values"][stream_index], "frozen_public_neural_stream")
                        rows.append({"record": name, "arm": arm, "seed": seed,
                                     "checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                                     "variance_group": variance_label, "variant": variant,
                                     "variance": variance, "stream": stream, "geometry": geometry,
                                     "A": a_mean, "B": b, "ES": e_mean})
            np.savez_compressed(raw / f"{name}_{arm}_{seed}_terms.npz", A_per_window=a_windows,
                                B=b_terms, ES_per_window=e_windows, test_weights=weights,
                                starts_test=data[name]["starts_test"], variants=VARIANTS,
                                streams=STREAMS, geometries=GEOMETRIES)
        ref_key = (name, variance_label)
        ref_hash = metadata["references"]["sha256"]
        if ref_key in reference_cache:
            require(reference_cache[ref_key] == ref_hash, "Duplicate reference is not identical")
        else:
            reference_cache[ref_key] = ref_hash
            with np.load(reference_path, allow_pickle=False) as arrays:
                checks.close(arrays["test_weights"], weights, "reference_test_weights")
                checks.close(arrays["ppca_matched__mean"], covariances[name]["mean"], "matched_PPCA_mean")
                checks.close(arrays["ppca_matched__covariance"], covariances[name][variance_label + "_covariance"], "matched_PPCA_covariance")
                draws = arrays["ppca_matched__prior_samples"]
                require(draws.shape == (3, 1024, 192) and np.isfinite(draws).all(), "Reference draw dimensions")
                ref_a, ref_b, ref_e = np.empty((3, 3, n)), np.empty((3, 3)), np.empty((3, 3, n))
                for stream_index, stream in enumerate((74001, 74002, 74003)):
                    for geometry_index, geometry in enumerate(GEOMETRIES):
                        a, b, energy = terms(transform(draws[stream_index], geometry), transform(observations, geometry))
                        ref_a[stream_index, geometry_index], ref_b[stream_index, geometry_index], ref_e[stream_index, geometry_index] = a, b, energy
                        checks.close(energy, arrays[f"ppca_matched__{geometry}_per_window"][stream_index], "reference_per_window_score")
                        mean_score = float(weights @ energy)
                        checks.close(mean_score, arrays[f"ppca_matched__{geometry}_streams"][stream_index], "reference_weighted_stream_score")
                        checks.close(mean_score, records[name]["references"][variance_label]["ppca_matched"]["scores"][geometry]["weighted"]["values"][stream_index], "frozen_public_reference_stream")
                        reference_rows.append({"record": name, "variance_group": variance_label,
                                               "stream": stream, "geometry": geometry,
                                               "A": float(weights @ a), "B": b, "ES": mean_score})
                np.savez_compressed(raw / f"{name}_{variance_label}_reference_terms.npz",
                                    A_per_window=ref_a, B=ref_b, ES_per_window=ref_e,
                                    test_weights=weights, starts_test=data[name]["starts_test"])
        print(f"Completed fit {ordinal}/84: {name}/{arm}/{seed}", flush=True)

    require(len(rows) == 1512 and len(reference_rows) == 126, "Incomplete stream tables")
    write_csv(args.output / "energy_terms_streams.csv", rows)
    write_csv(args.output / "PPCA_reference_streams.csv", reference_rows)
    fit_rows = fit_means(rows, ("record", "arm", "seed", "variance_group", "variant", "geometry"), ("A", "B", "ES"))
    write_csv(args.output / "energy_terms_fits.csv", fit_rows)
    lookup = {(r["record"], r["arm"], r["seed"], r["variant"], r["stream"], r["geometry"]): r for r in rows}
    reset, skip = [], []
    for name, arm, seed, stream, geometry in itertools.product(RECORDS, "ABCD", SEEDS, STREAMS, GEOMETRIES):
        before = lookup[(name, arm, seed, "original", stream, geometry)]
        after = lookup[(name, arm, seed, "variance_refit", stream, geometry)]
        row = {"record": name, "arm": arm, "seed": seed, "variance_group": before["variance_group"],
               "stream": stream, "geometry": geometry, "variance_ratio": after["variance"] / before["variance"]}
        for term in ("A", "B", "ES"):
            row["delta_" + term] = after[term] - before[term]
        checks.close(row["delta_A"] - row["delta_B"], row["delta_ES"], "paired_reset_accounting")
        reset.append(row)
    for name, pair, seed, stream, geometry in itertools.product(RECORDS, ("C-A", "B-D"), SEEDS, STREAMS, GEOMETRIES):
        left, right = (lookup[(name, a, seed, "original", stream, geometry)] for a in pair.split("-"))
        checks.close(left["variance"], right["variance"], "matched_skip_variance")
        row = {"record": name, "pair": pair, "seed": seed, "variance_group": left["variance_group"], "stream": stream, "geometry": geometry}
        for term in ("A", "B", "ES"):
            row["delta_" + term] = left[term] - right[term]
        checks.close(row["delta_A"] - row["delta_B"], row["delta_ES"], "paired_skip_accounting")
        skip.append(row)
    require(len(reset) == 756 and len(skip) == 378, "Incomplete contrast streams")
    for label, items, arm_key in (("reset", reset, "arm"), ("skip", skip, "pair")):
        write_csv(args.output / (label + "_contrasts_streams.csv"), items)
        keys = ("record", arm_key, "seed", "variance_group", "geometry")
        means = fit_means(items, keys, ("delta_A", "delta_B", "delta_ES"))
        for row in means:
            checks.close(row["delta_A"] - row["delta_B"], row["delta_ES"], "fit_mean_contrast_accounting")
        write_csv(args.output / (label + "_contrasts_fits.csv"), means)
        write_csv(args.output / (label + "_contrasts_seed_summary.csv"), seed_summaries(means, ("record", arm_key, "variance_group", "geometry")))

    ref_means = fit_means(reference_rows, ("record", "variance_group", "geometry"), ("A", "B", "ES"))
    write_csv(args.output / "PPCA_reference_means.csv", ref_means)
    reference_lookup = {(r["record"], r["variance_group"], r["geometry"]): r for r in ref_means}
    comparisons = []
    for row in fit_rows:
        if row["variant"] != "original":
            continue
        ref = reference_lookup[(row["record"], row["variance_group"], row["geometry"])]
        comparisons.append({k: row[k] for k in ("record", "arm", "seed", "variance_group", "geometry")} |
                           {"neural_ES": row["ES"], "PPCA_ES": ref["ES"],
                            "neural_minus_PPCA": row["ES"] - ref["ES"], "stream_pairing": "none"})
    require(len(comparisons) == 252, "Incomplete PPCA comparisons")
    write_csv(args.output / "PPCA_matched_comparisons.csv", comparisons)

    # Rehash each consumed input after computation; preserve raw per-window output remotely.
    for receipt in receipts.values():
        require(digest(receipt["path"]) == receipt["actual_sha256"], "Input changed during calculation")
    write_json(args.output / "verification.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "success": True,
        "unique_fits": len(seen), "neural_stream_geometry_variant_rows": len(rows),
        "reset_stream_contrasts": len(reset), "skip_stream_contrasts": len(skip),
        "covariance_rows": len(covariance_rows), "unique_reference_models": len(reference_cache),
        "matched_PPCA_comparisons": len(comparisons), "input_files": len(receipts),
        "inputs_rehashed_after_completion": True, "checks": checks.groups,
        "limits": "Saved-array arithmetic and frozen provenance; no neural execution, posterior diagnosis, population inference, or causal attribution."})
    artifacts = {str(p.relative_to(args.output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "output_manifest.json", artifacts)
    print(json.dumps({"success": True, "output": str(args.output), "input_files": len(receipts),
                      "neural_score_max_abs_error": checks.groups["neural_per_window_score"]["max_abs_error"]}), flush=True)


if __name__ == "__main__":
    main()
