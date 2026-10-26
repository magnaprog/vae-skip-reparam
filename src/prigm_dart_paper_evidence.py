"""Private-array verification and closed public DART evidence, independent of builders.

No neural replay is performed here. The parent must already have validated the
original core evidence. Fresh completion verification requires real h5py; there
is deliberately no signature-only fallback. Public numbers stay standardized.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
CORE_SHA256 = "37cc7445600eee541ee50aa580f6fdd6c0f37cf420074dbecc635c8d202946f5"
EVALUATOR_REVIEWED_SHA256 = "17282a2b7a596a5ba399f50b90a8439fd6c12c7191f8b9624c5a76d9ce9e0f64"
STAGE_SOURCES = {
    "prigm_metrics.py": "58ca29a18223dd21231380c81358e4cf9c2c2980246803b0aa8bb485e81c7ea5",
    "prigm_v2_data.py": "d89dd3e14068992d1fb148a57e0fddc498bc4d357ff3bfc1bb29dbe96bc13465",
    "prigm_v2_experiment.py": "33a4e1bc8bac81976d1558261c230e573a13cd3559e594fcd18f3e8abd2c3bd0",
    "prigm_v2_metrics.py": "6e87ca76b7e00a27b04ca2863399916bd82141d3cd95342d88b74d9e6e1704a8",
    "prigm_v2_model.py": "e4ddd0e0381a21fdf50212a157a6c225b5cad83a231e90dd24112ac4875d6125",
}
CORE_SOURCES = {
    "prigm_data.py": "62e954ea8dcd559060e7ad7b56b2b766d5a2b213119e496b330462c9ef367712",
    "prigm_experiment.py": "ac28e26c522b791869beda557ce3b92aa4fb5f1f2156d85926bd8f8ea4dc7cce",
    "prigm_metrics.py": STAGE_SOURCES["prigm_metrics.py"],
    "prigm_model.py": "440ea791759bac02424093715af302e7c9c9c126256da93665bd6f510bea53e8",
}
# Reviewed current derivation bytes, independent of the literal producer pins above.
REVIEWED_SOURCES = {
    "prigm_data.py": "62e954ea8dcd559060e7ad7b56b2b766d5a2b213119e496b330462c9ef367712",
    "prigm_experiment.py": "5e736920f0cf602cbe02fdfb1e763c7c21756191ee3ca097581c686221003acf",
    "prigm_metrics.py": "2f11f09b5c70ee2be4c563db9fadfde181303c9f4f31b8c7d29f2a541c0d70e9",
    "prigm_model.py": "c6e0a19862b00565fa5b6123c35efbb9f2cad8b5eaaa4b3429abb74509ac660b",
    "prigm_v2_data.py": "bea46030c7d517326423a5ce60b1f2e9cef70e12d0a85f46e0f01921134e84a5",
    "prigm_v2_experiment.py": "6f56655920529ce1dfd03b3ca6d78c42f79384d6b9e9d9a575e4498302065b09",
    "prigm_v2_metrics.py": "6e87ca76b7e00a27b04ca2863399916bd82141d3cd95342d88b74d9e6e1704a8",
    "prigm_v2_model.py": "4a63857d333df0b49708e5354ec2ceb34958455b78ebc4b3eb560cd78408fd1a",
    "eval_gaussian_final_checkpoints.py": "17282a2b7a596a5ba399f50b90a8439fd6c12c7191f8b9624c5a76d9ce9e0f64",
}
DART_DIAGNOSTICS = {
    "prior_samples": 1024, "prior_seeds": [71001, 71002, 71003],
    "variance_samples": 32, "variance_seeds": [72001, 72002, 72003],
    "reference_seeds": [74001, 74002, 74003],
    "iw_samples": 256, "iw_seeds": [42001, 42003, 42004],
    "density_coordinates": "observation",
}
OLD = ("23461t2022", "44402t2016", "46407t2011", "52403t2020")
PRIMARY = "dart_44402_2018"
OPTIONAL = ("dart_21415_2018", "dart_21418_2018")
SEEDS = (42, 1729, 31415)
ARMS = ("A", "B", "C", "D")
ARM_MAP = {("base", "low"): "A", ("raw", "high"): "B",
           ("raw", "low"): "C", ("base", "high"): "D"}
SCORES = ("joint", "centered", "increment")
COMPONENTS = ("log_score", "ess", "log_weights", "sample_log_likelihood",
              "sample_logq_minus_logprior", "posterior_expected_nll",
              "posterior_expected_mse", "posterior_mean_mse", "kl", "kl_by_latent",
              "negative_elbo_analytic_kl", "sample_negative_elbo", "jensen_gap")
REDUCTIONS = ("posterior_mean_mse", "posterior_expected_mse", "nll_per_coordinate",
              "kl_per_coordinate", "negative_elbo_per_coordinate", "negative_iw_per_coordinate",
              "jensen_gap_per_coordinate", "mean_ess")
METHODS = {
    "window_dimension": 192, "latent_dimension": 16, "cadence_seconds": 900,
    "diagnostics": DART_DIAGNOSTICS,
    "selected_secondary_streams": {"prior_samples": 1024, "prior_seeds": [62001, 62002, 62003],
                                   "iw_samples": 256, "iw_seeds": [65001, 65002, 65003],
                                   "density_coordinates": "observation"},
    "coordinates": "standardized_observation; no conversion to metres in scores or MSE",
    "score_units": {"joint": "standardized_joint_Euclidean", "centered": "standardized_centered_Euclidean",
                    "increment": "standardized_lag_normalized_Euclidean_1_4_16_48_96"},
    "endpoints": "old: original selected; fresh: final 10000 primary, validation-selected secondary; selected-at-10000 aliases archived final evaluation, not the new diagnostic MC streams",
    "variance_refit": "training-window posterior-expected MSE, 3x32 draws; pooled scalar; fixed learned functions",
    "contrasts": "C-A and B-D paired optimizer seeds; matched ORIGINAL variance only; refits are model-specific",
    "variation": "median and full range of three optimizer-seed stream means; MC streams reported separately",
    "references": "rank16 PPCA at original matched noise and training-fitted floor1e-4; diagonal max(training variance, original noise); three shared streams, not per-model replicates or refit-matched references",
    "blocks": "14/28-day start-time blocks anchored at January 1; descriptive conditional window-weighted means, no population CIs",
    "limits": "Overlapping windows, UTC bins, MC streams and optimizer seeds are not independent environmental samples. Low-ESS IW scores are diagnostics, not true likelihood rankings. Refit is not marginal MLE, instrument noise or guaranteed calibration.",
    "metadata_limits": "Selected annual case studies, not deployment effects or a random survey. Catalogue coverage does not establish unchanged hardware or stationarity. HEIGHT is pressure-equivalent recorded water-column height, not corrected sea surface or tsunami labels.",
}
VERIFICATION = {
    "saved_array_identities": True, "input_files_rehashed": True,
    "fresh_hdf5_contents_verified": True, "cpu_neural_replay": False,
    "old_core_truth": "parent-validated; selected checkpoint/data/artifacts rebound here",
    "neural_limit": "Reviewed producer attests checkpoint replay. CPU checks saved replay components and unchanged saved means/proposals, not neural forward passes or training posterior draws.",
    "qualification_limit": "Fresh prepared supports and frozen metadata checked by runner; raw acquisition and hardware history not independently reacquired here.",
}


def _require(ok, label):
    if not ok:
        raise ValueError(label)


def _keys(value, names, label):
    _require(type(value) is dict and set(value) == set(names), f"Invalid {label} fields")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(actual, expected, label):
    _require(_json(actual) == _json(expected), f"Inconsistent {label}")


def _number(value, positive=False):
    _require(type(value) in (int, float) and np.isfinite(value)
             and (not positive or value > 0), "Expected finite numeric scalar")
    return value


def _digest(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def _path(path, directory=False):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input")
    _require(path.is_dir() if directory else path.is_file(), "Expected regular input path")
    return path


def _sha(path):
    with _path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(_path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    _json(value)  # Includes overflowing exponent literals, not just NaN tokens.
    return value


def _inventory(path, files):
    path = _path(path, True)
    expected = set(files)
    directories = {str(parent) for name in files for parent in Path(name).parents if str(parent) != "."}
    entries = list(path.rglob("*"))
    _require({p.relative_to(path).as_posix() for p in entries} == expected | directories,
             "Missing or unexpected artifact inventory")
    for name in expected:
        _path(path / name)
    for name in directories:
        _path(path / name, True)


def _array(value, shape, label, integer=False):
    value = np.asarray(value)
    _require(value.shape == tuple(shape) and value.dtype.kind in ("iu" if integer else "fiu")
             and np.isfinite(value).all(), f"Invalid finite array shape/dtype: {label}")
    return value


def _close(actual, expected, label, rtol=1e-10, atol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape and actual.dtype.kind in "fiu" and expected.dtype.kind in "fiu"
             and np.isfinite(actual).all() and np.isfinite(expected).all()
             and np.allclose(actual, expected, rtol=rtol, atol=atol),
             f"Array identity differs: {label}")


def _load(path, shapes=None):
    _path(path)
    with np.load(path, allow_pickle=False) as store:
        _require(len(store.files) == len(set(store.files)), "Duplicate NPZ member")
        result = {key: store[key] for key in store.files}
    if shapes is not None:
        _keys(result, shapes, "NPZ")
        for key, shape in shapes.items():
            _array(result[key], shape, key)
    return result


def _imports():
    # Check actual reviewed bytes before any project import, including cached modules.
    for name, expected in REVIEWED_SOURCES.items():
        _require(_sha(ROOT / "src" / name) == expected, "Unreviewed current source")
        module = sys.modules.get(name[:-3])
        if module is not None:
            _require(Path(module.__file__).absolute() == ROOT / "src" / name,
                     "Project module imported from another source tree")
    return tuple(importlib.import_module(name) for name in
                 ("prigm_v2_metrics", "prigm_v2_experiment", "prigm_experiment",
                  "eval_gaussian_final_checkpoints"))


def _weights(data, split, fresh):
    n = len(data[split])
    _require(n > 0, "Empty partition")
    if not fresh:
        return np.full(n, 1 / n)
    key = "eval_weights" if split == "test" else split + "_weights"
    w = _array(data[key], (n,), key)
    starts = _array(data["starts_" + split], (n,), "start times", integer=True)
    bins = ((starts % 86400) // 14400).astype(int)
    counts = np.bincount(bins, minlength=6)
    _require(np.all(counts > 0) and np.all(starts % 14400 == 0), "Missing UTC bin/grid")
    _close(w, 1 / (6 * counts[bins]), "UTC target weights", rtol=0, atol=1e-12)
    _close(w.sum(), 1., "normalized target", rtol=0, atol=1e-12)
    return w / w.sum()


def _streams(values):
    x = _array(values, (3,), "three streams")
    return {"values": x.tolist(), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def _seed_summary(values):
    x = _array(values, (3,), "three optimizer seeds")
    return {"values": x.tolist(), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def _blocks(starts, year):
    origin = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    return {str(days): (starts - origin) // (days * 86400) for days in (14, 28)}


def _score_summary(values, weights, blocks):
    result = {"weighted": _streams(values @ weights), "uniform": _streams(values.mean(axis=1)), "blocks": {}}
    for days, labels in blocks.items():
        rows = []
        for block in np.unique(labels):
            mask = labels == block
            mass = weights[mask].sum()
            rows.append({"block": int(block), "windows": int(mask.sum()), "mass": float(mass),
                         "streams": _streams(values[:, mask] @ (weights[mask] / mass))})
        result["blocks"][days] = rows
    return result


def _component_shapes(n):
    return {key: (3, n, 256) if key in COMPONENTS[2:5] else
            (3, n, 16) if key == "kl_by_latent" else (3, n) for key in COMPONENTS}


def _components(arrays, n, variance, weights, metrics):
    _number(variance, True)
    _keys(arrays, COMPONENTS, "probability component")
    for key, shape in _component_shapes(n).items():
        a = _array(arrays[key], shape, key)
        _require(a.dtype == np.dtype("float64"), "Probability components must be float64")
    for key in ("posterior_mean_mse", "posterior_expected_mse", "kl", "kl_by_latent"):
        _require(np.all(arrays[key] >= 0), "Negative MSE/KL")
    for key in ("posterior_mean_mse", "kl", "kl_by_latent"):
        _require(all(np.array_equal(arrays[key][0], row) for row in arrays[key][1:]),
                 "Analytic component differs across MC streams")
    logw = arrays["log_weights"]
    _close(logw, arrays["sample_log_likelihood"] - arrays["sample_logq_minus_logprior"], "log weights")
    nll = -arrays["sample_log_likelihood"].mean(axis=2)
    _close(nll, 96 * (np.log(2 * np.pi * variance) + arrays["posterior_expected_mse"] / variance),
           "Gaussian likelihood from expected MSE")
    checks = {"posterior_expected_nll": nll, "kl": arrays["kl_by_latent"].sum(axis=2),
              "negative_elbo_analytic_kl": nll + arrays["kl"],
              "sample_negative_elbo": -logw.mean(axis=2),
              "jensen_gap": arrays["log_score"] - logw.mean(axis=2)}
    _require(np.all(arrays["jensen_gap"] >= -1e-8), "Negative Jensen gap")
    for key, value in checks.items():
        _close(arrays[key], value, key)
    for i in range(3):
        stats = metrics.importance_statistics(logw[i])
        for key in ("log_score", "ess"):
            _close(arrays[key][i], stats[key], key)
    values = {"posterior_mean_mse": arrays["posterior_mean_mse"],
              "posterior_expected_mse": arrays["posterior_expected_mse"],
              "nll_per_coordinate": nll / 192, "kl_per_coordinate": arrays["kl"] / 192,
              "negative_elbo_per_coordinate": arrays["negative_elbo_analytic_kl"] / 192,
              "negative_iw_per_coordinate": -arrays["log_score"] / 192,
              "jensen_gap_per_coordinate": arrays["jensen_gap"] / 192, "mean_ess": arrays["ess"]}
    return {"weighted": {k: _streams(v @ weights) for k, v in values.items()},
            "uniform": {k: _streams(v.mean(axis=1)) for k, v in values.items()},
            "median_ess": float(np.median(arrays["ess"]))}


def _diagnostic_shapes(n, training, fresh):
    shapes = {"diagnostic_test_weights": (n,), "variance_refit_training_weights": (training,),
              "variance_refit_training_mse": (3, training), "variance_refit_stream_variances": (3,),
              "variance_refit_observation_variance": ()}
    for prefix in ("original__", "variance_refit__"):
        shapes.update({prefix + k: s for k, s in _component_shapes(n).items()})
        shapes.update({prefix + "observation_variance": (), prefix + "samples": (), prefix + "seeds": (3,),
                       prefix + "prior_means": (3, 1024, 192), prefix + "prior_samples": (3, 1024, 192)})
        for key in ("energy", "centered_energy", "increment_energy"):
            shapes.update({prefix + key + "_per_window": (3, n), prefix + key + "_streams": (3,),
                           prefix + key + "_uniform_streams": (3,)})
    shapes.update({("stage_replay__" if fresh else "") + k: s for k, s in _component_shapes(n).items()})
    return shapes


def _variance_fit(arrays, weights):
    mse = _array(arrays["variance_refit_training_mse"], (3, len(weights)), "training MSE")
    _require(np.all(mse >= 0), "Negative training posterior MSE")
    _close(arrays["variance_refit_training_weights"], weights, "training measure")
    estimates = mse @ weights
    _require(np.all(estimates > 0), "Nonpositive stream variance")
    pooled = float(estimates.mean())
    _close(arrays["variance_refit_stream_variances"], estimates, "training variance streams")
    _close(arrays["variance_refit_observation_variance"], pooled, "pooled training variance")
    return {"streams": _streams(estimates), "pooled": pooled}


def _observation_draw(means, variance, seed):
    """Reproduce producer RNG order; scientific array shapes are checked by the caller."""
    rng = np.random.default_rng(seed)
    rng.standard_normal((len(means), 16))
    return means + np.sqrt(variance) * rng.standard_normal(means.shape)


def _fixed_functions(arrays):
    for key in ("prior_means", "posterior_mean_mse", "posterior_expected_mse", "kl", "kl_by_latent", "sample_logq_minus_logprior"):
        _require(np.array_equal(arrays["original__" + key], arrays["variance_refit__" + key]),
                 "Saved learned means/posterior identity changed during variance refit")


def _verify_arrays(arrays, data, variance, fresh, blocks, metrics):
    n = len(data["test"])
    shapes = _diagnostic_shapes(n, len(data["train"]), fresh)
    _keys(arrays, shapes, "diagnostic arrays")
    for key, shape in shapes.items():
        value = _array(arrays[key], shape, key)
        _require(value.dtype.kind in "iu" if key.endswith(("__samples", "__seeds")) else value.dtype == np.dtype("float64"),
                 "Diagnostic numeric storage must match the reviewed producer dtype")
    w = _weights(data, "test", fresh)
    _close(arrays["diagnostic_test_weights"], w, "test measure")
    refit = _variance_fit(arrays, _weights(data, "train", fresh))
    _fixed_functions(arrays)
    functions = (metrics.energy_score, metrics.centered_energy_score, metrics.increment_energy_score)
    result = {}
    for label, v in (("original", variance), ("variance_refit", refit["pooled"])):
        prefix = label + "__"
        _close(arrays[prefix + "observation_variance"], v, "explicit observation variance")
        _require(arrays[prefix + "samples"].dtype.kind in "iu" and arrays[prefix + "samples"].item() == 1024,
                 "Incorrect prior sample count")
        _array(arrays[prefix + "seeds"], (3,), "prior seeds", True)
        _require(np.array_equal(arrays[prefix + "seeds"], DART_DIAGNOSTICS["prior_seeds"]), "Prior seeds differ")
        for i, seed in enumerate(DART_DIAGNOSTICS["prior_seeds"]):
            _close(arrays[prefix + "prior_samples"][i], _observation_draw(arrays[prefix + "prior_means"][i], v, seed),
                   "seeded observation noise", rtol=1e-12, atol=1e-12)
        scores = {}
        for name, saved, function in zip(SCORES, ("energy", "centered_energy", "increment_energy"), functions):
            values = np.stack([function(draws, data["test"]) for draws in arrays[prefix + "prior_samples"]])
            _close(arrays[prefix + saved + "_per_window"], values, name + " energy")
            _close(arrays[prefix + saved + "_streams"], values @ w, "weighted energy streams")
            _close(arrays[prefix + saved + "_uniform_streams"], values.mean(axis=1), "uniform energy streams")
            scores[name] = _score_summary(values, w, blocks)
        components = _components({k: arrays[prefix + k] for k in COMPONENTS}, n, v, w, metrics)
        result[label] = {"observation_variance": float(v), "variance_fit": deepcopy(refit),
                         "scores": scores, "components": components}
    # Same posterior draws imply the entire likelihood array transforms, not just its mean.
    old_ll = arrays["original__sample_log_likelihood"]
    squared = -2 * variance * (old_ll + 96 * np.log(2 * np.pi * variance))
    _require(np.all(squared >= -1e-8), "Negative likelihood-implied squared error")
    v = refit["pooled"]
    _close(arrays["variance_refit__sample_log_likelihood"], -squared / (2 * v) - 96 * np.log(2 * np.pi * v),
           "same-draw variance-refit likelihood")
    return result


def _reference_shapes(n):
    result = {"test_weights": (n,)}
    for name in ("ppca_matched", "ppca_fitted", "diagonal_matched"):
        p = name + "__"
        result.update({p + "mean": (192,), p + "covariance": (192, 192),
                       p + "prior_samples": (3, 1024, 192), p + "log_density": (n,)})
        for score in SCORES:
            result.update({p + score + "_per_window": (3, n), p + score + "_streams": (3,),
                           p + score + "_uniform_streams": (3,)})
    return result


def _reference_models(data, weights, variance, metrics):
    mean, cov = metrics.weighted_moments(data["train"], weights)
    fitted = max(float(np.linalg.eigvalsh(cov)[:-16].mean()), 1e-4)
    return fitted, {"ppca_matched": metrics.fixed_noise_ppca(mean, cov, variance, rank=16),
                    "ppca_fitted": metrics.fixed_noise_ppca(mean, cov, fitted, rank=16),
                    "diagonal_matched": metrics.noise_matched_diagonal(data["train"], weights, variance)}


def _references(arrays, data, variance, fresh, blocks, metrics):
    _require(all(value.dtype == np.dtype("float64") for value in arrays.values()), "Reference arrays must be float64")
    w = _weights(data, "test", fresh)
    _close(arrays["test_weights"], w, "reference test weights")
    fitted, references = _reference_models(data, _weights(data, "train", fresh), variance, metrics)
    output = {}
    for name, reference in references.items():
        p = name + "__"
        for key in ("mean", "covariance"):
            _close(arrays[p + key], reference[key], "training-fitted reference " + key)
        root = np.linalg.cholesky(reference["covariance"])
        draws = np.stack([np.random.default_rng(seed).standard_normal((1024, 192)) @ root.T + reference["mean"]
                          for seed in DART_DIAGNOSTICS["reference_seeds"]])
        _close(arrays[p + "prior_samples"], draws, "direct Cholesky reference draws")
        density = metrics.gaussian_log_prob(data["test"], reference["mean"], reference["covariance"])
        _close(arrays[p + "log_density"], density, "exact reference density")
        scores = {}
        for score, function in zip(SCORES, (metrics.energy_score, metrics.centered_energy_score,
                                            metrics.increment_energy_score)):
            values = np.stack([function(stream, data["test"]) for stream in draws])
            for suffix, expected in (("per_window", values), ("streams", values @ w),
                                     ("uniform_streams", values.mean(axis=1))):
                _close(arrays[p + score + "_" + suffix], expected, "reference " + score + " " + suffix)
            scores[score] = _score_summary(values, w, blocks)
        output[name] = {"noise_variance": fitted if name == "ppca_fitted" else float(variance),
                        "scores": scores, "weighted_log_density_per_coordinate": float(w @ density / 192),
                        "uniform_log_density_per_coordinate": float(density.mean() / 192)}
    return output


def _admission(value, stage_sha, fresh_names):
    names = set(fresh_names)
    _require(names in ({PRIMARY}, {PRIMARY, *OPTIONAL}), "Publication needs exactly 12 or 36 fresh fits")
    if value is None:
        _require(names == {PRIMARY}, "Optional completed blocks require admission")
        return {"optional_admitted": False, "published_fits": 12, "admission_sha256": None}
    _keys(value, ("admitted_records", "allowance_per_optional_block_seconds", "basis", "created_utc",
                  "fits_per_record", "optional_neural_results_exist", "purpose", "remaining_seconds_before_reserve",
                  "schema_version", "stage_manifest_sha256", "total_new_fit_ceiling", "verified_counts"), "admission")
    for key, expected in {"admitted_records": list(OPTIONAL), "fits_per_record": 12,
                          "optional_neural_results_exist": False, "purpose": "dart_optional_execution_admission",
                          "schema_version": 1, "stage_manifest_sha256": stage_sha, "total_new_fit_ceiling": 36,
                          "verified_counts": {"diagnostics-core-pilot": 12, "diagnostics-core-remaining": 36,
                                              "diagnostics-primary": 12}}.items():
        _same(value[key], expected, "admission " + key)
    allowance = _number(value["allowance_per_optional_block_seconds"], True)
    _require(2 * allowance <= _number(value["remaining_seconds_before_reserve"], True), "Optional resource budget exceeded")
    _require(type(value["basis"]) is str and value["basis"], "Missing admission basis")
    stamp = datetime.fromisoformat(value["created_utc"])
    _require(stamp.utcoffset() is not None, "Admission requires aware UTC time")
    return {"optional_admitted": True, "published_fits": 12 * len(names), "admission_sha256": None}


def _metadata_coverage(record, dataset):
    """Check frozen recorded coverage, not live NOAA metadata or hardware history."""
    qualification, deployment = record["qualification"], record["deployment"]
    _require(qualification["eligible"] is True and .8 <= _number(qualification["coverage"]) <= 1,
             "Unqualified annual coverage")
    primary = dataset == PRIMARY
    if not primary:
        _same(qualification["calendar_days"], 365, "optional whole-year qualification")
    identifier = deployment["catalog_entry"] if primary else deployment["catalogue_deployment"]["id"]
    match = re.fullmatch(re.escape(dataset.split("_")[1]) + r"_([0-9]{8})to([0-9]{8})", identifier)
    _require(match is not None, "Deployment/station identity mismatch")
    start, end = (datetime.strptime(value, "%Y%m%d").date() for value in match.groups())
    lower, upper = datetime(2018, 1, 1).date(), datetime(2019, 1, 1).date()
    _require(start <= lower and end >= upper, "Annual interval outside recorded deployment")
    if primary:
        processed = deployment["tcon_exact_deployment_fields"]
        _same(processed["name"], identifier, "processed deployment name")
        start = datetime.fromisoformat(processed["data_start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(processed["data_end"].replace("Z", "+00:00"))
        _require(start <= datetime(2018, 1, 1, tzinfo=timezone.utc)
                 and end >= datetime(2019, 1, 1, tzinfo=timezone.utc), "Annual interval outside recorded processed coverage")
    else:
        _require(deployment["full_2018_inside_catalogue_and_processed_dates"] is True
                 and deployment["recovered_qc_not_input"] is True, "Optional metadata/input distinction missing")
        processed = deployment["processed_catalog_filename_dates"]
        processed_start, processed_end = (datetime.strptime(processed[key], "%Y%m%d").date() for key in ("start", "end"))
        # Processed coverage can end before the catalogue deployment (21418 does).
        _require(processed_start <= lower and processed_end >= upper,
                 "Annual interval outside recorded processed calendar coverage")


def _publication_blocks(root, manifest, published):
    """Do not hide an optional complete-looking block by omitting its batch root.

    Marker presence never accepts a fit: all published fits still go through
    verified_completion. For omitted blocks it is a conservative refusal only.
    """
    for dataset in set(manifest["prepared_sha256"]) - set(published):
        jobs = [job for job in manifest["jobs"] if job[0] == dataset]
        markers = [root / "jobs" / "_".join(map(str, job)) / "complete.json" for job in jobs]
        for marker in markers:
            _require(not any(p.is_symlink() for p in (marker, *marker.parents)), "Symlink in omitted job")
        _require(not (len(markers) == 12 and all(p.exists() for p in markers)),
                 "Completed optional block omitted: publication requires all 36 fresh fits and diagnostic arrays")


def _batch_records(roots, fresh, manifest, bind):
    _require(type(roots) in (list, tuple) and roots, "Explicit nonempty batch-root list required")
    paths = [_path(root, True) for root in roots]
    _require(len(set(paths)) == len(paths), "Duplicate batch root")
    seen, output = set(), []
    for batch_index, root in enumerate(paths):
        _require({p.name for p in root.iterdir()} == {f"worker-{i}" for i in range(4)}, "Batch needs exactly four workers")
        all_jobs = None
        for worker in range(4):
            directory = _path(root / f"worker-{worker}", True)
            logical = f"{'stage' if fresh else 'core'}-batch-{batch_index:03d}/worker-{worker}"
            protocol = _read(directory / "protocol.json")
            complete = _read(directory / "complete.json")
            protocol_sha = bind(logical + "/protocol.json", directory / "protocol.json")
            bind(logical + "/complete.json", directory / "complete.json")
            purpose = "dart_stage_checkpoint_diagnostics" if fresh else "dart_fixed_checkpoint_diagnostics"
            sources = {"eval_gaussian_final_checkpoints.py": EVALUATOR_REVIEWED_SHA256}
            if not fresh:
                sources["prigm_v2_metrics.py"] = STAGE_SOURCES["prigm_v2_metrics.py"]
            common = {"schema_version": 2, "purpose": purpose, "evaluator_source_sha256": sources,
                      "worker_index": worker, "worker_count": 4, "dart_diagnostics": DART_DIAGNOSTICS,
                      "density_coordinates": "observation" if fresh else "residual", "probability_dtype": "float64",
                      "network_dtype": "float32", "chunk": 32,
                      "training_measure": "declared_train_weights" if fresh else "original_uniform_training_windows",
                      "evaluation_measure": "declared_eval_weights" if fresh else "original_uniform_test_windows"}
            manifest_key = "stage_manifest_sha256" if fresh else "core_manifest_sha256"
            common[manifest_key] = manifest["_sha"]
            if fresh:
                common.update(stage=manifest["stage"], stage_purpose=manifest["purpose"],
                              stage_source_sha256=manifest["code_sha256"],
                              archived_replay={"samples": 256, "streams": 3, "first_seed": 65001, "rtol": 1e-5, "atol": 1e-6})
            else:
                common.update(samples=256, stream_seeds=DART_DIAGNOSTICS["iw_seeds"])
            _keys(protocol, {*common, "jobs", "all_jobs", "runtime", *(["started_utc"] if fresh else [])}, "worker protocol")
            for key, value in common.items():
                _same(protocol[key], value, "protocol " + key)
            if all_jobs is None:
                all_jobs = protocol["all_jobs"]
                _require(type(all_jobs) is list and all_jobs, "Empty batch matrix")
                admitted = set(manifest["prepared_sha256"]) if fresh else set(OLD)
                datasets = set()
                for job in all_jobs:
                    _require(type(job) is list and len(job) == (5 if fresh else 4), "Invalid batch job")
                    _require(job[0] in admitted and type(job[2]) is int and job[2] in SEEDS
                             and job[-1] == ("final" if fresh else "selected"), "Invalid diagnostic endpoint/job")
                    _require((job[1], job[3]) in ARM_MAP if fresh else job[1] in ARMS, "Invalid condition")
                    datasets.add(job[0])
                canonical = ([j + ["final"] for j in manifest["jobs"] if j[0] in datasets] if fresh else
                             [[d, a, s, "selected"] for d in OLD if d in datasets for s in SEEDS for a in ARMS])
                _same(all_jobs, canonical, "complete batch matrix/order")
            _same(protocol["all_jobs"], all_jobs, "worker shared matrix")
            _same(protocol["jobs"], all_jobs[worker::4], "worker ownership")
            complete_fields = {"schema_version", "purpose", "protocol_sha256", manifest_key,
                               "evaluator_source_sha256", "artifact_sha256"}
            if fresh:
                complete_fields |= {"stage_source_sha256", "started_utc", "completed_utc", "elapsed_seconds"}
            _keys(complete, complete_fields, "worker completion")
            for key in ("schema_version", "purpose", manifest_key, "evaluator_source_sha256"):
                _same(complete[key], common[key], "completion " + key)
            _same(complete["protocol_sha256"], protocol_sha, "protocol hash")
            source_files = {"source/" + name: sha for name, sha in sources.items()}
            if fresh:
                _same(complete["stage_source_sha256"], manifest["code_sha256"], "completion source")
                source_files.update({"source/" + name: STAGE_SOURCES[name] for name in
                                     ("prigm_v2_metrics.py", "prigm_metrics.py")})
            artifacts = complete["artifact_sha256"]
            _require(type(artifacts) is dict, "Invalid worker artifacts")
            records, expected = [], dict(source_files)
            for job in protocol["jobs"]:
                identity = tuple(job)
                _require(identity not in seen, "Duplicate diagnostic block/job")
                seen.add(identity)
                identifier = "_".join(map(str, job[:-1]))
                stem = identifier + "-" + job[-1]
                record = _read(directory / (stem + ".json"))
                ref = record["references"]
                refname = ref["file"]
                _require(type(refname) is str and re.fullmatch(r"references-" + re.escape(job[0]) + r"-[0-9a-f]{16}\.npz", refname),
                         "Invalid reference filename")
                expected[stem + ".json"] = artifacts.get(stem + ".json")
                expected[stem + ".npz"] = record["array_sha256"]
                expected[refname] = ref["sha256"]
                _same(record["protocol_sha256"], protocol_sha, "record protocol")
                records.append((job, directory, logical, stem, record))
            # v1 workers do not list preserved source bytes in artifact_sha256.
            artifact_expected = expected if fresh else {k: v for k, v in expected.items() if not k.startswith("source/")}
            _same(artifacts, artifact_expected, "closed worker artifact map")
            _inventory(directory, {*expected, "protocol.json", "complete.json"})
            for name, sha in expected.items():
                bind(logical + "/" + name, directory / name, sha)
            output.extend(records)
    return output


def _validated_original(root, job, record, manifest, bind, experiment):
    """Bind the original selected record via the reviewed frozen runner, never a new endpoint."""
    dataset, arm, seed, endpoint = job
    identifier = f"{dataset}_{arm}_{seed}"
    marker = root / "jobs" / identifier / "complete.json"
    completion_sha = bind("core/jobs/" + identifier + "/complete.json", marker)
    completion = _read(marker)
    _require(re.fullmatch(r"attempt-[A-Za-z0-9-]+", completion.get("attempt", "")), "Unsafe core attempt")
    attempt = _path(marker.parent / completion["attempt"], True)
    for filename, digest in completion["artifact_sha256"].items():
        _require(Path(filename).name == filename, "Unsafe core artifact")
        bind(f"core/jobs/{identifier}/{filename}", attempt / filename, digest)
    _require(experiment.verified_completion(root, identifier) == attempt, "Missing verified original selected job")
    summary, config = _read(attempt / "summary.json"), _read(attempt / "configuration.json")
    _same(summary, manifest["_summaries"][identifier], "original selected root/job record")
    for name, digest in completion["artifact_sha256"].items():
        _same(digest, manifest["_archive"].get(f'jobs/{identifier}/{completion["attempt"]}/{name}'),
              "parent-validated original archive artifact")
        if name.endswith(".json"):
            _read(attempt / name)
    _same(completion_sha, manifest["_archive"].get(f"jobs/{identifier}/complete.json"), "original archived completion")
    _same(config["specification"], manifest["specification"], "original frozen specification")
    _same(config["run_manifest_sha256"], CORE_SHA256, "original configuration root")
    checks = {"identifier": identifier, "dataset": dataset, "arm": arm, "seed": seed,
              "checkpoint": endpoint, "checkpoint_sha256": _sha(attempt / "best.weights.h5"),
              "core_completion_sha256": completion_sha, "prepared_sha256": manifest["prepared_sha256"][dataset + ".npz"],
              "core_evaluation_sha256": _sha(attempt / "evaluation.npz"), "selected_step": summary["best_step"],
              "steps_run": summary["steps_run"], "frozen_evaluator_reproduced_exactly": True}
    for key, expected in checks.items():
        _same(record[key], expected, "original record " + key)
    return attempt, summary


def _replay(arrays, record, attempt, data, variance, fresh, metrics):
    n = len(data["test"])
    prefix = "stage_replay__" if fresh else ""
    components = {k: arrays[prefix + k] for k in COMPONENTS}
    _components(components, n, variance, _weights(data, "test", fresh), metrics)
    for key in ("posterior_mean_mse", "kl", "kl_by_latent"):
        _close(arrays["original__" + key], components[key], "diagnostic deterministic replay " + key,
               rtol=1e-5, atol=1e-6)
    evaluation = attempt / (record["stage_evaluation_file"] if fresh else "evaluation.npz")
    differences, exact = {}, True
    with np.load(_path(evaluation), allow_pickle=False) as saved:
        if fresh:
            for key, value in components.items():
                original = saved[{"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key)]
                _close(value, original, "archived stage replay", rtol=1e-5, atol=1e-6)
                differences[key] = float(np.max(np.abs(value - original)))
                exact = exact and np.array_equal(value, original)
            expected = {"samples": 256, "stream_seeds": [65001, 65002, 65003],
                        "max_abs_differences": differences, "reproduced_exactly": bool(exact), "rtol": 1e-5, "atol": 1e-6}
            _same(record["archived_component_replay"], expected, "stage replay receipt")
            return None
        for key, saved_key in (("log_score", "log_score"), ("ess", "ess"), ("posterior_mean_mse", "corrected_mse")):
            value, original = components[key][0], saved[saved_key]
            _close(value, original, "archived original replay", rtol=1e-5, atol=1e-6)
            if key != "posterior_mean_mse":
                _require(np.array_equal(value, original), "Original IW replay not exact")
            differences[key + "_max_abs"] = float(np.max(np.abs(value - original)))
        _same(record["archived_selected_replay_differences"], differences, "old replay receipt")
        raw = _array(saved["self_mse"], (n,), "archived raw self MSE")
        mean = _array(saved["posterior_mean"], (n, 192), "archived posterior mean")
        b = 1 - .8 if record["arm"] in ("B", "C") else 1.
        _close(saved["corrected_mse"], np.square(data["test"] - mean).mean(axis=1), "archived corrected MSE")
        _close(raw, b**2 * saved["corrected_mse"], "actual saved raw self MSE", rtol=1e-5, atol=1e-8)
        return float(raw.mean())


def _selected(attempt, summary, data, variance, blocks, metrics):
    step = summary["selected_step"]
    alias = step == 10000
    label = "final" if alias else "selected"
    path = attempt / f"evaluation-{label}.npz"
    # Already fully checked by verified_completion, including HDF5 and runner _check_evaluation.
    arrays = _load(path)
    w = _weights(data, "test", True)
    for i, seed in enumerate(METHODS["selected_secondary_streams"]["prior_seeds"]):
        _close(arrays["prior_samples"][i], _observation_draw(arrays["prior_means"][i], variance, seed),
               "archived secondary seeded noise", rtol=1e-12, atol=1e-12)
    scores = {}
    for name, key, function in zip(SCORES, ("energy", None, "increment_energy"),
                                   (metrics.energy_score, metrics.centered_energy_score, metrics.increment_energy_score)):
        values = (np.stack([function(draw, data["test"]) for draw in arrays["prior_samples"]])
                  if key is None else arrays[key + "_per_window"])
        scores[name] = _score_summary(values, w, blocks)
    components = {key: arrays[{"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key)] for key in COMPONENTS}
    return {"checkpoint": "selected", "step": step, "aliases_final": alias,
            "weights_sha256": _sha(attempt / "selected.weights.h5"), "evaluation_sha256": _sha(path),
            "scores": scores, "components": _components(components, len(w), variance, w, metrics),
            "raw_self_mse": None}


def _aggregate(models):
    summaries, contrasts = {}, {}
    for variant in ("original", "variance_refit"):
        for arm in ARMS:
            fits = [models[f"{arm}/{seed}/{variant}"] for seed in SEEDS]
            summaries[f"{arm}/{variant}"] = {
                "scores": {name: _seed_summary([fit["scores"][name]["weighted"]["mean"] for fit in fits]) for name in SCORES},
                "components": {name: _seed_summary([fit["components"]["weighted"][name]["mean"] for fit in fits]) for name in REDUCTIONS},
                "observation_variance": _seed_summary([fit["observation_variance"] for fit in fits])}
        for left, right in (("C", "A"), ("B", "D")):
            paired = {}
            for score in SCORES:
                stream_deltas = []
                for seed in SEEDS:
                    a, b = (models[f"{arm}/{seed}/{variant}"] for arm in (left, right))
                    if variant == "original":
                        _close(a["observation_variance"], b["observation_variance"], "matched original variance", atol=0)
                    stream_deltas.append(np.asarray(a["scores"][score]["weighted"]["values"]) - b["scores"][score]["weighted"]["values"])
                paired[score] = {"optimizer_seeds": _seed_summary([x.mean() for x in stream_deltas]),
                                 "paired_mc_streams": [_streams(x) for x in stream_deltas]}
            contrasts[f"{left}-{right}/{variant}"] = {"matched_variance": variant == "original", "scores": paired}
    return summaries, contrasts


def verify_dart_extension(core_root, stage_root, core_diagnostics_roots,
                          stage_diagnostics_roots, expected_stage_sha256, admission_file):
    """Return (closed public schema-1 dict, private paths/data), or fail closed.

    Each batch root contains exactly worker-0..3. Batches contain whole record
    blocks, cannot overlap, and never rewrite/merge producer protocols. Only
    final fresh diagnostics are inputs; selected secondary arrays come from the
    verified runner completion (with its final alias rule).
    """
    core_root, stage_root = _path(core_root, True), _path(stage_root, True)
    _digest(expected_stage_sha256)
    files, snapshot, inventories = {}, {}, {}

    def bind(logical, path, expected=None):
        path = _path(path)
        digest = _sha(path)
        if expected is not None:
            _require(digest == _digest(expected), "Input SHA256 differs")
        _require(logical not in files or files[logical] == digest, "Logical input changed")
        _require(path not in snapshot or snapshot[path] == digest, "Input changed during verification")
        files[logical], snapshot[path] = digest, digest
        return digest

    for name, expected in REVIEWED_SOURCES.items():
        bind("reviewed/" + name, ROOT / "src" / name, expected)
    helper_sha = bind("reviewed/prigm_dart_paper_evidence.py", Path(__file__))
    bind("core/run_manifest.json", core_root / "run_manifest.json", CORE_SHA256)
    bind("stage/stage_manifest.json", stage_root / "stage_manifest.json", expected_stage_sha256)
    core, stage = _read(core_root / "run_manifest.json"), _read(stage_root / "stage_manifest.json")
    for root, manifest, sources, kind in ((core_root, core, CORE_SOURCES, "core"),
                                          (stage_root, stage, STAGE_SOURCES, "stage")):
        _same(manifest["code_sha256"], {"src/" + k: v for k, v in sources.items()}, "reviewed archive sources")
        inventory = {"src/" + name for name in sources}
        _inventory(root / "source", inventory)
        inventories[root / "source"] = inventory
        for name, digest in manifest["code_sha256"].items():
            bind(kind + "/source/" + name, root / "source" / name, digest)
        prepared = ({name + ".npz": sha for name, sha in manifest["prepared_sha256"].items()}
                    if kind == "stage" else dict(manifest["prepared_sha256"]))
        if kind == "stage":
            prepared["records.json"] = manifest["records_sha256"]
        _inventory(root / "prepared", prepared)
        inventories[root / "prepared"] = set(prepared)
        for name, digest in prepared.items():
            bind(kind + "/prepared/" + name, root / "prepared" / name, digest)
            if name.endswith(".json"):
                _read(root / "prepared" / name)
    metrics, runner, original_runner, evaluator = _imports()
    _same(evaluator.verified_core_source(core_root, CORE_SHA256), core, "verified original source")
    _same(evaluator.verified_stage_source(stage_root, expected_stage_sha256), stage, "verified stage source")
    stage_identity = {"manifest_sha256": expected_stage_sha256,
                      "code_sha256": {"src/" + name: digest for name, digest in STAGE_SOURCES.items()}}
    _same(runner.verify_stage(stage_root, historical_identity=stage_identity), stage, "verified DART stage")
    importlib.import_module("h5py")  # Fail early; never substitute HDF5 signature checks.
    core_manifest, stage_manifest = dict(core, _sha=CORE_SHA256), dict(stage, _sha=expected_stage_sha256)
    for name in ("archive_manifest.json", "results.json"):
        bind("core/" + name, core_root / name)
    archive = _read(core_root / "archive_manifest.json")
    results = _read(core_root / "results.json")
    _same(results["run_manifest_sha256"], CORE_SHA256, "original result root")
    _same(results["specification"], core["specification"], "original result specification")
    summaries = {}
    for row in results["results"]:
        identifier = f'{row["dataset"]}_{row["arm"]}_{row["seed"]}'
        _require(identifier not in summaries, "Duplicate original result")
        summaries[identifier] = row
    core_manifest.update(_summaries=summaries, _archive=archive["artifact_sha256"])
    old_records = _batch_records(core_diagnostics_roots, False, core_manifest, bind)
    new_records = _batch_records(stage_diagnostics_roots, True, stage_manifest, bind)
    _require({j[0] for j, *_ in old_records} == set(OLD) and len(old_records) == 48, "Missing original diagnostic block")
    names = {j[0] for j, *_ in new_records}
    admission = _read(admission_file) if admission_file is not None else None
    admission_public = _admission(admission, expected_stage_sha256, names)
    if admission is not None:
        admission_public["admission_sha256"] = bind("admission.json", admission_file)
    _require(len(new_records) == 12 * len(names), "Incomplete fresh diagnostic blocks")
    _publication_blocks(stage_root, stage, names)
    # Registration is not completion; optional records may be running but unpublished.
    metadata = _read(core_root / "prepared/metadata.json")
    fresh_metadata = _read(stage_root / "prepared/records.json")["records"]
    public_records, data_cache, block_cache = {}, {}, {}
    for dataset in (*OLD, *sorted(names)):
        fresh = dataset in names
        root = stage_root if fresh else core_root
        data = _load(root / "prepared" / (dataset + ".npz"))
        year = 2018 if fresh else int(dataset[-4:])
        station = dataset.split("_")[1] if fresh else dataset[:5]
        splits = {}
        for split in ("train", "validation", "test"):
            n = len(data[split])
            _array(data[split], (n, 192), split)
            starts = _array(data["starts_" + split], (n,), "prepared timestamps", True)
            _require(np.all(np.diff(starts) > 0), "Nonmonotonic prepared starts")
            _weights(data, split, fresh)
            splits[split] = {"windows": n, "first_start_unix": int(starts[0]), "last_start_unix": int(starts[-1]),
                             "packs": stage["record_summaries"][dataset][split]["packs"] if fresh else None}
        for key in ("mean", "scale"):
            _array(data[key], (), key)
            _number(float(data[key]), key == "scale")
        normalization = "unique_covered_training_source_rows" if fresh else "original_valid_training_source_rows"
        public_records[dataset] = {"station": station, "year": year, "target": "six_UTC_bins_equal" if fresh else "original_uniform_windows",
                                   "primary_checkpoint": "final" if fresh else "selected", "splits": splits,
                                   "mean": float(data["mean"]), "scale": float(data["scale"]), "normalization": normalization,
                                   "training_measure": "six_UTC_bins_equal" if fresh else "original_uniform_windows",
                                   "window_measures": {}, "metadata_sha256": stage["records_sha256"] if fresh else core["prepared_sha256"]["metadata.json"],
                                   "prepared_sha256": stage["prepared_sha256"][dataset] if fresh else core["prepared_sha256"][dataset + ".npz"],
                                   "metadata_scope": "catalogued_annual_coverage_hardware_unestablished" if fresh else "original_record_protocol_retained",
                                   "selection_role": ("retrospective_additional_year_same_station" if dataset == PRIMARY else
                                                      "data_only_qualified_optional_site_resource_admitted" if fresh else "original_case_study"),
                                   "models": {}, "references": {}, "optimizer_summaries": {}, "contrasts": {}}
        for split in splits:
            stats = metrics.window_spectrum_statistics(data[split], _weights(data, split, fresh))
            public_records[dataset]["window_measures"][split] = {key: float(stats[key]) for key in
                    ("within_window_variance", "between_window_mean_variance", "pooled_entry_variance")}
        if fresh:
            _metadata_coverage(fresh_metadata[dataset], dataset)
        else:
            _close(data["mean"], metadata[dataset]["mean"], "original normalization mean")
            _close(data["scale"], metadata[dataset]["scale"], "original normalization scale")
        data_cache[dataset] = data
        block_cache[dataset] = _blocks(data["starts_test"], year)
    private = {"model_paths": {}, "data": data_cache, "reference_paths": {}}
    ref_cache = {}
    for fresh, entries, root in ((False, old_records, core_root), (True, new_records, stage_root)):
        for job, directory, logical, stem, record in entries:
            dataset, condition, seed = job[:3]
            arm = ARM_MAP[(condition, job[3])] if fresh else condition
            variance_label = "low" if arm in ("A", "C") else "high"
            data, blocks = data_cache[dataset], block_cache[dataset]
            identifier = "_".join(map(str, job[:-1]))
            common_fields = {"schema_version", "purpose", "identifier", "dataset", "seed", "checkpoint",
                             "checkpoint_sha256", "prepared_sha256", "observation_variance", "dimension", "test_windows",
                             "protocol_sha256", "array_sha256", "array_shapes", "dart_diagnostics", "references"}
            extra = ({"condition", "variance_label", "step", "stage_completion_sha256", "stage_manifest_sha256",
                      "stage_evaluation_file", "stage_evaluation_sha256", "archived_component_replay"} if fresh else
                     {"arm", "core_completion_sha256", "selected_step", "steps_run", "frozen_evaluator_reproduced_exactly",
                      "core_evaluation_sha256", "archived_selected_replay_differences"})
            _keys(record, common_fields | extra, "diagnostic record")
            _same(record["schema_version"], 2, "record schema")
            _same(record["purpose"], "dart_stage_checkpoint_diagnostics" if fresh else "dart_fixed_checkpoint_diagnostics", "record purpose")
            if fresh:
                marker = root / "jobs" / identifier / "complete.json"
                complete_sha = bind(f"stage/jobs/{identifier}/complete.json", marker)
                completion = _read(marker)
                attempt_name = completion.get("attempt", "")
                _require(re.fullmatch(r"attempt-[0-9a-f]{32}", attempt_name), "Unsafe fresh attempt")
                attempt = _path(marker.parent / attempt_name, True)
                bind(f"stage/jobs/{identifier}/attempt-complete.json", attempt / "complete.json")
                for name, digest in completion["artifact_sha256"].items():
                    _require(Path(name).name == name, "Unsafe fresh artifact")
                    bind(f"stage/jobs/{identifier}/{name}", attempt / name, digest)
                    if name.endswith(".json"):
                        _read(attempt / name)
                _require(runner.verified_completion(root, identifier, historical_identity=stage_identity) == attempt,
                         "Fresh job not fully completed")
                inventories[attempt] = {*completion["artifact_sha256"], "complete.json"}
                summary = _read(attempt / "summary.json")
                variance = stage["job_parameters"][identifier]["observation_variance"]
                checks = {"identifier": identifier, "dataset": dataset, "condition": condition, "seed": seed,
                          "variance_label": job[3], "checkpoint": "final", "step": 10000,
                          "checkpoint_sha256": _sha(attempt / "final.weights.h5"),
                          "stage_completion_sha256": complete_sha, "stage_manifest_sha256": expected_stage_sha256,
                          "prepared_sha256": stage["prepared_sha256"][dataset], "stage_evaluation_file": "evaluation-final.npz",
                          "stage_evaluation_sha256": _sha(attempt / "evaluation-final.npz")}
                for key, value in checks.items():
                    _same(record[key], value, "fresh record " + key)
                if dataset in OPTIONAL:
                    config = _read(attempt / "configuration.json")
                    # Chronology of source admission is attested; no outcome-based rule is inferred.
                    _require(admission is not None, "Optional fit without resource admission")
                    _require(config["dataset"] == dataset, "Optional record configuration")
                step = 10000
            else:
                attempt, summary = _validated_original(root, job, record, core_manifest, bind, original_runner)
                base = float(data["base_variance"])
                b = 1 - .8 if arm in ("B", "C") else 1.
                variance = (base * b**2 if arm == "C" else base / (1 - .8)**2 if arm == "D" else base) / b**2
                step = summary["best_step"]
            _same(record["dimension"], 192, "model dimension")
            _same(record["test_windows"], len(data["test"]), "test count")
            _close(record["observation_variance"], variance, "original variance", atol=0)
            shapes = _diagnostic_shapes(len(data["test"]), len(data["train"]), fresh)
            _same(record["array_shapes"], {k: list(v) for k, v in shapes.items()}, "declared array shapes")
            arrays = _load(directory / (stem + ".npz"), shapes)
            result = _verify_arrays(arrays, data, variance, fresh, blocks, metrics)
            raw_mse = _replay(arrays, record, attempt, data, variance, fresh, metrics)
            del arrays
            selected = _selected(attempt, summary, data, variance, blocks, metrics) if fresh else None
            _keys(record["dart_diagnostics"], result, "diagnostic variants")
            for label, model in result.items():
                receipt = {"observation_variance": model["observation_variance"],
                           **{name + "_energy": model["scores"][name]["weighted"]["mean"] for name in SCORES},
                           "mean_iw_log_density_per_coordinate": -model["components"]["weighted"]["negative_iw_per_coordinate"]["mean"],
                           "median_ess": model["components"]["median_ess"]}
                _keys(record["dart_diagnostics"][label], receipt, "diagnostic scalar receipt")
                for key, expected in receipt.items():
                    _close(record["dart_diagnostics"][label][key], expected, "diagnostic scalar " + key)
                model.update(condition=arm, seed=seed, original_variance_label=variance_label,
                             checkpoint="final" if fresh else "selected", step=step,
                             checkpoint_sha256=record["checkpoint_sha256"], original_observation_variance=float(variance),
                             raw_self_mse=raw_mse, raw_self_mse_source="not_saved_by_fresh_runner" if fresh else "original_evaluation_self_mse",
                             selected_secondary=selected if label == "original" else None)
                public_records[dataset]["models"][f"{arm}/{seed}/{label}"] = model
            private["model_paths"][dataset + "/" + arm + "/" + str(seed)] = directory / (stem + ".npz")
            reference = record["references"]
            _keys(reference, {"file", "sha256", "observation_variance", "summary", *(["array_shapes"] if fresh else [])}, "reference record")
            _close(reference["observation_variance"], variance, "reference matched noise", atol=0)
            key = (dataset, float(variance))
            if key not in ref_cache:
                shapes = _reference_shapes(len(data["test"]))
                ref_arrays = _load(directory / reference["file"], shapes)
                result = _references(ref_arrays, data, variance, fresh, blocks, metrics)
                del ref_arrays
                ref_cache[key] = (reference["sha256"], result)
            else:
                _same(reference["sha256"], ref_cache[key][0], "copied reference arrays; not new streams")
            reference_result = ref_cache[key][1]
            if fresh:
                _same(reference["array_shapes"], {k: list(v) for k, v in _reference_shapes(len(data["test"])).items()}, "reference shapes")
            _keys(reference["summary"], reference_result, "reference summary")
            for name, values in reference_result.items():
                _keys(reference["summary"][name], {s + "_energy" for s in SCORES}, "reference score receipt")
                for score in SCORES:
                    _close(reference["summary"][name][score + "_energy"], values["scores"][score]["weighted"]["mean"], "reference scalar")
            public_records[dataset]["references"][variance_label] = deepcopy(reference_result)
            private["reference_paths"][dataset + "/" + variance_label] = directory / reference["file"]
    for record in public_records.values():
        record["optimizer_summaries"], record["contrasts"] = _aggregate(record["models"])
    index = {"core_manifest_sha256": CORE_SHA256, "stage_manifest_sha256": expected_stage_sha256,
             "reviewed_source_sha256": dict(REVIEWED_SOURCES), "helper_sha256": helper_sha, "files": files}
    public = {"schema_version": 1, "purpose": "dart_window_diagnostics", "input_index": index,
              "methods": deepcopy(METHODS), "verification": deepcopy(VERIFICATION),
              "admission": admission_public, "records": public_records}
    validate_dart_extension(public)
    _publication_blocks(stage_root, stage, names)
    for path, expected in snapshot.items():
        _require(_sha(path) == expected, "Input changed after verification")
    for path, expected in inventories.items():
        _inventory(path, expected)
    # Recheck worker inventories too; a newly added unlisted file must fail.
    for _, directory, logical, _, _ in old_records + new_records:
        names = {name[len(logical) + 1:] for name in files if name.startswith(logical + "/")}
        _inventory(directory, names)
    return public, private


def _validate_stream(value, optimizer=False):
    key = "median" if optimizer else "mean"
    _keys(value, ("values", key, "min", "max"), "variation summary")
    _require(type(value["values"]) is list and len(value["values"]) == 3, "Exactly three streams/seeds required")
    for v in value["values"]:
        _number(v)
    _same(value, _seed_summary(value["values"]) if optimizer else _streams(value["values"]), "variation reductions")


def _validate_scores(value, n, *, uniform_target=False):
    _keys(value, SCORES, "scores")
    for score in value.values():
        _keys(score, ("weighted", "uniform", "blocks"), "score")
        for measure in ("weighted", "uniform"):
            _validate_stream(score[measure])
        if uniform_target:
            _close(score["weighted"]["values"], score["uniform"]["values"], "uniform target score streams")
        _keys(score["blocks"], ("14", "28"), "start-time blocks")
        for rows in score["blocks"].values():
            _require(type(rows) is list and rows, "Missing descriptive blocks")
            seen, count, mass = set(), 0, 0.
            pooled = np.zeros(3)
            for row in rows:
                _keys(row, ("block", "windows", "mass", "streams"), "block")
                _require(type(row["block"]) is int and 0 <= row["block"] <= 26 and row["block"] not in seen, "Duplicate/invalid block")
                _require(type(row["windows"]) is int and row["windows"] > 0, "Invalid block window count")
                _number(row["mass"], True)
                if uniform_target:
                    _close(row["mass"], row["windows"] / n, "uniform target block mass")
                _validate_stream(row["streams"])
                seen.add(row["block"])
                count += row["windows"]
                mass += row["mass"]
                pooled += row["mass"] * np.asarray(row["streams"]["values"])
            _require(count == n, "Block partition window count differs")
            _close(mass, 1., "block mass")
            _close(pooled, score["weighted"]["values"], "descriptive block reductions")


def _validate_components(value, variance, *, uniform_target=False):
    _keys(value, ("weighted", "uniform", "median_ess"), "component summary")
    _require(1 - 1e-7 <= _number(value["median_ess"]) <= 256 + 1e-7, "Invalid median ESS")
    for measure in ("weighted", "uniform"):
        _keys(value[measure], REDUCTIONS, "component reductions")
        for stream in value[measure].values():
            _validate_stream(stream)
        for ess in value[measure]["mean_ess"]["values"]:
            _require(1 - 1e-7 <= ess <= 256 + 1e-7, "Invalid mean ESS")
        for key in ("posterior_mean_mse", "posterior_expected_mse", "kl_per_coordinate", "jensen_gap_per_coordinate"):
            _require(min(value[measure][key]["values"]) >= -1e-10, "Negative public MSE/KL/Jensen gap")
        nll = np.asarray(value[measure]["nll_per_coordinate"]["values"])
        mse = np.asarray(value[measure]["posterior_expected_mse"]["values"])
        _close(nll, .5 * (np.log(2 * np.pi * variance) + mse / variance), "public normalized Gaussian likelihood")
        _close(value[measure]["negative_elbo_per_coordinate"]["values"],
               nll + value[measure]["kl_per_coordinate"]["values"], "public analytic ELBO")
    if uniform_target:
        for key in REDUCTIONS:
            _close(value["weighted"][key]["values"], value["uniform"][key]["values"],
                   "uniform target component " + key)


def _validate_diagnostic_index(files, datasets):
    for fresh, kind in ((False, "core"), (True, "stage")):
        expected = set()
        for dataset in datasets:
            if (dataset not in OLD) != fresh:
                continue
            for arm in ARMS:
                for seed in SEEDS:
                    condition, label = ("base" if arm in ("A", "D") else "raw"), ("low" if arm in ("A", "C") else "high")
                    expected.add(f"{dataset}_{condition}_{seed}_{label}-final" if fresh else f"{dataset}_{arm}_{seed}-selected")
        paths = [name for name in files if name.startswith(kind + "-batch-")]
        workers = {"/".join(name.split("/")[:2]) for name in paths}
        batches = {name.split("/")[0] for name in workers}
        _require(batches == {f"{kind}-batch-{i:03d}" for i in range(len(batches))} and batches,
                 "Missing/noncanonical diagnostic batch index")
        _require(workers == {batch + f"/worker-{i}" for batch in batches for i in range(4)}, "Incomplete indexed worker block")
        seen = set()
        for worker in workers:
            members = {name[len(worker) + 1:] for name in paths if name.startswith(worker + "/")}
            sources = {"eval_gaussian_final_checkpoints.py": EVALUATOR_REVIEWED_SHA256,
                       "prigm_v2_metrics.py": STAGE_SOURCES["prigm_v2_metrics.py"]}
            if fresh:
                sources["prigm_metrics.py"] = STAGE_SOURCES["prigm_metrics.py"]
            for name, digest in sources.items():
                _same(files.get(worker + "/source/" + name), digest, "indexed reviewed producer source")
            _require({"protocol.json", "complete.json"} <= members, "Missing indexed worker receipt")
            stems = {name[:-4] for name in members if name.endswith(".npz") and not name.startswith("references-")}
            json_stems = {name[:-5] for name in members if name.endswith(".json") and name not in ("protocol.json", "complete.json")}
            _require(stems and stems == json_stems and not (stems & seen), "Missing/duplicate indexed diagnostic arrays")
            _require(any(name.startswith("references-") for name in members), "Missing indexed references")
            seen.update(stems)
        _require(seen == expected, "Indexed diagnostic blocks differ from public models")


def validate_dart_extension(value, core=None, input_index=None):
    """Pure closed-schema/reduction validation. No filesystem, imports or self-hash.

    ``input_index`` is this helper's input index, not the containing combined
    manifest. ``core`` may be the parent's original public core evidence.
    Cryptographic revalidation of private inputs remains verify_dart_extension's
    responsibility; an input digest is not an authenticity signature.
    """
    _keys(value, ("schema_version", "purpose", "input_index", "methods", "verification", "admission", "records"), "DART public")
    _same(value["schema_version"], 1, "public schema")
    _same(value["purpose"], "dart_window_diagnostics", "public purpose")
    _same(value["methods"], METHODS, "methods")
    _same(value["verification"], VERIFICATION, "verification scope")
    index = value["input_index"]
    _keys(index, ("core_manifest_sha256", "stage_manifest_sha256", "reviewed_source_sha256", "helper_sha256", "files"), "input index")
    _same(index["core_manifest_sha256"], CORE_SHA256, "approved core")
    _digest(index["stage_manifest_sha256"])
    _digest(index["helper_sha256"])
    _same(index["reviewed_source_sha256"], REVIEWED_SOURCES, "reviewed source pins")
    _require(type(index["files"]) is dict and index["files"], "Empty input index")
    # Only stable names synthesized by this helper; no UUID, host, PID or root paths.
    file_pattern = (r"(?:reviewed/(?:prigm(?:_v2)?_(?:data|model|metrics|experiment)|eval_gaussian_final_checkpoints|prigm_dart_paper_evidence)\.py"
                    r"|admission\.json|(?:core|stage)/(?:run_manifest\.json|stage_manifest\.json|archive_manifest\.json|results\.json"
                    r"|prepared/(?:(?:metadata|records)\.json|(?:gaussian|[0-9]{5}t[0-9]{4}|dart_[0-9]{5}_2018)\.npz)"
                    r"|source/src/prigm(?:_v2)?_(?:data|model|metrics|experiment)\.py"
                    r"|jobs/(?:[0-9]{5}t[0-9]{4}_[ABCD]_(?:42|1729|31415)|dart_[0-9]{5}_2018_(?:base|raw)_(?:42|1729|31415)_(?:low|high))/"
                    r"(?:(?:configuration|history|progress|summary|complete|attempt-complete)\.json|(?:history|probes)\.jsonl"
                    r"|(?:best|final|selected)\.weights\.h5|evaluation(?:-final|-selected)?\.npz|last_checkpoint\.(?:index|data-00000-of-00001)))"
                    r"|(?:core|stage)-batch-[0-9]{3}/worker-[0-3]/(?:source/(?:eval_gaussian_final_checkpoints|prigm_v2_metrics|prigm_metrics)\.py"
                    r"|(?:protocol|complete)\.json|references-(?:[0-9]{5}t[0-9]{4}|dart_[0-9]{5}_2018)-[0-9a-f]{16}\.npz"
                    r"|(?:[0-9]{5}t[0-9]{4}_[ABCD]_(?:42|1729|31415)-selected|dart_[0-9]{5}_2018_(?:base|raw)_(?:42|1729|31415)_(?:low|high)-final)\.(?:json|npz)))")
    for name, digest in index["files"].items():
        _require(type(name) is str and re.fullmatch(file_pattern, name) and ".." not in name, "Private/unsafe input name")
        _digest(digest)
    for name, digest in {"core/run_manifest.json": CORE_SHA256, "stage/stage_manifest.json": index["stage_manifest_sha256"],
                         "reviewed/prigm_dart_paper_evidence.py": index["helper_sha256"],
                         **{"reviewed/" + k: v for k, v in REVIEWED_SOURCES.items()}}.items():
        _same(index["files"].get(name), digest, "indexed source/root identity")
    if input_index is not None:
        _same(index, input_index, "combined input binding")
    if core is not None:
        _same(core["run_manifest_sha256"], CORE_SHA256, "parent core identity")
    records = value["records"]
    _require(type(records) is dict, "Invalid record map")
    _require(set(records) in ({*OLD, PRIMARY}, {*OLD, PRIMARY, *OPTIONAL}), "Missing/partial publication record block")
    _validate_diagnostic_index(index["files"], records)
    admission = value["admission"]
    _keys(admission, ("optional_admitted", "published_fits", "admission_sha256"), "public admission")
    _require(type(admission["optional_admitted"]) is bool, "Invalid admission flag")
    _same(admission["published_fits"], 12 * (len(records) - 4), "published fresh count")
    if admission["optional_admitted"]:
        _digest(admission["admission_sha256"])
        _same(index["files"].get("admission.json"), admission["admission_sha256"], "admission binding")
    else:
        _require(admission["published_fits"] == 12 and admission["admission_sha256"] is None
                 and "admission.json" not in index["files"], "Unadmitted optional block")
    for dataset, record in records.items():
        fresh = dataset not in OLD
        _keys(record, ("station", "year", "target", "primary_checkpoint", "splits", "mean", "scale", "normalization",
                       "training_measure", "window_measures", "metadata_sha256", "prepared_sha256", "metadata_scope", "selection_role",
                       "models", "references", "optimizer_summaries", "contrasts"), "record")
        _same(record["station"], dataset.split("_")[1] if fresh else dataset[:5], "station")
        _same(record["year"], 2018 if fresh else int(dataset[-4:]), "year")
        target = "six_UTC_bins_equal" if fresh else "original_uniform_windows"
        _same(record["target"], target, "window target")
        _same(record["training_measure"], target, "training target")
        _same(record["primary_checkpoint"], "final" if fresh else "selected", "primary endpoint")
        _same(record["normalization"], "unique_covered_training_source_rows" if fresh else "original_valid_training_source_rows", "normalization")
        _same(record["metadata_scope"], "catalogued_annual_coverage_hardware_unestablished" if fresh else "original_record_protocol_retained", "metadata limits")
        _same(record["selection_role"], "retrospective_additional_year_same_station" if dataset == PRIMARY else
              "data_only_qualified_optional_site_resource_admitted" if fresh else "original_case_study", "selection/admission limits")
        _number(record["mean"])
        _number(record["scale"], True)
        for key in ("metadata_sha256", "prepared_sha256"):
            _digest(record[key])
        kind = "stage" if fresh else "core"
        _same(record["prepared_sha256"], index["files"].get(f"{kind}/prepared/{dataset}.npz"), "prepared binding")
        _same(record["metadata_sha256"], index["files"].get(f"{kind}/prepared/" + ("records.json" if fresh else "metadata.json")), "metadata binding")
        _keys(record["splits"], ("train", "validation", "test"), "splits")
        _keys(record["window_measures"], record["splits"], "window measures")
        for split, row in record["splits"].items():
            _keys(row, ("windows", "first_start_unix", "last_start_unix", "packs"), "split support")
            _require(type(row["windows"]) is int and row["windows"] > 0, "Invalid window count")
            _require(type(row["first_start_unix"]) is int and type(row["last_start_unix"]) is int
                     and row["first_start_unix"] <= row["last_start_unix"], "Invalid start support")
            _require(type(row["packs"]) is int and 0 < row["packs"] <= row["windows"] if fresh else row["packs"] is None,
                     "Invalid disjoint support count")
            moments = record["window_measures"][split]
            _keys(moments, ("within_window_variance", "between_window_mean_variance", "pooled_entry_variance"), "window moments")
            for v in moments.values():
                _require(_number(v) >= 0, "Negative window variance")
            _close(moments["pooled_entry_variance"], moments["within_window_variance"] + moments["between_window_mean_variance"], "window variance decomposition")
        n = record["splits"]["test"]["windows"]
        models = record["models"]
        _keys(models, {f"{a}/{s}/{v}" for a in ARMS for s in SEEDS for v in ("original", "variance_refit")}, "complete 12-fit model block")
        for key, model in models.items():
            arm, seed, variant = key.split("/")
            _keys(model, ("observation_variance", "variance_fit", "scores", "components", "condition", "seed",
                          "original_variance_label", "checkpoint", "step", "checkpoint_sha256", "original_observation_variance",
                          "raw_self_mse", "raw_self_mse_source", "selected_secondary"), "model")
            _same(model["condition"], arm, "condition mapping")
            _same(model["seed"], int(seed), "optimizer seed")
            _same(model["original_variance_label"], "low" if arm in ("A", "C") else "high", "variance label")
            _same(model["checkpoint"], record["primary_checkpoint"], "model endpoint")
            _require(type(model["step"]) is int and (model["step"] == 10000 if fresh else 0 < model["step"] <= 10000), "Invalid endpoint step")
            _digest(model["checkpoint_sha256"])
            condition = "base" if arm in ("A", "D") else "raw"
            identifier = (f'{dataset}_{condition}_{seed}_{model["original_variance_label"]}' if fresh else f"{dataset}_{arm}_{seed}")
            job_prefix = f"{kind}/jobs/{identifier}/"
            weight_file = "final.weights.h5" if fresh else "best.weights.h5"
            _same(index["files"].get(job_prefix + weight_file), model["checkpoint_sha256"], "indexed primary weights")
            _number(model["observation_variance"], True)
            _number(model["original_observation_variance"], True)
            fit = model["variance_fit"]
            _keys(fit, ("streams", "pooled"), "variance fit")
            _validate_stream(fit["streams"])
            _require(min(fit["streams"]["values"]) > 0, "Nonpositive public training variance")
            _number(fit["pooled"], True)
            _same(fit["pooled"], fit["streams"]["mean"], "pooled refit")
            _same(model["observation_variance"], model["original_observation_variance"] if variant == "original" else fit["pooled"], "model variance role")
            _validate_scores(model["scores"], n, uniform_target=not fresh)
            _validate_components(model["components"], model["observation_variance"], uniform_target=not fresh)
            _same(model["raw_self_mse_source"], "not_saved_by_fresh_runner" if fresh else "original_evaluation_self_mse", "raw self MSE source")
            _require(model["raw_self_mse"] is None if fresh else _number(model["raw_self_mse"]) >= 0, "Invalid raw self MSE")
            secondary = model["selected_secondary"]
            if fresh and variant == "original":
                _keys(secondary, ("checkpoint", "step", "aliases_final", "weights_sha256", "evaluation_sha256", "scores", "components", "raw_self_mse"), "selected secondary")
                _same(secondary["checkpoint"], "selected", "secondary endpoint")
                _require(type(secondary["step"]) is int and 0 < secondary["step"] <= 10000, "Invalid selected step")
                _same(secondary["aliases_final"], secondary["step"] == 10000, "selected final alias")
                _digest(secondary["weights_sha256"])
                _digest(secondary["evaluation_sha256"])
                _same(index["files"].get(job_prefix + "selected.weights.h5"), secondary["weights_sha256"], "indexed selected weights")
                evaluation_file = "evaluation-final.npz" if secondary["aliases_final"] else "evaluation-selected.npz"
                _same(index["files"].get(job_prefix + evaluation_file), secondary["evaluation_sha256"], "indexed selected evaluation")
                _require(secondary["raw_self_mse"] is None, "Fresh raw self MSE not saved")
                _validate_scores(secondary["scores"], n)
                _validate_components(secondary["components"], model["original_observation_variance"])
                if secondary["aliases_final"]:
                    for measure in ("weighted", "uniform"):
                        for field in ("posterior_mean_mse", "kl_per_coordinate"):
                            _close(secondary["components"][measure][field]["values"], model["components"][measure][field]["values"],
                                   "selected final deterministic alias " + field, rtol=1e-5, atol=1e-6)
            else:
                _require(secondary is None, "Unexpected secondary/refit endpoint")
            if core is not None and not fresh:
                original = core["jobs"][f"{dataset}_{arm}_{seed}"]
                _same(model["checkpoint_sha256"], original["artifact_sha256"]["best.weights.h5"], "parent selected checkpoint")
                _same(model["step"], original["best_step"], "parent selected step")
        for arm in ARMS:
            for seed in SEEDS:
                original = models[f"{arm}/{seed}/original"]
                refit = models[f"{arm}/{seed}/variance_refit"]
                for field in ("checkpoint_sha256", "step", "original_observation_variance", "variance_fit", "raw_self_mse"):
                    _same(refit[field], original[field], "fixed checkpoint/refit binding")
                _close(original["observation_variance"], models["A/42/original"]["observation_variance"] * (1 if arm in ("A", "C") else 25),
                       "frozen two-variance matrix", atol=0)
                for measure in ("weighted", "uniform"):
                    for field in ("posterior_mean_mse", "posterior_expected_mse", "kl_per_coordinate"):
                        _same(refit["components"][measure][field], original["components"][measure][field], "fixed posterior public summary")
        summaries, contrasts = _aggregate(models)
        _same(record["optimizer_summaries"], summaries, "optimizer summaries")
        _same(record["contrasts"], contrasts, "paired seed contrasts")
        _keys(record["references"], ("low", "high"), "matched references")
        _same(record["references"]["low"]["ppca_fitted"], record["references"]["high"]["ppca_fitted"],
              "shared training-fitted reference is not another MC replicate")
        for label, refs in record["references"].items():
            _keys(refs, ("ppca_matched", "ppca_fitted", "diagonal_matched"), "reference models")
            for name, ref in refs.items():
                _keys(ref, ("noise_variance", "scores", "weighted_log_density_per_coordinate", "uniform_log_density_per_coordinate"), "reference")
                _number(ref["noise_variance"], True)
                if name == "ppca_fitted":
                    _require(ref["noise_variance"] >= 1e-4, "PPCA training variance floor")
                else:
                    _close(ref["noise_variance"], models[f"{'A' if label == 'low' else 'B'}/42/original"]["observation_variance"], "matched reference variance", atol=0)
                _number(ref["weighted_log_density_per_coordinate"])
                _number(ref["uniform_log_density_per_coordinate"])
                if not fresh:
                    _close(ref["weighted_log_density_per_coordinate"], ref["uniform_log_density_per_coordinate"],
                           "uniform target reference density")
                _validate_scores(ref["scores"], n, uniform_target=not fresh)
    _json(value)
    return value
