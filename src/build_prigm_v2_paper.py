"""CPU-only derivation from original and DART-extension PriGM evidence.

No training, neural forward passes, manuscript editing, or release verification
is performed. The schema-1 builder is reused unchanged in a private stage.
Only the selected profile's output set is published, with the manifest last.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

import build_prigm_paper as legacy
from prigm_metrics import energy_score, importance_statistics
from prigm_v2_metrics import (INCREMENT_LAGS, fixed_noise_ppca, increment_energy_score,
                              increment_features, noise_matched_diagonal, weighted_moments,
                              window_spectrum_statistics)
from prigm_v2_experiment import SPEC as NUMERICAL_SPEC

ROOT = Path(__file__).resolve().parents[1]
DATASETS, ARMS, SEEDS = legacy.DATASETS, legacy.ARMS, legacy.SEEDS
WORKERS = tuple(f"worker-{i}" for i in range(4))
STREAM_SEEDS = (42001, 42003, 42004)
EVALUATOR_SOURCES = ("eval_gaussian_final_checkpoints.py", "prigm_v2_metrics.py")
FROZEN_EVALUATOR_SHA256 = {
    "eval_gaussian_final_checkpoints.py": "14dc51defb3c418389081782091b8a40ca8a31cdaa5245703d694e2e0eaf22d6",
    "prigm_v2_metrics.py": "673fe2b482b685dc873d7fd2f08b03e2966b77b62b6663672dd12c19dd40e64a",
}
NUMERICAL_DRIVER = "prigm_canonical_pair_validation.py"
# Archived names -> reviewed repository files. Release integration must check
# these actual bytes separately from the current paper-derivation SOURCE_IDS.
NUMERICAL_REVIEWED_SOURCES = {
    **{f"src/{name}": f"src/{name}" for name in (
        "prigm_v2_model.py", "prigm_v2_experiment.py", "prigm_v2_metrics.py",
        "prigm_v2_data.py", "prigm_metrics.py")},
    NUMERICAL_DRIVER: "runs-v2/" + NUMERICAL_DRIVER,
}
# Producer identities from the verified pre-extension submission, not declarations
# taken from a supplied archive. The DART profile preserves this existing control.
FROZEN_NUMERICAL_SOURCE_SHA256 = {
    "src/prigm_metrics.py": "58ca29a18223dd21231380c81358e4cf9c2c2980246803b0aa8bb485e81c7ea5",
    "src/prigm_v2_data.py": "d89dd3e14068992d1fb148a57e0fddc498bc4d357ff3bfc1bb29dbe96bc13465",
    "src/prigm_v2_experiment.py": "59f40502bd353e5f1c0185121ac636ac200f42bd77efc8e859f3949483507443",
    "src/prigm_v2_metrics.py": "7d5bb09aa6dcb5da1e9c18f3b4b2c55912b9fb1e618bbf981bbe2c75617b7bcf",
    "src/prigm_v2_model.py": "e4ddd0e0381a21fdf50212a157a6c225b5cad83a231e90dd24112ac4875d6125",
    NUMERICAL_DRIVER: "8258c1da62ceee3f90b4cc6a1537f659a541f42206a613d8834a0a4e43e89aa4",
}
NUMERICAL_PREPARED_SHA256 = "03bb770461757c811886a569b74c9e1917750098844ed2e2cfeb5a01720d2101"
NUMERICAL_PAIRS = {"full-low": 0.0625, "full-high": 1.5625}
NUMERICAL_SELF_TEST = {"deliberate_output_bias_mismatch_rejected": True,
                       "original_state_restored_and_verified": True, "comparison_jit_compile": False}
NUMERICAL_COMPARISON = "Exact finite tensor equality after every update, including parameters, Adam states, rates, counters, clipping, and common-draw outputs"
NUMERICAL_SCOPE = "CPU verification checks exact finite NPZ states and trace closure. Common-output equality is attested by the reviewed producer's online checks, not replayed on CPU. HDF5 files are bound by SHA256 and signature only; Keras loading and HDF5 contents are not verified. Smoke attempts do not count as full trajectories. No scientific effect is estimated."


def numerical_state_layout():
    """Storage-3 order from BypassVAE and TF 2.15 Adam.build, without imports.

    0:14 parameters (encoder, posterior heads, decoder, output head); 14 body
    iteration; 15:39 interleaved m/v for 12 body tensors; 39 head iteration;
    40:44 interleaved head m/v; 44:46 rates; 46 scale; 47 clipping-active.
    """
    parameters = [(192, 128), (128,), (128, 64), (64,), (64, 16), (16,),
                  (64, 16), (16,), (16, 64), (64,), (64, 128), (128,), (128, 192), (192,)]
    floating = lambda shapes: [{"shape": list(shape), "dtype": "float32"} for shape in shapes]
    counter = {"shape": [], "dtype": "int64"}
    return (floating(parameters) + [dict(counter)]
            + floating([shape for shape in parameters[:12] for _ in range(2)]) + [dict(counter)]
            + floating([shape for shape in parameters[12:] for _ in range(2)])
            + floating([(), (), ()]) + [{"shape": [], "dtype": "bool"}])


SOURCE_IDS = (*legacy.experiment.CODE_FILES, "src/prigm_refine_evaluation.py",
              "src/build_prigm_paper.py", "src/build_prigm_v2_paper.py", "src/prigm_v2_metrics.py")
AUDIT_OUTPUTS = (
    *(name for name in legacy.OUTPUTS if name != "figures/prior_samples.png"),
    "generated/refinement_rows.tex", "generated/component_rows.tex",
    "generated/dart_diagnostic_rows.tex", "generated/dart_energy_rows.tex", "generated/combined_binding.tex",
    "figures/dart_distribution_diagnostics.png", "figures/dart_support_spectra.png",
)
DART_EXTENSION_SOURCE_IDS = (*SOURCE_IDS, "src/prigm_dart_paper_evidence.py",
                             "src/prigm_v2_model.py", "src/prigm_v2_data.py",
                             "src/prigm_v2_experiment.py", "src/eval_gaussian_final_checkpoints.py")
DART_EXTENSION_OUTPUTS = (
    *(name for name in AUDIT_OUTPUTS if name not in (
        "generated/dart_diagnostic_rows.tex", "generated/dart_energy_rows.tex",
        "figures/dart_distribution_diagnostics.png", "figures/dart_support_spectra.png")),
    "generated/dart_record_rows.tex", "generated/dart_extension_rows.tex",
    "generated/dart_variance_rows.tex", "generated/dart_reference_rows.tex",
    "generated/dart_selected_rows.tex", "generated/dart_block_rows.tex",
    "figures/dart_ecdf_historical.png", "figures/dart_ecdf_fresh.png",
    "figures/dart_spectra_historical.png", "figures/dart_spectra_fresh.png",
    "figures/dart_variance_refit.png",
)
EVIDENCE_NAME = legacy.EVIDENCE_NAME
ARRAY_KEYS = (
    "log_score", "ess", "log_weights", "sample_log_likelihood",
    "sample_logq_minus_logprior", "posterior_expected_nll", "posterior_expected_mse",
    "posterior_mean_mse", "kl", "kl_by_latent", "negative_elbo_analytic_kl",
    "sample_negative_elbo", "jensen_gap",
)
TABLE_METRICS = (
    "posterior_mean_mse", "posterior_expected_mse", "posterior_expected_nll_per_coordinate",
    "kl_per_coordinate", "negative_elbo_analytic_kl_per_coordinate",
    "negative_iw_per_coordinate",
)
STREAM_METRICS = (*TABLE_METRICS, "sample_negative_elbo_per_coordinate",
                  "jensen_gap_per_coordinate", "mean_ess", "median_ess")
REPLAY_KEYS = ("log_score_max_abs", "ess_max_abs", "posterior_mean_mse_max_abs")
RECORD_KEYS = (
    "identifier", "dataset", "arm", "seed", "checkpoint", "checkpoint_sha256",
    "core_completion_sha256", "prepared_sha256", "selected_step", "steps_run",
    "observation_variance", "test_windows", "dimension", "array_shapes", "array_sha256",
    "frozen_evaluator_reproduced_exactly", "core_evaluation_sha256",
    "archived_selected_replay_differences",
)
METHODS = {
    "components": "Uniform test-window arithmetic means, then equal means over three Monte Carlo streams and three optimization fits. Components are not medians and do not decompose a median score. Densities use prepared standardized coordinates, residual likelihood arithmetic, float64 probabilities and float32 networks.",
    "replay": "The exact-reproduction flag attests only the evaluator's same-process comparison with frozen importance_evaluate. Archived selected-checkpoint differences are recomputed and reported, not required to be zero across devices.",
    "support": "Stored normalized training windows have uniform window weights. One-sided within-window DFT bin power uses every positive frequency from 0.5 to 48 cycles per day. Physical power multiplies by training scale squared once. Parseval and the within plus between variance identity are checked. The 1 and 2 cycles per day shares are DFT-bin shares, not tidal constituent attribution.",
    "distribution": "All recorded training and test windows and all 512 saved unconditional draws per optimization fit use training normalization only. Window means are relative to the stored training center, in meters. CDFs use uniform window weights within each split; spectra use uniform mean bin power. Each arm shows the pointwise median and full range of three fit CDFs or spectra, not a confidence interval or IID precision estimate.",
    "decoder_means": "Diagnostic decoder means are reconstructed by subtracting the known observation noise from saved noisy prior samples. NumPy default_rng(43001) first draws (512,16) standard normals cast to float32, then (512,192) standard normals; subtract sqrt(observation_variance) times the latter. Solid colored curves are noisy samples and dashed colored curves reconstructed decoder means. Black solid curves are recorded test windows; black dotted curves are training windows. This arithmetic reconstruction is not new physical cleaning or observed truth.",
    "energy": "Joint energy uses the complete raw-window target and the frozen unbiased off-diagonal sample-pair term. Increment energy uses concatenated normalized lag differences and discards window level; it does not replace joint energy. All saved draws and test windows are retained. Standardized scores multiply by the positive training scale once to give meters. Neural table means average windows then three optimization fits. References use one shared fixed integration stream, not three optimization replicates.",
    "blocks": "Complete test windows are assigned by start to 14 and 28 day blocks from the earliest retained training start, with global mass 1/N per window. Sparse occupied blocks remain. Conditional block means times block masses reconstruct each fit's global mean. Table ranges span all actual conditional block means, including all three neural fits, and are dependent descriptive summaries, not confidence intervals. UTC phase counts are descriptive 15-minute start bins only; historical weights remain uniform.",
}
DART_EXTENSION_METHODS = {
    "ecdf": "Exact inclusive weighted ECDFs use each observed partition's measure and equal generated-draw mass. Mean relative to training center, population SD and lag-normalized increment norm are computed per window in meters before aggregation. Three sampling-stream CDFs are averaged within each fit; pointwise medians and full descriptive seed ranges use three fits, never pooled seeds. Fitted PPCA is one model with three streams. Feature marginals do not identify the full window distribution or dependence between level and shape.",
    "spectra": "Demeaned one-sided DFT bin power, not PSD, uses unchanged 192-point, 900-second window statistics and all positive bins 0.5-48 cycles/day. Stream powers are averaged linearly before seed median/range and log display. Zero power is masked only for log display, never replaced. Saved prior-decoder means are dashed, original noisy samples solid; neither is a posterior reconstruction. Scale is applied once to amplitudes and squared once to powers. Each distribution passes Parseval and within plus between equals pooled variance; median spectra need not integrate to median variance. Bins do not resolve tidal constituents.",
    "variance_refit": "All three seed points show pooled v_hat / v_original and paired joint, centered and increment energy-score changes after the variance reset, in standardized coordinates. Corresponding stream differences precede stream averaging; thin ranges span those three stream differences, not confidence intervals. No new inference or independently reset same-variance comparison is performed.",
    "components": "Original records use selected weights and uniform test-window means; 2018 records use final weights and UTC start-hour weights. Each record's declared test target is then averaged over three Monte Carlo streams and three optimization fits. Components are not medians and do not decompose a median score. Densities use prepared standardized coordinates, residual likelihood arithmetic, float64 probabilities and float32 networks.",
    "replay": METHODS["replay"],
    "dart_extension": "Original selected DART checkpoints retain their original uniform-window target. Fresh final checkpoints use equal mass in six UTC start bins; uniform-window reductions and validation-selected endpoints are secondary. Per-record methods, independent Monte Carlo streams, optimizer-seed contrasts and fixed-function variance diagnostics are specified in dart_extension. Records with different protocols are not pooled.",
    "numerical": "The original four canonical-control trajectories are preserved under independently pinned pre-extension producer hashes. Their complete stored states and trace closure are verified without new Gaussian training. This historical control does not attest to execution of subsequently added DART code; fresh training and diagnostics have separate source identities.",
}
DART_PROTOCOL_FILE = "runs-v2/prigm-dart-dependent-diagnostics-protocol-20260908.json"
DART_PROTOCOL_SHA256 = "a662bb4b95e1ffec13f23262f2166622ba378e949e58a7dec7505c1f210a6bbb"
DART_CORE_SHA256 = "37cc7445600eee541ee50aa580f6fdd6c0f37cf420074dbecc635c8d202946f5"
DART_PROTOCOL = {
    "schema_version": 1, "records": list(DATASETS[1:]), "arms": list(ARMS), "optimization_seeds": list(SEEDS),
    "dimension": 192, "draw_count": 512, "arithmetic_dtype": "float64",
    "array_digest": "sha256_c_order_little_endian_float64",
    "energy_units": "m", "spectrum_units": "m2", "moment_units": "standardized",
    "training_weighting": "uniform_population", "test_weighting": "uniform_windows",
    "baseline_sampling_seed": 45001, "baseline_normal_shape": [512, 192],
    "baseline_sampling": "shared_default_rng_normal_covariance_cholesky", "reference_rank_max": 16,
    "reference_variance_multipliers": [1, 25], "diagonal_floor": "max_population_variance_observation_variance",
    "increment_lags": [1, 4, 16, 48, 96], "increment_block_widths": [191, 188, 176, 144, 96],
    "increment_dimension": 795, "increment_normalization": "each_lag_divided_by_sqrt_width",
    "block_days": [14, 28], "block_origin": "minimum_training_start_unix_s",
    "block_membership": "floor_start_offset_full_windows_no_trimming", "phase_bin_seconds": 900,
    "float64_relative_tolerance": float(128 * np.finfo(np.float64).eps),
    "float64_absolute_tolerance": float(128 * np.finfo(np.float64).eps),
}
DART_REFERENCES = {f"{model}_{level}": (model, multiplier)
                   for model in ("ppca", "diagonal") for level, multiplier in (("low", 1), ("high", 25))}
DART_SCORES = ("joint_m", "increment_m")

_require, _close = legacy._require, legacy._close


def canonical_sha256(value):
    """SHA256 of sorted, compact, finite JSON; generated assets are not inputs."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")).hexdigest()


def _keys(value, expected, label):
    _require(isinstance(value, dict) and set(value) == set(expected), f"Invalid {label} schema")


def _integer(value, label, minimum=1):
    _require(type(value) is int and value >= minimum, f"Invalid integer {label}")
    return value


def _number(value, label, minimum=None):
    _require(type(value) in (int, float) and np.isfinite(value)
             and (minimum is None or value >= minimum), f"Invalid finite {label}")
    return value


def _digests(value, names, label):
    _keys(value, names, label)
    for digest in value.values():
        legacy._digest(digest)


def canonical_jobs():
    return [[d, a, s, "selected"] for d in DATASETS for s in SEEDS for a in ARMS]


def _directory(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input directory")
    _require(path.is_dir(), "Missing input directory")
    return path


def _inventory(directory, names):
    _require({p.name for p in directory.iterdir()} == set(names), "Unexpected or missing artifact inventory")
    for name in names:
        legacy._safe_path(directory, name)


def _variance(arm, base):
    # Frozen arm_settings without importing the TensorFlow model.
    b = 1.0 - 0.8 if arm in ("B", "C") else 1.0
    residual = base * b**2 if arm == "C" else base
    if arm == "D":
        residual /= (1.0 - 0.8)**2
    return residual / b**2


def _core_attempt(root, identifier, evidence):
    completion_path = legacy._safe_path(root, f"jobs/{identifier}/complete.json")
    completion = legacy._read_json(completion_path)
    _require(completion["run_manifest_sha256"] == evidence["run_manifest_sha256"],
             "Core completion belongs to another run")
    attempt = completion["attempt"]
    _require(isinstance(attempt, str) and re.fullmatch(r"attempt-[A-Za-z0-9-]+", attempt),
             "Unsafe core attempt")
    _require(completion["artifact_sha256"] == evidence["jobs"][identifier]["artifact_sha256"],
             "Core completion differs from verified evidence")
    return legacy._safe_path(root, f"jobs/{identifier}/{attempt}"), completion_path


def component_statistics(arrays, n, dimension, variance):
    """Validate same-draw identities and reduce arrays, never scalar receipts."""
    _integer(n, "test windows")
    _require(dimension == 192 and type(dimension) is int, "Invalid component dimension")
    _number(variance, "observation variance", 0)
    _require(variance > 0, "Nonpositive observation variance")
    _keys(arrays, ARRAY_KEYS, "component array")
    for key, array in arrays.items():
        shape = (3, n, 256) if key in ARRAY_KEYS[2:5] else (
            (3, n, 16) if key == "kl_by_latent" else (3, n))
        legacy._array(array, shape, key)
        _require(array.dtype == np.dtype("float64"), "Component arrays must be float64")
    for key in ("posterior_mean_mse", "posterior_expected_mse", "kl", "kl_by_latent"):
        _require(np.all(arrays[key] >= 0), f"Negative component: {key}")
    for key in ("posterior_mean_mse", "kl", "kl_by_latent"):
        _require(all(np.array_equal(arrays[key][0], arrays[key][i]) for i in (1, 2)),
                 f"Shared analytic component differs across streams: {key}")
    logw = arrays["log_weights"]
    _close(logw, arrays["sample_log_likelihood"] - arrays["sample_logq_minus_logprior"], "log weights")
    nll = -arrays["sample_log_likelihood"].mean(axis=2)
    _close(arrays["posterior_expected_nll"], nll, "posterior NLL")
    _close(arrays["posterior_expected_mse"],
           2 * variance * (nll / dimension - 0.5 * np.log(2 * np.pi * variance)), "expected MSE units")
    kl = arrays["kl_by_latent"].sum(axis=2)
    _close(arrays["kl"], kl, "analytic KL sum")
    _close(arrays["negative_elbo_analytic_kl"], nll + kl, "analytic negative ELBO")
    _close(arrays["sample_negative_elbo"], -logw.mean(axis=2), "sample negative ELBO")
    _close(arrays["jensen_gap"], arrays["log_score"] - logw.mean(axis=2), "Jensen gap")
    _require(np.all(arrays["jensen_gap"] >= -1e-10), "Negative Jensen gap")
    streams = []
    for i, seed in enumerate(STREAM_SEEDS):
        stats = importance_statistics(logw[i])
        for key in ("log_score", "ess"):
            _close(arrays[key][i], stats[key], key)
        mean = lambda key: float(arrays[key][i].mean())
        streams.append({
            "stream_seed": seed,
            "posterior_mean_mse": mean("posterior_mean_mse"),
            "posterior_expected_mse": mean("posterior_expected_mse"),
            "posterior_expected_nll_per_coordinate": float(nll[i].mean() / dimension),
            "kl_per_coordinate": float(kl[i].mean() / dimension),
            "negative_elbo_analytic_kl_per_coordinate": float((nll[i] + kl[i]).mean() / dimension),
            "negative_iw_per_coordinate": -float(stats["log_score"].mean() / dimension),
            "sample_negative_elbo_per_coordinate": -float(logw[i].mean() / dimension),
            "jensen_gap_per_coordinate": mean("jensen_gap") / dimension,
            "mean_ess": float(stats["ess"].mean()), "median_ess": float(np.median(stats["ess"])),
            "mean_kl_by_latent": arrays["kl_by_latent"][i].mean(axis=0).tolist(),
        })
    return streams


def _launcher_sources(root, sources, core_metrics_sha256):
    """Verify the optional launcher and its frozen metrics dependency."""
    directory = legacy._safe_path(root, "src")
    if not directory.exists():
        return
    _directory(directory)
    names = {p.name for p in directory.iterdir()}
    _require(names in (set(EVALUATOR_SOURCES), {*EVALUATOR_SOURCES, "prigm_metrics.py"}),
             "Unexpected launcher source inventory")
    expected = dict(sources)
    if "prigm_metrics.py" in names:
        expected["prigm_metrics.py"] = core_metrics_sha256
    legacy._verify_archive(directory, expected)


def verify_components(components_root, core_root, core_evidence):
    """Return (sanitized evidence, private arrays) for exactly 60 selected jobs.

    core_evidence must come from legacy.verify_evidence. This lower-level seam
    also accepts explicitly synthetic closed evidence in CPU unit tests; the
    production build always runs the genuine legacy closure first.
    """
    root, core = _directory(components_root), _directory(core_root)
    _require({p.name for p in root.iterdir()} in (set(WORKERS), {*WORKERS, "src"}),
             "Expected exactly worker-0 through worker-3 and optional src")
    legacy._index_jobs(list(core_evidence["jobs"].values()))
    _require(core_evidence["schema_version"] == 1 and core_evidence["purpose"] == "core"
             and core_evidence["specification"] == legacy.experiment.specification("core"),
             "Expected verified frozen core evidence")
    pin = legacy._digest(core_evidence["run_manifest_sha256"])
    _require(legacy.experiment.sha256(legacy._safe_path(core, "run_manifest.json")) == pin,
             "Core manifest changed")
    archive_path = legacy._safe_path(core, "archive_manifest.json")
    archive_sha = legacy._digest(core_evidence["archive_manifest_sha256"])
    _require(legacy.experiment.sha256(archive_path) == archive_sha, "Core archive identity changed")
    core_artifacts = legacy._read_json(archive_path)["artifact_sha256"]
    sources = dict(FROZEN_EVALUATOR_SHA256)
    _launcher_sources(root, sources, core_evidence["code_sha256"]["src/prigm_metrics.py"])
    jobs, private, workers = {}, {}, {}
    snapshots = [(archive_path, archive_sha), (core / "run_manifest.json", pin)]
    inventories = [(root, {p.name for p in root.iterdir()})]
    prepared = {}
    for dataset in DATASETS:
        name = dataset + ".npz"
        path = legacy._safe_path(core, "prepared/" + name)
        expected = core_evidence["prepared_sha256"][name]
        _require(legacy.experiment.sha256(path) == expected, "Prepared input changed")
        prepared[dataset] = legacy._load_npz(path)
        snapshots.append((path, expected))
    for index, worker_name in enumerate(WORKERS):
        worker = _directory(legacy._safe_path(root, worker_name))
        full_jobs, owned = canonical_jobs(), canonical_jobs()[index::4]
        artifact_names = {legacy.experiment.job_id(*job[:3]) + "-selected" + suffix
                          for job in owned for suffix in (".npz", ".json")}
        _inventory(worker, {*artifact_names, "source", "protocol.json", "complete.json"})
        source = _directory(legacy._safe_path(worker, "source"))
        _inventory(source, EVALUATOR_SOURCES)
        inventories.extend([(worker, {*artifact_names, "source", "protocol.json", "complete.json"}),
                            (source, set(EVALUATOR_SOURCES))])
        legacy._verify_archive(source, sources)
        protocol_path, complete_path = worker / "protocol.json", worker / "complete.json"
        protocol_sha = legacy.experiment.sha256(protocol_path)
        complete_sha = legacy.experiment.sha256(complete_path)
        protocol = legacy._read_json(protocol_path)
        expected_protocol = {
            "schema_version": 1, "purpose": "frozen_checkpoint_components", "core_manifest_sha256": pin,
            "evaluator_source_sha256": sources, "worker_index": index, "worker_count": 4,
            "jobs": owned, "all_jobs": full_jobs, "samples": 256, "stream_seeds": list(STREAM_SEEDS),
            "density_coordinates": "residual", "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32,
        }
        _keys(protocol, {*expected_protocol, "runtime"}, "worker protocol")
        _require(isinstance(protocol["runtime"], dict), "Invalid private runtime receipt")
        _require(canonical_sha256({k: v for k, v in protocol.items() if k != "runtime"})
                 == canonical_sha256(expected_protocol), "Worker protocol differs")
        complete = legacy._read_json(complete_path)
        _keys(complete, ("schema_version", "purpose", "protocol_sha256", "core_manifest_sha256",
                         "evaluator_source_sha256", "artifact_sha256"), "worker completion")
        _require(type(complete["schema_version"]) is int and complete["schema_version"] == 1
                 and complete["purpose"] == expected_protocol["purpose"]
                 and complete["core_manifest_sha256"] == pin
                 and complete["evaluator_source_sha256"] == sources, "Worker completion identity differs")
        _require(complete["protocol_sha256"] == protocol_sha, "Worker protocol SHA256 differs")
        _digests(complete["artifact_sha256"], artifact_names, "worker artifacts")
        legacy._verify_archive(worker, complete["artifact_sha256"])
        workers[worker_name] = {"completion_sha256": complete_sha, "protocol_sha256": protocol_sha}
        snapshots.extend([(complete_path, workers[worker_name]["completion_sha256"]),
                          (protocol_path, protocol_sha)])
        snapshots.extend((source / name, digest) for name, digest in sources.items())
        snapshots.extend((worker / name, digest) for name, digest in complete["artifact_sha256"].items())
        for dataset, arm, seed, label in owned:
            identifier = legacy.experiment.job_id(dataset, arm, seed)
            _require(identifier not in jobs, "Duplicate component job")
            stem = identifier + "-" + label
            record = legacy._read_json(worker / (stem + ".json"))
            _keys(record, RECORD_KEYS, "component record")
            original = core_evidence["jobs"][identifier]
            attempt, completion_path = _core_attempt(core, identifier, core_evidence)
            hashes = original["artifact_sha256"]
            for name in ("best.weights.h5", "evaluation.npz"):
                path = legacy._safe_path(attempt, name)
                _require(legacy.experiment.sha256(path) == hashes[name], "Core job input changed")
                snapshots.append((path, hashes[name]))
            completion_sha = legacy.experiment.sha256(completion_path)
            _require(core_artifacts.get(f"jobs/{identifier}/complete.json") == completion_sha,
                     "Core completion differs from verified archive")
            snapshots.append((completion_path, completion_sha))
            data = prepared[dataset]
            n, dimension = data["test"].shape
            _require(n == core_evidence["dataset_statistics"][dataset]["windows"]["test"]
                     and dimension == 192, "Prepared evaluation dimensions differ")
            base = float(data["base_variance"])
            _close(base, core_evidence["dataset_statistics"][dataset]["base_variance"], "base variance")
            variance = _variance(arm, base)
            expected_record = {
                "identifier": identifier, "dataset": dataset, "arm": arm, "seed": seed, "checkpoint": "selected",
                "checkpoint_sha256": hashes["best.weights.h5"], "core_completion_sha256": completion_sha,
                "prepared_sha256": core_evidence["prepared_sha256"][dataset + ".npz"],
                "selected_step": original["best_step"], "steps_run": original["steps_run"],
                "test_windows": n, "dimension": dimension, "frozen_evaluator_reproduced_exactly": True,
                "core_evaluation_sha256": hashes["evaluation.npz"],
                "array_sha256": complete["artifact_sha256"][stem + ".npz"],
            }
            _require(canonical_sha256({k: record[k] for k in expected_record})
                     == canonical_sha256(expected_record), "Component record/core identity differs")
            _number(record["observation_variance"], "observation variance", 0)
            _close(record["observation_variance"], variance, "observation variance")
            arrays = legacy._load_npz(worker / (stem + ".npz"))
            statistics = component_statistics(arrays, n, dimension, variance)
            _require(canonical_sha256(record["array_shapes"])
                     == canonical_sha256({k: list(v.shape) for k, v in arrays.items()}), "Recorded array shapes differ")
            saved = legacy._load_npz(attempt / "evaluation.npz")
            legacy._array(saved["posterior_mean"], (n, dimension), "saved posterior mean")
            _close(saved["corrected_mse"], np.square(data["test"] - saved["posterior_mean"]).mean(axis=1),
                   "archived posterior mean MSE")
            differences = {}
            for key, archived_key, report_key in zip(("log_score", "ess", "posterior_mean_mse"),
                                                    ("log_score", "ess", "corrected_mse"), REPLAY_KEYS):
                legacy._array(saved[archived_key], (n,), archived_key)
                differences[report_key] = float(np.max(np.abs(arrays[key][0] - saved[archived_key])))
            _keys(record["archived_selected_replay_differences"], REPLAY_KEYS, "archived replay")
            for key, value in differences.items():
                _number(record["archived_selected_replay_differences"][key], key, 0)
                _close(record["archived_selected_replay_differences"][key], value, key)
            jobs[identifier] = {**expected_record, "observation_variance": variance,
                                "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
                                "archived_selected_replay_differences": differences, "streams": statistics}
            private[identifier] = arrays
    _require(len(jobs) == 60, "Incomplete component job matrix")
    for path, digest in snapshots:
        _require(legacy.experiment.sha256(path) == digest, "Input changed during component verification")
    for directory, names in inventories:
        _inventory(directory, names)
    _launcher_sources(root, sources, core_evidence["code_sha256"]["src/prigm_metrics.py"])
    return {
        "schema_version": 1, "purpose": "frozen_checkpoint_components", "job_count": 60, "worker_count": 4,
        "samples": 256, "stream_seeds": list(STREAM_SEEDS), "density_coordinates": "residual",
        "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32,
        "evaluator_source_sha256": sources, "workers": workers, "jobs": jobs,
    }, private


def _numerical_steps(mode):
    return [0, *range(1000, 10001, 1000)] if mode == "full" else [0, 64]


def _numerical_artifacts(mode):
    return ("protocol.json", "runtime.json", "comparator_self_test.json", "trace.jsonl",
            "base.weights.h5", "canonical.weights.h5",
            *(f"state-{step:05d}.npz" for step in _numerical_steps(mode)))


def _numerical_snapshot(root, profile="corrected_audit"):
    """Closed private inventory, including smoke guards; never a public payload."""
    _require(profile in ("corrected_audit", "dart_extension"), "Unknown numerical source policy")
    _require(".." not in Path(root).parts, "Unsafe numerical root")
    root = _directory(root)
    _inventory(root, ("src", "prepared", NUMERICAL_DRIVER,
                      "smoke-low", "smoke-high", *NUMERICAL_PAIRS))
    _inventory(_directory(root / "src"), [Path(name).name for name in NUMERICAL_REVIEWED_SOURCES if name.startswith("src/")])
    _inventory(_directory(root / "prepared"), ("gaussian_v2.npz",))
    paths = [root / name for name in NUMERICAL_REVIEWED_SOURCES]
    paths.append(root / "prepared/gaussian_v2.npz")
    for name in ("smoke-low", "smoke-high", *NUMERICAL_PAIRS):
        directory = _directory(root / name)
        artifacts = (*_numerical_artifacts(name.split("-")[0]), "complete.json")
        _inventory(directory, artifacts)
        paths.extend(directory / artifact for artifact in artifacts)
    # The default profile binds current originals; the DART profile instead
    # checks literal historical producer pins and binds its own current sources.
    if profile == "corrected_audit":
        paths.extend(legacy._safe_path(ROOT, name) for name in NUMERICAL_REVIEWED_SOURCES.values())
    return {path: legacy.experiment.sha256(path) for path in paths}


def _numerical_inputs(root):
    digest = lambda name: legacy.experiment.sha256(legacy._safe_path(root, name))
    return {"pairs": {name: {"protocol_sha256": digest(name + "/protocol.json"),
                             "completion_sha256": digest(name + "/complete.json")}
                      for name in NUMERICAL_PAIRS},
            "source_sha256": {name: digest(name) for name in NUMERICAL_REVIEWED_SOURCES},
            "prepared_sha256": digest("prepared/gaussian_v2.npz")}


def _numerical_protocol(mode, variance, sources, prepared):
    return {"schema_version": 1, "purpose": "canonical_pair_numerical_validation",
            "mode": mode, "conditions": ["base", "canonical"], "seed": 42,
            "observation_variance": variance, "specification": deepcopy(NUMERICAL_SPEC),
            "planned_updates_per_trajectory": 10000 if mode == "full" else 64,
            "full_training_trajectories_in_attempt": 2 if mode == "full" else 0,
            "short_validation_trajectories_in_attempt": 0 if mode == "full" else 2,
            "absolute_state_tolerance": 0.0, "relative_state_tolerance": 0.0,
            "comparison": NUMERICAL_COMPARISON, "source_sha256": dict(sources), "prepared_sha256": prepared,
            "maximum_elapsed_seconds": 1800, "snapshot_steps": _numerical_steps(mode),
            "restart_only": True, "scientific_effect_estimation": False}


def _same_json(actual, expected, label):
    _require(canonical_sha256(actual) == canonical_sha256(expected), f"Invalid numerical {label}")


def _utc(value):
    _require(isinstance(value, str), "Invalid numerical UTC timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid numerical UTC timestamp") from error
    _require(result.utcoffset() == timezone.utc.utcoffset(result), "Numerical timestamp is not UTC")
    return result


def _numerical_attestation(receipt):
    _keys(receipt, ("gpu_uuid", "pid", "placement", "process_inventory", "tensorflow", "python", "numpy",
                    "deterministic_ops_requested", "tf32_enabled"), "numerical GPU attestation")
    _require(isinstance(receipt["gpu_uuid"], str) and re.fullmatch(
        r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", receipt["gpu_uuid"]), "Invalid GPU UUID")
    pid = _integer(receipt["pid"], "GPU process")
    _require(receipt["placement"] == "/job:localhost/replica:0/task:0/device:GPU:0",
             "Invalid numerical GPU placement")
    inventory = receipt["process_inventory"]
    _require(isinstance(inventory, list) and len(inventory) == 1 and isinstance(inventory[0], str),
             "Invalid numerical GPU process inventory")
    fields = [part.strip() for part in inventory[0].split(",")]
    _require(len(fields) == 4 and fields[:2] == [str(pid), receipt["gpu_uuid"]]
             and bool(fields[2]) and re.fullmatch(r"[0-9]+ MiB", fields[3]), "GPU process attestation differs")
    _require(isinstance(receipt["python"], str) and bool(receipt["python"]), "Missing private Python version")
    public = {key: receipt[key] for key in ("tensorflow", "numpy", "deterministic_ops_requested", "tf32_enabled")}
    _validate_numerical_attestation(public)
    return public


def _validate_numerical_attestation(public):
    _keys(public, ("tensorflow", "numpy", "deterministic_ops_requested", "tf32_enabled"), "public GPU attestation")
    _require(isinstance(public["tensorflow"], str) and re.fullmatch(r"2\.15\.[0-9]+", public["tensorflow"]),
             "Numerical state order requires TensorFlow 2.15")
    _require(isinstance(public["numpy"], str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", public["numpy"]),
             "Invalid numerical NumPy version")
    _require(public["deterministic_ops_requested"] is True and type(public["tf32_enabled"]) is bool,
             "Invalid numerical determinism attestation")


def _float32(value, label, minimum=None):
    value = _number(value, label, minimum)
    _require(abs(value) <= np.finfo(np.float32).max and float(np.float32(value)) == value,
             f"Invalid float32 numerical {label}")
    return value


def _clip_scale(norm):
    return float(np.float32(10) / max(np.float32(norm), np.float32(10)))


def _numerical_trace(path, steps):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key in numerical trace")
            result[key] = value
        return result

    def invalid(text):
        raise ValueError("Nonfinite JSON constant in numerical trace")

    rows = []
    with path.open(encoding="utf-8") as stream:
        for step, line in enumerate(stream, 1):
            _require(step <= steps and line.endswith("\n"), "Truncated or extra numerical trace")
            row = json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)
            _keys(row, ("step", "pre_update_loss", "pre_update_kl", "gradient_norm", "clipping_scale",
                        "clipping_active", "all_state_equal", "checked_state_tensors_per_model"), "numerical trace row")
            _require(_integer(row["step"], "trace step") == step, "Nonsequential numerical trace")
            _require(row["all_state_equal"] is True
                     and _integer(row["checked_state_tensors_per_model"], "checked states") == 48,
                     "Numerical trace lacks exact 48-state comparison")
            for key in ("pre_update_loss", "pre_update_kl", "gradient_norm", "clipping_scale"):
                _float32(row[key], key, 0 if key in ("gradient_norm", "clipping_scale") else None)
            scale = row["clipping_scale"]
            _require(0 < scale <= 1 and scale == _clip_scale(row["gradient_norm"])
                     and row["clipping_active"] is (scale < 1), "Invalid numerical clipping reduction")
            rows.append(row)
    _require(len(rows) == steps, "Truncated numerical trace")
    return rows


def _numerical_states(path, step, row):
    arrays = legacy._load_npz(path)
    layout = numerical_state_layout()
    _keys(arrays, [f"{condition}_state_{index:02d}" for condition in ("base", "canonical")
                   for index in range(48)], "numerical snapshot")
    for index, spec in enumerate(layout):
        left, right = (arrays[f"{condition}_state_{index:02d}"] for condition in ("base", "canonical"))
        for array in (left, right):
            _require(array.shape == tuple(spec["shape"]) and array.dtype == np.dtype(spec["dtype"])
                     and np.isfinite(array).all(), f"Invalid numerical state shape/dtype/finiteness at {index}")
        _require(np.array_equal(left, right), "Nonzero base/canonical state difference")
        if index in (14, 39):
            _require(left.item() == step, "Numerical optimizer counter differs from snapshot step")
        elif index in (44, 45):
            _require(left.item() == float(np.float32(0.001)), "Numerical learning rate differs")
        elif index == 46:
            _require(left.item() == (row["clipping_scale"] if step else 1.0), "Snapshot clipping scale differs")
        elif index == 47:
            _require(left.item() is (row["clipping_active"] if step else False), "Snapshot clipping flag differs")
        elif step == 0 and 15 <= index <= 43:
            _require(np.all(left == 0), "Nonzero initial Adam moment")


def _clipping_summary(rows):
    return {"active_updates": sum(row["clipping_active"] for row in rows),
            "minimum_scale": min(row["clipping_scale"] for row in rows),
            "maximum_scale": max(row["clipping_scale"] for row in rows),
            "minimum_gradient_norm": min(row["gradient_norm"] for row in rows),
            "maximum_gradient_norm": max(row["gradient_norm"] for row in rows)}


def _verify_numerical_pair(directory, mode, variance, sources, prepared):
    protocol, complete = (legacy._read_json(directory / name) for name in ("protocol.json", "complete.json"))
    expected = _numerical_protocol(mode, variance, sources, prepared)
    _keys(protocol, (*expected, "started_utc", "launch_cutoff_utc"), "numerical protocol")
    _same_json({key: protocol[key] for key in expected}, expected, "protocol")
    started = _utc(protocol["started_utc"])
    if mode == "full" or protocol["launch_cutoff_utc"] is not None:
        _require(started <= _utc(protocol["launch_cutoff_utc"]), "Numerical launch exceeded UTC cutoff")
    _keys(complete, ("schema_version", "purpose", "mode", "trajectories", "verified_updates_per_trajectory",
                     "exact_state_equality", "elapsed_seconds", "protocol_sha256", "artifact_sha256"), "numerical completion")
    steps = expected["planned_updates_per_trajectory"]
    fixed = {"schema_version": 1, "purpose": expected["purpose"], "mode": mode, "trajectories": 2,
             "verified_updates_per_trajectory": steps, "exact_state_equality": True}
    _same_json({key: complete[key] for key in fixed}, fixed, "completion")
    _require(0 < _number(complete["elapsed_seconds"], "elapsed seconds") <= 1800, "Numerical elapsed limit exceeded")
    artifacts = complete["artifact_sha256"]
    _digests(artifacts, _numerical_artifacts(mode), "numerical artifacts")
    _require(complete["protocol_sha256"] == artifacts["protocol.json"], "Numerical protocol identity differs")
    legacy._verify_archive(directory, artifacts)
    attestation = _numerical_attestation(legacy._read_json(directory / "runtime.json"))
    self_test = legacy._read_json(directory / "comparator_self_test.json")
    _same_json(self_test, NUMERICAL_SELF_TEST, "comparator self-test")
    rows = _numerical_trace(directory / "trace.jsonl", steps)
    for step in expected["snapshot_steps"]:
        _numerical_states(directory / f"state-{step:05d}.npz", step, rows[step - 1] if step else None)
    for condition in ("base", "canonical"):
        with (directory / f"{condition}.weights.h5").open("rb") as stream:
            _require(stream.read(8) == b"\x89HDF\r\n\x1a\n", "Invalid HDF5 signature")
    return {"protocol": expected, "protocol_sha256": complete["protocol_sha256"],
            "completion_sha256": legacy.experiment.sha256(directory / "complete.json"),
            "artifact_sha256": artifacts, "gpu_attestation": attestation,
            "launch_cutoff_enforced": mode == "full", "elapsed_seconds": complete["elapsed_seconds"],
            "verified_updates_per_trajectory": steps, "trace_rows": len(rows),
            "checked_state_tensors_per_model": 48, "npz_snapshot_count": len(expected["snapshot_steps"]),
            "exact_state_equality": True, "max_absolute_state_difference": 0.0,
            "max_relative_state_difference": 0.0, "online_common_output_checks_attested": True,
            "comparator_self_test": self_test, "clipping": _clipping_summary(rows)}


def verify_numerical_validation(root, *, profile="corrected_audit"):
    """Verify existing trajectories against an independently reviewed producer.

    The default retains current-source matching. The DART extension preserves
    the exact previously verified producer, without adding Gaussian training.
    Neither policy accepts source identities supplied solely by the archive.
    """
    _require(profile in ("corrected_audit", "dart_extension"), "Unknown numerical source policy")
    initial = _numerical_snapshot(root, profile=profile)
    root = _directory(root)
    inputs = _numerical_inputs(root)
    expected_sources = (FROZEN_NUMERICAL_SOURCE_SHA256 if profile == "dart_extension" else
                        {archived: initial[legacy._safe_path(ROOT, reviewed)]
                         for archived, reviewed in NUMERICAL_REVIEWED_SOURCES.items()})
    _same_json(inputs["source_sha256"], expected_sources, "Unreviewed numerical source identity")
    _require(inputs["prepared_sha256"] == NUMERICAL_PREPARED_SHA256, "Invalid numerical prepared identity")
    pairs = {}
    for name in ("smoke-low", "smoke-high", *NUMERICAL_PAIRS):
        mode, label = name.split("-")
        record = _verify_numerical_pair(root / name, mode, NUMERICAL_PAIRS["full-" + label],
                                        inputs["source_sha256"], inputs["prepared_sha256"])
        if mode == "full":
            pairs[name] = record
    public = {"schema_version": 1, "purpose": "canonical_pair_numerical_validation",
              "full_training_trajectories": 4, "source_sha256": inputs["source_sha256"],
              "prepared_sha256": inputs["prepared_sha256"], "state_layout": numerical_state_layout(),
              "pairs": pairs, "verification_scope": NUMERICAL_SCOPE}
    validate_numerical_validation(public, inputs)
    _require(_numerical_snapshot(root, profile=profile) == initial, "Numerical inputs changed during verification")
    return public


def validate_numerical_validation(value, input_index=None):
    """Pure public gate. Mutual consistency only; private closure checks bytes.

    Release verifiers compare source_sha256 with current reviewed bytes for
    schema 2, or the independently pinned historical producer for schema 3.
    No source files, hosts or artifacts are read here.
    """
    _public_tree(value)
    _keys(value, ("schema_version", "purpose", "full_training_trajectories", "source_sha256", "prepared_sha256",
                  "state_layout", "pairs", "verification_scope"), "numerical validation")
    _same_json({key: value[key] for key in ("schema_version", "purpose", "full_training_trajectories")},
               {"schema_version": 1, "purpose": "canonical_pair_numerical_validation", "full_training_trajectories": 4},
               "public trajectory count")
    _digests(value["source_sha256"], NUMERICAL_REVIEWED_SOURCES, "numerical source identities")
    _require(value["prepared_sha256"] == NUMERICAL_PREPARED_SHA256, "Invalid numerical prepared identity")
    _same_json(value["state_layout"], numerical_state_layout(), "state layout")
    _require(value["verification_scope"] == NUMERICAL_SCOPE, "Invalid numerical verification scope")
    _keys(value["pairs"], NUMERICAL_PAIRS, "numerical full pairs")
    identities = {}
    for name, variance in NUMERICAL_PAIRS.items():
        pair = value["pairs"][name]
        fixed = {"launch_cutoff_enforced": True, "verified_updates_per_trajectory": 10000, "trace_rows": 10000,
                 "checked_state_tensors_per_model": 48, "npz_snapshot_count": 11, "exact_state_equality": True,
                 "max_absolute_state_difference": 0.0, "max_relative_state_difference": 0.0,
                 "online_common_output_checks_attested": True, "comparator_self_test": NUMERICAL_SELF_TEST}
        _keys(pair, (*fixed, "protocol", "protocol_sha256", "completion_sha256", "artifact_sha256",
                     "gpu_attestation", "elapsed_seconds", "clipping"), "public numerical pair")
        _same_json({key: pair[key] for key in fixed}, fixed, "public exact comparisons")
        _same_json(pair["protocol"], _numerical_protocol("full", variance, value["source_sha256"], value["prepared_sha256"]),
                   "public protocol")
        _digests(pair["artifact_sha256"], _numerical_artifacts("full"), "public numerical artifacts")
        identities[name] = {key: legacy._digest(pair[key]) for key in ("protocol_sha256", "completion_sha256")}
        _require(pair["protocol_sha256"] == pair["artifact_sha256"]["protocol.json"], "Numerical protocol identity differs")
        _require(0 < _number(pair["elapsed_seconds"], "elapsed seconds") <= 1800, "Numerical elapsed limit exceeded")
        _validate_numerical_attestation(pair["gpu_attestation"])
        clip = pair["clipping"]
        _keys(clip, ("active_updates", "minimum_scale", "maximum_scale", "minimum_gradient_norm", "maximum_gradient_norm"),
              "public numerical clipping")
        count = _integer(clip["active_updates"], "clipped updates", 0)
        for key in ("minimum_scale", "maximum_scale", "minimum_gradient_norm", "maximum_gradient_norm"):
            _float32(clip[key], key, 0)
        low, high = clip["minimum_scale"], clip["maximum_scale"]
        _require(count <= 10000 and 0 < low <= high <= 1
                 and clip["minimum_gradient_norm"] <= clip["maximum_gradient_norm"]
                 and low == _clip_scale(clip["maximum_gradient_norm"])
                 and high == _clip_scale(clip["minimum_gradient_norm"])
                 and (count == 0) == (low == 1) and (count == 10000) == (high < 1),
                 "Invalid public numerical clipping reduction")
    if input_index is not None:
        _same_json(input_index, {"pairs": identities, "source_sha256": value["source_sha256"],
                                 "prepared_sha256": value["prepared_sha256"]}, "input identities")
    return value


def _dart_close(actual, expected, label):
    actual = legacy._array(actual, np.shape(expected), label)
    _require(np.allclose(actual, expected, rtol=DART_PROTOCOL["float64_relative_tolerance"],
                        atol=DART_PROTOCOL["float64_absolute_tolerance"]),
             f"DART float64 reduction differs: {label}")


def _dart_array_digest(array):
    array = np.asarray(array, dtype="<f8")
    _require(np.isfinite(array).all(), "Nonfinite DART derived array")
    return sha256(array.tobytes(order="C")).hexdigest()


def _dart_blocks(starts, origin):
    """Occupied blocks only; every full window keeps its original global mass."""
    starts = np.asarray(starts)
    n = len(starts)
    result = {}
    for days in DART_PROTOCOL["block_days"]:
        membership = np.floor_divide(starts - origin, days * 86400).astype(np.int64)
        ids, counts = np.unique(membership, return_counts=True)
        result[str(days)] = {"block_ids": ids.tolist(), "window_block_ids": membership.tolist(),
                             "counts": counts.tolist(), "mass": (counts / n).tolist()}
    return result


def _dart_score_summary(joint, increment, blocks):
    """Keep fit-by-window values and their mass-weighted block reductions."""
    arrays = {"joint_m": np.asarray(joint), "increment_m": np.asarray(increment)}
    result = {"per_window": {}, "per_fit_mean": {}, "mean": {}, "blocks": {}}
    for key, values in arrays.items():
        _require(values.ndim == 2 and min(values.shape) > 0 and np.isfinite(values).all(), "Invalid DART scores")
        result["per_window"][key] = values.tolist()
        result["per_fit_mean"][key] = values.mean(axis=1).tolist()
        result["mean"][key] = float(values.mean(axis=1).mean())
    for days, block in blocks.items():
        membership, mass = np.asarray(block["window_block_ids"]), np.asarray(block["mass"])
        result["blocks"][days] = {}
        for key, values in arrays.items():
            conditional = np.stack([values[:, membership == i].mean(axis=1) for i in block["block_ids"]], axis=1)
            contribution = conditional * mass
            reconstructed = contribution.sum(axis=1)
            _dart_close(reconstructed, values.mean(axis=1), "weighted block reconstruction")
            result["blocks"][days][key] = {
                "conditional_mean": conditional.tolist(), "weighted_contribution": contribution.tolist(),
                "reconstructed_global_mean": reconstructed.tolist(),
                "range": [float(conditional.min()), float(conditional.max())],
            }
    return result


def _dart_reductions(actual, expected, label):
    """Strict keys and shapes, float64 arithmetic tolerance at numeric leaves."""
    if isinstance(expected, dict):
        _keys(actual, expected, label)
        for key in expected:
            _dart_reductions(actual[key], expected[key], label + " " + key)
    else:
        _dart_close(actual, expected, label)


def _dart_phase_counts(starts):
    return np.bincount((np.asarray(starts, dtype=np.int64) % 86400) // 900, minlength=96).tolist()


def _dart_plot_values(stats):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stats.items()}


def physical_window_statistics(windows, scale):
    """Stored normalized coordinates in; uniform weights and physical units out."""
    windows = np.asarray(windows, dtype=np.float64)
    _require(windows.ndim == 2 and windows.shape[0] >= 2 and windows.shape[1] == 192,
             "DART diagnostics need at least two 192-coordinate windows")
    _number(scale, "training scale", 0)
    _require(scale > 0, "Nonpositive training scale")
    stats = window_spectrum_statistics(windows, np.ones(len(windows), dtype=np.float64))
    _close(stats["pooled_bin_power"].sum(), stats["within_window_variance"], "Parseval")
    _close(stats["within_window_variance"] + stats["between_window_mean_variance"],
           stats["pooled_entry_variance"], "within plus between variance")
    factor = scale**2
    return {
        "window_count": len(windows), "frequency_cycles_per_day": stats["frequency_cycles_per_day"],
        "bin_power_m2": stats["pooled_bin_power"] * factor,
        "within_variance_m2": stats["within_window_variance"] * factor,
        "between_variance_m2": stats["between_window_mean_variance"] * factor,
        "pooled_variance_m2": stats["pooled_entry_variance"] * factor,
        "within_to_pooled_ratio": stats["within_to_pooled_ratio"],
        "window_mean_relative_training_center_m": windows.mean(axis=1) * scale,
        "within_window_sd_m": windows.std(axis=1) * scale,
        "increment_magnitude_m": np.linalg.norm(increment_features(windows), axis=1) * scale,
    }


def reconstruct_decoder_means(prior_samples, observation_variance):
    """Undo only the exactly specified frozen observation-noise draw sequence."""
    legacy._array(prior_samples, (512, 192), "saved unconditional draws")
    _number(observation_variance, "observation variance", 0)
    _require(observation_variance > 0, "Nonpositive observation variance")
    rng = np.random.default_rng(43001)
    rng.standard_normal((512, 16)).astype(np.float32)
    noise = rng.standard_normal((512, 192))
    return prior_samples - np.sqrt(observation_variance) * noise


def dart_diagnostics(core_root, evidence):
    """Return public support, strict descriptive evidence, and plotting inputs.

    The caller supplies verified core evidence. No model or test-fitted transform
    is used. Source/core hashes are rechecked by the build's transitive closure.
    """
    root = _directory(core_root)
    _require(tuple(DART_PROTOCOL["increment_lags"]) == INCREMENT_LAGS, "Increment helper protocol differs")
    support, records, private = {}, {}, {}
    normal = np.random.default_rng(45001).standard_normal((512, 192))
    for dataset in DATASETS[1:]:
        path = legacy._safe_path(root, f"prepared/{dataset}.npz")
        _require(legacy.experiment.sha256(path) == evidence["prepared_sha256"][dataset + ".npz"],
                 "DART prepared input changed")
        data = legacy._load_npz(path)
        for split in ("train", "test"):
            n = evidence["dataset_statistics"][dataset]["windows"][split]
            legacy._array(data[split], (n, 192), "DART " + split)
            _require(data[split].dtype == np.dtype("float64"), "DART windows must be float64")
            starts = legacy._array(data["starts_" + split], (n,), "DART starts")
            _require(starts.dtype.kind in "iu" and np.all(starts >= 0), "Invalid UNIX seconds")
        scale = float(legacy._array(data["scale"], (), "DART scale"))
        base = float(legacy._array(data["base_variance"], (), "DART base variance"))
        _require(scale > 0 and base > 0, "Nonpositive DART normalization")
        _dart_close(scale, evidence["dataset_statistics"][dataset]["std"], "training scale")
        _dart_close(base, evidence["dataset_statistics"][dataset]["base_variance"], "base variance")
        origin = int(data["starts_train"].min())
        blocks = _dart_blocks(data["starts_test"], origin)
        mean, covariance = weighted_moments(data["train"], np.ones(len(data["train"])))
        record = {
            "scale_m": scale, "base_variance_standardized2": base,
            "training_mean_standardized": mean.tolist(), "training_covariance_standardized2": covariance.tolist(),
            "train_starts_unix_s": data["starts_train"].tolist(), "test_starts_unix_s": data["starts_test"].tolist(),
            "phase_counts_utc": {split: _dart_phase_counts(data["starts_" + split]) for split in ("train", "test")},
            "block_origin_unix_s": origin, "global_window_weights": np.full(len(data["test"]), 1 / len(data["test"])).tolist(),
            "blocks": blocks, "neural": {}, "references": {}, "paired": {},
        }
        training = physical_window_statistics(data["train"], scale)
        _require(training["within_variance_m2"] > 0 and training["pooled_variance_m2"] > 0,
                 "Training spectrum requires positive within and pooled variance")
        frequencies = training["frequency_cycles_per_day"]
        shares = training["bin_power_m2"] / training["within_variance_m2"]
        support[dataset] = {
            "training_windows": training["window_count"], "weighting": "uniform_windows",
            "power_units": "m2", "frequency_units": "cycles_per_day",
            "one_cpd_share": float(shares[np.flatnonzero(frequencies == 1)[0]]),
            "two_cpd_share": float(shares[np.flatnonzero(frequencies == 2)[0]]),
            **{key: training[key] for key in ("within_variance_m2", "between_variance_m2",
                                             "pooled_variance_m2", "within_to_pooled_ratio")},
            "frequency_cycles_per_day": frequencies[1:].tolist(), "bin_power_m2": training["bin_power_m2"][1:].tolist(),
        }
        fits = {}
        for arm in ARMS:
            fits[arm], joint, increment, saved = [], [], [], []
            for seed in SEEDS:
                identifier = legacy.experiment.job_id(dataset, arm, seed)
                attempt, _ = _core_attempt(root, identifier, evidence)
                path = legacy._safe_path(attempt, "evaluation.npz")
                _require(legacy.experiment.sha256(path) == evidence["jobs"][identifier]["artifact_sha256"]["evaluation.npz"],
                         "DART evaluation input changed")
                arrays = legacy._load_npz(path)
                noisy = legacy._array(arrays["prior_samples"], (512, 192), "DART prior samples")
                archived = legacy._array(arrays["energy"], (len(data["test"]),), "DART saved energy")
                recomputed = energy_score(noisy, data["test"])
                _dart_close(recomputed, archived, "saved joint energy")
                saved.append(archived)
                joint.append(recomputed * scale)
                increment.append(increment_energy_score(noisy, data["test"]) * scale)
                means = reconstruct_decoder_means(noisy, _variance(arm, base))
                fits[arm].append({"noisy": physical_window_statistics(noisy, scale),
                                  "decoder_mean": physical_window_statistics(means, scale)})
            record["neural"][arm] = {
                "optimization_seeds": list(SEEDS), "observation_variance_standardized2": _variance(arm, base),
                "saved_joint_standardized": np.asarray(saved).tolist(),
                "scores": _dart_score_summary(joint, increment, blocks),
                "distribution": [{kind: _dart_plot_values(stats) for kind, stats in fit.items()} for fit in fits[arm]],
            }
        for name, (model, multiplier) in DART_REFERENCES.items():
            variance = base * multiplier
            fitted = (fixed_noise_ppca(mean, covariance, variance, rank=16) if model == "ppca" else
                      noise_matched_diagonal(data["train"], np.ones(len(data["train"])), variance))
            samples = fitted["mean"] + normal @ np.linalg.cholesky(fitted["covariance"]).T
            record["references"][name] = {
                "model": model, "variance_multiplier": multiplier, "observation_variance_standardized2": variance,
                "sampling_seed": 45001, "integration_streams": 1,
                "effective_rank": fitted["effective_rank"] if model == "ppca" else None,
                "covariance_standardized2": fitted["covariance"].tolist(),
                "covariance_sha256": _dart_array_digest(fitted["covariance"]),
                "prior_samples_sha256": _dart_array_digest(samples),
                "scores": _dart_score_summary([energy_score(samples, data["test"]) * scale],
                                               [increment_energy_score(samples, data["test"]) * scale], blocks),
            }
        for arm in ARMS[1:]:
            left, right = (record["neural"][a]["scores"]["per_window"] for a in (arm, "A"))
            record["paired"][arm + "-A"] = {
                "optimization_seeds": list(SEEDS), "scores": _dart_score_summary(
                    *(np.asarray(left[key]) - right[key] for key in DART_SCORES), blocks),
            }
        record["train_distribution"] = _dart_plot_values(training)
        record["test_distribution"] = _dart_plot_values(physical_window_statistics(data["test"], scale))
        records[dataset] = record
        private[dataset] = {"train": training, "test": physical_window_statistics(data["test"], scale), "fits": fits}
    diagnostics = {"protocol": deepcopy(DART_PROTOCOL), "records": records}
    validate_dart_diagnostics(diagnostics, evidence)
    return support, diagnostics, private


def _validate_dart_plot(value, n):
    fields = ("window_mean_relative_training_center_m", "within_window_sd_m", "increment_magnitude_m")
    _keys(value, ("window_count", "frequency_cycles_per_day", "bin_power_m2", "within_variance_m2",
                  "between_variance_m2", "pooled_variance_m2", "within_to_pooled_ratio", *fields), "DART distribution")
    _require(_integer(value["window_count"], "plot window count") == n, "DART plot count differs")
    _dart_close(value["frequency_cycles_per_day"], np.arange(97) / 2, "plot frequency")
    power = legacy._array(value["bin_power_m2"], (97,), "plot power")
    arrays = {key: legacy._array(value[key], (n,), key) for key in fields}
    _require(np.all(power >= 0) and np.all(arrays[fields[1]] >= 0) and np.all(arrays[fields[2]] >= 0),
             "Negative DART power or magnitude")
    within = np.square(arrays[fields[1]]).mean()
    between = arrays[fields[0]].var()
    _dart_close(value["within_variance_m2"], within, "plot within variance")
    _dart_close(value["between_variance_m2"], between, "plot between variance")
    _dart_close(value["pooled_variance_m2"], within + between, "plot pooled variance")
    _dart_close(power.sum(), within, "plot Parseval")
    if within + between > 0:
        _dart_close(value["within_to_pooled_ratio"], within / (within + between), "plot variance fraction")
    else:
        _require(value["within_to_pooled_ratio"] is None, "Invalid zero-variance ratio")


def _validate_dart_scores(value, fits, n, blocks):
    _keys(value, ("per_window", "per_fit_mean", "mean", "blocks"), "DART score summary")
    _keys(value["per_window"], DART_SCORES, "DART score arrays")
    arrays = [legacy._array(value["per_window"][key], (fits, n), key) for key in DART_SCORES]
    _dart_reductions(value, _dart_score_summary(*arrays, blocks), "score summary")


def validate_dart_diagnostics(value, core):
    """Pure descriptive gate; input archives, not this gate, attest observations."""
    _public_tree(value)
    _keys(value, ("protocol", "records"), "DART diagnostics")
    _require(canonical_sha256(value["protocol"]) == canonical_sha256(DART_PROTOCOL), "DART protocol differs")
    _keys(value["records"], DATASETS[1:], "DART diagnostic records")
    for dataset, record in value["records"].items():
        _keys(record, ("scale_m", "base_variance_standardized2", "training_mean_standardized",
                       "training_covariance_standardized2", "train_starts_unix_s", "test_starts_unix_s",
                       "phase_counts_utc", "block_origin_unix_s", "global_window_weights", "blocks", "neural",
                       "references", "paired", "train_distribution", "test_distribution"), "DART record")
        stats = core["dataset_statistics"][dataset]
        n = stats["windows"]["test"]
        scale = _number(record["scale_m"], "DART scale", 0)
        base = _number(record["base_variance_standardized2"], "DART variance", 0)
        _require(scale > 0 and base > 0, "Nonpositive DART normalization")
        _dart_close(scale, stats["std"], "DART scale")
        _dart_close(base, stats["base_variance"], "DART base variance")
        mean = legacy._array(record["training_mean_standardized"], (192,), "training mean")
        covariance = legacy._array(record["training_covariance_standardized2"], (192, 192), "training covariance")
        _dart_close(covariance, covariance.T, "population covariance symmetry")
        _require(np.linalg.eigvalsh(covariance).min() >= -DART_PROTOCOL["float64_absolute_tolerance"]
                 * max(1, np.linalg.norm(covariance)), "Invalid population covariance")
        _keys(record["phase_counts_utc"], ("train", "test"), "DART phase counts")
        for split in ("train", "test"):
            starts = legacy._array(record[split + "_starts_unix_s"], (stats["windows"][split],), "window starts")
            _require(starts.dtype.kind in "iu" and np.all(starts >= 0), "Invalid UNIX seconds")
            _require(canonical_sha256(record["phase_counts_utc"][split]) == canonical_sha256(_dart_phase_counts(starts)),
                     "DART phase counts differ")
        origin = _integer(record["block_origin_unix_s"], "block origin", 0)
        _require(origin == min(record["train_starts_unix_s"]), "DART block origin differs")
        weights = legacy._array(record["global_window_weights"], (n,), "DART global weights")
        _require(np.array_equal(weights, np.full(n, 1 / n)), "DART global weights differ")
        blocks = _dart_blocks(record["test_starts_unix_s"], origin)
        _require(canonical_sha256(record["blocks"]) == canonical_sha256(blocks), "DART block topology or mass differs")
        _validate_dart_plot(record["train_distribution"], stats["windows"]["train"])
        _validate_dart_plot(record["test_distribution"], n)
        training = record["train_distribution"]
        _dart_close(np.mean(training["window_mean_relative_training_center_m"]),
                    mean.mean() * scale, "training window-mean average")
        _dart_close(training["between_variance_m2"], covariance.mean() * scale**2,
                    "training window-mean variance")
        _dart_close(training["pooled_variance_m2"], (np.trace(covariance) / 192 + mean.var()) * scale**2,
                    "training pooled variance")
        _keys(record["neural"], ARMS, "DART neural arms")
        for arm, fit in record["neural"].items():
            _keys(fit, ("optimization_seeds", "observation_variance_standardized2", "saved_joint_standardized",
                        "scores", "distribution"), "DART neural record")
            _require(canonical_sha256(fit["optimization_seeds"]) == canonical_sha256(list(SEEDS)), "DART seeds differ")
            _dart_close(fit["observation_variance_standardized2"], _variance(arm, base), "neural variance")
            _validate_dart_scores(fit["scores"], 3, n, blocks)
            archived = legacy._array(fit["saved_joint_standardized"], (3, n), "archived energies")
            _dart_close(fit["scores"]["per_window"]["joint_m"], archived * scale, "joint meter conversion")
            for i, seed in enumerate(SEEDS):
                job = core["jobs"][legacy.experiment.job_id(dataset, arm, seed)]
                _dart_close(archived[i].mean(), job["energy_score"], "core energy mean")
                _dart_close(fit["scores"]["per_fit_mean"]["joint_m"][i], job["energy_score_meters"], "core meter energy mean")
            _require(isinstance(fit["distribution"], list) and len(fit["distribution"]) == 3, "Invalid fit plot count")
            for distribution in fit["distribution"]:
                _keys(distribution, ("noisy", "decoder_mean"), "fit distributions")
                for plot in distribution.values():
                    _validate_dart_plot(plot, 512)
        _keys(record["references"], DART_REFERENCES, "DART references")
        for name, (model, multiplier) in DART_REFERENCES.items():
            reference = record["references"][name]
            _keys(reference, ("model", "variance_multiplier", "observation_variance_standardized2", "sampling_seed",
                              "integration_streams", "effective_rank", "covariance_standardized2", "covariance_sha256",
                              "prior_samples_sha256", "scores"),
                  "DART reference")
            fixed = {"model": model, "variance_multiplier": multiplier, "sampling_seed": 45001, "integration_streams": 1}
            _require(canonical_sha256({k: reference[k] for k in fixed}) == canonical_sha256(fixed), "DART reference settings differ")
            variance = base * multiplier
            _dart_close(reference["observation_variance_standardized2"], variance, "reference variance")
            if model == "ppca":
                fitted = fixed_noise_ppca(mean, covariance, variance, rank=16)
                expected_covariance, rank = fitted["covariance"], fitted["effective_rank"]
            else:
                expected_covariance, rank = np.diag(np.maximum(np.diag(covariance), variance)), None
            _require(canonical_sha256(reference["effective_rank"]) == canonical_sha256(rank), "Reference rank differs")
            retained_covariance = legacy._array(reference["covariance_standardized2"], (192, 192), "reference covariance")
            _dart_close(retained_covariance, expected_covariance, "reference covariance or floor")
            _require(reference["covariance_sha256"] == _dart_array_digest(retained_covariance), "Reference covariance identity differs")
            # This is the producer's draw identity, like component array_sha256,
            # not a claim of public byte replay across BLAS implementations.
            legacy._digest(reference["prior_samples_sha256"])
            _validate_dart_scores(reference["scores"], 1, n, blocks)
        _keys(record["paired"], ("B-A", "C-A", "D-A"), "DART paired differences")
        for contrast, paired in record["paired"].items():
            _keys(paired, ("optimization_seeds", "scores"), "DART paired record")
            _require(canonical_sha256(paired["optimization_seeds"]) == canonical_sha256(list(SEEDS)), "DART paired seeds differ")
            left, right = (record["neural"][arm]["scores"]["per_window"] for arm in contrast.split("-"))
            expected = _dart_score_summary(*(np.asarray(left[key]) - right[key] for key in DART_SCORES), blocks)
            _dart_reductions(paired["scores"], expected, "paired difference")
    return value


def table_fragments(components, support, diagnostics, combined_sha256):
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            streams = [stream for seed in SEEDS for stream in components["jobs"][
                legacy.experiment.job_id(dataset, arm, seed)]["streams"]]
            rows.append(legacy._row([legacy.LABELS[dataset], arm, *[
                legacy._number(np.mean([stream[key] for stream in streams])) for key in TABLE_METRICS]]))
    energy_rows = []
    for dataset, record in diagnostics["records"].items():
        models = [(arm, record["neural"][arm]["scores"]) for arm in ARMS]
        for model in ("ppca", "diagonal"):
            low, high = (record["references"][model + "_" + level] for level in ("low", "high"))
            label = "PPCA" if model == "ppca" else "Diag."
            # Combine only byte-identical covariances, draws AND score reductions.
            if (low["covariance_sha256"] == high["covariance_sha256"]
                    and low["prior_samples_sha256"] == high["prior_samples_sha256"]
                    and canonical_sha256(low["scores"]) == canonical_sha256(high["scores"])):
                models.append((label + r" (both $v_0,25v_0$)", low["scores"]))
            else:
                models.extend([(label + r" ($v_0$)", low["scores"]), (label + r" ($25v_0$)", high["scores"])])
        for label, scores in models:
            ranges = ["[" + ", ".join(legacy._number(x) for x in scores["blocks"][str(days)]["joint_m"]["range"]) + "]"
                      for days in (14, 28)]
            energy_rows.append(legacy._row([legacy.LABELS[dataset], label,
                                           *[legacy._number(scores["mean"][key]) for key in DART_SCORES], *ranges]))
    diagnostic_rows = [legacy._row([legacy.LABELS[dataset], *[
        legacy._number(support[dataset][key]) for key in ("one_cpd_share", "two_cpd_share", "within_to_pooled_ratio")]])
        for dataset in DATASETS[1:]]
    return {
        "generated/component_rows.tex": " \\\\\n".join(rows) + "\n",
        "generated/dart_diagnostic_rows.tex": " \\\\\n".join(diagnostic_rows) + "\n",
        "generated/dart_energy_rows.tex": " \\\\\n".join(energy_rows) + "\n",
        "generated/combined_binding.tex": "\\newcommand{\\PrigmCombinedSHA}{" + legacy._digest(combined_sha256) + "}\n",
    }


def _dart_order(records):
    return [name for name in (*DATASETS[1:], "dart_44402_2018", "dart_21415_2018", "dart_21418_2018")
            if name in records]


def _dart_weighted_ecdf(values, weights, grid):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    _require(values.ndim == 1 and weights.shape == values.shape and len(values) > 0
             and np.isfinite(values).all() and np.isfinite(weights).all()
             and np.all(weights >= 0) and weights.sum() > 0, "Invalid ECDF measure")
    order = np.argsort(values, kind="stable")
    mass = np.r_[0., np.cumsum(weights[order] / weights.sum())]
    return mass[np.searchsorted(values[order], grid, side="right")]


def _dart_distribution(windows, weights, scale):
    """Compact per-window features and linear physical power; no MC averaging here."""
    windows = np.asarray(windows, dtype=float)
    _require(windows.ndim == 2 and windows.shape[1] == 192 and np.isfinite(scale) and scale > 0,
             "Invalid DART distribution shape or scale")
    stats = window_spectrum_statistics(windows, weights)
    within, between, pooled = (stats[k] for k in (
        "within_window_variance", "between_window_mean_variance", "pooled_entry_variance"))
    _close(stats["pooled_bin_power"].sum(), within, "DART renderer Parseval")
    _close(within + between, pooled, "DART renderer within plus between")
    return {"features": np.column_stack((windows.mean(axis=1), windows.std(axis=1),
                                         np.linalg.norm(increment_features(windows), axis=1))) * scale,
            "weights": np.asarray(weights) / np.sum(weights),
            "power": stats["pooled_bin_power"][1:] * scale**2,
            "fractions": np.array([within, between]) / pooled if pooled > 0 else None}


def _dart_archive_summary(path, files, keys, scale):
    """Only caller-supplied verified paths; one archive open, selected arrays only."""
    path = Path(path)
    legacy._safe_path(path.parent, path.name)
    digest = legacy.experiment.sha256(path)
    _require(any(Path(name).name == path.name and expected == digest for name, expected in files.items()),
             "DART plot archive not in verified digest inventory")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            draws = archive[key]
            _require(draws.shape == (3, 1024, 192) and draws.dtype == np.dtype("float64"),
                     "Invalid DART plot archive shape or dtype")
            result[key] = [_dart_distribution(stream, np.full(1024, 1 / 1024), scale) for stream in draws]
            del draws
    _require(legacy.experiment.sha256(path) == digest, "DART plot archive changed during rendering")
    return result


def _dart_ecdf_summary(fits, feature):
    grid = np.unique(np.concatenate([stream["features"][:, feature] for fit in fits for stream in fit]))
    return grid, _dart_ecdf_on_grid(fits, feature, grid)


def _dart_ecdf_on_grid(fits, feature, grid):
    # The fit axis survives; only the MC stream axis is averaged.
    return np.array([np.mean([_dart_weighted_ecdf(s["features"][:, feature], s["weights"], grid)
                             for s in fit], axis=0) for fit in fits])


def _dart_refit_points(record, arm):
    ratios, deltas = [], []
    for seed in SEEDS:
        original = record["models"][f"{arm}/{seed}/original"]
        refit = record["models"][f"{arm}/{seed}/variance_refit"]
        ratios.append(refit["variance_fit"]["pooled"] / original["observation_variance"])
        deltas.append([(np.asarray(refit["scores"][score]["weighted"]["values"])
                        - np.asarray(original["scores"][score]["weighted"]["values"]))
                       for score in ("joint", "centered", "increment")])
    return np.asarray(ratios), np.asarray(deltas)


def _dart_extension_figures(directory, evidence, private, combined_sha256):
    """Schema-3 diagnostics from verified saved draws and public paired scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7",
              "PPCA": "#6A3D9A", "train": "0.35", "test": "black"}
    styles = {"train": ":", "test": "-", "PPCA": "-"}
    metadata = {"Software": "PriGM CPU evidence builder", "CombinedInputSHA256": combined_sha256,
                "Description": "Verified original noisy prior samples; three sampling streams within each fit, descriptive seed ranges. "
                               "Fitted PPCA is one model. ECDF features in meters; spectra in square meters. "
                               "Refit energy differences remain standardized; variance ratios are dimensionless."}
    records = evidence["records"]
    reduced = {}
    for name in _dart_order(records):
        record, data = records[name], private["data"][name]
        scale = record["scale"]
        _close(float(data["scale"]), scale, "DART plot training scale")
        fresh = record["target"] == "six_UTC_bins_equal"
        case = {}
        for split in ("train", "test"):
            weights = data["train_weights" if split == "train" else "eval_weights"] if fresh else np.ones(len(data[split]))
            case[split] = [[_dart_distribution(data[split], weights, scale)]]
        for arm in ARMS:
            case[arm], case[arm + " means"] = [], []
            for seed in SEEDS:
                summary = _dart_archive_summary(private["model_paths"][f"{name}/{arm}/{seed}"],
                    evidence["input_index"]["files"], ("original__prior_samples", "original__prior_means"), scale)
                case[arm].append(summary["original__prior_samples"])
                case[arm + " means"].append(summary["original__prior_means"])
        # Low/high copies share fitted PPCA draws; retain exactly one model's streams.
        summary = _dart_archive_summary(private["reference_paths"][name + "/low"],
            evidence["input_index"]["files"], ("ppca_fitted__prior_samples",), scale)
        case["PPCA"] = [summary["ppca_fitted__prior_samples"]]
        reduced[name] = case
    with plt.rc_context({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif",
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}):
        handles = [Line2D([], [], color=colors[k], linestyle=styles.get(k, "-"),
                          label={"train": "Observed train", "test": "Observed test", "PPCA": "Fitted PPCA"}.get(k, k))
                   for k in ("train", "test", *ARMS, "PPCA")]
        def save(fig, name):
            fig.savefig(Path(directory) / "figures" / name, dpi=200, metadata=metadata)
            plt.close(fig)
        for group, fresh in (("historical", False), ("fresh", True)):
            names = [n for n in reduced if (records[n]["target"] == "six_UTC_bins_equal") == fresh]
            _require(len(names) == (3 if fresh else 4), "Incomplete DART diagnostic case coverage")
            fig, axes = plt.subplots(len(names), 3, figsize=(5.5, 1.65 * len(names) + .65), squeeze=False)
            for row, name in enumerate(names):
                case = reduced[name]
                for feature, axis in enumerate(axes[row]):
                    fits = [fit for key in ("train", "test", *ARMS, "PPCA") for fit in case[key]]
                    grid, _ = _dart_ecdf_summary(fits, feature)
                    # Include the zero left limit without moving or dropping any breakpoint.
                    grid = np.r_[np.nextafter(grid[0], -np.inf), grid]
                    for key in ("train", "test", *ARMS, "PPCA"):
                        curves = _dart_ecdf_on_grid(case[key], feature, grid)
                        axis.fill_between(grid, curves.min(0), curves.max(0), step="post", color=colors[key], alpha=.16)
                        axis.step(grid, np.median(curves, axis=0), where="post", color=colors[key],
                                  linestyle=styles.get(key, "-"), linewidth=.9)
                    axis.set_ylim(0, 1)
                    axis.set_xlabel(("Window mean\nminus train mean (m)", "Window SD (m)", "Increment length (m)")[feature])
                    axis.tick_params(axis="x", labelrotation=25)
                axes[row, 0].set_ylabel(f"{records[name]['station']}/{records[name]['year']}\nECDF")
            fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .91 if fresh else .93))
            save(fig, "dart_ecdf_" + group + ".png")
            fig, axes = plt.subplots(len(names), 2, figsize=(5.5, 1.55 * len(names) + 1.), squeeze=False)
            frequency = np.arange(1, 97) / 2
            for row, name in enumerate(names):
                case = reduced[name]
                for col, arms in enumerate((("A", "C"), ("B", "D"))):
                    axis = axes[row, col]
                    for key in ("train", "test", "PPCA", *arms, *(a + " means" for a in arms)):
                        power = np.array([np.mean([s["power"] for s in fit], axis=0) for fit in case[key]])
                        color = colors[key.split()[0]]
                        positive = lambda x: np.ma.masked_less_equal(x, 0)
                        axis.fill_between(frequency, positive(power.min(0)), positive(power.max(0)), color=color, alpha=.12)
                        axis.plot(frequency, positive(np.median(power, axis=0)), color=color,
                                  linestyle="--" if key.endswith(" means") else styles.get(key, "-"), linewidth=.9)
                    axis.set_yscale("log", nonpositive="mask")
                    axis.set_xlim(.5, 48)
                    ticks = [1, 2, 12, 24, 48]
                    axis.set_xticks(ticks)
                    # Keep both 1 and 2 cycles/day. Put 2 on the next line so the
                    # adjacent labels cannot collide into a false "12" at print width.
                    axis.set_xticklabels(["1", "\n2", "12", "24", "48"])
                    axis.set_xlabel("Cycles/day")
                    if row == 0:
                        axis.set_title("Low observation variance: A/C" if col == 0
                                       else "High observation variance: B/D")
                limits = (min(a.get_ylim()[0] for a in axes[row]),
                          max(a.get_ylim()[1] for a in axes[row]))
                for axis in axes[row]:
                    axis.set_ylim(limits)
                axes[row, 0].set_ylabel(f"{records[name]['station']}/{records[name]['year']}\nBin power (m$^2$)")
            extra = [Line2D([], [], color="gray", linestyle="--", label="Prior decoder means")]
            fig.legend(handles=handles + extra, loc="upper center", ncol=4, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .86 if fresh else .89))
            save(fig, "dart_spectra_" + group + ".png")
        fig, axes = plt.subplots(1, 4, figsize=(5.5, 7.2), sharey=True)
        labels = []
        for row, (name, arm) in enumerate((n, a) for n in reduced for a in ARMS):
            record = records[name]
            labels.append(f"{record['station']}/{record['year']} {arm}")
            ratios, deltas = _dart_refit_points(record, arm)
            for fit, offset in enumerate((-.2, 0, .2)):
                y = row + offset
                axes[0].plot(ratios[fit], y, "o", color=colors[arm], markersize=3)
                for col in range(3):
                    values = deltas[fit, col]
                    axes[col + 1].hlines(y, values.min(), values.max(), color=colors[arm], linewidth=.6)
                    axes[col + 1].plot(values.mean(), y, "o", color=colors[arm], markersize=3)
        for col, axis in enumerate(axes):
            axis.axvline(1 if col == 0 else 0, color="gray", linewidth=.7, linestyle=":")
            axis.set_xlabel(("Variance\nratio", "Joint\nchange", "Centered\nchange", "Increment\nchange")[col])
            axis.tick_params(axis="x", labelrotation=45)
        axes[0].set_xscale("log")
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        fig.suptitle("Observation variance reset\nThree seeds. Thin bars span three sampling streams", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .96), w_pad=.5)
        save(fig, "dart_variance_refit.png")


def _dart_interval(summary, center="median", *, stacked=False):
    values = [legacy._number(float(format(summary[key], ".3g"))) for key in (center, "min", "max")]
    interval = "[" + ", ".join(values[1:]) + "]"
    return r"\shortstack{" + values[0] + r"\\{" + interval + "}}" if stacked else values[0] + " " + interval


def _dart_table_fragments(components, evidence, combined_sha256, core):
    records = evidence["records"]
    ordered = _dart_order(records)
    rows = {name: [] for name in ("component", "dart_record", "dart_extension", "dart_variance",
                                  "dart_reference", "dart_selected", "dart_block")}
    for arm in ARMS:
        streams = [stream for seed in SEEDS for stream in components["jobs"][
            legacy.experiment.job_id("gaussian", arm, seed)]["streams"]]
        raw = np.mean([core["jobs"][legacy.experiment.job_id("gaussian", arm, seed)]["self_mse"] for seed in SEEDS])
        rows["component"].append(legacy._row(["Gaussian", arm, legacy._number(raw), *[
            legacy._number(np.mean([stream[key] for stream in streams])) for key in TABLE_METRICS]]))
    for dataset in ordered:
        record = records[dataset]
        label = str(record["station"]) + "/" + str(record["year"])
        fresh = record["primary_checkpoint"] == "final"
        models = record["models"]
        if fresh:
            rows["dart_record"].append(legacy._row([label, *[record["splits"][split]["windows"]
                for split in ("train", "validation", "test")], format(record["mean"], ".4f"),
                legacy._number(record["scale"]), legacy._number(models["A/42/original"]["observation_variance"])]))
        for arm in ARMS:
            fits = [models[f"{arm}/{seed}/original"] for seed in SEEDS]
            fields = ("posterior_mean_mse", "posterior_expected_mse", "nll_per_coordinate",
                      "kl_per_coordinate", "negative_elbo_per_coordinate", "negative_iw_per_coordinate")
            values = [float(np.mean([fit["components"]["weighted"][key]["mean"] for fit in fits])) for key in fields]
            # Fresh runs did not store the bypass-space floating-point residual.
            # This column is its algebraic equivalent, not an extra measurement.
            raw = (1 - (.8 if arm in ("B", "C") else 0))**2 * values[0] if fresh else np.mean(
                [fit["raw_self_mse"] for fit in fits])
            rows["component"].append(legacy._row([label, arm, legacy._number(raw), *map(legacy._number, values)]))
            variance = record["optimizer_summaries"][f"{arm}/variance_refit"]["observation_variance"]
            changes = []
            for score in ("joint", "centered", "increment"):
                delta = [1000 * (models[f"{arm}/{seed}/variance_refit"]["scores"][score]["weighted"]["mean"]
                                 - models[f"{arm}/{seed}/original"]["scores"][score]["weighted"]["mean"]) for seed in SEEDS]
                changes.append(_dart_interval({"median": float(np.median(delta)), "min": min(delta), "max": max(delta)}, stacked=True))
            displayed_variance = {key: 10000 * variance[key] for key in ("median", "min", "max")}
            rows["dart_variance"].append(legacy._row([label, arm, _dart_interval(displayed_variance, stacked=True), *changes]))
        block_ranges = []
        for contrast in ("C-A", "B-D"):
            left, right = contrast.split("-")
            for days in ("14", "28"):
                delta = [a["streams"]["mean"] - b["streams"]["mean"] for seed in SEEDS for a, b in zip(
                    models[f"{left}/{seed}/original"]["scores"]["joint"]["blocks"][days],
                    models[f"{right}/{seed}/original"]["scores"]["joint"]["blocks"][days], strict=True)]
                block_ranges.append("[" + ", ".join(legacy._number(float(format(value, ".3g")))
                                                       for value in (min(delta), max(delta))) + "]")
            values = record["contrasts"][contrast + "/original"]["scores"]
            cells = [_dart_interval(values[score]["optimizer_seeds"], stacked=True) for score in ("joint", "centered", "increment")]
            rows["dart_extension"].append(legacy._row([label, contrast, "Seed", *cells]))
            mc_widths = [legacy._number(max(stream["max"] - stream["min"]
                                           for stream in values[score]["paired_mc_streams"]))
                         for score in ("joint", "centered", "increment")]
            rows["dart_extension"].append(legacy._row([label, contrast, "Monte Carlo width", *mc_widths]))
            if fresh:
                changes = []
                for score in ("joint", "centered", "increment"):
                    delta = [models[f"{left}/{seed}/original"]["selected_secondary"]["scores"][score]["weighted"]["mean"]
                             - models[f"{right}/{seed}/original"]["selected_secondary"]["scores"][score]["weighted"]["mean"]
                             for seed in SEEDS]
                    changes.append(_dart_interval({"median": float(np.median(delta)), "min": min(delta), "max": max(delta)}, stacked=True))
                steps = [models[f"{arm}/{seed}/original"]["selected_secondary"]["step"]
                         for arm in (left, right) for seed in SEEDS]
                rows["dart_selected"].append(legacy._row([label, contrast, "Selected/UTC", f"{min(steps)}--{max(steps)}", *changes]))
                uniform_changes = []
                for score in ("joint", "centered", "increment"):
                    delta = [models[f"{left}/{seed}/original"]["scores"][score]["uniform"]["mean"]
                             - models[f"{right}/{seed}/original"]["scores"][score]["uniform"]["mean"] for seed in SEEDS]
                    uniform_changes.append(_dart_interval({"median": float(np.median(delta)), "min": min(delta), "max": max(delta)}, stacked=True))
                rows["dart_selected"].append(legacy._row([label, contrast, "Final/uniform", "10000", *uniform_changes]))
        counts = "/".join(str(len(models["A/42/original"]["scores"]["joint"]["blocks"][days])) for days in ("14", "28"))
        rows["dart_block"].append(legacy._row([label, counts, *block_ranges]))
        for arm in ARMS:
            scores = record["optimizer_summaries"][arm + "/original"]["scores"]
            rows["dart_reference"].append(legacy._row([label, arm, *[
                _dart_interval(scores[score]) for score in ("joint", "centered", "increment")]]))
        reference_rows = [("low", "ppca_fitted", "PPCA fitted")]
        reference_rows.extend((level, kind, name + (r" ($v$)" if level == "low" else r" ($25v$)"))
                              for level in ("low", "high") for kind, name in
                              (("ppca_matched", "PPCA matched"), ("diagonal_matched", "Diagonal")))
        for level, kind, name in reference_rows:
            reference = record["references"][level][kind]
            rows["dart_reference"].append(legacy._row([label, name, *[
                _dart_interval(reference["scores"][score]["weighted"], "mean") for score in ("joint", "centered", "increment")]]))
    fragments = {"generated/" + name + "_rows.tex": " \\\\\n".join(values) + "\n" for name, values in rows.items()}
    fragments["generated/combined_binding.tex"] = "\\newcommand{\\PrigmCombinedSHA}{" + legacy._digest(combined_sha256) + "}\n"
    return fragments


def _dart_comparison_figure(directory, evidence, combined_sha256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = evidence["records"]
    ordered = _dart_order(records)
    with plt.rc_context({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif",
                         "font.size": 9, "axes.formatter.useoffset": False, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.6), sharey=True)
        for axis, score, title in zip(axes, ("joint", "centered", "increment"),
                                      ("Complete windows", "Centered windows", "Increments")):
            for contrast, offset, color, marker, label in (
                ("C-A", -.12, "#009E73", "s", r"C-A ($\sigma_0^2$)"),
                ("B-D", .12, "#CC79A7", "o", r"B-D ($25\sigma_0^2$)")):
                stats = [records[name]["contrasts"][contrast + "/original"]["scores"][score]["optimizer_seeds"] for name in ordered]
                middle = np.array([value["median"] for value in stats])
                error = np.array([[value["median"] - value["min"] for value in stats],
                                  [value["max"] - value["median"] for value in stats]])
                axis.errorbar(middle, np.arange(len(ordered)) + offset, xerr=error,
                              fmt=marker, color=color, capsize=2, markersize=4, label=label)
            axis.axvline(0, color="0.5", linewidth=.8)
            axis.axhline(3.5, color="0.65", linewidth=.7, linestyle=":")
            axis.grid(axis="x", alpha=.2)
            axis.set_title(title)
            axis.set_xlabel("Difference\n(standardized)")
        axes[0].set_yticks(np.arange(len(ordered)), [str(records[name]["station"]) + "/" + str(records[name]["year"])
                                                  for name in ordered], fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Station / year")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.58, 1), ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9))
        fig.savefig(Path(directory) / "figures/reconstruction_generation.png", dpi=200, metadata={
            "Software": "PriGM CPU evidence builder", "CombinedInputSHA256": combined_sha256,
            "Description": "Same observation variance, skip minus alpha=0 energy score; median and full range of three seed stream means. Historical selected and 2018 final endpoints are separate case studies."})
        plt.close(fig)


def _ecdf(values, grid):
    return np.searchsorted(np.sort(values), grid, side="right") / len(values)


def _comparison_figure(directory, paired, combined_sha256):
    """Render the verified paired summaries without recomputing statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B": "#C43D58", "C": "#93B042", "D": "#6A7D92"}
    markers = {"B": "o", "C": "s", "D": "^"}
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
                              fmt=markers[arm], linestyle="none", capsize=3, markersize=4,
                              color=colors[arm], label=arm + suffix)
            axis.set_xticks(np.arange(len(DATASETS)),
                            ["Gaussian" if d == "gaussian" else d.replace("t", "\n") for d in DATASETS],
                            fontsize=8.5, rotation=0, ha="center")
            axis.set_xlabel("DART station / year")
            axis.margins(x=0.02)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=9.5)
        axes[0].set_yscale("log")
        axes[0].axhline(1, color="0.5", linewidth=0.8)
        axes[0].set_ylabel("MSE ratio to A (log scale)")
        axes[0].set_title("Reconstruction error")
        axes[1].axhline(0, color="0.5", linewidth=0.8)
        axes[1].set_ylabel("Change in negative IW estimate\nnats per coordinate")
        axes[1].set_title("Estimated log density")
        fig.tight_layout()
        fig.savefig(Path(directory) / "figures" / "reconstruction_generation.png", dpi=200, metadata={
            "Software": "PriGM CPU evidence builder",
            "Description": "All five datasets, all three paired optimization seeds; medians and full seed ranges.",
            "CombinedInputSHA256": combined_sha256})
        plt.close(fig)


def _figures(directory, support, private, combined_sha256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, LogLocator

    colors = {"A": "#C4A04A", "B": "#C43D58", "C": "#93B042", "D": "#6A7D92"}
    def save(fig, name, description, training=False):
        handles = [Line2D([], [], color=colors[arm], label=arm) for arm in ARMS]
        handles += [Line2D([], [], color="0.4", linestyle=style, label=label)
                    for label, style in (("Noisy", "-"), ("Decoder mean", "--"))]
        handles.append(Line2D([], [], color="black", linewidth=2, label="Test"))
        if training:
            handles.append(Line2D([], [], color="black", linestyle=":", label="Train"))
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8,
                   handlelength=1.6, columnspacing=1, bbox_to_anchor=(0.5, 1))
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.8, w_pad=0.8)
        fig.savefig(Path(directory) / "figures" / name, dpi=200, metadata={
            "Software": "PriGM corrected audit CPU builder", "Description": description,
            "CombinedInputSHA256": combined_sha256})
        plt.close(fig)

    def fit_curves(axis, record, field, grid, ecdf):
        for arm in ARMS:
            for kind, linestyle in (("noisy", "-"), ("decoder_mean", "--")):
                curves = np.array([_ecdf(fit[kind][field], grid) if ecdf else fit[kind][field][1:]
                                   for fit in record["fits"][arm]])
                axis.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=colors[arm],
                                  alpha=0.09, step="post" if ecdf else None)
                axis.plot(grid, np.median(curves, axis=0), color=colors[arm], linestyle=linestyle,
                          linewidth=0.9, drawstyle="steps-post" if ecdf else "default",
                          label=arm + (" noisy" if kind == "noisy" else " decoder mean"))

    with plt.rc_context({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif",
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(4, 2, figsize=(5.5, 7.3), gridspec_kw={"width_ratios": [3.3, 1.2]})
        for row, dataset in enumerate(DATASETS[1:]):
            stats, record = support[dataset], private[dataset]
            axis, fractions = axes[row]
            grid = np.asarray(stats["frequency_cycles_per_day"])
            axis.plot(grid, stats["bin_power_m2"], color="black", linestyle=":", linewidth=1.6, label="Train", zorder=10)
            fit_curves(axis, record, "bin_power_m2", grid, False)
            axis.plot(grid, record["test"]["bin_power_m2"][1:], color="black", linewidth=2, label="Test", zorder=10)
            axis.set(xlabel="Frequency (cycles/day)", ylabel="DFT power (m$^2$)",
                     title="Station " + dataset.replace("t", ", "), xlim=(0.5, 48), yscale="log", xticks=[1, 12, 24, 48])
            axis.yaxis.set_major_locator(LogLocator(numticks=3))
            axis.minorticks_off()
            ratio = stats["within_to_pooled_ratio"]
            fractions.bar([0, 1], [ratio, 1 - ratio], color=["#444444", "#BBBBBB"])
            fractions.set(xticks=[0, 1], xticklabels=["Within", "Between"], ylim=(0, 1), yticks=[0, 0.5, 1],
                          ylabel="Variance share")
        save(fig, "dart_support_spectra.png", METHODS["support"] + " " + METHODS["distribution"], training=True)

        fig, axes = plt.subplots(4, 3, figsize=(5.5, 7.3), sharey=True)
        fields = ("window_mean_relative_training_center_m", "within_window_sd_m", "increment_magnitude_m")
        labels = ("Window mean\nminus train mean (m)", "Window SD (m)", "Increment length (m)")
        for row, dataset in enumerate(DATASETS[1:]):
            record = private[dataset]
            for column, field in enumerate(fields):
                axis = axes[row, column]
                target, training = record["test"][field], record["train"][field]
                values = np.concatenate([training, target, *[fit[kind][field] for arm in ARMS
                                        for fit in record["fits"][arm] for kind in ("noisy", "decoder_mean")]])
                # Exact ECDF breakpoints, not fitted distributions or quantile averages.
                grid = np.unique(values)
                fit_curves(axis, record, field, grid, True)
                axis.plot(grid, _ecdf(training, grid), color="black", linestyle=":", linewidth=1.6,
                          drawstyle="steps-post", label=f"Observed train (N={len(training)})", zorder=10)
                axis.plot(grid, _ecdf(target, grid), color="black", linewidth=2, drawstyle="steps-post",
                          label=f"Observed test (N={len(target)})", zorder=10)
                axis.set(ylim=(0, 1), yticks=[0, 0.5, 1], xlabel=labels[column])
                axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
                if column == 0:
                    axis.set(ylabel="ECDF", title="Station " + dataset.replace("t", ", "))
        save(fig, "dart_distribution_diagnostics.png", METHODS["distribution"] + " " + METHODS["decoder_means"], training=True)


def _public_tree(value):
    """Pure structural privacy check: no host lookup and no private file reads."""
    if isinstance(value, dict):
        for key, child in value.items():
            _require(isinstance(key, str) and not re.search(
                r"(?:^|_)(?:host|hostname|username|uid|uuid|runtime|timestamp|created_at|path)(?:$|_)", key, re.I),
                "Private field in public evidence")
            _public_tree(key)
            _public_tree(child)
    elif isinstance(value, list):
        for child in value:
            _public_tree(child)
    elif isinstance(value, str):
        _require(not value.startswith(("/", "~")) and not re.search(
            r"(?:^|[\s\"'=])/(?:[^\s]+)|[A-Za-z]:[\\/]|\\\\|file://|\.\./|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
            r"|\b(?:GPU|MIG)-[A-Za-z0-9-]+|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", value),
            "Internal path or machine identity in public evidence")
    else:
        _require(type(value) in (int, float, bool, type(None)), "Non-JSON public value")
        if type(value) is float:
            _number(value, "public value")


def _validate_streams(streams):
    _require(isinstance(streams, list) and len(streams) == 3, "Invalid component streams")
    for seed, stream in zip(STREAM_SEEDS, streams):
        _keys(stream, (*STREAM_METRICS, "stream_seed", "mean_kl_by_latent"), "stream statistics")
        for key in ("posterior_mean_mse", "kl_per_coordinate", "mean_kl_by_latent"):
            _require(stream[key] == streams[0][key], "Public shared analytic streams differ")
        _require(type(stream["stream_seed"]) is int and stream["stream_seed"] == seed, "Invalid stream seed")
        for key in STREAM_METRICS:
            _number(stream[key], key)
        for key in ("posterior_mean_mse", "posterior_expected_mse", "kl_per_coordinate"):
            _number(stream[key], key, 0)
        for key in ("mean_ess", "median_ess"):
            _require(1 - 1e-7 <= stream[key] <= 256 * (1 + 1e-7), "Invalid ESS statistic")
        latent = stream["mean_kl_by_latent"]
        _require(isinstance(latent, list) and len(latent) == 16, "Invalid latent KL statistics")
        for value in latent:
            _number(value, "latent KL", 0)
        _close(sum(latent) / 192, stream["kl_per_coordinate"], "public latent KL")
        _close(stream["negative_elbo_analytic_kl_per_coordinate"],
               stream["posterior_expected_nll_per_coordinate"] + stream["kl_per_coordinate"], "public analytic ELBO")
        _number(stream["jensen_gap_per_coordinate"], "Jensen gap", -1e-10)
        _close(stream["jensen_gap_per_coordinate"], stream["sample_negative_elbo_per_coordinate"]
               - stream["negative_iw_per_coordinate"], "public Jensen gap")


def validate_combined_manifest(manifest, expected_core_sha256=None):
    """Pure schema-2/3 gate returning manifest; no file, source, PDF or host reads.

    A release verifier must additionally compare output hashes to actual files
    and source hashes to current code. This validates public consistency, not a
    replacement for private scientific evidence verification.
    """
    _public_tree(manifest)
    extension = (isinstance(manifest, dict) and type(manifest.get("schema_version")) is int
                 and manifest["schema_version"] == 3)
    diagnostic_fields = ("dart_extension",) if extension else ("dart_support", "dart_diagnostics")
    _keys(manifest, ("schema_version", "purpose", "profile", "combined_input_sha256", "input_index",
                     "output_sha256", "builder_sha256", "core", "refinement", "components", "numerical_validation",
                     *diagnostic_fields, "methods"), "combined manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == (3 if extension else 2)
             and manifest["purpose"] == "combined", "Expected combined schema 2 or 3")
    _require(manifest["profile"] == ("dart_extension" if extension else "corrected_audit"), "Unknown combined profile")
    _require(manifest["methods"] == (DART_EXTENSION_METHODS if extension else METHODS), "Unknown public methods")
    _digests(manifest["output_sha256"], DART_EXTENSION_OUTPUTS if extension else AUDIT_OUTPUTS, "audit output set")
    index = manifest["input_index"]
    _keys(index, ("core", "refinement", "components", "numerical_validation",
                  "dart_extension" if extension else "dart_diagnostics", "source_sha256",
                  "evaluator_source_sha256", "decision_sha256"), "input index")
    validate_numerical_validation(manifest["numerical_validation"], index["numerical_validation"])
    if extension:
        _same_json(manifest["numerical_validation"]["source_sha256"], FROZEN_NUMERICAL_SOURCE_SHA256,
                   "Preserved numerical producer identity")
    else:
        _require(canonical_sha256(index["dart_diagnostics"]) == canonical_sha256({
            "approved_protocol_sha256": DART_PROTOCOL_SHA256, "settings": DART_PROTOCOL}), "DART derivation settings differ")
    for key in ("core", "refinement"):
        _digests(index[key], ("manifest_sha256", "archive_manifest_sha256"), key + " input")
    _keys(index["components"], WORKERS, "component input workers")
    for worker in WORKERS:
        _digests(index["components"][worker], ("completion_sha256", "protocol_sha256"), "component input")
    _digests(index["source_sha256"], DART_EXTENSION_SOURCE_IDS if extension else SOURCE_IDS, "source identities")
    _digests(index["evaluator_source_sha256"], EVALUATOR_SOURCES, "frozen evaluator identities")
    _require(index["evaluator_source_sha256"] == FROZEN_EVALUATOR_SHA256,
             "Unreviewed frozen evaluator identity")
    legacy._digest(index["decision_sha256"])
    _require(legacy._digest(manifest["combined_input_sha256"]) == canonical_sha256(index), "Combined input digest differs")
    _require(manifest["builder_sha256"] == index["source_sha256"]["src/build_prigm_v2_paper.py"], "Builder identity differs")
    core = manifest["core"]
    _keys(core, ("schema_version", "purpose", "run_manifest_sha256", "archive_manifest_sha256", "code_sha256",
                 "prepared_sha256", "public_source_sha256", "specification", "dataset_statistics", "jobs", "baselines",
                 "builder_sha256", "summary_statistics"), "nested core")
    _require(type(core["schema_version"]) is int and core["schema_version"] == 1 and core["purpose"] == "core",
             "Invalid nested core identity")
    _require(core["run_manifest_sha256"] == index["core"]["manifest_sha256"]
             and core["archive_manifest_sha256"] == index["core"]["archive_manifest_sha256"], "Core input identity differs")
    if expected_core_sha256 is not None:
        _require(core["run_manifest_sha256"] == legacy._digest(expected_core_sha256), "Expected core SHA256 differs")
    _require(core["builder_sha256"] == index["source_sha256"]["src/build_prigm_paper.py"], "Legacy builder identity differs")
    _digests(core["code_sha256"], legacy.experiment.CODE_FILES, "core sources")
    if extension:
        from prigm_dart_paper_evidence import CORE_SOURCES
        _same_json(core["code_sha256"], {"src/" + name: digest for name, digest in CORE_SOURCES.items()},
                   "Historical core source pins")
    else:
        _require(all(index["source_sha256"][name] == digest for name, digest in core["code_sha256"].items()), "Core source differs")
    _digests(core["prepared_sha256"], ("metadata.json", *(d + ".npz" for d in DATASETS)), "prepared identities")
    _digests(core["public_source_sha256"], (f"datasets/{d}.txt" for d in DATASETS[1:]), "public source identities")
    _require(canonical_sha256(core["specification"]) == canonical_sha256(legacy.experiment.specification("core")),
             "Frozen specification differs")
    _keys(core["dataset_statistics"], DATASETS, "core dataset statistics")
    for dataset, stats in core["dataset_statistics"].items():
        _keys(stats, ("windows", "mean", "std", "base_variance", *(() if dataset == "gaussian" else ("observed_t1_rows",))),
              "dataset statistics")
        _keys(stats["windows"], ("train", "validation", "test"), "split counts")
        for count in stats["windows"].values():
            _integer(count, "window count")
        if dataset != "gaussian":
            _integer(stats["observed_t1_rows"], "observed row count")
        _number(stats["mean"], "training mean")
        _require(_number(stats["std"], "training scale", 0) > 0
                 and _number(stats["base_variance"], "base variance", 0) > 0, "Nonpositive normalization")
    _keys(core["summary_statistics"], ("paired", "job_count"), "core summary")
    _require(_integer(core["summary_statistics"]["job_count"], "core jobs") == 60, "Invalid core job count")
    _require(isinstance(core["jobs"], dict), "Invalid core jobs")
    scalar_keys = (legacy.SCORE, "negative_log_score_meter_coordinates_per_dimension", "median_effective_samples",
                   "median_normalized_ess", "self_mse", "corrected_mse", "kl_nats_per_window", "energy_score",
                   "energy_score_meters", "permutation_corrected_mse_increase",
                   "stability_mean_log_score_change_per_dimension", "replicate_mean_log_score_change_per_dimension")
    for identifier, job in core["jobs"].items():
        _keys(job, ("dataset", "arm", "seed", "best_step", "steps_run", "test_windows", "mapped_control", "artifact_sha256",
                    *scalar_keys), "core job")
        _require(identifier == legacy.experiment.job_id(job["dataset"], job["arm"], job["seed"]), "Core job key differs")
        for key in scalar_keys:
            _number(job[key], key)
        _keys(job["mapped_control"], ("target_arm", "max_absolute_log_density_difference", "max_absolute_sample_mean_difference",
                                    "relative_log_density_tolerance", "absolute_log_density_tolerance"), "mapped control")
        legacy._mapped_statistics(job["mapped_control"], job["arm"])
    indexed = legacy._index_jobs(list(core["jobs"].values()))
    _require(set(indexed) == set(core["jobs"]), "Core job identifiers differ")
    _require(core["summary_statistics"]["paired"] == legacy.paired_statistics(core["jobs"]), "Core paired statistics differ")
    _keys(core["baselines"], DATASETS, "baseline datasets")
    for dataset, models in core["baselines"].items():
        _keys(models, ("ppca", "diagonal_gaussian", *(("oracle",) if dataset == "gaussian" else ())), "baseline models")
        for stats in models.values():
            _keys(stats, (legacy.SCORE, "energy_score"), "baseline statistics")
            for value in stats.values():
                _number(value, "baseline statistic")
    refinement = manifest["refinement"]
    _keys(refinement, ("manifest_sha256", "archive_manifest_sha256", "code_sha256", "results"), "refinement")
    _require(all(refinement[key] == digest for key, digest in index["refinement"].items()), "Refinement input identity differs")
    _digests(refinement["code_sha256"], ("src/prigm_refine_evaluation.py",), "refinement source")
    _require(refinement["code_sha256"]["src/prigm_refine_evaluation.py"] == index["source_sha256"]["src/prigm_refine_evaluation.py"],
             "Refinement source differs")
    _keys(refinement["results"], ("specification", "jobs", "paired_comparisons"), "refinement results")
    from prigm_refine_evaluation import specification as refinement_specification
    _require(canonical_sha256(refinement["results"]["specification"]) == canonical_sha256(refinement_specification()),
             "Refinement specification differs")
    _require(isinstance(refinement["results"]["jobs"], list) and len(refinement["results"]["jobs"]) == 60,
             "Invalid refinement job count")
    refinement_jobs = {}
    for position, job in enumerate(refinement["results"]["jobs"]):
        _keys(job, ("core_job_id", "dataset", "arm", "seed", "count", "input_dim", "worker_index",
                    "core_completion_sha256", "core_checkpoint_sha256", "core_evaluation_sha256", "metrics"),
              "refinement job")
        dataset, arm, seed, _ = canonical_jobs()[position]
        identifier = legacy.experiment.job_id(dataset, arm, seed)
        _require((job["core_job_id"], job["dataset"], job["arm"], job["seed"])
                 == (identifier, dataset, arm, seed), "Refinement job identity differs")
        for key in ("seed", "count", "input_dim"):
            _integer(job[key], key)
        _integer(job["worker_index"], "refinement worker", 0)
        _require(job["worker_index"] == position % 4 and job["input_dim"] == 192
                 and job["count"] == min(32, core["dataset_statistics"][dataset]["windows"]["test"]),
                 "Refinement counts or worker differ")
        for key in ("core_completion_sha256", "core_checkpoint_sha256", "core_evaluation_sha256"):
            legacy._digest(job[key])
        _keys(job["metrics"], ("256", "4096", "16384"), "refinement budgets")
        for budget, stats in job["metrics"].items():
            _keys(stats, ("mean_nll_nats_per_dimension", "median_ess", "mean_log_score_change_from_256_per_dimension"),
                  "refinement metrics")
            for value in stats.values():
                _number(value, "refinement statistic")
            _require(1 - 1e-7 <= stats["median_ess"] <= int(budget) * (1 + 1e-7), "Invalid refinement ESS")
        refinement_jobs[identifier] = job
    pairs = refinement["results"]["paired_comparisons"]
    _keys(pairs, DATASETS, "refinement paired datasets")
    for dataset in DATASETS:
        _keys(pairs[dataset], ("B-A", "C-A", "B-D"), "refinement contrasts")
        for contrast, budgets in pairs[dataset].items():
            _keys(budgets, ("256", "4096", "16384"), "refinement paired budgets")
            for budget, stats in budgets.items():
                _keys(stats, ("training_seeds", "paired_seed_deltas", "median", "range"), "refinement paired statistics")
                _require(canonical_sha256(stats["training_seeds"]) == canonical_sha256(list(SEEDS)), "Invalid paired seeds")
                left, right = contrast.split("-")
                values = [refinement_jobs[legacy.experiment.job_id(dataset, left, seed)]["metrics"][budget]["mean_nll_nats_per_dimension"]
                          - refinement_jobs[legacy.experiment.job_id(dataset, right, seed)]["metrics"][budget]["mean_nll_nats_per_dimension"]
                          for seed in SEEDS]
                _close(stats["paired_seed_deltas"], values, "refinement paired deltas")
                _close(stats["median"], float(np.median(values)), "refinement paired median")
                _close(stats["range"], [min(values), max(values)], "refinement paired range")
    components = manifest["components"]
    _keys(components, ("schema_version", "purpose", "job_count", "worker_count", "samples", "stream_seeds",
                       "density_coordinates", "probability_dtype", "network_dtype", "chunk", "evaluator_source_sha256", "workers", "jobs"),
          "components")
    expected = {"schema_version": 1, "purpose": "frozen_checkpoint_components", "job_count": 60, "worker_count": 4,
                "samples": 256, "stream_seeds": list(STREAM_SEEDS), "density_coordinates": "residual",
                "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32}
    _require(canonical_sha256({k: components[k] for k in expected}) == canonical_sha256(expected), "Invalid component protocol")
    _require(components["workers"] == index["components"], "Component input identities differ")
    _digests(components["evaluator_source_sha256"], EVALUATOR_SOURCES, "evaluator source identities")
    _require(index["evaluator_source_sha256"] == components["evaluator_source_sha256"],
             "Evaluator source differs")
    _keys(components["jobs"], indexed, "component jobs")
    for dataset, arm, seed, _ in canonical_jobs():
        identifier = legacy.experiment.job_id(dataset, arm, seed)
        record, job = components["jobs"][identifier], core["jobs"][identifier]
        _keys(record, (*RECORD_KEYS, "streams"), "public component record")
        _require((record["identifier"], record["dataset"], record["arm"], record["seed"], record["checkpoint"])
                 == (identifier, dataset, arm, seed, "selected"), "Public job identity differs")
        for key in ("seed", "selected_step", "steps_run", "dimension", "test_windows"):
            _integer(record[key], key)
        for key in ("best_step", "steps_run", "test_windows"):
            _integer(job[key], key)
        _require(record["selected_step"] == job["best_step"] and record["steps_run"] == job["steps_run"]
                 and 0 < record["selected_step"] <= record["steps_run"] <= 10000
                 and record["selected_step"] % 250 == 0, "Invalid selected step")
        n = core["dataset_statistics"][dataset]["windows"]["test"]
        _require(record["dimension"] == 192 and record["test_windows"] == n == job["test_windows"], "Invalid evaluation counts")
        _require(record["frozen_evaluator_reproduced_exactly"] is True, "Missing frozen evaluator comparison")
        _digests(job["artifact_sha256"], legacy.experiment.JOB_ARTIFACTS, "core job artifacts")
        for key in ("checkpoint_sha256", "core_completion_sha256", "prepared_sha256", "array_sha256", "core_evaluation_sha256"):
            legacy._digest(record[key])
        refined = refinement_jobs[identifier]
        _require(record["checkpoint_sha256"] == job["artifact_sha256"]["best.weights.h5"] == refined["core_checkpoint_sha256"]
                 and record["core_evaluation_sha256"] == job["artifact_sha256"]["evaluation.npz"] == refined["core_evaluation_sha256"]
                 and record["core_completion_sha256"] == refined["core_completion_sha256"]
                 and record["prepared_sha256"] == core["prepared_sha256"][dataset + ".npz"], "Public component input differs")
        _number(record["observation_variance"], "observation variance", 0)
        _close(record["observation_variance"], _variance(arm, core["dataset_statistics"][dataset]["base_variance"]), "public variance")
        shapes = {k: [3, n, 256] if k in ARRAY_KEYS[2:5] else [3, n, 16] if k == "kl_by_latent" else [3, n]
                  for k in ARRAY_KEYS}
        _require(canonical_sha256(record["array_shapes"]) == canonical_sha256(shapes), "Invalid public array shapes")
        _keys(record["archived_selected_replay_differences"], REPLAY_KEYS, "public replay")
        for value in record["archived_selected_replay_differences"].values():
            _number(value, "replay difference", 0)
        _validate_streams(record["streams"])
        for stream in record["streams"]:
            _close(stream["posterior_expected_mse"], 2 * record["observation_variance"] * (
                stream["posterior_expected_nll_per_coordinate"] - 0.5 * np.log(2 * np.pi * record["observation_variance"])),
                "public MSE units")
    if extension:
        from prigm_dart_paper_evidence import validate_dart_extension
        _require(core["run_manifest_sha256"] == DART_CORE_SHA256, "Approved DART core identity differs")
        validate_dart_extension(manifest["dart_extension"], core, index["dart_extension"])
        dart_index = index["dart_extension"]
        _require(dart_index["helper_sha256"] == index["source_sha256"]["src/prigm_dart_paper_evidence.py"],
                 "DART derivation helper identity differs")
        for name, digest in dart_index["reviewed_source_sha256"].items():
            _require(index["source_sha256"]["src/" + name] == digest, "DART reviewed source identity differs")
        return manifest
    _keys(manifest["dart_support"], DATASETS[1:], "DART support")
    for dataset, support in manifest["dart_support"].items():
        _keys(support, ("training_windows", "weighting", "power_units", "frequency_units", "one_cpd_share", "two_cpd_share",
                        "within_variance_m2", "between_variance_m2", "pooled_variance_m2", "within_to_pooled_ratio",
                        "frequency_cycles_per_day", "bin_power_m2"), "DART support record")
        _require(_integer(support["training_windows"], "training windows", 2)
                 == core["dataset_statistics"][dataset]["windows"]["train"], "Training window count differs")
        _require(support["weighting"] == "uniform_windows" and support["power_units"] == "m2"
                 and support["frequency_units"] == "cycles_per_day", "Invalid DART units or weighting")
        _close(support["frequency_cycles_per_day"], np.arange(1, 97) / 2, "positive DFT frequencies")
        power = legacy._array(support["bin_power_m2"], (96,), "physical bin power")
        _require(np.all(power >= 0), "Negative physical power")
        for key in ("within_variance_m2", "between_variance_m2", "pooled_variance_m2"):
            _number(support[key], key, 0)
        within, pooled = support["within_variance_m2"], support["pooled_variance_m2"]
        _require(within > 0 and pooled > 0, "Nonpositive training variance")
        _close(power.sum(), within, "public Parseval")
        _close(within + support["between_variance_m2"], pooled, "public within plus between")
        _close(support["within_to_pooled_ratio"], within / pooled, "public within fraction")
        _close(support["one_cpd_share"], power[1] / within, "1 cpd share")
        _close(support["two_cpd_share"], power[3] / within, "2 cpd share")
    _require(core["run_manifest_sha256"] == DART_CORE_SHA256, "Approved DART core identity differs")
    validate_dart_diagnostics(manifest["dart_diagnostics"], core)
    for dataset, record in manifest["dart_diagnostics"]["records"].items():
        training, support = record["train_distribution"], manifest["dart_support"][dataset]
        for key in ("within_variance_m2", "between_variance_m2", "pooled_variance_m2", "within_to_pooled_ratio"):
            _dart_close(training[key], support[key], "training support " + key)
        _dart_close(training["bin_power_m2"][1:], support["bin_power_m2"], "training support spectrum")
    return manifest


def _input_index(core_root, refinement_root, components_root, decision_file, numerical_root, *, dart_index=None):
    from prigm_refine_evaluation import MANIFEST

    digest = legacy.experiment.sha256
    if dart_index is None:
        protocol_file = legacy._safe_path(ROOT, DART_PROTOCOL_FILE)
        _require(digest(protocol_file) == DART_PROTOCOL_SHA256, "Approved DART protocol changed")
        for name, expected in legacy._read_json(protocol_file)["source_sha256"].items():
            _require(digest(legacy._safe_path(ROOT, name)) == expected, "Approved DART helper source changed")
    _require(digest(legacy._safe_path(core_root, "run_manifest.json")) == DART_CORE_SHA256,
             "Approved DART core identity differs")
    diagnostic_index = ({"dart_diagnostics": {"approved_protocol_sha256": DART_PROTOCOL_SHA256,
                                             "settings": deepcopy(DART_PROTOCOL)}} if dart_index is None
                        else {"dart_extension": deepcopy(dart_index)})
    source_ids = SOURCE_IDS if dart_index is None else DART_EXTENSION_SOURCE_IDS
    return {
        **diagnostic_index,
        "core": {"manifest_sha256": digest(legacy._safe_path(core_root, "run_manifest.json")),
                 "archive_manifest_sha256": digest(legacy._safe_path(core_root, "archive_manifest.json"))},
        "refinement": {"manifest_sha256": digest(legacy._safe_path(refinement_root, MANIFEST)),
                       "archive_manifest_sha256": digest(legacy._safe_path(refinement_root, "archive_manifest.json"))},
        "components": {worker: {"completion_sha256": digest(legacy._safe_path(components_root, worker + "/complete.json")),
                                 "protocol_sha256": digest(legacy._safe_path(components_root, worker + "/protocol.json"))}
                       for worker in WORKERS},
        "source_sha256": {name: digest(legacy._safe_path(ROOT, name)) for name in source_ids},
        "evaluator_source_sha256": dict(FROZEN_EVALUATOR_SHA256),
        "decision_sha256": digest(decision_file),
        "numerical_validation": _numerical_inputs(numerical_root),
    }


def build_paper(run_root, paper_dir, expected_core_sha256, refinement_root, components_root, decision_file,
                numerical_root, profile="corrected_audit", *, dart_core_diagnostics_roots=None,
                dart_stage_root=None, dart_stage_diagnostics_roots=None,
                expected_dart_stage_sha256=None, dart_selection_file=None):
    """Verify source-bound evidence, stage rendering, and publish the manifest last."""
    _require(profile in ("corrected_audit", "dart_extension"), "Unknown combined profile")
    extension = profile == "dart_extension"
    dart_arguments = (dart_core_diagnostics_roots, dart_stage_root, dart_stage_diagnostics_roots,
                      expected_dart_stage_sha256, dart_selection_file)
    _require(all(value is not None for value in dart_arguments) if extension else
             all(value is None for value in dart_arguments), "DART inputs require the complete dart_extension profile")
    outputs = DART_EXTENSION_OUTPUTS if extension else AUDIT_OUTPUTS
    core_root, refinement_root, components_root = map(_directory, (run_root, refinement_root, components_root))
    decision_file = Path(decision_file).absolute()
    _directory(decision_file.parent)
    legacy._safe_path(decision_file.parent, decision_file.name)
    numerical_snapshot = _numerical_snapshot(numerical_root, profile=profile)
    numerical_root = _directory(numerical_root)
    dart_public = None
    if extension:
        from prigm_dart_paper_evidence import OLD, PRIMARY, OPTIONAL, verify_dart_extension
        dart_public, dart_private = verify_dart_extension(
            core_root, dart_stage_root, dart_core_diagnostics_roots, dart_stage_diagnostics_roots,
            expected_dart_stage_sha256, dart_selection_file)
        _require(set(dart_public["records"]) == {*OLD, PRIMARY, *OPTIONAL},
                 "dart_extension paper requires all seven DART records")
        initial_index = _input_index(core_root, refinement_root, components_root, decision_file, numerical_root,
                                     dart_index=dart_public["input_index"])
    else:
        initial_index = _input_index(core_root, refinement_root, components_root, decision_file, numerical_root)
    decision = legacy._read_json(decision_file)
    _require(isinstance(decision, dict) and type(decision.get("schema_version")) is int and decision["schema_version"] == 1
             and decision.get("new_scientific_training_authorized_for_launch") is False,
             "Corrected audit requires the stop-scientific-matrix decision")
    legacy_policy = {"historical": True} if extension else {}
    core = legacy.verify_evidence(core_root, expected_core_sha256, **legacy_policy)
    legacy.verify_refinement_evidence(core_root, refinement_root, **legacy_policy)
    components, private_arrays = verify_components(components_root, core_root, core)
    del private_arrays  # Probability arrays are never serialized or used as plot shortcuts.
    if extension:
        numerical = verify_numerical_validation(numerical_root, profile=profile)
    else:
        numerical = verify_numerical_validation(numerical_root)
        support, diagnostics, private_diagnostics = dart_diagnostics(core_root, core)
    combined_sha = canonical_sha256(initial_index)
    destination = Path(paper_dir).absolute()
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)), "Symlink paper directory")
    for name in (*outputs, EVIDENCE_NAME):
        target = legacy._safe_path(destination, name)
        _require(not target.exists() or target.is_file(), "Output target is not a file")
    with tempfile.TemporaryDirectory(prefix="prigm-combined-private-") as temporary:
        stage = Path(temporary)
        old_public = legacy.build_paper(core_root, stage, expected_core_sha256, refinement_root, **legacy_policy)
        nested_core = {key: value for key, value in old_public.items() if key not in ("refinement", "output_sha256")}
        nested_core["summary_statistics"] = {key: old_public["summary_statistics"][key] for key in ("paired", "job_count")}
        if extension:
            fragments = _dart_table_fragments(components, dart_public, combined_sha, old_public)
            _dart_comparison_figure(stage, dart_public, combined_sha)
            _dart_extension_figures(stage, dart_public, dart_private, combined_sha)
            for name in ("generated/baseline_rows.tex", "generated/all_runs_rows.tex"):
                fragments[name] = "".join(line for line in (stage / name).read_text(encoding="ascii").splitlines(keepends=True)
                                          if line.startswith("Gaussian &"))
            diagnostic_fields = {"dart_extension": dart_public}
        else:
            _comparison_figure(stage, old_public["summary_statistics"]["paired"], combined_sha)
            fragments = table_fragments(components, support, diagnostics, combined_sha)
            _figures(stage, support, private_diagnostics, combined_sha)
            diagnostic_fields = {"dart_support": support, "dart_diagnostics": diagnostics}
        for name, text in fragments.items():
            (stage / name).write_text(text, encoding="ascii")
        public = {
            "schema_version": 3 if extension else 2, "purpose": "combined", "profile": profile,
            "combined_input_sha256": combined_sha, "input_index": initial_index,
            "builder_sha256": initial_index["source_sha256"]["src/build_prigm_v2_paper.py"],
            "core": nested_core, "refinement": old_public["refinement"], "components": components,
            **diagnostic_fields, "numerical_validation": numerical,
            "methods": dict(DART_EXTENSION_METHODS if extension else METHODS),
            "output_sha256": {name: legacy.experiment.sha256(stage / name) for name in outputs},
        }
        validate_combined_manifest(public, expected_core_sha256)
        # Rehash the transitive closures, not just their unchanged index files.
        for root in (core_root, refinement_root):
            legacy._verify_archive(root, legacy._read_json(root / "archive_manifest.json")["artifact_sha256"])
        _require({p.name for p in components_root.iterdir()} in (set(WORKERS), {*WORKERS, "src"}),
                 "Component worker inventory changed during rendering")
        for index, worker in enumerate(WORKERS):
            directory = _directory(legacy._safe_path(components_root, worker))
            artifact_names = {legacy.experiment.job_id(*job[:3]) + "-selected" + suffix
                              for job in canonical_jobs()[index::4] for suffix in (".npz", ".json")}
            _inventory(directory, {*artifact_names, "source", "protocol.json", "complete.json"})
            _inventory(legacy._safe_path(directory, "source"), EVALUATOR_SOURCES)
            legacy._verify_archive(directory, legacy._read_json(directory / "complete.json")["artifact_sha256"])
            legacy._verify_archive(directory / "source", components["evaluator_source_sha256"])
        _launcher_sources(components_root, components["evaluator_source_sha256"],
                          core["code_sha256"]["src/prigm_metrics.py"])
        _require(_numerical_snapshot(numerical_root, profile=profile) == numerical_snapshot,
                 "Numerical inputs changed during rendering")
        if extension:
            final_dart, _ = verify_dart_extension(
                core_root, dart_stage_root, dart_core_diagnostics_roots, dart_stage_diagnostics_roots,
                expected_dart_stage_sha256, dart_selection_file)
            _same_json(final_dart, dart_public, "DART evidence changed during rendering")
            final_index = _input_index(core_root, refinement_root, components_root, decision_file, numerical_root,
                                       dart_index=final_dart["input_index"])
        else:
            final_index = _input_index(core_root, refinement_root, components_root, decision_file, numerical_root)
        _require(final_index == initial_index, "Combined inputs changed during rendering")
        (stage / EVIDENCE_NAME).write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
        # A sibling stage keeps os.replace on the destination filesystem.
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prigm-combined-", dir=destination) as publishing:
            publish = Path(publishing)
            for name in (*outputs, EVIDENCE_NAME):
                target = legacy._safe_path(publish, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((stage / name).read_bytes())
            for name in (*outputs, EVIDENCE_NAME):
                target = legacy._safe_path(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(publish / name, target)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "refinement-root", "components-root", "decision-file", "numerical-root", "paper-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--profile", choices=("corrected_audit", "dart_extension"), default="corrected_audit")
    parser.add_argument("--dart-core-diagnostics-roots", type=Path, nargs="+")
    parser.add_argument("--dart-stage-diagnostics-roots", type=Path, nargs="+")
    parser.add_argument("--dart-stage-root", type=Path)
    parser.add_argument("--expected-dart-stage-sha256")
    parser.add_argument("--dart-selection-file", type=Path)
    args = parser.parse_args(argv)
    extra = {name: getattr(args, name) for name in (
        "dart_core_diagnostics_roots", "dart_stage_root", "dart_stage_diagnostics_roots",
        "expected_dart_stage_sha256", "dart_selection_file")}
    if args.profile == "dart_extension":
        if any(value is None for value in extra.values()):
            parser.error("dart_extension requires all five DART input arguments")
        result = build_paper(args.run_root, args.paper_dir, args.expected_core_sha256,
                             args.refinement_root, args.components_root, args.decision_file, args.numerical_root,
                             profile=args.profile, **extra)
    else:
        if any(value is not None for value in extra.values()):
            parser.error("DART inputs require --profile dart_extension")
        result = build_paper(args.run_root, args.paper_dir, args.expected_core_sha256,
                             args.refinement_root, args.components_root, args.decision_file, args.numerical_root)
    outputs = DART_EXTENSION_OUTPUTS if args.profile == "dart_extension" else AUDIT_OUTPUTS
    print(json.dumps({"combined_input_sha256": result["combined_input_sha256"],
                      "artifacts": [*outputs, EVIDENCE_NAME]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
