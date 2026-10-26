"""v2 staged experiment: fixed updates, probes, evaluation, dispatcher.

Contract summary (frozen before any fit):
- Exactly 10,000 updates; batch 64; Adam lr 1e-3, beta1 .9, beta2 .999,
  eps 1e-7, amsgrad False; raw global-norm cap 10 across all groups; no L2,
  no warm-up, no early stopping.
- Training stream: np.random.default_rng(seed + 50000); probe/prior/
  validation/evaluation streams are separate and fixed below.
- Every update logs the pre-clipping global gradient norm (in optimizer
  coordinates) and whether clipping was active.
- Probes on updates 1-250 and every 1000: 64 fixed training windows, 128
  fixed latent vectors; full and head-only decoder-mean RMS displacement;
  encoder symmetric KL on the fixed windows.
- Validation energy every 1000 updates with 512 fixed prior draws; selected
  checkpoint keeps the earlier step on exact ties.
- Primary evaluation at the final update: joint energy (1024 draws x 3
  streams), increment energy, posterior-code reassignment, K=256 IW x 3
  streams, and the analytic linear controls at K = 64/256/1024.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

# Direct snapshot CLI launches, like campaign workers, must not mutate source closure.
sys.dont_write_bytecode = True
from prigm_metrics import importance_statistics

ROOT = Path(__file__).resolve().parents[1]
V2_CODE_FILES = ("src/prigm_v2_data.py", "src/prigm_v2_model.py",
                 "src/prigm_v2_metrics.py", "src/prigm_v2_experiment.py")
# The v2 experiment imports frozen v1 metrics; the stage snapshot must carry
# them so the released closure is self-contained.
DEPS_V1 = ("src/prigm_metrics.py",)
DATASETS = ("gaussian_v2", "mixture_v2", "dart_44402_2018")
CONDITIONS = ("base", "raw", "canonical")
VARIANCES = {"low": 0.0625, "high": 1.5625}
DART_VARIANCES = {"low": 0.0001, "high": 0.0025}  # selected base and 25x
STAGES = {
    "numerical_null": [("gaussian_v2", condition, 42, variance)
                       for variance in VARIANCES for condition in ("base", "canonical")],
}
SPEC = {
    "window": 192, "latent_dim": 16, "synthetic_latent_dim": 8,
    "encoder_widths": [128, 64], "decoder_widths": [64, 128],
    "batch_size": 64, "learning_rate": 0.001,
    "adam": {"beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7, "amsgrad": False},
    "kl_beta": 1.0, "max_steps": 10000,
    "model_storage_contract": 3, "purpose": "numerical_validation",
    "gradient_clipping_enabled": True, "gradient_global_norm": 10.0,
    "posterior_log_variance_bounds": [-12.0, 12.0],
    "synthetic_observation_variances": [0.0625, 1.5625],
    "validation_every": 1000, "validation_prior_samples": 512,
    "evaluation_prior_samples": 1024, "evaluation_streams": 3,
    "iw_samples": 256, "iw_streams": 3, "linear_control_k": [64, 256, 1024],
    "probe_windows": 64, "probe_latents": 128,
    "probe_steps": sorted(set(range(1, 251)) | set(range(1000, 10001, 1000))),
    "head_rate_multiplier": "b",
    "training_seed_offset": 50000,
    "probe_seed": 61001, "prior_seed": 62001, "validation_seed": 63001,
    "evaluation_seed": 64001, "iw_seed": 65001, "reassignment_seed": 66001,
    "exact_resume_supported": False,
}
REQUIRED_JOB_ARTIFACTS = ("configuration.json", "history.jsonl", "probes.jsonl",
                           "final.weights.h5", "selected.weights.h5", "summary.json",
                           "evaluation-final.npz", "progress.json")
OPTIONAL_JOB_ARTIFACTS = ("evaluation-selected.npz",)
JOB_ARTIFACTS = REQUIRED_JOB_ARTIFACTS + OPTIONAL_JOB_ARTIFACTS + ("complete.json",)


def sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular artifact: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return _parse_json(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, record):
    path = Path(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


# ---- stage manifest --------------------------------------------------------


def job_parameters(stage):
    """Closed numerical-validation matrix, not a research-stage registry."""
    return {job_id(*job): {"observation_variance": VARIANCES[job[3]],
                          "gradient_clipping_enabled": SPEC["gradient_clipping_enabled"],
                          "gradient_global_norm": SPEC["gradient_global_norm"]}
            for job in STAGES[stage]}


def _same(actual, expected, label):
    # JSON equality preserves bool/int distinctions in frozen identities.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"Inconsistent {label}")


DART_STAGE = "dart_record_extension"
DART_PURPOSE = "dart_input_skip_evaluation"
DART_NOISE_GRID = (1e-4, 5e-4, 2.5e-3, 1.25e-2, 6.25e-2)


def _resolved_spec(stage):
    """Return an isolated specification; never change the numerical-null defaults."""
    spec = deepcopy(SPEC)
    if stage == DART_STAGE:
        spec["purpose"] = DART_PURPOSE
    elif stage not in STAGES:
        raise ValueError("Unknown stage in manifest")
    return spec


def _dart_records(directory):
    path = directory / "records.json"
    sha256(path)
    metadata = read_json(path)
    if not isinstance(metadata, dict) or set(metadata) != {"schema_version", "records"}:
        raise ValueError("Invalid DART records metadata fields")
    _same(metadata["schema_version"], 1, "records schema")
    records = metadata["records"]
    if (not isinstance(records, dict) or not 1 <= len(records) <= 3
            or "dart_44402_2018" not in records):
        raise ValueError("DART requires primary 44402/2018 and at most two other sites")
    fields = {"station", "interval_start", "interval_end", "raw_sha256", "raw_filename",
              "deployment", "qualification", "selection_status"}
    for name, record in records.items():
        if (re.fullmatch(r"dart_[0-9]{5}_2018", name) is None
                or name[5:10] in {"23461", "46407", "52403"}
                or not isinstance(record, dict) or set(record) != fields):
            raise ValueError("Invalid DART record name or metadata fields")
        if record["station"] != name[5:10]:
            raise ValueError("DART station does not match dataset name")
        # Dates identify a complete UTC calendar year, not an arbitrary subinterval.
        _same([record["interval_start"], record["interval_end"]],
              ["2018-01-01", "2019-01-01"], "DART whole-year interval")
        if (not isinstance(record["raw_sha256"], str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", record["raw_sha256"]) is None
                or any(not isinstance(record[key], str) or not record[key].strip()
                       for key in ("raw_filename", "selection_status"))
                or any(not isinstance(record[key], dict) for key in ("deployment", "qualification"))):
            raise ValueError("Invalid DART provenance metadata")
        # raw_filename is provenance only and is never resolved or opened.
        _finite(record, "DART provenance")
    return metadata


def _dart_prepared(path, name, spec):
    """Validate prepared supports, not raw acquisition/deployment qualification."""
    from prigm_v2_data import same_phase_donors
    splits = ("train", "validation", "test")
    fields = {"dataset_name", "mean", "scale", "train_weights", "validation_weights",
              "eval_weights", "reassignment_block_origin", "reassignment_available",
              "reassignment_donors", *splits,
              *(f"starts_{split}" for split in splits),
              *(f"source_starts_{split}" for split in splits)}
    sha256(path)
    with np.load(path, allow_pickle=False) as store:
        if set(store.files) != fields or len(store.files) != len(fields):
            raise ValueError("Incorrect DART prepared array inventory")
        data = dict(store)
    if data["dataset_name"].shape != () or data["dataset_name"].item() != name:
        raise ValueError("Prepared dataset name mismatch")
    for key in fields - {"dataset_name"}:
        value = data[key]
        if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite DART array: {key}")
    for key in ("mean", "scale"):
        if data[key].shape != () or data[key].dtype.kind not in "fi":
            raise ValueError("Invalid training normalization scalar")
    if data["scale"].item() <= 0:
        raise ValueError("Training scale must be positive")
    origin = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
    day, span = 86400, 191 * 900
    bounds = ((origin, origin + 217 * day),
              (origin + 221 * day, origin + 290 * day),
              (origin + 294 * day, origin + 365 * day))
    summaries = {}
    previous_time = previous_source = -1
    for split, (lower, upper), minimum in zip(splits, bounds, (60, 20, 20)):
        x, starts, sources = data[split], data[f"starts_{split}"], data[f"source_starts_{split}"]
        n = len(x) if x.ndim else 0
        if x.shape != (n, 192) or n < minimum or x.dtype.kind not in "fi":
            raise ValueError(f"Invalid DART {split} window shape/count")
        for key, values in (("starts", starts), ("source_starts", sources)):
            if (values.shape != (n,) or values.dtype.kind not in "iu"
                    or np.any(values[1:] <= values[:-1])):
                raise ValueError(f"Invalid monotonic {split} {key}")
        if (np.any(starts % 14400 != 0) or starts[0] < lower
                or int(starts[-1]) + span >= upper or starts[0] <= previous_time
                or sources[0] <= previous_source or int(sources[-1]) > np.iinfo(np.int64).max - 192):
            raise ValueError("Invalid DART split time/source support or embargo")
        previous_time, previous_source = int(starts[-1]) + span, int(sources[-1]) + 191
        weights = data["eval_weights" if split == "test" else f"{split}_weights"]
        bins = ((starts % day) // 14400).astype(int)
        counts = np.bincount(bins, minlength=6)
        if (weights.shape != (n,) or np.any(weights <= 0) or np.any(counts == 0)
                or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
                or not np.allclose(weights, 1.0 / (6 * counts[bins]), rtol=0, atol=1e-12)):
            raise ValueError(f"Invalid DART {split} UTC-bin weights")
        packs, time_end, source_end = 0, -1, -1
        for start, source in zip(starts, sources):
            if start > time_end and source > source_end:
                packs += 1
                time_end, source_end = int(start) + span, int(source) + 191
        if packs < minimum:
            raise ValueError(f"Insufficient DART {split} disjoint-support packs")
        summaries[split] = {"windows": n, "packs": packs, "utc_bin_counts": counts.tolist()}
    if (data["reassignment_block_origin"].shape != ()
            or data["reassignment_block_origin"].dtype.kind not in "iu"
            or data["reassignment_block_origin"].item() != origin
            or data["reassignment_available"].shape != ()
            or data["reassignment_available"].dtype.kind != "b"):
        raise ValueError("Invalid DART reassignment identity")
    donors = same_phase_donors(data["starts_test"], spec["reassignment_seed"],
                              data["source_starts_test"], origin)
    expected = np.empty(0, dtype=np.int64) if donors is None else donors
    if (data["reassignment_available"].item() != (donors is not None)
            or data["reassignment_donors"].dtype.kind not in "iu"
            or not np.array_equal(data["reassignment_donors"], expected)):
        raise ValueError("Invalid DART reassignment support")
    return data, summaries


def _dart_design(directory, records, spec):
    from prigm_v2_metrics import select_noise_variance
    selections, summaries = {}, {}
    for name in sorted(records):
        data, summaries[name] = _dart_prepared(directory / f"{name}.npz", name, spec)
        result = select_noise_variance(data["train"], data["train_weights"],
                                       data["validation"], data["validation_weights"],
                                       grid=DART_NOISE_GRID, rank=16)
        selections[name] = {key: result[key] for key in
                            ("selection", "grid_results", "boundary_selection")}
        _finite(selections[name], "PPCA variance selection")
    jobs = [[name, condition, seed, label] for name in sorted(records)
            for seed in (42, 1729, 31415) for label in ("low", "high")
            for condition in ("base", "raw")]
    parameters = {job_id(*job): {
        "observation_variance": selections[job[0]]["selection"]["variance"] * (25 if job[3] == "high" else 1),
        "gradient_clipping_enabled": spec["gradient_clipping_enabled"],
        "gradient_global_norm": spec["gradient_global_norm"]} for job in jobs}
    return jobs, parameters, selections, summaries


def build_dart_stage(run_root, dataset_path):
    """Freeze a fresh bounded DART stage from prepare_dart_record NPZs and records.json.

    Raw hashes/deployment/qualification are bound provenance supplied by the caller;
    acquisition and independent raw-data qualification remain outside this runner.
    """
    root, directory = Path(run_root).absolute(), Path(dataset_path)
    if root.exists():
        raise FileExistsError("Use a fresh stage directory")
    metadata = _dart_records(directory)
    names = set(metadata["records"])
    expected = {"records.json", *(f"{name}.npz" for name in names)}
    if (directory.is_symlink() or {p.name for p in directory.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in directory.iterdir())):
        raise ValueError("Incorrect DART input directory inventory")
    # Validate before creating a stage; then bind the exact copied bytes, not caller paths.
    spec = _resolved_spec(DART_STAGE)
    for name in sorted(names):
        _dart_prepared(directory / f"{name}.npz", name, spec)
    root.mkdir(parents=True)
    (root / "source" / "src").mkdir(parents=True)
    for folder in ("prepared", "jobs", "logs"):
        (root / folder).mkdir()
    for name in (*V2_CODE_FILES, *DEPS_V1):
        shutil.copyfile(ROOT / name, root / "source" / name)
    for name in sorted(expected):
        shutil.copyfile(directory / name, root / "prepared" / name)
    copied = _dart_records(root / "prepared")
    _same(copied, metadata, "copied DART metadata")
    jobs, parameters, selections, summaries = _dart_design(root / "prepared", copied["records"], spec)
    manifest = {
        "schema_version": 4, "stage": DART_STAGE, "purpose": DART_PURPOSE,
        "run_id": str(uuid.uuid4()), "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs, "job_parameters": parameters, "specification": spec,
        "code_sha256": {name: sha256(root / "source" / name) for name in (*V2_CODE_FILES, *DEPS_V1)},
        "prepared_sha256": {name: sha256(root / "prepared" / f"{name}.npz") for name in sorted(names)},
        "records_sha256": sha256(root / "prepared" / "records.json"),
        "variance_selection": selections, "record_summaries": summaries,
    }
    atomic_json(root / "stage_manifest.json", manifest)
    print(json.dumps({"stage_root": str(root), "stage": DART_STAGE,
                      "manifest_sha256": sha256(root / "stage_manifest.json")}, indent=2))


def build_stage(run_root, stage, dataset_path, purpose="numerical_validation"):
    """Create a fresh stage directory with source snapshot and input hashes.

    dataset_path points to a serialized prepared .npz per dataset name.
    The stage manifest freezes source identity and inputs; workers launch
    from this snapshot, never from the working tree.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if purpose != "numerical_validation":
        raise ValueError("Only numerical_validation stages are enabled")
    root = Path(run_root).absolute()
    if root.exists():
        raise FileExistsError("Use a fresh stage directory")
    jobs = STAGES[stage]
    datasets = sorted({j[0] for j in jobs})
    root.mkdir(parents=True)
    (root / "source" / "src").mkdir(parents=True)
    (root / "prepared").mkdir()
    (root / "jobs").mkdir()
    (root / "logs").mkdir()
    for name in (*V2_CODE_FILES, *DEPS_V1):
        shutil.copyfile(ROOT / name, root / "source" / name)
    prepared = {}
    dataset_dir = Path(dataset_path) if dataset_path else ROOT / "runs-v2"
    for name in datasets:
        source = dataset_dir / f"{name}.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Missing prepared dataset: {source}")
        shutil.copyfile(source, root / "prepared" / f"{name}.npz")
        prepared[name] = sha256(root / "prepared" / f"{name}.npz")
    spec = dict(SPEC)
    manifest = {
        "schema_version": 3, "stage": stage, "purpose": purpose,
        "run_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [list(j) for j in jobs],
        "job_parameters": job_parameters(stage),
        "specification": spec,
        "code_sha256": {name: sha256(root / "source" / name)
                        for name in (*V2_CODE_FILES, *DEPS_V1)},
        "prepared_sha256": prepared,
    }
    atomic_json(root / "stage_manifest.json", manifest)
    print(json.dumps({"stage_root": str(root), "stage": stage,
                      "manifest_sha256": sha256(root / "stage_manifest.json")}, indent=2))


def verify_stage(root, check_code=True, *, historical_identity=None):
    """Verify archive bytes, with explicit approved producer pins for historical reads.

    Current derivation authentication belongs to the evidence reader supplying
    historical_identity. All training/publication callers retain strict defaults.
    """
    root = Path(root).resolve()
    sha256(root / "stage_manifest.json")
    manifest = read_json(root / "stage_manifest.json")
    stage = manifest.get("stage")
    dart = stage == DART_STAGE
    if manifest.get("schema_version") != (4 if dart else 3):
        raise ValueError("Unsupported stage schema; old stages require archived-source replay")
    spec = _resolved_spec(stage)
    if dart:
        _same(manifest["schema_version"], 4, "DART stage schema")
    fields = {"schema_version", "stage", "purpose", "run_id", "created_utc", "jobs",
              "job_parameters", "specification", "code_sha256", "prepared_sha256"}
    if dart:
        fields.update(("records_sha256", "variance_selection", "record_summaries"))
    _same(sorted(manifest), sorted(fields), "manifest fields")
    _same(manifest["purpose"], spec["purpose"], "stage purpose")
    _same(manifest["specification"], spec, "frozen specification")
    if dart:
        if sha256(root / "prepared" / "records.json") != manifest["records_sha256"]:
            raise ValueError("DART records metadata changed")
        records = _dart_records(root / "prepared")["records"]
        datasets = set(records)
    else:
        _same(manifest["jobs"], [list(job) for job in STAGES[stage]], "stage job matrix")
        _same(manifest["job_parameters"], job_parameters(stage), "job parameters")
        datasets = {job[0] for job in STAGES[stage]}
    if str(uuid.UUID(manifest["run_id"])) != manifest["run_id"]:
        raise ValueError("Invalid run identity")
    datetime.fromisoformat(manifest["created_utc"])
    code = set((*V2_CODE_FILES, *DEPS_V1))
    if set(manifest["code_sha256"]) != code:
        raise ValueError("Incomplete v2 source identity")
    if historical_identity is not None:
        if (check_code is not True or type(historical_identity) is not dict
                or set(historical_identity) != {"manifest_sha256", "code_sha256"}
                or sha256(root / "stage_manifest.json") != historical_identity["manifest_sha256"]
                or manifest["code_sha256"] != historical_identity["code_sha256"]):
            raise ValueError("Historical stage identity differs from approved pins")
    if set(manifest["prepared_sha256"]) != datasets:
        raise ValueError("Incorrect prepared dataset set")
    prepared_files = {f"{name}.npz" for name in datasets} | ({"records.json"} if dart else set())
    for folder, expected in (("source", code), ("prepared", prepared_files)):
        base = root / folder
        entries = list(base.rglob("*"))
        if base.is_symlink() or any(p.is_symlink() for p in entries):
            raise ValueError(f"Symlink in stage {folder}")
        directories = {str(Path(name).parent) for name in expected} - {"."}
        if {p.relative_to(base).as_posix() for p in entries} != expected | directories:
            raise ValueError(f"Incorrect stage {folder} inventory")
        if any(not (base / name).is_dir() for name in directories):
            raise ValueError(f"Invalid stage {folder} directory")
    for name in (*V2_CODE_FILES, *DEPS_V1):
        expected = manifest["code_sha256"][name]
        if sha256(root / "source" / name) != expected:
            raise ValueError(f"Archived v2 source changed: {name}")
        if check_code and historical_identity is None and sha256(ROOT / name) != expected:
            raise ValueError(f"v2 source changed after stage build: {name}")
    for name in datasets:
        if sha256(root / "prepared" / f"{name}.npz") != manifest["prepared_sha256"][name]:
            raise ValueError(f"Prepared input changed: {name}")
    if dart:
        jobs, parameters, selections, summaries = _dart_design(root / "prepared", records, spec)
        _same(manifest["jobs"], jobs, "DART stage job matrix")
        _same(manifest["job_parameters"], parameters, "DART job parameters")
        _same(manifest["record_summaries"], summaries, "DART prepared support counts")
        # Permit only floating-point reduction roundoff across CPU/BLAS runtimes.
        # Grid, ranks, selected variance and boundary flag remain exact identities.
        recorded = deepcopy(manifest["variance_selection"])
        if not isinstance(recorded, dict) or set(recorded) != set(selections):
            raise ValueError("Incorrect DART variance selection inventory")
        for name, expected in selections.items():
            actual = recorded[name]
            if (not isinstance(actual, dict) or set(actual) != set(expected)
                    or not isinstance(actual["grid_results"], list)
                    or len(actual["grid_results"]) != len(DART_NOISE_GRID)):
                raise ValueError("Incorrect DART variance grid fields")
            for row, expected_row in zip([actual["selection"], *actual["grid_results"]],
                                          [expected["selection"], *expected["grid_results"]]):
                key = "weighted_validation_log_density"
                if (not isinstance(row, dict) or set(row) != set(expected_row)
                        or type(row[key]) not in (int, float) or not np.isfinite(row[key])
                        or not np.isclose(row[key], expected_row[key], rtol=1e-10, atol=1e-8)):
                    raise ValueError("Inconsistent DART PPCA validation density")
                row[key] = expected_row[key]
        _same(recorded, selections, "DART pre-fit variance selection")
    return manifest


def _matrix_job(manifest, identifier):
    # Entry points verify the complete derived matrix before consulting this list.
    for job in manifest["jobs"]:
        if job_id(*job) == identifier:
            return job
    raise ValueError("Job outside the frozen stage matrix")


def _configuration_identity(root, manifest, identifier, attempt):
    dataset, condition, seed, label = _matrix_job(manifest, identifier)
    variance = manifest["job_parameters"][identifier]["observation_variance"]
    spec = manifest["specification"]
    alpha = 0.0 if condition == "base" else 0.8
    b = 1.0 - alpha
    # These are the only admitted heads. Match condition_settings without importing TF.
    coordinates, scale = ("r", b) if condition == "raw" else ("w", 1.0)
    return {"schema_version": manifest["schema_version"], "purpose": manifest["purpose"],
            "run_id": manifest["run_id"], "identifier": identifier, "attempt": attempt,
            "dataset": dataset, "condition": condition, "seed": seed,
            "variance_label": label, "observation_variance": variance,
            "alpha": alpha, "b": b, "residual_variance": b**2 * variance,
            "model_storage_contract": 3, "head_coordinates": coordinates, "head_scale": scale,
            "model_dtype": "float32", "head_learning_rate": float(np.float32(spec["learning_rate"])),
            "body_learning_rate": float(np.float32(spec["learning_rate"])),
            "epsilon": spec["adam"]["epsilon"],
            "epsilon_float32": float(np.float32(spec["adam"]["epsilon"])),
            "gradient_clipping_enabled": spec["gradient_clipping_enabled"],
            "gradient_global_norm": spec["gradient_global_norm"],
            "specification": spec, "code_sha256": manifest["code_sha256"],
            "prepared_sha256": manifest["prepared_sha256"][dataset],
            "stage_manifest_sha256": sha256(root / "stage_manifest.json")}


def _finite(value, label):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item, label)
    elif isinstance(value, list):
        for item in value:
            _finite(item, label)
    elif isinstance(value, (float, int)) and not np.isfinite(value):
        raise ValueError(f"Nonfinite {label}")


def _rows(path):
    rows = [_parse_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid rows: {path.name}")
    _finite(rows, path.name)
    return rows


def _check_history(attempt, spec=None):
    spec = SPEC if spec is None else spec
    rows = iter(_rows(attempt / "history.jsonl"))
    best_score, best_step = float("inf"), 0
    training_fields = {"step", "pre_update_loss", "pre_update_minibatch_kl",
                       "gradient_norm", "clipping_enabled", "clipping_scale", "clipping_active"}
    for step in range(1, spec["max_steps"] + 1):
        row = next(rows, {})
        if set(row) != training_fields or type(row["step"]) is not int or row["step"] != step:
            raise ValueError(f"Invalid ordered training history at step {step}")
        for key in ("pre_update_loss", "pre_update_minibatch_kl", "gradient_norm", "clipping_scale"):
            if type(row[key]) not in (int, float):
                raise ValueError(f"Invalid training numeric field: {key}")
        norm, scale = row["gradient_norm"], row["clipping_scale"]
        cap = np.float32(spec["gradient_global_norm"])
        expected_scale = float(cap / max(np.float32(norm), cap))
        if (norm < 0 or not 0 < scale <= 1 or row["clipping_enabled"] is not True
                or row["clipping_active"] is not (scale < 1)
                or row["clipping_active"] is not (expected_scale < 1)
                or not np.isclose(scale, expected_scale, rtol=2e-7, atol=0)):
            raise ValueError(f"Inconsistent clipping history at step {step}")
        if step % spec["validation_every"] == 0:
            row = next(rows, {})
            if (set(row) != {"step", "validation_energy", "best_step"}
                    or type(row["step"]) is not int or row["step"] != step
                    or type(row["validation_energy"]) not in (int, float)):
                raise ValueError(f"Invalid scheduled validation history at step {step}")
            if row["validation_energy"] < best_score:
                best_score, best_step = row["validation_energy"], step
            _same(row["best_step"], best_step, "earlier-tie validation selection")
    if next(rows, None) is not None:
        raise ValueError("Unexpected trailing history rows")
    probes = _rows(attempt / "probes.jsonl")
    _same([row.get("step") for row in probes], spec["probe_steps"], "probe schedule")
    for row in probes:
        if set(row) != {"step", "full_decoder_rms", "head_only_rms", "encoder_symmetric_kl"}:
            raise ValueError("Invalid probe fields")
        if any(type(row[key]) not in (int, float) or row[key] < -1e-10
               for key in ("full_decoder_rms", "head_only_rms", "encoder_symmetric_kl")):
            raise ValueError("Invalid probe values")
    progress = read_json(attempt / "progress.json")
    _finite(progress, "progress")
    _same({key: progress.get(key) for key in ("step", "best_step", "best_score")},
          {"step": spec["max_steps"], "best_step": best_step, "best_score": best_score},
          "end progress")
    if (set(progress) != {"step", "best_step", "best_score", "elapsed_seconds"}
            or type(progress["elapsed_seconds"]) not in (int, float)
            or progress["elapsed_seconds"] < 0):
        raise ValueError("Invalid progress fields")
    return best_step, best_score


def _check_weights(path, configuration):
    """Inspect HDF5, without importing TensorFlow or allocating a GPU."""
    # Reject fake bytes even on machines without the optional HDF5 reader.
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError("Expected genuine HDF5 weights")
    import h5py
    expected_pairs = {"layers/dense": (192, 128), "layers/dense_1": (128, 64),
                      "layers/dense_2": (16, 64), "layers/dense_3": (64, 128),
                      "mean_head": (64, 16), "logvar_head": (64, 16),
                      "output_head": (128, 192)}
    expected_groups = {"", "layers", "vars", *expected_pairs,
                       *(name + "/vars" for name in expected_pairs)}
    with h5py.File(path, "r") as store:
        for key, expected in {"storage_contract": configuration["model_storage_contract"],
                              "head_coordinates": configuration["head_coordinates"],
                              "head_scale": configuration["head_scale"],
                              "observation_variance": configuration["observation_variance"]}.items():
            if key not in store.attrs or store.attrs[key] != expected:
                raise ValueError(f"Incompatible weight storage contract: {key}")
        pairs = {}
        seen = {h5py.h5o.get_info(store.id).addr}
        groups = set()

        def inspect(group):
            group_name = group.name.lstrip("/")
            if group_name not in expected_groups:
                raise ValueError("Incorrect Keras weight group inventory")
            groups.add(group_name)
            datasets = []
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("External/soft weight links are not self-contained")
                item = group[name]
                address = h5py.h5o.get_info(item.id).addr
                if address in seen:
                    raise ValueError("Aliased/cyclic weight objects are not permitted")
                seen.add(address)
                if isinstance(item, h5py.Group):
                    inspect(item)
                else:
                    if item.is_virtual or item.external:
                        raise ValueError("Weight tensors must be stored inside the checkpoint")
                    value = item[()]
                    if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                        raise ValueError("Invalid weight tensor dtype or value")
                    datasets.append((name, value.shape))
            if datasets:
                # Keras .weights.h5 stores each Dense's kernel/bias as vars/0, vars/1.
                shapes = dict(datasets)
                if (set(shapes) != {"0", "1"} or len(shapes["0"]) != 2
                        or shapes["1"] != (shapes["0"][1],)):
                    raise ValueError("Invalid Dense weight tensor inventory")
                if not group_name.endswith("/vars"):
                    raise ValueError("Incorrect Keras weight tensor location")
                pairs[group_name[:-5]] = shapes["0"]

        inspect(store)
        if pairs != expected_pairs or groups != expected_groups:
            raise ValueError("Incorrect model weight tensor inventory")


def check_attempt(root, identifier, attempt, *, historical_identity=None):
    """Validate a restart-only attempt BEFORE publishing any completion marker.

    Reuse runs this same closure, including all rows, saved reductions and
    finite contract-3 HDF5 tensors. No optimizer/RNG resume is supported.
    """
    root, attempt = Path(root).resolve(), Path(attempt)
    manifest = verify_stage(root, historical_identity=historical_identity)
    _matrix_job(manifest, identifier)
    job_root = root / "jobs" / identifier
    if (job_root.is_symlink() or (root / "jobs").is_symlink()
            or attempt.parent != job_root or attempt.is_symlink() or not attempt.is_dir()
            or not attempt.name.startswith("attempt-")
            or len(attempt.name) != 40
            or any(c not in "0123456789abcdef" for c in attempt.name[8:])):
        raise ValueError("Invalid job attempt path")
    artifacts = list(attempt.iterdir())
    entries = {p.name for p in artifacts}
    if (not set(REQUIRED_JOB_ARTIFACTS) <= entries or entries - set(JOB_ARTIFACTS)
            or any(p.is_symlink() or not p.is_file() for p in artifacts)):
        raise ValueError("Incorrect attempt artifact inventory")
    configuration = read_json(attempt / "configuration.json")
    _finite(configuration, "configuration")
    expected = _configuration_identity(root, manifest, identifier, attempt.name)
    _same({key: value for key, value in configuration.items() if key != "runtime"},
          expected, "configuration identity/model storage/optimizer")
    runtime = configuration.get("runtime")
    runtime_fields = {"gpu_uuid", "pid", "placement", "process_inventory", "tensorflow",
                      "python", "numpy", "deterministic_ops_requested"}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise ValueError("Missing runtime attestation")
    if (any(not isinstance(runtime[key], str) or not runtime[key]
            for key in ("gpu_uuid", "placement", "tensorflow", "python", "numpy"))
            or type(runtime["pid"]) is not int or runtime["pid"] <= 0
            or "GPU:0" not in runtime["placement"]
            or runtime["deterministic_ops_requested"] is not True
            or not isinstance(runtime["process_inventory"], list)
            or not runtime["process_inventory"]
            or any(not isinstance(line, str) for line in runtime["process_inventory"])
            or not any(line.split(",", 1)[0].strip() == str(runtime["pid"])
                       and runtime["gpu_uuid"] in line for line in runtime["process_inventory"])):
        raise ValueError("Inconsistent runtime attestation")
    spec = manifest["specification"]
    best_step, best_score = _check_history(attempt, spec)
    summary = read_json(attempt / "summary.json")
    _finite(summary, "summary")
    _same(sorted(summary), sorted(("identifier", "attempt", "run_id", "selected_step",
                                   "best_score", "final", "selected")), "summary fields")
    _same({key: summary.get(key) for key in ("identifier", "attempt", "run_id", "selected_step", "best_score")},
          {"identifier": identifier, "attempt": attempt.name, "run_id": manifest["run_id"],
           "selected_step": best_step, "best_score": best_score}, "summary selection/identity")
    earlier = best_step < spec["max_steps"]
    required = set(REQUIRED_JOB_ARTIFACTS) | ({"evaluation-selected.npz"} if earlier else set())
    if earlier and "evaluation-selected.npz" not in entries:
        raise ValueError(f"Missing selected evaluation: {identifier}")
    if entries - {"complete.json"} != required:
        raise ValueError("Incorrect attempt artifact inventory")
    recorded = {name: sha256(attempt / name) for name in sorted(required)}
    with np.load(root / "prepared" / f'{configuration["dataset"]}.npz', allow_pickle=False) as store:
        data = {key: store[key] for key in store.files}
    for label, step in (("final", spec["max_steps"]), ("selected", best_step)):
        if label == "selected" and not earlier:
            if summary.get("selected") is not None:
                raise ValueError("Unexpected selected evaluation summary")
            break
        result = summary.get(label)
        if not isinstance(result, dict):
            raise ValueError(f"Missing {label} checkpoint summary")
        _same({key: result.get(key) for key in ("identifier", "seed", "checkpoint", "step",
                                               "weights_sha256", "evaluation_sha256")},
              {"identifier": identifier, "seed": configuration["seed"], "checkpoint": label,
               "step": step, "weights_sha256": recorded[f"{label}.weights.h5"],
               "evaluation_sha256": recorded[f"evaluation-{label}.npz"]},
              f"{label} checkpoint identity")
        _check_evaluation(attempt / f"evaluation-{label}.npz", result, data,
                          configuration["observation_variance"], spec)
    for label in ("final", "selected"):
        _check_weights(attempt / f"{label}.weights.h5", configuration)
    if not earlier and recorded["selected.weights.h5"] != recorded["final.weights.h5"]:
        # HDF5 container metadata may differ; compare tensors, not file bytes.
        import h5py
        with h5py.File(attempt / "final.weights.h5", "r") as final, h5py.File(
                attempt / "selected.weights.h5", "r") as selected:
            def compare(name, item):
                if isinstance(item, h5py.Dataset) and (name not in selected
                        or not np.array_equal(item[()], selected[name][()])):
                    raise ValueError("Selected final-step weights differ from final weights")
            final.visititems(compare)
    completion = {"schema_version": manifest["schema_version"], "identifier": identifier, "run_id": manifest["run_id"],
                  "attempt": attempt.name, "stage_manifest_sha256": expected["stage_manifest_sha256"],
                  "artifact_sha256": recorded}
    if "complete.json" in entries:
        sha256(attempt / "complete.json")
        _same(read_json(attempt / "complete.json"), completion, "attempt completion marker")
    return completion


def _publish_completion(root, identifier, attempt):
    """Called under the job writer lock; validation precedes both markers."""
    completion = check_attempt(root, identifier, attempt)
    atomic_json(attempt / "complete.json", completion)
    atomic_json(attempt.parent / "complete.json", completion)


def verified_completion(root, identifier, *, historical_identity=None):
    """Return only a fully validated attempt, never trust marker existence."""
    root = Path(root).resolve()
    manifest = verify_stage(root, historical_identity=historical_identity)
    _matrix_job(manifest, identifier)
    job_root = root / "jobs" / identifier
    if job_root.is_symlink() or (root / "jobs").is_symlink():
        raise ValueError("Invalid job directory")
    path = job_root / "complete.json"
    if not path.exists() and not path.is_symlink():
        return None
    sha256(path)
    completion = read_json(path)
    if completion.get("stage_manifest_sha256") != sha256(root / "stage_manifest.json"):
        raise ValueError("Job belongs to a different stage manifest")
    name = completion.get("attempt", "")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid job attempt path")
    attempt = job_root / name
    if (attempt.is_symlink() or not attempt.is_dir() or not name.startswith("attempt-")
            or len(name) != 40 or any(c not in "0123456789abcdef" for c in name[8:])):
        raise ValueError("Invalid job attempt path")
    _same(sorted(completion), sorted(("schema_version", "identifier", "run_id", "attempt",
                                     "stage_manifest_sha256", "artifact_sha256")), "completion fields")
    _same([completion["schema_version"], completion["identifier"], completion["run_id"]],
          [manifest["schema_version"], identifier, manifest["run_id"]], "completion identity")
    recorded = completion["artifact_sha256"]
    if (not isinstance(recorded, dict) or not set(REQUIRED_JOB_ARTIFACTS) <= set(recorded)
            or set(recorded) - set(REQUIRED_JOB_ARTIFACTS + OPTIONAL_JOB_ARTIFACTS)):
        raise ValueError("Incorrect completion artifact inventory")
    for artifact, expected in recorded.items():
        if sha256(attempt / artifact) != expected:
            raise ValueError(f"Completed job artifact changed: {artifact}")
    _same(completion, check_attempt(root, identifier, attempt,
                                    historical_identity=historical_identity), "completion marker")
    return attempt


def job_id(dataset, condition, seed, variance_label):
    if ((dataset not in DATASETS and re.fullmatch(r"dart_[0-9]{5}_2018", dataset) is None)
            or condition not in ("base", "raw", "canonical", "raw_half", "head_rate")):
        raise ValueError("Job outside the v2 matrix")
    return f"{dataset}_{condition}_{seed}_{variance_label}"


# ---- GPU runtime -----------------------------------------------------------


def gpu_runtime(expected_uuid):
    import tensorflow as tf
    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise RuntimeError(f"Worker requires exactly one GPU, found {len(physical)}")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
        probe.numpy()
    if "GPU:0" not in probe.device:
        raise RuntimeError("GPU tensor-placement probe failed")
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
         "--format=csv,noheader"], check=True, text=True, capture_output=True, timeout=20,
    ).stdout
    matches = [line for line in output.splitlines()
               if line.split(",", 1)[0].strip() == str(os.getpid())]
    if not any(expected_uuid in line for line in matches):
        raise RuntimeError("Current process is not attested on the requested GPU UUID")
    return {"gpu_uuid": expected_uuid, "pid": os.getpid(), "placement": probe.device,
            "process_inventory": matches, "tensorflow": tf.__version__,
            "python": sys.version, "numpy": np.__version__,
            "deterministic_ops_requested": True}


# ---- training --------------------------------------------------------------


def training_indices(rng, n, batch_size, weights=None):
    """Draw a training batch. Weighted draws follow the declared window measure."""
    if n <= 0:
        raise ValueError("empty training set")
    if weights is None:
        return rng.integers(0, n, size=batch_size, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError("train_weights length must match the training set")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("train_weights must be finite, nonnegative, and have mass")
    return rng.choice(n, size=batch_size, replace=True, p=weights / weights.sum()).astype(np.int32)


def _probe_state(model, windows, latent):
    """Decoder means on fixed latent vectors plus encoder proposals."""
    mean, log_variance = model.encode(windows)
    decoded = model.decode_mean(latent)
    return (np.asarray(decoded, dtype=np.float64),
            np.asarray(mean, dtype=np.float64),
            np.asarray(log_variance, dtype=np.float64))


def _decoder_hidden_numpy(model, latent):
    """Final decoder hidden activations (pre-head) as float64 numpy."""
    import tensorflow as tf
    hidden = tf.reshape(latent, (-1, model.latent_dim))
    for layer in model.decoder_hidden:
        hidden = layer(hidden)
    return np.asarray(hidden, dtype=np.float64)


def _symmetric_kl(mu_a, lv_a, mu_b, lv_b):
    """Symmetric KL between diagonal Gaussians, per row."""
    def kl(mu1, lv1, mu2, lv2):
        return 0.5 * np.sum(lv2 - lv1 + (np.exp(lv1) + (mu1 - mu2) ** 2) / np.exp(lv2) - 1.0,
                            axis=-1)
    return kl(mu_a, lv_a, mu_b, lv_b) + kl(mu_b, lv_b, mu_a, lv_a)


def _prior_draws(model, spec, stream_index, count):
    """Fixed prior latent and noise draws for one evaluation stream."""
    rng = np.random.default_rng(spec["prior_seed"] + stream_index)
    latent = rng.standard_normal((count, model.latent_dim)).astype(np.float32)
    noise = rng.standard_normal((count, model.input_dim))
    return latent, noise


def _validation_energy(model, data, spec):
    """Phase-weighted validation energy with the fixed validation stream."""
    from prigm_metrics import energy_score
    latent, noise = _prior_draws(model, spec, stream_index=-1,
                                 count=spec["validation_prior_samples"])
    mean = np.asarray(model.decode_mean(latent), dtype=np.float64)
    draws = mean + np.sqrt(model.residual_variance / model.b ** 2) * noise
    x = data["validation"].astype(np.float64)
    if "validation_weights" in data:
        weights = data["validation_weights"].astype(np.float64)
        scores = energy_score(draws, x)
        return float(weights @ scores / weights.sum())
    return float(energy_score(draws, x).mean())


def train(stage_root, dataset, condition, seed, observation_variance,
          variance_label, gpu_uuid):
    """Run one fixed-update fit with probes, selection, and evaluation."""
    root = Path(stage_root).resolve()
    manifest = verify_stage(root)
    spec = manifest["specification"]
    identifier = job_id(dataset, condition, seed, variance_label)
    _same([dataset, condition, seed, variance_label],
          list(_matrix_job(manifest, identifier)), "CLI stage job matrix")
    expected_variance = manifest["job_parameters"][identifier]["observation_variance"]
    if (isinstance(observation_variance, bool) or not np.isfinite(observation_variance)
            or observation_variance != expected_variance):
        raise ValueError("CLI observation variance differs from the frozen matrix")
    job_root = root / "jobs" / identifier
    if ((root / "jobs").is_symlink() or job_root.is_symlink()
            or (job_root / ".writer.lock").is_symlink()):
        raise ValueError("Invalid job lock directory")
    job_root.mkdir(exist_ok=True)
    lock = (job_root / ".writer.lock").open("a+")
    directory = job_root / ("attempt-" + uuid.uuid4().hex)
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        if verified_completion(root, identifier) is not None:
            print(f"VERIFIED REUSE {identifier}", flush=True)
            return
        directory.mkdir()
        runtime = gpu_runtime(gpu_uuid)
        snapshot_src = str(root / "source" / "src")
        if snapshot_src not in sys.path:
            sys.path.insert(0, snapshot_src)
        import tensorflow as tf
        from prigm_v2_model import BypassVAE, CanonicalBypassOptimizer
        tf.keras.utils.set_random_seed(seed)
        with np.load(root / "prepared" / f"{dataset}.npz", allow_pickle=False) as store:
            data = {key: store[key] for key in store.files}
        train_x = tf.constant(data["train"].astype(np.float32))

        model = BypassVAE(input_dim=spec["window"], latent_dim=spec["latent_dim"],
                          condition=condition, observation_variance=observation_variance,
                          seed=seed)
        optimizer = CanonicalBypassOptimizer(
            model, learning_rate=spec["learning_rate"],
            global_norm=spec["gradient_global_norm"],
            beta_1=spec["adam"]["beta_1"], beta_2=spec["adam"]["beta_2"],
            epsilon=spec["adam"]["epsilon"],
            clip=spec["gradient_clipping_enabled"])

        # Fixed probe/prior streams, independent of training randomness.
        probe_rng = np.random.default_rng(spec["probe_seed"])
        probe_windows = tf.constant(
            data["train"][probe_rng.integers(0, len(data["train"]),
                                             spec["probe_windows"], dtype=np.int64)]
            .astype(np.float32))
        probe_latent = tf.constant(probe_rng.standard_normal(
            (spec["probe_latents"], spec["latent_dim"])).astype(np.float32))

        rng = np.random.default_rng(seed + spec["training_seed_offset"])
        train_weights = data["train_weights"].astype(np.float64) if "train_weights" in data else None

        @tf.function
        def update(indices, epsilon):
            batch = tf.gather(train_x, indices)
            with tf.GradientTape() as tape:
                terms = model.loss_terms(batch, epsilon)
                loss = tf.reduce_mean(terms["negative_elbo"])
            gradients = tape.gradient(loss, model.trainable_variables)
            norm = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, tf.reduce_mean(terms["kl"]), norm

        configuration = _configuration_identity(root, manifest, identifier, directory.name)
        # Record actual model/optimizer storage, including float32 LR variables.
        actual = dict(configuration, alpha=model.alpha, b=model.b,
                      observation_variance=model.observation_variance,
                      residual_variance=model.residual_variance,
                      model_storage_contract=model.storage_contract,
                      head_coordinates=model.head_coordinates, head_scale=model.head_scale,
                      model_dtype=model.compute_dtype,
                      body_learning_rate=float(optimizer._optimizers["body"].learning_rate.numpy()),
                      head_learning_rate=float(optimizer._optimizers["head"].learning_rate.numpy()),
                      epsilon=float(optimizer.epsilon),
                      epsilon_float32=float(np.float32(optimizer._optimizers["head"].epsilon)),
                      gradient_clipping_enabled=optimizer.clip,
                      gradient_global_norm=optimizer.global_norm)
        _same(actual, configuration, "actual model/optimizer configuration")
        atomic_json(directory / "configuration.json", dict(actual, runtime=runtime))

        probe_steps = set(spec["probe_steps"])
        best_score = np.inf
        best_step = 0
        started = time.monotonic()
        for step in range(1, spec["max_steps"] + 1):
            probe_now = step in probe_steps
            pre_decode = pre_mean = pre_lv = pre_hidden = None
            pre_kernel = pre_bias = None
            if probe_now:
                pre_decode, pre_mean, pre_lv = _probe_state(
                    model, probe_windows, probe_latent)
                pre_hidden = _decoder_hidden_numpy(model, probe_latent)
                pre_kernel = np.asarray(model.output_head.kernel, dtype=np.float64)
                pre_bias = np.asarray(model.output_head.bias, dtype=np.float64)
            indices = training_indices(rng, len(data["train"]), spec["batch_size"],
                                       train_weights)
            epsilon = rng.standard_normal((spec["batch_size"],
                                           spec["latent_dim"])).astype(np.float32)
            loss, kl, norm = update(indices, epsilon)
            record = {"step": step, "pre_update_loss": float(loss.numpy()),
                      "pre_update_minibatch_kl": float(kl.numpy()),
                      "gradient_norm": float(norm.numpy()),
                      "clipping_enabled": optimizer.clip,
                      "clipping_scale": float(optimizer.clipping_scale.numpy()),
                      "clipping_active": bool(optimizer.clipping_active.numpy())}
            append_jsonl(directory / "history.jsonl", record)
            if probe_now:
                post_decode, post_mean, post_lv = _probe_state(
                    model, probe_windows, probe_latent)
                full_rms = float(np.sqrt(np.mean((post_decode - pre_decode) ** 2)))
                # Hold pre-update hidden activations fixed to isolate the head.
                post_kernel = np.asarray(model.output_head.kernel, dtype=np.float64)
                post_bias = np.asarray(model.output_head.bias, dtype=np.float64)
                head_pre = (pre_hidden @ pre_kernel + pre_bias) / model.head_scale
                head_post = (pre_hidden @ post_kernel + post_bias) / model.head_scale
                head_only_rms = float(np.sqrt(np.mean((head_post - head_pre) ** 2)))
                encoder_kl = float(_symmetric_kl(pre_mean, pre_lv,
                                                 post_mean, post_lv).mean())
                append_jsonl(directory / "probes.jsonl", {
                    "step": step, "full_decoder_rms": full_rms,
                    "head_only_rms": head_only_rms,
                    "encoder_symmetric_kl": encoder_kl,
                })
            if step % spec["validation_every"] == 0:
                score = _validation_energy(model, data, spec)
                improved = score < best_score
                if improved:
                    best_score, best_step = score, step
                    temporary = directory / "selected.pending.weights.h5"
                    model.save_weights(str(temporary))
                    os.replace(temporary, directory / "selected.weights.h5")
                append_jsonl(directory / "history.jsonl",
                             {"step": step, "validation_energy": score,
                              "best_step": best_step})
                atomic_json(directory / "progress.json", {
                    "step": step, "best_step": best_step, "best_score": best_score,
                    "elapsed_seconds": time.monotonic() - started,
                })
        model.save_weights(str(directory / "final.weights.h5"))
        final_summary = _evaluate(directory, model, data, spec, identifier, seed,
                                  checkpoint_label="final")
        # Selected-checkpoint evaluation: reload and rerun the same contract.
        selected_summary = None
        if best_step and best_step < spec["max_steps"]:
            model.load_weights(str(directory / "selected.weights.h5"))
            selected_summary = _evaluate(directory, model, data, spec, identifier,
                                         seed, checkpoint_label="selected")
            model.load_weights(str(directory / "final.weights.h5"))
        for label, result, step in (("final", final_summary, spec["max_steps"]),
                                    ("selected", selected_summary, best_step)):
            if result is not None:
                result.update(step=step, weights_sha256=sha256(directory / f"{label}.weights.h5"),
                              evaluation_sha256=sha256(directory / f"evaluation-{label}.npz"))
        summary = {"identifier": identifier, "attempt": directory.name, "run_id": manifest["run_id"],
                   "final": final_summary, "best_score": best_score,
                   "selected_step": best_step, "selected": selected_summary}
        atomic_json(directory / "summary.json", summary)
        # Publish the reusable pointer last, only after attempt validation.
        _publish_completion(root, identifier, directory)
        print(f"DONE {identifier}", flush=True)
    except BaseException as error:
        if directory.exists():
            atomic_json(directory / "failure.json",
                        {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock.close()


def _evaluate(directory, model, data, spec, identifier, seed,
              checkpoint_label="final"):
    """Final-update evaluation: energies, reassignment, IW, linear controls."""
    from prigm_metrics import energy_score, gaussian_log_prob, importance_statistics
    from prigm_v2_metrics import (increment_energy_score,
                                  linear_model_iw_control,
                                  noise_matched_diagonal,
                                  posterior_code_reassignment)
    x_test = data["test"].astype(np.float64)
    weights = data["eval_weights"].astype(np.float64) if "eval_weights" in data \
        else np.ones(len(x_test))
    if (weights.shape != (len(x_test),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid evaluation weights")
    weights = weights / weights.sum()

    energy_windows, increment_windows, prior_means, prior_samples = [], [], [], []
    for stream in range(spec["evaluation_streams"]):
        latent, noise = _prior_draws(model, spec, stream,
                                     spec["evaluation_prior_samples"])
        mean = np.asarray(model.decode_mean(latent), dtype=np.float64)
        draws = mean + np.sqrt(model.observation_variance) * noise
        prior_means.append(mean)
        prior_samples.append(draws)
        energy_windows.append(energy_score(draws, x_test))
        increment_windows.append(increment_energy_score(draws, x_test))
    energy_windows = np.asarray(energy_windows)
    increment_windows = np.asarray(increment_windows)
    energies = energy_windows @ weights
    increment_energies = increment_windows @ weights

    # Posterior-code reassignment: donors uniformly from other windows for
    # synthetic data; DART donors are computed in the prepared arrays.
    reassignment_available = bool(data.get("reassignment_available", True))
    if "reassignment_donors" in data:
        donors = data["reassignment_donors"]
    else:
        donor_rng = np.random.default_rng(spec["reassignment_seed"])
        donors = np.array([(i + 1 + donor_rng.integers(0, len(x_test) - 1)) % len(x_test)
                           for i in range(len(x_test))], dtype=np.int64)
    if reassignment_available:
        reassignment = posterior_code_reassignment(
            model, x_test, donors, seed=spec["reassignment_seed"])
    else:
        if donors.shape != (0,) or donors.dtype.kind not in "iu":
            raise ValueError("Unavailable reassignment must have empty donor support")
        reassignment = {key: np.empty(0, dtype=np.float64)
                        for key in ("own_mse", "reassignment_gain")}

    components = [_iw_evaluate(model, x_test, spec["iw_samples"],
                               spec["iw_seed"] + stream)
                  for stream in range(spec["iw_streams"])]
    components = {key: np.stack([stream[key] for stream in components])
                  for key in components[0]}
    iw_scores, iw_ess = components["log_score"], components["ess"]

    # Analytic linear controls on the same test windows.
    train_x = data["train"].astype(np.float64)
    train_w = data["train_weights"].astype(np.float64) if "train_weights" in data \
        else np.ones(len(train_x))
    from prigm_v2_metrics import weighted_moments
    mean_w, cov_w = weighted_moments(train_x, train_w)
    controls, control_arrays = {}, {"linear_model_mean": mean_w}
    control_rank = 16 if str(data.get("dataset_name", "")).startswith("dart") \
        else spec["synthetic_latent_dim"]
    for k in spec["linear_control_k"]:
        control = linear_model_iw_control(
            cov_w, model.observation_variance, x_test[:32] - mean_w, k,
            seed=spec["iw_seed"] + 100 + int(np.log2(k)), rank=control_rank)
        controls[f"k{k}"] = {
            "mean_error": float(np.mean(control["error"])),
            "max_abs_error": float(np.max(np.abs(control["error"]))),
            "median_ess": float(np.median(control["ess"])),
        }
        for key in ("exact_log_marginal", "log_weights", "iw_log_score", "ess", "error"):
            control_arrays[f"linear_k{k}_{key}"] = control[key]

    baseline = noise_matched_diagonal(train_x, train_w, model.observation_variance)
    baseline_score = float(weights @ gaussian_log_prob(
        x_test, baseline["mean"], baseline["covariance"]) / weights.sum())

    arrays = {
        "evaluation_weights": weights,
        "energy_streams": energies,
        "increment_energy_streams": increment_energies,
        "energy_per_window": energy_windows,
        "increment_energy_per_window": increment_windows,
        "prior_means": np.asarray(prior_means),
        "prior_samples": np.asarray(prior_samples),
        "reassignment_donors": donors,
        "reassignment_own_mse": reassignment["own_mse"],
        "reassignment_gain": reassignment["reassignment_gain"],
        "iw_log_scores": iw_scores,
        "iw_ess": iw_ess,
    }
    arrays.update({key: value for key, value in components.items()
                   if key not in ("log_score", "ess")})
    arrays.update(control_arrays)
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite evaluation artifact: {key}")
    np.savez_compressed(directory / f"evaluation-{checkpoint_label}.npz", **arrays)
    return {
        "identifier": identifier, "seed": seed, "checkpoint": checkpoint_label,
        "energy_streams": [float(e) for e in energies],
        "increment_energy_streams": [float(e) for e in increment_energies],
        "reassignment_available": reassignment_available,
        "reassignment_gain_weighted_mean": (float(weights @ reassignment["reassignment_gain"])
                                             if reassignment_available else None),
        "iw_mean_log_score": float(np.mean(iw_scores @ weights / weights.sum())),
        "iw_median_ess": float(np.median(iw_ess)),
        "posterior_kl_nats_per_window": float(np.mean(components["kl"] @ weights)),
        "posterior_expected_nll_nats_per_window": float(np.mean(
            components["posterior_expected_nll"] @ weights)),
        "negative_elbo_nats_per_window": float(np.mean(
            components["negative_elbo_analytic_kl"] @ weights)),
        "posterior_mean_mse": float(np.mean(components["posterior_mean_mse"] @ weights)),
        "posterior_expected_mse": float(np.mean(
            components["posterior_expected_mse"] @ weights)),
        "sample_jensen_gap_nats_per_window": float(np.mean(
            components["jensen_gap"] @ weights)),
        "linear_controls": controls,
        "noise_matched_diagonal_log_density": baseline_score,
    }


def _check_evaluation(path, summary, data, variance, spec=None):
    """Check saved components and independently reduce the declared window measure."""
    spec = SPEC if spec is None else spec
    from prigm_metrics import energy_score, gaussian_log_prob
    from prigm_v2_metrics import (increment_energy_score, linear_model_iw_control,
                                  noise_matched_diagonal, weighted_moments)
    n, d = data["test"].shape
    e, p = spec["evaluation_streams"], spec["evaluation_prior_samples"]
    i, k, latent = spec["iw_streams"], spec["iw_samples"], spec["latent_dim"]
    if n < 2 or d != spec["window"]:
        raise ValueError("Invalid evaluation data shape")
    reassignment_available = bool(data.get("reassignment_available", True))
    if summary.get("reassignment_available") is not reassignment_available:
        raise ValueError("Incorrect reassignment availability")
    donor_count = n if reassignment_available else 0
    shapes = {"evaluation_weights": (n,), "energy_streams": (e,),
              "increment_energy_streams": (e,), "energy_per_window": (e, n),
              "increment_energy_per_window": (e, n), "prior_means": (e, p, d),
              "prior_samples": (e, p, d), "reassignment_donors": (donor_count,),
              "reassignment_own_mse": (donor_count,), "reassignment_gain": (donor_count,),
              "linear_model_mean": (d,), "kl_by_latent": (i, n, latent)}
    for key in ("iw_log_scores", "iw_ess", "posterior_expected_nll", "posterior_expected_mse",
                "posterior_mean_mse", "kl", "negative_elbo_analytic_kl", "sample_negative_elbo", "jensen_gap"):
        shapes[key] = (i, n)
    for key in ("log_weights", "sample_log_likelihood", "sample_logq_minus_logprior"):
        shapes[key] = (i, n, k)
    for count in spec["linear_control_k"]:
        for key in ("exact_log_marginal", "log_weights", "iw_log_score", "ess", "error"):
            shapes[f"linear_k{count}_{key}"] = ((min(n, 32), count) if key == "log_weights"
                                                else (min(n, 32),))
    with np.load(path, allow_pickle=False) as store:
        if set(store.files) != set(shapes) or len(store.files) != len(shapes):
            raise ValueError("Incorrect evaluation array inventory")
        arrays = {key: store[key] for key in shapes}
    for key, value in arrays.items():
        if (value.shape != shapes[key] or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid evaluation array: {key}")

    def close(actual, expected, label):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if (actual.shape != expected.shape or actual.dtype.kind not in "fiu"
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=1e-9, atol=1e-8)):
            raise ValueError(f"Inconsistent saved reduction/component: {label}")

    def component(key, expected):
        close(arrays[key], expected, key)

    def reduced(key, expected):
        if key not in summary:
            raise ValueError(f"Missing evaluation summary: {key}")
        close(summary[key], expected, key)

    weights = np.asarray(data.get("eval_weights", np.ones(n)), dtype=np.float64)
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid prepared evaluation weights")
    weights = weights / weights.sum()
    component("evaluation_weights", weights)
    for stream in range(e):
        # Consume latent draws as well to reproduce the fixed prior noise stream.
        rng = np.random.default_rng(spec["prior_seed"] + stream)
        rng.standard_normal((p, latent))
        noise = rng.standard_normal((p, d))
        close(arrays["prior_samples"][stream],
              arrays["prior_means"][stream] + np.sqrt(variance) * noise, "prior observation noise")
        close(arrays["energy_per_window"][stream],
              energy_score(arrays["prior_samples"][stream], data["test"]), "per-window energy")
        close(arrays["increment_energy_per_window"][stream],
              increment_energy_score(arrays["prior_samples"][stream], data["test"]),
              "per-window increment energy")
    for prefix in ("energy", "increment_energy"):
        value = arrays[f"{prefix}_per_window"] @ weights
        component(f"{prefix}_streams", value)
        reduced(f"{prefix}_streams", value)
    donors = arrays["reassignment_donors"]
    if (donors.dtype.kind not in "iu" or np.any(donors < 0) or np.any(donors >= n)
            or (reassignment_available and np.any(donors == np.arange(n)))):
        raise ValueError("Invalid reassignment donors")
    expected_donors = data.get("reassignment_donors")
    if expected_donors is None:
        rng = np.random.default_rng(spec["reassignment_seed"])
        expected_donors = np.array([(index + 1 + rng.integers(0, n - 1)) % n for index in range(n)])
    component("reassignment_donors", expected_donors)
    if reassignment_available:
        with np.errstate(over="ignore", invalid="ignore"):
            donor_mse = arrays["reassignment_own_mse"] + arrays["reassignment_gain"]
        if not np.all(np.isfinite(donor_mse)) or np.any(donor_mse < -1e-8):
            raise ValueError("Invalid reassignment donor MSE")
        reduced("reassignment_gain_weighted_mean", weights @ arrays["reassignment_gain"])
    elif ("reassignment_gain_weighted_mean" not in summary
          or summary["reassignment_gain_weighted_mean"] is not None):
        raise ValueError("Unavailable reassignment cannot report a numerical gain")
    component("log_weights", arrays["sample_log_likelihood"] - arrays["sample_logq_minus_logprior"])
    stats = importance_statistics(arrays["log_weights"].reshape(i * n, k))
    component("iw_log_scores", stats["log_score"].reshape(i, n))
    component("iw_ess", stats["ess"].reshape(i, n))
    component("posterior_expected_nll", -arrays["sample_log_likelihood"].mean(axis=-1))
    component("posterior_expected_mse", (2 * arrays["posterior_expected_nll"] / d
                                         - np.log(2 * np.pi * variance)) * variance)
    component("kl", arrays["kl_by_latent"].sum(axis=-1))
    component("negative_elbo_analytic_kl", arrays["posterior_expected_nll"] + arrays["kl"])
    component("sample_negative_elbo", -arrays["log_weights"].mean(axis=-1))
    component("jensen_gap", arrays["iw_log_scores"] + arrays["sample_negative_elbo"])
    for key in ("posterior_expected_mse", "posterior_mean_mse", "reassignment_own_mse",
                "kl_by_latent", "jensen_gap"):
        if np.any(arrays[key] < -1e-8):
            raise ValueError(f"Negative evaluation component: {key}")
    reduced("iw_mean_log_score", np.mean(arrays["iw_log_scores"] @ weights / weights.sum()))
    reduced("iw_median_ess", np.median(arrays["iw_ess"]))
    for key, component_key in {
            "posterior_kl_nats_per_window": "kl",
            "posterior_expected_nll_nats_per_window": "posterior_expected_nll",
            "negative_elbo_nats_per_window": "negative_elbo_analytic_kl",
            "posterior_mean_mse": "posterior_mean_mse",
            "posterior_expected_mse": "posterior_expected_mse",
            "sample_jensen_gap_nats_per_window": "jensen_gap"}.items():
        reduced(key, np.mean(arrays[component_key] @ weights))
    train_x = data["train"].astype(np.float64)
    train_w = np.asarray(data.get("train_weights", np.ones(len(train_x))), dtype=np.float64)
    mean, covariance = weighted_moments(train_x, train_w)
    component("linear_model_mean", mean)
    rank = 16 if str(data.get("dataset_name", "")).startswith("dart") else spec["synthetic_latent_dim"]
    controls = {}
    for count in spec["linear_control_k"]:
        control = linear_model_iw_control(
            covariance, variance, data["test"][:32].astype(np.float64) - mean, count,
            seed=spec["iw_seed"] + 100 + int(np.log2(count)), rank=rank)
        for key in ("exact_log_marginal", "log_weights", "iw_log_score", "ess", "error"):
            component(f"linear_k{count}_{key}", control[key])
        controls[f"k{count}"] = {"mean_error": float(np.mean(control["error"])),
                                 "max_abs_error": float(np.max(np.abs(control["error"]))),
                                 "median_ess": float(np.median(control["ess"]))}
    if set(summary.get("linear_controls", {})) != set(controls):
        raise ValueError("Incorrect linear control summary inventory")
    for label, values in controls.items():
        if set(summary["linear_controls"][label]) != set(values):
            raise ValueError("Incorrect linear control summary fields")
        for key, value in values.items():
            close(summary["linear_controls"][label][key], value, f"{label}/{key}")
    baseline = noise_matched_diagonal(train_x, train_w, variance)
    reduced("noise_matched_diagonal_log_density", float(weights @ gaussian_log_prob(
        data["test"].astype(np.float64), baseline["mean"], baseline["covariance"]) / weights.sum()))


def _iw_evaluate(model, observations, samples, seed, chunk=32):
    """Same-draw IW, ELBO, and posterior reconstruction components."""
    from prigm_v2_metrics import posterior_score_components
    return posterior_score_components(model, observations, samples, seed, chunk=chunk)


# ---- dispatcher ------------------------------------------------------------


def campaign(stage_root, gpu_uuids):
    """Exclusive, restart-only stage execution on four owned workers.

    Signal handlers only set a Python flag. Cleanup runs outside signal context
    and BEFORE executor shutdown, so blocked worker waits cannot delay cancellation.
    """
    root = Path(stage_root).resolve()
    manifest = verify_stage(root)
    if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4:
        raise ValueError("Campaign requires four distinct GPU UUIDs")
    lock_path = root / ".campaign.lock"
    if lock_path.is_symlink():
        raise ValueError("Invalid campaign lock")
    lock = lock_path.open("a+")
    handlers, executor = {}, None
    jobs = [tuple(job) for job in manifest["jobs"]]
    pending = deque(jobs)
    guard = threading.Lock()
    errors, completed = [], set()
    stop = threading.Event()
    children = {}
    interrupted = False

    def cancel(signum, frame):
        # Never acquire guard or Event's internal lock in a signal handler.
        nonlocal interrupted
        interrupted = True

    def reap_owned():
        stop.set()
        with guard:
            active = list(children.values())
        for child in active:
            if child.poll() is None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 2.0
        for child in active:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for child in active:
            if child.poll() is None:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
        for child in active:
            child.wait()

    def worker(gpu):
        while True:
            with guard:
                if stop.is_set() or not pending:
                    return
                job = pending.popleft()
            dataset, condition, seed, variance_label = job
            identifier = job_id(*job)
            log = root / "logs" / f"{identifier}-{uuid.uuid4().hex}.log"
            try:
                if verified_completion(root, identifier) is None:
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                                       TF_CPP_MIN_LOG_LEVEL="2", TF_DETERMINISTIC_OPS="1",
                                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                                       PYTHONDONTWRITEBYTECODE="1")
                    runner = root / "source" / "src" / "prigm_v2_experiment.py"
                    command = [sys.executable, str(runner), "train",
                               "--stage-root", str(root), "--dataset", dataset,
                               "--condition", condition, "--seed", str(seed),
                               "--variance-label", variance_label,
                               "--observation-variance", str(manifest["job_parameters"][identifier]["observation_variance"]),
                               "--gpu-uuid", gpu]
                    print(f"START {identifier} {gpu}", flush=True)
                    with log.open("x") as stream:
                        with guard:
                            if stop.is_set():
                                return
                            child = subprocess.Popen(command, env=environment,
                                                     stdout=stream, stderr=subprocess.STDOUT)
                            children[identifier] = child
                        returncode = child.wait()
                    if returncode:
                        raise RuntimeError(f"Worker exited {returncode}; log={log.name}")
                    if verified_completion(root, identifier) is None:
                        raise ValueError("Worker exited without a verified completion")
                with guard:
                    children.pop(identifier, None)
                    completed.add(identifier)
                    atomic_json(root / "campaign_status.json", {
                        "pending": len(pending), "errors": list(errors), "completed": len(completed),
                    })
                print(f"VERIFIED {identifier}", flush=True)
            except BaseException as error:
                stop.set()
                with guard:
                    errors.append({"job": identifier, "type": type(error).__name__,
                                   "message": str(error), "log": log.name})
                return

    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, cancel)
        executor = ThreadPoolExecutor(max_workers=4)
        futures = {executor.submit(worker, gpu) for gpu in gpu_uuids}
        while futures and not stop.is_set() and not interrupted:
            _, futures = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
        if interrupted or stop.is_set():
            raise RuntimeError(f"Campaign stopped before completion: {errors}")
    finally:
        try:
            if executor is not None:
                reap_owned()
                executor.shutdown(wait=True, cancel_futures=True)
                atomic_json(root / "campaign_status.json", {
                    "pending": len(pending), "errors": list(errors), "completed": len(completed),
                    "interrupted": interrupted,
                })
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-synthetic")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--family", choices=("gaussian_v2", "mixture_v2"), required=True)
    build = commands.add_parser("build-stage")
    build.add_argument("--run-root", required=True)
    build.add_argument("--stage", required=True)
    build.add_argument("--dataset-dir", default=None)
    dart = commands.add_parser("build-dart")
    dart.add_argument("--run-root", required=True)
    dart.add_argument("--dataset-path", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--stage-root", required=True)
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--condition", required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--observation-variance", type=float, default=None)
    fit.add_argument("--variance-label", default="v1")
    fit.add_argument("--gpu-uuid", required=True)
    execute = commands.add_parser("campaign")
    execute.add_argument("--stage-root", required=True)
    execute.add_argument("--gpu-uuids", nargs=4, required=True)
    args = parser.parse_args()
    if args.command == "prepare-synthetic":
        from prigm_v2_data import prepare_gaussian, prepare_mixture
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prepared = (prepare_gaussian() if args.family == "gaussian_v2"
                    else prepare_mixture())
        arrays = {split: prepared[split] for split in ("train", "validation", "test")}
        arrays["oracle_covariance"] = prepared["oracle_covariance"]
        np.savez_compressed(output / f"{args.family}.npz", **arrays)
        atomic_json(output / f"{args.family}-metadata.json", prepared["metadata"])
        print(json.dumps({"output": str(output / f'{args.family}.npz'),
                          "sha256": sha256(output / f"{args.family}.npz"),
                          "metadata": prepared["metadata"]}, indent=2))
    elif args.command == "build-stage":
        build_stage(args.run_root, args.stage, args.dataset_dir)
    elif args.command == "build-dart":
        build_dart_stage(args.run_root, args.dataset_path)
    elif args.command == "train":
        variance = args.observation_variance
        if variance is None:
            manifest = verify_stage(args.stage_root)
            identifier = job_id(args.dataset, args.condition, args.seed, args.variance_label)
            _matrix_job(manifest, identifier)
            variance = manifest["job_parameters"][identifier]["observation_variance"]
        train(args.stage_root, args.dataset, args.condition, args.seed,
              variance, args.variance_label, args.gpu_uuid)
    else:
        campaign(args.stage_root, args.gpu_uuids)


if __name__ == "__main__":
    main()
