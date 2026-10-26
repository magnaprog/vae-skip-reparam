"""Evaluate probability components of verified frozen v1 checkpoints.

Each invocation is a separate process and writes a fresh private worker bundle.
The model and original metrics are imported from the hash-verified core source.
No training, checkpoint selection, or historical artifact is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np


CORE_FILES = tuple(f"src/prigm_{name}.py" for name in ("data", "metrics", "model", "experiment"))
DATASETS = ("gaussian", "23461t2022", "44402t2016", "46407t2011", "52403t2020")
ARMS = ("A", "B", "C", "D")
SEEDS = (42, 1729, 31415)


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def _write_json(path, value):
    with Path(path).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verified_core_source(root, expected_sha256):
    """Verify the source closure before importing executable archived code."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected lowercase core SHA256")
    root = Path(root).absolute()
    for directory in (root, root / "source", root / "source/src", root / "prepared"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Core directories must be regular directories")
    if _sha(root / "run_manifest.json") != expected_sha256:
        raise ValueError("Core manifest SHA256 differs")
    manifest = _read_json(root / "run_manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("purpose") != "core"
            or manifest.get("datasets") != list(DATASETS)
            or set(manifest.get("code_sha256", {})) != set(CORE_FILES)
            or set(manifest.get("prepared_sha256", {}))
            != {"metadata.json", *(name + ".npz" for name in DATASETS)}):
        raise ValueError("Unsupported frozen core identity")
    if {entry.name for entry in (root / "source/src").iterdir()} != {Path(name).name for name in CORE_FILES}:
        raise ValueError("Frozen source inventory differs")
    for name, expected in manifest["code_sha256"].items():
        if _sha(root / "source" / name) != expected:
            raise ValueError(f"Frozen source differs: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        if _sha(root / "prepared" / name) != expected:
            raise ValueError(f"Prepared input differs: {name}")
    return manifest


DART_DIAGNOSTICS = {
    "prior_samples": 1024,
    "prior_seeds": [71001, 71002, 71003],
    "variance_samples": 32,
    "variance_seeds": [72001, 72002, 72003],
    "reference_seeds": [74001, 74002, 74003],
    "iw_samples": 256,
    "iw_seeds": [42001, 42003, 42004],
    "density_coordinates": "observation",
}


def dart_checkpoint_diagnostics(model, data):
    """Score fixed learned functions before and after training-only variance refit."""
    from prigm_v2_metrics import (posterior_score_components,
                                  refit_observation_variance,
                                  unconditional_score_streams)
    train_weights = data.get("train_weights", np.ones(len(data["train"])))
    test_weights = np.asarray(data.get("eval_weights", np.ones(len(data["test"]))),
                              dtype=np.float64)
    test_weights = test_weights / test_weights.sum()
    refit = refit_observation_variance(
        model, data["train"], train_weights,
        samples=DART_DIAGNOSTICS["variance_samples"],
        seeds=tuple(DART_DIAGNOSTICS["variance_seeds"]))
    arrays = {
        "diagnostic_test_weights": test_weights,
        "variance_refit_training_weights": refit["normalized_training_weights"],
        "variance_refit_training_mse": refit["posterior_expected_mse_per_window"],
        "variance_refit_stream_variances": refit["stream_variances"],
        "variance_refit_observation_variance": np.asarray(refit["observation_variance"]),
    }
    summaries = {}
    for label, variance in (("original", float(model.observation_variance)),
                            ("variance_refit", float(refit["observation_variance"]))):
        scores = unconditional_score_streams(
            model, data["test"], test_weights, variance,
            samples=DART_DIAGNOSTICS["prior_samples"],
            seeds=tuple(DART_DIAGNOSTICS["prior_seeds"]))
        arrays.update({label + "__" + key: np.asarray(value) for key, value in scores.items()})
        components = [posterior_score_components(
            model, data["test"], DART_DIAGNOSTICS["iw_samples"], seed,
            observation_variance=variance) for seed in DART_DIAGNOSTICS["iw_seeds"]]
        for key in components[0]:
            arrays[label + "__" + key] = np.stack([value[key] for value in components])
        summaries[label] = {
            "observation_variance": variance,
            "joint_energy": float(np.mean(scores["energy_streams"])),
            "centered_energy": float(np.mean(scores["centered_energy_streams"])),
            "increment_energy": float(np.mean(scores["increment_energy_streams"])),
            "mean_iw_log_density_per_coordinate": float(np.mean(
                arrays[label + "__log_score"] @ test_weights) / model.input_dim),
            "median_ess": float(np.median(arrays[label + "__ess"])),
        }
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite DART diagnostic: {key}")
    return arrays, summaries


def dart_reference_diagnostics(data, observation_variance):
    """Replicated unconditional references on the same retained test windows."""
    from prigm_v2_metrics import (centered_energy_score, energy_score,
                                  fixed_noise_ppca, gaussian_log_prob,
                                  increment_energy_score, noise_matched_diagonal,
                                  weighted_moments)
    train_weights = np.asarray(data.get("train_weights", np.ones(len(data["train"]))),
                               dtype=np.float64)
    test_weights = np.asarray(data.get("eval_weights", np.ones(len(data["test"]))),
                              dtype=np.float64)
    test_weights = test_weights / test_weights.sum()
    mean, covariance = weighted_moments(data["train"], train_weights)
    eigenvalues = np.linalg.eigvalsh(covariance)
    fitted_noise = max(float(eigenvalues[:-16].mean()), 1e-4)
    references = {
        "ppca_matched": fixed_noise_ppca(mean, covariance, observation_variance, rank=16),
        "ppca_fitted": fixed_noise_ppca(mean, covariance, fitted_noise, rank=16),
        "diagonal_matched": noise_matched_diagonal(
            data["train"], train_weights, observation_variance),
    }
    arrays = {"test_weights": test_weights}
    summaries = {}
    for name, reference in references.items():
        root = np.linalg.cholesky(reference["covariance"])
        draws = np.stack([
            np.random.default_rng(seed).standard_normal(
                (DART_DIAGNOSTICS["prior_samples"], len(mean))) @ root.T + reference["mean"]
            for seed in DART_DIAGNOSTICS["reference_seeds"]])
        arrays[name + "__mean"] = reference["mean"]
        arrays[name + "__covariance"] = reference["covariance"]
        arrays[name + "__prior_samples"] = draws
        arrays[name + "__log_density"] = gaussian_log_prob(
            data["test"], reference["mean"], reference["covariance"])
        summaries[name] = {}
        for score_name, score_function in (("joint", energy_score),
                                            ("centered", centered_energy_score),
                                            ("increment", increment_energy_score)):
            values = np.stack([score_function(stream, data["test"]) for stream in draws])
            arrays[name + "__" + score_name + "_per_window"] = values
            arrays[name + "__" + score_name + "_streams"] = values @ test_weights
            arrays[name + "__" + score_name + "_uniform_streams"] = values.mean(axis=1)
            summaries[name][score_name + "_energy"] = float(np.mean(values @ test_weights))
    return arrays, summaries


STAGE_FILES = tuple(f"src/prigm_v2_{name}.py" for name in
                    ("data", "model", "metrics", "experiment")) + ("src/prigm_metrics.py",)


def verified_stage_source(root, expected_sha256):
    """Check the schema-4 executable inventory without importing any prigm module."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected lowercase stage SHA256")
    root = Path(root).absolute()
    for directory in (*root.parents, root, root / "source", root / "source/src", root / "prepared"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Stage directories must be regular directories")
    if _sha(root / "stage_manifest.json") != expected_sha256:
        raise ValueError("Stage manifest SHA256 differs")
    manifest = _read_json(root / "stage_manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 4
            or manifest.get("stage") != "dart_record_extension"
            or manifest.get("purpose") != "dart_input_skip_evaluation"
            or not isinstance(manifest.get("code_sha256"), dict)
            or set(manifest["code_sha256"]) != set(STAGE_FILES)
            or not isinstance(manifest.get("prepared_sha256"), dict)
            or not manifest["prepared_sha256"]
            or any(re.fullmatch(r"dart_[0-9]{5}_2018", name) is None
                   for name in manifest["prepared_sha256"])):
        raise ValueError("Unsupported frozen DART stage identity")
    if ({entry.name for entry in (root / "source").iterdir()} != {"src"}
            or {entry.name for entry in (root / "source/src").iterdir()}
            != {Path(name).name for name in STAGE_FILES}):
        raise ValueError("Frozen stage source inventory differs")
    for name, expected in manifest["code_sha256"].items():
        if _sha(root / "source" / name) != expected:
            raise ValueError(f"Frozen stage source differs: {name}")
    prepared = {name + ".npz": value for name, value in manifest["prepared_sha256"].items()}
    prepared["records.json"] = manifest.get("records_sha256")
    if {entry.name for entry in (root / "prepared").iterdir()} != set(prepared):
        raise ValueError("Stage prepared inventory differs")
    for name, expected in prepared.items():
        if _sha(root / "prepared" / name) != expected:
            raise ValueError(f"Stage prepared input differs: {name}")
    return manifest


def _stage_jobs(manifest, datasets, checkpoints, worker_index):
    """Partition only the archived runner's verified admitted matrix."""
    admitted = set(manifest["prepared_sha256"])
    datasets = sorted(admitted) if datasets is None else datasets
    if (not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= admitted
            or not checkpoints or len(set(checkpoints)) != len(checkpoints)
            or not set(checkpoints) <= {"final", "selected"}
            or type(worker_index) is not int or worker_index not in range(4)):
        raise ValueError("Invalid stage dataset/checkpoint/worker selection")
    matrix = manifest["jobs"]
    if (not matrix or any(not isinstance(job, (list, tuple)) or len(job) != 4
                         or job[0] not in admitted or job[1] not in ("base", "raw")
                         or type(job[2]) is not int or job[2] not in SEEDS
                         or job[3] not in ("low", "high") for job in matrix)
            or len({tuple(job) for job in matrix}) != len(matrix)):
        raise ValueError("Invalid stage job matrix")
    all_jobs = [(*job, label) for job in matrix if job[0] in datasets for label in checkpoints]
    jobs = all_jobs[worker_index::4]
    if not jobs:
        raise ValueError("Worker has no declared jobs")
    return all_jobs, jobs


def _stage_replay(model, data, evaluation_path, spec):
    """Compare archived-seed components, not the different diagnostic draw streams."""
    from prigm_v2_experiment import _iw_evaluate
    streams = [_iw_evaluate(model, data["test"].astype(np.float64), spec["iw_samples"],
                            spec["iw_seed"] + index) for index in range(spec["iw_streams"])]
    arrays = {key: np.stack([stream[key] for stream in streams]) for key in streams[0]}
    differences = {}
    exact = True
    with np.load(evaluation_path, allow_pickle=False) as saved:
        for key, value in arrays.items():
            saved_key = {"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key)
            original = saved[saved_key]
            if value.shape != original.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid stage replay component: {key}")
            # Allow float32 neural reduction roundoff across GPU/runtime versions.
            np.testing.assert_allclose(value, original, rtol=1e-5, atol=1e-6,
                                       err_msg=f"Archived checkpoint replay differs: {key}")
            differences[key] = float(np.max(np.abs(value - original)))
            exact = exact and np.array_equal(value, original)
    return {"stage_replay__" + key: value for key, value in arrays.items()}, {
        "samples": spec["iw_samples"],
        "stream_seeds": [spec["iw_seed"] + index for index in range(spec["iw_streams"])],
        "max_abs_differences": differences, "reproduced_exactly": bool(exact),
        "rtol": 1e-5, "atol": 1e-6,
    }


def _stage_save_arrays(path, arrays):
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite stage diagnostic: {key}")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Serialized diagnostic inventory differs")
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    return _sha(path)


def stage_main(argv=None):
    """Evaluate fresh DART checkpoints with archived schema-4 code in a new process."""
    import time
    from datetime import datetime, timezone
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=stage_main.__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--expected-stage-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--worker-index", type=int, choices=range(4), required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=("final", "selected"), default=["final"])
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES must equal the requested GPU UUID")
    if any(name == "prigm" or name.startswith("prigm_") for name in sys.modules):
        raise RuntimeError("Frozen stage evaluation requires a fresh Python process")
    root, output = args.stage_root.absolute(), args.output_dir.absolute()
    manifest = verified_stage_source(root, args.expected_stage_sha256)
    all_jobs, jobs = _stage_jobs(manifest, args.datasets, args.checkpoints, args.worker_index)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh adjunct worker directory")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Adjunct output must be outside the frozen stage")
    evaluator_source = Path(__file__).resolve()
    evaluator_sources = {evaluator_source.name: _sha(evaluator_source)}
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "source/src"))
    from prigm_v2_experiment import (gpu_runtime, job_id, _matrix_job,
                                     verified_completion, verify_stage)
    # The archived runner checks the full prepared/metadata/parameters closure.
    if verify_stage(root) != manifest:
        raise ValueError("Stage changed during verification")
    runtime = gpu_runtime(args.gpu_uuid)
    from prigm_v2_model import BypassVAE
    output.mkdir()
    (output / "source").mkdir()
    artifacts = {}
    source_files = (evaluator_source, root / "source/src/prigm_v2_metrics.py",
                    root / "source/src/prigm_metrics.py")
    for source in source_files:
        name = "source/" + source.name
        expected = (evaluator_sources[source.name] if source == evaluator_source
                    else manifest["code_sha256"]["src/" + source.name])
        shutil.copyfile(source, output / name)
        if _sha(source) != expected or _sha(output / name) != expected:
            raise RuntimeError("Diagnostic source changed during preservation")
        artifacts[name] = expected
    spec = manifest["specification"]
    protocol = {
        "schema_version": 2, "purpose": "dart_stage_checkpoint_diagnostics",
        "stage": manifest["stage"], "stage_purpose": manifest["purpose"],
        "stage_manifest_sha256": args.expected_stage_sha256,
        "stage_source_sha256": manifest["code_sha256"],
        "evaluator_source_sha256": evaluator_sources,
        "worker_index": args.worker_index, "worker_count": 4,
        "jobs": [list(job) for job in jobs], "all_jobs": [list(job) for job in all_jobs],
        "dart_diagnostics": DART_DIAGNOSTICS, "density_coordinates": "observation",
        "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32,
        "training_measure": "declared_train_weights", "evaluation_measure": "declared_eval_weights",
        "archived_replay": {"samples": spec["iw_samples"], "streams": spec["iw_streams"],
                            "first_seed": spec["iw_seed"], "rtol": 1e-5, "atol": 1e-6},
        "runtime": runtime, "started_utc": started_utc,
    }
    _write_json(output / "protocol.json", protocol)
    protocol_sha = _sha(output / "protocol.json")
    inputs, loaded_data, reference_files = {}, {}, {}
    for dataset, condition, seed, variance_label, label in jobs:
        identifier = job_id(dataset, condition, seed, variance_label)
        if list(_matrix_job(manifest, identifier)) != [dataset, condition, seed, variance_label]:
            raise ValueError("Archived matrix identity differs")
        completion_path = root / "jobs" / identifier / "complete.json"
        completion_sha = _sha(completion_path) if completion_path.exists() else None
        attempt = verified_completion(root, identifier)
        if attempt is None:
            raise RuntimeError(f"Missing verified stage job: {identifier}")
        if (_sha(completion_path) != completion_sha
                or (completion_path in inputs and inputs[completion_path] != completion_sha)):
            raise RuntimeError("Stage completion changed during verification")
        completion = _read_json(completion_path)
        job_inputs = {completion_path: completion_sha,
                      attempt / "complete.json": _sha(attempt / "complete.json"),
                      **{attempt / name: expected
                         for name, expected in completion["artifact_sha256"].items()}}
        for path, expected in job_inputs.items():
            if path in inputs and inputs[path] != expected:
                raise RuntimeError("Stage input changed between checkpoints")
            inputs[path] = expected
        if dataset not in loaded_data:
            with np.load(root / "prepared" / f"{dataset}.npz", allow_pickle=False) as store:
                loaded_data[dataset] = {key: store[key].copy() for key in store.files}
        data = loaded_data[dataset]
        summary = _read_json(attempt / "summary.json")
        step = spec["max_steps"] if label == "final" else summary["selected_step"]
        evaluation_label = "final" if step == spec["max_steps"] else "selected"
        evaluation_path = attempt / f"evaluation-{evaluation_label}.npz"
        weight_path = attempt / f"{label}.weights.h5"
        variance = manifest["job_parameters"][identifier]["observation_variance"]
        model = BypassVAE(input_dim=spec["window"], latent_dim=spec["latent_dim"],
                          condition=condition, observation_variance=variance, seed=seed, dtype="float32")
        model.load_weights(str(weight_path))
        arrays, replay = _stage_replay(model, data, evaluation_path, spec)
        diagnostic_arrays, diagnostic_summary = dart_checkpoint_diagnostics(model, data)
        arrays.update(diagnostic_arrays)
        # One private reference per dataset/variance, never shared mutable worker state.
        reference_key = (dataset, float(variance))
        if reference_key not in reference_files:
            reference_arrays, reference_summary = dart_reference_diagnostics(data, variance)
            variance_tag = hashlib.sha256(repr(float(variance)).encode("ascii")).hexdigest()[:16]
            reference_name = f"references-{dataset}-{variance_tag}.npz"
            artifacts[reference_name] = _stage_save_arrays(output / reference_name, reference_arrays)
            reference_files[reference_key] = {
                "file": reference_name, "sha256": artifacts[reference_name],
                "observation_variance": float(variance), "summary": reference_summary,
                "array_shapes": {key: list(value.shape) for key, value in reference_arrays.items()},
            }
        stem = f"{identifier}-{label}"
        npz_name, json_name = stem + ".npz", stem + ".json"
        artifacts[npz_name] = _stage_save_arrays(output / npz_name, arrays)
        record = {
            "schema_version": 2, "purpose": protocol["purpose"],
            "identifier": identifier, "dataset": dataset, "condition": condition, "seed": seed,
            "variance_label": variance_label, "checkpoint": label, "step": step,
            "checkpoint_sha256": inputs[weight_path], "stage_completion_sha256": completion_sha,
            "stage_manifest_sha256": args.expected_stage_sha256,
            "prepared_sha256": manifest["prepared_sha256"][dataset],
            "stage_evaluation_file": evaluation_path.name,
            "stage_evaluation_sha256": inputs[evaluation_path], "archived_component_replay": replay,
            "observation_variance": float(variance), "dimension": spec["window"],
            "test_windows": len(data["test"]), "protocol_sha256": protocol_sha,
            "array_sha256": artifacts[npz_name],
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "dart_diagnostics": diagnostic_summary, "references": reference_files[reference_key],
        }
        _write_json(output / json_name, record)
        artifacts[json_name] = _sha(output / json_name)
        print(json.dumps({"completed": stem}), flush=True)
    verified_stage_source(root, args.expected_stage_sha256)
    for path, expected in inputs.items():
        if _sha(path) != expected:
            raise RuntimeError(f"Stage input changed during evaluation: {path}")
    if _sha(evaluator_source) != evaluator_sources[evaluator_source.name]:
        raise RuntimeError("Evaluator source changed during execution")
    expected_outputs = {**artifacts, "protocol.json": protocol_sha}
    entries = list(output.rglob("*"))
    if (output.is_symlink() or any(path.is_symlink() for path in entries)
            or {path.relative_to(output).as_posix() for path in entries}
            != set(expected_outputs) | {"source"}):
        raise RuntimeError("Diagnostic output inventory changed during execution")
    for name, expected in expected_outputs.items():
        if _sha(output / name) != expected:
            raise RuntimeError(f"Diagnostic output changed during execution: {name}")
    _write_json(output / "complete.json", {
        "schema_version": 2, "purpose": protocol["purpose"], "protocol_sha256": protocol_sha,
        "stage_manifest_sha256": args.expected_stage_sha256,
        "stage_source_sha256": manifest["code_sha256"], "evaluator_source_sha256": evaluator_sources,
        "artifact_sha256": artifacts, "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    })
    return 0


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if any(value == "--stage-root" or value.startswith("--stage-root=") for value in arguments):
        return stage_main(arguments)
    argv = arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--checkpoints", nargs="+", choices=("selected", "final"), default=["selected"])
    parser.add_argument("--worker-index", type=int, choices=range(4), required=True)
    parser.add_argument("--dart-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if args.datasets is None:
        args.datasets = list(DATASETS[1:] if args.dart_diagnostics else DATASETS)
    if args.dart_diagnostics and ("gaussian" in args.datasets or args.checkpoints != ["selected"]):
        parser.error("DART diagnostics require original DART selected checkpoints")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error("datasets and checkpoint labels must be unique")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES must equal the requested GPU UUID")
    root = args.run_root.absolute()
    manifest = verified_core_source(root, args.expected_core_sha256)
    if any(name in sys.modules for name in ("prigm_data", "prigm_model", "prigm_experiment", "prigm_metrics")):
        raise RuntimeError("Frozen checkpoint evaluation requires a fresh Python process")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh adjunct worker directory")

    # Keep the archived source inventory closed through the final verification.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "source/src"))
    from prigm_experiment import (gpu_runtime, importance_evaluate, job_id, load_prepared,
                                  verified_completion, verify_run)
    from prigm_model import GaussianVAE
    from prigm_v2_metrics import posterior_score_components
    runtime = gpu_runtime(args.gpu_uuid)
    verify_run(root, check_code=False)
    spec = manifest["specification"]
    # The first stream reproduces the original K=256 evaluation draw sequence.
    stream_seeds = (42001, 42003, 42004)
    all_jobs = [(dataset, arm, seed, label) for dataset in DATASETS if dataset in args.datasets
                for seed in SEEDS for arm in ARMS for label in args.checkpoints]
    jobs = all_jobs[args.worker_index::4]
    if not jobs:
        raise ValueError("Worker has no declared jobs")
    args.output_dir.mkdir()
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    helper = Path(sys.modules["prigm_v2_metrics"].__file__).resolve()
    source_files = (Path(__file__).resolve(), helper)
    evaluator_sources = {}
    for source in source_files:
        destination = source_dir / source.name
        shutil.copyfile(source, destination)
        evaluator_sources[source.name] = _sha(destination)
        if evaluator_sources[source.name] != _sha(source):
            raise RuntimeError("Evaluator source changed during preservation")
    protocol = {
        "schema_version": 1, "purpose": "frozen_checkpoint_components",
        "core_manifest_sha256": args.expected_core_sha256,
        "evaluator_source_sha256": evaluator_sources,
        "worker_index": args.worker_index, "worker_count": 4,
        "jobs": [list(job) for job in jobs], "all_jobs": [list(job) for job in all_jobs],
        "samples": 256, "stream_seeds": list(stream_seeds),
        "density_coordinates": "residual", "probability_dtype": "float64",
        "network_dtype": "float32", "chunk": 32,
        "runtime": runtime,
    }
    if args.dart_diagnostics:
        protocol.update(schema_version=2, purpose="dart_fixed_checkpoint_diagnostics",
                        dart_diagnostics=DART_DIAGNOSTICS,
                        training_measure="original_uniform_training_windows",
                        evaluation_measure="original_uniform_test_windows")
    _write_json(args.output_dir / "protocol.json", protocol)
    artifacts = {}
    loaded_data = {}
    reference_files = {}
    for dataset, arm, seed, label in jobs:
        identifier = job_id(dataset, arm, seed)
        attempt = verified_completion(root, identifier)
        if attempt is None:
            raise RuntimeError(f"Missing verified core job: {identifier}")
        if dataset not in loaded_data:
            loaded_data[dataset] = load_prepared(root, dataset)
        data = loaded_data[dataset]
        original_summary = _read_json(attempt / "summary.json")
        weight_name = "best.weights.h5" if label == "selected" else "final.weights.h5"
        weight_sha = _sha(attempt / weight_name)
        model = GaussianVAE(input_dim=spec["window"], latent_dim=spec["latent_dim"],
                            arm=arm, base_variance=float(data["base_variance"]), seed=seed)
        model.load_weights(str(attempt / weight_name))
        streams = [posterior_score_components(model, data["test"], 256, stream_seed,
                                              density_coordinates="residual")
                   for stream_seed in stream_seeds]
        arrays = {key: np.stack([stream[key] for stream in streams]) for key in streams[0]}
        original = importance_evaluate(model, data["test"], 256, stream_seeds[0])
        np.testing.assert_array_equal(arrays["log_score"][0], original["log_score"])
        np.testing.assert_array_equal(arrays["ess"][0], original["ess"])
        replay_differences = None
        if label == "selected":
            with np.load(attempt / "evaluation.npz", allow_pickle=False) as saved:
                replay_differences = {
                    "log_score_max_abs": float(np.max(np.abs(arrays["log_score"][0] - saved["log_score"]))),
                    "ess_max_abs": float(np.max(np.abs(arrays["ess"][0] - saved["ess"]))),
                    "posterior_mean_mse_max_abs": float(np.max(np.abs(
                        arrays["posterior_mean_mse"][0] - saved["corrected_mse"]))),
                }
        diagnostic_summary = None
        reference_record = None
        if args.dart_diagnostics:
            diagnostic_arrays, diagnostic_summary = dart_checkpoint_diagnostics(model, data)
            arrays.update(diagnostic_arrays)
            reference_key = (dataset, float(model.observation_variance))
            if reference_key not in reference_files:
                reference_arrays, reference_summary = dart_reference_diagnostics(
                    data, model.observation_variance)
                variance_tag = hashlib.sha256(repr(reference_key[1]).encode("ascii")).hexdigest()[:16]
                reference_name = f"references-{dataset}-{variance_tag}.npz"
                np.savez_compressed(args.output_dir / reference_name, **reference_arrays)
                artifacts[reference_name] = _sha(args.output_dir / reference_name)
                reference_files[reference_key] = {
                    "file": reference_name, "sha256": artifacts[reference_name],
                    "observation_variance": reference_key[1], "summary": reference_summary,
                }
            reference_record = reference_files[reference_key]
        if _sha(attempt / weight_name) != weight_sha:
            raise RuntimeError("Input checkpoint changed during evaluation")
        stem = f"{identifier}-{label}"
        npz_name, json_name = stem + ".npz", stem + ".json"
        np.savez_compressed(args.output_dir / npz_name, **arrays)
        record = {
            "identifier": identifier, "dataset": dataset, "arm": arm, "seed": seed,
            "checkpoint": label, "checkpoint_sha256": weight_sha,
            "core_completion_sha256": _sha(root / "jobs" / identifier / "complete.json"),
            "prepared_sha256": manifest["prepared_sha256"][dataset + ".npz"],
            "selected_step": original_summary["best_step"],
            "steps_run": original_summary["steps_run"],
            "observation_variance": float(model.observation_variance),
            "test_windows": len(data["test"]), "dimension": spec["window"],
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "array_sha256": _sha(args.output_dir / npz_name),
            "frozen_evaluator_reproduced_exactly": True,
            "core_evaluation_sha256": _sha(attempt / "evaluation.npz"),
            "archived_selected_replay_differences": replay_differences,
        }
        if args.dart_diagnostics:
            record.update(schema_version=2, purpose=protocol["purpose"],
                          protocol_sha256=_sha(args.output_dir / "protocol.json"),
                          dart_diagnostics=diagnostic_summary,
                          references=reference_record)
        _write_json(args.output_dir / json_name, record)
        artifacts[npz_name] = record["array_sha256"]
        artifacts[json_name] = _sha(args.output_dir / json_name)
        print(json.dumps({"completed": stem}), flush=True)
    verified_core_source(root, args.expected_core_sha256)
    for source in source_files:
        if _sha(source) != evaluator_sources[source.name]:
            raise RuntimeError("Evaluator source changed during execution")
    _write_json(args.output_dir / "complete.json", {
        "schema_version": protocol["schema_version"], "purpose": protocol["purpose"],
        "protocol_sha256": _sha(args.output_dir / "protocol.json"),
        "core_manifest_sha256": args.expected_core_sha256,
        "evaluator_source_sha256": evaluator_sources, "artifact_sha256": artifacts,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
