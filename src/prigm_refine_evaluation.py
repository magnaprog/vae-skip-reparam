"""Fixed, evaluation-only importance-budget validation of the completed PriGM core."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import numpy as np

import prigm_experiment as core

CORE_MANIFEST_SHA256 = "37cc7445600eee541ee50aa580f6fdd6c0f37cf420074dbecc635c8d202946f5"
SOURCE_FILE = Path(__file__).resolve()
SOURCE_NAME = "src/prigm_refine_evaluation.py"
MANIFEST = "refinement_manifest.json"
SAMPLE_COUNTS = (4096, 16384)
SAMPLING_SEEDS = (44001, 44002)
BUDGETS = (256, *SAMPLE_COUNTS)
ARTIFACTS = ("evaluation.npz", "summary.json")
PAIRS = (("B", "A"), ("C", "A"), ("B", "D"))


def specification():
    return {
        "sample_counts": list(SAMPLE_COUNTS), "sampling_seeds": list(SAMPLING_SEEDS),
        "window_selection": "first_test_windows", "max_test_windows": 32, "chunk": 4,
        "original_samples": 256, "original_sampling_seed": 42001,
        "checkpoint": "best.weights.h5", "workers": 4,
        "job_order": "dataset_then_training_seed_then_arm", "shard": "jobs[index::4]",
        "datasets": list(core.DATASETS), "training_seeds": list(core.SEEDS),
        "arms": list(core.ARMS), "training_or_checkpoint_selection": False,
    }


def canonical_jobs():
    return [(dataset, arm, seed) for dataset in core.DATASETS
            for seed in core.SEEDS for arm in core.ARMS]


def shard_jobs(index):
    if type(index) is not int or index not in range(4):
        raise ValueError("Worker index must be an integer in 0..3")
    return canonical_jobs()[index::4]


def read_json(path):
    core.sha256(path)  # Reject missing files and symlinks, including dangling links.
    return core.read_json(path)


def same_json(actual, expected):
    # Python equality conflates true/1 and 4.0/4; frozen settings require exact JSON types.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False)


def require_directory(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Expected a regular directory: {path}")


def publish_json(path, payload):
    """Publish a fully written JSON atomically, without replacing any existing file."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Same filesystem; exclusive publication, unlike replace().
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def verify_core(root):
    if core.sha256(Path(root) / "run_manifest.json") != CORE_MANIFEST_SHA256:
        raise ValueError("Core manifest differs from the approved SHA256")
    manifest = core.verify_run(root)
    if manifest["purpose"] != "core":
        raise ValueError("Refinement requires the completed core study")
    return manifest


def test_subset(data):
    x = data["test"]
    starts = data["starts_test"]
    if (x.ndim != 2 or x.shape[1] != core.SPEC["window"] or len(x) == 0
            or starts.shape != (len(x),) or starts.dtype.kind not in "iu"):
        raise ValueError("Invalid prepared test-window identity")
    count = min(32, len(x))
    if not np.isfinite(x[:count]).all():
        raise ValueError("Nonfinite prepared test windows")
    return x[:count], starts[:count]


def core_job_identity(root, dataset, arm, seed, count):
    identifier = core.job_id(dataset, arm, seed)
    job_root = Path(root) / "jobs" / identifier
    require_directory(job_root.parent)
    require_directory(job_root)
    completion_hash = core.sha256(job_root / "complete.json")
    attempt = core.verified_completion(root, identifier)
    if attempt is None:
        raise ValueError(f"Missing completed core job: {identifier}")
    require_directory(attempt)
    config = read_json(attempt / "configuration.json")
    if (config.get("run_manifest_sha256") != CORE_MANIFEST_SHA256
            or not same_json(config.get("specification"), core.specification("core"))):
        raise ValueError(f"Core configuration differs from the protocol: {identifier}")
    identity = {
        "core_job_id": identifier, "dataset": dataset, "arm": arm, "seed": seed,
        "count": count, "input_dim": core.SPEC["window"],
        "worker_index": canonical_jobs().index((dataset, arm, seed)) % 4,
        "core_completion_sha256": completion_hash,
        "core_checkpoint_sha256": core.sha256(attempt / "best.weights.h5"),
        "core_evaluation_sha256": core.sha256(attempt / "evaluation.npz"),
    }
    if core.sha256(job_root / "complete.json") != completion_hash:
        raise ValueError("Core completion changed during verification")
    return identity, attempt


def expected_manifest(core_root):
    verify_core(core_root)
    jobs = {}
    for dataset in core.DATASETS:
        x, _ = test_subset(core.load_prepared(core_root, dataset))
        for seed in core.SEEDS:
            for arm in core.ARMS:
                identity, _ = core_job_identity(core_root, dataset, arm, seed, len(x))
                jobs[identity["core_job_id"]] = identity
    return {
        "schema_version": 1, "purpose": "importance_budget_refinement",
        "core_manifest_sha256": CORE_MANIFEST_SHA256,
        "code_sha256": {SOURCE_NAME: core.sha256(SOURCE_FILE)},
        "specification": specification(), "jobs": jobs,
    }


def verify_frozen(core_root, output_root, manifest):
    verify_core(core_root)
    if not same_json(read_json(Path(output_root) / MANIFEST), manifest):
        raise ValueError("Refinement manifest changed")
    expected = manifest["code_sha256"][SOURCE_NAME]
    if (core.sha256(SOURCE_FILE) != expected
            or core.sha256(Path(output_root) / "source" / SOURCE_NAME) != expected):
        raise ValueError("Frozen refinement source changed")


def verify_refinement(core_root, output_root):
    output_root = Path(output_root)
    require_directory(output_root)
    require_directory(output_root / "jobs")
    manifest = read_json(output_root / MANIFEST)
    if not same_json(manifest, expected_manifest(core_root)):
        raise ValueError("Refinement manifest differs from frozen protocol or core identity")
    verify_frozen(core_root, output_root, manifest)
    return manifest


def prepare(core_root, output_root):
    core_root, output_root = Path(core_root).resolve(), Path(output_root).absolute()
    if core_root == output_root.resolve() or core_root in output_root.resolve().parents:
        raise ValueError("Refinement output must be outside the frozen core root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Use a fresh refinement output directory")
    manifest = expected_manifest(core_root)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "jobs").mkdir()
    (output_root / "source" / "src").mkdir(parents=True)
    shutil.copyfile(SOURCE_FILE, output_root / "source" / SOURCE_NAME)
    publish_json(output_root / MANIFEST, manifest)
    verify_refinement(core_root, output_root)
    print(f"PREPARED {len(manifest['jobs'])} jobs; manifest SHA256 "
          f"{core.sha256(output_root / MANIFEST)}", flush=True)


def original_arrays(attempt, data):
    x, starts = test_subset(data)
    arrays = {"window_index": np.arange(len(x), dtype=np.int64), "test_starts": starts.copy()}
    with np.load(attempt / "evaluation.npz", allow_pickle=False) as evaluation:
        for key in ("log_score", "ess"):
            values = evaluation[key]
            if values.shape != (len(data["test"]),):
                raise ValueError("Core evaluation does not match prepared test windows")
            arrays[f"{key}_256"] = values[:len(x)].copy()
    return arrays


def validate_arrays(arrays, original, count):
    keys = {"window_index", "test_starts"} | {
        f"{key}_{k}" for k in BUDGETS for key in ("log_score", "ess")}
    if set(arrays) != keys:
        raise ValueError("Incomplete refinement array schema")
    for key, value in arrays.items():
        if value.shape != (count,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid refinement array: {key}")
    for key, expected in original.items():
        if arrays[key].dtype != expected.dtype or not np.array_equal(arrays[key], expected):
            raise ValueError(f"Refinement differs from the original same-window subset: {key}")
    for k in BUDGETS:
        ess = arrays[f"ess_{k}"]
        if np.any(ess < 1 - 1e-7) or np.any(ess > k * (1 + 1e-7)):
            raise ValueError(f"ESS outside importance-sampling bounds at K={k}")


def metrics(arrays, dimension):
    original = np.asarray(arrays["log_score_256"], dtype=np.float64)
    return {str(k): {
        "mean_nll_nats_per_dimension": float(-np.mean(arrays[f"log_score_{k}"], dtype=np.float64) / dimension),
        "median_ess": float(np.median(arrays[f"ess_{k}"])),
        "mean_log_score_change_from_256_per_dimension": float(np.mean(
            np.asarray(arrays[f"log_score_{k}"], dtype=np.float64) - original) / dimension),
    } for k in BUDGETS}


def validate_runtime(runtime):
    if (not isinstance(runtime, dict)
            or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                                str(runtime.get("gpu_uuid", "")))
            or "GPU:0" not in str(runtime.get("placement", ""))
            or runtime.get("deterministic_ops_requested") is not True
            or type(runtime.get("pid")) is not int or runtime["pid"] <= 0
            or type(runtime.get("elapsed_seconds")) not in (int, float)
            or not np.isfinite(runtime["elapsed_seconds"]) or runtime["elapsed_seconds"] < 0):
        raise ValueError("Invalid private GPU runtime record")


def job_summary(output_root, manifest, identifier, arrays, runtime):
    validate_runtime(runtime)
    identity = manifest["jobs"][identifier]
    return {
        "schema_version": 1, **identity,
        "refinement_manifest_sha256": core.sha256(Path(output_root) / MANIFEST),
        "specification": specification(), "metrics": metrics(arrays, identity["input_dim"]),
        "runtime_private": runtime,
    }


def verify_job(core_root, output_root, manifest, identifier):
    """Verify a job against an already verified manifest, including its core baseline."""
    identity = manifest["jobs"][identifier]
    actual, core_attempt = core_job_identity(
        core_root, identity["dataset"], identity["arm"], identity["seed"], identity["count"])
    if actual != identity:
        raise ValueError(f"Core job identity changed: {identifier}")
    job_root = Path(output_root) / "jobs" / identifier
    if job_root.is_symlink():
        raise ValueError("Symlinked refinement job directory")
    if not job_root.exists():
        return None
    require_directory(job_root)
    marker = job_root / "complete.json"
    if not marker.exists() and not marker.is_symlink():
        return None
    completion = read_json(marker)
    if (set(completion) != {"schema_version", "core_job_id", "refinement_manifest_sha256",
                            "attempt", "artifact_sha256"}
            or type(completion["schema_version"]) is not int or completion["schema_version"] != 1
            or completion["core_job_id"] != identifier
            or completion["refinement_manifest_sha256"] != core.sha256(Path(output_root) / MANIFEST)
            or not re.fullmatch(r"attempt-[0-9a-f]{32}", str(completion["attempt"]))
            or set(completion["artifact_sha256"]) != set(ARTIFACTS)):
        raise ValueError("Invalid refinement completion identity")
    attempt = job_root / completion["attempt"]
    require_directory(attempt)
    for name, expected in completion["artifact_sha256"].items():
        if core.sha256(attempt / name) != expected:
            raise ValueError(f"Completed refinement artifact changed: {identifier}/{name}")
    with np.load(attempt / "evaluation.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
        if len(stored.files) != len(arrays):
            raise ValueError("Duplicate refinement arrays")
    data = core.load_prepared(core_root, identity["dataset"])
    validate_arrays(arrays, original_arrays(core_attempt, data), identity["count"])
    summary = read_json(attempt / "summary.json")
    if not same_json(summary, job_summary(
            output_root, manifest, identifier, arrays, summary.get("runtime_private"))):
        raise ValueError("Refinement summary identity, settings or metrics changed")
    return attempt, summary, arrays


def run_job(core_root, output_root, manifest, job, runtime, model_class, tf):
    dataset, arm, seed = job
    identifier = core.job_id(*job)
    verify_frozen(core_root, output_root, manifest)
    job_root = Path(output_root) / "jobs" / identifier
    if job_root.is_symlink():
        raise ValueError("Symlinked refinement job directory")
    job_root.mkdir(exist_ok=True)
    lock_path = job_root / ".writer.lock"
    if lock_path.is_symlink():
        raise ValueError("Symlinked writer lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify_job(core_root, output_root, manifest, identifier) is not None:
            print(f"VERIFIED REUSE {identifier}", flush=True)
            return
        if any(job_root.glob("attempt-*")):
            print(f"RESTART {identifier} in a fresh attempt; incomplete artifacts retained", flush=True)
        directory = job_root / ("attempt-" + uuid.uuid4().hex)
        directory.mkdir()
        data = core.load_prepared(core_root, dataset)
        x, _ = test_subset(data)
        identity, core_attempt = core_job_identity(core_root, dataset, arm, seed, len(x))
        if identity != manifest["jobs"][identifier]:
            raise ValueError("Core job changed before evaluation")
        arrays = original_arrays(core_attempt, data)
        started = time.monotonic()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = model_class(input_dim=identity["input_dim"], latent_dim=core.SPEC["latent_dim"],
                            arm=arm, base_variance=float(data["base_variance"]), seed=seed)
        model.load_weights(str(core_attempt / "best.weights.h5"))
        for k, sampling_seed in zip(SAMPLE_COUNTS, SAMPLING_SEEDS):
            stats = core.importance_evaluate(model, x, k, sampling_seed, chunk=4)
            for key in ("log_score", "ess"):
                arrays[f"{key}_{k}"] = stats[key]
            print(f"SCORED {identifier} K={k} windows={len(x)}", flush=True)
        del model
        validate_arrays(arrays, original_arrays(core_attempt, data), len(x))
        private = dict(runtime, elapsed_seconds=time.monotonic() - started)
        summary = job_summary(output_root, manifest, identifier, arrays, private)
        with (directory / "evaluation.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        publish_json(directory / "summary.json", summary)
        verify_frozen(core_root, output_root, manifest)
        actual, _ = core_job_identity(core_root, dataset, arm, seed, len(x))
        if actual != identity:
            raise ValueError("Core job changed during evaluation")
        publish_json(job_root / "complete.json", {
            "schema_version": 1, "core_job_id": identifier, "attempt": directory.name,
            "refinement_manifest_sha256": core.sha256(Path(output_root) / MANIFEST),
            "artifact_sha256": {name: core.sha256(directory / name) for name in ARTIFACTS},
        })
        verify_job(core_root, output_root, manifest, identifier)
        print(f"DONE {identifier}", flush=True)


def worker(core_root, output_root, worker_index, gpu_uuid):
    jobs = shard_jobs(worker_index)
    core_root, output_root = Path(core_root).resolve(), Path(output_root).resolve()
    manifest = verify_refinement(core_root, output_root)
    if "tensorflow" in sys.modules:
        raise RuntimeError("Run each GPU worker in a fresh process before importing TensorFlow")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid):
        raise ValueError("Expected a full physical GPU UUID")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, TF_DETERMINISTIC_OPS="1",
                      TF_CPP_MIN_LOG_LEVEL="2", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    runtime = core.gpu_runtime(gpu_uuid)
    if runtime.get("gpu_uuid") != gpu_uuid:
        raise ValueError("GPU runtime UUID mismatch")
    validate_runtime(dict(runtime, elapsed_seconds=0.0))
    import tensorflow as tf
    from prigm_model import GaussianVAE
    for job in jobs:
        run_job(core_root, output_root, manifest, job, runtime, GaussianVAE, tf)
    verify_frozen(core_root, output_root, manifest)


def paired_comparisons(records):
    """Median/range of three paired-seed mean NLL differences on identical windows."""
    expected = {core.job_id(*job) for job in canonical_jobs()}
    if set(records) != expected:
        raise ValueError("Paired comparison requires exactly the 60 canonical jobs")
    results = {}
    for dataset in core.DATASETS:
        results[dataset] = {}
        for left, right in PAIRS:
            budgets = {}
            for k in BUDGETS:
                differences = []
                for seed in core.SEEDS:
                    a = records[core.job_id(dataset, left, seed)]
                    b = records[core.job_id(dataset, right, seed)]
                    for key in ("window_index", "test_starts"):
                        if not np.array_equal(a[key], b[key]):
                            raise ValueError("Paired comparison requires the same ordered windows")
                    score_a, score_b = a[f"log_score_{k}"], b[f"log_score_{k}"]
                    if score_a.shape != a["window_index"].shape or score_b.shape != score_a.shape:
                        raise ValueError("Paired score/window shape mismatch")
                    differences.append(float(np.mean(
                        np.asarray(score_b, dtype=np.float64) - np.asarray(score_a, dtype=np.float64)
                    ) / core.SPEC["window"]))
                budgets[str(k)] = {"training_seeds": list(core.SEEDS), "paired_seed_deltas": differences,
                                   "median": float(np.median(differences)),
                                   "range": [min(differences), max(differences)]}
            results[dataset][f"{left}-{right}"] = budgets
    return results


def summarize(core_root, output_root):
    core_root, output_root = Path(core_root).resolve(), Path(output_root).resolve()
    manifest = verify_refinement(core_root, output_root)
    files = [output_root / MANIFEST, output_root / "source" / SOURCE_NAME,
             output_root / "results.json"]
    summaries, records = [], {}
    for job in canonical_jobs():
        identifier = core.job_id(*job)
        verified = verify_job(core_root, output_root, manifest, identifier)
        if verified is None:
            raise ValueError(f"Missing completed refinement job: {identifier}")
        attempt, summary, arrays = verified
        # Explicit public whitelist: never propagate runtime UUIDs, PIDs, paths or inventories.
        summaries.append({**manifest["jobs"][identifier], "metrics": summary["metrics"]})
        records[identifier] = arrays
        files.append(attempt.parent / "complete.json")
        files.extend(attempt / name for name in ARTIFACTS)
    output = {
        "schema_version": 1, "purpose": manifest["purpose"],
        "core_manifest_sha256": CORE_MANIFEST_SHA256,
        "refinement_manifest_sha256": core.sha256(output_root / MANIFEST),
        "specification": specification(), "coordinates": "prepared_standardized",
        "comparison": "NLL(left) minus NLL(right), nats/dimension; positive means left is worse; "
                      "median and range across three paired training seeds on the same test prefix",
        "jobs": summaries, "paired_comparisons": paired_comparisons(records),
    }
    verify_frozen(core_root, output_root, manifest)
    lock_path = output_root / ".summary.lock"
    if lock_path.is_symlink():
        raise ValueError("Symlinked summary lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path, archive_path = output_root / "results.json", output_root / "archive_manifest.json"
        present = [p.exists() or p.is_symlink() for p in (result_path, archive_path)]
        if any(present) and not all(present):
            raise FileExistsError("Incomplete summary publication; refusing to overwrite or silently reuse")
        if all(present):
            if not same_json(read_json(result_path), output):
                raise ValueError("Published results differ from verified jobs")
        else:
            publish_json(result_path, output)
        archive = {
            "schema_version": 1, "core_manifest_sha256": CORE_MANIFEST_SHA256,
            "scope": "Refinement manifest, source, successful job markers/arrays/private summaries, and sanitized results",
            "artifact_sha256": {str(p.relative_to(output_root)): core.sha256(p) for p in sorted(files)},
        }
        if all(present):
            if not same_json(read_json(archive_path), archive):
                raise ValueError("Published archive identity changed")
        else:
            publish_json(archive_path, archive)
    print("ALL DONE: 60 verified refinement jobs; sanitized results.json and archive_manifest.json", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "worker", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--core-root", required=True)
        command.add_argument("--output-root", required=True)
        if name == "worker":
            command.add_argument("--worker-index", type=int, choices=range(4), required=True)
            command.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.core_root, args.output_root)
    elif args.command == "worker":
        worker(args.core_root, args.output_root, args.worker_index, args.gpu_uuid)
    else:
        summarize(args.core_root, args.output_root)


if __name__ == "__main__":
    main()
