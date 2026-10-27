"""Prepare data and train the four input-skip and observation-variance conditions."""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from data import prepare_dataset
from metrics import (energy_score, fit_ppca, gaussian_kl,
                     gaussian_log_prob, importance_statistics)

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = tuple(f"src/{name}.py" for name in ("data", "metrics", "model", "experiment"))
DATASETS = ("gaussian", "23461t2022", "44402t2016", "46407t2011", "52403t2020")
CONDITIONS = ("no_skip_base", "skip_base", "no_skip_high", "skip_high")
SEEDS = (42, 1729, 31415)
JOB_ARTIFACTS = ("configuration.json", "history.json", "progress.json", "best.weights.h5",
                 "final.weights.h5", "summary.json", "evaluation.npz",
                 "last_checkpoint.index", "last_checkpoint.data-00000-of-00001")
SPEC = {
    "window": 192, "training_stride": 16, "latent_dim": 16,
    "encoder_widths": [128, 64], "decoder_widths": [64, 128],
    "batch_size": 64, "learning_rate": 0.001, "kl_beta": 1.0,
    "max_steps": 10000, "validation_every": 250, "patience": 8,
    "gradient_global_norm": 10.0, "posterior_log_variance_bounds": [-12.0, 12.0],
    "variance_floor": 0.0001, "validation_samples": 64,
    "evaluation_samples": 256, "stability_samples": 1024,
    "stability_windows": 32, "prior_samples": 512,
    "training_seeds": list(SEEDS), "conditions": list(CONDITIONS),
    "checkpoint_criterion": "minimum_validation_negative_importance_log_score",
    "validation_sampling_seed": 41001, "evaluation_sampling_seed": 42001,
    "stability_sampling_seed": 42002, "prior_sampling_seed": 43001,
    "exact_resume_supported": False,
}


def sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
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


def read_json(path):
    return json.loads(Path(path).read_text())


def specification(purpose):
    if purpose not in ("core", "smoke"):
        raise ValueError("Unknown experiment purpose")
    spec = dict(SPEC)
    if purpose == "smoke":
        spec.update(max_steps=4, validation_every=2, patience=2,
                    validation_samples=8, evaluation_samples=16,
                    stability_samples=32, stability_windows=8, prior_samples=32)
    return spec


def verify_run(run_root):
    """Check the run's settings, inputs and saved code against this implementation."""
    root = Path(run_root).resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported run manifest: schema_version=2 with "
                         "descriptive condition names is required; prepare a fresh run")
    if manifest["datasets"] != list(DATASETS) or manifest["specification"] != specification(manifest["purpose"]):
        raise ValueError("Manifest differs from the experiment settings")
    if set(manifest["code_sha256"]) != set(CODE_FILES):
        raise ValueError("Incomplete source identity")
    if set(manifest["prepared_sha256"]) != {"metadata.json", *(name + ".npz" for name in DATASETS)}:
        raise ValueError("Incomplete prepared-data identity")
    for name, expected in manifest["code_sha256"].items():
        if sha256(root / "source" / name) != expected:
            raise ValueError(f"Saved experiment source changed: {name}")
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Experiment source changed: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        if sha256(root / "prepared" / name) != expected:
            raise ValueError(f"Prepared dataset changed: {name}")
    return manifest


def prepare(run_root, purpose="core"):
    root = Path(run_root).absolute()
    if root.exists():
        raise FileExistsError("Use a fresh run directory")
    root.mkdir(parents=True)
    (root / "prepared").mkdir()
    (root / "jobs").mkdir()
    (root / "logs").mkdir()
    (root / "source" / "src").mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copyfile(ROOT / name, root / "source" / name)
    spec = specification(purpose)
    metadata = {}
    for name in DATASETS:
        data = prepare_dataset(name, ROOT / "data")
        ppca = fit_ppca(data.train, rank=spec["latent_dim"], variance_floor=spec["variance_floor"])
        diagonal = np.maximum(np.var(data.train, axis=0), spec["variance_floor"])
        arrays = {
            "train": data.train, "validation": data.validation, "test": data.test,
            "mean": np.array(data.mean), "scale": np.array(data.scale),
            "base_variance": np.array(ppca["variance"]),
            "ppca_mean": ppca["mean"], "ppca_covariance": ppca["covariance"],
            "diagonal_covariance": np.diag(diagonal),
        }
        arrays.update({f"starts_{split}": value for split, value in data.starts.items()})
        if data.oracle_covariance is not None:
            arrays["oracle_covariance"] = data.oracle_covariance
        np.savez_compressed(root / "prepared" / f"{name}.npz", **arrays)
        metadata[name] = dict(data.metadata)
        metadata[name].update(
            base_variance=float(ppca["variance"]), mean=float(data.mean), scale=float(data.scale),
            windows={s: int(len(getattr(data, s))) for s in ("train", "validation", "test")},
        )
    atomic_json(root / "prepared" / "metadata.json", metadata)
    manifest = {
        "schema_version": 2, "purpose": purpose, "run_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(), "specification": spec,
        "datasets": list(DATASETS),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "prepared_sha256": {p.name: sha256(p) for p in sorted((root / "prepared").iterdir())},
        "claim_scope": "Controlled window-distribution study; observed DART records are not clean-signal labels.",
    }
    atomic_json(root / "run_manifest.json", manifest)
    print(json.dumps({"run_root": str(root), "manifest_sha256": sha256(root / "run_manifest.json"),
                      "purpose": purpose, "prepared": metadata}, indent=2))


def load_prepared(root, name):
    if name not in DATASETS:
        raise ValueError("Unknown dataset")
    with np.load(Path(root) / "prepared" / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def importance_evaluate(model, observations, samples, seed, chunk=32):
    rng = np.random.default_rng(seed)
    pieces = {key: [] for key in ("log_score", "ess", "normalized_ess")}
    for start in range(0, len(observations), chunk):
        x = np.asarray(observations[start:start + chunk], dtype=np.float64)
        mean, log_variance = model.encode(x.astype(np.float32))
        mean, log_variance = np.asarray(mean, dtype=np.float64), np.asarray(log_variance, dtype=np.float64)
        epsilon = rng.standard_normal((len(x), samples, model.latent_dim))
        z = mean[:, None, :] + np.exp(0.5 * log_variance[:, None, :]) * epsilon
        residual = np.asarray(model.decode_residual(z.reshape(-1, model.latent_dim).astype(np.float32)),
                              dtype=np.float64).reshape(len(x), samples, -1)
        dimension = x.shape[1]
        log_likelihood = (-0.5 * np.square(model.b * x[:, None, :] - residual).sum(axis=-1)
                          / model.residual_variance
                          - 0.5 * dimension * np.log(2 * np.pi * model.residual_variance)
                          + dimension * np.log(model.b))
        log_prior = -0.5 * (np.square(z) + np.log(2 * np.pi)).sum(axis=-1)
        log_q = -0.5 * (np.square(epsilon) + log_variance[:, None, :] + np.log(2 * np.pi)).sum(axis=-1)
        stats = importance_statistics(log_likelihood + log_prior - log_q)
        for key in pieces:
            pieces[key].append(stats[key])
    return {key: np.concatenate(value) for key, value in pieces.items()}


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
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True, timeout=20,
    ).stdout
    matches = [line for line in output.splitlines()
               if line.split(",", 1)[0].strip() == str(os.getpid())]
    if not any(expected_uuid in line for line in matches):
        raise RuntimeError("Current process is not attested on the requested GPU UUID")
    return {"gpu_uuid": expected_uuid, "pid": os.getpid(), "placement": probe.device,
            "process_inventory": matches, "tensorflow": tf.__version__, "python": sys.version,
            "numpy": np.__version__, "deterministic_ops_requested": True}


def job_id(dataset, condition, seed):
    if dataset not in DATASETS or condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Job is outside the experiment's dataset, condition and seed combinations")
    return f"{dataset}_{condition}_{seed}"


def verified_completion(root, identifier):
    root = Path(root)
    path = root / "jobs" / identifier / "complete.json"
    if not path.exists():
        return None
    completion = read_json(path)
    if completion["run_manifest_sha256"] != sha256(root / "run_manifest.json"):
        raise ValueError("Job belongs to a different run manifest")
    attempt = root / "jobs" / identifier / completion["attempt"]
    if attempt.parent != root / "jobs" / identifier or attempt.is_symlink():
        raise ValueError("Invalid job attempt path")
    if set(completion["artifact_sha256"]) != set(JOB_ARTIFACTS):
        raise ValueError("Incomplete job artifact identity")
    for name, expected in completion["artifact_sha256"].items():
        if Path(name).name != name or sha256(attempt / name) != expected:
            raise ValueError(f"Completed job artifact changed: {identifier}/{name}")
    for filename in ("configuration.json", "summary.json"):
        record = read_json(attempt / filename)
        if job_id(record["dataset"], record["condition"], record["seed"]) != identifier:
            raise ValueError("Completed job identity does not match its condition slot")
    return attempt


def evaluate_job(model, data, spec, directory):
    x = data["test"]
    stats = importance_evaluate(model, x, spec["evaluation_samples"], spec["evaluation_sampling_seed"])
    count = min(len(x), spec["stability_windows"])
    stable = importance_evaluate(model, x[:count], spec["stability_samples"], spec["stability_sampling_seed"])
    replicate = importance_evaluate(model, x[:count], spec["evaluation_samples"], spec["stability_sampling_seed"] + 1)
    mean, log_variance = model.encode(x.astype(np.float32))
    mean, log_variance = np.asarray(mean), np.asarray(log_variance)
    residual = np.asarray(model.decode_residual(mean), dtype=np.float64)
    generative_mean = residual / model.b
    corrected_mse = np.square(x - generative_mean).mean(axis=1)
    self_mse = np.square(x - (residual + model.alpha * x)).mean(axis=1)
    permuted = np.asarray(model.decode_mean(np.roll(mean, 1, axis=0)), dtype=np.float64)
    permutation_delta = np.square(x - permuted).mean(axis=1) - corrected_mse
    kl = gaussian_kl(mean, log_variance)
    rng = np.random.default_rng(spec["prior_sampling_seed"])
    latent = rng.standard_normal((spec["prior_samples"], model.latent_dim)).astype(np.float32)
    noise = rng.standard_normal((spec["prior_samples"], model.input_dim))
    draws = np.asarray(model.decode_mean(latent), dtype=np.float64) + np.sqrt(
        model.residual_variance / model.b ** 2) * noise
    energy = energy_score(draws, x)
    scale = float(data["scale"])
    arrays = dict(stats, self_mse=self_mse, corrected_mse=corrected_mse,
                  kl=kl, permutation_delta=permutation_delta, energy=energy,
                  stability_log_score=stable["log_score"], stability_ess=stable["ess"],
                  replicate_log_score=replicate["log_score"],
                  stability_primary_log_score=stats["log_score"][:count],
                  prior_samples=draws, posterior_mean=generative_mean,
                  posterior_kl_per_dimension=0.5 * (mean.astype(float) ** 2 + np.exp(log_variance.astype(float)) - 1 - log_variance))
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite evaluation artifact: {key}")
    np.savez_compressed(directory / "evaluation.npz", **arrays)
    summary = {
        "test_windows": len(x), "negative_log_score_nats_per_dimension": float(-stats["log_score"].mean() / model.input_dim),
        "negative_log_score_meter_coordinates_per_dimension": float(-stats["log_score"].mean() / model.input_dim + np.log(scale)),
        "median_effective_samples": float(np.median(stats["ess"])),
        "median_normalized_ess": float(np.median(stats["normalized_ess"])),
        "self_mse": float(self_mse.mean()), "corrected_mse": float(corrected_mse.mean()),
        "kl_nats_per_window": float(kl.mean()), "energy_score": float(energy.mean()),
        "energy_score_meters": float(energy.mean() * scale),
        "permutation_corrected_mse_increase": float(permutation_delta.mean()),
        "stability_mean_log_score_change_per_dimension": float((stable["log_score"] - stats["log_score"][:count]).mean() / model.input_dim),
        "replicate_mean_log_score_change_per_dimension": float((replicate["log_score"] - stats["log_score"][:count]).mean() / model.input_dim),
    }
    return summary


def mapped_control(model, data):
    from model import GaussianVAE
    target_condition = {
        "no_skip_base": "skip_base", "skip_base": "no_skip_base",
        "no_skip_high": "skip_high", "skip_high": "no_skip_high",
    }[model.condition]
    mapped = GaussianVAE(input_dim=model.input_dim, latent_dim=model.latent_dim,
                         condition=target_condition, base_variance=float(data["base_variance"]), seed=42)
    mapped.map_from(model)
    x = data["validation"][:8].astype(np.float32)
    z = np.random.default_rng(2002).standard_normal((len(x), model.latent_dim)).astype(np.float32)
    difference = float(np.max(np.abs(np.asarray(model.log_likelihood(x, z)) - np.asarray(mapped.log_likelihood(x, z)))))
    mean_difference = float(np.max(np.abs(np.asarray(model.decode_mean(z)) - np.asarray(mapped.decode_mean(z)))))
    # Float32 heads and very small observation variance amplify absolute log-density rounding.
    source_ll = np.asarray(model.log_likelihood(x, z), dtype=float)
    mapped_ll = np.asarray(mapped.log_likelihood(x, z), dtype=float)
    if not np.allclose(source_ll, mapped_ll, rtol=3e-5, atol=0.01):
        raise RuntimeError("Mapped density identity failed")
    if not np.allclose(model.decode_mean(z), mapped.decode_mean(z), rtol=3e-5, atol=2e-5):
        raise RuntimeError("Mapped sampling-mean identity failed")
    return {"target_condition": target_condition, "max_absolute_log_density_difference": difference,
            "max_absolute_sample_mean_difference": mean_difference,
            "relative_log_density_tolerance": 3e-5, "absolute_log_density_tolerance": 0.01}


def train(run_root, dataset, condition, seed, gpu_uuid):
    root = Path(run_root).resolve()
    manifest = verify_run(root)
    identifier = job_id(dataset, condition, seed)
    if verified_completion(root, identifier):
        print(f"VERIFIED REUSE {identifier}", flush=True)
        return
    job_root = root / "jobs" / identifier
    job_root.mkdir(exist_ok=True)
    lock = (job_root / ".writer.lock").open("a+")
    directory = job_root / ("attempt-" + uuid.uuid4().hex)
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        if verified_completion(root, identifier):
            return
        directory.mkdir()
        runtime = gpu_runtime(gpu_uuid)
        import tensorflow as tf
        from model import GaussianVAE
        spec = manifest["specification"]
        tf.keras.utils.set_random_seed(seed)
        data = load_prepared(root, dataset)
        model = GaussianVAE(input_dim=spec["window"], latent_dim=spec["latent_dim"],
                            condition=condition, base_variance=float(data["base_variance"]), seed=seed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=spec["learning_rate"])
        optimizer.build(model.trainable_variables)
        rng = np.random.default_rng(seed + 50000)
        train_x = tf.constant(data["train"].astype(np.float32))

        @tf.function
        def update(indices, epsilon):
            batch = tf.gather(train_x, indices)
            with tf.GradientTape() as tape:
                terms = model.loss_terms(batch, epsilon)
                loss = tf.reduce_mean(terms["negative_elbo"])
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(g is None for g in gradients):
                raise RuntimeError("Disconnected trainable variable")
            tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, "Nonfinite training gradient")
            clipped, norm = tf.clip_by_global_norm(gradients, spec["gradient_global_norm"])
            optimizer.apply_gradients(zip(clipped, model.trainable_variables))
            return loss, tf.reduce_mean(terms["kl"]), norm

        atomic_json(directory / "configuration.json", {
            "dataset": dataset, "condition": condition, "seed": seed, "runtime": runtime,
            "specification": spec, "alpha": model.alpha,
            "residual_variance": model.residual_variance,
            "observation_variance": model.residual_variance / model.b ** 2,
            "parameter_count": int(model.count_params()),
            "run_manifest_sha256": sha256(root / "run_manifest.json"),
        })
        best = np.inf
        best_step = 0
        stale = 0
        history = []
        started = time.monotonic()
        for step in range(1, spec["max_steps"] + 1):
            indices = rng.integers(0, len(data["train"]), size=spec["batch_size"], dtype=np.int32)
            epsilon = rng.standard_normal((spec["batch_size"], spec["latent_dim"])).astype(np.float32)
            loss, rate, norm = update(indices, epsilon)
            if step % spec["validation_every"] and step != spec["max_steps"]:
                continue
            validation = importance_evaluate(model, data["validation"], spec["validation_samples"],
                                             spec["validation_sampling_seed"])
            score = float(-validation["log_score"].mean())
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite validation score")
            improved = score < best
            if improved:
                best, best_step, stale = score, step, 0
                temporary = directory / "best.pending.weights.h5"
                model.save_weights(str(temporary))
                os.replace(temporary, directory / "best.weights.h5")
            else:
                stale += 1
            history.append({"step": step, "training_negative_elbo": float(loss.numpy()),
                            "training_kl": float(rate.numpy()), "gradient_norm": float(norm.numpy()),
                            "validation_negative_log_score": score, "best_step": best_step,
                            "elapsed_seconds": time.monotonic() - started})
            atomic_json(directory / "history.json", history)
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            checkpoint.write(str(directory / "last_checkpoint"))
            atomic_json(directory / "progress.json", {
                "step": step, "best_step": best_step, "best_validation_score": best,
                "elapsed_seconds": time.monotonic() - started,
                "exact_resume_supported": False,
            })
            print(f"{identifier} step={step} validation={score:.6f} best={best_step}", flush=True)
            if stale >= spec["patience"]:
                break
        model.save_weights(str(directory / "final.weights.h5"))
        model.load_weights(str(directory / "best.weights.h5"))
        selected_validation = importance_evaluate(model, data["validation"], spec["validation_samples"],
                                                 spec["validation_sampling_seed"])
        reloaded_score = float(-selected_validation["log_score"].mean())
        if not np.isclose(reloaded_score, best, rtol=1e-6, atol=1e-5):
            raise RuntimeError("Reloaded checkpoint does not reproduce selected validation score")
        summary = evaluate_job(model, data, spec, directory)
        summary.update(dataset=dataset, condition=condition, seed=seed, best_step=best_step,
                       steps_run=step, best_validation_score=best,
                       elapsed_seconds=time.monotonic() - started,
                       mapped_control=mapped_control(model, data))
        atomic_json(directory / "summary.json", summary)
        verify_run(root)
        completion = {"attempt": directory.name, "run_manifest_sha256": sha256(root / "run_manifest.json"),
                      "artifact_sha256": {name: sha256(directory / name) for name in JOB_ARTIFACTS}}
        atomic_json(job_root / "complete.json", completion)
        print(f"DONE {identifier}", flush=True)
    except BaseException as error:
        if directory.exists():
            atomic_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock.close()


def baseline_results(root, manifest):
    results = {}
    for name in manifest["datasets"]:
        data = load_prepared(root, name)
        records = {}
        covariances = {"ppca": data["ppca_covariance"], "diagonal_gaussian": data["diagonal_covariance"]}
        if "oracle_covariance" in data:
            covariances["oracle"] = data["oracle_covariance"]
        for label, covariance in covariances.items():
            mean = np.zeros(data["test"].shape[1]) if label == "oracle" else data["ppca_mean"]
            score = gaussian_log_prob(data["test"], mean, covariance)
            rng = np.random.default_rng(manifest["specification"]["prior_sampling_seed"])
            draws = mean + rng.standard_normal((manifest["specification"]["prior_samples"], len(mean))) @ np.linalg.cholesky(covariance).T
            energy = energy_score(draws, data["test"])
            records[label] = {"negative_log_score_nats_per_dimension": float(-score.mean() / len(mean)),
                              "energy_score": float(energy.mean())}
        results[name] = records
    return results


def summarize(run_root):
    root = Path(run_root).resolve()
    manifest = verify_run(root)
    summaries = []
    files = [root / "run_manifest.json", root / "results.json"]
    files.extend(root / "prepared" / name for name in manifest["prepared_sha256"])
    files.extend(root / "source" / name for name in CODE_FILES)
    for dataset in manifest["datasets"]:
        for seed in SEEDS:
            for condition in CONDITIONS:
                identifier = job_id(dataset, condition, seed)
                attempt = verified_completion(root, identifier)
                if attempt is None:
                    raise RuntimeError(f"Missing completed job: {identifier}")
                summaries.append(read_json(attempt / "summary.json"))
                files.append(root / "jobs" / identifier / "complete.json")
                files.extend(attempt / name for name in JOB_ARTIFACTS)
    output = {"schema_version": 2, "purpose": manifest["purpose"],
              "run_manifest_sha256": sha256(root / "run_manifest.json"),
              "specification": manifest["specification"], "results": summaries,
              "baselines": baseline_results(root, manifest),
              "dataset_metadata": read_json(root / "prepared" / "metadata.json")}
    atomic_json(root / "results.json", output)
    atomic_json(root / "archive_manifest.json", {
        "schema_version": 1,
        "scope": "Completed jobs, prepared inputs, source code, settings and results.",
        "artifact_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(files)},
    })
    print(f"ALL DONE: {len(summaries)} verified jobs; {root / 'results.json'}", flush=True)


def campaign(run_root, gpu_uuids):
    root = Path(run_root).resolve()
    manifest = verify_run(root)
    if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4:
        raise ValueError("Campaign requires four distinct GPU UUIDs")
    jobs = [(dataset, condition, seed) for dataset in manifest["datasets"]
            for seed in SEEDS for condition in CONDITIONS]
    np.random.default_rng(20260905).shuffle(jobs)
    pending = deque(jobs)
    guard = threading.Lock()
    errors = []
    stop = threading.Event()
    children = {}
    interrupted = False

    def request_stop(signum, frame):
        # A signal can interrupt cleanup while it holds a lock.
        nonlocal interrupted
        interrupted = True

    def cancel():
        stop.set()
        with guard:
            active = list(children.values())
        for child in active:
            if child.poll() is None:
                child.terminate()
        for child in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def worker(gpu):
        while True:
            with guard:
                if stop.is_set() or interrupted or not pending:
                    return
                dataset, condition, seed = pending.popleft()
            identifier = job_id(dataset, condition, seed)
            log = root / "logs" / f"{identifier}-{uuid.uuid4().hex}.log"
            try:
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TF_CPP_MIN_LOG_LEVEL="2",
                                   TF_DETERMINISTIC_OPS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                                   PYTHONDONTWRITEBYTECODE="1")
                command = [sys.executable, str(Path(__file__).resolve()), "train", "--run-root", str(root),
                           "--dataset", dataset, "--condition", condition, "--seed", str(seed), "--gpu-uuid", gpu]
                print(f"START {identifier} {gpu}", flush=True)
                with log.open("x") as stream:
                    with guard:
                        if stop.is_set() or interrupted:
                            return
                        child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                        children[identifier] = child
                    returncode = child.wait()
                with guard:
                    children.pop(identifier, None)
                    if returncode:
                        errors.append({"job": identifier, "returncode": returncode, "log": log.name})
                        stop.set()
                    atomic_json(root / "campaign_status.json", {
                        "pending": len(pending), "errors": list(errors),
                        "completed": sum((root / "jobs" / job_id(*job) / "complete.json").exists() for job in jobs),
                    })
                print(f"EXIT {identifier} {returncode}", flush=True)
            except Exception as error:
                stop.set()
                with guard:
                    errors.append({"job": identifier, "type": type(error).__name__, "message": str(error)})
                raise

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                futures = {executor.submit(worker, gpu) for gpu in gpu_uuids}
                while futures and not interrupted:
                    done, futures = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    if stop.is_set():
                        break
            except BaseException:
                stop.set()
                raise
            finally:
                # Reap peers before the executor joins their blocked waits.
                if stop.is_set() or interrupted:
                    cancel()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if errors or stop.is_set() or interrupted:
        raise RuntimeError(f"Campaign stopped before completion: {errors}")
    summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run-root", required=True)
    prep.add_argument("--purpose", choices=("core", "smoke"), default="core")
    fit = commands.add_parser("train")
    fit.add_argument("--run-root", required=True)
    fit.add_argument("--dataset", choices=DATASETS, required=True)
    fit.add_argument("--condition", choices=CONDITIONS, required=True)
    fit.add_argument("--seed", type=int, choices=SEEDS, required=True)
    fit.add_argument("--gpu-uuid", required=True)
    execute = commands.add_parser("campaign")
    execute.add_argument("--run-root", required=True)
    execute.add_argument("--gpu-uuids", nargs=4, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_root, args.purpose)
    elif args.command == "train":
        train(args.run_root, args.dataset, args.condition, args.seed, args.gpu_uuid)
    elif args.command == "campaign":
        campaign(args.run_root, args.gpu_uuids)
    else:
        summarize(args.run_root)


if __name__ == "__main__":
    main()
