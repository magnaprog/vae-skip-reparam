"""CPU-only synthetic closures, not trained models or scientific results.

Reuse the frozen builder's fixture rather than inventing another core schema.
All checkpoint bytes are explicit placeholders and are never loaded. Component
NPZs are finite algebraic fixtures, not evidence of executing a neural network.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_prigm_paper as legacy
import build_prigm_v2_paper as paper
import prigm_refine_evaluation as refinement
from test_prigm_paper import campaign, read_json, reseal, write_json
from test_prigm_dart_paper_evidence import metrics, public_fixture as dart_public_fixture


SHA = legacy.experiment.sha256
PUBLIC_SOURCE_IDS = tuple(dict.fromkeys((
    *paper.SOURCE_IDS,
    *(reviewed for archived, reviewed in paper.NUMERICAL_REVIEWED_SOURCES.items()
      if archived != paper.NUMERICAL_DRIVER),
    *("src/" + name for name in paper.EVALUATOR_SOURCES),
)))


def test_synthetic_closures_work_in_checkout_without_private_files(tmp_path):
    checkout = tmp_path / "public-only-checkout"
    names = tuple(dict.fromkeys((*PUBLIC_SOURCE_IDS, *paper.DART_EXTENSION_SOURCE_IDS,
                                 "tests/test_prigm_paper.py", "tests/test_prigm_v2_paper.py",
                                 "tests/test_prigm_dart_paper_evidence.py")))
    for name in names:
        assert Path(name).parts[0] in ("src", "tests")
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert {path.relative_to(checkout).as_posix() for path in checkout.rglob("*") if path.is_file()} == set(names)
    assert not (checkout / "runs-v2").exists()
    assert not (checkout / "datasets").exists()
    # Explicit node IDs exclude this subprocess test, so it cannot recurse.
    selected = (
        "test_numerical_fixture_checks_four_full_trajectories_without_neural_execution",
        "test_numerical_self_declared_source_or_data_cannot_be_resealed",
        "test_numerical_reviewed_originals_are_rechecked",
        "test_components_verify_all_finite_arrays_and_nonzero_archived_replay",
        "test_manifest_is_pure_and_input_digest_excludes_generated_payloads",
        "test_combined_diagnostic_protocol_and_helper_bytes_are_checked",
        "test_combined_public_manifest_rejects_corruption[source_binding]",
        "test_combined_public_manifest_rejects_corruption[dart_protocol_pin]",
        "test_dart_train_and_test_distributions_preserve_all_windows_and_population_moments",
        "test_dart_training_distribution_rejects_missing_malformed_or_misaligned_values",
        "test_dart_self_consistent_training_distribution_must_match_training_moments",
        "test_combined_training_distribution_must_match_independently_valid_support",
        "test_figures_use_test_all_draws_fit_ranges_and_physical_labels",
        "test_full_synthetic_build_reuses_frozen_builder_and_publishes_exactly_14_manifest_last",
    )
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(checkout / "src"),
                       CUDA_VISIBLE_DEVICES="", TMPDIR=str(tmp_path))
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--confcutdir", str(checkout), "--basetemp", str(tmp_path / "child-pytest"),
         *(str(checkout / "tests/test_prigm_v2_paper.py") + "::" + name for name in selected)],
        cwd=checkout, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def component_arrays(data, arm, seed):
    """Independent finite Gaussian identities, retaining all per-draw values."""
    n, d = data["test"].shape
    variance = float(data["base_variance"]) * (25 if arm in ("B", "D") else 1)
    rng = np.random.default_rng(seed)
    mse = 0.1 + rng.uniform(0, 0.02, (3, n, 256))
    ll = -0.5 * d * (mse / variance + np.log(2 * np.pi * variance))
    sample_kl = rng.normal(0.3, 0.2, (3, n, 256))
    logw = ll - sample_kl
    latent = np.broadcast_to(np.arange(1, 17) / 1000, (3, n, 16)).copy()
    kl = latent.sum(axis=2)
    nll = -ll.mean(axis=2)
    iw = [paper.importance_statistics(lw) for lw in logw]
    arrays = {
        "log_score": np.stack([s["log_score"] for s in iw]),
        "ess": np.stack([s["ess"] for s in iw]), "log_weights": logw,
        "sample_log_likelihood": ll, "sample_logq_minus_logprior": sample_kl,
        "posterior_expected_nll": nll, "posterior_expected_mse": mse.mean(axis=2),
        "posterior_mean_mse": np.broadcast_to(np.square(data["test"]).mean(axis=1) + 1e-5, (3, n)).copy(),
        "kl": kl, "kl_by_latent": latent, "negative_elbo_analytic_kl": nll + kl,
        "sample_negative_elbo": -logw.mean(axis=2),
        "jensen_gap": np.stack([s["log_score"] for s in iw]) - logw.mean(axis=2),
    }
    return arrays, variance


def make_components(destination, core, evidence):
    destination.mkdir()
    launcher = destination / "src"
    launcher.mkdir()
    sources = {name: SHA(ROOT / "src" / name) for name in paper.EVALUATOR_SOURCES}
    for name in sources:
        shutil.copyfile(ROOT / "src" / name, launcher / name)
    for index, worker_name in enumerate(paper.WORKERS):
        worker = destination / worker_name
        (worker / "source").mkdir(parents=True)
        for name in sources:
            shutil.copyfile(launcher / name, worker / "source" / name)
        owned = paper.canonical_jobs()[index::4]
        protocol = {
            "schema_version": 1, "purpose": "frozen_checkpoint_components",
            "core_manifest_sha256": evidence["run_manifest_sha256"], "evaluator_source_sha256": sources,
            "worker_index": index, "worker_count": 4, "jobs": owned, "all_jobs": paper.canonical_jobs(),
            "samples": 256, "stream_seeds": list(paper.STREAM_SEEDS), "density_coordinates": "residual",
            "probability_dtype": "float64", "network_dtype": "float32", "chunk": 32,
            "runtime": {"fixture_only": True, "username": "PRIVATE-USER", "path": "/home/private/fixture"},
        }
        write_json(worker / "protocol.json", protocol)
        artifacts = {}
        for dataset, arm, seed, _ in owned:
            identifier = legacy.experiment.job_id(dataset, arm, seed)
            original = evidence["jobs"][identifier]
            completion_path = core / "jobs" / identifier / "complete.json"
            attempt = completion_path.parent / read_json(completion_path)["attempt"]
            data = legacy._load_npz(core / "prepared" / (dataset + ".npz"))
            saved = legacy._load_npz(attempt / "evaluation.npz")
            arrays, variance = component_arrays(data, arm, seed)
            stem = identifier + "-selected"
            np.savez_compressed(worker / (stem + ".npz"), **arrays)
            differences = {key: float(np.max(np.abs(arrays[a][0] - saved[b]))) for key, a, b in zip(
                paper.REPLAY_KEYS, ("log_score", "ess", "posterior_mean_mse"), ("log_score", "ess", "corrected_mse"))}
            record = {
                "identifier": identifier, "dataset": dataset, "arm": arm, "seed": seed, "checkpoint": "selected",
                "checkpoint_sha256": original["artifact_sha256"]["best.weights.h5"], "core_completion_sha256": SHA(completion_path),
                "prepared_sha256": evidence["prepared_sha256"][dataset + ".npz"],
                "selected_step": original["best_step"], "steps_run": original["steps_run"],
                "observation_variance": variance, "test_windows": len(data["test"]), "dimension": 192,
                "array_shapes": {k: list(v.shape) for k, v in arrays.items()}, "array_sha256": SHA(worker / (stem + ".npz")),
                "frozen_evaluator_reproduced_exactly": True, "core_evaluation_sha256": SHA(attempt / "evaluation.npz"),
                "archived_selected_replay_differences": differences,
            }
            write_json(worker / (stem + ".json"), record)
            for suffix in (".npz", ".json"):
                artifacts[stem + suffix] = SHA(worker / (stem + suffix))
        write_json(worker / "complete.json", {
            "schema_version": 1, "purpose": protocol["purpose"], "protocol_sha256": SHA(worker / "protocol.json"),
            "core_manifest_sha256": evidence["run_manifest_sha256"], "evaluator_source_sha256": sources,
            "artifact_sha256": artifacts,
        })


def make_refinement(core, output):
    """Real closure verifier over synthetic finite arrays, no GPU worker call."""
    refinement.prepare(core, output)
    manifest = refinement.verify_refinement(core, output)
    for job in refinement.canonical_jobs():
        identifier = legacy.experiment.job_id(*job)
        attempt = legacy.experiment.verified_completion(core, identifier)
        data = legacy.experiment.load_prepared(core, job[0])
        arrays = refinement.original_arrays(attempt, data)
        for budget in refinement.SAMPLE_COUNTS:
            arrays[f"log_score_{budget}"] = arrays["log_score_256"] + 0.25
            arrays[f"ess_{budget}"] = arrays["ess_256"].copy()
        output_attempt = output / "jobs" / identifier / ("attempt-" + "a" * 32)
        output_attempt.mkdir(parents=True)
        np.savez_compressed(output_attempt / "evaluation.npz", **arrays)
        runtime = {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000", "pid": 1,
                   "placement": "/device:GPU:0", "deterministic_ops_requested": True,
                   "elapsed_seconds": 0.1, "synthetic_fixture_not_execution": True}
        write_json(output_attempt / "summary.json", refinement.job_summary(output, manifest, identifier, arrays, runtime))
        write_json(output_attempt.parent / "complete.json", {
            "schema_version": 1, "core_job_id": identifier, "attempt": output_attempt.name,
            "refinement_manifest_sha256": SHA(output / refinement.MANIFEST),
            "artifact_sha256": {name: SHA(output_attempt / name) for name in refinement.ARTIFACTS},
        })
    refinement.summarize(core, output)


def fixture_numerical_states(step, scale=1.0):
    """Full production dimensions, compressible zeros; NOT neural execution."""
    shapes = [(192, 128), (128,), (128, 64), (64,), (64, 16), (16,), (64, 16), (16,),
              (16, 64), (64,), (64, 128), (128,), (128, 192), (192,)]
    values = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    values += [np.array(step, dtype=np.int64)]
    values += [np.zeros(shape, dtype=np.float32) for shape in shapes[:12] for _ in range(2)]
    values += [np.array(step, dtype=np.int64)]
    values += [np.zeros(shape, dtype=np.float32) for shape in shapes[12:] for _ in range(2)]
    values += [np.array(value, dtype=np.float32) for value in (0.001, 0.001, scale)]
    values += [np.array(scale < 1)]
    assert len(values) == 48
    return {f"{condition}_state_{index:02d}": value.copy()
            for condition in ("base", "canonical") for index, value in enumerate(values)}


def reseal_numerical(directory):
    complete = read_json(directory / "complete.json")
    complete["protocol_sha256"] = SHA(directory / "protocol.json")
    complete["artifact_sha256"] = {name: SHA(directory / name) for name in complete["artifact_sha256"]}
    write_json(directory / "complete.json", complete)


@pytest.fixture(scope="module")
def reviewed_checkout_master(tmp_path_factory):
    """Real public bytes, but no private driver or diagnostic approval receipt."""
    checkout = tmp_path_factory.mktemp("synthetic-reviewed-checkout")
    for name in PUBLIC_SOURCE_IDS:
        assert Path(name).parts[0] == "src"
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    driver = checkout / paper.NUMERICAL_REVIEWED_SOURCES[paper.NUMERICAL_DRIVER]
    driver.parent.mkdir(parents=True, exist_ok=True)
    driver.write_bytes(
        b'"""Synthetic numerical driver identity, NOT execution or scientific evidence."""\n'
        b'raise RuntimeError("Synthetic driver bytes must never be executed")\n'
    )
    protocol = checkout / paper.DART_PROTOCOL_FILE
    protocol.parent.mkdir(parents=True, exist_ok=True)
    write_json(protocol, {
        "schema_version": 1, "purpose": "synthetic_dart_diagnostics_not_execution",
        "synthetic_fixture_not_execution": True,
        "source_sha256": {name: SHA(checkout / name) for name in paper.SOURCE_IDS},
    })
    return checkout


@pytest.fixture(scope="module")
def numerical_master(tmp_path_factory, reviewed_checkout_master):
    """Synthetic receipts and header-only HDF5 placeholders, never GPU evidence."""
    root = tmp_path_factory.mktemp("synthetic-numerical-not-execution")
    (root / "src").mkdir()
    (root / "prepared").mkdir()
    for archived, reviewed in paper.NUMERICAL_REVIEWED_SOURCES.items():
        shutil.copyfile(reviewed_checkout_master / reviewed, root / archived)
    np.savez_compressed(root / "prepared/gaussian_v2.npz", train=np.zeros((8192, 192), dtype=np.float64))
    sources = {name: SHA(root / name) for name in paper.NUMERICAL_REVIEWED_SOURCES}
    prepared = SHA(root / "prepared/gaussian_v2.npz")
    for mode in ("smoke", "full"):
        steps = 64 if mode == "smoke" else 10000
        snapshots = [0, 64] if mode == "smoke" else list(range(0, 10001, 1000))
        for label, variance in (("low", 0.0625), ("high", 1.5625)):
            directory = root / (mode + "-" + label)
            directory.mkdir()
            protocol = {
                "schema_version": 1, "purpose": "canonical_pair_numerical_validation",
                "mode": mode, "conditions": ["base", "canonical"], "seed": 42,
                "observation_variance": variance, "specification": deepcopy(paper.NUMERICAL_SPEC),
                "planned_updates_per_trajectory": steps, "full_training_trajectories_in_attempt": 2 if mode == "full" else 0,
                "short_validation_trajectories_in_attempt": 2 if mode == "smoke" else 0,
                "absolute_state_tolerance": 0.0, "relative_state_tolerance": 0.0,
                "comparison": paper.NUMERICAL_COMPARISON, "source_sha256": sources, "prepared_sha256": prepared,
                "started_utc": "2026-09-07T00:00:00+00:00", "launch_cutoff_utc": "2026-09-07T01:00:00+00:00" if mode == "full" else None,
                "maximum_elapsed_seconds": 1800, "snapshot_steps": snapshots,
                "restart_only": True, "scientific_effect_estimation": False,
            }
            write_json(directory / "protocol.json", protocol)
            write_json(directory / "runtime.json", {
                "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000", "pid": 123,
                "placement": "/job:localhost/replica:0/task:0/device:GPU:0",
                "process_inventory": ["123, GPU-00000000-0000-0000-0000-000000000000, /PRIVATE/fixture-not-execution/python, 264 MiB"],
                "python": "PRIVATE synthetic fixture, not execution", "numpy": "1.26.4", "tensorflow": "2.15.0",
                "deterministic_ops_requested": True, "tf32_enabled": True,
            })
            write_json(directory / "comparator_self_test.json", dict(paper.NUMERICAL_SELF_TEST))
            with (directory / "trace.jsonl").open("w") as stream:
                for step in range(1, steps + 1):
                    stream.write(json.dumps({"step": step, "pre_update_loss": 1.0, "pre_update_kl": 0.0,
                                             "gradient_norm": 20.0 if step % 2 else 5.0,
                                             "clipping_scale": 0.5 if step % 2 else 1.0,
                                             "clipping_active": bool(step % 2), "all_state_equal": True,
                                             "checked_state_tensors_per_model": 48}, separators=(",", ":")) + "\n")
            for step in snapshots:
                np.savez_compressed(directory / f"state-{step:05d}.npz", **fixture_numerical_states(step))
            for condition in ("base", "canonical"):
                (directory / (condition + ".weights.h5")).write_bytes(b"\x89HDF\r\n\x1a\nfixture header only, NOT Keras weights")
            artifacts = {path.name: SHA(path) for path in directory.iterdir()}
            write_json(directory / "complete.json", {
                "schema_version": 1, "purpose": protocol["purpose"], "mode": mode, "trajectories": 2,
                "verified_updates_per_trajectory": steps, "exact_state_equality": True, "elapsed_seconds": 100.0,
                "protocol_sha256": SHA(directory / "protocol.json"), "artifact_sha256": artifacts,
            })
    return root


@pytest.fixture
def numerical(numerical_master, reviewed_checkout_master, tmp_path, monkeypatch):
    root = tmp_path / "synthetic-numerical"
    shutil.copytree(numerical_master, root)
    checkout = tmp_path / "synthetic-checkout"
    shutil.copytree(reviewed_checkout_master, checkout)
    # Bind identities, not verifiers. Reviewed originals stay separate from the
    # archived copies, so resealing corrupt source bytes still fails closed.
    monkeypatch.setattr(paper, "ROOT", checkout)
    monkeypatch.setattr(paper, "DART_PROTOCOL_SHA256", SHA(checkout / paper.DART_PROTOCOL_FILE))
    monkeypatch.setattr(paper, "NUMERICAL_PREPARED_SHA256", SHA(numerical_master / "prepared/gaussian_v2.npz"))
    return root


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    base = tmp_path_factory.mktemp("synthetic-combined")
    with pytest.MonkeyPatch.context() as patch:
        core, _ = campaign.__wrapped__(base, patch)
        # Nonzero known training support, unequal split counts and train-only
        # ECDF breakpoints. No test-fit transform or neural execution.
        metadata = read_json(core / "prepared/metadata.json")
        for dataset in paper.DATASETS:
            path = core / "prepared" / (dataset + ".npz")
            data = legacy._load_npz(path)
            t = np.arange(192) / 96
            data["train"] = np.concatenate((data["train"], data["train"][:1] + 0.5))
            data["train"] += np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)
            origin = 1609459200
            data["starts_train"] = origin + np.array([0, 900, 86400, 86400 + 900], dtype=np.int64)
            data["starts_test"] = origin + np.array([40 * 86400, 40 * 86400 + 900, 71 * 86400], dtype=np.int64)
            np.savez_compressed(path, **data)
            metadata[dataset]["windows"]["train"] = len(data["train"])
            metadata[dataset]["counts"]["train"]["windows"] = len(data["train"])
        write_json(core / "prepared/metadata.json", metadata)
        manifest = read_json(core / "run_manifest.json")
        manifest["prepared_sha256"] = {p.name: SHA(p) for p in (core / "prepared").iterdir()}
        write_json(core / "run_manifest.json", manifest)
        pin = SHA(core / "run_manifest.json")
        summaries = {}
        for completion_path in (core / "jobs").glob("*/complete.json"):
            completion = read_json(completion_path)
            completion["run_manifest_sha256"] = pin
            config_path = completion_path.parent / completion["attempt"] / "configuration.json"
            config = read_json(config_path)
            config["run_manifest_sha256"] = pin
            write_json(config_path, config)
            write_json(completion_path, completion)
            # Synthetic draws, not neural execution. Unlike the legacy fixture,
            # these energies must really reduce the entire stored ensemble.
            attempt = config_path.parent
            data = legacy._load_npz(core / "prepared" / (config["dataset"] + ".npz"))
            arrays = legacy._load_npz(attempt / "evaluation.npz")
            rng = np.random.default_rng(config["seed"] + paper.ARMS.index(config["arm"]))
            # Repeated four-coordinate patterns keep per-test archive copies
            # small, while retaining 512 distinct finite full-width draws.
            arrays["prior_samples"] = np.tile(rng.normal(size=(512, 4)), (1, 48)) + 0.4 * paper.ARMS.index(config["arm"])
            arrays["energy"] = paper.energy_score(arrays["prior_samples"], data["test"])
            np.savez_compressed(attempt / "evaluation.npz", **arrays)
            summary = read_json(attempt / "summary.json")
            summary["energy_score"] = float(arrays["energy"].mean())
            summary["energy_score_meters"] = float(arrays["energy"].mean() * data["scale"])
            write_json(attempt / "summary.json", summary)
            summaries[completion_path.parent.name] = summary
        results = read_json(core / "results.json")
        results["results"] = [summaries[legacy.experiment.job_id(job["dataset"], job["arm"], job["seed"])]
                              for job in results["results"]]
        results["run_manifest_sha256"] = pin
        results["dataset_metadata"] = metadata
        write_json(core / "results.json", results)
        reseal(core)
        evidence = legacy.verify_evidence(core, pin)
        patch.setattr(refinement, "CORE_MANIFEST_SHA256", pin)
        make_refinement(core, base / "refinement")
        make_components(base / "components", core, evidence)
        decision = base / "decision.json"
        write_json(decision, {"schema_version": 1, "new_scientific_training_authorized_for_launch": False,
                              "private_note": "/home/private/stop-new-matrix"})
    return base, evidence


@pytest.fixture
def bundle(master, tmp_path, monkeypatch, numerical):
    base, evidence = master
    for name in ("run", "refinement", "components"):
        shutil.copytree(base / name, tmp_path / name)
    shutil.copyfile(base / "decision.json", tmp_path / "decision.json")
    monkeypatch.setattr(legacy, "ROOT", base / "public")
    monkeypatch.setattr(paper, "FROZEN_EVALUATOR_SHA256", {
        name: SHA(base / "components/worker-0/source" / name) for name in paper.EVALUATOR_SOURCES})
    monkeypatch.setattr(refinement, "CORE_MANIFEST_SHA256", evidence["run_manifest_sha256"])
    monkeypatch.setattr(paper, "DART_CORE_SHA256", evidence["run_manifest_sha256"])
    return tmp_path, deepcopy(evidence)


def verify(bundle):
    base, core = bundle
    return paper.verify_components(base / "components", base / "run", core)


def build(bundle):
    base, core = bundle
    return paper.build_paper(base / "run", base / "paper", core["run_manifest_sha256"],
                             base / "refinement", base / "components", base / "decision.json", base / "synthetic-numerical")


def reseal_worker(worker):
    complete = read_json(worker / "complete.json")
    complete["protocol_sha256"] = SHA(worker / "protocol.json")
    for name in complete["artifact_sha256"]:
        if name.endswith(".npz"):
            record_path = worker / (Path(name).stem + ".json")
            record = read_json(record_path)
            record["array_sha256"] = SHA(worker / name)
            write_json(record_path, record)
    complete["artifact_sha256"] = {name: SHA(worker / name) for name in complete["artifact_sha256"]}
    write_json(worker / "complete.json", complete)


def public_fixture(bundle, dart_master):
    base, core = bundle
    components, _ = verify(bundle)
    # Public-schema mutations share one derived synthetic diagnostic fixture;
    # private derivation and full builds are exercised independently below.
    support, diagnostics, _ = deepcopy(dart_master)
    index = paper._input_index(base / "run", base / "refinement", base / "components", base / "decision.json", base / "synthetic-numerical")
    public_core = {key: value for key, value in core.items() if key not in ("contexts", "prior")}
    public_core.update(builder_sha256=SHA(paper.ROOT / "src/build_prigm_paper.py"),
                       summary_statistics={"paired": legacy.paired_statistics(core["jobs"]), "job_count": 60})
    return {"schema_version": 2, "purpose": "combined", "profile": "corrected_audit", "input_index": index,
            "combined_input_sha256": paper.canonical_sha256(index), "builder_sha256": SHA(paper.ROOT / "src/build_prigm_v2_paper.py"),
            "core": public_core, "components": components, "dart_support": support, "dart_diagnostics": diagnostics,
            "numerical_validation": paper.verify_numerical_validation(base / "synthetic-numerical"),
            "refinement": legacy.verify_refinement_evidence(base / "run", base / "refinement"),
            "output_sha256": dict.fromkeys(paper.AUDIT_OUTPUTS, "a" * 64), "methods": dict(paper.METHODS)}


def test_dart_numerical_policy_preserves_only_reviewed_producer(numerical, monkeypatch):
    reviewed = paper._numerical_inputs(numerical)["source_sha256"]
    monkeypatch.setattr(paper, "FROZEN_NUMERICAL_SOURCE_SHA256", dict(reviewed))
    current = paper.ROOT / "src/prigm_v2_metrics.py"
    current.write_bytes(current.read_bytes() + b"\n# unrelated extension\n")
    with pytest.raises(ValueError, match="Unreviewed numerical source identity"):
        paper.verify_numerical_validation(numerical)
    public = paper.verify_numerical_validation(numerical, profile="dart_extension")
    assert public["source_sha256"] == reviewed
    assert public["full_training_trajectories"] == 4
    monkeypatch.setitem(paper.FROZEN_NUMERICAL_SOURCE_SHA256, "src/prigm_v2_metrics.py", "0" * 64)
    with pytest.raises(ValueError, match="Unreviewed numerical source identity"):
        paper.verify_numerical_validation(numerical, profile="dart_extension")


def test_dart_numerical_snapshot_requires_archive_not_current_driver(numerical, monkeypatch):
    reviewed = paper._numerical_inputs(numerical)["source_sha256"]
    monkeypatch.setattr(paper, "FROZEN_NUMERICAL_SOURCE_SHA256", dict(reviewed))
    current = paper.ROOT / paper.NUMERICAL_REVIEWED_SOURCES[paper.NUMERICAL_DRIVER]
    current.unlink()
    with pytest.raises(ValueError, match="Expected a regular artifact"):
        paper._numerical_snapshot(numerical)
    snapshot = paper._numerical_snapshot(numerical, profile="dart_extension")
    assert current not in snapshot
    assert numerical / paper.NUMERICAL_DRIVER in snapshot
    public = paper.verify_numerical_validation(numerical, profile="dart_extension")
    assert public["source_sha256"] == reviewed
    assert public["full_training_trajectories"] == 4


def test_numerical_source_policy_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown numerical source policy"):
        paper.verify_numerical_validation(tmp_path / "absent", profile="unchecked")


def test_numerical_fixture_checks_four_full_trajectories_without_neural_execution(numerical, monkeypatch):
    public = paper.verify_numerical_validation(numerical)
    assert public["full_training_trajectories"] == 4
    assert set(public["pairs"]) == {"full-low", "full-high"}
    assert len(public["state_layout"]) == 48
    for index, array in enumerate(fixture_numerical_states(0).values()):
        assert public["state_layout"][index % 48] == {"shape": list(array.shape), "dtype": str(array.dtype)}
    for name, pair in public["pairs"].items():
        assert pair["trace_rows"] == pair["verified_updates_per_trajectory"] == 10000
        assert pair["npz_snapshot_count"] == 11
        assert pair["clipping"] == {"active_updates": 5000, "minimum_scale": 0.5, "maximum_scale": 1.0,
                                    "minimum_gradient_norm": 5.0, "maximum_gradient_norm": 20.0}
        assert pair["protocol_sha256"] == SHA(numerical / name / "protocol.json")
        assert pair["completion_sha256"] == SHA(numerical / name / "complete.json")
        for artifact, digest in pair["artifact_sha256"].items():
            assert digest == SHA(numerical / name / artifact)
    text = json.dumps(public)
    for private in ("GPU-", "PRIVATE", "started_utc", "launch_cutoff_utc", '"pid"', "process_inventory", "placement"):
        assert private not in text
    assert "Keras loading and HDF5 contents are not verified" in public["verification_scope"]
    index = paper._numerical_inputs(numerical)
    def forbidden(*args, **kwargs):
        pytest.fail("Public numerical validator read host/files")
    for method in ("open", "read_bytes", "read_text", "stat", "iterdir"):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(os, "uname", forbidden)
    assert paper.validate_numerical_validation(public, index) is public


@pytest.mark.parametrize("mutation", ["missing_full", "missing_smoke", "failure", "extra", "pending", "missing_snapshot",
                                      "missing_self_test", "extra_source", "extra_prepared", "symlink_root", "symlink_source",
                                      "symlink_snapshot", "unsafe_artifact", "missing_artifact_hash", "artifact_hash", "protocol_hash"])
def test_numerical_inventory_and_hash_closure(numerical, mutation):
    directory = numerical / "full-low"
    if mutation in ("missing_full", "missing_smoke"):
        shutil.rmtree(numerical / ("full-high" if mutation == "missing_full" else "smoke-high"))
    elif mutation in ("failure", "extra", "pending"):
        (directory / {"failure": "failure.json", "extra": "extra.json", "pending": "state-10000.npz.pending.npz"}[mutation]).write_text("{}")
    elif mutation in ("missing_snapshot", "missing_self_test"):
        (directory / ("state-09000.npz" if mutation == "missing_snapshot" else "comparator_self_test.json")).unlink()
    elif mutation in ("extra_source", "extra_prepared"):
        (numerical / ("src/unbound.py" if mutation == "extra_source" else "prepared/extra.npz")).write_text("extra")
    elif mutation.startswith("symlink"):
        path = {"symlink_root": numerical, "symlink_source": numerical / "src/prigm_v2_model.py",
                "symlink_snapshot": directory / "state-10000.npz"}[mutation]
        renamed = numerical.parent / (path.name + "-original")
        path.rename(renamed)
        path.symlink_to(renamed, target_is_directory=renamed.is_dir())
    else:
        complete = read_json(directory / "complete.json")
        if mutation == "unsafe_artifact":
            complete["artifact_sha256"]["../outside.npz"] = "a" * 64
        elif mutation == "missing_artifact_hash":
            complete["artifact_sha256"].pop("runtime.json")
        elif mutation == "artifact_hash":
            complete["artifact_sha256"]["base.weights.h5"] = "a" * 64
        else:
            complete["protocol_sha256"] = "a" * 64
        write_json(directory / "complete.json", complete)
    with pytest.raises(ValueError):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("mutation", ["steps", "mode", "conditions", "seed", "variance", "storage", "tolerance", "relative",
                                      "effect", "restart", "cutoff_missing", "cutoff_late", "cutoff_naive", "start_offset",
                                      "limit", "snapshots", "source", "prepared", "completion_steps", "completion_equal", "elapsed"])
def test_numerical_protocol_and_completion_rejected_after_resealing(numerical, mutation):
    directory = numerical / "full-low"
    protocol = read_json(directory / "protocol.json")
    changes = {"steps": ("planned_updates_per_trajectory", 64), "mode": ("mode", "smoke"),
               "conditions": ("conditions", ["base", "raw"]), "seed": ("seed", 42.0), "variance": ("observation_variance", 1.5625),
               "tolerance": ("absolute_state_tolerance", 1e-8), "relative": ("relative_state_tolerance", 1e-8),
               "effect": ("scientific_effect_estimation", True), "restart": ("restart_only", False),
               "cutoff_missing": ("launch_cutoff_utc", None), "cutoff_late": ("launch_cutoff_utc", "2026-09-06T00:00:00Z"),
               "cutoff_naive": ("launch_cutoff_utc", "2026-09-07T01:00:00"),
               "start_offset": ("started_utc", "2026-09-07T00:00:00+01:00"), "limit": ("maximum_elapsed_seconds", 3600),
               "snapshots": ("snapshot_steps", [0, 10000]), "prepared": ("prepared_sha256", "b" * 64)}
    if mutation in changes:
        key, value = changes[mutation]
        protocol[key] = value
    elif mutation == "storage":
        protocol["specification"]["model_storage_contract"] = 2
    elif mutation == "source":
        protocol["source_sha256"]["src/prigm_v2_model.py"] = "a" * 64
    else:
        complete = read_json(directory / "complete.json")
        key, value = {"completion_steps": ("verified_updates_per_trajectory", 9999),
                      "completion_equal": ("exact_state_equality", 1), "elapsed": ("elapsed_seconds", 1800.01)}[mutation]
        complete[key] = value
        write_json(directory / "complete.json", complete)
    write_json(directory / "protocol.json", protocol)
    reseal_numerical(directory)
    with pytest.raises(ValueError):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("profile", ["corrected_audit", "dart_extension"])
@pytest.mark.parametrize("name", [*paper.NUMERICAL_REVIEWED_SOURCES, "prepared/gaussian_v2.npz"])
def test_numerical_self_declared_source_or_data_cannot_be_resealed(numerical, name, profile, monkeypatch):
    monkeypatch.setattr(paper, "FROZEN_NUMERICAL_SOURCE_SHA256",
                        paper._numerical_inputs(numerical)["source_sha256"])
    path = numerical / name
    path.write_bytes(path.read_bytes() + b"\n# changed fixture bytes\n")
    for directory in (numerical / pair for pair in ("smoke-low", "smoke-high", *paper.NUMERICAL_PAIRS)):
        protocol = read_json(directory / "protocol.json")
        if name.startswith("prepared/"):
            protocol["prepared_sha256"] = SHA(path)
        else:
            protocol["source_sha256"][name] = SHA(path)
        write_json(directory / "protocol.json", protocol)
        reseal_numerical(directory)
    with pytest.raises(ValueError, match="source identity|prepared identity"):
        paper.verify_numerical_validation(numerical, profile=profile)


@pytest.mark.parametrize("mutation", ["truncate", "extra", "nonsequential", "no_newline", "blank", "duplicate", "nan", "overflow",
                                      "false_equal", "int_equal", "47_states", "bool_states", "clip_scale", "clip_flag", "negative_norm"])
def test_numerical_all_trace_rows_validated(numerical, mutation):
    directory = numerical / "full-low"
    path = directory / "trace.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    if mutation == "truncate":
        lines.pop()
    elif mutation == "extra":
        lines.append(lines[-1])
    elif mutation == "no_newline":
        lines[-1] = lines[-1].rstrip("\n")
    elif mutation == "blank":
        lines[9998] = "\n"
    elif mutation == "duplicate":
        lines[9998] = lines[9998].replace('"step":9999', '"step":9999,"step":9999')
    elif mutation in ("nan", "overflow"):
        lines[9998] = lines[9998].replace('"pre_update_loss":1.0', '"pre_update_loss":' + ("NaN" if mutation == "nan" else "1e999"))
    else:
        row = json.loads(lines[9998])
        key, value = {"nonsequential": ("step", 9998), "false_equal": ("all_state_equal", False),
                      "int_equal": ("all_state_equal", 1), "47_states": ("checked_state_tensors_per_model", 47),
                      "bool_states": ("checked_state_tensors_per_model", True), "clip_scale": ("clipping_scale", 0.75),
                      "clip_flag": ("clipping_active", False), "negative_norm": ("gradient_norm", -20.0)}[mutation]
        row[key] = value
        lines[9998] = json.dumps(row) + "\n"
    path.write_text("".join(lines))
    reseal_numerical(directory)
    with pytest.raises(ValueError):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "infinity", "missing", "extra", "difference", "body_counter", "head_counter",
                                      "body_rate", "head_rate", "clipping", "active_dtype", "active", "moment_shape", "initial_moment"])
def test_numerical_snapshots_validate_exact_48_state_contract(numerical, mutation):
    directory = numerical / "full-low"
    path = directory / ("state-00000.npz" if mutation == "initial_moment" else "state-10000.npz")
    arrays = legacy._load_npz(path)
    for condition in ("base", "canonical"):
        key = lambda index: f"{condition}_state_{index:02d}"
        if mutation == "shape":
            arrays[key(0)] = arrays[key(0)].T
        elif mutation == "dtype":
            arrays[key(0)] = arrays[key(0)].astype(np.float64)
        elif mutation in ("nan", "infinity"):
            arrays[key(0)].flat[0] = np.nan if mutation == "nan" else np.inf
        elif mutation == "missing":
            arrays.pop(key(1))
        elif mutation == "extra":
            arrays[key(48)] = np.array(0)
        elif mutation == "difference":
            arrays["canonical_state_00"].flat[0] = np.nextafter(np.float32(0), np.float32(1))
        elif mutation in ("body_counter", "head_counter"):
            arrays[key(14 if mutation == "body_counter" else 39)] = np.array(9999, dtype=np.int64)
        elif mutation in ("body_rate", "head_rate"):
            arrays[key(44 if mutation == "body_rate" else 45)] = np.array(0.002, dtype=np.float32)
        elif mutation == "clipping":
            arrays[key(46)] = np.array(0.5, dtype=np.float32)
        elif mutation == "active_dtype":
            arrays[key(47)] = np.array(0, dtype=np.int64)
        elif mutation == "active":
            arrays[key(47)] = np.array(True)
        elif mutation == "moment_shape":
            arrays[key(40)] = np.zeros((192, 128), dtype=np.float32)
        else:
            arrays[key(15)].flat[0] = 1.0
    np.savez_compressed(path, **arrays)
    reseal_numerical(directory)
    with pytest.raises(ValueError):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("mutation", ["mismatch", "restoration", "jit", "pid", "uuid", "placement", "determinism", "version", "tf32", "hdf5",
                                      "duplicate_json", "nonfinite_json", "unknown_runtime"])
def test_numerical_comparator_runtime_and_hdf5_guards(numerical, mutation):
    directory = numerical / "full-low"
    if mutation in ("mismatch", "restoration", "jit"):
        path = directory / "comparator_self_test.json"
        data = read_json(path)
        key = {"mismatch": "deliberate_output_bias_mismatch_rejected", "restoration": "original_state_restored_and_verified",
               "jit": "comparison_jit_compile"}[mutation]
        data[key] = not data[key]
        write_json(path, data)
    elif mutation == "hdf5":
        (directory / "canonical.weights.h5").write_bytes(b"not an HDF5 file")
    elif mutation == "duplicate_json":
        path = directory / "complete.json"
        path.write_text(path.read_text().replace('"trajectories": 2', '"trajectories": 2, "trajectories": 2'))
        with pytest.raises(ValueError, match="Duplicate"):
            paper.verify_numerical_validation(numerical)
        return
    elif mutation == "nonfinite_json":
        path = directory / "complete.json"
        path.write_text(path.read_text().replace('"elapsed_seconds": 100.0', '"elapsed_seconds": 1e999'))
        with pytest.raises(ValueError, match="Nonfinite"):
            paper.verify_numerical_validation(numerical)
        return
    else:
        path = directory / "runtime.json"
        data = read_json(path)
        key, value = {"pid": ("pid", 124), "uuid": ("gpu_uuid", "GPU-11111111-1111-1111-1111-111111111111"),
                      "placement": ("placement", "/device:CPU:0"), "determinism": ("deterministic_ops_requested", False),
                      "version": ("tensorflow", "2.16.0"), "tf32": ("tf32_enabled", 1),
                      "unknown_runtime": ("unbound", True)}[mutation]
        data[key] = value
        write_json(path, data)
    reseal_numerical(directory)
    with pytest.raises(ValueError):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("mutation", ["trajectory_count", "pair_missing", "pair_extra", "trace_rows", "state_layout", "clip_count", "clip_reduction",
                                      "source", "prepared", "protocol_pin", "index_pin", "mismatch", "restoration", "scope", "private", "spec_alias"])
def test_public_numerical_consistency(numerical, mutation):
    public = paper.verify_numerical_validation(numerical)
    index = paper._numerical_inputs(numerical)
    pair = public["pairs"]["full-low"]
    if mutation == "trajectory_count":
        public["full_training_trajectories"] = 2
    elif mutation == "pair_missing":
        public["pairs"].pop("full-high")
    elif mutation == "pair_extra":
        public["pairs"]["smoke-low"] = deepcopy(pair)
    elif mutation == "trace_rows":
        pair["trace_rows"] = 9999
    elif mutation == "state_layout":
        public["state_layout"][0]["shape"] = [1]
    elif mutation == "clip_count":
        pair["clipping"]["active_updates"] = 10000
    elif mutation == "clip_reduction":
        pair["clipping"]["minimum_scale"] = 0.25
    elif mutation == "source":
        pair["protocol"]["source_sha256"] = {**public["source_sha256"], "src/prigm_v2_model.py": "0" * 64}
    elif mutation == "prepared":
        public["prepared_sha256"] = "0" * 64
    elif mutation == "protocol_pin":
        pair["protocol_sha256"] = "0" * 64
    elif mutation == "index_pin":
        index["pairs"]["full-low"]["completion_sha256"] = "0" * 64
    elif mutation in ("mismatch", "restoration"):
        key = "deliberate_output_bias_mismatch_rejected" if mutation == "mismatch" else "original_state_restored_and_verified"
        pair["comparator_self_test"][key] = False
    elif mutation == "scope":
        public["verification_scope"] = "Keras weights loaded on CPU"
    elif mutation == "spec_alias":
        pair["protocol"]["specification"]["model_storage_contract"] = 2
        assert paper.NUMERICAL_SPEC["model_storage_contract"] == 3
    else:
        pair["gpu_attestation"]["pid"] = 123
    with pytest.raises(ValueError):
        paper.validate_numerical_validation(public, index)


def test_numerical_public_sources_are_consistency_not_a_claim_of_host_byte_checks(numerical):
    public = paper.verify_numerical_validation(numerical)
    public["source_sha256"]["src/prigm_v2_model.py"] = "a" * 64
    for pair in public["pairs"].values():
        pair["protocol"]["source_sha256"] = dict(public["source_sha256"])
    paper.validate_numerical_validation(public)
    # The corresponding actual-byte forgery is rejected by the private test above.


def test_numerical_change_during_verification_is_rejected(numerical, monkeypatch):
    original = paper._numerical_states
    def changed(path, *args):
        original(path, *args)
        if path.parent.name == "full-high" and path.name == "state-10000.npz":
            target = numerical / "full-low/base.weights.h5"
            target.write_bytes(target.read_bytes() + b"changed after hash check")
            reseal_numerical(target.parent)
    monkeypatch.setattr(paper, "_numerical_states", changed)
    with pytest.raises(ValueError, match="changed during verification"):
        paper.verify_numerical_validation(numerical)


@pytest.mark.parametrize("name", ["src/prigm_v2_model.py", paper.NUMERICAL_DRIVER])
def test_numerical_reviewed_originals_are_rechecked(numerical, monkeypatch, name):
    checkout = numerical.parent / "synthetic-reviewed-checkout"
    for archived, reviewed in paper.NUMERICAL_REVIEWED_SOURCES.items():
        target = checkout / reviewed
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(numerical / archived, target)
    monkeypatch.setattr(paper, "ROOT", checkout)
    original = paper._numerical_states
    def changed(path, *args):
        original(path, *args)
        if path.parent.name == "full-high" and path.name == "state-10000.npz":
            (checkout / paper.NUMERICAL_REVIEWED_SOURCES[name]).write_text("changed reviewed source during verification")
    monkeypatch.setattr(paper, "_numerical_states", changed)
    with pytest.raises(ValueError, match="changed during verification"):
        paper.verify_numerical_validation(numerical)


def test_numerical_unsafe_root_is_rejected(numerical):
    with pytest.raises(ValueError, match="Unsafe numerical root"):
        paper.verify_numerical_validation(numerical / "full-low/..")


def test_components_verify_all_finite_arrays_and_nonzero_archived_replay(bundle):
    components, arrays = verify(bundle)
    assert len(components["jobs"]) == len(arrays) == 60
    assert len(components["workers"]) == 4
    assert sum(len(read_json(bundle[0] / "components" / w / "complete.json")["artifact_sha256"])
               for w in paper.WORKERS) == 120
    for identifier, record in components["jobs"].items():
        assert record["frozen_evaluator_reproduced_exactly"] is True
        assert record["archived_selected_replay_differences"]["posterior_mean_mse_max_abs"] == pytest.approx(1e-5)
        for i, stream in enumerate(record["streams"]):
            a = arrays[identifier]
            assert stream["posterior_expected_nll_per_coordinate"] == pytest.approx(-a["sample_log_likelihood"][i].mean() / 192)
            assert stream["negative_iw_per_coordinate"] == pytest.approx(-a["log_score"][i].mean() / 192)
            assert stream["mean_ess"] == pytest.approx(a["ess"][i].mean())
    text = json.dumps(components)
    assert "PRIVATE" not in text and "/home/" not in text and "runtime" not in text


@pytest.mark.parametrize("mutation", ["stream_seed", "stream_count", "duplicate_job", "job_order", "slice", "dtype", "coordinates", "bool_count"])
def test_protocol_changes_rejected_after_resealing(bundle, mutation):
    worker = bundle[0] / "components/worker-0"
    protocol = read_json(worker / "protocol.json")
    if mutation == "stream_seed":
        protocol["stream_seeds"][1] = 42002
    elif mutation == "stream_count":
        protocol["stream_seeds"].pop()
    elif mutation == "duplicate_job":
        protocol["all_jobs"][-1] = protocol["all_jobs"][0]
    elif mutation == "job_order":
        protocol["all_jobs"].reverse()
    elif mutation == "slice":
        protocol["jobs"][-1] = protocol["all_jobs"][1]
    elif mutation == "dtype":
        protocol["network_dtype"] = "float64"
    elif mutation == "coordinates":
        protocol["density_coordinates"] = "observation"
    else:
        protocol["worker_index"] = False
    write_json(worker / "protocol.json", protocol)
    reseal_worker(worker)
    with pytest.raises(ValueError, match="protocol differs"):
        verify(bundle)


@pytest.mark.parametrize("mutation", ["missing_worker", "extra_worker", "extra_output", "missing_output", "unsafe_artifact", "mismatched_source", "source_hash", "completion_hash", "symlink"])
def test_component_inventory_source_and_path_closure(bundle, mutation):
    root = bundle[0] / "components"
    worker = root / "worker-0"
    if mutation == "missing_worker":
        shutil.rmtree(root / "worker-3")
    elif mutation == "extra_worker":
        (root / "worker-4").mkdir()
    elif mutation == "extra_output":
        (worker / "unbound.json").write_text("{}")
    elif mutation == "missing_output":
        (worker / "gaussian_A_42-selected.npz").unlink()
    elif mutation == "unsafe_artifact":
        complete = read_json(worker / "complete.json")
        complete["artifact_sha256"]["../outside.npz"] = "a" * 64
        write_json(worker / "complete.json", complete)
    elif mutation == "mismatched_source":
        (worker / "source/prigm_v2_metrics.py").write_text("# altered source\n")
        protocol = read_json(worker / "protocol.json")
        protocol["evaluator_source_sha256"]["prigm_v2_metrics.py"] = SHA(worker / "source/prigm_v2_metrics.py")
        write_json(worker / "protocol.json", protocol)
        reseal_worker(worker)
    elif mutation == "source_hash":
        (root / "src/eval_gaussian_final_checkpoints.py").write_text("# changed launcher\n")
    elif mutation == "completion_hash":
        complete = read_json(worker / "complete.json")
        complete["protocol_sha256"] = "0" * 64
        write_json(worker / "complete.json", complete)
    else:
        path = worker / "source/prigm_v2_metrics.py"
        path.unlink()
        path.symlink_to(ROOT / "src/prigm_v2_metrics.py")
    with pytest.raises(ValueError):
        verify(bundle)


@pytest.mark.parametrize("key", paper.ARRAY_KEYS)
def test_every_array_reduction_or_shape_checked_after_resealing(bundle, key):
    worker = bundle[0] / "components/worker-0"
    path = worker / "gaussian_A_42-selected.npz"
    arrays = legacy._load_npz(path)
    # Also catches the non-stochastic posterior MSE via shared-stream equality.
    arrays[key].flat[0] += 0.5
    np.savez_compressed(path, **arrays)
    reseal_worker(worker)
    with pytest.raises(ValueError, match="Recomputed|Shared analytic"):
        verify(bundle)


@pytest.mark.parametrize("mutation", ["nonfinite", "float32", "missing", "extra", "truncated", "negative_kl", "mse_units"])
def test_component_array_schema_dtype_finiteness_and_units(bundle, mutation):
    worker = bundle[0] / "components/worker-0"
    path = worker / "gaussian_A_42-selected.npz"
    arrays = legacy._load_npz(path)
    if mutation == "nonfinite":
        arrays["log_score"][0, 0] = np.inf
    elif mutation == "float32":
        arrays["log_score"] = arrays["log_score"].astype(np.float32)
    elif mutation == "missing":
        arrays.pop("log_weights")
    elif mutation == "extra":
        arrays["unused"] = np.zeros((3, 3))
    elif mutation == "truncated":
        arrays["log_weights"] = arrays["log_weights"][:, :, :-1]
    elif mutation == "negative_kl":
        arrays["kl_by_latent"][:] = -0.1
    else:
        arrays["posterior_expected_mse"] *= 4
    np.savez_compressed(path, **arrays)
    reseal_worker(worker)
    with pytest.raises(ValueError):
        verify(bundle)


@pytest.mark.parametrize("field,value", [
    ("selected_step", 500), ("steps_run", 750), ("observation_variance", 2.0),
    ("test_windows", 4), ("dimension", 16), ("seed", 1729), ("checkpoint", "final"),
    ("checkpoint_sha256", "0" * 64), ("core_completion_sha256", "0" * 64),
    ("prepared_sha256", "0" * 64), ("core_evaluation_sha256", "0" * 64),
    ("frozen_evaluator_reproduced_exactly", 1),
])
def test_job_receipt_cross_references_core(bundle, field, value):
    worker = bundle[0] / "components/worker-0"
    path = worker / "gaussian_A_42-selected.json"
    record = read_json(path)
    record[field] = value
    write_json(path, record)
    reseal_worker(worker)
    with pytest.raises(ValueError):
        verify(bundle)


def test_archived_replay_claim_is_recomputed_not_trusted(bundle):
    worker = bundle[0] / "components/worker-0"
    path = worker / "gaussian_A_42-selected.json"
    record = read_json(path)
    record["archived_selected_replay_differences"]["log_score_max_abs"] = 0
    write_json(path, record)
    reseal_worker(worker)
    with pytest.raises(ValueError, match="log_score_max_abs"):
        verify(bundle)


@pytest.mark.parametrize("target", ["prepared/gaussian.npz", "jobs/gaussian_A_42/complete.json", "jobs/gaussian_A_42/attempt-PRIVATE-UUID/best.weights.h5"])
def test_core_inputs_cannot_change_under_verified_evidence(bundle, target):
    path = bundle[0] / "run" / target
    if path.suffix == ".json":
        completion = read_json(path)
        completion["artifact_sha256"]["best.weights.h5"] = "0" * 64
        write_json(path, completion)
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        verify(bundle)


@pytest.mark.parametrize("mutation", ["protocol", "helper"])
def test_combined_diagnostic_protocol_and_helper_bytes_are_checked(bundle, mutation):
    base, _ = bundle
    arguments = (base / "run", base / "refinement", base / "components", base / "decision.json",
                 base / "synthetic-numerical")
    paper._input_index(*arguments)
    target = paper.ROOT / (paper.DART_PROTOCOL_FILE if mutation == "protocol" else "src/prigm_v2_metrics.py")
    target.write_bytes(target.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="Approved DART (protocol|helper source) changed"):
        paper._input_index(*arguments)


def test_manifest_is_pure_and_input_digest_excludes_generated_payloads(bundle, dart_master, monkeypatch):
    public = public_fixture(bundle, dart_master)
    pin = bundle[1]["run_manifest_sha256"]
    def forbidden(*args, **kwargs):
        pytest.fail("Pure validator tried to read a file")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    assert paper.validate_combined_manifest(public, pin) is public
    digest = public["combined_input_sha256"]
    public["output_sha256"][paper.AUDIT_OUTPUTS[0]] = "b" * 64
    assert paper.validate_combined_manifest(public)["combined_input_sha256"] == digest
    assert digest == paper.canonical_sha256(dict(reversed(list(public["input_index"].items()))))
    public["input_index"]["decision_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="input digest"):
        paper.validate_combined_manifest(public)


@pytest.mark.parametrize("mutation", ["schema", "bool_schema", "profile", "purpose", "count", "array_count", "extra_output", "missing_output", "input_extra", "input_source", "input_digest", "private_key", "private_value", "units", "power", "stream_units", "float_seed", "refinement_duplicate", "refinement_count", "stream_shared", "source_binding", "source_id", "dart_settings", "dart_protocol_pin", "dart_missing", "malformed_job"])
def test_combined_public_manifest_rejects_corruption(bundle, dart_master, mutation):
    public = public_fixture(bundle, dart_master)
    if mutation in ("schema", "bool_schema", "profile", "purpose"):
        key, value = {"schema": ("schema_version", 1), "bool_schema": ("schema_version", True),
                      "profile": ("profile", "future_scientific_matrix"), "purpose": ("purpose", "core")}[mutation]
        public[key] = value
    elif mutation == "count":
        public["components"]["job_count"] = 60.0
    elif mutation == "array_count":
        public["components"]["jobs"]["gaussian_A_42"]["test_windows"] = True
    elif mutation == "extra_output":
        public["output_sha256"]["figures/prior_samples.png"] = "a" * 64
    elif mutation == "missing_output":
        public["output_sha256"].pop(paper.AUDIT_OUTPUTS[0])
    elif mutation == "input_extra":
        public["input_index"]["generated_assets"] = {"main.pdf": "a" * 64}
    elif mutation == "input_source":
        public["input_index"]["source_sha256"]["../private.py"] = "a" * 64
    elif mutation == "input_digest":
        public["input_index"]["decision_sha256"] = "0" * 64
    elif mutation == "private_key":
        public["core"]["runtime"] = {"pid": 1}
    elif mutation == "private_value":
        public["core"]["jobs"]["gaussian_A_42"]["note"] = "/home/private/run"
    elif mutation in ("units", "power"):
        support = public["dart_support"][paper.DATASETS[1]]
        if mutation == "units":
            support["power_units"] = "normalized_squared"
        else:
            support["bin_power_m2"][1] *= 4
    elif mutation == "stream_units":
        public["components"]["jobs"]["gaussian_A_42"]["streams"][0]["posterior_expected_mse"] *= 4
    elif mutation == "float_seed":
        public["components"]["jobs"]["gaussian_A_42"]["seed"] = 42.0
    elif mutation == "refinement_duplicate":
        public["refinement"]["results"]["jobs"][-1] = public["refinement"]["results"]["jobs"][0]
    elif mutation == "refinement_count":
        public["refinement"]["results"]["jobs"][0]["count"] = 3.0
    elif mutation == "stream_shared":
        public["components"]["jobs"]["gaussian_A_42"]["streams"][1]["posterior_mean_mse"] += 0.1
    elif mutation == "source_binding":
        public["input_index"]["evaluator_source_sha256"]["prigm_v2_metrics.py"] = "f" * 64
        public["combined_input_sha256"] = paper.canonical_sha256(public["input_index"])
    elif mutation == "source_id":
        public["input_index"]["source_sha256"]["src/unexpected.py"] = "f" * 64
        public["combined_input_sha256"] = paper.canonical_sha256(public["input_index"])
    elif mutation in ("dart_settings", "dart_protocol_pin"):
        diagnostic_index = public["input_index"]["dart_diagnostics"]
        if mutation == "dart_settings":
            diagnostic_index["settings"]["baseline_sampling_seed"] += 1
        else:
            diagnostic_index["approved_protocol_sha256"] = "0" * 64
        public["combined_input_sha256"] = paper.canonical_sha256(public["input_index"])
    elif mutation == "dart_missing":
        public.pop("dart_diagnostics")
    else:
        public["core"]["jobs"]["gaussian_A_42"].pop("dataset")
    with pytest.raises(ValueError):
        paper.validate_combined_manifest(public)


@pytest.mark.parametrize("mutation", [None, "dependency", "cache"])
def test_launcher_binds_optional_frozen_metrics_and_rejects_cache(bundle, mutation):
    base, core = bundle
    launcher = base / "components/src"
    shutil.copyfile(ROOT / "src/prigm_metrics.py", launcher / "prigm_metrics.py")
    if mutation == "dependency":
        (launcher / "prigm_metrics.py").write_text("changed dependency")
    elif mutation == "cache":
        (launcher / "__pycache__").mkdir()
    arguments = (base / "components", paper.FROZEN_EVALUATOR_SHA256,
                 core["code_sha256"]["src/prigm_metrics.py"])
    if mutation:
        with pytest.raises(ValueError):
            paper._launcher_sources(*arguments)
    else:
        paper._launcher_sources(*arguments)


def test_frozen_evaluator_identity_is_separate_from_current_derivation_code(bundle, dart_master):
    public = public_fixture(bundle, dart_master)
    public["input_index"]["source_sha256"]["src/prigm_v2_metrics.py"] = "a" * 64
    public["combined_input_sha256"] = paper.canonical_sha256(public["input_index"])
    assert (public["input_index"]["evaluator_source_sha256"]["prigm_v2_metrics.py"]
            != public["input_index"]["source_sha256"]["src/prigm_v2_metrics.py"])
    paper.validate_combined_manifest(public)


def test_components_table_uses_arithmetic_means_not_medians(bundle):
    components, _ = verify(bundle)
    support, diagnostics, _ = paper.dart_diagnostics(bundle[0] / "run", bundle[1])
    for seed, value in zip(paper.SEEDS, (1, 2, 90)):
        for stream in components["jobs"][f"gaussian_A_{seed}"]["streams"]:
            stream["posterior_mean_mse"] = value
    fragments = paper.table_fragments(components, support, diagnostics, "c" * 64)
    rows = fragments["generated/component_rows.tex"].splitlines()
    assert len(rows) == 20 and all(len(row.split(" & ")) == 8 for row in rows)
    assert rows[0].split(" & ")[2] == "31"
    assert all(row.endswith("\\\\") for row in rows[:-1]) and not rows[-1].endswith("\\\\")
    assert len(fragments["generated/dart_diagnostic_rows.tex"].splitlines()) == 4
    assert r"\newcommand{\PrigmCombinedSHA}{" + "c" * 64 + "}" in fragments["generated/combined_binding.tex"]


@pytest.fixture(scope="module")
def dart_master(master):
    base, evidence = master
    return paper.dart_diagnostics(base / "run", evidence)


def test_dart_unequal_block_masses_reconstruct_not_unweighted_block_average():
    origin = 1609459200
    starts = origin + np.array([0, 900, 14 * 86400, 28 * 86400, 29 * 86400])
    blocks = paper._dart_blocks(starts, origin)
    assert blocks["14"]["counts"] == [2, 1, 2]
    assert blocks["28"]["counts"] == [3, 2]
    values = np.array([[1., 3., 50., 5., 7.], [2., 4., 100., 6., 8.]])
    summary = paper._dart_score_summary(values, values / 2, blocks)
    paper._validate_dart_scores(summary, 2, 5, blocks)
    for days in ("14", "28"):
        block = summary["blocks"][days]["joint_m"]
        conditional = np.asarray(block["conditional_mean"])
        np.testing.assert_allclose(block["reconstructed_global_mean"], values.mean(axis=1), rtol=1e-15)
        np.testing.assert_allclose(block["weighted_contribution"], conditional * blocks[days]["mass"])
        assert not np.allclose(conditional.mean(axis=1), values.mean(axis=1))
        broken = deepcopy(summary)
        broken["blocks"][days]["joint_m"]["reconstructed_global_mean"] = conditional.mean(axis=1).tolist()
        with pytest.raises(ValueError, match="float64"):
            paper._validate_dart_scores(broken, 2, 5, blocks)
    # The window starting at the boundary goes into the new block, untrimmed.
    assert blocks["14"]["window_block_ids"] == [0, 0, 1, 2, 2]


def test_dart_retains_all_seeds_draws_windows_and_single_meter_scaling(master, dart_master):
    base, evidence = master
    _, diagnostics, private = dart_master
    assert list(diagnostics["records"]) == list(paper.DATASETS[1:])
    for dataset, record in diagnostics["records"].items():
        data = legacy._load_npz(base / "run/prepared" / (dataset + ".npz"))
        scale, n = float(data["scale"]), len(data["test"])
        for arm in paper.ARMS:
            fit = record["neural"][arm]
            assert fit["optimization_seeds"] == list(paper.SEEDS)
            assert np.shape(fit["scores"]["per_window"]["joint_m"]) == (3, n)
            for i, seed in enumerate(paper.SEEDS):
                identifier = legacy.experiment.job_id(dataset, arm, seed)
                attempt, _ = paper._core_attempt(base / "run", identifier, evidence)
                saved = legacy._load_npz(attempt / "evaluation.npz")
                expected_joint = paper.energy_score(saved["prior_samples"], data["test"])
                expected_increment = paper.increment_energy_score(saved["prior_samples"], data["test"])
                np.testing.assert_array_equal(fit["saved_joint_standardized"][i], expected_joint)
                np.testing.assert_allclose(fit["scores"]["per_window"]["joint_m"][i], expected_joint * scale, rtol=1e-15)
                np.testing.assert_allclose(fit["scores"]["per_window"]["increment_m"][i], expected_increment * scale, rtol=1e-15)
                magnitude = np.linalg.norm(paper.increment_features(saved["prior_samples"]), axis=1) * scale
                np.testing.assert_allclose(fit["distribution"][i]["noisy"]["increment_magnitude_m"], magnitude)
                assert len(magnitude) == 512
                for kind in ("noisy", "decoder_mean"):
                    plot = private[dataset]["fits"][arm][i][kind]
                    assert plot["window_count"] == 512
                    np.testing.assert_array_equal(plot["increment_magnitude_m"], fit["distribution"][i][kind]["increment_magnitude_m"])
        for arm in paper.ARMS[1:]:
            paired = record["paired"][arm + "-A"]
            for key in paper.DART_SCORES:
                expected = (np.asarray(record["neural"][arm]["scores"]["per_window"][key])
                            - record["neural"]["A"]["scores"]["per_window"][key])
                np.testing.assert_array_equal(paired["scores"]["per_window"][key], expected)
                for days in ("14", "28"):
                    left, right = (np.asarray(record["neural"][a]["scores"]["blocks"][days][key]["conditional_mean"])
                                   for a in (arm, "A"))
                    np.testing.assert_allclose(paired["scores"]["blocks"][days][key]["conditional_mean"], left - right, atol=1e-14)
        assert record["blocks"]["14"]["counts"] == [2, 1]
        assert record["blocks"]["28"]["counts"] == [2, 1]
        assert sum(record["phase_counts_utc"]["train"]) == len(data["train"])
        assert sum(record["phase_counts_utc"]["test"]) == n
        np.testing.assert_array_equal(record["global_window_weights"], np.full(n, 1 / n))


def test_dart_train_and_test_distributions_preserve_all_windows_and_population_moments(master, dart_master):
    support, diagnostics, private = dart_master
    for dataset, record in diagnostics["records"].items():
        data = legacy._load_npz(master[0] / "run/prepared" / (dataset + ".npz"))
        scale = float(data["scale"])
        assert len(data["train"]) == 4 and len(data["test"]) == 3
        for split in ("train", "test"):
            windows = data[split]
            means = windows.mean(axis=1) * scale
            sd = windows.std(axis=1) * scale
            increments = np.concatenate([(windows[:, lag:] - windows[:, :-lag]) / np.sqrt(192 - lag)
                                         for lag in (1, 4, 16, 48, 96)], axis=1)
            centered = windows - windows.mean(axis=1, keepdims=True)
            power = np.abs(np.fft.rfft(centered, axis=1))**2 / 192**2
            power[:, 1:-1] *= 2
            within, between = np.square(sd).mean(), means.var()
            expected = {
                "frequency_cycles_per_day": np.arange(97) / 2,
                "bin_power_m2": power.mean(axis=0) * scale**2,
                "window_mean_relative_training_center_m": means,
                "within_window_sd_m": sd,
                "increment_magnitude_m": np.linalg.norm(increments, axis=1) * scale,
                "within_variance_m2": within, "between_variance_m2": between,
                "pooled_variance_m2": (windows * scale).var(),
                "within_to_pooled_ratio": within / (within + between),
            }
            public = record[split + "_distribution"]
            plot = private[dataset][split]
            assert public["window_count"] == plot["window_count"] == len(windows)
            assert set(public) == set(plot) == {*expected, "window_count"}
            for field, values in expected.items():
                np.testing.assert_allclose(public[field], values, rtol=1e-14, atol=1e-14)
                np.testing.assert_array_equal(plot[field], public[field])
        training = record["train_distribution"]
        mean = np.asarray(record["training_mean_standardized"])
        covariance = np.asarray(record["training_covariance_standardized2"])
        assert np.mean(training["window_mean_relative_training_center_m"]) == pytest.approx(mean.mean() * scale)
        assert training["between_variance_m2"] == pytest.approx(covariance.mean() * scale**2)
        assert training["pooled_variance_m2"] == pytest.approx((np.trace(covariance) / 192 + mean.var()) * scale**2)
        assert mean.var() > 0  # The coordinate-mean contribution must not disappear.
        np.testing.assert_array_equal(training["bin_power_m2"][1:], support[dataset]["bin_power_m2"])
        for field in ("within_variance_m2", "between_variance_m2", "pooled_variance_m2", "within_to_pooled_ratio"):
            assert training[field] == support[dataset][field]


def test_dart_reference_population_covariances_floors_and_shared_normal(master, dart_master):
    base, _ = master
    _, diagnostics, _ = dart_master
    normal = np.random.default_rng(45001).standard_normal((512, 192))
    for dataset, record in diagnostics["records"].items():
        data = legacy._load_npz(base / "run/prepared" / (dataset + ".npz"))
        train = data["train"]
        mean = train.mean(axis=0)
        covariance = (train - mean).T @ (train - mean) / len(train)
        np.testing.assert_allclose(record["training_mean_standardized"], mean, atol=1e-15)
        np.testing.assert_allclose(record["training_covariance_standardized2"], covariance, atol=1e-15)
        # Use the actual stored float64 moment arithmetic for deterministic bytes.
        mean, covariance = paper.weighted_moments(train, np.ones(len(train)))
        assert not np.allclose(covariance, np.cov(train, rowvar=False, ddof=1))
        for name, (model, multiplier) in paper.DART_REFERENCES.items():
            reference = record["references"][name]
            variance = float(data["base_variance"]) * multiplier
            if model == "ppca":
                fitted = paper.fixed_noise_ppca(mean, covariance, variance, rank=16)
                expected = fitted["covariance"]
                assert reference["effective_rank"] == fitted["effective_rank"] <= 16
                np.testing.assert_allclose(np.linalg.eigvalsh(expected)[:176], variance, atol=1e-13)
            else:
                expected = np.diag(np.maximum(np.diag(covariance), variance))
                np.testing.assert_array_equal(paper.noise_matched_diagonal(train, np.ones(len(train)), variance)["covariance"], expected)
                assert np.all(np.diag(expected) >= variance)
                if multiplier == 25:
                    assert np.any(np.diag(covariance) < variance)  # active-floor synthetic case
            samples = mean + normal @ np.linalg.cholesky(expected).T
            assert reference["covariance_sha256"] == paper._dart_array_digest(expected)
            assert reference["prior_samples_sha256"] == paper._dart_array_digest(samples)
            assert reference["sampling_seed"] == 45001 and reference["integration_streams"] == 1
            assert np.shape(reference["scores"]["per_window"]["joint_m"]) == (1, len(data["test"]))
            np.testing.assert_allclose(reference["scores"]["per_window"]["joint_m"][0],
                                       paper.energy_score(samples, data["test"]) * data["scale"])
        # Floors can also be inactive at both settings. Identity here is exact,
        # not an allclose decision based on a rounded score table.
        low = paper.noise_matched_diagonal(train, np.ones(len(train)), 0.001)["covariance"]
        high = paper.noise_matched_diagonal(train, np.ones(len(train)), 0.025)["covariance"]
        np.testing.assert_array_equal(low, high)


def test_increment_features_widths_level_invariance_and_norm_meter_scaling():
    rng = np.random.default_rng(42)
    x, draws = rng.normal(size=(3, 192)), rng.normal(size=(512, 192))
    widths = [192 - lag for lag in (1, 4, 16, 48, 96)]
    assert widths == paper.DART_PROTOCOL["increment_block_widths"]
    assert sum(widths) == paper.DART_PROTOCOL["increment_dimension"] == 795
    features = paper.increment_features(x)
    assert features.shape == (3, 795)
    offset = 0
    for lag, width in zip((1, 4, 16, 48, 96), widths):
        np.testing.assert_array_equal(features[:, offset:offset + width], (x[:, lag:] - x[:, :-lag]) / np.sqrt(width))
        offset += width
    shifted = x + np.array([10., 20., 30.])[:, None]
    np.testing.assert_allclose(paper.increment_features(shifted), features, atol=1e-14)
    np.testing.assert_allclose(paper.increment_energy_score(draws, x), paper.increment_energy_score(draws, shifted), atol=1e-14)
    assert not np.allclose(paper.energy_score(draws, x), paper.energy_score(draws, shifted))
    first, scaled = (paper.physical_window_statistics(x, scale) for scale in (1., 3.))
    np.testing.assert_allclose(scaled["increment_magnitude_m"], np.linalg.norm(features, axis=1) * 3)
    np.testing.assert_allclose(scaled["bin_power_m2"], first["bin_power_m2"] * 9)
    np.testing.assert_allclose(paper.energy_score(draws * 3, x * 3), paper.energy_score(draws, x) * 3)
    np.testing.assert_allclose(paper.increment_energy_score(draws * 3, x * 3), paper.increment_energy_score(draws, x) * 3)


def test_dart_reference_validation_allows_arithmetic_roundoff_not_changed_covariance(master, dart_master, monkeypatch):
    public = deepcopy(dart_master[1])
    original = paper.fixed_noise_ppca
    def rounded(*args, **kwargs):
        fitted = original(*args, **kwargs)
        fitted["covariance"] = np.nextafter(fitted["covariance"], np.inf)
        return fitted
    monkeypatch.setattr(paper, "fixed_noise_ppca", rounded)
    paper.validate_dart_diagnostics(public, master[1])
    reference = public["records"][paper.DATASETS[1]]["references"]["ppca_low"]
    # Draw SHA is a producer identity, not public replay of BLAS-specific bytes.
    reference["prior_samples_sha256"] = "a" * 64
    paper.validate_dart_diagnostics(public, master[1])
    covariance = np.asarray(reference["covariance_standardized2"])
    covariance[0, 0] += 1e-8
    reference["covariance_standardized2"] = covariance.tolist()
    reference["covariance_sha256"] = paper._dart_array_digest(covariance)
    with pytest.raises(ValueError, match="covariance or floor"):
        paper.validate_dart_diagnostics(public, master[1])


def test_dart_schema_is_pure(master, dart_master, monkeypatch):
    _, public, _ = dart_master
    def forbidden(*args, **kwargs):
        pytest.fail("DART public validator read files or host identity")
    for method in ("open", "read_bytes", "read_text", "stat", "iterdir"):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(os, "uname", forbidden)
    assert paper.validate_dart_diagnostics(public, master[1]) is public


@pytest.mark.parametrize("mutation", ["units", "lags", "widths", "draw_count", "block_days", "unknown", "missing_record",
    "scale", "moment_shape", "covariance", "starts", "phase", "origin", "weights", "block_ids", "membership", "count", "mass", "sparse",
    "seed", "seed_order", "missing_arm", "joint_shape", "increment_shape", "joint_scale", "last_fit_window", "mean", "fit_mean",
    "conditional", "contribution", "reconstruction", "range", "paired", "paired_seed", "reference_seed", "reference_streams",
    "reference_variance", "reference_rank", "reference_covariance", "reference_samples", "reference_shape", "plot_shape", "plot_count", "plot_power", "plot_negative"])
def test_dart_strict_schema_rejects_corrupt_arrays_units_weights_and_reductions(master, dart_master, mutation):
    public = deepcopy(dart_master[1])
    dataset = paper.DATASETS[1]
    record = public["records"][dataset]
    fit = record["neural"]["A"]
    scores = fit["scores"]
    block = scores["blocks"]["14"]["joint_m"]
    reference = record["references"]["ppca_low"]
    if mutation in ("units", "lags", "widths", "draw_count", "block_days"):
        key, value = {"units": ("energy_units", "standardized"), "lags": ("increment_lags", [1]),
                      "widths": ("increment_block_widths", [192] * 5), "draw_count": ("draw_count", 511),
                      "block_days": ("block_days", [7, 28])}[mutation]
        public["protocol"][key] = value
    elif mutation == "unknown":
        record["unbound"] = 1
    elif mutation == "missing_record":
        public["records"].pop(dataset)
    elif mutation == "scale":
        record["scale_m"] *= 2
    elif mutation == "moment_shape":
        record["training_mean_standardized"].pop()
    elif mutation == "covariance":
        record["training_covariance_standardized2"][0][1] += 1
    elif mutation == "starts":
        record["test_starts_unix_s"].pop()
    elif mutation == "phase":
        record["phase_counts_utc"]["test"][0] += 1
    elif mutation == "origin":
        record["block_origin_unix_s"] += 900
    elif mutation == "weights":
        record["global_window_weights"] = [0.25, 0.25, 0.5]
    elif mutation in ("block_ids", "membership", "count", "mass", "sparse"):
        topology = record["blocks"]["14"]
        if mutation == "sparse":
            for key in ("block_ids", "counts", "mass"):
                topology[key].pop()
        else:
            key = {"block_ids": "block_ids", "membership": "window_block_ids", "count": "counts", "mass": "mass"}[mutation]
            topology[key][0] += 1
    elif mutation in ("seed", "seed_order"):
        fit["optimization_seeds"] = list(reversed(paper.SEEDS)) if mutation == "seed_order" else [42., 123, 2024]
    elif mutation == "missing_arm":
        record["neural"].pop("D")
    elif mutation in ("joint_shape", "increment_shape"):
        scores["per_window"]["joint_m" if mutation == "joint_shape" else "increment_m"][-1].pop()
    elif mutation == "joint_scale":
        fit["saved_joint_standardized"] = (np.asarray(fit["saved_joint_standardized"]) * record["scale_m"]).tolist()
    elif mutation == "last_fit_window":
        scores["per_window"]["increment_m"][-1][-1] += 1
    elif mutation == "mean":
        scores["mean"]["joint_m"] += 1
    elif mutation == "fit_mean":
        scores["per_fit_mean"]["joint_m"] = np.asarray(block["conditional_mean"]).mean(axis=1).tolist()
    elif mutation in ("conditional", "contribution", "reconstruction", "range"):
        key = {"conditional": "conditional_mean", "contribution": "weighted_contribution",
               "reconstruction": "reconstructed_global_mean", "range": "range"}[mutation]
        if mutation in ("conditional", "contribution"):
            block[key][-1][-1] += 1
        else:
            block[key][-1] += 1
    elif mutation in ("paired", "paired_seed"):
        paired = record["paired"]["D-A"]
        if mutation == "paired_seed":
            paired["optimization_seeds"].reverse()
        else:
            paired["scores"]["blocks"]["28"]["increment_m"]["conditional_mean"][-1][-1] += 1
    elif mutation.startswith("reference_"):
        key, value = {"reference_seed": ("sampling_seed", 45002), "reference_streams": ("integration_streams", 3),
                      "reference_variance": ("observation_variance_standardized2", 99), "reference_rank": ("effective_rank", 17),
                      "reference_covariance": ("covariance_sha256", "0" * 64), "reference_samples": ("prior_samples_sha256", "not-a-digest"),
                      "reference_shape": ("scores", deepcopy(reference["scores"]))}[mutation]
        reference[key] = value
        if mutation == "reference_shape":
            reference["scores"]["per_window"]["joint_m"] *= 3
    else:
        plot = fit["distribution"][-1]["decoder_mean"]
        if mutation == "plot_shape":
            plot["increment_magnitude_m"].pop()
        elif mutation == "plot_count":
            plot["window_count"] = 511
        elif mutation == "plot_power":
            plot["bin_power_m2"][-1] += 1
        else:
            plot["increment_magnitude_m"][-1] = -1
    with pytest.raises(ValueError):
        paper.validate_dart_diagnostics(public, master[1])


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown_field", "missing_field", "count", "float_count",
    "mean_shape", "sd_shape", "increment_shape", "frequency_shape", "power_shape", "frequency_alignment",
    "nonfinite", "negative_sd", "negative_increment", "negative_power", "parseval", "test_substitution"])
def test_dart_training_distribution_rejects_missing_malformed_or_misaligned_values(master, dart_master, mutation):
    public = deepcopy(dart_master[1])
    record = public["records"][paper.DATASETS[1]]
    training = record["train_distribution"]
    if mutation == "missing":
        record.pop("train_distribution")
    elif mutation == "malformed":
        record["train_distribution"] = []
    elif mutation == "unknown_field":
        training["unbound"] = 1
    elif mutation == "missing_field":
        training.pop("window_mean_relative_training_center_m")
    elif mutation in ("count", "float_count"):
        training["window_count"] = 3 if mutation == "count" else 4.0
    elif mutation.endswith("_shape"):
        field = {"mean_shape": "window_mean_relative_training_center_m", "sd_shape": "within_window_sd_m",
                 "increment_shape": "increment_magnitude_m", "frequency_shape": "frequency_cycles_per_day",
                 "power_shape": "bin_power_m2"}[mutation]
        training[field].pop()
    elif mutation == "frequency_alignment":
        training["frequency_cycles_per_day"].reverse()
    elif mutation == "nonfinite":
        training["window_mean_relative_training_center_m"][-1] = float("nan")
    elif mutation.startswith("negative_"):
        field = {"negative_sd": "within_window_sd_m", "negative_increment": "increment_magnitude_m",
                 "negative_power": "bin_power_m2"}[mutation]
        training[field][-1] = -1
    elif mutation == "parseval":
        training["bin_power_m2"][-1] += 1
    else:
        record["train_distribution"] = deepcopy(record["test_distribution"])
    with pytest.raises(ValueError):
        paper.validate_dart_diagnostics(public, master[1])


@pytest.mark.parametrize("mutation,label", [
    ("mean", "training window-mean average"),
    ("between", "training window-mean variance"),
    ("pooled", "training pooled variance"),
])
def test_dart_self_consistent_training_distribution_must_match_training_moments(master, dart_master, mutation, label):
    public = deepcopy(dart_master[1])
    record = public["records"][paper.DATASETS[1]]
    training = record["train_distribution"]
    means = np.asarray(training["window_mean_relative_training_center_m"])
    if mutation == "mean":
        training["window_mean_relative_training_center_m"] = (means + 0.5).tolist()
    elif mutation == "between":
        means = means.mean() + 2 * (means - means.mean())
        training["window_mean_relative_training_center_m"] = means.tolist()
        training["between_variance_m2"] = float(means.var())
    else:
        training["within_window_sd_m"] = (2 * np.asarray(training["within_window_sd_m"])).tolist()
        training["bin_power_m2"] = (4 * np.asarray(training["bin_power_m2"])).tolist()
        training["within_variance_m2"] *= 4
    training["pooled_variance_m2"] = training["within_variance_m2"] + training["between_variance_m2"]
    training["within_to_pooled_ratio"] = training["within_variance_m2"] / training["pooled_variance_m2"]
    # All local reductions, shapes and Parseval still pass. Only the independent
    # training moments expose the misalignment; do not stub either validator.
    paper._validate_dart_plot(training, 4)
    with pytest.raises(ValueError, match=label):
        paper.validate_dart_diagnostics(public, master[1])


@pytest.mark.parametrize("mutation,label", [
    ("training_spectrum", "training support spectrum"),
    ("support_spectrum", "training support spectrum"),
    ("support_within", "training support within_variance_m2"),
    ("support_between", "training support between_variance_m2"),
])
def test_combined_training_distribution_must_match_independently_valid_support(bundle, dart_master, mutation, label):
    public = public_fixture(bundle, dart_master)
    dataset = paper.DATASETS[1]
    training = public["dart_diagnostics"]["records"][dataset]["train_distribution"]
    support = public["dart_support"][dataset]
    if mutation == "training_spectrum":
        # Move positive-frequency energy without changing Parseval or moments.
        power = training["bin_power_m2"]
        power[2], power[4] = power[4], power[2]
    elif mutation == "support_spectrum":
        power = support["bin_power_m2"]
        power[1], power[3] = power[3], power[1]
    elif mutation == "support_within":
        support["bin_power_m2"] = (2 * np.asarray(support["bin_power_m2"])).tolist()
        support["within_variance_m2"] *= 2
    else:
        support["between_variance_m2"] *= 2
    support["pooled_variance_m2"] = support["within_variance_m2"] + support["between_variance_m2"]
    support["within_to_pooled_ratio"] = support["within_variance_m2"] / support["pooled_variance_m2"]
    support["one_cpd_share"] = support["bin_power_m2"][1] / support["within_variance_m2"]
    support["two_cpd_share"] = support["bin_power_m2"][3] / support["within_variance_m2"]
    assert sum(support["bin_power_m2"]) == pytest.approx(support["within_variance_m2"])
    paper.validate_dart_diagnostics(public["dart_diagnostics"], public["core"])
    # Matching this late error also proves all earlier support checks passed.
    with pytest.raises(ValueError, match=label):
        paper.validate_combined_manifest(public)


def test_dart_table_six_columns_fit_mean_actual_ranges_and_exact_only_combining(dart_master):
    support, diagnostics, _ = deepcopy(dart_master)
    # Component rows are checked separately; keep this test focused on the new fragment.
    components = {"jobs": {legacy.experiment.job_id(d, a, s): {"streams": [dict.fromkeys(paper.TABLE_METRICS, 1.)]}
                            for d in paper.DATASETS for a in paper.ARMS for s in paper.SEEDS}}
    record = diagnostics["records"][paper.DATASETS[1]]
    values = np.broadcast_to(np.array([1., 2., 90.])[:, None], (3, 3))
    record["neural"]["A"]["scores"] = paper._dart_score_summary(values, values / 2, record["blocks"])
    low, high = (record["references"]["diagonal_" + level] for level in ("low", "high"))
    for key in ("covariance_standardized2", "covariance_sha256", "prior_samples_sha256", "scores"):
        high[key] = deepcopy(low[key])
    fragment = paper.table_fragments(components, support, diagnostics, "c" * 64)["generated/dart_energy_rows.tex"]
    rows = fragment.splitlines()
    assert len(rows) == 31  # 4 neural + 4 references per record, one exact pair combined
    assert all(len(row.split(" & ")) == 6 for row in rows)
    assert rows[0].split(" & ")[2:4] == ["31", "15.5"]
    assert rows[0].split(" & ")[4] == "[1, 90]"
    assert sum("both $v_0,25v_0$" in row for row in rows) == 1
    assert all(row.endswith("\\\\") for row in rows[:-1]) and not rows[-1].endswith("\\\\")
    # Merely close scores must never be merged, even with equal covariance IDs.
    high["scores"]["mean"]["joint_m"] = float(np.nextafter(high["scores"]["mean"]["joint_m"], np.inf))
    fragment = paper.table_fragments(components, support, diagnostics, "c" * 64)["generated/dart_energy_rows.tex"]
    assert len(fragment.splitlines()) == 32 and "both $v_0,25v_0$" not in fragment


@pytest.mark.parametrize("mutation", ["saved_energy", "prior_truncation", "test_float32", "negative_scale"])
def test_dart_private_inputs_reject_semantic_corruption_after_hash_reseal(bundle, mutation):
    base, evidence = bundle
    dataset = paper.DATASETS[1]
    identifier = legacy.experiment.job_id(dataset, "A", paper.SEEDS[0])
    if mutation in ("saved_energy", "prior_truncation"):
        attempt, _ = paper._core_attempt(base / "run", identifier, evidence)
        path = attempt / "evaluation.npz"
        arrays = legacy._load_npz(path)
        if mutation == "saved_energy":
            arrays["energy"][-1] += 1e-8
        else:
            arrays["prior_samples"] = arrays["prior_samples"][:-1]
        np.savez_compressed(path, **arrays)
        completion_path = attempt.parent / "complete.json"
        completion = read_json(completion_path)
        completion["artifact_sha256"]["evaluation.npz"] = SHA(path)
        write_json(completion_path, completion)
        evidence["jobs"][identifier]["artifact_sha256"]["evaluation.npz"] = SHA(path)
    else:
        path = base / "run/prepared" / (dataset + ".npz")
        arrays = legacy._load_npz(path)
        arrays["test"] = arrays["test"].astype(np.float32) if mutation == "test_float32" else arrays["test"]
        if mutation == "negative_scale":
            arrays["scale"] *= -1
        np.savez_compressed(path, **arrays)
        evidence["prepared_sha256"][dataset + ".npz"] = SHA(path)
    with pytest.raises(ValueError):
        paper.dart_diagnostics(base / "run", evidence)


def test_physical_spectrum_uniform_weights_parseval_and_scale_once():
    t = np.arange(192) / 96
    waves = np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)
    windows = np.array([waves - 2, waves + 2])
    stats = paper.physical_window_statistics(windows, 3.0)
    assert stats["within_variance_m2"] == pytest.approx(0.625 * 9)
    assert stats["between_variance_m2"] == pytest.approx(4 * 9)
    assert stats["pooled_variance_m2"] == pytest.approx(4.625 * 9)
    assert stats["bin_power_m2"][2] == pytest.approx(0.5 * 9)
    assert stats["bin_power_m2"][4] == pytest.approx(0.125 * 9)
    np.testing.assert_allclose(stats["window_mean_relative_training_center_m"], [-6, 6])
    duplicated = paper.physical_window_statistics(np.repeat(windows, 2, axis=0), 3.0)
    np.testing.assert_allclose(stats["bin_power_m2"], duplicated["bin_power_m2"])
    assert stats["frequency_cycles_per_day"][-1] == 48
    with pytest.raises(ValueError, match="at least two"):
        paper.physical_window_statistics(windows[:1], 3.0)


@pytest.mark.parametrize("key", ["pooled_bin_power", "pooled_entry_variance"])
def test_spectral_identity_failures_are_rejected(monkeypatch, key):
    original = paper.window_spectrum_statistics
    def broken(*args):
        result = original(*args)
        result[key] *= 2
        return result
    monkeypatch.setattr(paper, "window_spectrum_statistics", broken)
    with pytest.raises(ValueError, match="Parseval|within plus between"):
        paper.physical_window_statistics(np.arange(384).reshape(2, 192), 2.0)


def test_decoder_mean_noise_recovery_uses_latent_draws_first():
    rng = np.random.default_rng(43001)
    latent = rng.standard_normal((512, 16)).astype(np.float32)
    noise = rng.standard_normal((512, 192))
    means = np.broadcast_to(latent[:, :1], (512, 192)).astype(np.float64)
    samples = means + np.sqrt(0.625) * noise
    np.testing.assert_allclose(paper.reconstruct_decoder_means(samples, 0.625), means, rtol=0, atol=1e-15)
    wrong_noise = np.random.default_rng(43001).standard_normal((512, 192))
    assert not np.allclose(samples - np.sqrt(0.625) * wrong_noise, means)


def test_comparison_figure_preserves_paired_ranges_and_readable_labels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    paired = {}
    for i, dataset in enumerate(paper.DATASETS):
        paired[dataset] = {}
        for field, suffix in (("self_mse_ratio", "/A"), ("negative_iw_score_difference", "-A")):
            paired[dataset][field] = {}
            for j, arm in enumerate(("B", "C", "D")):
                middle = (i + j + 1) / 8 if suffix == "/A" else i - j - 0.5
                values = [middle - (0.0625 if suffix == "/A" else 2), middle, middle + 0.5]
                paired[dataset][field][arm + suffix] = {
                    "median": middle, "min": values[0], "max": values[2],
                    "by_seed": dict(zip(("42", "123", "456"), values))}
    original = deepcopy(paired)
    figures, metadata = {}, {}
    def capture(fig, path, **kwargs):
        figures[Path(path).name] = fig
        metadata.update(kwargs["metadata"])
    monkeypatch.setattr(Figure, "savefig", capture)
    paper._comparison_figure(tmp_path, paired, "e" * 64)
    assert paired == original
    assert set(figures) == {"reconstruction_generation.png"}
    fig = figures["reconstruction_generation.png"]
    np.testing.assert_array_equal(fig.get_size_inches(), [6.4, 3.3])
    assert len(fig.axes) == 2
    labels = ["Gaussian", "23461\n2022", "44402\n2016", "46407\n2011", "52403\n2020"]
    for axis, field, suffix, scale, reference, title, ylabel in zip(
        fig.axes, ("self_mse_ratio", "negative_iw_score_difference"), ("/A", "-A"),
        ("log", "linear"), (1, 0), ("Reconstruction error", "Estimated log density"),
        ("MSE ratio to A (log scale)", "Change in negative IW estimate\nnats per coordinate"),
    ):
        assert axis.get_yscale() == scale
        np.testing.assert_array_equal(axis.lines[-1].get_ydata(), [reference, reference])
        assert axis.get_title() == title and axis.get_ylabel() == ylabel
        assert axis.get_xlabel() == "DART station / year"
        assert [tick.get_text() for tick in axis.get_xticklabels()] == labels
        assert all(tick.get_rotation() == 0 for tick in axis.get_xticklabels())
        assert len(axis.containers) == 3
        for container, arm, offset, color, marker in zip(
            axis.containers, ("B", "C", "D"), (-0.22, 0, 0.22),
            ("#C43D58", "#93B042", "#6A7D92"), ("o", "s", "^"),
        ):
            line, caps, bars = container.lines
            x = np.arange(5) + offset
            stats = [paired[d][field][arm + suffix] for d in paper.DATASETS]
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), [s["median"] for s in stats])
            assert line.get_color() == color and line.get_marker() == marker
            assert line.get_linestyle() == "None"
            for cap, endpoint in zip(caps, ("min", "max")):
                np.testing.assert_array_equal(cap.get_ydata(), [s[endpoint] for s in stats])
            assert len(bars[0].get_segments()) == 5
            for position, segment, stats in zip(x, bars[0].get_segments(), stats):
                np.testing.assert_array_equal(segment[:, 0], [position, position])
                np.testing.assert_array_equal(segment[:, 1], [stats["min"], stats["max"]])
        assert [text.get_text() for text in axis.get_legend().get_texts()] == [arm + suffix for arm in ("B", "C", "D")]
    assert metadata == {
        "Software": "PriGM CPU evidence builder",
        "Description": "All five datasets, all three paired optimization seeds; medians and full seed ranges.",
        "CombinedInputSHA256": "e" * 64}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.get_tightbbox(renderer)
    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 6.4 and box.y1 <= 3.3
    for axis in fig.axes:
        boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
        assert all(right.x0 - left.x1 >= 3 for left, right in zip(boxes, boxes[1:]))


def test_figures_use_test_all_draws_fit_ranges_and_physical_labels(bundle, monkeypatch):
    from matplotlib.figure import Figure
    support, _, private = paper.dart_diagnostics(bundle[0] / "run", bundle[1])
    figures, metadata = {}, {}
    def capture(fig, path, **kwargs):
        figures[Path(path).name] = fig
        metadata[Path(path).name] = kwargs["metadata"]
    monkeypatch.setattr(Figure, "savefig", capture)
    paper._figures(bundle[0], support, private, "d" * 64)
    colors = {"A": "#C4A04A", "B": "#C43D58", "C": "#93B042", "D": "#6A7D92"}
    titles = ["Station 23461, 2022", "Station 44402, 2016", "Station 46407, 2011", "Station 52403, 2020"]
    support_fig = figures["dart_support_spectra.png"]
    assert len(support_fig.axes) == 8
    for i, dataset in enumerate(paper.DATASETS[1:]):
        axis, fractions = support_fig.axes[2 * i:2 * i + 2]
        assert axis.get_title() == titles[i]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(1, 97) / 2)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), support[dataset]["bin_power_m2"])
        assert "m$^2$" in axis.get_ylabel()
        assert axis.lines[0].get_linestyle() == ":" and axis.lines[0].get_color() == "black"
        assert axis.lines[0].get_label() == "Train" and axis.lines[-1].get_label() == "Test"
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), private[dataset]["train"]["bin_power_m2"][1:])
        assert axis.lines[-1].get_linestyle() == "-" and axis.lines[-1].get_color() == "black"
        np.testing.assert_allclose(axis.lines[-1].get_ydata(), private[dataset]["test"]["bin_power_m2"][1:])
        assert len(axis.lines) == 10 and len(axis.collections) == 8
        for j, arm in enumerate(paper.ARMS):
            for offset, kind in enumerate(("noisy", "decoder_mean")):
                curves = np.array([fit[kind]["bin_power_m2"][1:] for fit in private[dataset]["fits"][arm]])
                assert axis.lines[1 + 2 * j + offset].get_color() == colors[arm]
                np.testing.assert_allclose(axis.lines[1 + 2 * j + offset].get_ydata(), np.median(curves, axis=0))
                vertices = axis.collections[2 * j + offset].get_paths()[0].vertices
                assert vertices[:, 1].min() == pytest.approx(curves.min())
                assert vertices[:, 1].max() == pytest.approx(curves.max())
        assert sum(p.get_height() for p in fractions.patches) == pytest.approx(1)
    distribution = figures["dart_distribution_diagnostics.png"]
    assert len(distribution.axes) == 12
    for row, dataset in enumerate(paper.DATASETS[1:]):
        record = private[dataset]
        for column, axis in enumerate(distribution.axes[row * 3:row * 3 + 3]):
            assert len(axis.lines) == 10 and len(axis.collections) == 8
            field = ("window_mean_relative_training_center_m", "within_window_sd_m", "increment_magnitude_m")[column]
            assert axis.get_xlabel() == ("Window mean\nminus train mean (m)", "Window SD (m)", "Increment length (m)")[column]
            if column == 0:
                assert axis.get_ylabel() == "ECDF"
                assert axis.get_title() == titles[row]
            train, test = (record[split][field] for split in ("train", "test"))
            assert len(train) == 4 and len(test) == 3
            neural = [fit[kind][field] for arm in paper.ARMS for fit in record["fits"][arm]
                      for kind in ("noisy", "decoder_mean")]
            grid = np.unique(np.concatenate([train, test, *neural]))
            # A test/neural-only grid drops actual training breakpoints.
            assert np.setdiff1d(train, np.concatenate([test, *neural])).size > 0
            for line in axis.lines:
                np.testing.assert_array_equal(line.get_xdata(), grid)
                assert line.get_drawstyle() == "steps-post"
            for line, values, split, style, width in (
                (axis.lines[-2], train, "train", ":", 1.6),
                (axis.lines[-1], test, "test", "-", 2),
            ):
                assert line.get_color() == "black" and line.get_linestyle() == style
                assert line.get_linewidth() == width
                assert line.get_label() == f"Observed {split} (N={len(values)})"
                expected = np.count_nonzero(values[:, None] <= grid, axis=0) / len(values)
                np.testing.assert_array_equal(line.get_ydata(), expected)
                assert line.get_ydata()[-1] == 1
            for j, arm in enumerate(paper.ARMS):
                for offset, kind in enumerate(("noisy", "decoder_mean")):
                    line = axis.lines[2 * j + offset]
                    assert line.get_color() == colors[arm]
                    assert line.get_linestyle() == ("-" if offset == 0 else "--")
                    assert line.get_label() == arm + (" noisy" if kind == "noisy" else " decoder mean")
                    values = [fit[kind][field] for fit in record["fits"][arm]]
                    assert len(values) == 3 and all(len(v) == 512 for v in values)
                    curves = np.array([np.count_nonzero(v[:, None] <= grid, axis=0) / 512 for v in values])
                    np.testing.assert_array_equal(line.get_ydata(), np.median(curves, axis=0))
                    vertices = axis.collections[2 * j + offset].get_paths()[0].vertices
                    assert vertices[:, 1].min() == pytest.approx(curves.min())
                    assert vertices[:, 1].max() == pytest.approx(curves.max())
    assert all(info["CombinedInputSHA256"] == "d" * 64 for info in metadata.values())
    for fig in (support_fig, distribution):
        np.testing.assert_array_equal(fig.get_size_inches(), [5.5, 7.3])
        assert fig._suptitle is None
        assert len(fig.legends) == 1 and all(axis.get_legend() is None for axis in fig.axes)
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [*paper.ARMS, "Noisy", "Decoder mean", "Test", "Train"]
        test_handle, train_handle = fig.legends[0].get_lines()[-2:]
        assert test_handle.get_color() == train_handle.get_color() == "black"
        assert test_handle.get_linestyle() == "-" and train_handle.get_linestyle() == ":"
        assert all(text.get_fontsize() == 8 for text in fig.legends[0].get_texts())
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = fig.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 5.5 and box.y1 <= 7.3
        legend = fig.legends[0].get_window_extent(renderer)
        assert all(axis.get_tightbbox(renderer).y1 < legend.y0 for axis in fig.axes)


def test_full_synthetic_build_reuses_frozen_builder_and_publishes_exactly_14_manifest_last(bundle, monkeypatch):
    base, core = bundle
    destination = base / "paper"
    (destination / "figures").mkdir(parents=True)
    (destination / "figures/prior_samples.png").write_bytes(b"old unused asset must remain")
    (destination / "main.tex").write_text("untouched manuscript")
    old_builder_bytes = (ROOT / "src/build_prigm_paper.py").read_bytes()
    calls, replacements = [], []
    original_build, original_replace = legacy.build_paper, paper.os.replace
    original_comparison = paper._comparison_figure
    rendered = {}
    def track(*args, **kwargs):
        calls.append(args)
        assert Path(args[1]) != destination
        result = original_build(*args, **kwargs)
        rendered["paired"] = result["summary_statistics"]["paired"]
        rendered["paired_snapshot"] = deepcopy(rendered["paired"])
        rendered["legacy_sha"] = SHA(Path(args[1]) / "figures/reconstruction_generation.png")
        return result
    def comparison(stage, paired, combined_sha):
        assert paired is rendered["paired"]
        original_comparison(stage, paired, combined_sha)
        rendered["sha"] = SHA(Path(stage) / "figures/reconstruction_generation.png")
        assert rendered["sha"] != rendered["legacy_sha"]
        from PIL import Image
        with Image.open(Path(stage) / "figures/reconstruction_generation.png") as image:
            assert image.info["CombinedInputSHA256"] == combined_sha
    def replace(source, target):
        if Path(target).is_relative_to(destination) and ".prigm-combined-" not in str(target):
            replacements.append(Path(target).relative_to(destination).as_posix())
        return original_replace(source, target)
    monkeypatch.setattr(legacy, "build_paper", track)
    monkeypatch.setattr(paper, "_comparison_figure", comparison)
    monkeypatch.setattr(paper.os, "replace", replace)
    result = build(bundle)
    assert len(calls) == 1
    assert result["core"]["summary_statistics"]["paired"] == rendered["paired_snapshot"]
    assert result["output_sha256"]["figures/reconstruction_generation.png"] == rendered["sha"]
    assert replacements == [*paper.AUDIT_OUTPUTS, paper.EVIDENCE_NAME]
    assert len(result["output_sha256"]) == len(paper.AUDIT_OUTPUTS) == 14
    assert paper.validate_combined_manifest(result, core["run_manifest_sha256"]) == result
    assert result == read_json(destination / paper.EVIDENCE_NAME)
    for name, digest in result["output_sha256"].items():
        assert SHA(destination / name) == digest
    assert (destination / "figures/prior_samples.png").read_bytes() == b"old unused asset must remain"
    assert (destination / "main.tex").read_text() == "untouched manuscript"
    assert (ROOT / "src/build_prigm_paper.py").read_bytes() == old_builder_bytes
    text = (destination / paper.EVIDENCE_NAME).read_text()
    assert "/home/" not in text and "PRIVATE" not in text and '"runtime":' not in text
    assert r"\PrigmCoreSHA" in (destination / "generated/summary_values.tex").read_text()
    assert r"\PrigmCombinedSHA" in (destination / "generated/combined_binding.tex").read_text()


@pytest.mark.parametrize("failure", ["render", "legacy_closure", "decision", "changed_input", "profile"])
def test_failure_publishes_nothing(bundle, monkeypatch, failure):
    base, core = bundle
    if failure == "render":
        monkeypatch.setattr(paper, "_figures", lambda *args: (_ for _ in ()).throw(RuntimeError("render failure")))
    elif failure == "legacy_closure":
        (base / "run/results.json").write_text("{}")
    elif failure == "decision":
        write_json(base / "decision.json", {"schema_version": 1, "new_scientific_training_authorized_for_launch": True})
    elif failure == "changed_input":
        original = paper._figures
        def change(*args):
            original(*args)
            (base / "decision.json").write_text("changed after verification")
        monkeypatch.setattr(paper, "_figures", change)
    else:
        with pytest.raises(ValueError, match="Unknown combined profile"):
            paper.build_paper(base / "run", base / "paper", core["run_manifest_sha256"], base / "refinement",
                              base / "components", base / "decision.json", base / "synthetic-numerical", profile="new_matrix")
        assert not (base / "paper").exists()
        return
    with pytest.raises((ValueError, RuntimeError)):
        build(bundle)
    assert not (base / "paper").exists()


@pytest.mark.parametrize("mutation", ["extra_worker", "extra_artifact", "array_bytes", "source_bytes"])
def test_post_render_component_closure_changes_block_publication(bundle, monkeypatch, mutation):
    base, _ = bundle
    def render_and_change(stage, *args):
        # A renderer double isolates final closure checks; real PNG rendering is
        # exercised separately by the full synthetic integration test.
        for name in ("dart_support_spectra.png", "dart_distribution_diagnostics.png"):
            (stage / "figures" / name).write_bytes(b"synthetic renderer payload")
        worker = base / "components/worker-0"
        if mutation == "extra_worker":
            (base / "components/worker-4").mkdir()
        elif mutation == "extra_artifact":
            (worker / "extra.json").write_text("{}")
        elif mutation == "array_bytes":
            path = worker / "gaussian_A_42-selected.npz"
            path.write_bytes(path.read_bytes() + b"changed")
        else:
            (worker / "source/prigm_v2_metrics.py").write_text("# changed after validation")
    monkeypatch.setattr(paper, "_figures", render_and_change)
    with pytest.raises(ValueError, match="inventory|changed"):
        build(bundle)
    assert not (base / "paper").exists()


@pytest.mark.parametrize("mutation", ["failure", "snapshot", "hdf5_resealed", "prepared", "source", "smoke"])
def test_post_render_numerical_closure_changes_block_publication(bundle, monkeypatch, mutation):
    base, _ = bundle
    def render_and_change(stage, *args):
        for name in ("dart_support_spectra.png", "dart_distribution_diagnostics.png"):
            (stage / "figures" / name).write_bytes(b"synthetic renderer payload")
        root = base / "synthetic-numerical"
        if mutation == "failure":
            (root / "full-high/failure.json").write_text("{}")
        else:
            target = root / {"snapshot": "full-low/state-01000.npz", "hdf5_resealed": "full-low/base.weights.h5",
                             "prepared": "prepared/gaussian_v2.npz", "source": "src/prigm_v2_model.py",
                             "smoke": "smoke-low/trace.jsonl"}[mutation]
            target.write_bytes(target.read_bytes() + b"changed after verification")
            if mutation == "hdf5_resealed":
                reseal_numerical(target.parent)
    monkeypatch.setattr(paper, "_figures", render_and_change)
    with pytest.raises(ValueError, match="inventory|changed"):
        build(bundle)
    assert not (base / "paper").exists()


@pytest.mark.parametrize("reseal_archive", [False, True])
def test_completion_receipt_is_bound_to_previously_verified_core_archive(bundle, reseal_archive):
    base, _ = bundle
    path = base / "run/jobs/gaussian_A_42/complete.json"
    completion = read_json(path)
    completion["extra_note"] = "changed without changing any core artifact"
    write_json(path, completion)
    worker = base / "components/worker-0"
    record_path = worker / "gaussian_A_42-selected.json"
    record = read_json(record_path)
    record["core_completion_sha256"] = SHA(path)
    write_json(record_path, record)
    reseal_worker(worker)
    if reseal_archive:
        reseal(base / "run")
    with pytest.raises(ValueError, match="archive"):
        verify(bundle)


def test_worker_added_during_array_verification_is_rejected(bundle, monkeypatch):
    base, _ = bundle
    load = legacy._load_npz
    def changing_load(path):
        value = load(path)
        if Path(path).name == "gaussian_A_42-selected.npz":
            (base / "components/worker-4").mkdir()
        return value
    monkeypatch.setattr(legacy, "_load_npz", changing_load)
    with pytest.raises(ValueError, match="inventory"):
        verify(bundle)


@pytest.mark.parametrize("source", ["decision.json", "components"])
def test_symlink_input_roots_or_decision_are_rejected(bundle, source):
    base, _ = bundle
    path = base / source
    renamed = base / (source + "-original")
    path.rename(renamed)
    path.symlink_to(renamed, target_is_directory=renamed.is_dir())
    with pytest.raises(ValueError, match="Symlink"):
        build(bundle)
    assert not (base / "paper").exists()


def test_cpu_import_has_no_tensorflow_h5py_or_release_import():
    script = f"""import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import build_prigm_v2_paper
assert 'tensorflow' not in sys.modules
assert 'h5py' not in sys.modules
assert 'verify_prigm_release' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


@pytest.mark.parametrize("mutation", [None, "helper", "reviewed", "numerical", "output", "ecdf_output", "core", "delegate"])
def test_dart_combined_dispatch_checks_sources_and_preserves_core_validation(
        bundle, dart_master, dart_public_fixture, monkeypatch, mutation):
    import prigm_dart_paper_evidence as dart
    value = public_fixture(bundle, dart_master)
    monkeypatch.setattr(dart, "CORE_SOURCES", {Path(k).name: v for k, v in value["core"]["code_sha256"].items()})
    extension = deepcopy(dart_public_fixture)
    value.update(schema_version=3, profile="dart_extension", dart_extension=extension,
                 methods=dict(paper.DART_EXTENSION_METHODS), output_sha256=dict.fromkeys(paper.DART_EXTENSION_OUTPUTS, "a" * 64))
    del value["dart_support"], value["dart_diagnostics"]
    index = value["input_index"]
    del index["dart_diagnostics"]
    for name in paper.DART_EXTENSION_SOURCE_IDS:
        index["source_sha256"].setdefault(name, "d" * 64)
    inner = extension["input_index"]
    inner["helper_sha256"] = index["source_sha256"]["src/prigm_dart_paper_evidence.py"]
    inner["reviewed_source_sha256"] = {name: index["source_sha256"]["src/" + name] for name in dart.REVIEWED_SOURCES}
    index["dart_extension"] = inner
    monkeypatch.setattr(paper, "FROZEN_NUMERICAL_SOURCE_SHA256", deepcopy(value["numerical_validation"]["source_sha256"]))
    calls = []
    # This test isolates the combined boundary; the helper's schema and private
    # scientific checks have their own fixtures and real-artifact integration.
    def validate(received, core, input_index):
        assert received is extension and core is value["core"] and input_index is inner
        calls.append(True)
        if mutation == "delegate":
            raise ValueError("DART helper rejected evidence")
    monkeypatch.setattr(dart, "validate_dart_extension", validate)
    if mutation == "helper":
        inner["helper_sha256"] = "e" * 64
    elif mutation == "reviewed":
        inner["reviewed_source_sha256"]["prigm_v2_metrics.py"] = "e" * 64
    elif mutation == "numerical":
        value["numerical_validation"]["source_sha256"][paper.NUMERICAL_DRIVER] = "e" * 64
        index["numerical_validation"]["source_sha256"][paper.NUMERICAL_DRIVER] = "e" * 64
    elif mutation == "output":
        value["output_sha256"].pop("generated/dart_block_rows.tex")
    elif mutation == "ecdf_output":
        value["output_sha256"].pop("figures/dart_ecdf_fresh.png")
    elif mutation == "core":
        value["core"]["summary_statistics"]["job_count"] = 59
    value["combined_input_sha256"] = paper.canonical_sha256(index)
    if mutation is None:
        assert paper.validate_combined_manifest(value) is value
        assert calls == [True]
    else:
        with pytest.raises(ValueError):
            paper.validate_combined_manifest(value)


@pytest.mark.parametrize("mutation", [None, "core_producer", "stage_producer", "refiner_producer",
                                      "current_core", "current_model", "numerical_producer"])
def test_maintained_public_contract_separates_literal_producers_from_current_sources(mutation):
    import prigm_dart_paper_evidence as dart

    # Public schema fixture only, not a claim that the historical evidence was
    # regenerated. Keep its independent literal producer identities unchanged.
    public = json.loads((ROOT / "paper/prigm/paper_evidence_manifest.json").read_text())
    old_core = deepcopy(public["core"]["code_sha256"])
    index = public["input_index"]
    index["source_sha256"] = {name: SHA(ROOT / name) for name in paper.DART_EXTENSION_SOURCE_IDS}
    inner = public["dart_extension"]["input_index"]
    inner["reviewed_source_sha256"] = dict(dart.REVIEWED_SOURCES)
    inner["helper_sha256"] = SHA(ROOT / "src/prigm_dart_paper_evidence.py")
    inner["files"].update({"reviewed/" + name: digest for name, digest in dart.REVIEWED_SOURCES.items()})
    inner["files"]["reviewed/prigm_dart_paper_evidence.py"] = inner["helper_sha256"]
    index["dart_extension"] = deepcopy(inner)
    public["builder_sha256"] = index["source_sha256"]["src/build_prigm_v2_paper.py"]
    public["core"]["builder_sha256"] = index["source_sha256"]["src/build_prigm_paper.py"]
    assert old_core["src/prigm_metrics.py"] != index["source_sha256"]["src/prigm_metrics.py"]
    assert old_core["src/prigm_model.py"] != index["source_sha256"]["src/prigm_model.py"]
    public["combined_input_sha256"] = paper.canonical_sha256(index)
    # Every corruption case starts from a passing real combined/helper closure.
    assert paper.validate_combined_manifest(public, dart.CORE_SHA256) is public
    if mutation == "core_producer":
        public["core"]["code_sha256"]["src/prigm_metrics.py"] = index["source_sha256"]["src/prigm_metrics.py"]
    elif mutation == "stage_producer":
        name = next(k for k in inner["files"] if k.startswith("stage-batch-") and k.endswith("source/prigm_metrics.py"))
        inner["files"][name] = dart.REVIEWED_SOURCES["prigm_metrics.py"]
    elif mutation == "refiner_producer":
        public["refinement"]["code_sha256"]["src/prigm_refine_evaluation.py"] = "e" * 64
    elif mutation in ("current_core", "current_model"):
        name = "prigm_metrics.py" if mutation == "current_core" else "prigm_model.py"
        index["source_sha256"]["src/" + name] = dart.CORE_SOURCES[name]
        inner["reviewed_source_sha256"][name] = dart.CORE_SOURCES[name]
        inner["files"]["reviewed/" + name] = dart.CORE_SOURCES[name]
    elif mutation == "numerical_producer":
        public["numerical_validation"]["source_sha256"]["src/prigm_metrics.py"] = dart.REVIEWED_SOURCES["prigm_metrics.py"]
        index["numerical_validation"]["source_sha256"] = deepcopy(public["numerical_validation"]["source_sha256"])
    index["dart_extension"] = deepcopy(inner)
    public["combined_input_sha256"] = paper.canonical_sha256(index)
    if mutation is None:
        assert dart.validate_dart_extension(public["dart_extension"], public["core"], inner) is public["dart_extension"]
        assert paper.validate_combined_manifest(public, dart.CORE_SHA256) is public
        assert public["core"]["code_sha256"] == old_core
    else:
        with pytest.raises(ValueError):
            paper.validate_combined_manifest(public, dart.CORE_SHA256)


def test_dart_rendering_keeps_endpoints_weights_and_raw_mse_distinct(dart_public_fixture):
    import prigm_dart_paper_evidence as dart
    evidence = deepcopy(dart_public_fixture)
    fresh = evidence["records"][dart.PRIMARY]
    for seed in paper.SEEDS:
        model = fresh["models"][f"C/{seed}/original"]
        model["components"]["weighted"]["posterior_mean_mse"]["mean"] = 2.
        fresh["models"][f"C/{seed}/variance_refit"]["scores"]["joint"]["weighted"]["mean"] = 1.002
        for score in dart.SCORES:
            model["scores"][score]["uniform"]["mean"] = 8.
            model["selected_secondary"]["scores"][score]["weighted"]["mean"] = 4.
    components, core = {"jobs": {}}, {"jobs": {}}
    for arm in paper.ARMS:
        for seed in paper.SEEDS:
            identifier = legacy.experiment.job_id("gaussian", arm, seed)
            components["jobs"][identifier] = {"streams": [{key: 2. for key in paper.TABLE_METRICS}] * 3}
            core["jobs"][identifier] = {"self_mse": .125}
    fragments = paper._dart_table_fragments(components, evidence, "f" * 64, core)
    rows = fragments["generated/component_rows.tex"].splitlines()
    assert len(rows) == 24
    assert rows[0].startswith("Gaussian & A & 0.125 & 2 &")
    assert next(row for row in rows if row.startswith("44402/2018 & C &")).startswith("44402/2018 & C & 0.08 & 2 &")
    assert next(row for row in rows if row.startswith("23461/2022 & C &")).startswith("23461/2022 & C & 0.1 &")
    secondary = fragments["generated/dart_selected_rows.tex"]
    assert r"C-A & Selected/UTC & 10000--10000 & \shortstack{3\\{[3, 3]}}" in secondary
    assert r"C-A & Final/uniform & 10000 & \shortstack{7\\{[7, 7]}}" in secondary
    variance_row = next(row for row in fragments["generated/dart_variance_rows.tex"].splitlines()
                        if row.startswith("44402/2018 & C &"))
    assert r"\shortstack{3000\\{[3000, 3000]}} & \shortstack{2\\{[2, 2]}}" in variance_row
    assert fragments["generated/dart_extension_rows.tex"].count("Monte Carlo width") == 10
    assert "MC width" not in fragments["generated/dart_extension_rows.tex"]
    assert len(fragments["generated/dart_reference_rows.tex"].splitlines()) == 45
    assert len(fragments["generated/dart_block_rows.tex"].splitlines()) == 5
    assert all(text.endswith("\n") and text.isascii() for text in fragments.values())
    assert all(" \\\\\n" in text for name, text in fragments.items() if name not in
               ("generated/combined_binding.tex", "generated/dart_record_rows.tex"))


def test_dart_figure_labels_and_statistics_are_bound(dart_public_fixture, tmp_path, monkeypatch):
    import matplotlib.axes
    from PIL import Image
    calls = []
    original = matplotlib.axes.Axes.errorbar
    def record(axis, x, y, **kwargs):
        calls.append((np.asarray(x), np.asarray(y), kwargs))
        return original(axis, x, y, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", record)
    (tmp_path / "figures").mkdir()
    paper._dart_comparison_figure(tmp_path, dart_public_fixture, "f" * 64)
    assert len(calls) == 6
    assert all(len(x) == 5 and np.array_equal(x, np.zeros(5)) for x, _, _ in calls)
    assert [call[2]["color"] for call in calls[:2]] == ["#009E73", "#CC79A7"]
    with Image.open(tmp_path / "figures/reconstruction_generation.png") as image:
        assert image.info["CombinedInputSHA256"] == "f" * 64
        assert image.size == (1280, 720)
        assert "2018 final" in image.info["Description"]


def test_extension_weighted_ecdf_ties_and_fit_not_seed_pooling():
    grid = np.array([-1., 0., 1., 2.])
    np.testing.assert_allclose(paper._dart_weighted_ecdf([0, 0, 2], [1, 2, 3], grid), [0, .5, .5, 1])
    np.testing.assert_allclose(paper._dart_weighted_ecdf([0, 0, 2], [1, 1, 4], grid), [0, 1/3, 1/3, 1])
    def stream(x):
        return {"features": np.full((2, 3), x), "weights": np.array([.5, .5])}
    fits = [[stream(0), stream(2), stream(4)], [stream(8)] * 3, [stream(10)] * 3]
    support, curves = paper._dart_ecdf_summary(fits, 0)
    np.testing.assert_array_equal(support, [0, 2, 4, 8, 10])
    np.testing.assert_allclose(curves[0], [1/3, 2/3, 1, 1, 1])
    assert np.median(curves[:, 2]) == 0 and curves[:, 2].mean() == pytest.approx(1/3)


def test_extension_features_before_mc_and_physical_power():
    wave = np.cos(2 * np.pi * np.arange(192) / 192)
    windows = np.array([wave - 2, -wave + 2])
    summary = paper._dart_distribution(windows, [.25, .75], 3.)
    np.testing.assert_allclose(summary["features"][:, 0], [-6, 6])
    np.testing.assert_allclose(summary["features"][:, 1], 3 / np.sqrt(2))
    np.testing.assert_allclose(summary["features"][:, 2], np.linalg.norm(paper.increment_features(windows), axis=1) * 3)
    assert summary["power"].sum() == pytest.approx(4.5)
    assert summary["power"][0] == pytest.approx(4.5)
    np.testing.assert_allclose(summary["fractions"], [.5 / 3.5, 3 / 3.5])
    averaged = paper._dart_distribution(windows.mean(axis=0)[None], [1], 3.)
    assert averaged["features"][0, 1] == 0
    assert np.all(averaged["power"] == 0) and averaged["fractions"] is None
    nyquist = paper._dart_distribution(((-1.) ** np.arange(192))[None], [1], 2.)
    assert nyquist["power"][-1] == pytest.approx(4.)


@pytest.mark.parametrize("key", ["pooled_bin_power", "pooled_entry_variance"])
def test_extension_spectral_identity_rejection(monkeypatch, key):
    original = paper.window_spectrum_statistics
    def broken(*args):
        value = original(*args)
        value[key] *= 2
        return value
    monkeypatch.setattr(paper, "window_spectrum_statistics", broken)
    with pytest.raises(ValueError, match="Parseval|within plus between"):
        paper._dart_distribution(np.arange(384).reshape(2, 192), [1, 1], 2.)


def test_extension_refit_preserves_stream_pairing(dart_public_fixture):
    assert "standardized coordinates" in paper.DART_EXTENSION_METHODS["variance_refit"]
    assert "in meters" not in paper.DART_EXTENSION_METHODS["variance_refit"]
    assert "UTC start-hour weights" in paper.DART_EXTENSION_METHODS["components"]
    assert "Observed fractions" not in paper.DART_EXTENSION_METHODS["spectra"]
    assert paper.METHODS["components"].startswith("Uniform test-window arithmetic means")
    record = deepcopy(next(iter(dart_public_fixture["records"].values())))
    record["scale"] = 2.
    for index, seed in enumerate(paper.SEEDS):
        original = record["models"][f"A/{seed}/original"]
        refit = record["models"][f"A/{seed}/variance_refit"]
        original["observation_variance"] = 4.
        refit["variance_fit"]["pooled"] = 8. + index
        for score in ("joint", "centered", "increment"):
            original["scores"][score]["weighted"]["values"] = [1., 20., 3.]
            refit["scores"][score]["weighted"]["values"] = [4., 19., 5.]
    ratios, deltas = paper._dart_refit_points(record, "A")
    np.testing.assert_allclose(ratios, [2, 2.25, 2.5])
    np.testing.assert_allclose(deltas, np.broadcast_to([3, -1, 2], (3, 3, 3)))


def test_extension_real_saved_draw_renderer(tmp_path, monkeypatch, dart_public_fixture):
    from PIL import Image
    evidence = deepcopy(dart_public_fixture)
    # The public fixture has five records; clone historical protocol to fill all seven cases.
    template = deepcopy(next(iter(evidence["records"].values())))
    for name in paper.DATASETS[1:]:
        evidence["records"].setdefault(name, deepcopy(template))
        evidence["records"][name].update(station=name[:5], year=int(name[-4:]), target="original_uniform_windows")
    for name in ("dart_44402_2018", "dart_21415_2018", "dart_21418_2018"):
        evidence["records"].setdefault(name, deepcopy(template))
        evidence["records"][name].update(station=name.split("_")[1], year=2018, target="six_UTC_bins_equal")
    draws = np.broadcast_to(np.cos(2 * np.pi * np.arange(192) / 192), (3, 1024, 192)).copy()
    draws[:, -1] += 100  # retained outlier, not clipped or subsampled
    path = tmp_path / "draws.npz"
    np.savez(path, original__prior_samples=draws, original__prior_means=draws * .5,
             ppca_fitted__prior_samples=draws, irrelevant_likelihood=np.array([object()], dtype=object))
    evidence["input_index"]["files"] = {"diagnostics/draws.npz": SHA(path)}
    private = {"data": {}, "model_paths": {}, "reference_paths": {}}
    for name, record in evidence["records"].items():
        private["data"][name] = {"train": draws[0, :6], "test": draws[1, :6],
            "train_weights": np.array([1, 1, 1, 1, 1, 2]) / 7,
            "eval_weights": np.array([2, 1, 1, 1, 1, 1]) / 7, "scale": record["scale"]}
        for arm in paper.ARMS:
            for seed in paper.SEEDS:
                private["model_paths"][f"{name}/{arm}/{seed}"] = path
        private["reference_paths"][name + "/low"] = path
    (tmp_path / "figures").mkdir()
    from matplotlib.figure import Figure
    savefig = Figure.savefig
    def inspect_figure(fig, filename, **kwargs):
        name = Path(filename).name
        assert fig.get_figwidth() == 5.5
        assert fig.get_figheight() <= 9 * .81
        for axis in fig.axes:
            assert axis.xaxis.label.get_fontsize() >= 8
        if name.startswith("dart_ecdf"):
            assert len(fig.axes) == (12 if "historical" in name else 9)
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            assert "Observed train" in labels and "Observed test" in labels
            assert "Recorded train" not in labels and "Recorded test" not in labels
            assert fig.axes[0].get_xlabel() == "Window mean\nminus train mean (m)"
            assert fig.axes[1].get_xlabel() == "Window SD (m)"
            assert fig.axes[2].get_xlabel() == "Increment length (m)"
            for axis in fig.axes:
                for line in axis.lines:
                    x = line.get_xdata()
                    assert axis.get_xlim()[0] <= min(x) <= max(x) <= axis.get_xlim()[1]
            # The single extreme final draw is in every common mean-feature support.
            assert max(fig.axes[0].lines[0].get_xdata()) == pytest.approx(
                100 * evidence["records"][paper.DATASETS[1] if "historical" in name else "dart_44402_2018"]["scale"])
        elif name.startswith("dart_spectra"):
            legend = [text.get_text() for text in fig.legends[0].get_texts()]
            assert "Prior decoder means" in legend
            assert "Within window" not in legend and "Window means" not in legend
            assert "Variance share" not in [axis.get_title() for axis in fig.axes]
            n_rows = 4 if "historical" in name else 3
            assert len(fig.axes) == 2 * n_rows
            for row in range(n_rows):
                low, high = fig.axes[row * 2:row * 2 + 2]
                np.testing.assert_allclose(low.get_ylim(), high.get_ylim())
                for axis in (low, high):
                    np.testing.assert_allclose(axis.get_xticks(), [1, 2, 12, 24, 48])
                    labels = [tick.get_text() for tick in axis.get_xticklabels()]
                    assert [text.strip() for text in labels] == ["1", "2", "12", "24", "48"]
                    assert any("\n" in text for text in labels)
                    for line in axis.lines:
                        positive = np.asarray(line.get_ydata())
                        positive = positive[positive > 0]
                        assert axis.get_ylim()[0] <= positive.min() <= positive.max() <= axis.get_ylim()[1]
            assert fig.axes[0].get_title() == "Low observation variance: A/C"
            assert fig.axes[1].get_title() == "High observation variance: B/D"
        elif name == "dart_variance_refit.png":
            assert len(fig.axes) == 4 and len(fig.axes[0].get_yticks()) == 28
            assert all(len(axis.lines) == 85 for axis in fig.axes)  # 84 optimizer points plus reference
            assert all(len(axis.collections) == 84 for axis in fig.axes[1:])
            title = fig._suptitle.get_text() if fig._suptitle is not None else ""
            assert title == "Observation variance reset\nThree seeds. Thin bars span three sampling streams"
            assert "Fixed-function" not in title and "MC-stream" not in title
        return savefig(fig, filename, **kwargs)
    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    paper._dart_extension_figures(tmp_path, evidence, private, "e" * 64)
    names = {"dart_ecdf_historical.png", "dart_ecdf_fresh.png", "dart_spectra_historical.png",
             "dart_spectra_fresh.png", "dart_variance_refit.png"}
    assert {p.name for p in (tmp_path / "figures").iterdir()} == names
    assert len(paper.DART_EXTENSION_OUTPUTS) == 21
    for name in names:
        assert "figures/" + name in paper.DART_EXTENSION_OUTPUTS
        with Image.open(tmp_path / "figures" / name) as image:
            assert image.info["CombinedInputSHA256"] == "e" * 64
            assert image.width == 1100
    with pytest.raises(ValueError, match="digest inventory"):
        paper._dart_archive_summary(path, {"draws.npz": "0" * 64}, ("original__prior_samples",), 1.)
    original = paper._dart_distribution
    def mutate(*args):
        result = original(*args)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result
    monkeypatch.setattr(paper, "_dart_distribution", mutate)
    with pytest.raises(ValueError, match="changed during rendering"):
        paper._dart_archive_summary(path, evidence["input_index"]["files"], ("original__prior_samples",), 1.)


def test_dart_cli_requires_explicit_profile_and_complete_inputs(tmp_path, monkeypatch, capsys):
    arguments = [item for name in ("run-root", "refinement-root", "components-root", "decision-file",
                                  "numerical-root", "paper-dir")
                 for item in ("--" + name, str(tmp_path / name))]
    arguments += ["--expected-core-sha256", "a" * 64]
    extra = ["--dart-core-diagnostics-roots", str(tmp_path / "old-pilot"), str(tmp_path / "old-rest"),
             "--dart-stage-diagnostics-roots", str(tmp_path / "primary"), str(tmp_path / "optional"),
             "--dart-stage-root", str(tmp_path / "stage"), "--expected-dart-stage-sha256", "b" * 64,
             "--dart-selection-file", str(tmp_path / "admission.json")]
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {"combined_input_sha256": "f" * 64}
    monkeypatch.setattr(paper, "build_paper", fake)
    for invalid in ([*arguments, *extra], [*arguments, "--profile", "dart_extension"]):
        with pytest.raises(SystemExit) as error:
            paper.main(invalid)
        assert error.value.code == 2
    assert calls == []
    assert paper.main([*arguments, "--profile", "dart_extension", *extra]) == 0
    assert calls[0][1] == {
        "profile": "dart_extension", "dart_core_diagnostics_roots": [tmp_path / "old-pilot", tmp_path / "old-rest"],
        "dart_stage_root": tmp_path / "stage", "dart_stage_diagnostics_roots": [tmp_path / "primary", tmp_path / "optional"],
        "expected_dart_stage_sha256": "b" * 64, "dart_selection_file": tmp_path / "admission.json"}
    result = json.loads(capsys.readouterr().out)
    assert result["artifacts"] == [*paper.DART_EXTENSION_OUTPUTS, paper.EVIDENCE_NAME]


def test_cli_requires_all_inputs_and_only_reports_public_digest(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        paper.main([])
    assert error.value.code == 2
    calls = []
    def fake(*args):
        calls.append(args)
        return {"combined_input_sha256": "f" * 64}
    monkeypatch.setattr(paper, "build_paper", fake)
    arguments = []
    for name in ("run-root", "refinement-root", "components-root", "decision-file", "numerical-root", "paper-dir"):
        arguments += ["--" + name, str(tmp_path / name)]
    assert paper.main([*arguments, "--expected-core-sha256", "a" * 64]) == 0
    assert calls == [(tmp_path / "run-root", tmp_path / "paper-dir", "a" * 64,
                      tmp_path / "refinement-root", tmp_path / "components-root", tmp_path / "decision-file", tmp_path / "numerical-root")]
    result = json.loads(capsys.readouterr().out)
    assert result == {"combined_input_sha256": "f" * 64, "artifacts": [*paper.AUDIT_OUTPUTS, paper.EVIDENCE_NAME]}


@pytest.mark.parametrize("mutation, error", [
    (None, None),
    ("helper", "Current derivation source SHA256 differs: src/prigm_dart_paper_evidence.py"),
    ("metrics", "Current derivation source SHA256 differs: src/prigm_v2_metrics.py"),
    ("missing_pin", "Combined evidence requires --expected-combined-sha256"),
    ("wrong_pin", "Evidence differs from pinned combined SHA256"),
    ("resealed_numerical_driver", "Preserved numerical producer SHA256 differs"),
    ("resealed_numerical_metrics", "Preserved numerical producer SHA256 differs"),
    ("missing_asset", "Missing release file: generated/dart_block_rows.tex"),
    ("changed_asset", "Generated asset SHA256 differs: generated/dart_block_rows.tex"),
    ("profile", "Unknown combined profile or purpose"),
    ("schema", "Unknown combined profile or purpose"),
    ("numerical_driver_changed", None),
    ("numerical_driver_missing", None),
])
def test_dart_release_binds_current_derivation_and_historical_numerical_sources(
        bundle, dart_master, dart_public_fixture, tmp_path, monkeypatch, mutation, error):
    from test_prigm_release import paper as release_fixture, put, OLD, NEW
    import prigm_dart_paper_evidence as dart
    import verify_prigm_release as release

    public = public_fixture(bundle, dart_master)
    directory, commands, poppler_calls = release_fixture.__wrapped__(tmp_path, monkeypatch)
    for name in dict.fromkeys((*paper.DART_EXTENSION_SOURCE_IDS, *paper.NUMERICAL_REVIEWED_SOURCES.values())):
        source = paper.ROOT / name
        if name == "src/prigm_dart_paper_evidence.py" and not source.is_file():
            source = ROOT / name
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    # Independent reviewed bytes establish synthetic historical pins BEFORE any
    # tampering. Neither the archive nor a resealed public map is an authority.
    numerical_pins = {name: SHA(tmp_path / reviewed)
                      for name, reviewed in paper.NUMERICAL_REVIEWED_SOURCES.items()}
    assert public["numerical_validation"]["source_sha256"] == numerical_pins
    monkeypatch.setattr(paper, "FROZEN_NUMERICAL_SOURCE_SHA256", dict(numerical_pins))
    extension = deepcopy(dart_public_fixture)
    public.update(schema_version=3, profile="dart_extension", dart_extension=extension,
                  methods=dict(paper.DART_EXTENSION_METHODS))
    del public["dart_support"], public["dart_diagnostics"]
    index = public["input_index"]
    del index["dart_diagnostics"]
    index["source_sha256"] = {name: SHA(tmp_path / name) for name in paper.DART_EXTENSION_SOURCE_IDS}
    inner = extension["input_index"]
    inner["helper_sha256"] = index["source_sha256"]["src/prigm_dart_paper_evidence.py"]
    inner["reviewed_source_sha256"] = {name: index["source_sha256"]["src/" + name]
                                       for name in dart.REVIEWED_SOURCES}
    index["dart_extension"] = inner
    core_pin = public["core"]["run_manifest_sha256"]
    validation_calls = []
    # Release-boundary isolation only. The real combined validator and DART
    # scientific checks are tested separately; these assets and PDF are synthetic.
    def validate(received, expected_core_sha256):
        assert received == public
        assert received["schema_version"] == 3 and received["profile"] == "dart_extension"
        assert expected_core_sha256 == core_pin
        validation_calls.append(expected_core_sha256)
        return received
    monkeypatch.setattr(paper, "validate_combined_manifest", validate)
    if mutation in ("helper", "metrics"):
        name = "prigm_dart_paper_evidence.py" if mutation == "helper" else "prigm_v2_metrics.py"
        target = tmp_path / "src" / name
        target.write_bytes(target.read_bytes() + b"\n# changed current DART derivation\n")
    elif mutation in ("resealed_numerical_driver", "resealed_numerical_metrics"):
        name = paper.NUMERICAL_DRIVER if mutation == "resealed_numerical_driver" else "src/prigm_v2_metrics.py"
        public["numerical_validation"]["source_sha256"][name] = "0" * 64
        index["numerical_validation"]["source_sha256"][name] = "0" * 64
        assert public["numerical_validation"]["source_sha256"] == index["numerical_validation"]["source_sha256"]
        assert index["numerical_validation"]["source_sha256"] != numerical_pins
    elif mutation == "profile":
        public["profile"] = "corrected_audit"
    elif mutation == "schema":
        public["schema_version"] = 2
    elif mutation in ("numerical_driver_changed", "numerical_driver_missing"):
        name = paper.NUMERICAL_REVIEWED_SOURCES[paper.NUMERICAL_DRIVER]
        assert name not in index["source_sha256"]
        driver = tmp_path / name
        if mutation == "numerical_driver_changed":
            driver.write_bytes(driver.read_bytes() + b"\n# obsolete current producer changed\n")
            assert SHA(driver) != numerical_pins[paper.NUMERICAL_DRIVER]
        else:
            driver.unlink()
            assert not driver.exists()
    # Reseal the combined digest AND both visible bindings, so numerical-map
    # mutations must fail against the independent pin, not incidental staleness.
    combined_pin = paper.canonical_sha256(index)
    public["combined_input_sha256"] = combined_pin
    for name in paper.DART_EXTENSION_OUTPUTS:
        put(directory / name, "synthetic generated asset\n")
    put(directory / "generated/summary_values.tex", r"\newcommand{\PrigmCoreSHA}{" + core_pin + "}\n")
    put(directory / "generated/combined_binding.tex", r"\newcommand{\PrigmCombinedSHA}{" + combined_pin + "}\n")
    tex = (directory / "main.tex").read_text()
    tex = tex.replace(r"\usepackage[hidelinks]{hyperref}",
                      r"\input{generated/combined_binding.tex}" + "\n" + r"\usepackage[hidelinks]{hyperref}")
    tex = tex.replace(r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA}",
                      r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA; combined SHA256: \PrigmCombinedSHA}")
    extra = [name for name in paper.DART_EXTENSION_OUTPUTS
             if name not in release.OUTPUTS and name != "generated/combined_binding.tex"]
    includes = "\n".join((r"\input{" if name.endswith(".tex") else r"\includegraphics{")
                         + name + "}" for name in extra)
    put(directory / "main.tex", tex.replace(r"\includegraphics{figures/prior_samples.png}", includes))
    required = [name for name in (*release.SOURCES, *paper.DART_EXTENSION_OUTPUTS) if name != "references.bib"]
    put(directory / "main.fls", "PWD " + str(directory) + "\n"
        + "".join("INPUT ./" + name + "\n" for name in required) + "OUTPUT main.pdf\n", NEW)
    public["output_sha256"] = {name: SHA(directory / name) for name in paper.DART_EXTENSION_OUTPUTS}
    commands["pdfinfo"] = (f"Title: Fixture\nAuthor: \nSubject: PriGM core SHA256: {core_pin}; "
                           f"combined SHA256: {combined_pin}\nPages: 6\n")
    expected_pin = None if mutation == "missing_pin" else "0" * 64 if mutation == "wrong_pin" else combined_pin
    if mutation == "missing_asset":
        (directory / "generated/dart_block_rows.tex").unlink()
    elif mutation == "changed_asset":
        put(directory / "generated/dart_block_rows.tex", "changed DART asset\n")
    put(directory / release.EVIDENCE, json.dumps(public), OLD)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            release.verify_release(directory, core_pin, expected_pin)
        assert poppler_calls == []
    else:
        result = release.verify_release(directory, core_pin, expected_pin)
        assert result["schema_version"] == 3 and result["profile"] == "dart_extension"
        assert result["run_manifest_sha256"] == core_pin
        assert result["combined_input_sha256"] == combined_pin
        assert result["generated_sha256"] == public["output_sha256"]
        assert set(result["generated_sha256"]) == set(paper.DART_EXTENSION_OUTPUTS)
        assert len(result["generated_sha256"]) == 21
        assert result["derivation_source_sha256"] == index["source_sha256"]
        assert result["numerical_source_sha256"] == numerical_pins
        assert result["pdf"]["pages"] == 6 and result["pdf"]["main_pages"] == 4
        assert all(result["verification"][name] is True for name in (
            "subject_core_bound", "subject_combined_bound", "author_metadata_blank", "first_page_anonymous",
            "fonts_embedded", "recorder_inputs_bound", "dependencies_not_newer_than_pdf"))
        assert [args[0] for args, _ in poppler_calls] == ["pdfinfo", "pdftotext", "pdffonts"]
        assert not (directory / release.MANIFEST).exists()
    early_rejection = mutation in ("missing_pin", "wrong_pin", "profile", "schema")
    assert validation_calls == ([] if early_rejection else [core_pin])
    assert paper.FROZEN_NUMERICAL_SOURCE_SHA256 == numerical_pins


@pytest.mark.parametrize("mutation", [None, "missing_pin", "wrong_pin", "profile", "source",
                                      "numerical_driver_changed", "numerical_driver_missing",
                                      "asset", "macro", "subject", "stale", "extra_output"])
def test_combined_release_binds_both_pins_sources_and_all_assets(bundle, dart_master, tmp_path, monkeypatch, mutation):
    from test_prigm_release import paper as release_fixture, put, OLD, NEW
    import verify_prigm_release as release

    public = public_fixture(bundle, dart_master)
    directory, commands, _ = release_fixture.__wrapped__(tmp_path, monkeypatch)
    for name in dict.fromkeys((*paper.SOURCE_IDS, *paper.NUMERICAL_REVIEWED_SOURCES.values())):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((paper.ROOT / name).read_bytes())
    core_pin = public["core"]["run_manifest_sha256"]
    combined_pin = public["combined_input_sha256"]
    for name in paper.AUDIT_OUTPUTS:
        put(directory / name, "synthetic generated asset\n")
    put(directory / "generated/summary_values.tex", r"\newcommand{\PrigmCoreSHA}{" + core_pin + "}\n")
    put(directory / "generated/combined_binding.tex", r"\newcommand{\PrigmCombinedSHA}{" + combined_pin + "}\n")
    tex = (directory / "main.tex").read_text()
    tex = tex.replace(r"\usepackage[hidelinks]{hyperref}",
                      r"\input{generated/combined_binding.tex}" + "\n" + r"\usepackage[hidelinks]{hyperref}")
    tex = tex.replace(r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA}",
                      r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA; combined SHA256: \PrigmCombinedSHA}")
    extra = [name for name in paper.AUDIT_OUTPUTS
             if name not in release.OUTPUTS and name != "generated/combined_binding.tex"]
    includes = "\n".join((r"\input{" if name.endswith(".tex") else r"\includegraphics{")
                           + name + "}" for name in extra)
    tex = tex.replace(r"\includegraphics{figures/prior_samples.png}", includes)
    put(directory / "main.tex", tex)
    required = [name for name in (*release.SOURCES, *paper.AUDIT_OUTPUTS) if name != "references.bib"]
    put(directory / "main.fls", "PWD " + str(directory) + "\n"
        + "".join("INPUT ./" + name + "\n" for name in required) + "OUTPUT main.pdf\n", NEW)
    public["output_sha256"] = {name: SHA(directory / name) for name in paper.AUDIT_OUTPUTS}
    commands["pdfinfo"] = (f"Title: Fixture\nAuthor: \nSubject: PriGM core SHA256: {core_pin}; "
                           f"combined SHA256: {combined_pin}\nPages: 6\n")
    expected_pin = combined_pin
    if mutation == "missing_pin":
        expected_pin = None
    elif mutation == "wrong_pin":
        expected_pin = "0" * 64
    elif mutation == "profile":
        public["profile"] = "unknown"
    elif mutation == "source":
        with (tmp_path / "src/prigm_v2_metrics.py").open("a") as stream:
            stream.write("\n# changed derivation\n")
    elif mutation in ("numerical_driver_changed", "numerical_driver_missing"):
        driver = tmp_path / paper.NUMERICAL_REVIEWED_SOURCES[paper.NUMERICAL_DRIVER]
        if mutation == "numerical_driver_changed":
            driver.write_bytes(driver.read_bytes() + b"\n# changed reviewed numerical driver\n")
        else:
            driver.unlink()
    elif mutation == "asset":
        put(directory / "generated/component_rows.tex", "changed")
    elif mutation == "macro":
        put(directory / "generated/combined_binding.tex", r"\newcommand{\PrigmCombinedSHA}{" + "0" * 64 + "}\n")
        public["output_sha256"]["generated/combined_binding.tex"] = SHA(directory / "generated/combined_binding.tex")
    elif mutation == "subject":
        commands["pdfinfo"] = commands["pdfinfo"].replace(combined_pin, "0" * 64)
    elif mutation == "stale":
        os.utime(directory / "generated/component_rows.tex", ns=(NEW, NEW))
    elif mutation == "extra_output":
        public["output_sha256"]["generated/unsupported.tex"] = "0" * 64
    put(directory / release.EVIDENCE, json.dumps(public), OLD)
    if mutation is not None:
        with pytest.raises(ValueError):
            release.verify_release(directory, core_pin, expected_pin)
    else:
        result = release.verify_release(directory, core_pin, expected_pin)
        assert result["schema_version"] == 2
        assert result["combined_input_sha256"] == combined_pin
        assert result["verification"]["subject_combined_bound"] is True
        assert set(result["generated_sha256"]) == set(paper.AUDIT_OUTPUTS)
        assert len(result["generated_sha256"]) == 14
        assert result["numerical_source_sha256"] == public["input_index"]["numerical_validation"]["source_sha256"]
        assert result["pdf"]["main_pages"] == 4


@pytest.mark.parametrize("fresh_count", [1, 2, 3])
def test_dart_build_requires_seven_records_before_input_index(tmp_path, monkeypatch, fresh_count):
    import prigm_dart_paper_evidence as dart
    records = dict.fromkeys((*dart.OLD, dart.PRIMARY, *dart.OPTIONAL[:fresh_count - 1]))
    # Isolate the build boundary after private verification; the helper's
    # generic validator intentionally also accepts the primary-only subset.
    monkeypatch.setattr(dart, "verify_dart_extension", lambda *args: (
        {"records": records, "input_index": {}}, {}))
    monkeypatch.setattr(paper, "_numerical_snapshot", lambda *args, **kwargs: {})
    decision = tmp_path / "decision.json"
    decision.write_text("{}")
    destination = tmp_path / "paper"

    def reached_input_index(*args, **kwargs):
        raise RuntimeError("complete records reached input index")

    monkeypatch.setattr(paper, "_input_index", reached_input_index)
    error = RuntimeError if fresh_count == 3 else ValueError
    message = "complete records reached input index" if fresh_count == 3 else "requires all seven DART records"
    with pytest.raises(error, match=message):
        paper.build_paper(
            tmp_path, destination, "a" * 64, tmp_path, tmp_path, decision, tmp_path,
            profile="dart_extension", dart_core_diagnostics_roots=[tmp_path],
            dart_stage_root=tmp_path, dart_stage_diagnostics_roots=[tmp_path],
            expected_dart_stage_sha256="b" * 64, dart_selection_file=decision)
    assert not destination.exists()
