"""CPU-only protocol, provenance, restart and paired-window refinement tests."""
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prigm_experiment as core
import prigm_refine_evaluation as refine

GPU = "GPU-12345678-1234-1234-1234-123456789abc"
RUNTIME = {"gpu_uuid": GPU, "pid": 123, "placement": "/device:GPU:0",
           "deterministic_ops_requested": True, "process_inventory": ["private inventory"],
           "tensorflow": "fake", "python": "fake", "numpy": np.__version__}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Real hash verifiers, synthetic files, full frozen 60-slot matrix, no TensorFlow."""
    core_root = tmp_path / "core"
    output_root = tmp_path / "refinement"
    (core_root / "source" / "src").mkdir(parents=True)
    (core_root / "prepared").mkdir()
    (core_root / "jobs").mkdir()
    for name in core.CODE_FILES:
        shutil.copyfile(ROOT / name, core_root / "source" / name)
    for dataset, count in zip(core.DATASETS, (35, 3, 8, 1, 33)):
        np.savez_compressed(core_root / "prepared" / f"{dataset}.npz",
                            test=np.arange(count * 192, dtype=float).reshape(count, 192),
                            starts_test=np.arange(count, dtype=np.int64) * 192,
                            base_variance=np.array(0.25))
    write_json(core_root / "prepared" / "metadata.json", {})
    manifest = {
        "schema_version": 1, "purpose": "core", "datasets": list(core.DATASETS),
        "specification": core.specification("core"),
        "code_sha256": {name: core.sha256(ROOT / name) for name in core.CODE_FILES},
        "prepared_sha256": {p.name: core.sha256(p) for p in (core_root / "prepared").iterdir()},
    }
    write_json(core_root / "run_manifest.json", manifest)
    monkeypatch.setattr(refine, "CORE_MANIFEST_SHA256", core.sha256(core_root / "run_manifest.json"))
    # Only the temporary source copy is mutated in source-integrity tests.
    source = tmp_path / "refiner.py"
    shutil.copyfile(refine.SOURCE_FILE, source)
    monkeypatch.setattr(refine, "SOURCE_FILE", source)
    for dataset, arm, seed in refine.canonical_jobs():
        identifier = core.job_id(dataset, arm, seed)
        attempt = core_root / "jobs" / identifier / ("attempt-" + "a" * 32)
        attempt.mkdir(parents=True)
        record = {"dataset": dataset, "arm": arm, "seed": seed,
                  "run_manifest_sha256": refine.CORE_MANIFEST_SHA256,
                  "specification": core.specification("core")}
        for name in core.JOB_ARTIFACTS:
            write_json(attempt / name, record)
        count = len(core.load_prepared(core_root, dataset)["test"])
        nll = np.arange(count, dtype=float) + core.ARMS.index(arm) * (core.SEEDS.index(seed) + 1)
        nll[32:] += 1e6  # Full-test averaging must not leak into the prefix comparison.
        np.savez_compressed(attempt / "evaluation.npz", log_score=-192 * nll,
                            ess=np.full(count, 2.0), stability_log_score=np.full(min(32, count), 999.0))
        write_json(attempt.parent / "complete.json", {
            "attempt": attempt.name, "run_manifest_sha256": refine.CORE_MANIFEST_SHA256,
            "artifact_sha256": {name: core.sha256(attempt / name) for name in core.JOB_ARTIFACTS},
        })
    refine.prepare(core_root, output_root)
    return SimpleNamespace(core=core_root, output=output_root,
                           manifest=refine.verify_refinement(core_root, output_root), source=source)


def completed_job(campaign, job=("gaussian", "A", 42)):
    identifier = core.job_id(*job)
    core_attempt = core.verified_completion(campaign.core, identifier)
    arrays = refine.original_arrays(core_attempt, core.load_prepared(campaign.core, job[0]))
    for k, shift in ((4096, 0.125), (16384, 0.25)):
        arrays[f"log_score_{k}"] = arrays["log_score_256"] + 192 * shift
        arrays[f"ess_{k}"] = np.full(len(arrays["window_index"]), k / 4)
    attempt = campaign.output / "jobs" / identifier / ("attempt-" + "b" * 32)
    attempt.mkdir(parents=True)
    np.savez_compressed(attempt / "evaluation.npz", **arrays)
    summary = refine.job_summary(campaign.output, campaign.manifest, identifier, arrays,
                                 dict(RUNTIME, elapsed_seconds=1.0))
    write_json(attempt / "summary.json", summary)
    write_json(attempt.parent / "complete.json", {
        "schema_version": 1, "core_job_id": identifier, "attempt": attempt.name,
        "refinement_manifest_sha256": core.sha256(campaign.output / refine.MANIFEST),
        "artifact_sha256": {name: core.sha256(attempt / name) for name in refine.ARTIFACTS},
    })
    return attempt


def rehash(attempt, name):
    marker = attempt.parent / "complete.json"
    record = core.read_json(marker)
    record["artifact_sha256"][name] = core.sha256(attempt / name)
    write_json(marker, record)


def verify_one(campaign, identifier="gaussian_A_42"):
    return refine.verify_job(campaign.core, campaign.output, campaign.manifest, identifier)


def test_frozen_protocol_and_canonical_four_shards():
    assert refine.CORE_MANIFEST_SHA256 == "37cc7445600eee541ee50aa580f6fdd6c0f37cf420074dbecc635c8d202946f5"
    spec = refine.specification()
    assert spec["sample_counts"] == [4096, 16384]
    assert spec["sampling_seeds"] == [44001, 44002]
    assert spec["max_test_windows"] == 32 and spec["chunk"] == 4
    assert spec["original_samples"] == 256 and spec["original_sampling_seed"] == 42001
    assert spec["checkpoint"] == "best.weights.h5"
    jobs = refine.canonical_jobs()
    assert len(jobs) == len(set(jobs)) == 60
    assert jobs[:5] == [("gaussian", a, 42) for a in "ABCD"] + [("gaussian", "A", 1729)]
    assert jobs[-1] == ("52403t2020", "D", 31415)
    assert [len(refine.shard_jobs(i)) for i in range(4)] == [15] * 4
    assert set().union(*(set(refine.shard_jobs(i)) for i in range(4))) == set(jobs)
    for index in (-1, 4, True, 1.0):
        with pytest.raises(ValueError):
            refine.shard_jobs(index)


def test_import_and_cli_do_not_load_tensorflow():
    script = ("import sys; sys.path.insert(0, " + repr(str(ROOT / "src")) + "); "
              "import prigm_refine_evaluation; "
              "assert 'tensorflow' not in sys.modules; assert 'prigm_model' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", script], check=True)
    help_result = subprocess.run([sys.executable, "-B", str(refine.SOURCE_FILE), "--help"],
                                 check=True, capture_output=True, text=True)
    assert "prepare,worker,summarize" in help_result.stdout
    tree = ast.parse(refine.SOURCE_FILE.read_text())
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"fit", "train", "GradientTape", "optimizers", "save_weights", "apply_gradients"}


def test_prepare_binds_all_inputs_and_requires_fresh_output(campaign):
    manifest = campaign.manifest
    assert len(manifest["jobs"]) == 60
    assert manifest["jobs"]["gaussian_A_42"]["count"] == 32
    assert manifest["jobs"]["23461t2022_B_1729"]["count"] == 3
    assert manifest["jobs"]["46407t2011_D_31415"]["count"] == 1
    assert core.sha256(campaign.output / "source" / refine.SOURCE_NAME) == core.sha256(campaign.source)
    with pytest.raises(FileExistsError):
        refine.prepare(campaign.core, campaign.output)
    with pytest.raises(ValueError, match="outside"):
        refine.prepare(campaign.core, campaign.core / "forbidden-refinement")
    assert not (campaign.core / "forbidden-refinement").exists()


@pytest.mark.parametrize("mutation", ["samples", "seeds", "chunk", "float_chunk", "boolean_type",
                                      "selection", "job", "hash", "source_hash"])
def test_rejects_manifest_protocol_or_identity_changes(campaign, mutation):
    changed = copy.deepcopy(campaign.manifest)
    if mutation == "samples":
        changed["specification"]["sample_counts"][0] = 1024
    elif mutation == "seeds":
        changed["specification"]["sampling_seeds"].reverse()
    elif mutation == "chunk":
        changed["specification"]["chunk"] = 32
    elif mutation == "float_chunk":
        changed["specification"]["chunk"] = 4.0
    elif mutation == "boolean_type":
        changed["specification"]["training_or_checkpoint_selection"] = 0
    elif mutation == "selection":
        changed["specification"]["window_selection"] = "random_test_windows"
    elif mutation == "job":
        changed["jobs"].pop("gaussian_A_42")
    elif mutation == "hash":
        changed["jobs"]["gaussian_A_42"]["core_checkpoint_sha256"] = "0" * 64
    else:
        changed["code_sha256"][refine.SOURCE_NAME] = "0" * 64
    write_json(campaign.output / refine.MANIFEST, changed)
    with pytest.raises(ValueError, match="manifest differs"):
        refine.verify_refinement(campaign.core, campaign.output)


@pytest.mark.parametrize("target", ["core_manifest", "core_source", "prepared", "own_source", "snapshot"])
def test_rejects_changed_frozen_inputs(campaign, target):
    paths = {"core_manifest": campaign.core / "run_manifest.json",
             "core_source": campaign.core / "source" / core.CODE_FILES[0],
             "prepared": campaign.core / "prepared" / "gaussian.npz",
             "own_source": campaign.source,
             "snapshot": campaign.output / "source" / refine.SOURCE_NAME}
    with paths[target].open("ab") as stream:
        stream.write(b"\nchanged")
    with pytest.raises(ValueError):
        refine.verify_refinement(campaign.core, campaign.output)


@pytest.mark.parametrize("name", ["complete.json", "best.weights.h5", "evaluation.npz"])
@pytest.mark.parametrize("operation", ["delete", "change"])
def test_rejects_missing_or_changed_core_artifacts(campaign, name, operation):
    job = campaign.core / "jobs" / "gaussian_A_42"
    path = job / name if name == "complete.json" else job / ("attempt-" + "a" * 32) / name
    if operation == "delete":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"\nchanged")
    with pytest.raises(ValueError):
        refine.verify_refinement(campaign.core, campaign.output)


def test_rehashed_core_checkpoint_still_rejects_bound_identity(campaign):
    attempt = core.verified_completion(campaign.core, "gaussian_A_42")
    (attempt / "best.weights.h5").write_bytes(b"different checkpoint")
    rehash(attempt, "best.weights.h5")
    assert core.verified_completion(campaign.core, "gaussian_A_42") == attempt
    with pytest.raises(ValueError, match="core identity"):
        refine.verify_refinement(campaign.core, campaign.output)


def test_complete_identity_and_original_prefix_not_stability_scores(campaign):
    attempt = completed_job(campaign)
    verified, summary, arrays = verify_one(campaign)
    assert verified == attempt
    assert summary["count"] == 32
    assert summary["core_job_id"] == "gaussian_A_42"
    np.testing.assert_array_equal(arrays["log_score_256"], -192 * np.arange(32))
    assert summary["metrics"]["256"]["mean_nll_nats_per_dimension"] == 15.5
    assert summary["metrics"]["4096"]["mean_log_score_change_from_256_per_dimension"] == 0.125
    assert summary["metrics"]["16384"]["median_ess"] == 4096
    assert summary["runtime_private"]["gpu_uuid"] == GPU


@pytest.mark.parametrize("name", refine.ARTIFACTS)
@pytest.mark.parametrize("operation", ["delete", "change", "symlink"])
def test_rejects_missing_changed_or_symlinked_refinement_artifact(campaign, name, operation):
    attempt = completed_job(campaign)
    path = attempt / name
    if operation == "delete":
        path.unlink()
    elif operation == "change":
        with path.open("ab") as stream:
            stream.write(b"changed")
    else:
        relocated = attempt / (name + ".moved")
        path.rename(relocated)
        path.symlink_to(relocated)
    with pytest.raises(ValueError):
        verify_one(campaign)


@pytest.mark.parametrize("field", ["core_job_id", "refinement_manifest_sha256", "attempt", "artifact_sha256"])
def test_rejects_invalid_completion_marker(campaign, field):
    attempt = completed_job(campaign)
    marker = core.read_json(attempt.parent / "complete.json")
    marker[field] = {"core_job_id": "gaussian_B_42", "refinement_manifest_sha256": "0" * 64,
                     "attempt": "../escape", "artifact_sha256": {}}[field]
    write_json(attempt.parent / "complete.json", marker)
    with pytest.raises(ValueError, match="completion identity"):
        verify_one(campaign)


@pytest.mark.parametrize("field", ["core_job_id", "count", "core_evaluation_sha256", "specification", "metrics", "runtime_private"])
def test_rehashed_summary_must_match_identity_settings_and_math(campaign, field):
    attempt = completed_job(campaign)
    summary = core.read_json(attempt / "summary.json")
    summary[field] = {"core_job_id": "gaussian_B_42", "count": 31,
                      "core_evaluation_sha256": "0" * 64, "specification": {}, "metrics": {},
                      "runtime_private": {}}[field]
    write_json(attempt / "summary.json", summary)
    rehash(attempt, "summary.json")
    with pytest.raises(ValueError):
        verify_one(campaign)


@pytest.mark.parametrize("mutation", ["baseline", "order", "short", "nan", "ess", "missing"])
def test_rehashed_npz_must_preserve_windows_baseline_shapes_and_finite_values(campaign, mutation):
    attempt = completed_job(campaign)
    with np.load(attempt / "evaluation.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if mutation == "baseline":
        arrays["log_score_256"][0] += 1
    elif mutation == "order":
        arrays["test_starts"] = arrays["test_starts"][::-1]
    elif mutation == "short":
        arrays["log_score_4096"] = arrays["log_score_4096"][:-1]
    elif mutation == "nan":
        arrays["log_score_16384"][0] = np.nan
    elif mutation == "ess":
        arrays["ess_4096"][0] = 4097
    else:
        arrays.pop("ess_256")
    np.savez_compressed(attempt / "evaluation.npz", **arrays)
    rehash(attempt, "evaluation.npz")
    with pytest.raises(ValueError):
        verify_one(campaign)


def fake_execution(monkeypatch):
    loaded, calls = [], []

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["input_dim"] == 192 and kwargs["latent_dim"] == 16
            assert kwargs["base_variance"] == 0.25

        def load_weights(self, path):
            loaded.append(path)
            assert Path(path).name == "best.weights.h5"

    tf = SimpleNamespace(keras=SimpleNamespace(
        backend=SimpleNamespace(clear_session=lambda: None),
        utils=SimpleNamespace(set_random_seed=lambda seed: None)))

    def score(model, x, k, seed, chunk):
        calls.append((x.copy(), k, seed, chunk))
        return {"log_score": np.full(len(x), -192.0), "ess": np.full(len(x), 1.0)}

    monkeypatch.setattr(core, "importance_evaluate", score)
    return Model, tf, loaded, calls


@pytest.mark.parametrize("dataset,count", [("gaussian", 32), ("23461t2022", 3)])
def test_job_restart_uses_fresh_attempt_and_verified_reuse_never_evaluates(
        campaign, monkeypatch, capsys, dataset, count):
    job = (dataset, "B", 1729)
    identifier = core.job_id(*job)
    incomplete = campaign.output / "jobs" / identifier / ("attempt-" + "c" * 32)
    incomplete.mkdir(parents=True)
    (incomplete / "evaluation.npz").write_bytes(b"interrupted")
    assert verify_one(campaign, identifier) is None
    Model, tf, loaded, calls = fake_execution(monkeypatch)
    refine.run_job(campaign.core, campaign.output, campaign.manifest, job, RUNTIME, Model, tf)
    attempt, _, _ = verify_one(campaign, identifier)
    assert attempt != incomplete and (incomplete / "evaluation.npz").read_bytes() == b"interrupted"
    assert "RESTART" in capsys.readouterr().out
    assert len(loaded) == 1
    assert [(len(x), k, seed, chunk) for x, k, seed, chunk in calls] == [
        (count, 4096, 44001, 4), (count, 16384, 44002, 4)]
    np.testing.assert_array_equal(calls[0][0], core.load_prepared(campaign.core, dataset)["test"][:count])
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    marker_hash = core.sha256(attempt.parent / "complete.json")
    refine.run_job(campaign.core, campaign.output, campaign.manifest, job, RUNTIME, Model, tf)
    assert len(calls) == 2 and len(loaded) == 1
    assert core.sha256(attempt.parent / "complete.json") == marker_hash
    (attempt / "evaluation.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        refine.run_job(campaign.core, campaign.output, campaign.manifest, job, RUNTIME, Model, tf)
    assert len(calls) == 2


def test_worker_requests_gpu_attestation_and_dispatches_only_its_shard(campaign, monkeypatch):
    dispatched = []
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    for name in ("CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS", "TF_CPP_MIN_LOG_LEVEL",
                 "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(name, "must-be-replaced-before-runtime")
    tf = SimpleNamespace()
    model = object()

    def runtime(gpu):
        assert gpu == GPU and os.environ["CUDA_VISIBLE_DEVICES"] == GPU
        assert os.environ["TF_DETERMINISTIC_OPS"] == "1"
        monkeypatch.setitem(sys.modules, "tensorflow", tf)
        monkeypatch.setitem(sys.modules, "prigm_model", SimpleNamespace(GaussianVAE=model))
        return RUNTIME

    monkeypatch.setattr(core, "gpu_runtime", runtime)
    monkeypatch.setattr(refine, "run_job", lambda c, o, m, j, r, model_class, tensorflow:
                        dispatched.append((j, r, model_class, tensorflow)))
    refine.worker(campaign.core, campaign.output, 2, GPU)
    assert [record[0] for record in dispatched] == refine.canonical_jobs()[2::4]
    assert all(record[1:] == (RUNTIME, model, tf) for record in dispatched)


@pytest.mark.parametrize("failed_budget", [4096, 16384])
def test_failed_evaluation_publishes_no_completion(campaign, monkeypatch, failed_budget):
    Model, tf, _, _ = fake_execution(monkeypatch)
    original = core.importance_evaluate

    def fail(model, x, k, seed, chunk):
        if k == failed_budget:
            raise RuntimeError("synthetic evaluation failure")
        return original(model, x, k, seed, chunk)

    monkeypatch.setattr(core, "importance_evaluate", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        refine.run_job(campaign.core, campaign.output, campaign.manifest,
                       ("gaussian", "A", 42), RUNTIME, Model, tf)
    assert verify_one(campaign) is None
    assert not (campaign.output / "jobs" / "gaussian_A_42" / "complete.json").exists()


def test_concurrent_writer_fails_before_creating_attempt(campaign, monkeypatch):
    Model, tf, loaded, calls = fake_execution(monkeypatch)
    job_root = campaign.output / "jobs" / "gaussian_A_42"
    job_root.mkdir()
    with (job_root / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            refine.run_job(campaign.core, campaign.output, campaign.manifest,
                           ("gaussian", "A", 42), RUNTIME, Model, tf)
    assert not loaded and not calls and not list(job_root.glob("attempt-*"))


@pytest.mark.parametrize("bad_state", ["source", "completion", "worker_index", "gpu_uuid", "tensorflow"])
def test_worker_rejects_bad_state_before_gpu_initialization(campaign, monkeypatch, bad_state):
    def forbidden(*args):
        pytest.fail("GPU initialization must not run after failed preflight")

    monkeypatch.setattr(core, "gpu_runtime", forbidden)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    index, gpu = 0, GPU
    if bad_state == "source":
        campaign.source.write_text("changed")
    elif bad_state == "completion":
        (campaign.core / "jobs" / "gaussian_A_42" / "complete.json").unlink()
    elif bad_state == "worker_index":
        index = 4
    elif bad_state == "gpu_uuid":
        gpu = "0,1"
    else:
        monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace())
    with pytest.raises((ValueError, RuntimeError)):
        refine.worker(campaign.core, campaign.output, index, gpu)


def test_dangling_completion_marker_is_not_an_incomplete_job(campaign):
    job_root = campaign.output / "jobs" / "gaussian_A_42"
    job_root.mkdir()
    (job_root / "complete.json").symlink_to(job_root / "nonexistent")
    with pytest.raises(ValueError, match="regular artifact"):
        verify_one(campaign)


def test_paired_math_uses_seed_differences_not_unpaired_medians():
    records = {}
    # Median(B-A)=1, but median(B)-median(A)=2: a deliberate pairing trap.
    nll = {"A": [0, 100, 101], "B": [1, 200, 102], "C": [3, 103, 104], "D": [-1, 198, 98]}
    for dataset, arm, seed in refine.canonical_jobs():
        arrays = {"window_index": np.array([0, 1]), "test_starts": np.array([5, 197])}
        for k, multiplier in zip(refine.BUDGETS, (1, 2, 4)):
            mean = nll[arm][core.SEEDS.index(seed)] * multiplier
            arrays[f"log_score_{k}"] = -192 * np.array([mean - 10.0, mean + 10.0])
        records[core.job_id(dataset, arm, seed)] = arrays
    results = refine.paired_comparisons(records)["gaussian"]
    for k, multiplier in zip(refine.BUDGETS, (1, 2, 4)):
        assert results["B-A"][str(k)]["paired_seed_deltas"] == [multiplier, 100 * multiplier, multiplier]
        assert results["B-A"][str(k)]["median"] == multiplier
        assert results["B-A"][str(k)]["range"] == [multiplier, 100 * multiplier]
        assert results["C-A"][str(k)]["median"] == 3 * multiplier
        assert results["B-D"][str(k)]["range"] == [2 * multiplier, 4 * multiplier]
    records["gaussian_B_42"]["test_starts"] = np.array([197, 5])
    with pytest.raises(ValueError, match="same ordered windows"):
        refine.paired_comparisons(records)
    records.pop("gaussian_A_42")
    with pytest.raises(ValueError, match="exactly the 60"):
        refine.paired_comparisons(records)


def test_summarize_requires_all60_sanitizes_and_archives_exact_artifacts(campaign):
    with pytest.raises(ValueError, match="Missing completed refinement"):
        refine.summarize(campaign.core, campaign.output)
    assert not (campaign.output / "results.json").exists()
    for job in refine.canonical_jobs():
        completed_job(campaign, job)
    output = refine.summarize(campaign.core, campaign.output)
    assert len(output["jobs"]) == 60
    assert output["paired_comparisons"]["gaussian"]["B-A"]["256"]["median"] == 2.0
    assert output["jobs"][0]["metrics"]["256"]["mean_nll_nats_per_dimension"] == 15.5
    text = (campaign.output / "results.json").read_text()
    for private in (GPU, "runtime_private", "process_inventory", "private inventory", str(campaign.core)):
        assert private not in text
    archive = core.read_json(campaign.output / "archive_manifest.json")
    assert len(archive["artifact_sha256"]) == 183  # manifest, source, results, 60 * 3 artifacts
    for name, expected in archive["artifact_sha256"].items():
        assert not Path(name).is_absolute() and core.sha256(campaign.output / name) == expected
    initial = core.sha256(campaign.output / "results.json")
    assert refine.summarize(campaign.core, campaign.output) == output
    assert core.sha256(campaign.output / "results.json") == initial
    archive["artifact_sha256"]["results.json"] = "0" * 64
    write_json(campaign.output / "archive_manifest.json", archive)
    with pytest.raises(ValueError, match="archive identity"):
        refine.summarize(campaign.core, campaign.output)


def test_json_publication_is_exclusive_atomic_and_rejects_nan(tmp_path):
    path = tmp_path / "complete.json"
    with pytest.raises(ValueError):
        refine.publish_json(path, {"score": float("nan")})
    assert not path.exists() and not list(tmp_path.iterdir())
    refine.publish_json(path, {"score": 1})
    with pytest.raises(FileExistsError):
        refine.publish_json(path, {"score": 2})
    assert core.read_json(path) == {"score": 1}
    assert list(tmp_path.iterdir()) == [path]
