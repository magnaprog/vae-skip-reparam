"""Check experiment execution, saved files and the importance evaluator."""
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    implementation = tmp_path / "implementation"
    run = tmp_path / "run"
    for name in experiment.CODE_FILES:
        for directory in (implementation, run / "source"):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}\n")
    (run / "prepared").mkdir()
    for name in ["metadata.json", *(f"{dataset}.npz" for dataset in experiment.DATASETS)]:
        (run / "prepared" / name).write_bytes(b"input identity fixture")
    manifest = {
        "schema_version": 2,
        "purpose": "smoke",
        "datasets": list(experiment.DATASETS),
        "specification": experiment.specification("smoke"),
        "code_sha256": {name: experiment.sha256(implementation / name)
                        for name in experiment.CODE_FILES},
        "prepared_sha256": {path.name: experiment.sha256(path)
                            for path in (run / "prepared").iterdir()},
    }
    experiment.atomic_json(run / "run_manifest.json", manifest)
    monkeypatch.setattr(experiment, "ROOT", implementation)
    return run, implementation, manifest


def test_verify_run_accepts_consistent_source_and_inputs(prepared_run):
    run, _, manifest = prepared_run
    assert experiment.verify_run(run) == manifest


@pytest.mark.parametrize("schema_version", [None, 1, 3])
def test_verify_run_rejects_unsupported_schema_before_loading_inputs(tmp_path, schema_version):
    experiment.atomic_json(tmp_path / "run_manifest.json", {"schema_version": schema_version})
    with pytest.raises(ValueError, match="schema_version=2.*fresh run"):
        experiment.verify_run(tmp_path)


def test_verify_run_rejects_unsupported_specification_field(prepared_run):
    run, _, manifest = prepared_run
    manifest["specification"] = dict(manifest["specification"])
    manifest["specification"]["unsupported_conditions"] = manifest["specification"].pop("conditions")
    experiment.atomic_json(run / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="Manifest differs"):
        experiment.verify_run(run)


def test_prepare_writes_and_verifies_condition_manifest(tmp_path, monkeypatch):
    train = np.random.default_rng(17).standard_normal((32, 192))
    data = SimpleNamespace(
        train=train, validation=train[:8], test=train[8:16], mean=0.0, scale=1.0,
        starts={}, oracle_covariance=None, metadata={"fixture": "synthetic"},
    )
    monkeypatch.setattr(experiment, "prepare_dataset", lambda name, root: data)
    root = tmp_path / "prepared-run"
    experiment.prepare(root, purpose="smoke")
    manifest = experiment.verify_run(root)
    assert manifest["schema_version"] == 2
    assert manifest["specification"]["conditions"] == [
        "no_skip_base", "skip_base", "no_skip_high", "skip_high",
    ]


@pytest.mark.parametrize("changed", ["implementation", "saved_source", "input", "settings"])
def test_verify_run_rejects_changed_source_inputs_or_settings(prepared_run, changed):
    run, implementation, manifest = prepared_run
    if changed == "implementation":
        (implementation / experiment.CODE_FILES[0]).write_text("changed source")
        error = "Experiment source changed"
    elif changed == "saved_source":
        (run / "source" / experiment.CODE_FILES[0]).write_text("changed source")
        error = "Saved experiment source changed"
    elif changed == "input":
        (run / "prepared" / "gaussian.npz").write_bytes(b"changed input")
        error = "Prepared dataset changed"
    else:
        manifest["specification"]["max_steps"] += 1
        experiment.atomic_json(run / "run_manifest.json", manifest)
        error = "Manifest differs"
    with pytest.raises(ValueError, match=error):
        experiment.verify_run(run)


class IndependentGaussian:
    latent_dim = 2
    b = 0.2
    residual_variance = 0.04

    def encode(self, x):
        return np.zeros((len(x), self.latent_dim)), np.zeros((len(x), self.latent_dim))

    def decode_residual(self, z):
        return np.zeros((len(z), 3))


def test_importance_evaluator_includes_bypass_jacobian():
    x = np.array([[1.0, 0.0, -0.5], [0.2, 0.4, 0.8]])
    output = experiment.importance_evaluate(IndependentGaussian(), x, samples=32, seed=3)
    np.testing.assert_allclose(output["log_score"], norm.logpdf(x).sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(output["ess"], 32, atol=1e-10)


def test_importance_evaluation_does_not_depend_on_chunk_for_constant_weights():
    x = np.arange(21).reshape(7, 3) / 10
    first = experiment.importance_evaluate(IndependentGaussian(), x, 32, 42, chunk=1)
    second = experiment.importance_evaluate(IndependentGaussian(), x, 32, 42, chunk=7)
    np.testing.assert_allclose(first["log_score"], second["log_score"])


def test_smoke_and_core_have_separate_settings():
    core = experiment.specification("core")
    smoke = experiment.specification("smoke")
    assert core["max_steps"] == 10000
    assert smoke["max_steps"] == 4
    assert core["conditions"] == ["no_skip_base", "skip_base", "no_skip_high", "skip_high"]
    assert smoke["conditions"] == core["conditions"]
    with pytest.raises(ValueError):
        experiment.specification("unsupported")


def test_completion_rejects_tampered_artifact(tmp_path):
    experiment.atomic_json(tmp_path / "run_manifest.json", {"example": 1})
    root = tmp_path / "jobs" / "gaussian_no_skip_base_42"
    directory = root / "attempt-1"
    directory.mkdir(parents=True)
    for name in experiment.JOB_ARTIFACTS:
        experiment.atomic_json(directory / name, {"dataset": "gaussian", "condition": "no_skip_base", "seed": 42})
    completion = {
        "run_manifest_sha256": experiment.sha256(tmp_path / "run_manifest.json"),
        "attempt": directory.name,
        "artifact_sha256": {name: experiment.sha256(directory / name) for name in experiment.JOB_ARTIFACTS},
    }
    experiment.atomic_json(root / "complete.json", completion)
    assert experiment.verified_completion(tmp_path, "gaussian_no_skip_base_42") == directory
    experiment.atomic_json(directory / "summary.json", {"value": 2})
    with pytest.raises(ValueError, match="artifact changed"):
        experiment.verified_completion(tmp_path, "gaussian_no_skip_base_42")


def test_completion_rejects_manifest_change(tmp_path):
    experiment.atomic_json(tmp_path / "run_manifest.json", {"example": 1})
    directory = tmp_path / "jobs" / "gaussian_no_skip_base_42"
    directory.mkdir(parents=True)
    experiment.atomic_json(directory / "complete.json", {
        "run_manifest_sha256": "0" * 64, "attempt": "attempt-1", "artifact_sha256": {},
    })
    with pytest.raises(ValueError, match="different run"):
        experiment.verified_completion(tmp_path, "gaussian_no_skip_base_42")


def test_completion_rejects_empty_artifact_manifest(tmp_path):
    experiment.atomic_json(tmp_path / "run_manifest.json", {"example": 1})
    directory = tmp_path / "jobs" / "gaussian_no_skip_base_42"
    directory.mkdir(parents=True)
    experiment.atomic_json(directory / "complete.json", {
        "run_manifest_sha256": experiment.sha256(tmp_path / "run_manifest.json"),
        "attempt": "attempt-1", "artifact_sha256": {},
    })
    with pytest.raises(ValueError, match="Incomplete job"):
        experiment.verified_completion(tmp_path, "gaussian_no_skip_base_42")


def test_completion_rejects_misplaced_job(tmp_path):
    experiment.atomic_json(tmp_path / "run_manifest.json", {"example": 1})
    directory = tmp_path / "jobs" / "gaussian_skip_high_42" / "attempt-1"
    directory.mkdir(parents=True)
    for name in experiment.JOB_ARTIFACTS:
        experiment.atomic_json(directory / name, {"dataset": "gaussian", "condition": "no_skip_base", "seed": 42})
    experiment.atomic_json(directory.parent / "complete.json", {
        "run_manifest_sha256": experiment.sha256(tmp_path / "run_manifest.json"),
        "attempt": directory.name,
        "artifact_sha256": {name: experiment.sha256(directory / name) for name in experiment.JOB_ARTIFACTS},
    })
    with pytest.raises(ValueError, match="condition slot"):
        experiment.verified_completion(tmp_path, "gaussian_skip_high_42")


def test_campaign_launch_exception_stops_queue(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(experiment, "verify_run", lambda root: {"datasets": ["gaussian"]})
    launched = []

    def failing_launch(*args, **kwargs):
        launched.append(args)
        command = args[0]
        assert command[command.index("--condition") + 1] in experiment.CONDITIONS
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(experiment.subprocess, "Popen", failing_launch)
    with pytest.raises(OSError, match="synthetic launch failure"):
        experiment.campaign(tmp_path, ["gpu0", "gpu1", "gpu2", "gpu3"])
    assert len(launched) <= 4


@pytest.mark.parametrize("payload,error", [({"value": float("nan")}, ValueError),
                                          ({"value": object()}, TypeError)])
def test_atomic_json_removes_failed_payload_and_preserves_destination(tmp_path, payload, error):
    path = tmp_path / "value.json"
    experiment.atomic_json(path, {"value": 3})
    with pytest.raises(error):
        experiment.atomic_json(path, payload)
    assert json.loads(path.read_text()) == {"value": 3}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_json_does_not_remove_an_unowned_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    temporary = tmp_path / "value.json.existing.tmp"
    temporary.write_text("another writer")
    monkeypatch.setattr(experiment.uuid, "uuid4", lambda: SimpleNamespace(hex="existing"))
    with pytest.raises(FileExistsError):
        experiment.atomic_json(path, {"value": 3})
    assert temporary.read_text() == "another writer"
    assert not path.exists()


@pytest.mark.parametrize("condition", experiment.CONDITIONS)
def test_job_identity_stays_inside_protocol(condition):
    assert experiment.job_id("gaussian", condition, 1729) == f"gaussian_{condition}_1729"
    with pytest.raises(ValueError):
        experiment.job_id("../unrelated", condition, 1729)


def test_job_identity_rejects_unsupported_condition():
    with pytest.raises(ValueError, match="condition"):
        experiment.job_id("gaussian", "invalid_condition", 1729)


def test_train_help_lists_conditions_in_order():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/experiment.py"), "train", "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "--condition {no_skip_base,skip_base,no_skip_high,skip_high}" in result.stdout


@pytest.mark.parametrize("condition", experiment.CONDITIONS)
def test_train_cli_passes_condition(monkeypatch, condition):
    called = []
    monkeypatch.setattr(experiment, "train", lambda *args: called.append(args))
    monkeypatch.setattr(sys, "argv", [
        "experiment.py", "train", "--run-root", "run", "--dataset", "gaussian",
        "--condition", condition, "--seed", "42", "--gpu-uuid", "fixture",
    ])
    experiment.main()
    assert called == [("run", "gaussian", condition, 42, "fixture")]


@pytest.mark.parametrize("arguments", [
    ["--condition", "invalid_condition"],
    ["--condition", "no_skip_base", "--unsupported", "value"],
])
def test_train_cli_rejects_unsupported_arguments_before_training(monkeypatch, arguments):
    monkeypatch.setattr(experiment, "train", lambda *args: pytest.fail("training must not start"))
    monkeypatch.setattr(sys, "argv", [
        "experiment.py", "train", "--run-root", "run", "--dataset", "gaussian",
        "--seed", "42", "--gpu-uuid", "fixture", *arguments,
    ])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2


@pytest.mark.parametrize("condition,partner", [
    ("no_skip_base", "skip_base"), ("skip_base", "no_skip_base"),
    ("no_skip_high", "skip_high"), ("skip_high", "no_skip_high"),
])
def test_mapped_control_switches_skip_at_the_same_variance(condition, partner):
    tf = pytest.importorskip("tensorflow")
    from model import GaussianVAE

    with tf.device("/CPU:0"):
        model = GaussianVAE(input_dim=5, latent_dim=3, condition=condition, base_variance=0.37)
        data = {"base_variance": 0.37,
                "validation": np.random.default_rng(7).standard_normal((8, 5))}
        control = experiment.mapped_control(model, data)
    assert control["target_condition"] == partner


@pytest.mark.parametrize("failure_mode", ["worker_exit", "thread_start"])
def test_campaign_failure_reaps_blocked_peers_before_executor_join(tmp_path, failure_mode):
    """Exercise worker exit and partial executor startup with real CPU children."""
    (tmp_path / "logs").mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("""import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
(root / (gpu + ".pid")).write_text(str(os.getpid()))
if gpu == "gpu3":
    deadline = time.monotonic() + 10
    while len(list(root.glob("*.pid"))) != 4:
        if time.monotonic() > deadline:
            sys.exit(8)
        time.sleep(0.01)
    sys.exit(7)
while True:
    time.sleep(1)
""")
    driver = tmp_path / "driver.py"
    driver.write_text("""import subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import experiment
experiment.verify_run = lambda root: {"datasets": ["gaussian"]}
original_popen = subprocess.Popen
def cpu_worker(command, **kwargs):
    return original_popen([sys.executable, sys.argv[3], sys.argv[2]], **kwargs)
experiment.subprocess.Popen = cpu_worker
if sys.argv[4] == "thread_start":
    original_start = threading.Thread.start
    started = 0
    def thread_start(thread, *args, **kwargs):
        global started
        if thread.name.startswith("ThreadPoolExecutor-"):
            started += 1
            if started == 2:
                deadline = time.monotonic() + 10
                while not (Path(sys.argv[2]) / "gpu0.pid").exists():
                    if time.monotonic() > deadline:
                        raise AssertionError("First CPU child did not start")
                    time.sleep(0.01)
                raise RuntimeError("synthetic thread start failure")
        return original_start(thread, *args, **kwargs)
    threading.Thread.start = thread_start
experiment.campaign(sys.argv[2], ["gpu0", "gpu1", "gpu2", "gpu3"])
""")
    process = subprocess.Popen(
        [sys.executable, str(driver), str(ROOT / "src"), str(tmp_path), str(worker), failure_mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1"))
    try:
        expected_children = 4 if failure_mode == "worker_exit" else 1
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob("*.pid"))) != expected_children:
            assert process.poll() is None, "Campaign exited before starting the expected CPU children"
            assert time.monotonic() < deadline, "Campaign did not start the expected CPU children"
            time.sleep(0.01)
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("Campaign failure left CPU children blocked during executor shutdown")
        assert process.returncode != 0
        if failure_mode == "worker_exit":
            assert "Campaign stopped before completion" in stderr
            status = json.loads((tmp_path / "campaign_status.json").read_text())
            assert any(error.get("returncode") == 7 for error in status["errors"])
        else:
            assert "synthetic thread start failure" in stderr
        pids = [int(path.read_text()) for path in tmp_path.glob("*.pid")]
        assert len(pids) == expected_children
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        # This session owns only the disposable driver and its CPU children.
        if process.poll() is None:
            # Let the coordinator's signal cleanup reap its children after a failure.
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)


@pytest.fixture
def repeated_signal_campaign(tmp_path):
    """Disposable coordinator whose guard injects a real signal during cleanup."""
    (tmp_path / "logs").mkdir()
    worker = tmp_path / "signal_worker.py"
    worker.write_text("""import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
pid_file = root / (os.environ["CUDA_VISIBLE_DEVICES"] + ".pid")
pending_pid = pid_file.with_suffix(".tmp")
pending_pid.write_text(str(os.getpid()))
pending_pid.replace(pid_file)
while True:
    time.sleep(1)
""")
    driver = tmp_path / "signal_driver.py"
    driver.write_text("""import faulthandler, json, signal, subprocess, sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import experiment
root = Path(sys.argv[2])
injected_signal = int(sys.argv[4])
experiment.verify_run = lambda root: {"datasets": ["gaussian"]}
original_popen = subprocess.Popen
def cpu_worker(command, **kwargs):
    return original_popen([sys.executable, sys.argv[3], str(root)], **kwargs)
experiment.subprocess.Popen = cpu_worker

# The campaign creates its guard before Event/executor locks. Instrument only
# that lock, without depending on the signal handler or cleanup function name.
original_lock = threading.Lock
main_ident = threading.get_ident()
captured = []
class Guard:
    def __init__(self):
        self.lock = original_lock()
        self.injected = False
    def acquire(self, *args, **kwargs):
        acquired = self.lock.acquire(*args, **kwargs)
        if acquired and threading.get_ident() == main_ident and not self.injected:
            self.injected = True
            assert self.lock.locked()
            (root / "signal-under-guard.json").write_text(json.dumps({
                "signal": injected_signal, "main_thread": True, "guard_locked": True}))
            signal.raise_signal(injected_signal)
        return acquired
    def release(self):
        self.lock.release()
    def locked(self):
        return self.lock.locked()
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, *args):
        self.release()
def capture_guard():
    if not captured:
        captured.append(Guard())
        return captured[0]
    return original_lock()
experiment.threading.Lock = capture_guard

def original_handler(signum, frame):
    raise AssertionError("Original handler called before campaign cleanup finished")
previous = {}
for signum in (signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, original_handler)
    previous[signum] = signal.getsignal(signum)
faulthandler.dump_traceback_later(5)
try:
    experiment.campaign(root, ["fixture-" + str(i) for i in range(4)])
except RuntimeError as error:
    assert "Campaign stopped" in str(error), str(error)
else:
    raise AssertionError("Interrupted campaign returned success")
finally:
    faulthandler.cancel_dump_traceback_later()
    restored = {str(sig): signal.getsignal(sig) is handler for sig, handler in previous.items()}
    (root / "signal-cleanup.json").write_text(json.dumps({"restored_handlers": restored}))
""")
    processes = []

    def launch(injected_signal):
        process = subprocess.Popen(
            [sys.executable, str(driver), str(ROOT / "src"), str(tmp_path),
             str(worker), str(int(injected_signal))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
                     OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        processes.append(process)
        return process

    yield tmp_path, launch
    for process in processes:
        # Only this fixture's new session contains these disposable CPU workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)


@pytest.mark.parametrize("initial_signal,injected_signal", [
    (signal.SIGINT, signal.SIGTERM), (signal.SIGTERM, signal.SIGINT),
])
def test_campaign_repeated_signal_during_guarded_cleanup_reaps_children(
        repeated_signal_campaign, initial_signal, injected_signal):
    root, launch = repeated_signal_campaign
    process = launch(injected_signal)
    deadline = time.monotonic() + 10
    while len(list(root.glob("*.pid"))) != 4:
        assert process.poll() is None, "Coordinator exited before starting four CPU workers"
        assert time.monotonic() < deadline, "Coordinator did not start four CPU workers"
        time.sleep(0.01)
    pids = [int(path.read_text()) for path in root.glob("*.pid")]
    process.send_signal(initial_signal)
    try:
        _, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        assert (root / "signal-under-guard.json").exists(), "Cleanup never acquired the guard"
        pytest.fail("Repeated signal deadlocked campaign cleanup while it owned the guard")
    assert process.returncode == 0, stderr
    assert json.loads((root / "signal-under-guard.json").read_text()) == {
        "signal": int(injected_signal), "main_thread": True, "guard_locked": True}
    restored = json.loads((root / "signal-cleanup.json").read_text())["restored_handlers"]
    assert restored == {str(int(sig)): True for sig in (signal.SIGINT, signal.SIGTERM)}
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
