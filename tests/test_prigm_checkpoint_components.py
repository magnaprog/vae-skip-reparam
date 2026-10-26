"""CPU tests of the frozen-evaluator import boundary, not GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_gaussian_final_checkpoints as evaluator


class _ConstantDartModel:
    input_dim = 192
    latent_dim = 2
    b = 0.2
    observation_variance = 0.5
    residual_variance = b**2 * observation_variance

    def encode(self, x):
        return np.zeros((len(x), 2)), np.zeros((len(x), 2))

    def decode_mean(self, z):
        return np.zeros((len(z), 192), dtype=np.float32)

    def decode_residual(self, z):
        return self.b * self.decode_mean(z)


@pytest.fixture
def small_dart_protocol(monkeypatch):
    protocol = dict(evaluator.DART_DIAGNOSTICS)
    protocol.update(prior_samples=8, variance_samples=4, iw_samples=8)
    monkeypatch.setattr(evaluator, "DART_DIAGNOSTICS", protocol)
    return protocol


def test_dart_refit_uses_only_weighted_training_windows(small_dart_protocol):
    rng = np.random.default_rng(123)
    data = {"train": rng.normal(size=(5, 192)), "test": rng.normal(size=(3, 192)),
            "train_weights": np.array([1, 1, 2, 3, 7.]), "eval_weights": np.array([1, 2, 3.])}
    model = _ConstantDartModel()
    arrays, summary = evaluator.dart_checkpoint_diagnostics(model, data)
    assert all(isinstance(value, np.ndarray) for value in arrays.values())
    expected = np.average(np.square(data["train"]).mean(axis=1), weights=data["train_weights"])
    np.testing.assert_allclose(arrays["variance_refit_stream_variances"], expected)
    assert summary["variance_refit"]["observation_variance"] == pytest.approx(expected)
    changed = dict(data, test=data["test"] + 100)
    other, _ = evaluator.dart_checkpoint_diagnostics(model, changed)
    np.testing.assert_array_equal(other["variance_refit_stream_variances"],
                                  arrays["variance_refit_stream_variances"])
    np.testing.assert_allclose(arrays["original__prior_samples"] / np.sqrt(0.5),
                               arrays["variance_refit__prior_samples"] / np.sqrt(expected))
    assert model.observation_variance == 0.5
    assert model.residual_variance == pytest.approx(0.02)
    for label in ("original", "variance_refit"):
        np.testing.assert_allclose(arrays[label + "__energy_streams"],
                                   arrays[label + "__energy_per_window"] @ np.array([1, 2, 3]) / 6)
        assert summary[label]["joint_energy"] == pytest.approx(
            arrays[label + "__energy_streams"].mean())


def test_dart_uniform_weights_preserve_original_target(small_dart_protocol):
    rng = np.random.default_rng(7)
    data = {"train": rng.normal(size=(4, 192)), "test": rng.normal(size=(3, 192))}
    arrays, _ = evaluator.dart_checkpoint_diagnostics(_ConstantDartModel(), data)
    np.testing.assert_array_equal(arrays["variance_refit_training_weights"], np.full(4, 0.25))
    np.testing.assert_array_equal(arrays["diagnostic_test_weights"], np.full(3, 1 / 3))


def test_dart_references_save_recomputable_independent_streams(small_dart_protocol):
    from prigm_metrics import energy_score
    rng = np.random.default_rng(8)
    data = {"train": rng.normal(size=(20, 192)), "test": rng.normal(size=(3, 192)),
            "train_weights": np.arange(1., 21), "eval_weights": np.array([1., 2., 3.])}
    arrays, summaries = evaluator.dart_reference_diagnostics(data, 0.5)
    for name in ("ppca_matched", "ppca_fitted", "diagonal_matched"):
        draws = arrays[name + "__prior_samples"]
        assert draws.shape == (3, 8, 192)
        assert not np.array_equal(draws[0], draws[1])
        expected = np.stack([energy_score(stream, data["test"]) for stream in draws])
        np.testing.assert_array_equal(arrays[name + "__joint_per_window"], expected)
        assert summaries[name]["joint_energy"] == pytest.approx(
            np.mean(expected @ (data["eval_weights"] / data["eval_weights"].sum())))


@pytest.mark.parametrize("extra", [["--datasets", "gaussian"], ["--checkpoints", "final"]])
def test_dart_mode_rejects_unapproved_dataset_or_checkpoint(tmp_path, extra):
    with pytest.raises(SystemExit):
        evaluator.main(["--dart-diagnostics", "--run-root", str(tmp_path / "absent"),
                        "--expected-core-sha256", "0" * 64, "--output-dir", str(tmp_path / "out"),
                        "--gpu-uuid", "GPU-test", "--worker-index", "0", *extra])
    assert not (tmp_path / "out").exists()


def _fixture(tmp_path, sources=None):
    root = tmp_path / "core"
    (root / "source/src").mkdir(parents=True)
    (root / "prepared").mkdir()
    for name in evaluator.CORE_FILES:
        content = (sources[name] if sources is not None else
                   "raise RuntimeError('must not execute during hash verification')\n")
        (root / "source" / name).write_text(content)
    names = ("metadata.json", *(name + ".npz" for name in evaluator.DATASETS))
    for name in names:
        (root / "prepared" / name).write_bytes(b"source-boundary fixture; not training data")
    manifest = {
        "schema_version": 1, "purpose": "core", "datasets": list(evaluator.DATASETS),
        "code_sha256": {name: evaluator._sha(root / "source" / name) for name in evaluator.CORE_FILES},
        "prepared_sha256": {name: evaluator._sha(root / "prepared" / name) for name in names},
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    return root, evaluator._sha(root / "run_manifest.json"), manifest


def test_source_verification_does_not_execute_archived_code(tmp_path):
    root, pin, manifest = _fixture(tmp_path)
    assert evaluator.verified_core_source(root, pin) == manifest


def test_incorrect_core_pin_rejected_before_source_import(tmp_path):
    root, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Core manifest SHA256 differs"):
        evaluator.verified_core_source(root, "0" * 64)


@pytest.mark.parametrize("target", ["source/src/prigm_model.py", "prepared/gaussian.npz"])
def test_tampered_source_or_prepared_data_rejected(tmp_path, target):
    root, pin, _ = _fixture(tmp_path)
    with (root / target).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differs"):
        evaluator.verified_core_source(root, pin)


def test_source_directory_symlink_rejected(tmp_path):
    root, pin, _ = _fixture(tmp_path)
    (root / "source/src").rename(root / "actual-src")
    (root / "source/src").symlink_to(root / "actual-src", target_is_directory=True)
    with pytest.raises(ValueError, match="regular directories"):
        evaluator.verified_core_source(root, pin)


def test_extra_source_field_rejected_even_when_manifest_pin_matches(tmp_path):
    root, _, manifest = _fixture(tmp_path)
    manifest["code_sha256"]["../../outside.py"] = "0" * 64
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    pin = evaluator._sha(root / "run_manifest.json")
    with pytest.raises(ValueError, match="Unsupported frozen core identity"):
        evaluator.verified_core_source(root, pin)


def test_duplicate_manifest_key_rejected(tmp_path):
    root, _, _ = _fixture(tmp_path)
    payload = '{"schema_version": 1, "schema_version": 2}'
    (root / "run_manifest.json").write_text(payload)
    pin = hashlib.sha256(payload.encode()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        evaluator.verified_core_source(root, pin)


@pytest.mark.parametrize("name, kind", [
    ("__init__.py", "file"),
    ("prigm_experiment.pyc", "file"),
    ("prigm_experiment.so", "file"),
    ("prigm_v2_metrics.py", "file"),
    (".hidden", "file"),
    ("__pycache__", "directory"),
    ("prigm_experiment", "directory"),
    ("extra", "directory"),
    ("prigm_data", "symlink"),
    ("extra.py", "dangling_symlink"),
])
def test_undeclared_source_entry_rejected_with_unchanged_pin(tmp_path, name, kind):
    root, pin, _ = _fixture(tmp_path)
    entry = root / "source/src" / name
    if kind == "file":
        entry.write_text("undeclared")
    elif kind == "directory":
        entry.mkdir()
    else:
        target = "prigm_data.py" if kind == "symlink" else "missing.py"
        entry.symlink_to(entry.parent / target)
    assert evaluator._sha(root / "run_manifest.json") == pin
    with pytest.raises(ValueError, match="Frozen source inventory differs"):
        evaluator.verified_core_source(root, pin)


@pytest.mark.parametrize("name", evaluator.CORE_FILES)
@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_declared_source_must_remain_a_regular_file(tmp_path, name, kind):
    root, pin, _ = _fixture(tmp_path)
    entry = root / "source" / name
    entry.rename(tmp_path / "original.py")
    if kind == "directory":
        entry.mkdir()
    elif kind == "symlink":
        entry.symlink_to(tmp_path / "original.py")
    with pytest.raises(ValueError, match="inventory differs|regular file"):
        evaluator.verified_core_source(root, pin)


@pytest.mark.parametrize("name", [Path(name).stem for name in evaluator.CORE_FILES])
def test_main_rejects_preloaded_core_dependency(tmp_path, monkeypatch, name):
    root, pin, _ = _fixture(tmp_path)
    monkeypatch.setitem(sys.modules, name, object())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-probe")
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="requires a fresh Python process"):
        evaluator.main([
            "--run-root", str(root), "--expected-core-sha256", pin,
            "--output-dir", str(output), "--gpu-uuid", "GPU-cpu-probe",
            "--worker-index", "0",
        ])
    assert not output.exists()


def _main_subprocess(tmp_path, root, pin):
    # Run the real __main__ entry point, with no site packages or GPU libraries.
    argv = [str(Path(evaluator.__file__).resolve()),
            "--run-root", str(root), "--expected-core-sha256", pin,
            "--output-dir", str(tmp_path / "output"), "--gpu-uuid", "GPU-cpu-probe",
            "--worker-index", "0"]
    probe = textwrap.dedent(f"""\
        from pathlib import Path
        import runpy
        import sys
        import types

        class GPUImportTripwire:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'tensorflow', 'keras', 'torch', 'jax', 'cupy'}}:
                    Path({str(tmp_path / 'gpu-imported')!r}).write_text(fullname)
                    raise AssertionError('GPU import attempted: ' + fullname)

        sys.meta_path.insert(0, GPUImportTripwire())
        sys.modules['numpy'] = types.ModuleType('numpy')
        helper = types.ModuleType('prigm_v2_metrics')
        helper.posterior_score_components = None
        sys.modules[helper.__name__] = helper
        sys.argv = {argv!r}
        # Do not let the parent's PYTHONDONTWRITEBYTECODE mask a missing safeguard.
        sys.dont_write_bytecode = False
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except RuntimeError as exc:
            if str(exc) != 'CPU_IMPORT_PROBE_COMPLETE':
                raise
            assert sys.dont_write_bytecode
            for name in {evaluator.CORE_FILES!r}:
                module = sys.modules[Path(name).stem]
                assert Path(module.__file__) == Path({str(root / 'source')!r}) / name
            print('CPU_IMPORT_PROBE_COMPLETE')
    """)
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe], cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-cpu-probe"},
        capture_output=True, text=True, timeout=30,
    )


def test_main_rejects_shadow_package_before_execution_output_or_gpu_import(tmp_path):
    root, pin, _ = _fixture(tmp_path)
    shadow = root / "source/src/prigm_experiment"
    shadow.mkdir()
    sentinel = tmp_path / "shadow-executed"
    (shadow / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
        "print('shadow package executed', flush=True)\nimport tensorflow\n"
    )
    assert evaluator._sha(root / "run_manifest.json") == pin
    result = _main_subprocess(tmp_path, root, pin)
    assert not sentinel.exists(), result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "gpu-imported").exists()
    assert result.stdout == ""
    assert result.returncode != 0
    assert "ValueError: Frozen source inventory differs" in result.stderr
    assert not list((root / "source").rglob("__pycache__"))


def test_main_imports_closed_cpu_sources_without_writing_bytecode(tmp_path):
    sources = {
        "src/prigm_data.py": "DATA_IMPORTED = True\n",
        "src/prigm_metrics.py": "METRICS_IMPORTED = True\n",
        "src/prigm_model.py": (
            "from prigm_data import DATA_IMPORTED\n"
            "from prigm_metrics import METRICS_IMPORTED\n"
            "class GaussianVAE: pass\n"
        ),
        "src/prigm_experiment.py": (
            "import prigm_data\nimport prigm_metrics\nfrom prigm_model import GaussianVAE\n"
            "def gpu_runtime(uuid):\n"
            "    raise RuntimeError('CPU_IMPORT_PROBE_COMPLETE')\n"
            "importance_evaluate = job_id = load_prepared = verified_completion = verify_run = None\n"
        ),
    }
    root, pin, manifest = _fixture(tmp_path, sources=sources)
    result = _main_subprocess(tmp_path, root, pin)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "CPU_IMPORT_PROBE_COMPLETE\n"
    assert result.stderr == ""
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "gpu-imported").exists()
    assert not list((root / "source").rglob("__pycache__"))
    assert not list((root / "source").rglob("*.pyc"))
    assert evaluator.verified_core_source(root, pin) == manifest


def test_receipt_writer_does_not_overwrite_existing_output(tmp_path):
    path = tmp_path / "receipt.json"
    evaluator._write_json(path, {"first": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator._write_json(path, {"first": False})
    assert path.read_bytes() == original


def _stage_fixture(tmp_path, sources=None):
    root = tmp_path / "stage"
    (root / "source/src").mkdir(parents=True)
    (root / "prepared").mkdir()
    for name in evaluator.STAGE_FILES:
        (root / "source" / name).write_text(
            sources[name] if sources else "raise AssertionError('archived code executed')\n")
    dataset = "dart_44402_2018"
    (root / "prepared" / f"{dataset}.npz").write_bytes(b"prepared fixture")
    (root / "prepared/records.json").write_text("{}")
    jobs = [[dataset, condition, seed, variance] for seed in evaluator.SEEDS
            for variance in ("low", "high") for condition in ("base", "raw")]
    manifest = {
        "schema_version": 4, "stage": "dart_record_extension",
        "purpose": "dart_input_skip_evaluation", "jobs": jobs,
        "code_sha256": {name: evaluator._sha(root / "source" / name)
                        for name in evaluator.STAGE_FILES},
        "prepared_sha256": {dataset: evaluator._sha(root / "prepared" / f"{dataset}.npz")},
        "records_sha256": evaluator._sha(root / "prepared/records.json"),
    }
    pin = _stage_manifest(root, manifest)
    return root, pin, manifest


def _stage_manifest(root, manifest):
    (root / "stage_manifest.json").write_text(json.dumps(manifest))
    return evaluator._sha(root / "stage_manifest.json")


def _stage_argv(root, pin, output, *extra):
    return ["--stage-root", str(root), "--expected-stage-sha256", pin,
            "--output-dir", str(output), "--gpu-uuid", "GPU-cpu-probe",
            "--worker-index", "0", *extra]


@pytest.fixture
def fresh_prigm_boundary(monkeypatch):
    # Only use this for rejection tests which never import archived code.
    for name in list(sys.modules):
        if name == "prigm" or name.startswith("prigm_"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-probe")


def test_stage_pin_verification_never_imports_code(tmp_path):
    root, pin, manifest = _stage_fixture(tmp_path)
    assert evaluator.verified_stage_source(root, pin) == manifest


@pytest.mark.parametrize("pin", ["0" * 64, "A" * 64, "../manifest", "", None])
def test_stage_bad_pin_rejected(tmp_path, pin):
    root, _, _ = _stage_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        evaluator.verified_stage_source(root, pin)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 3), ("stage", "numerical_null"),
    ("purpose", "numerical_validation"), ("prepared_sha256", {"../outside": "0" * 64}),
    ("prepared_sha256", {}), ("code_sha256", None),
])
def test_stage_unapproved_identity_rejected_with_matching_pin(tmp_path, field, value):
    root, _, manifest = _stage_fixture(tmp_path)
    manifest[field] = value
    with pytest.raises(ValueError, match="Unsupported frozen DART stage identity"):
        evaluator.verified_stage_source(root, _stage_manifest(root, manifest))


@pytest.mark.parametrize("name", ["../../outside.py", "src/prigm_model.py", "src/prigm_v2_extra.py"])
def test_stage_unapproved_source_path_rejected(tmp_path, name):
    root, _, manifest = _stage_fixture(tmp_path)
    manifest["code_sha256"][name] = "0" * 64
    with pytest.raises(ValueError, match="Unsupported frozen DART stage identity"):
        evaluator.verified_stage_source(root, _stage_manifest(root, manifest))


@pytest.mark.parametrize("name", ["source/__init__.py", "source/src/prigm_v2_model.so",
                                  "source/src/__pycache__", "source/src/prigm_v2_experiment",
                                  "source/src/.hidden", "prepared/extra.npz"])
def test_stage_extra_inventory_rejected(tmp_path, name):
    root, pin, _ = _stage_fixture(tmp_path)
    (root / name).write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        evaluator.verified_stage_source(root, pin)


@pytest.mark.parametrize("name", ["source", "source/src", "prepared",
                                  *["source/" + name for name in evaluator.STAGE_FILES],
                                  "prepared/records.json", "prepared/dart_44402_2018.npz",
                                  "stage_manifest.json"])
def test_stage_symlink_rejected(tmp_path, name):
    root, pin, _ = _stage_fixture(tmp_path)
    entry = root / name
    destination = tmp_path / "original"
    entry.rename(destination)
    entry.symlink_to(destination, target_is_directory=destination.is_dir())
    with pytest.raises(ValueError, match="regular"):
        evaluator.verified_stage_source(root, pin)


@pytest.mark.parametrize("name", [*["source/" + name for name in evaluator.STAGE_FILES],
                                  "prepared/records.json", "prepared/dart_44402_2018.npz"])
def test_stage_changed_input_rejected(tmp_path, name):
    root, pin, _ = _stage_fixture(tmp_path)
    with (root / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differs"):
        evaluator.verified_stage_source(root, pin)


@pytest.mark.parametrize("name", ["prigm_data", "prigm_model", "prigm_metrics", "prigm_experiment",
                                  "prigm_v2_data", "prigm_v2_model", "prigm_v2_metrics",
                                  "prigm_v2_experiment", "prigm_v2_model.child"])
def test_stage_dispatch_rejects_every_preloaded_prigm_module(
        tmp_path, monkeypatch, fresh_prigm_boundary, name):
    root, pin, _ = _stage_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, name, object())
    with pytest.raises(RuntimeError, match="requires a fresh Python process"):
        evaluator.main(_stage_argv(root, pin, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("extra", [("--datasets", "gaussian_v2"),
                                    ("--datasets", "dart_44402_2018", "dart_44402_2018"),
                                    ("--checkpoints", "final", "final"),
                                    ("--checkpoints", "best"), ("--worker-index", "4")])
def test_stage_invalid_cli_leaves_no_output(tmp_path, fresh_prigm_boundary, extra):
    root, pin, _ = _stage_fixture(tmp_path)
    with pytest.raises((ValueError, SystemExit)):
        evaluator.main(_stage_argv(root, pin, tmp_path / "out", *extra))
    assert not (tmp_path / "out").exists()


def test_stage_rejects_wrong_gpu_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(SystemExit):
        evaluator.stage_main(_stage_argv(tmp_path / "missing", "0" * 64, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_stage_rejects_inplace_or_existing_output(tmp_path, fresh_prigm_boundary, existing):
    root, pin, _ = _stage_fixture(tmp_path)
    output = tmp_path / "out" if existing else root / "adjunct"
    if existing:
        output.mkdir()
    with pytest.raises((FileExistsError, ValueError), match="fresh adjunct|outside the frozen stage"):
        evaluator.main(_stage_argv(root, pin, output))


@pytest.mark.parametrize("checkpoints", [["final"], ["selected"], ["final", "selected"]])
def test_stage_matrix_worker_partition_is_complete_disjoint_and_manifest_ordered(tmp_path, checkpoints):
    _, _, manifest = _stage_fixture(tmp_path)
    partitions = [evaluator._stage_jobs(manifest, None, checkpoints, worker)
                  for worker in range(4)]
    expected = [(*job, label) for job in manifest["jobs"] for label in checkpoints]
    assigned = []
    for worker, (all_jobs, jobs) in enumerate(partitions):
        assert all_jobs == expected
        assert jobs == expected[worker::4]
        assigned.extend(jobs)
    assert len(assigned) == len(set(assigned)) == len(expected)
    assert set(assigned) == set(expected)


def test_stage_matrix_explicit_dataset_subset(tmp_path):
    _, _, manifest = _stage_fixture(tmp_path)
    second = "dart_46409_2018"
    manifest["prepared_sha256"][second] = "0" * 64
    manifest["jobs"] += [[second, *job[1:]] for job in list(manifest["jobs"])]
    all_jobs, _ = evaluator._stage_jobs(manifest, [second], ["final"], 0)
    assert len(all_jobs) == 12
    assert {job[0] for job in all_jobs} == {second}


@pytest.mark.parametrize("bad_job", [["dart_44402_2018", "canonical", 42, "low"],
                                    ["dart_44402_2018", "base", 43, "low"],
                                    ["dart_44402_2018", "base", True, "low"],
                                    ["dart_44402_2018", "base", 42, "other"],
                                    ["gaussian_v2", "base", 42, "low"], []])
def test_stage_matrix_rejects_invalid_jobs(tmp_path, bad_job):
    _, _, manifest = _stage_fixture(tmp_path)
    manifest["jobs"][0] = bad_job
    with pytest.raises(ValueError, match="job matrix"):
        evaluator._stage_jobs(manifest, None, ["final"], 0)


def test_stage_matrix_rejects_duplicate_or_empty_worker(tmp_path):
    _, _, manifest = _stage_fixture(tmp_path)
    manifest["jobs"].append(manifest["jobs"][0])
    with pytest.raises(ValueError, match="job matrix"):
        evaluator._stage_jobs(manifest, None, ["final"], 0)
    manifest["jobs"] = manifest["jobs"][:1]
    with pytest.raises(ValueError, match="no declared jobs"):
        evaluator._stage_jobs(manifest, None, ["final"], 3)


def test_stage_array_serialization_is_finite_exact_and_exclusive(tmp_path):
    path = tmp_path / "arrays.npz"
    arrays = {"scalar": np.asarray(0.5), "windows": np.arange(12.).reshape(3, 4)}
    assert evaluator._stage_save_arrays(path, arrays) == evaluator._sha(path)
    with np.load(path, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    with pytest.raises(FileExistsError):
        evaluator._stage_save_arrays(path, arrays)
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluator._stage_save_arrays(tmp_path / "bad.npz", {"bad": np.array([np.nan])})
    assert not (tmp_path / "bad.npz").exists()


def test_stage_replay_uses_saved_seeds_and_checks_all_components(tmp_path):
    from prigm_v2_experiment import _iw_evaluate
    model = _ConstantDartModel()
    data = {"test": np.random.default_rng(5).normal(size=(3, 192))}
    spec = {"iw_samples": 8, "iw_seed": 65001, "iw_streams": 3}
    streams = [_iw_evaluate(model, data["test"], 8, 65001 + index) for index in range(3)]
    saved = {{"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key):
             np.stack([stream[key] for stream in streams]) for key in streams[0]}
    path = tmp_path / "evaluation-final.npz"
    np.savez_compressed(path, **saved)
    arrays, replay = evaluator._stage_replay(model, data, path, spec)
    assert replay["stream_seeds"] == [65001, 65002, 65003]
    assert replay["reproduced_exactly"] is True
    assert set(replay["max_abs_differences"].values()) == {0.0}
    np.testing.assert_array_equal(arrays["stage_replay__posterior_mean_mse"], saved["posterior_mean_mse"])
    saved["posterior_mean_mse"][0, 0] += 1e-7
    np.savez_compressed(path, **saved)
    _, replay = evaluator._stage_replay(model, data, path, spec)
    assert replay["reproduced_exactly"] is False
    saved["iw_log_scores"][0, 0] += 1
    np.savez_compressed(path, **saved)
    with pytest.raises(AssertionError, match="Archived checkpoint replay differs: log_score"):
        evaluator._stage_replay(model, data, path, spec)


def _cpu_stage(tmp_path, selected_step=10000):
    """Pinned CPU import/output fixture, not a substitute for the runner's HDF5 tests."""
    source_root = Path(evaluator.__file__).parent
    sources = {name: (source_root / Path(name).name).read_text()
               for name in ("src/prigm_metrics.py", "src/prigm_v2_metrics.py")}
    sources["src/prigm_v2_data.py"] = "ARCHIVED_DATA = True\n"
    sources["src/prigm_v2_model.py"] = textwrap.dedent("""\
        import json
        import numpy as np
        from pathlib import Path
        import prigm_v2_data
        class BypassVAE:
            def __init__(self, input_dim, latent_dim, condition, observation_variance, seed, dtype):
                assert dtype == 'float32'
                self.input_dim, self.latent_dim = input_dim, latent_dim
                self.condition, self.seed = condition, seed
                self.b = 1.0 if condition == 'base' else 0.2
                self.observation_variance = observation_variance
                self.residual_variance = self.b**2 * observation_variance
            def load_weights(self, path):
                value = json.loads(Path(path).read_text())
                assert value == [self.condition, self.seed, self.observation_variance]
            def encode(self, x):
                return np.zeros((len(x), self.latent_dim)), np.zeros((len(x), self.latent_dim))
            def decode_mean(self, z):
                return np.zeros((len(z), self.input_dim), dtype=np.float32)
        """)
    sources["src/prigm_v2_experiment.py"] = textwrap.dedent("""\
        import hashlib
        import json
        from pathlib import Path
        from prigm_v2_metrics import posterior_score_components
        def verify_stage(root):
            return json.loads((root / 'stage_manifest.json').read_text())
        def gpu_runtime(uuid):
            return {'test_only_cpu_runtime': True, 'gpu_uuid': uuid}
        def job_id(*job):
            return '_'.join(map(str, job))
        def _matrix_job(manifest, identifier):
            return next(job for job in manifest['jobs'] if job_id(*job) == identifier)
        def verified_completion(root, identifier):
            complete = root / 'jobs' / identifier / 'complete.json'
            if not complete.exists():
                return None
            value = json.loads(complete.read_text())
            attempt = complete.parent / value['attempt']
            for name, expected in value['artifact_sha256'].items():
                assert hashlib.sha256((attempt / name).read_bytes()).hexdigest() == expected
            return attempt
        def _iw_evaluate(model, observations, samples, seed):
            return posterior_score_components(model, observations, samples, seed)
        """)
    root, _, manifest = _stage_fixture(tmp_path, sources)
    rng = np.random.default_rng(23)
    data = {"train": rng.normal(size=(20, 192)), "test": rng.normal(size=(3, 192)),
            "train_weights": np.arange(1., 21), "eval_weights": np.array([1., 2., 3.])}
    prepared = root / "prepared/dart_44402_2018.npz"
    np.savez_compressed(prepared, **data)
    manifest["prepared_sha256"]["dart_44402_2018"] = evaluator._sha(prepared)
    manifest["specification"] = {"window": 192, "latent_dim": 2, "max_steps": 10000,
                                 "iw_samples": 8, "iw_seed": 65001, "iw_streams": 3}
    manifest["job_parameters"] = {}
    from prigm_v2_metrics import posterior_score_components
    for dataset, condition, seed, variance_label in manifest["jobs"]:
        identifier = f"{dataset}_{condition}_{seed}_{variance_label}"
        variance = 0.5 if variance_label == "low" else 12.5
        manifest["job_parameters"][identifier] = {"observation_variance": variance}
        attempt = root / "jobs" / identifier / ("attempt-" + "a" * 32)
        attempt.mkdir(parents=True)
        for label in ("final", "selected"):
            (attempt / f"{label}.weights.h5").write_text(json.dumps([condition, seed, variance]))
        (attempt / "summary.json").write_text(json.dumps({"selected_step": selected_step}))
        model = _ConstantDartModel()
        model.b = 1.0 if condition == "base" else 0.2
        model.observation_variance = variance
        model.residual_variance = model.b**2 * variance
        streams = [posterior_score_components(model, data["test"], 8, 65001 + index)
                   for index in range(3)]
        arrays = {{"log_score": "iw_log_scores", "ess": "iw_ess"}.get(key, key):
                  np.stack([stream[key] for stream in streams]) for key in streams[0]}
        np.savez_compressed(attempt / "evaluation-final.npz", **arrays)
        if selected_step < 10000:
            np.savez_compressed(attempt / "evaluation-selected.npz", **arrays)
        completion = {"attempt": attempt.name,
                      "artifact_sha256": {path.name: evaluator._sha(path) for path in attempt.iterdir()}}
        evaluator._write_json(attempt / "complete.json", completion)
        evaluator._write_json(attempt.parent / "complete.json", completion)
    return root, _stage_manifest(root, manifest), manifest


def _stage_process(tmp_path, root, pin, extra=(), mutation=None):
    # Keep a private evaluator copy so the source-change test never edits the repository.
    script = tmp_path / "eval_gaussian_final_checkpoints.py"
    script.write_bytes(Path(evaluator.__file__).read_bytes())
    probe = textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        class GPUImportTripwire:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'tensorflow', 'keras', 'torch', 'jax', 'cupy'}}:
                    raise AssertionError('GPU import attempted: ' + fullname)
        sys.meta_path.insert(0, GPUImportTripwire())
        # Explicitly admit the installed CPU packages even with isolated Python.
        sys.path.append({str(Path(np.__file__).resolve().parents[1])!r})
        sys.path.insert(0, {str(tmp_path)!r})
        import eval_gaussian_final_checkpoints as evaluator
        assert not any(name.startswith('prigm_') for name in sys.modules)
        evaluator.DART_DIAGNOSTICS.update(prior_samples=8, variance_samples=4, iw_samples=8)
        sys.dont_write_bytecode = False
        root, output = Path({str(root)!r}), Path({str(tmp_path / 'output')!r})
        mutation = {mutation!r}
        if mutation:
            original = evaluator._stage_save_arrays
            def save_then_mutate(path, arrays):
                sha = original(path, arrays)
                evaluator._stage_save_arrays = original
                if mutation == 'source':
                    target = root / 'source/src/prigm_v2_metrics.py'
                elif mutation == 'prepared':
                    target = root / 'prepared/dart_44402_2018.npz'
                elif mutation == 'completion':
                    target = root / 'jobs/dart_44402_2018_base_42_low/complete.json'
                elif mutation == 'weights':
                    target = next((root / 'jobs/dart_44402_2018_base_42_low').glob('attempt-*/final.weights.h5'))
                elif mutation == 'protocol':
                    target = output / 'protocol.json'
                elif mutation == 'evaluator':
                    target = Path(evaluator.__file__)
                else:
                    target = path
                with target.open('ab') as stream:
                    stream.write(b' ')
                return sha
            evaluator._stage_save_arrays = save_then_mutate
        result = evaluator.main({_stage_argv(root, pin, tmp_path / 'output', *extra)!r})
        assert result == 0 and sys.dont_write_bytecode
        for name in evaluator.STAGE_FILES:
            module = sys.modules[Path(name).stem]
            assert Path(module.__file__) == root / 'source' / name
        """)
    return subprocess.run([sys.executable, "-I", "-c", probe], cwd=tmp_path,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-cpu-probe",
                               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                               "PYTHONDONTWRITEBYTECODE": "1"},
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("selected_step", [10000, 1000])
@pytest.mark.parametrize("extra", [(), ("--checkpoints", "final", "selected", "--worker-index", "1")])
def test_stage_cpu_archived_import_replay_output_closure_and_reference_cache(tmp_path, selected_step, extra):
    root, pin, manifest = _cpu_stage(tmp_path, selected_step)
    original_inputs = {path: evaluator._sha(path) for path in root.rglob("*") if path.is_file()}
    result = _stage_process(tmp_path, root, pin, extra)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "output"
    complete = evaluator._read_json(output / "complete.json")
    protocol = evaluator._read_json(output / "protocol.json")
    assert complete["purpose"] == protocol["purpose"] == "dart_stage_checkpoint_diagnostics"
    assert complete["schema_version"] == protocol["schema_version"] == 2
    assert protocol["stage_purpose"] == "dart_input_skip_evaluation"
    assert protocol["stage_manifest_sha256"] == pin
    assert protocol["stage_source_sha256"] == manifest["code_sha256"]
    assert set(protocol["evaluator_source_sha256"]) == {"eval_gaussian_final_checkpoints.py"}
    assert complete["protocol_sha256"] == evaluator._sha(output / "protocol.json")
    assert complete["elapsed_seconds"] > 0
    assert complete["started_utc"] <= complete["completed_utc"]
    labels = ["final", "selected"] if extra else ["final"]
    all_jobs, jobs = evaluator._stage_jobs(manifest, None, labels, 1 if extra else 0)
    assert protocol["jobs"] == [list(job) for job in jobs]
    assert protocol["all_jobs"] == [list(job) for job in all_jobs]
    assert len(result.stdout.splitlines()) == len(jobs)
    references = set()
    for dataset, condition, seed, variance_label, label in jobs:
        identifier = f"{dataset}_{condition}_{seed}_{variance_label}"
        stem = f"{identifier}-{label}"
        record = evaluator._read_json(output / f"{stem}.json")
        attempt = root / "jobs" / identifier / ("attempt-" + "a" * 32)
        assert record["checkpoint_sha256"] == evaluator._sha(attempt / f"{label}.weights.h5")
        assert record["stage_completion_sha256"] == evaluator._sha(attempt.parent / "complete.json")
        assert record["prepared_sha256"] == manifest["prepared_sha256"][dataset]
        assert record["array_sha256"] == evaluator._sha(output / f"{stem}.npz")
        assert record["protocol_sha256"] == complete["protocol_sha256"]
        assert record["step"] == (10000 if label == "final" else selected_step)
        evaluation_label = "selected" if label == "selected" and selected_step < 10000 else "final"
        assert record["stage_evaluation_file"] == f"evaluation-{evaluation_label}.npz"
        assert record["stage_evaluation_sha256"] == evaluator._sha(attempt / record["stage_evaluation_file"])
        assert record["archived_component_replay"]["reproduced_exactly"] is True
        assert (record["dimension"], record["test_windows"]) == (192, 3)
        assert set(record["dart_diagnostics"]) == {"original", "variance_refit"}
        with np.load(output / f"{stem}.npz", allow_pickle=False) as arrays:
            assert record["array_shapes"] == {key: list(arrays[key].shape) for key in arrays.files}
            np.testing.assert_allclose(arrays["diagnostic_test_weights"], np.array([1, 2, 3]) / 6)
        references.add(record["references"]["file"])
    assert len(references) == len({(job[0], job[3]) for job in jobs})
    assert len(list(output.glob("references-*.npz"))) == len(references)
    for name, sha in complete["artifact_sha256"].items():
        assert evaluator._sha(output / name) == sha
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == (
        set(complete["artifact_sha256"]) | {"protocol.json", "complete.json"})
    assert {path: evaluator._sha(path) for path in root.rglob("*") if path.is_file()} == original_inputs
    assert not list(root.rglob("__pycache__"))


@pytest.mark.parametrize("mutation", ["source", "prepared", "completion", "weights",
                                      "protocol", "arrays", "evaluator"])
def test_stage_input_output_changes_prevent_completion_after_serialization(tmp_path, mutation):
    root, pin, _ = _cpu_stage(tmp_path)
    result = _stage_process(tmp_path, root, pin, mutation=mutation)
    assert result.returncode != 0
    assert "changed during" in result.stderr or "differs" in result.stderr, result.stderr
    assert not (tmp_path / "output/complete.json").exists()


@pytest.mark.parametrize("failure", ["pin", "shadow", "missing_completion", "replay"])
def test_stage_cpu_failures_never_publish_completion(tmp_path, failure):
    root, pin, _ = _cpu_stage(tmp_path)
    if failure == "pin":
        pin = "0" * 64
    elif failure == "shadow":
        shadow = root / "source/src/prigm_v2_experiment"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("raise AssertionError('shadow executed')\n")
    elif failure == "missing_completion":
        (root / "jobs/dart_44402_2018_base_42_low/complete.json").unlink()
    else:
        attempt = next((root / "jobs/dart_44402_2018_base_42_low").glob("attempt-*"))
        with np.load(attempt / "evaluation-final.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["posterior_mean_mse"] += 1
        np.savez_compressed(attempt / "evaluation-final.npz", **arrays)
        completion = evaluator._read_json(attempt / "complete.json")
        completion["artifact_sha256"]["evaluation-final.npz"] = evaluator._sha(attempt / "evaluation-final.npz")
        for path in (attempt / "complete.json", attempt.parent / "complete.json"):
            path.write_text(json.dumps(completion))
    result = _stage_process(tmp_path, root, pin)
    assert result.returncode != 0
    assert not (tmp_path / "output/complete.json").exists()
    assert "GPU import attempted" not in result.stderr
    assert "shadow executed" not in result.stderr
    if failure in ("pin", "shadow"):
        assert not (tmp_path / "output").exists()
    messages = {"pin": "Stage manifest SHA256 differs", "shadow": "source inventory differs",
                "missing_completion": "Missing verified stage job",
                "replay": "Archived checkpoint replay differs: posterior_mean_mse"}
    assert messages[failure] in result.stderr


@pytest.mark.parametrize("payload", ['[]', '{"schema_version": 4, "schema_version": 3}'])
def test_stage_malformed_or_duplicate_json_rejected(tmp_path, payload):
    root, _, _ = _stage_fixture(tmp_path)
    (root / "stage_manifest.json").write_text(payload)
    with pytest.raises(ValueError, match="Unsupported frozen|Duplicate JSON key"):
        evaluator.verified_stage_source(root, evaluator._sha(root / "stage_manifest.json"))


def test_stage_calls_real_archived_closure_checker_before_gpu_or_model_import(tmp_path):
    source_root = Path(evaluator.__file__).parent
    sources = {name: (source_root / Path(name).name).read_text() for name in evaluator.STAGE_FILES}
    root, pin, _ = _stage_fixture(tmp_path, sources)
    # Source hashes and matrix are valid, but the skeletal metadata/specification is not.
    result = _stage_process(tmp_path, root, pin)
    assert result.returncode != 0
    assert "Inconsistent manifest fields" in result.stderr
    assert "GPU import attempted" not in result.stderr
    assert not (tmp_path / "output").exists()
    assert not list(root.rglob("__pycache__"))
