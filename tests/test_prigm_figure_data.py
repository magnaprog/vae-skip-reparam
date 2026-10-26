"""CPU cache fidelity and provenance checks; no private archives required."""
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_figure_data", ROOT / "analysis/prigm/figures/prepare_figure_data.py")
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


@pytest.fixture(scope="module")
def frozen():
    return cache.load_frozen(ROOT / "paper/prigm/paper_evidence_manifest.json")


def test_observed_weights_and_features_are_the_frozen_reduction(frozen):
    _, paper, _ = frozen
    rng = np.random.default_rng(173)
    data = {"train": rng.normal(size=(7, 192)), "test": rng.normal(size=(11, 192)),
            "train_weights": np.array([1., 2., 3., 1., 2., 1., 1.]),
            "eval_weights": np.arange(1., 12.), "scale": np.array(2.3)}
    for target in ("original_uniform_windows", "six_UTC_bins_equal"):
        record = {"scale": 2.3, "target": target}
        seen = []

        def reduce(x, w, scale):
            seen.append(np.array(w))
            return paper._dart_distribution(x, w, scale)

        actual = cache.observed_distributions(data, record, reduce)
        for i, split in enumerate(("train", "test")):
            weights = (np.ones(len(data[split])) if target == "original_uniform_windows"
                       else data["train_weights" if split == "train" else "eval_weights"])
            np.testing.assert_array_equal(seen[i], weights)
            expected = paper._dart_distribution(data[split], weights, 2.3)
            for key in ("features", "weights", "power"):
                np.testing.assert_array_equal(actual[split][key], expected[key])


def test_streams_and_mean_power_remain_separate(frozen):
    _, paper, _ = frozen
    wave = np.cos(2 * np.pi * np.arange(192) / 192)
    draws = np.broadcast_to(wave, (3, 1024, 192)).copy()
    draws[1] *= 2
    draws[2] = (-1.) ** np.arange(192)
    draws[:, -1] += 100  # retain the full feature support, including its outlier
    samples = cache.draw_distributions(draws, 3., paper._dart_distribution)
    means = cache.draw_distributions(draws, 3., paper._dart_distribution, means=True)
    assert len(samples) == len(means) == 3
    for stream, result, mean in zip(draws, samples, means):
        expected = paper._dart_distribution(stream, np.full(1024, 1 / 1024), 3.)
        for key in ("features", "weights", "power"):
            np.testing.assert_array_equal(result[key], expected[key])
        assert set(mean) == {"power"}
        np.testing.assert_array_equal(mean["power"], expected["power"])
        assert result["features"][-1, 0] == pytest.approx(300.)
    assert samples[0]["power"][0] == pytest.approx(4.5)
    assert samples[1]["power"][0] == pytest.approx(18.)
    assert samples[2]["power"][-1] == pytest.approx(9.)
    assert samples[0]["features"].shape == (1024, 3)
    assert samples[0]["power"].shape == (96,)


@pytest.mark.parametrize("shape,dtype", [((1, 1024, 192), "float64"),
                                         ((3, 512, 192), "float64"),
                                         ((3, 1024, 192), "float32")])
def test_saved_draw_contract_rejects_reduced_or_lower_precision_arrays(shape, dtype, frozen):
    with pytest.raises(ValueError, match="shape or dtype"):
        cache.draw_distributions(np.zeros(shape, dtype=dtype), 1., frozen[1]._dart_distribution)


def test_manifest_and_source_tampering_rejected(tmp_path, frozen):
    path = tmp_path / "manifest.json"
    path.write_bytes((ROOT / "paper/prigm/paper_evidence_manifest.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Scientific manifest hash"):
        cache.load_frozen(path)
    (tmp_path / "src").mkdir()
    sources = frozen[2]
    for name in sources:
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    with (tmp_path / "src/prigm_v2_metrics.py").open("ab") as handle:
        handle.write(b"\n# changed\n")
    with pytest.raises(ValueError, match="Source hash differs: src/prigm_v2_metrics.py"):
        cache.load_frozen(ROOT / "paper/prigm/paper_evidence_manifest.json", root=tmp_path)


def test_input_receipts_are_authenticated_and_public_metadata_excludes_locations(tmp_path):
    path = tmp_path / "operator-private.npz"
    path.write_bytes(b"retained arrays")
    expected = {"core/prepared/record.npz": cache.digest(path)}
    mapping = {"files": {next(iter(expected)): {"local_path": str(path),
               "sha256": cache.digest(path), "bytes": path.stat().st_size, "role": "prepared"}}}
    inputs = cache.Inputs(mapping, expected)
    inputs.bind(next(iter(expected)))
    inputs.recheck()
    assert str(tmp_path) not in json.dumps(inputs.files)
    assert inputs.files == {next(iter(expected)): {"sha256": cache.digest(path), "bytes": 15}}
    bad = deepcopy(mapping)
    bad["files"][next(iter(expected))]["bytes"] += 1
    with pytest.raises(ValueError, match="Transfer size"):
        cache.Inputs(bad, expected).bind(next(iter(expected)))
    path.write_bytes(b"replaced arrays")
    with pytest.raises(ValueError, match="changed during extraction"):
        inputs.recheck()
    with pytest.raises(ValueError, match="Input hash differs"):
        cache.Inputs(mapping, expected).bind(next(iter(expected)))


@pytest.mark.parametrize("name", ["../data", "/private/data", "a//b", "a/./b", "a\\b"])
def test_private_locations_cannot_become_public_logical_names(name):
    with pytest.raises(ValueError, match="logical input name"):
        cache.Inputs({name: "/irrelevant"}, {name: "0" * 64})


def synthetic_plan(tmp_path, *, omit=None, wrong_checkpoint=False):
    """Small authenticated worker metadata; arrays need no contents for planning."""
    source_pins = {"src/eval_gaussian_final_checkpoints.py": "1" * 64,
                   "src/prigm_v2_metrics.py": "2" * 64}
    stage_pins = {"src/prigm_v2_data.py": "6" * 64}
    diagnostics = {"prior_seeds": list(cache.PRIOR_SEEDS),
                   "reference_seeds": list(cache.REFERENCE_SEEDS), "prior_samples": 1024}
    files, mapping, records = {}, {}, {}

    def write(logical, value):
        path = tmp_path / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        files[logical] = cache.digest(path)
        mapping[logical] = str(path)

    for name in cache.RECORDS:
        models = {f"{arm}/{seed}/original": {"checkpoint": "final" if name.startswith("dart_") else "selected",
                  "checkpoint_sha256": hashlib.sha256(f"{name}/{arm}/{seed}".encode()).hexdigest(),
                  "observation_variance": .001 if arm in "AC" else .1}
                  for arm, seed in itertools.product(cache.ARMS, cache.SEEDS)}
        records[name] = {"prepared_sha256": "3" * 64, "models": models}
    groups = [["44402t2016"], ["23461t2022", "46407t2011", "52403t2020"],
              ["dart_44402_2018"], ["dart_21415_2018", "dart_21418_2018"]]
    for batch, names in zip(cache.BATCHES, groups):
        fresh = batch.startswith("stage")
        all_jobs = []
        arms = "ACDB" if fresh else "ABCD"
        reverse = {arm: key for key, arm in cache.ARM_MAP.items()}
        for name, seed, arm in itertools.product(names, cache.SEEDS, arms):
            if (name, arm, seed) == omit:
                continue
            all_jobs.append([name, reverse[arm][0], seed, reverse[arm][1], "final"] if fresh
                            else [name, arm, seed, "selected"])
        for worker in range(4):
            prefix = f"{batch}/worker-{worker}"
            common = {"schema_version": 2,
                      "purpose": "dart_stage_checkpoint_diagnostics" if fresh else "dart_fixed_checkpoint_diagnostics",
                      "stage_manifest_sha256" if fresh else "core_manifest_sha256": "4" * 64,
                      "evaluator_source_sha256": {k.removeprefix("src/"): v for k, v in source_pins.items()
                                                 if fresh is False or k.endswith("checkpoints.py")}}
            protocol = {**common, "worker_index": worker, "worker_count": 4,
                        "all_jobs": all_jobs, "jobs": all_jobs[worker::4], "dart_diagnostics": diagnostics}
            if fresh:
                common["stage_source_sha256"] = stage_pins
                protocol["stage_source_sha256"] = stage_pins
            write(prefix + "/protocol.json", protocol)
            artifacts = {}
            for job in protocol["jobs"]:
                name, condition, seed = job[:3]
                arm = cache.ARM_MAP[condition, job[3]] if fresh else condition
                public = records[name]["models"][f"{arm}/{seed}/original"]
                stem = "_".join(map(str, job[:-1])) + "-" + job[-1]
                reference = f"references-{name}-{'low' if arm in 'AC' else 'high'}.npz"
                write(prefix + "/" + reference, b"reference")
                write(prefix + "/" + stem + ".npz", b"samples")
                meta = {**public, "dataset": name, "seed": seed, "dimension": 192,
                        "prepared_sha256": "3" * 64,
                        "protocol_sha256": files[prefix + "/protocol.json"],
                        "array_sha256": files[prefix + "/" + stem + ".npz"],
                        "references": {"file": reference, "sha256": files[prefix + "/" + reference],
                                       "observation_variance": public["observation_variance"]}}
                meta.update({"condition": condition, "variance_label": job[3]} if fresh else {"arm": arm})
                if wrong_checkpoint:
                    meta["checkpoint_sha256"] = "9" * 64
                write(prefix + "/" + stem + ".json", meta)
                for filename in (reference, stem + ".npz", stem + ".json"):
                    artifacts[filename] = files[prefix + "/" + filename]
            write(prefix + "/complete.json", {**common, "protocol_sha256": files[prefix + "/protocol.json"],
                                               "artifact_sha256": artifacts})
    files.update({"stage/source/" + name: digest for name, digest in stage_pins.items()})
    manifest = {"input_index": {"source_sha256": {**source_pins, **stage_pins}}, "dart_extension": {
        "records": records, "methods": {"diagnostics": diagnostics},
        "input_index": {"files": files, "stage_manifest_sha256": "4" * 64, "core_manifest_sha256": "4" * 64}}}
    return manifest, cache.Inputs(mapping, files)


def test_protocol_plan_retains_84_identities_and_exactly_seven_ppca_copies(tmp_path):
    manifest, inputs = synthetic_plan(tmp_path)
    jobs, refs = cache.plan_inputs(manifest, inputs)
    assert set(jobs) == set(itertools.product(cache.RECORDS, cache.ARMS, cache.SEEDS))
    assert set(refs) == set(cache.RECORDS)
    assert all("/worker-0/" in name and name.endswith("-low.npz") for name in refs.values())
    assert {n for n in inputs.files if "/references-" in n} == set(refs.values())


def test_plan_distinguishes_historical_stage_and_current_derivation(tmp_path):
    manifest, inputs = synthetic_plan(tmp_path)
    manifest["input_index"]["source_sha256"]["src/prigm_v2_data.py"] = "8" * 64
    jobs, refs = cache.plan_inputs(manifest, inputs)
    assert len(jobs) == 84 and len(refs) == 7


@pytest.mark.parametrize("producer_map", [{}, {"src/prigm_v2_data.py": "9" * 64}])
def test_plan_rejects_resealed_wrong_stage_producer(tmp_path, producer_map):
    manifest, inputs = synthetic_plan(tmp_path)
    prefix = "stage-batch-000/worker-0/"
    protocol_name, completion_name = prefix + "protocol.json", prefix + "complete.json"
    protocol_path, completion_path = Path(inputs.mapping[protocol_name]), Path(inputs.mapping[completion_name])
    protocol = json.loads(protocol_path.read_text())
    completion = json.loads(completion_path.read_text())
    protocol["stage_source_sha256"] = producer_map
    completion["stage_source_sha256"] = producer_map
    protocol_path.write_text(json.dumps(protocol))
    inputs.expected[protocol_name] = cache.digest(protocol_path)
    completion["protocol_sha256"] = inputs.expected[protocol_name]
    for name, location in inputs.mapping.items():
        if name.startswith(prefix) and name.endswith(".json") and name not in {protocol_name, completion_name}:
            path = Path(location)
            metadata = json.loads(path.read_text())
            metadata["protocol_sha256"] = inputs.expected[protocol_name]
            path.write_text(json.dumps(metadata))
            inputs.expected[name] = cache.digest(path)
            completion["artifact_sha256"][path.name] = inputs.expected[name]
    completion_path.write_text(json.dumps(completion))
    inputs.expected[completion_name] = cache.digest(completion_path)
    with pytest.raises(ValueError, match="Stage source identity differs"):
        cache.plan_inputs(manifest, inputs)


@pytest.mark.parametrize("defect,message", [("omit", "Incomplete 84-fit coverage"),
                                            ("wrong_checkpoint", "metadata identity")])
def test_protocol_plan_rejects_missing_fit_and_inconsistent_checkpoint(tmp_path, defect, message):
    options = {"omit": ("dart_21418_2018", "B", 31415)} if defect == "omit" else {"wrong_checkpoint": True}
    manifest, inputs = synthetic_plan(tmp_path, **options)
    with pytest.raises(ValueError, match=message):
        cache.plan_inputs(manifest, inputs)


def test_existing_output_is_refused_before_work(tmp_path):
    with pytest.raises(ValueError, match="must be fresh"):
        cache.prepare(tmp_path / "unused-map.json", tmp_path, tmp_path / "unused-manifest.json")
