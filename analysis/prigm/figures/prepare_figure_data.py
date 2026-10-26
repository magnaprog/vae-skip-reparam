#!/usr/bin/env python3
"""Cache the frozen DART figure reductions from authenticated saved arrays.

Run on CPU beside the repository's 13 frozen ``src/`` files. The input map is
private: either logical names mapped to local paths, or a transfer receipt with
``files[logical] = {local_path, sha256, bytes, ...}``. Only logical names and
digests enter the output. No checkpoint loading, fitting or sampling occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import itertools
import json
from pathlib import Path, PurePosixPath
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCIENTIFIC_MANIFEST_SHA256 = "33a571728e3126a11ef8957e56319a363f130dab8dc7853fbd4f82f19ef2ddeb"
RECORDS = ("23461t2022", "44402t2016", "46407t2011", "52403t2020",
           "dart_44402_2018", "dart_21415_2018", "dart_21418_2018")
SEEDS = (42, 1729, 31415)
PRIOR_SEEDS = (71001, 71002, 71003)
REFERENCE_SEEDS = (74001, 74002, 74003)
ARMS = ("A", "B", "C", "D")
ARM_MAP = {("base", "low"): "A", ("raw", "low"): "C",
           ("base", "high"): "D", ("raw", "high"): "B"}
BATCHES = ("core-batch-000", "core-batch-001", "stage-batch-000", "stage-batch-001")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular input file")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def logical_name(name):
    require(isinstance(name, str) and name and "\\" not in name
            and not PurePosixPath(name).is_absolute()
            and all(part not in ("", ".", "..") and ":" not in part
                    for part in name.split("/"))
            and all(ord(c) >= 32 for c in name), "Invalid logical input name")
    return name


class Inputs:
    """Bind private locations to the trusted scientific inventory."""

    def __init__(self, mapping, expected):
        self.mapping = mapping.get("files", mapping)
        require(isinstance(self.mapping, dict), "Input map must contain a files mapping")
        self.expected = expected
        self.paths = {}
        self.files = {}
        for name in self.mapping:
            logical_name(name)
            require(name in expected, "Input map contains an unpinned logical name")

    def bind(self, name):
        require(name in self.mapping, f"Missing required input: {name}")
        entry = self.mapping[name]
        if isinstance(entry, str):
            entry = {"local_path": entry}
        require(isinstance(entry, dict) and isinstance(entry.get("local_path"), str),
                f"Invalid input binding: {name}")
        path = Path(entry["local_path"])
        if name not in self.files:
            actual = digest(path)
            size = path.stat().st_size
            require(actual == self.expected[name], f"Input hash differs: {name}")
            require(entry.get("sha256", actual) == actual, f"Transfer hash differs: {name}")
            require(entry.get("bytes", size) == size, f"Transfer size differs: {name}")
            self.paths[name] = path
            self.files[name] = {"sha256": actual, "bytes": size}
        return self.paths[name]

    def json(self, name):
        return read_json(self.bind(name))

    def recheck(self):
        for name, path in self.paths.items():
            require(digest(path) == self.files[name]["sha256"]
                    and path.stat().st_size == self.files[name]["bytes"],
                    f"Input changed during extraction: {name}")


def check_sources(root, sources):
    for name, expected in sources.items():
        logical_name(name)
        require(digest(root / name) == expected, f"Source hash differs: {name}")


def load_frozen(manifest_path, root=ROOT):
    require(digest(manifest_path) == SCIENTIFIC_MANIFEST_SHA256,
            "Scientific manifest hash differs")
    manifest = read_json(manifest_path)
    sources = manifest["input_index"]["source_sha256"]
    require(len(sources) == 13, "Expected 13 frozen scientific sources")
    check_sources(root, sources)
    # The frozen builder imports only CPU derivation dependencies. Authenticate
    # before importing it, and refuse an already loaded module from another tree.
    for name in sources:
        module = sys.modules.get(Path(name).stem)
        if module is not None:
            require(Path(module.__file__).resolve() == (root / name).resolve(),
                    f"Scientific module loaded from another tree: {name}")
    sys.path.insert(0, str(root / "src"))
    paper = importlib.import_module("build_prigm_v2_paper")
    paper.validate_combined_manifest(manifest)
    check_sources(root, sources)
    return manifest, paper, dict(sources)


def plan_inputs(manifest, inputs):
    """Resolve all 84 fits and one worker-0 low-variance PPCA copy per record."""
    extension = manifest["dart_extension"]
    records, index = extension["records"], extension["input_index"]
    require(set(records) == set(RECORDS), "Expected exactly seven DART records")
    jobs, references = {}, {}
    stage_producers = {name.removeprefix("stage/source/"): digest
                       for name, digest in inputs.expected.items()
                       if name.startswith("stage/source/")}
    require(stage_producers, "Missing historical stage source identities")
    diagnostics = extension["methods"]["diagnostics"]
    require(diagnostics["prior_seeds"] == list(PRIOR_SEEDS)
            and diagnostics["reference_seeds"] == list(REFERENCE_SEEDS)
            and diagnostics["prior_samples"] == 1024, "Sampling stream identity differs")
    for batch in BATCHES:
        fresh = batch.startswith("stage")
        manifest_key = "stage_manifest_sha256" if fresh else "core_manifest_sha256"
        purpose = "dart_stage_checkpoint_diagnostics" if fresh else "dart_fixed_checkpoint_diagnostics"
        all_jobs = None
        for worker in range(4):
            prefix = f"{batch}/worker-{worker}"
            protocol_name = prefix + "/protocol.json"
            protocol = inputs.json(protocol_name)
            completion = inputs.json(prefix + "/complete.json")
            require(protocol["worker_index"] == worker and protocol["worker_count"] == 4,
                    "Worker identity differs")
            require(protocol["schema_version"] == completion["schema_version"] == 2
                    and protocol["purpose"] == completion["purpose"] == purpose
                    and protocol[manifest_key] == completion[manifest_key] == index[manifest_key]
                    and protocol["dart_diagnostics"] == diagnostics,
                    "Diagnostic protocol identity differs")
            require(completion["protocol_sha256"] == inputs.expected[protocol_name],
                    "Completion protocol binding differs")
            source_names = ["eval_gaussian_final_checkpoints.py"]
            if not fresh:
                source_names.append("prigm_v2_metrics.py")
            evaluator = {n: manifest["input_index"]["source_sha256"]["src/" + n]
                         for n in source_names}
            require(protocol["evaluator_source_sha256"] == completion["evaluator_source_sha256"] == evaluator,
                    "Evaluator source identity differs")
            if fresh:
                stage_sources = protocol["stage_source_sha256"]
                require(completion["stage_source_sha256"] == stage_sources
                        and stage_sources == stage_producers, "Stage source identity differs")
            if all_jobs is None:
                all_jobs = protocol["all_jobs"]
            require(protocol["all_jobs"] == all_jobs and protocol["jobs"] == all_jobs[worker::4],
                    "Worker job ownership differs")
            for job in protocol["jobs"]:
                require(len(job) == (5 if fresh else 4), "Invalid diagnostic job")
                name, condition, seed = job[:3]
                arm = ARM_MAP.get((condition, job[3])) if fresh else condition
                identity = (name, arm, seed)
                require(name in records and arm in ARMS and type(seed) is int and seed in SEEDS,
                        "Invalid fit identity")
                require(identity not in jobs, "Duplicate fit identity")
                endpoint = "final" if fresh else "selected"
                require(job[-1] == endpoint and name.startswith("dart_") == fresh,
                        "Checkpoint selection rule differs")
                stem = "_".join(map(str, job[:-1])) + "-" + job[-1]
                meta_name, array_name = prefix + "/" + stem + ".json", prefix + "/" + stem + ".npz"
                meta = inputs.json(meta_name)
                public = records[name]["models"][f"{arm}/{seed}/original"]
                require(meta["dataset"] == name and meta["seed"] == seed
                        and meta["checkpoint"] == endpoint == public["checkpoint"]
                        and meta["checkpoint_sha256"] == public["checkpoint_sha256"]
                        and meta["prepared_sha256"] == records[name]["prepared_sha256"]
                        and meta["observation_variance"] == public["observation_variance"]
                        and meta["dimension"] == 192, "Diagnostic metadata identity differs")
                require((meta["condition"], meta["variance_label"]) == (condition, job[3]) if fresh
                        else meta["arm"] == arm, "Diagnostic condition differs")
                require(meta["protocol_sha256"] == inputs.expected[protocol_name]
                        and meta["array_sha256"] == inputs.expected[array_name],
                        "Diagnostic artifact binding differs")
                ref = meta["references"]
                require(Path(ref["file"]).name == ref["file"], "Invalid reference filename")
                ref_name = prefix + "/" + ref["file"]
                require(ref["sha256"] == inputs.expected[ref_name], "Reference identity differs")
                for logical in (meta_name, array_name, ref_name):
                    require(completion["artifact_sha256"][Path(logical).name] == inputs.expected[logical],
                            "Completion artifact binding differs")
                inputs.bind(array_name)
                jobs[identity] = (array_name, meta)
                if worker == 0 and arm == "A":
                    require(ref["observation_variance"] == public["observation_variance"],
                            "PPCA reference variance identity differs")
                    require(name not in references or references[name] == ref_name,
                            "Multiple selected PPCA copies")
                    references[name] = ref_name
    require(set(jobs) == set(itertools.product(RECORDS, ARMS, SEEDS)), "Incomplete 84-fit coverage")
    require(set(references) == set(RECORDS), "Incomplete seven-record PPCA coverage")
    for ref_name in references.values():
        inputs.bind(ref_name)
    return jobs, references


def observed_distributions(data, record, distribution):
    require(float(data["scale"]) == record["scale"], "Prepared training scale differs")
    fresh = record["target"] == "six_UTC_bins_equal"
    require(fresh or record["target"] == "original_uniform_windows", "Unknown window measure")
    result = {}
    for split in ("train", "test"):
        # Preserve the frozen renderer's arithmetic: old records start with ones,
        # while fresh records use their archived weights without pre-normalizing.
        weights = data["train_weights" if split == "train" else "eval_weights"] if fresh else np.ones(len(data[split]))
        result[split] = distribution(data[split], weights, record["scale"])
    return result


def draw_distributions(draws, scale, distribution, *, means=False):
    require(draws.shape == (3, 1024, 192) and draws.dtype == np.dtype("float64"),
            "Invalid saved prior array shape or dtype")
    result = []
    for stream in draws:
        value = distribution(stream, np.full(1024, 1 / 1024), scale)
        result.append({"power": value["power"]} if means else value)
    return result


def prepare(input_map, output_dir, manifest_path, *, root=ROOT):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Output directory must be fresh")
    manifest, paper, sources = load_frozen(manifest_path, root)
    source_name = "analysis/prigm/figures/prepare_figure_data.py"
    sources[source_name] = digest(root / source_name)
    inputs = Inputs(read_json(input_map), manifest["dart_extension"]["input_index"]["files"])
    jobs, references = plan_inputs(manifest, inputs)
    arrays, distribution_index = {}, {}

    def store(prefix, summary):
        for field in ("features", "weights", "power"):
            if field in summary:
                key = prefix + "/" + field
                require(key not in arrays, "Duplicate cache array")
                value = np.ascontiguousarray(summary[field])
                require(np.isfinite(value).all(), "Nonfinite cache array")
                arrays[key] = value
        return prefix

    for name in RECORDS:
        record = manifest["dart_extension"]["records"][name]
        kind = "stage" if name.startswith("dart_") else "core"
        logical = f"{kind}/prepared/{name}.npz"
        require(inputs.expected[logical] == record["prepared_sha256"], "Prepared identity differs")
        case = distribution_index[name] = {}
        with np.load(inputs.bind(logical), allow_pickle=False) as data:
            for split, summary in observed_distributions(data, record, paper._dart_distribution).items():
                case[split] = [[store(f"{name}/{split}/0/0", summary)]]
            test_count = len(data["test"])
        for arm in ARMS:
            case[arm], case[arm + " means"] = [], []
            for seed in SEEDS:
                logical, meta = jobs[name, arm, seed]
                require(meta["test_windows"] == test_count, "Test window count differs")
                with np.load(inputs.bind(logical), allow_pickle=False) as archive:
                    for key, role, means in (("original__prior_samples", arm, False),
                                             ("original__prior_means", arm + " means", True)):
                        require(meta["array_shapes"][key] == [3, 1024, 192], "Declared prior shape differs")
                        summaries = draw_distributions(archive[key], record["scale"], paper._dart_distribution,
                                                       means=means)
                        case[role].append([store(f"{name}/{role.replace(' ', '_')}/{seed}/{stream}", value)
                                           for stream, value in zip(PRIOR_SEEDS, summaries)])
        with np.load(inputs.bind(references[name]), allow_pickle=False) as archive:
            summaries = draw_distributions(archive["ppca_fitted__prior_samples"], record["scale"],
                                           paper._dart_distribution)
            case["PPCA"] = [[store(f"{name}/PPCA/0/{stream}", value)
                             for stream, value in zip(REFERENCE_SEEDS, summaries)]]
    require(set(inputs.files) == set(inputs.mapping), "Input map contains unused files")
    require(len(inputs.files) == 214, "Expected 214 consumed diagnostic inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    cache = output_dir / "plot_data.npz"
    with cache.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    array_index = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                   for key, value in arrays.items()}
    result = {"schema_version": 1, "scientific_manifest_sha256": SCIENTIFIC_MANIFEST_SHA256,
              "cache_sha256": digest(cache), "files": inputs.files, "source_sha256": sources,
              "versions": {"python": platform.python_version(),
                           **{n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pandas")}},
              "training_seeds": list(SEEDS), "prior_stream_seeds": list(PRIOR_SEEDS),
              "reference_stream_seeds": list(REFERENCE_SEEDS),
              "distribution_index": distribution_index, "arrays": array_index}
    # Publish the manifest last, after rechecking every consumed input and source.
    inputs.recheck()
    check_sources(root, sources)
    require(digest(manifest_path) == SCIENTIFIC_MANIFEST_SHA256, "Scientific manifest changed during extraction")
    with (output_dir / "plot_data_manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper/prigm/paper_evidence_manifest.json")
    args = parser.parse_args(argv)
    result = prepare(args.input_map, args.output_dir, args.manifest)
    print(json.dumps({"cache_sha256": result["cache_sha256"], "arrays": len(result["arrays"]),
                      "input_files": len(result["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
