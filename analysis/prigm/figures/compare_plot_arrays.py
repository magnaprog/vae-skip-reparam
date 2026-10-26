#!/usr/bin/env python3
"""Compare numeric plotting arguments through frozen and revised DART loops.

The same authenticated compact cache supplies both rendering paths. This checks
rendering from retained reductions, not reproduction of the historical PNGs or
extraction from raw diagnostic arrays. No image files or model outputs are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

EVIDENCE = "33a571728e3126a11ef8957e56319a363f130dab8dc7853fbd4f82f19ef2ddeb"
EXPECTED = {
    "reconstruction_generation": (3, {"errorbar": 6}),
    "full_record_context": (4, {"scatter": 4}),
    "dart_ecdf_historical": (12, {"step": 84, "fill_between": 84}),
    "dart_ecdf_fresh": (9, {"step": 63, "fill_between": 63}),
    "dart_spectra_historical": (8, {"plot": 56, "fill_between": 56}),
    "dart_spectra_fresh": (6, {"plot": 42, "fill_between": 42}),
    "dart_variance_refit": (4, {"plot": 336, "hlines": 252}),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def import_file(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array(value):
    if value is None:
        return None
    value = np.ma.asarray(value)
    require(value.dtype.kind in "biufc", "Non-numeric plotting argument")
    # Copy because Matplotlib or a later loop might retain and mutate an input.
    return {"data": np.array(np.ma.getdata(value), copy=True, order="C"),
            "mask": np.array(np.ma.getmaskarray(value), copy=True, order="C")}


def arguments(method, args, kwargs):
    names = {"plot": ("x", "y"), "step": ("x", "y"),
             "fill_between": ("x", "y1", "y2"), "scatter": ("x", "y"),
             "hlines": ("y", "xmin", "xmax"),
             "errorbar": ("x", "y", "yerr", "xerr")}[method]
    result = {}
    for index, name in enumerate(names):
        if index < len(args):
            value = args[index]
        else:
            value = kwargs.get(name)
        # Only errorbar's optional error arrays may be absent in these loops.
        require(value is not None or (method == "errorbar" and name in ("xerr", "yerr")),
                "Missing numeric plotting argument")
        result[name] = array(value)
    if method == "errorbar":
        for coordinate in ("x", "y"):
            errors = result[coordinate + "err"]
            if errors is None:
                continue
            center = np.ma.array(result[coordinate]["data"], mask=result[coordinate]["mask"])
            err = np.ma.array(errors["data"], mask=errors["mask"])
            low, high = (err[0], err[1]) if err.ndim == 2 else (err, err)
            result[coordinate + "_endpoint_low"] = array(center - low)
            result[coordinate + "_endpoint_high"] = array(center + high)
    return result


class Capture:
    """Record outer plotting calls; ignore style and internal helper calls."""

    def __init__(self):
        self.depth = 0
        self.pending = {}
        self.figures = {}
        self.stack = ExitStack()

    def wrap(self, method, original):
        def wrapped(axis, *args, **kwargs):
            if self.depth == 0:
                record = {"axis": axis.figure.axes.index(axis), "method": method,
                          "arguments": arguments(method, args, kwargs),
                          "step_convention": kwargs.get("where", "pre") if method == "step" else kwargs.get("step") if method == "fill_between" else None}
                self.pending.setdefault(axis.figure, []).append(record)
            self.depth += 1
            try:
                return original(axis, *args, **kwargs)
            finally:
                self.depth -= 1
        return wrapped

    def save(self, fig, file, *args, **kwargs):
        name = Path(file).stem
        require(name in EXPECTED and name not in self.figures, "Unexpected or duplicate figure")
        calls = self.pending.pop(fig, [])
        require(len(fig.axes) == EXPECTED[name][0], "Unexpected axis count: " + name)
        counts = Counter(call["method"] for call in calls)
        require(dict(counts) == EXPECTED[name][1], "Unexpected call coverage: " + name)
        self.figures[name] = {"axes": len(fig.axes), "calls": calls}
        # Deliberately suppress all PDF/PNG output and drawing.

    def __enter__(self):
        for method in ("step", "fill_between", "plot", "errorbar", "hlines", "scatter"):
            self.stack.enter_context(patch.object(Axes, method, self.wrap(method, getattr(Axes, method))))
        self.stack.enter_context(patch.object(Figure, "savefig", lambda fig, *a, **k: self.save(fig, *a, **k)))
        self.stack.enter_context(patch.object(Figure, "tight_layout", lambda *a, **k: None))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        plt.close("all")
        if not exc[0]:
            require(not self.pending, "Unsealed figure calls")
            require(set(self.figures) == set(EXPECTED), "Incomplete seven-figure coverage")


class WindowToken:
    """Only identifies an observed cached partition, without fabricating values."""

    def __init__(self, record, split, size):
        self.record, self.split, self.size = record, split, size

    def __len__(self):
        return self.size


def baseline_figures(frozen, evidence, reduced, directory):
    records = evidence["dart_extension"]["records"]
    private = {"data": {}, "model_paths": {}, "reference_paths": {}}
    routes = {}
    observed_calls, archive_calls = [], []
    for name in frozen._dart_order(records):
        case = reduced[name]
        data = private["data"][name] = {"scale": records[name]["scale"]}
        for split in ("train", "test"):
            value = case[split][0][0]
            data[split] = WindowToken(name, split, len(value["weights"]))
            data["train_weights" if split == "train" else "eval_weights"] = value["weights"]
        for arm in frozen.ARMS:
            for i, seed in enumerate(frozen.SEEDS):
                name_key = f"{name}/{arm}/{seed}"
                route = "cache-model/" + name_key
                private["model_paths"][name_key] = route
                routes[route] = (name, arm, i)
        route = "cache-reference/" + name
        private["reference_paths"][name + "/low"] = route
        routes[route] = (name, "PPCA", 0)

    def distribution(token, weights, scale):
        require(isinstance(token, WindowToken), "Unexpected raw distribution call")
        name, split = token.record, token.split
        require(scale == records[name]["scale"], "Observed scale differs")
        value = reduced[name][split][0][0]
        expected = value["weights"] if records[name]["target"] == "six_UTC_bins_equal" else np.ones(len(token))
        require(np.array_equal(weights, expected), "Observed weighting route differs")
        observed_calls.append((name, split))
        return value

    def archive_summary(route, files, keys, scale):
        require(route in routes, "Unknown cache route")
        name, role, index = routes[route]
        require(scale == records[name]["scale"], "Generated scale differs")
        require(files is evidence["dart_extension"]["input_index"]["files"], "Wrong scientific inventory")
        archive_calls.append(route)
        if role == "PPCA":
            require(tuple(keys) == ("ppca_fitted__prior_samples",), "Unexpected reference fields")
            return {keys[0]: reduced[name]["PPCA"][0]}
        require(tuple(keys) == ("original__prior_samples", "original__prior_means"), "Unexpected neural fields")
        return {keys[0]: reduced[name][role][index], keys[1]: reduced[name][role + " means"][index]}

    with patch.object(frozen, "_dart_distribution", distribution), patch.object(frozen, "_dart_archive_summary", archive_summary):
        frozen._dart_comparison_figure(directory, evidence["dart_extension"], evidence["combined_input_sha256"])
        frozen._dart_extension_figures(directory, evidence["dart_extension"], private, evidence["combined_input_sha256"])
    require(len(observed_calls) == len(set(observed_calls)) == 14, "Incomplete observed cache routes")
    require(len(archive_calls) == len(set(archive_calls)) == 91, "Incomplete generated cache routes")
    return {"observed_partitions": 14, "neural_fits": 84, "fitted_PPCA_references": 7}


def baseline_context(frozen, evidence, repo, directory):
    # Use the authenticated frozen point reader, rather than executing unrelated
    # legacy figures from its surrounding multi-figure function.
    fig, axes = plt.subplots(2, 2)
    sources = evidence["core"]["public_source_sha256"]
    for axis, name in zip(axes.flat, frozen.DATASETS[1:]):
        matches = [key for key in sources if Path(key).name == name + ".txt"]
        require(len(matches) == 1, "Ambiguous context source")
        key = matches[0]
        value = frozen.legacy.read_observed_context(repo / key, sources[key])
        axis.scatter(value["days"], value["heights"])
    fig.savefig(directory / "full_record_context.png")
    plt.close(fig)


def compare_call(a, b, location):
    require(a["axis"] == b["axis"] and a["method"] == b["method"], "Call order differs: " + location)
    require(a.get("step_convention") == b.get("step_convention"), "Step convention differs: " + location)
    require(a["arguments"].keys() == b["arguments"].keys(), "Fields differ: " + location)
    fields = 0
    elements = 0
    masked = 0
    for key in a["arguments"]:
        aa, bb = a["arguments"][key], b["arguments"][key]
        if aa is None or bb is None:
            require(aa is None and bb is None, "Absent value differs: " + location)
            continue
        for part in ("data", "mask"):
            av, bv = aa[part], bb[part]
            require(av.shape == bv.shape and av.dtype == bv.dtype
                    and av.tobytes(order="C") == bv.tobytes(order="C"),
                    "Array differs: " + location + "/" + key + "/" + part)
        fields += 1
        elements += aa["data"].size
        masked += int(aa["mask"].sum())
    return fields, elements, masked


def transcript_digest(figure):
    h = hashlib.sha256()
    for call in figure["calls"]:
        h.update(json.dumps([call["axis"], call["method"], call.get("step_convention")], separators=(",", ":")).encode())
        for key, value in call["arguments"].items():
            h.update(key.encode() + b"\0")
            if value is None:
                h.update(b"null\0")
                continue
            for part in ("data", "mask"):
                arr = value[part]
                h.update(json.dumps([part, list(arr.shape), arr.dtype.str], separators=(",", ":")).encode())
                h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def canonical_step(reference, revised, location):
    """Expand revised post steps on the full frozen grid, without new helpers.

    Exact membership, monotonicity and endpoints rule out hidden added jumps or
    truncated support. Right-continuous values at every original breakpoint then
    establish equality on all intervals between those breakpoints as well.
    """
    require(reference["method"] == revised["method"] in ("step", "fill_between"),
            "Unexpected ECDF method: " + location)
    require(reference.get("step_convention") == revised.get("step_convention") == "post",
            "ECDF convention differs: " + location)
    x, compact = reference["arguments"]["x"], revised["arguments"]["x"]
    full, shorter = x["data"], compact["data"]
    require(full.ndim == shorter.ndim == 1 and len(full) > 1 and len(shorter) > 1
            and full.dtype == shorter.dtype and not x["mask"].any() and not compact["mask"].any()
            and np.all(np.diff(full) > 0) and np.all(np.diff(shorter) > 0),
            "Invalid ECDF support: " + location)
    positions = np.searchsorted(full, shorter, side="left")
    require(np.all(positions < len(full)) and np.array_equal(full[positions], shorter)
            and shorter[0] == full[0] and shorter[-1] == full[-1],
            "ECDF breakpoints or endpoints differ: " + location)
    take = np.searchsorted(shorter, full, side="right") - 1
    expanded = {**revised, "arguments": {"x": x}}
    for key, value in revised["arguments"].items():
        if key == "x":
            continue
        require(value is not None and value["data"].shape == shorter.shape,
                "ECDF ordinate shape differs: " + location)
        expanded["arguments"][key] = {part: value[part][take] for part in ("data", "mask")}
    return expanded


def compare_figures(before, after):
    result = {}
    require(set(before) == set(after) == set(EXPECTED), "Figure coverage differs")
    for name in EXPECTED:
        old, new = before[name], after[name]
        require(old["axes"] == new["axes"] and len(old["calls"]) == len(new["calls"]), "Figure shape differs")
        ecdf = name.startswith("dart_ecdf_")
        count = np.zeros(3, dtype=np.int64)
        canonical = {"axes": new["axes"], "calls": []}
        old_points = new_points = 0
        for index, (a, b) in enumerate(zip(old["calls"], new["calls"])):
            location = f"{name}/{index}"
            if ecdf:
                old_points += a["arguments"]["x"]["data"].size
                new_points += b["arguments"]["x"]["data"].size
                b = canonical_step(a, b, location)
            count += compare_call(a, b, location)
            canonical["calls"].append(b)
        old_digest, canonical_digest = transcript_digest(old), transcript_digest(canonical)
        require(old_digest == canonical_digest, "Canonical transcript digest differs")
        result[name] = {"axes": old["axes"], "calls": len(old["calls"]),
                        "call_types": dict(Counter(c["method"] for c in old["calls"])),
                        "numeric_array_pairs": int(count[0]), "numeric_elements": int(count[1]),
                        "masked_elements": int(count[2]),
                        "baseline_transcript_sha256": old_digest,
                        "revised_raw_transcript_sha256": transcript_digest(new),
                        "revised_canonical_transcript_sha256": canonical_digest,
                        "comparison": "right-continuous post-step values on the complete frozen union grid" if ecdf else "raw numeric argument bytes",
                        "compared_values_dtypes_shapes_masks_and_bytes_exact": True}
        if ecdf:
            result[name]["baseline_vertex_count"] = old_points
            result[name]["revised_vertex_count"] = new_points
            result[name]["exact_step_function_despite_redundant_vertex_removal"] = True
    return result


def detector_controls(capture):
    sample = capture["dart_ecdf_fresh"]["calls"][0]
    controls = {}
    for part in ("data", "mask"):
        changed = {"axis": sample["axis"], "method": sample["method"],
                   "step_convention": sample.get("step_convention"),
                   "arguments": {key: None if val is None else {p: ar.copy() for p, ar in val.items()}
                                 for key, val in sample["arguments"].items()}}
        item = changed["arguments"]["x"][part]
        if part == "data":
            item.flat[0] = np.nextafter(item.flat[0], np.inf)
        else:
            item.flat[0] = not item.flat[0]
        try:
            compare_call(sample, changed, "deliberate-perturbation")
        except ValueError:
            controls[part + "_perturbation_rejected"] = True
        else:
            raise ValueError("Comparison accepted a deliberately changed " + part)
    def fixture(x, y, convention="post"):
        return {"axis": 0, "method": "step", "step_convention": convention,
                "arguments": {"x": array(np.asarray(x, dtype=float)), "y": array(np.asarray(y, dtype=float))}}
    original = fixture([0, 1, 2, 3], [0, .5, .5, 1])
    compact = fixture([0, 1, 3], [0, .5, 1])
    compare_call(original, canonical_step(original, compact, "constant-segment-control"), "constant-segment-control")
    controls["constant_segment_compaction_accepted"] = True
    invalid = {"removed_nonzero_step_rejected": fixture([0, 3], [0, 1]),
               "misplaced_step_rejected": fixture([0, 2, 3], [0, .5, 1]),
               "step_convention_change_rejected": fixture([0, 1, 3], [0, .5, 1], "pre")}
    for label, value in invalid.items():
        try:
            compare_call(original, canonical_step(original, value, label), label)
        except ValueError:
            controls[label] = True
        else:
            raise ValueError("Comparison accepted an invalid step function")
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output receipt must be fresh")
    repo = args.repo.resolve()
    data_dir = (args.data_dir or repo / "analysis/prigm/figures").resolve()
    evidence_file = repo / "paper/prigm/paper_evidence_manifest.json"
    require(sha(evidence_file) == EVIDENCE, "Scientific evidence identity differs")
    evidence = read_json(evidence_file)
    files = {"paper/prigm/paper_evidence_manifest.json": evidence_file,
             "analysis/prigm/figures/render_figures.py": repo / "analysis/prigm/figures/render_figures.py",
             "analysis/prigm/figures/prepare_figure_data.py": repo / "analysis/prigm/figures/prepare_figure_data.py",
             "analysis/prigm/figures/plot_data.npz": data_dir / "plot_data.npz",
             "analysis/prigm/figures/plot_data_manifest.json": data_dir / "plot_data_manifest.json"}
    sources = evidence["input_index"]["source_sha256"]
    require(len(sources) == 13, "Unexpected frozen source coverage")
    for key, expected in sources.items():
        require(sha(repo / key) == expected, "Frozen scientific source differs: " + key)
        files[key] = repo / key
    for key, expected in evidence["core"]["public_source_sha256"].items():
        require(sha(repo / key) == expected, "Public raw data source differs")
        files[key] = repo / key
    before = {key: sha(path) for key, path in files.items()}
    own_sha = sha(__file__)
    metadata = read_json(data_dir / "plot_data_manifest.json")
    require(metadata["scientific_manifest_sha256"] == EVIDENCE, "Cache scientific identity differs")
    for key, expected in metadata["source_sha256"].items():
        require(key in files and before[key] == expected, "Cache producer source differs: " + key)
    require(metadata["training_seeds"] == [42, 1729, 31415]
            and metadata["prior_stream_seeds"] == [71001, 71002, 71003]
            and metadata["reference_stream_seeds"] == [74001, 74002, 74003], "Cache seed identity differs")
    for name, case in metadata["distribution_index"].items():
        for role, fits in case.items():
            if role in ("train", "test"):
                expected = [[f"{name}/{role}/0/0"]]
            elif role == "PPCA":
                expected = [[f"{name}/PPCA/0/{s}" for s in metadata["reference_stream_seeds"]]]
            else:
                expected = [[f"{name}/{role.replace(' ', '_')}/{seed}/{s}"
                             for s in metadata["prior_stream_seeds"]] for seed in metadata["training_seeds"]]
            require(fits == expected, "Cache fit/stream ordering differs")
    renderer = import_file("dart_review_renderer", files["analysis/prigm/figures/render_figures.py"])
    loaded_evidence, reduced = renderer.load_inputs(data_dir)
    require(loaded_evidence == evidence, "Loaded scientific evidence differs")
    frozen = renderer.frozen
    for key in sources:
        module = sys.modules.get(Path(key).stem)
        if module is not None:
            require(Path(module.__file__).resolve() == (repo / key).resolve(), "Imported frozen source from another tree")
    unused = Path("suppressed-figure-output")
    with Capture() as old:
        routes = baseline_figures(frozen, evidence, reduced, unused)
        baseline_context(frozen, evidence, repo, unused)
    with Capture() as new, plt.rc_context(renderer.STYLE):
        records = evidence["dart_extension"]["records"]
        renderer.comparison(unused, records)
        renderer.context(unused, evidence)
        renderer.diagnostics(unused, records, reduced)
        renderer.variance(unused, records)
    figures = compare_figures(old.figures, new.figures)
    controls = detector_controls(old.figures)
    require({key: sha(path) for key, path in files.items()} == before, "Input changed during comparison")
    require(sha(__file__) == own_sha, "Comparison script changed during execution")
    receipt = {"schema_version": 1, "passed": True,
               "purpose": "Exact numeric plotting-input comparison from one authenticated retained reduction cache",
               "scope": "Frozen extension/comparison loops and frozen context point reader versus revised renderer; ECDFs compared as exact post-step functions, other numeric arrays compared directly; style excluded",
               "limits": ["Does not certify extraction from private raw diagnostic arrays.",
                          "Does not compare with numeric coordinates recovered from historical PNG pixels.",
                          "Does not certify final typography, layout, axis transforms, or visual clarity."],
               "scientific_manifest_sha256": EVIDENCE,
               "comparison_script_sha256": own_sha, "inputs_sha256": before,
               "versions": {"python": platform.python_version(),
                            **{key: importlib.metadata.version(key) for key in ("numpy", "matplotlib", "scipy", "pandas")}},
               "cache_routes": routes, "figures": figures, "detector_controls": controls,
               "totals": {"figures": len(figures), **{key: sum(f[key] for f in figures.values())
                       for key in ("calls", "numeric_array_pairs", "numeric_elements", "masked_elements")}},
               "inputs_rehashed_after_comparison": True,
               "no_image_output_or_neural_execution": True}
    # Write only after all equality, coverage, deliberate-failure and stability checks pass.
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": True, "receipt_sha256": sha(args.output), **receipt["totals"]}))


if __name__ == "__main__":
    main()
