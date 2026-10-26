#!/usr/bin/env python3
"""Make compact, path-free scientific results from energy_two_term_analysis.py.

Usage: python summarize_energy_analysis.py --input-root RESULTS --output results.json
Only authenticated CSV/JSON reductions are read; the large raw NPZ files may
remain in the original archive. Output contains no hostnames or absolute paths.
The private input receipts and launch commands are deliberately not exported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.input_root / "output_manifest.json").read_text())

    def read(name, csv_file=False):
        path = args.input_root / name
        if sha(path) != index[name]["sha256"]:
            raise ValueError("Reduction hash differs: " + name)
        with path.open() as handle:
            return list(csv.DictReader(handle)) if csv_file else json.load(handle)

    verification = read("verification.json")
    protocol = read("protocol.json")
    receipts = read("input_receipts.json")
    if verification["success"] is not True or verification["unique_fits"] != 84:
        raise ValueError("The complete analysis has not passed")
    reset = read("reset_contrasts_fits.csv", True)
    skip = read("skip_contrasts_fits.csv", True)
    comparison = read("PPCA_matched_comparisons.csv", True)
    covariance = read("covariance_structure.csv", True)
    if (len(reset), len(skip), len(comparison), len(covariance)) != (252, 126, 252, 21):
        raise ValueError("Incomplete scientific reductions")

    def describe(values):
        values = [float(value) for value in values]
        return {"count": len(values), "min": min(values), "max": max(values),
                "median": statistics.median(values),
                "positive": sum(value > 0 for value in values),
                "negative": sum(value < 0 for value in values),
                "zero": sum(value == 0 for value in values)}

    def summarize(rows, group_key, groups):
        output = []
        for group in groups:
            for geometry in ("joint", "centered", "increment"):
                selected = [row for row in rows if row[group_key] == group and row["geometry"] == geometry]
                output.append({group_key: group, "geometry": geometry,
                               **{term: describe(row[term] for row in selected)
                                  for term in ("delta_A", "delta_B", "delta_ES")}})
        return output

    ppca = []
    for geometry in ("joint", "centered", "increment"):
        selected = [row for row in comparison if row["geometry"] == geometry]
        low = min(selected, key=lambda row: float(row["neural_minus_PPCA"]))
        ppca.append({"geometry": geometry,
                     "neural_minus_PPCA": describe(row["neural_minus_PPCA"] for row in selected),
                     "minimum_at": {k: low[k] for k in ("record", "arm", "seed", "variance_group")}})
    covariance_records = []
    for record in protocol["records"]:
        selected = [row for row in covariance if row["record"] == record]
        if len(selected) != 3:
            raise ValueError("Covariance record coverage differs")
        covariance_records.append({"record": record,
                                   "leading16_covariance_fraction": {row["geometry"]: float(row["leading_fraction"]) for row in selected},
                                   "fitted_PPCA_variance": float(selected[0]["fitted_variance"])})
    logical_hashes = {k: v["actual_sha256"] for k, v in receipts.items()}
    encoded = json.dumps(logical_hashes, sort_keys=True, separators=(",", ":")).encode()
    score_checks = ("neural_per_window_score", "neural_weighted_stream_score",
                    "reference_per_window_score", "reference_weighted_stream_score",
                    "paired_reset_accounting", "paired_skip_accounting")
    output = {
        "schema_version": 1,
        "purpose": "Retrospective energy accounting and covariance description of seven DART records",
        "provenance": {"frozen_evidence_sha256": protocol["manifest_sha256"],
                       "analysis_script_sha256": protocol["script_sha256"],
                       "summary_script_sha256": sha(Path(__file__)),
                       "consumed_logical_input_hashes_sha256": hashlib.sha256(encoded).hexdigest(),
                       "input_files": verification["input_files"],
                       "inputs_rehashed_after_completion": verification["inputs_rehashed_after_completion"]},
        "coverage": {"records": protocol["records"], "arms": protocol["arms"],
                     "training_seeds": protocol["training_seeds"], "prior_streams": protocol["prior_streams"],
                     "samples_per_stream": 1024, "unique_fits": 84,
                     "original_and_reset_stream_geometry_rows": 1512,
                     "reset_fit_geometry_contrasts": 252, "skip_fit_geometry_contrasts": 126,
                     "matched_PPCA_fit_geometry_comparisons": 252},
        "definitions": {"A": "Weighted mean observation-to-sample distance",
                        "B": "Half mean unordered distinct sample-pair distance: scipy.spatial.distance.pdist(draws).mean()/2",
                        "ES": "A-B; lower is better",
                        "reset": "Reset minus original within the same fixed fit and stream",
                        "skip": "C-A or B-D at original matched observation variance and paired training seed/stream",
                        "reduction": "Average the three streams within each fit before these descriptive fit counts and ranges",
                        "PPCA": "Each neural fit's stream mean minus the matched PPCA reference's stream mean; no cross-model stream pairing",
                        "covariance": "Leading16 weighted-training-covariance trace divided by full trace after each fixed transform; about the weighted mean window",
                        "scope": "Standardized coordinates; finite retained windows and samples; no new fitting, neural execution, or causal/population inference"},
        "verification": {"atol": protocol["atol"], "rtol": protocol["rtol"],
                         "checks": {key: verification["checks"][key] for key in score_checks}},
        "reset_term_summaries": summarize(reset, "variance_group", ("low", "high")),
        "skip_term_summaries": summarize(skip, "pair", ("C-A", "B-D")),
        "matched_PPCA_comparison": ppca,
        "covariance_structure": covariance_records,
        "limits": ["Neither energy term alone is a proper score or calibration diagnostic.",
                   "Medians of the two terms need not subtract to the median score.",
                   "Training covariance concentration does not establish Gaussianity or explain the neural score gap.",
                   "The accepted evidence manifest is preserved; this is a separately bound additive analysis."]}
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if str(args.input_root.absolute()) in text or str(args.output.absolute()) in text:
        raise ValueError("Private location leaked into public results")
    with args.output.open("x") as handle:
        handle.write(text)
    print(json.dumps({"public_result_bytes": len(text.encode()), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
