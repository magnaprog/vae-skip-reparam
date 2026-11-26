"""Rebuild the paired 21415/2018 comparisons from public stream scores."""
import argparse
import csv
from itertools import product
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "optimizer_coordinates"
SEEDS = (42, 1729, 31415)
VARIANCES = {"base": 0.0001, "high": 0.0025}
# Mode -> (model condition, effective residual Adam settings, clipping norm).
MODES = {
    "no_skip": ("base", False, "observation"),
    "matched_obs": ("canonical", False, "observation"),
    "matched_res": ("canonical", False, "residual"),
    "original_obs": ("canonical", True, "observation"),
    "original_res": ("canonical", True, "residual"),
    "native_residual": ("raw", False, "residual"),
}
# Names in the follow-up tables and the previously released tables, respectively.
GEOMETRIES = {"joint": "full_window", "centered": "without_window_mean",
              "increment": "time_differences"}
STREAMS = (71001, 71002, 71003)


def stream_columns(values):
    return {f"stream_{s}": float(v) for s, v in zip(STREAMS, values, strict=True)} | {
        "stream_mean": float(np.mean(values))}


def paired_differences(scores):
    at_obs = scores["matched_obs"] - scores["original_obs"]
    at_res = scores["matched_res"] - scores["original_res"]
    return {
        "head_matched_minus_original_at_obs": at_obs,
        "head_matched_minus_original_at_res": at_res,
        "clip_obs_minus_res_at_matched": scores["matched_obs"] - scores["matched_res"],
        "clip_obs_minus_res_at_original": scores["original_obs"] - scores["original_res"],
        "interaction_head_at_obs_minus_head_at_res": at_obs - at_res,
        **{mode + "_minus_no_skip": scores[mode] - scores["no_skip"]
           for mode in MODES if mode != "no_skip"},
        "native_residual_minus_original_res": scores["native_residual"] - scores["original_res"],
        "no_skip_minus_retained_base": scores["no_skip"] - scores["retained_base"],
        "native_residual_minus_retained_raw": scores["native_residual"] - scores["retained_raw"],
    }


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(fit_rows, historical):
    scores = {}
    weights = {}
    for row in fit_rows:
        variance, seed, mode = row["variance_label"], int(row["seed"]), row["mode"]
        fit = variance, seed, mode
        key = (*fit, row["geometry"])
        digest = row["weights_sha256"]
        if (key in scores or float(row["observation_variance"]) != VARIANCES[variance]
                or row["condition"] != MODES[mode][0] or not re.fullmatch("[a-f0-9]{64}", digest)
                or weights.setdefault(fit, digest) != digest):
            raise ValueError(f"Inconsistent fit identity: {key}")
        values = np.array([float(row[f"stream_{s}"]) for s in STREAMS])
        if not np.isfinite(values).all() or not np.isclose(
                values.mean(), float(row["stream_mean"]), atol=1e-13, rtol=1e-13):
            raise ValueError(f"Inconsistent stream scores: {key}")
        scores[key] = values
    if set(scores) != set(product(VARIANCES, SEEDS, MODES, GEOMETRIES)):
        raise ValueError("Expected all three geometries for each of the 36 final fits")

    previous = {}
    for row in read_csv(historical):
        if (row["record"], row["checkpoint"], row["variant"]) != ("dart_21415_2018", "final", "original"):
            continue
        key = (row["condition"], row["variance_group"], int(row["seed"]),
               row["geometry"], int(row["stream"]))
        value = float(row["energy_score"])
        if key in previous or not np.isfinite(value) or float(row["variance"]) != VARIANCES[key[1]]:
            raise ValueError(f"Inconsistent historical score: {key}")
        previous[key] = value

    rows = []
    for variance, seed, geometry in product(VARIANCES, SEEDS, GEOMETRIES):
        group = {mode: scores[variance, seed, mode, geometry] for mode in MODES}
        for mode, prefix in (("retained_base", "no_skip"), ("retained_raw", "skip")):
            condition = f"{prefix}_{variance}"
            group[mode] = np.array([previous[condition, variance, seed, GEOMETRIES[geometry], stream]
                                    for stream in STREAMS])
        for name, values in paired_differences(group).items():
            rows.append({"variance_label": variance, "geometry": geometry, "seed": seed,
                         "contrast": name, **stream_columns(values)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=RESULTS / "scores.csv")
    parser.add_argument("--output", type=Path, required=True, help="New comparisons CSV")
    args = parser.parse_args()
    rows = comparison_rows(read_csv(args.scores), ROOT / "results" / "energy_terms_streams.csv")
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
