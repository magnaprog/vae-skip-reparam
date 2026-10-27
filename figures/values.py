"""Numerical reductions for the retained DART figure inputs.

Streams are averaged within each fit. The fit axis remains available for
medians and ranges across training seeds. No models are fitted or sampled.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


RECORDS = ("23461t2022", "44402t2016", "46407t2011", "52403t2020",
           "dart_44402_2018", "dart_21415_2018", "dart_21418_2018")
CONDITIONS = ('no_skip_base', 'skip_base', 'no_skip_high', 'skip_high')
SEEDS = (42, 1729, 31415)
SCORES = ("joint", "centered", "increment")


def validate_comparisons(data, scientific_sha256):
    """Validate the seven-record comparisons without changing their values."""
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def keys(value, expected, label):
        require(isinstance(value, dict) and set(value) == set(expected), label + " keys differ")

    def number(value):
        return type(value) in (int, float) and np.isfinite(value)

    def triple(value, label):
        require(isinstance(value, list) and len(value) == 3
                and all(number(x) for x in value), label + " requires three finite values")

    def stats(value):
        keys(value, ("mean", "min", "max", "values"), "Stream statistics")
        triple(value["values"], "Stream statistics")
        require(all(number(value[k]) for k in ("mean", "min", "max")), "Invalid stream statistic")
        require(value["min"] == min(value["values"]) and value["max"] == max(value["values"])
                and value["mean"] == float(np.mean(value["values"])), "Stream statistics differ")

    keys(data, ("schema_version", "scientific_manifest_sha256", "training_seeds", "records"), "Comparison")
    require(type(data["schema_version"]) is int and data["schema_version"] == 2,
            "Unsupported comparison schema")
    require(data["scientific_manifest_sha256"] == scientific_sha256, "Comparisons belong to different evidence")
    require(data["training_seeds"] == list(SEEDS)
            and all(type(s) is int for s in data["training_seeds"]), "Training seed order differs")
    keys(data["records"], RECORDS, "Record")
    baseline_keys = ("low/ppca_fitted", "low/ppca_matched", "high/ppca_matched",
                     "low/diagonal_matched", "high/diagonal_matched")
    for name, record in data["records"].items():
        keys(record, ("station", "year", "interval_origin_unix", "ppca_comparison",
                      "neural_scores", "baseline_scores", "intervals"), "Record fields")
        station = name[5:10] if name.startswith("dart_") else name[:5]
        require(record["station"] == station and type(record["year"]) is int
                and record["year"] == int(name[-4:]), "Record station/year differs")
        origin = record["interval_origin_unix"]
        require(type(origin) is int and origin == int(datetime(record["year"], 1, 1,
                tzinfo=timezone.utc).timestamp()), "Interval origin differs")
        keys(record["baseline_scores"], baseline_keys, "Baseline")
        for baseline in record["baseline_scores"].values():
            keys(baseline, ("noise_variance", "scores"), "Baseline fields")
            require(number(baseline["noise_variance"]) and baseline["noise_variance"] > 0,
                    "Invalid baseline variance")
            keys(baseline["scores"], SCORES, "Baseline score")
            for value in baseline["scores"].values():
                stats(value)
        keys(record["neural_scores"], CONDITIONS, "Neural condition")
        keys(record["ppca_comparison"], CONDITIONS, "Comparison condition")
        for condition in CONDITIONS:
            keys(record["neural_scores"][condition], SCORES, "Neural score")
            keys(record["ppca_comparison"][condition], SCORES, "Comparison score")
            variance = "low" if condition in ("no_skip_base", "skip_base") else "high"
            for score in SCORES:
                fits = record["neural_scores"][condition][score]
                require(isinstance(fits, list) and len(fits) == 3, "Expected three neural fits")
                for fit in fits:
                    stats(fit)
                actual = record["ppca_comparison"][condition][score]
                triple(actual, "PPCA comparison")
                reference = record["baseline_scores"][variance + "/ppca_matched"]["scores"][score]["mean"]
                expected = np.array([fit["mean"] - reference for fit in fits])
                require(np.asarray(actual, dtype=np.float64).tobytes() == expected.tobytes(),
                        "PPCA comparison differs from matched absolute scores")
        keys(record["intervals"], ("14", "28"), "Interval duration")
        counts = []
        for days in (14, 28):
            intervals = record["intervals"][str(days)]
            require(isinstance(intervals, list) and len(intervals) > 0, "Missing intervals")
            previous_end = origin
            for item in intervals:
                keys(item, ("start_unix", "end_unix", "windows", "mass", "base", "high"), "Interval")
                start, end = item["start_unix"], item["end_unix"]
                require(type(start) is int and type(end) is int and start >= previous_end
                        and (start - origin) % (days * 86400) == 0
                        and end - start == days * 86400, "Invalid interval dates/order")
                previous_end = end
                require(type(item["windows"]) is int and item["windows"] > 0, "Invalid window count")
                require(number(item["mass"]) and 0 < item["mass"] <= 1, "Invalid interval mass")
                for variance in ("base", "high"):
                    triple(item[variance], "Interval contrast")
            require(np.isclose(sum(i["mass"] for i in intervals), 1, rtol=0, atol=1e-12),
                    "Interval masses do not sum to one")
            counts.append(sum(i["windows"] for i in intervals))
            for variance, left, right in (("base", "skip_base", "no_skip_base"), ("high", "skip_high", "no_skip_high")):
                whole = np.array([a["mean"] - b["mean"] for a, b in zip(
                    record["neural_scores"][left]["joint"], record["neural_scores"][right]["joint"], strict=True)])
                weighted = sum(i["mass"] * np.asarray(i[variance]) for i in intervals)
                require(np.allclose(weighted, whole, rtol=1e-12, atol=1e-12),
                        "Interval contrasts do not reconstruct whole-record scores")
        require(counts[0] == counts[1], "Interval durations cover different window counts")
    return data


def ppca_comparison_points(record, condition):
    """One row per training seed, with the three already reduced score differences."""
    return np.array([record["ppca_comparison"][condition][score] for score in SCORES]).T


def interval_points(record, days):
    """Retain actual interval boundaries and every seed; never interpolate dates."""
    intervals = record["intervals"][str(days)]
    return {key: np.asarray([item[key] for item in intervals])
            for key in ("start_unix", "end_unix", "windows", "mass", "base", "high")}


def record_order(records):
    """Original four records, then the three additional 2018 records."""
    return [name for name in RECORDS if name in records]


def weighted_ecdf(values, weights, grid):
    """Right-continuous empirical CDF, including all tied mass at each point."""
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if not (values.ndim == 1 and weights.shape == values.shape and len(values) > 0
            and np.isfinite(values).all() and np.isfinite(weights).all()
            and np.all(weights >= 0) and weights.sum() > 0):
        raise ValueError("Invalid ECDF measure")
    order = np.argsort(values, kind="stable")
    mass = np.r_[0., np.cumsum(weights[order] / weights.sum())]
    return mass[np.searchsorted(values[order], grid, side="right")]


def ecdf_grid(fits, feature):
    """Return the exact union of the streams' feature values."""
    return np.unique(np.concatenate([stream["features"][:, feature] for fit in fits for stream in fit]))


def ecdf_summary(fits, feature):
    """Evaluate each fit on the exact union of its streams' feature values."""
    grid = ecdf_grid(fits, feature)
    return grid, ecdf_on_grid(fits, feature, grid)


def ecdf_on_grid(fits, feature, grid):
    """Keep fits separate; average only their Monte Carlo streams."""
    return np.array([np.mean([weighted_ecdf(s["features"][:, feature], s["weights"], grid)
                             for s in fit], axis=0) for fit in fits])


def power_arrays(fits):
    """Average stream powers in linear units, retaining one row per fit."""
    return np.array([np.mean([stream["power"] for stream in fit], axis=0) for fit in fits])


def refit_points(record, condition):
    """Variance ratios and replacement-minus-original scores, seed then stream."""
    ratios, deltas = [], []
    for seed in SEEDS:
        original = record["models"][f"{condition}/{seed}/original"]
        refit = record["models"][f"{condition}/{seed}/variance_refit"]
        ratios.append(refit["variance_fit"]["pooled"] / original["observation_variance"])
        deltas.append([(np.asarray(refit["scores"][score]["weighted"]["values"])
                        - np.asarray(original["scores"][score]["weighted"]["values"]))
                       for score in ("joint", "centered", "increment")])
    return np.asarray(ratios), np.asarray(deltas)


def read_observed_context(path, expected_sha256):
    """Read all valid observed T=1 rows in source order for the context plot.

    This plot includes rows outside eligible windows and retains duplicate
    timestamps. Hash the exact bytes parsed before interpreting the annual data.
    """
    source = Path(path).read_bytes()
    if sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Public source hash differs")
    frame = pd.read_csv(BytesIO(source), sep=r"\s+", header=None, skiprows=2,
                        dtype=np.float64, float_precision="round_trip", skip_blank_lines=False)
    if frame.shape[1] != 8:
        raise ValueError("DART context requires eight columns")
    dates = pd.to_datetime(dict(zip(("year", "month", "day", "hour", "minute", "second"),
                                    (frame.iloc[:, i] for i in range(6)))), utc=True, errors="coerce")
    heights = frame.iloc[:, 7].to_numpy(dtype=np.float64)
    keep = (frame.iloc[:, 6].to_numpy() == 1) & np.isfinite(heights) & (heights <= 9000) & dates.notna().to_numpy()
    if not keep.any():
        raise ValueError("No observed context data")
    dates = dates[keep]
    year = int(Path(path).stem.split("t")[1])
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    days = 1 + (dates - start).dt.total_seconds().to_numpy() / 86400
    return {"days": days, "heights": heights[keep], "year": year}
