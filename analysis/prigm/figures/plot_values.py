"""Numerical reductions for the retained DART figure inputs.

Streams are averaged within each fit. The fit axis remains available for
medians and ranges across training seeds. No models are fitted or sampled.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


RECORDS = ("23461t2022", "44402t2016", "46407t2011", "52403t2020",
           "dart_44402_2018", "dart_21415_2018", "dart_21418_2018")
ARMS = ("A", "B", "C", "D")
SEEDS = (42, 1729, 31415)


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


def ecdf_summary(fits, feature):
    """Evaluate each fit on the exact union of its streams' feature values."""
    grid = np.unique(np.concatenate([stream["features"][:, feature] for fit in fits for stream in fit]))
    return grid, ecdf_on_grid(fits, feature, grid)


def ecdf_on_grid(fits, feature, grid):
    """Keep fits separate; average only their Monte Carlo streams."""
    return np.array([np.mean([weighted_ecdf(s["features"][:, feature], s["weights"], grid)
                             for s in fit], axis=0) for fit in fits])


def power_arrays(fits):
    """Average stream powers in linear units, retaining one row per fit."""
    return np.array([np.mean([stream["power"] for stream in fit], axis=0) for fit in fits])


def refit_points(record, arm):
    """Variance ratios and replacement-minus-original scores, seed then stream."""
    ratios, deltas = [], []
    for seed in SEEDS:
        original = record["models"][f"{arm}/{seed}/original"]
        refit = record["models"][f"{arm}/{seed}/variance_refit"]
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
