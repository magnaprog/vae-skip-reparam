"""Reduce the full-window output-update differences for DART 21415 in 2018."""
import argparse
from datetime import date, timedelta
from itertools import product
from math import fsum, isfinite
from pathlib import Path

import numpy as np

from optimizer_comparisons import RESULTS, ROOT, SEEDS, STREAMS, VARIANCES, read_csv, write_csv


WINDOWS = "output_update_windows.csv"
INTERVALS = "output_update_intervals.csv"
PAIRS = tuple(product(VARIANCES, SEEDS, ("obs", "res")))
FORTNIGHT = 14 * 86400


def window_rows(starts, scores):
    """Subtract full-window scores within stream, then average the three streams."""
    rows = []
    for variance, seed, norm in PAIRS:
        matched = scores[variance, seed, "matched_" + norm]
        original = scores[variance, seed, "original_" + norm]
        if matched.shape != original.shape or matched.shape != (len(STREAMS), len(starts)):
            raise ValueError("Expected one score per stream and test window in both fits")
        differences = (matched - original).mean(axis=0)
        if not np.isfinite(differences).all():
            raise ValueError("Nonfinite window difference")
        for start, delta in zip(starts, differences, strict=True):
            rows.append(dict(variance_label=variance, seed=seed, clipping_coordinates=norm,
                             window_start_unix=int(start), score_difference=float(delta)))
    return rows


def interval_rows(rows, data):
    """Join windows by UTC start and renormalize the published test weights in each subset."""
    starts = data["starts_test"].tolist()
    weights = dict(zip(starts, data["eval_weights"].tolist(), strict=True))
    origin = int(data["reassignment_block_origin"])
    if len(weights) != len(starts) or not all(isfinite(w) and w > 0 for w in weights.values()):
        raise ValueError("Expected unique test window starts and finite positive weights")

    paired = {pair: {} for pair in PAIRS}
    for row in rows:
        pair = row["variance_label"], int(row["seed"]), row["clipping_coordinates"]
        start, value = int(row["window_start_unix"]), float(row["score_difference"])
        if pair not in paired or start not in weights or start in paired[pair] or not isfinite(value):
            raise ValueError(f"Invalid or duplicate window difference: {pair}, {start}")
        paired[pair][start] = value

    intervals = {}
    for start in starts:
        left = origin + (start - origin) // FORTNIGHT * FORTNIGHT
        intervals.setdefault(left, []).append(start)
    total = fsum(weights.values())
    result = []
    for (variance, seed, norm), values in paired.items():
        if values.keys() != weights.keys():
            raise ValueError(f"Missing test window differences: {variance}, {seed}, {norm}")
        full = fsum(weights[s] * values[s] for s in starts) / total
        for left, inside in sorted(intervals.items()):
            outside = [s for s in starts if not left <= s < left + FORTNIGHT]
            mass = fsum(weights[s] for s in inside)
            within = fsum(weights[s] * values[s] for s in inside) / mass
            omitted = fsum(weights[s] * values[s] for s in outside) / fsum(weights[s] for s in outside)
            day = date(1970, 1, 1) + timedelta(days=left // 86400)
            result.append(dict(variance_label=variance, seed=seed, clipping_coordinates=norm,
                               interval_start_utc=day.isoformat(), window_count=len(inside),
                               test_weight=mass / total, interval_difference=within,
                               excluding_interval_difference=omitted, full_test_difference=full))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=RESULTS / WINDOWS)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/dart_21415_2018.npz")
    parser.add_argument("--output", type=Path, required=True, help="New interval CSV")
    args = parser.parse_args()
    with np.load(args.prepared, allow_pickle=False) as data:
        write_csv(args.output, interval_rows(read_csv(args.windows), data))


if __name__ == "__main__":
    main()
