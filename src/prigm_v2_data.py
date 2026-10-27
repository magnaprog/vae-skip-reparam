"""DART record eligibility, window preparation and synthetic data generation.

Eligibility uses only data criteria, never model scores. The coverage
denominator is every expected quarter-hour UTC slot in the complete
candidate interval. Calendar splits at floor(0.6*N) and floor(0.8*N) days
from D0, with two-day embargo on each side of each boundary. Nonoverlapping
support packs counted by earliest-finish greedy selection from eligible
four-hour-grid windows. Six UTC start phases (00/04/08/12/16/20) must be
nonempty in each partition. Window weights w(s,i) = 1/[6*n(s, h(i))].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CADENCE = timedelta(minutes=15)
WINDOW_LENGTH = 192
WINDOW_SPAN = timedelta(minutes=15) * (WINDOW_LENGTH - 1)  # 47h45m
GRID = timedelta(hours=4)
PHASES = (0, 4, 8, 12, 16, 20)
EMBARGO = timedelta(days=2)
MIN_DAYS = 240
MIN_COVERAGE = 0.80
PACK_MINIMUMS = {"train": 60, "validation": 20, "test": 20}
SPLITS = ("train", "validation", "test")


@dataclass
class Qualification:
    station: str
    interval_start: datetime
    interval_end: datetime  # exclusive
    eligible: bool
    reasons: list = field(default_factory=list)
    coverage: float | None = None
    splits: dict = field(default_factory=dict)
    packs: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    mean: float | None = None
    scale: float | None = None


def _parse_annual(path):
    from io import BytesIO
    source = Path(path).read_bytes()
    frame = pd.read_csv(BytesIO(source), sep=r"\s+", header=None, skiprows=2,
                        dtype=np.float64, float_precision="round_trip",
                        skip_blank_lines=False)
    if frame.shape[1] != 8:
        raise ValueError("DART annual file must have eight columns")
    dates = pd.to_datetime(
        dict(zip(("year", "month", "day", "hour", "minute", "second"),
                 (frame.iloc[:, i] for i in range(6)))),
        utc=True, errors="coerce")
    seconds = (dates - pd.Timestamp("1970-01-01T00:00:00Z")).dt.total_seconds().to_numpy()
    heights = frame.iloc[:, 7].to_numpy(dtype=np.float64)
    types = frame.iloc[:, 6].to_numpy()
    observed = (types == 1) & np.isfinite(heights) & (heights <= 9000) & dates.notna().to_numpy()
    duplicate = np.zeros(len(frame), dtype=bool)
    idx = np.flatnonzero(observed)
    duplicate[idx] = dates.iloc[idx].duplicated(keep=False).to_numpy()
    retained = observed & ~duplicate
    return seconds, heights, retained, int(duplicate.sum())


def qualify_candidate(path, interval_start, interval_end, station=None,
                      window_length=WINDOW_LENGTH):
    """Evaluate one fixed candidate interval; returns a Qualification."""
    interval_start = pd.to_datetime(interval_start, utc=True).to_pydatetime()
    interval_end = pd.to_datetime(interval_end, utc=True).to_pydatetime()
    if interval_end <= interval_start:
        raise ValueError("empty interval")
    seconds, heights, retained, duplicates = _parse_annual(path)
    station = station or Path(path).stem.split("t")[0]

    start_ts = interval_start.replace(tzinfo=timezone.utc).timestamp()
    end_ts = interval_end.replace(tzinfo=timezone.utc).timestamp()
    days = (interval_end - interval_start).days
    n_days_exact = (interval_end - interval_start) == timedelta(days=days)

    q = Qualification(station=station, interval_start=interval_start,
                      interval_end=interval_end, eligible=True)
    if days < MIN_DAYS:
        q.eligible = False
        q.reasons.append(f"interval {days} days < {MIN_DAYS}")

    inside = retained & (seconds >= start_ts) & (seconds < end_ts)
    in_seconds = seconds[inside]
    in_heights = heights[inside]

    # Coverage: retained unambiguous T1 rows on expected quarter-hour UTC slots
    # / all expected slots in the complete interval.
    total_slots = int((end_ts - start_ts) / 900)
    on_grid = (in_seconds % 900 == 0)
    coverage = float(on_grid.sum()) / total_slots if total_slots else 0.0
    q.coverage = coverage
    if coverage < MIN_COVERAGE:
        q.eligible = False
        q.reasons.append(f"coverage {coverage:.4f} < {MIN_COVERAGE}")

    # Calendar splits with floor rounding and 2-day embargo per side.
    cut1 = interval_start + timedelta(days=int(np.floor(0.6 * days)))
    cut2 = interval_start + timedelta(days=int(np.floor(0.8 * days)))
    if not n_days_exact:
        q.reasons.append("note: interval is not whole days; floors applied to day count")
    c1_ts = cut1.replace(tzinfo=timezone.utc).timestamp()
    c2_ts = cut2.replace(tzinfo=timezone.utc).timestamp()

    partitions = {
        "train": inside & (seconds < c1_ts - EMBARGO.total_seconds()),
        "validation": inside & (seconds >= c1_ts + EMBARGO.total_seconds())
        & (seconds < c2_ts - EMBARGO.total_seconds()),
        "test": inside & (seconds >= c2_ts + EMBARGO.total_seconds()),
    }
    # Eligible four-hour-grid windows: all 192 observations present at exact
    # cadence, window start on the grid, full support within the partition.
    windows = {"train": [], "validation": [], "test": []}
    for split, mask in partitions.items():
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            q.eligible = False
            q.reasons.append(f"{split}: no observations")
            continue
        sec = seconds[idx]
        hgt = heights[idx]
        # segment by cadence breaks
        breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(sec) != 900)) + 1
        segments = np.split(np.arange(len(idx)), breaks) if len(idx) else []
        starts = []
        for segment in segments:
            if len(segment) < window_length:
                continue
            seg_seconds = sec[segment]
            # window starts at positions 0, 4h, 8h, ... within the segment
            # where the start timestamp is on the global four-hour grid.
            for offset in range(0, len(segment) - window_length + 1):
                t0 = seg_seconds[offset]
                if t0 % GRID.total_seconds() != 0:
                    continue
                t_end = seg_seconds[offset + window_length - 1]
                if t_end - t0 == (window_length - 1) * 900:
                    starts.append((t0, idx[segment[offset]]))
        # deduplicate by t0 (a start is unique per segment position)
        seen = {}
        for t0, start_ix in starts:
            seen[t0] = start_ix
        windows[split] = sorted(seen.items())
        q.windows[split] = len(windows[split])
        # Greedy disjoint packs by earliest finish.
        packs = 0
        occupied_until = -np.inf
        for t0, _ in sorted(seen.items()):
            finish = t0 + (window_length - 1) * 900
            if t0 > occupied_until:
                packs += 1
                occupied_until = finish
        q.packs[split] = packs
        phase_counts = {h: 0 for h in PHASES}
        for t0, _ in windows[split]:
            hour_utc = int((t0 % 86400) // 3600)
            if hour_utc in phase_counts:
                phase_counts[hour_utc] += 1
        q.phases[split] = phase_counts
        if packs < PACK_MINIMUMS[split]:
            q.eligible = False
            q.reasons.append(f"{split}: {packs} disjoint packs < {PACK_MINIMUMS[split]}")
        if any(count == 0 for count in phase_counts.values()):
            q.eligible = False
            q.reasons.append(f"{split}: empty UTC phase stratum")
        q.splits[split] = {"rows": int(mask.sum()), "cut": str(cut1 if split == 'train' else cut2)}
    # Per-window weights within each split: 1/[6 * n(s, h)].
    for split in ("train", "validation", "test"):
        phase_counts = q.phases.get(split, {})
        weights = {}
        for i, (t0, _) in enumerate(windows[split]):
            hour_utc = int((t0 % 86400) // 3600)
            n_phase = phase_counts.get(hour_utc, 0)
            weights[i] = 1.0 / (6 * n_phase) if n_phase else 0.0
        q.weights[split] = weights

    # Normalization: unique source rows covered by eligible training windows.
    covered_rows = set()
    for _, start_ix in windows["train"]:
        covered_rows.update(range(start_ix, start_ix + window_length))
    if covered_rows:
        rows = heights[np.array(sorted(covered_rows))]
        q.mean = float(rows.mean())
        q.scale = float(rows.std(ddof=0))
    q.duplicates = duplicates
    q.window_grid = windows
    return q


def same_phase_donors(starts, seed, source_starts, block_origin,
                      window_length=WINDOW_LENGTH):
    """Uniform same-phase donors from disjoint support and another 14-day block.

    Blocks are anchored at the declared cohort start in UTC seconds. If any
    target has no admissible donor, the entire diagnostic is unavailable.
    """
    starts = np.asarray(starts)
    source_starts = np.asarray(source_starts)
    if (starts.ndim != 1 or len(starts) == 0 or source_starts.shape != starts.shape
            or not np.issubdtype(starts.dtype, np.integer)
            or not np.issubdtype(source_starts.dtype, np.integer)
            or np.any(source_starts < 0)):
        raise ValueError("starts and source_starts must be matching nonempty integer vectors")
    if (isinstance(block_origin, bool) or not isinstance(block_origin, (int, np.integer))
            or isinstance(window_length, bool)
            or not isinstance(window_length, (int, np.integer)) or window_length < 1):
        raise ValueError("block origin and positive window length must be integers")
    hours = ((starts % 86400) // 3600).astype(np.int64)
    blocks = (starts - block_origin) // (14 * 86400)
    rng = np.random.default_rng(seed)
    donors = np.empty(len(starts), dtype=np.int64)
    index = np.arange(len(starts))
    for i in range(len(starts)):
        # Subtract smaller from larger so unsigned inputs cannot wrap at earlier starts.
        source_distance = (np.maximum(source_starts, source_starts[i])
                           - np.minimum(source_starts, source_starts[i]))
        time_distance = np.maximum(starts, starts[i]) - np.minimum(starts, starts[i])
        candidates = index[(hours == hours[i]) & (blocks != blocks[i])
                           & (source_distance >= window_length)
                           & (time_distance > (window_length - 1) * 900)]
        if len(candidates) == 0:
            return None
        donors[i] = candidates[rng.integers(0, len(candidates))]
    return donors


def _windows_from_grid(heights, grid, mean, scale, window_length=WINDOW_LENGTH):
    arrays = np.empty((len(grid), window_length), dtype=np.float64)
    starts = np.empty(len(grid), dtype=np.int64)
    for i, (t0, start_ix) in enumerate(grid):
        window = heights[start_ix:start_ix + window_length]
        if len(window) != window_length or not np.isfinite(window).all():
            raise ValueError(f"incomplete window at raw index {start_ix}")
        arrays[i] = (window - mean) / scale
        starts[i] = t0
    return arrays, starts


def prepare_dart_record(path, interval_start, interval_end, station=None,
                        reassignment_seed=66001, window_length=WINDOW_LENGTH):
    """Serialize one eligible DART candidate to the prepared-array contract.

    Window weights follow w = 1/(6 n_phase). Donors must have disjoint source
    support and belong to another cohort-anchored 14-day block at the same UTC
    phase. Unavailable reassignment leaves the evaluation population intact.
    """
    q = qualify_candidate(path, interval_start, interval_end, station=station,
                          window_length=window_length)
    if not q.eligible:
        raise ValueError("ineligible DART candidate: " + "; ".join(q.reasons))
    if q.mean is None or q.scale is None or not np.isfinite(q.scale) or q.scale <= 0:
        raise ValueError("qualification produced an invalid normalization")
    seconds, heights, retained, _ = _parse_annual(path)
    arrays, starts, weights, source_starts = {}, {}, {}, {}
    for split in SPLITS:
        grid = q.window_grid[split]
        arrays[split], starts[split] = _windows_from_grid(
            heights, grid, q.mean, q.scale, window_length)
        source_starts[split] = np.asarray([row for _, row in grid], dtype=np.int64)
        weights[split] = np.array(
            [q.weights[split][i] for i in range(len(grid))], dtype=np.float64)
        if not np.isfinite(weights[split]).all() or weights[split].sum() <= 0:
            raise ValueError(f"{split} weights are empty or nonfinite")
    block_origin = int(q.interval_start.timestamp())
    donors = same_phase_donors(starts["test"], reassignment_seed, source_starts["test"],
                              block_origin, window_length)
    year = q.interval_start.year
    return {
        "train": arrays["train"],
        "validation": arrays["validation"],
        "test": arrays["test"],
        "train_weights": weights["train"],
        "validation_weights": weights["validation"],
        "eval_weights": weights["test"],
        "starts_train": starts["train"],
        "starts_validation": starts["validation"],
        "starts_test": starts["test"],
        "source_starts_train": source_starts["train"],
        "source_starts_validation": source_starts["validation"],
        "source_starts_test": source_starts["test"],
        "reassignment_block_origin": np.int64(block_origin),
        "reassignment_available": np.bool_(donors is not None),
        "reassignment_donors": donors if donors is not None else np.empty(0, dtype=np.int64),
        "mean": np.float64(q.mean),
        "scale": np.float64(q.scale),
        "dataset_name": np.array(f"dart_{q.station}_{year}"),
    }


# ---- synthetic families (recovered; frozen contracts) ----


SYNTHETIC_SIZES = (8192, 2048, 2048)
GAUSSIAN_SEEDS = dict(zip(SPLITS, (1101, 1102, 1103)))
MIXTURE_SEEDS = dict(zip(SPLITS, (2101, 2102, 2103)))
NOISE_STD = 0.25
MIXTURE_FIRST_VARIANCE = 16.0
MIXTURE_SIGN_SCALE = np.sqrt(MIXTURE_FIRST_VARIANCE - 4.0)  # 12 + 4 = 16


def sinusoidal_basis(window: int = 192) -> np.ndarray:
    """Eight orthonormal columns cos/sin(k), k=1..4, on t=0..window-1."""
    if window < 9:
        raise ValueError("window must be at least 9 for eight Fourier columns")
    time = np.arange(window, dtype=np.float64)
    basis = np.column_stack([
        function(2.0 * np.pi * harmonic * time / window)
        for harmonic in range(1, 5) for function in (np.cos, np.sin)
    ]) * np.sqrt(2.0 / window)
    if not np.allclose(basis.T @ basis, np.eye(8), atol=1e-12):
        raise RuntimeError("Synthetic basis lost orthogonality")
    return basis


def _prepared(arrays, starts, metadata, oracle_covariance=None):
    return {
        "train": arrays["train"], "validation": arrays["validation"], "test": arrays["test"],
        "starts": starts, "metadata": metadata, "mean": 0.0, "scale": 1.0,
        "oracle_covariance": oracle_covariance,
    }


def prepare_gaussian(window: int = 192, sizes=SYNTHETIC_SIZES, seeds=GAUSSIAN_SEEDS):
    """x = U diag(4..1) z + 0.25 eps; independent splits, recorded seeds."""
    sizes = tuple(int(n) for n in sizes)
    basis = sinusoidal_basis(window)
    singular_values = np.linspace(4.0, 1.0, 8)
    loading = basis * singular_values
    arrays, starts = {}, {}
    for split, size in zip(SPLITS, sizes):
        rng = np.random.default_rng(seeds[split])
        latent = rng.standard_normal((size, 8))
        noise = rng.standard_normal((size, window)) * NOISE_STD
        arrays[split] = latent @ loading.T + noise
        starts[split] = np.arange(size, dtype=np.int64)
    metadata = {
        "dataset": "gaussian_v2", "window": window, "sizes": list(sizes),
        "family": "linear-Gaussian", "seeds": dict(seeds),
        "singular_values": singular_values.tolist(), "noise_std": NOISE_STD,
        "normalization": "declared zero mean, unit scale",
    }
    return _prepared(arrays, starts, metadata,
                     oracle_covariance=loading @ loading.T + NOISE_STD**2 * np.eye(window))


def prepare_mixture(window: int = 192, sizes=SYNTHETIC_SIZES, seeds=MIXTURE_SEEDS):
    """x = C*sqrt(12)*u1 + U diag(2, s2..s8) z + 0.25 eps.

    C is an independent equiprobable sign; the first signal direction has
    variance 12 + 4 = 16, matching the Gaussian family's first singular value
    squared. The density is a two-Gaussian mixture with the same covariance as
    the Gaussian family.
    """
    sizes = tuple(int(n) for n in sizes)
    basis = sinusoidal_basis(window)
    gaussian_singular_values = np.linspace(4.0, 1.0, 8)
    singular_values = np.array([2.0, *gaussian_singular_values[1:]])
    loading = basis * singular_values
    arrays, starts = {}, {}
    for split, size in zip(SPLITS, sizes):
        rng = np.random.default_rng(seeds[split])
        sign = rng.integers(0, 2, size=size) * 2.0 - 1.0
        latent = rng.standard_normal((size, 8))
        noise = rng.standard_normal((size, window)) * NOISE_STD
        arrays[split] = (sign * MIXTURE_SIGN_SCALE)[:, None] * basis[:, 0][None, :] \
            + latent @ loading.T + noise
        starts[split] = np.arange(size, dtype=np.int64)
    metadata = {
        "dataset": "mixture_v2", "window": window, "sizes": list(sizes),
        "family": "two-Gaussian mixture, covariance-matched", "seeds": dict(seeds),
        "singular_values": singular_values.tolist(),
        "sign_scale": float(MIXTURE_SIGN_SCALE), "noise_std": NOISE_STD,
        "normalization": "declared zero mean, unit scale",
    }
    first = basis[:, 0]
    oracle = loading @ loading.T + NOISE_STD**2 * np.eye(window)
    # 12 from the sign component plus 4 from the scaled latent keeps the first
    # direction's variance at 16, matching the Gaussian family.
    oracle = oracle + MIXTURE_SIGN_SCALE**2 * np.outer(first, first)
    return _prepared(arrays, starts, metadata, oracle_covariance=oracle)
