"""Prepare Gaussian samples and DART windows for training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


DART_BOUNDARIES = {
    "23461t2022": ("2022-04-28T00:00:00Z", "2022-11-25T00:00:00Z"),
    "44402t2016": ("2016-08-09T00:00:00Z", "2016-10-21T00:00:00Z"),
    "46407t2011": ("2011-08-12T00:00:00Z", "2011-10-23T00:00:00Z"),
    "52403t2020": ("2020-09-21T00:00:00Z", "2020-10-30T00:00:00Z"),
}
SPLITS = ("train", "validation", "test")
CADENCE_SECONDS = 900
EMBARGO_SECONDS = 2 * 24 * 60 * 60


@dataclass
class PreparedData:
    """Float64 window matrices and their shared scalar normalization.

    ``starts`` contains int64 UNIX seconds for DART, or realization indices
    local to each independently generated synthetic split. DART windows remain
    in source order; only training windows may overlap.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mean: float
    scale: float
    metadata: dict
    starts: dict[str, np.ndarray]
    oracle_covariance: np.ndarray | None = None


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _pack_split(indices, seconds, heights, window, stride, mean, scale):
    """Segment eligible source indices before packing, without sorting."""
    breaks = np.flatnonzero(
        (np.diff(indices) != 1)
        | (np.diff(seconds[indices]) != CADENCE_SECONDS)
    ) + 1
    segments = np.split(indices, breaks) if len(indices) else []
    packs, starts = [], []
    for segment in segments:
        if len(segment) < window:
            continue
        values = (heights[segment] - mean) / scale
        packs.append(np.lib.stride_tricks.sliding_window_view(values, window)[::stride])
        starts.append(seconds[segment[:len(segment) - window + 1:stride]])
    return (
        np.concatenate(packs) if packs else np.empty((0, window), dtype=np.float64),
        np.concatenate(starts).astype(np.int64) if starts else np.empty(0, dtype=np.int64),
        len(segments),
    )


def prepare_dart(path, window=192, stride=16):
    """Form windows from routine observations using UTC splits and training statistics.

    The filename stem identifies the station, year and partition dates.
    Heights must be finite and <= 9000, with T=1 and a valid timestamp. Every
    member of a duplicate observed-T=1 timestamp group is rejected globally.
    Invalid or intervening rows and steps other than 900 seconds end a segment.
    Heights retain their recorded values and order. Each eligible training row
    contributes once to normalization. Validation and test stride equals ``window``.
    """
    window = _positive_integer(window, "window")
    stride = _positive_integer(stride, "stride")
    path = Path(path)
    if path.stem not in DART_BOUNDARIES:
        raise ValueError(f"No DART partition dates for {path.stem!r}")
    boundary_strings = DART_BOUNDARIES[path.stem]
    b1, b2 = (pd.Timestamp(value).timestamp() for value in boundary_strings)

    # Record the identity of the bytes used by the parser.
    source = path.read_bytes()
    frame = pd.read_csv(
        BytesIO(source), sep=r"\s+", header=None, skiprows=2,
        dtype=np.float64, float_precision="round_trip", skip_blank_lines=False,
    )
    if frame.shape[1] != 8:
        raise ValueError("DART input must have eight columns after two header lines")
    dates = pd.to_datetime(
        dict(zip(("year", "month", "day", "hour", "minute", "second"),
                 (frame.iloc[:, i] for i in range(6)))),
        utc=True, errors="coerce",
    )
    # Elapsed seconds work with either Pandas datetime resolution, us or ns.
    # Invalid timestamps remain ineligible.
    seconds = (dates - pd.Timestamp("1970-01-01T00:00:00Z")).dt.total_seconds().to_numpy()
    heights = frame.iloc[:, 7].to_numpy(dtype=np.float64)
    sample_types = frame.iloc[:, 6].to_numpy()
    observed = (sample_types == 1) & np.isfinite(heights) & (heights <= 9000) & dates.notna().to_numpy()
    duplicate = np.zeros(len(frame), dtype=bool)
    observed_indices = np.flatnonzero(observed)
    duplicate[observed_indices] = dates.iloc[observed_indices].duplicated(keep=False).to_numpy()
    retained = observed & ~duplicate
    masks = {
        "train": retained & (seconds < b1 - EMBARGO_SECONDS),
        "validation": retained & (seconds >= b1 + EMBARGO_SECONDS) & (seconds < b2 - EMBARGO_SECONDS),
        "test": retained & (seconds >= b2 + EMBARGO_SECONDS),
    }
    train_rows = heights[masks["train"]]
    if not len(train_rows):
        raise ValueError("No valid training source rows for normalization")
    mean = float(train_rows.mean())
    scale = float(train_rows.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training source rows must have a finite mean and positive finite scale")

    arrays, starts, counts = {}, {}, {}
    strides = {"train": stride, "validation": window, "test": window}
    for split in SPLITS:
        indices = np.flatnonzero(masks[split])
        arrays[split], starts[split], segment_count = _pack_split(
            indices, seconds, heights, window, strides[split], mean, scale,
        )
        counts[split] = {
            "rows": int(len(indices)), "segments": segment_count,
            "windows": int(len(arrays[split])),
        }

    valid_dates = dates[dates.notna()]
    metadata = {
        "dataset": path.stem,
        "station_id": path.stem.split("t")[0],
        "source_sha256": sha256(source).hexdigest(),
        "boundaries_utc": dict(zip(("b1", "b2"), boundary_strings)),
        "embargo_seconds_each_side": EMBARGO_SECONDS,
        "split_rules": {
            "train": "t < b1 - embargo",
            "validation": "b1 + embargo <= t < b2 - embargo",
            "test": "t >= b2 + embargo",
        },
        "cadence_seconds": CADENCE_SECONDS,
        "window": window,
        "window_span_seconds": (window - 1) * CADENCE_SECONDS,
        "strides": strides,
        "starts_unit": "unix_seconds",
        "mean": mean,
        "std": scale,
        "normalization": {"fit": "valid training source rows, each once", "ddof": 0},
        "height_units": "meters of recorded pressure-equivalent water-column HEIGHT",
        "placeholder_rule": "HEIGHT > 9000",
        "before_filter": {
            "rows": int(len(frame)),
            "type_counts": {
                format(float(value), "g"): int(count)
                for value, count in zip(*np.unique(sample_types, return_counts=True))
            },
            "placeholder_rows": int((heights > 9000).sum()),
            "nonfinite_height_rows": int((~np.isfinite(heights)).sum()),
            "invalid_timestamp_rows": int(dates.isna().sum()),
            "first_timestamp_utc": valid_dates.iloc[0].isoformat().replace("+00:00", "Z") if len(valid_dates) else None,
            "last_timestamp_utc": valid_dates.iloc[-1].isoformat().replace("+00:00", "Z") if len(valid_dates) else None,
        },
        "observed_t1_rows": int(observed.sum()),
        "duplicate_observed_t1_rows": int(duplicate.sum()),
        "retained_t1_rows": int(retained.sum()),
        "embargo_rows": int(retained.sum()) - sum(item["rows"] for item in counts.values()),
        "counts": counts,
    }
    return PreparedData(**arrays, mean=mean, scale=scale, metadata=metadata, starts=starts)


def prepare_synthetic(window=192, sizes=(8192, 2048, 2048)):
    """Generate samples from a Gaussian factor model with a known covariance.

    Columns alternate cos(k), sin(k) for k=1,...,4 on t=0,...,window-1.
    Each split draws all latent coordinates before drawing observation noise.
    Samples retain the dimensionless simulator coordinates without standardization.
    """
    window = _positive_integer(window, "window")
    if window < 9:
        raise ValueError("window must be at least 9 for eight orthonormal Fourier columns")
    if len(sizes) != 3 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0
        for n in sizes
    ):
        raise ValueError("sizes must contain three nonnegative integers")
    sizes = tuple(int(n) for n in sizes)
    time = np.arange(window, dtype=np.float64)
    basis = np.column_stack([
        function(2.0 * np.pi * harmonic * time / window)
        for harmonic in range(1, 5) for function in (np.cos, np.sin)
    ]) * np.sqrt(2.0 / window)
    singular_values = np.linspace(4.0, 1.0, 8)
    loading = basis * singular_values
    noise_std = 0.25
    seeds = dict(zip(SPLITS, (1101, 1102, 1103)))
    arrays, starts = {}, {}
    for split, size in zip(SPLITS, sizes):
        rng = np.random.default_rng(seeds[split])
        latent = rng.standard_normal((size, 8))
        noise = rng.standard_normal((size, window)) * noise_std
        arrays[split] = latent @ loading.T + noise
        starts[split] = np.arange(size, dtype=np.int64)
    metadata = {
        "dataset": "gaussian",
        "window": window,
        "mean": 0.0,
        "std": 1.0,
        "normalization": "none; known zero mean and identity scale",
        "starts_unit": "split_local_realization_index",
        "counts": {split: {"windows": size} for split, size in zip(SPLITS, sizes)},
        "generator": {
            "equation": "x = A z + epsilon",
            "latent_dimension": 8,
            "latent_distribution": "eight independent standard normal coordinates",
            "time_grid": "t = 0, ..., window - 1",
            "basis_formula": "sqrt(2/window) * cos or sin(2*pi*k*t/window)",
            "harmonics": [1, 2, 3, 4],
            "column_order": [f"{function}({k})" for k in range(1, 5) for function in ("cos", "sin")],
            "basis_normalization": float(np.sqrt(2.0 / window)),
            "singular_values": singular_values.tolist(),
            "loading_matrix": "A = basis * singular_values (columnwise)",
            "noise_distribution": "independent isotropic zero-mean Gaussian",
            "noise_std": noise_std,
            "oracle_covariance": "A @ A.T + noise_std**2 * I",
            "rng": "numpy.random.Generator(PCG64)",
            "seeds": seeds,
            "draw_order": "standard_normal((size, 8)) then standard_normal((size, window))",
        },
    }
    return PreparedData(
        **arrays, mean=0.0, scale=1.0, metadata=metadata, starts=starts,
        oracle_covariance=loading @ loading.T + noise_std**2 * np.eye(window),
    )


def prepare_dataset(name, datasets_dir):
    """Prepare ``gaussian`` or one of the four DART records."""
    if name == "gaussian":
        return prepare_synthetic()
    if name not in DART_BOUNDARIES:
        raise ValueError(f"Unknown dataset {name!r}")
    return prepare_dart(Path(datasets_dir) / f"{name}.txt")
