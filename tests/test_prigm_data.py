"""Local protocol checks for the separate PriGM data path."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prigm_data import PreparedData, prepare_dart, prepare_dataset, prepare_synthetic


SPLITS = ("train", "validation", "test")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
BASE = datetime(2022, 1, 1, tzinfo=timezone.utc)


def unix_seconds(date):
    return int((date - EPOCH).total_seconds())


def routine_rows(start, heights):
    return [(start + timedelta(seconds=900 * i), 1, value) for i, value in enumerate(heights)]


def write_station(path, rows):
    lines = ["#YY MM DD hh mm ss T HEIGHT", "#yr mo dy hr mn s - m"]
    for date, sample_type, height in rows:
        timestamp = date.strftime("%Y %m %d %H %M %S") if date is not None else "2022 13 01 00 00 00"
        lines.append(f"{timestamp} {sample_type} {height}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class LiveDartTest(unittest.TestCase):
    def test_live_counts_provenance_cadence_and_disjoint_support(self):
        expected = {
            "23461t2022": ((544, 13, 17), 18204, 5028, "2022-04-28", "2022-11-25"),
            "44402t2016": ((1153, 34, 35), 34980, 156, "2016-08-09", "2016-10-21"),
            "46407t2011": ((983, 26, 27), 33023, 2017, "2011-08-12", "2011-10-23"),
            "52403t2020": ((339, 17, 16), 18769, 4655, "2020-09-21", "2020-10-30"),
        }
        for station, (window_counts, observed_count, placeholders, first, second) in expected.items():
            with self.subTest(station=station):
                path = ROOT / "datasets" / f"{station}.txt"
                data = prepare_dataset(station, ROOT / "datasets")
                raw = np.loadtxt(path, skiprows=2, dtype=np.float64)
                # Independent stdlib timestamp conversion, not pandas integer units.
                times = np.array([
                    unix_seconds(datetime(*row[:6].astype(int), tzinfo=timezone.utc))
                    for row in raw
                ], dtype=np.int64)
                valid = (raw[:, 6] == 1) & np.isfinite(raw[:, 7]) & (raw[:, 7] <= 9000)
                valid_indices = np.flatnonzero(valid)
                self.assertEqual(len(np.unique(times[valid])), len(valid_indices))
                source_at = dict(zip(times[valid], valid_indices))
                b1 = unix_seconds(datetime.fromisoformat(first).replace(tzinfo=timezone.utc))
                b2 = unix_seconds(datetime.fromisoformat(second).replace(tzinfo=timezone.utc))
                embargo = 2 * 86400
                masks = {
                    "train": valid & (times < b1 - embargo),
                    "validation": valid & (times >= b1 + embargo) & (times < b2 - embargo),
                    "test": valid & (times >= b2 + embargo),
                }
                self.assertIsInstance(data, PreparedData)
                self.assertIsNone(data.oracle_covariance)
                self.assertIs(type(data.mean), float)
                self.assertIs(type(data.scale), float)
                self.assertEqual(data.mean, float(raw[masks["train"], 7].mean()))
                self.assertEqual(data.scale, float(raw[masks["train"], 7].std(ddof=0)))
                meta = data.metadata
                self.assertEqual(meta["station_id"], station.split("t")[0])
                self.assertEqual(meta["source_sha256"], sha256(path.read_bytes()).hexdigest())
                self.assertEqual(meta["boundaries_utc"], {"b1": first + "T00:00:00Z", "b2": second + "T00:00:00Z"})
                self.assertEqual(meta["observed_t1_rows"], observed_count)
                self.assertEqual(meta["duplicate_observed_t1_rows"], 0)
                self.assertEqual(meta["before_filter"]["placeholder_rows"], placeholders)
                self.assertEqual(meta["before_filter"]["rows"], len(raw))
                self.assertEqual(meta["before_filter"]["type_counts"], {
                    str(int(t)): int(n) for t, n in zip(*np.unique(raw[:, 6], return_counts=True))
                })
                self.assertEqual(meta["mean"], data.mean)
                self.assertEqual(meta["std"], data.scale)
                encoded = json.dumps(meta, allow_nan=False)
                self.assertNotIn(str(ROOT), encoded)
                supports = {}
                for split, count in zip(SPLITS, window_counts):
                    values, starts = getattr(data, split), data.starts[split]
                    self.assertEqual(values.shape, (count, 192))
                    self.assertEqual(values.dtype, np.float64)
                    self.assertEqual(starts.dtype, np.int64)
                    self.assertEqual(starts.shape, (count,))
                    self.assertTrue(np.isfinite(values).all())
                    indices = np.flatnonzero(masks[split])
                    breaks = (np.diff(indices) != 1) | (np.diff(times[indices]) != 900)
                    self.assertEqual(meta["counts"][split], {
                        "rows": len(indices), "segments": 1 + int(breaks.sum()), "windows": count,
                    })
                    source_indices = np.array([source_at[start] for start in starts])[:, None] + np.arange(192)
                    support = starts[:, None] + 900 * np.arange(192)
                    np.testing.assert_array_equal(times[source_indices], support)
                    self.assertTrue(masks[split][source_indices].all())
                    np.testing.assert_array_equal(values, (raw[source_indices, 7] - data.mean) / data.scale)
                    supports[split] = np.unique(support)
                    if split != "train":
                        self.assertEqual(supports[split].size, count * 192)
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                    self.assertEqual(np.intersect1d(supports[left], supports[right]).size, 0)


class DartFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "23461t2022.txt"

    def test_train_only_float64_population_normalization_once_per_source_row(self):
        train_heights = np.array([6000.0001, 6000.0002, 6000.0003, 6000.0004, 6000.0020, 6000.0006, 6000.0040])
        isolated_height = 6000.0200
        train_rows = routine_rows(BASE, train_heights) + routine_rows(BASE + timedelta(days=31), [isolated_height])
        validation_rows = routine_rows(datetime(2022, 5, 1, tzinfo=timezone.utc), 6000 + np.arange(7) / 100)
        test_date = datetime(2022, 12, 1, tzinfo=timezone.utc)
        test_rows = routine_rows(test_date, 6000 + np.arange(7) / 10)
        write_station(self.path, train_rows + validation_rows + test_rows)
        before = prepare_dart(self.path, window=3, stride=2)
        all_train = np.append(train_heights, isolated_height)
        self.assertEqual(before.mean, all_train.mean())
        self.assertEqual(before.scale, all_train.std(ddof=0))
        self.assertEqual(before.metadata["counts"]["train"], {"rows": 8, "segments": 2, "windows": 3})
        self.assertNotAlmostEqual(before.mean, (before.train * before.scale + before.mean).mean(), places=5)
        self.assertEqual(before.validation.shape, (2, 3))
        np.testing.assert_array_equal(before.starts["train"], unix_seconds(BASE) + np.array([0, 1800, 3600]))
        np.testing.assert_allclose(before.train[0] * before.scale + before.mean, train_heights[:3], rtol=0, atol=1e-12)
        different_packing = prepare_dart(self.path, window=4, stride=3)
        self.assertEqual((before.mean, before.scale), (different_packing.mean, different_packing.scale))

        changed_test = routine_rows(test_date, 7000 + np.arange(7) / 10)
        write_station(self.path, train_rows + validation_rows + changed_test)
        after = prepare_dart(self.path, window=3, stride=2)
        self.assertEqual((before.mean, before.scale), (after.mean, after.scale))
        np.testing.assert_array_equal(before.train, after.train)
        np.testing.assert_array_equal(before.validation, after.validation)
        np.testing.assert_array_equal(before.starts["test"], after.starts["test"])
        self.assertFalse(np.array_equal(before.test, after.test))
        self.assertGreater(after.test.min(), 5.0)  # No clipping of retained extremes.
        self.assertNotEqual(before.metadata["source_sha256"], after.metadata["source_sha256"])

    def test_intervening_invalid_or_nonroutine_rows_break_source_contiguity(self):
        blockers = [(2, 2.0), (3, 2.0), (1, 9000.1), (1, 9999.0),
                    (1, np.nan), (1, np.inf), (1, -np.inf), (None, 2.0)]
        heights = [1.0, 2.0, 9000.0, -999.0, 5.0, 6.0]
        for sample_type, height in blockers:
            with self.subTest(sample_type=sample_type, height=height):
                rows = routine_rows(BASE, heights)
                # Duplicate the prior timestamp, but not an observed-T1 group.
                blocker = (rows[2][0], sample_type, height) if sample_type is not None else (None, 1, height)
                rows.insert(3, blocker)
                write_station(self.path, rows)
                data = prepare_dart(self.path, window=3, stride=1)
                np.testing.assert_array_equal(data.starts["train"], unix_seconds(BASE) + np.array([0, 2700]))
                np.testing.assert_allclose(data.train * data.scale + data.mean, np.array(heights).reshape(2, 3), atol=1e-12)
                self.assertEqual(data.metadata["counts"]["train"], {"rows": 6, "segments": 2, "windows": 2})
                self.assertEqual(data.metadata["duplicate_observed_t1_rows"], 0)
                json.dumps(data.metadata, allow_nan=False)

    def test_all_duplicate_observed_timestamp_group_members_are_rejected(self):
        offsets = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 3600, 8100, 9000, 9900, 3600]
        rows = [(BASE + timedelta(seconds=t), 1, float(i)) for i, t in enumerate(offsets)]
        write_station(self.path, rows)
        data = prepare_dart(self.path, window=3, stride=1)
        np.testing.assert_array_equal(data.starts["train"], unix_seconds(BASE) + np.array([0, 900, 4500, 5400, 8100]))
        self.assertEqual(data.metadata["duplicate_observed_t1_rows"], 3)
        self.assertEqual(data.metadata["counts"]["train"], {"rows": 11, "segments": 3, "windows": 5})
        self.assertEqual(data.mean, np.delete(np.arange(14, dtype=float), [4, 9, 13]).mean())
        support = data.starts["train"][:, None] + 900 * np.arange(3)
        self.assertNotIn(unix_seconds(BASE) + 3600, support)

    def test_gaps_reversals_and_short_steps_break_without_sorting(self):
        offsets = [0, 900, 1800, 3600, 4500, 5400, 5399, 6299, 7199, 7259, 8159, 9059]
        write_station(self.path, [(BASE + timedelta(seconds=t), 1, float(i)) for i, t in enumerate(offsets)])
        data = prepare_dart(self.path, window=3, stride=2)
        np.testing.assert_array_equal(data.starts["train"], unix_seconds(BASE) + np.array([0, 3600, 5399, 7259]))
        np.testing.assert_allclose(data.train * data.scale + data.mean, np.arange(12).reshape(4, 3), atol=1e-14)
        self.assertEqual(data.metadata["counts"]["train"]["segments"], 4)

    def test_exact_half_open_embargo_edges_and_segment_anchored_evaluation(self):
        edges = [datetime(2022, 4, 26, tzinfo=timezone.utc), datetime(2022, 4, 30, tzinfo=timezone.utc),
                 datetime(2022, 11, 23, tzinfo=timezone.utc), datetime(2022, 11, 27, tzinfo=timezone.utc)]
        rows = []
        for edge, offsets in zip(edges, ((-1800, -900, 0, 900), (-900, 0, 900), (-1800, -900, 0, 900), (-900, 0, 900))):
            rows.extend((edge + timedelta(seconds=t), 1, float(len(rows) + i)) for i, t in enumerate(offsets))
        write_station(self.path, rows)
        data = prepare_dart(self.path, window=2, stride=1)
        expected = {"train": [unix_seconds(edges[0]) - 1800],
                    "validation": [unix_seconds(edges[1]), unix_seconds(edges[2]) - 1800],
                    "test": [unix_seconds(edges[3])]}
        for split in SPLITS:
            np.testing.assert_array_equal(data.starts[split], expected[split])
        self.assertEqual(data.metadata["embargo_rows"], 6)

    def test_microsecond_datetime_storage_still_returns_unix_seconds(self):
        write_station(self.path, routine_rows(BASE, [1.0, 2.0, 3.0, 4.0]))
        original = pd.to_datetime

        def microseconds(*args, **kwargs):
            result = original(*args, **kwargs).dt.as_unit("us")
            self.assertEqual(result.dtype.unit, "us")
            return result

        with mock.patch("prigm_data.pd.to_datetime", side_effect=microseconds):
            data = prepare_dart(self.path, window=3, stride=1)
        np.testing.assert_array_equal(data.starts["train"], [1640995200, 1640996100])

    def test_empty_window_packs_preserve_two_dimensions_and_dtype(self):
        write_station(self.path, routine_rows(BASE, [1.0, 2.0]))
        data = prepare_dart(self.path)
        for split in SPLITS:
            self.assertEqual(getattr(data, split).shape, (0, 192))
            self.assertEqual(getattr(data, split).dtype, np.float64)
            self.assertEqual(data.starts[split].shape, (0,))
            self.assertEqual(data.starts[split].dtype, np.int64)

    def test_invalid_normalization_parameters_and_unknown_station_fail(self):
        for heights, message in (([1.0, 1.0], "positive finite scale"), ([9999.0, np.nan], "No valid training")):
            write_station(self.path, routine_rows(BASE, heights))
            with self.assertRaisesRegex(ValueError, message):
                prepare_dart(self.path)
        for kwargs in ({"window": 0}, {"stride": 0}, {"stride": 1.5}, {"window": True}):
            with self.assertRaises(ValueError):
                prepare_dart(self.path, **kwargs)
        with self.assertRaisesRegex(ValueError, "approved DART boundaries"):
            prepare_dart(self.path.with_name("unknown.txt"))
        with self.assertRaisesRegex(ValueError, "Unknown PriGM dataset"):
            prepare_dataset("unknown", self.path.parent)

    def test_eight_source_columns_required(self):
        self.path.write_text("header\nheader\n2022 01 01 00 00 00 1\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "eight columns"):
            prepare_dart(self.path)


class SyntheticTest(unittest.TestCase):
    def test_default_dispatch_sizes_and_repeatability(self):
        data = prepare_dataset("gaussian", ROOT / "does-not-exist")
        repeated = prepare_synthetic()
        for split, size in zip(SPLITS, (8192, 2048, 2048)):
            array = getattr(data, split)
            self.assertEqual(array.shape, (size, 192))
            self.assertEqual(array.dtype, np.float64)
            self.assertTrue(np.isfinite(array).all())
            np.testing.assert_array_equal(array, getattr(repeated, split))
            np.testing.assert_array_equal(data.starts[split], np.arange(size, dtype=np.int64))
        self.assertEqual(data.mean, 0.0)
        self.assertEqual(data.scale, 1.0)
        self.assertFalse(np.array_equal(data.validation, data.test))
        self.assertEqual(data.metadata, repeated.metadata)
        json.dumps(data.metadata, allow_nan=False)
        # Independent splits: changing training size cannot change held-out draws.
        changed_train = prepare_synthetic(sizes=(7, 2048, 2048))
        np.testing.assert_array_equal(data.validation, changed_train.validation)
        np.testing.assert_array_equal(data.test, changed_train.test)

    def test_fourier_oracle_and_declared_rng_draws_without_standardization(self):
        window, sizes = 24, (5, 4, 3)
        data = prepare_synthetic(window=window, sizes=sizes)
        basis = np.empty((window, 8), dtype=np.float64)
        for harmonic in range(1, 5):
            angle = 2 * np.pi * harmonic * np.arange(window) / window
            basis[:, 2 * harmonic - 2] = np.cos(angle) * np.sqrt(2 / window)
            basis[:, 2 * harmonic - 1] = np.sin(angle) * np.sqrt(2 / window)
        np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=1e-14)
        singular_values = np.linspace(4.0, 1.0, 8)
        loading = basis @ np.diag(singular_values)
        covariance = loading @ loading.T + 0.25**2 * np.eye(window)
        np.testing.assert_allclose(data.oracle_covariance, covariance, rtol=1e-14, atol=1e-14)
        expected_eigenvalues = np.sort(np.r_[singular_values**2 + 0.25**2, np.full(window - 8, 0.25**2)])
        np.testing.assert_allclose(np.linalg.eigvalsh(data.oracle_covariance), expected_eigenvalues, atol=1e-14)
        for split, size, seed in zip(SPLITS, sizes, (1101, 1102, 1103)):
            rng = np.random.Generator(np.random.PCG64(seed))
            expected = rng.standard_normal((size, 8)) @ loading.T + 0.25 * rng.standard_normal((size, window))
            np.testing.assert_array_equal(getattr(data, split), expected)
        self.assertEqual(data.metadata["generator"]["singular_values"], singular_values.tolist())
        self.assertEqual(data.metadata["generator"]["seeds"], dict(zip(SPLITS, (1101, 1102, 1103))))
        self.assertEqual(data.metadata["generator"]["noise_std"], 0.25)
        self.assertEqual(data.oracle_covariance.dtype, np.float64)

    def test_synthetic_parameters_and_empty_splits(self):
        for kwargs in ({"window": 8}, {"window": 0}, {"sizes": (1, 2)},
                       {"sizes": (-1, 2, 3)}, {"sizes": (1.5, 2, 3)}):
            with self.assertRaises(ValueError):
                prepare_synthetic(**kwargs)
        data = prepare_synthetic(window=9, sizes=(1, 0, 0))
        self.assertEqual(data.validation.shape, (0, 9))
        self.assertEqual(data.test.shape, (0, 9))


if __name__ == "__main__":
    unittest.main()
