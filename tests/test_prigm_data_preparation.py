"""CPU tests for Gaussian simulations and DART interval eligibility; no TensorFlow required."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prigm_data_preparation import (GAUSSIAN_SEEDS, MIXTURE_SEEDS, MIXTURE_SIGN_SCALE,
                           NOISE_STD, PHASES, prepare_dart_record, prepare_gaussian,
                           prepare_mixture, qualify_candidate,
                           sinusoidal_basis)


def test_basis_orthonormal():
    basis = sinusoidal_basis(192)
    np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=1e-12)


def test_families_share_covariance():
    g, m = prepare_gaussian(), prepare_mixture()
    np.testing.assert_allclose(g["oracle_covariance"], m["oracle_covariance"],
                               atol=1e-10)


def test_first_projection_kurtosis_matches_family_fixtures():
    from scipy import stats
    basis = sinusoidal_basis(192)
    g, m = prepare_gaussian(), prepare_mixture()
    kg = stats.kurtosis(g["train"] @ basis[:, 0])
    km = stats.kurtosis(m["train"] @ basis[:, 0])
    assert abs(kg) < 0.2
    assert km < -0.8  # two-point mixture along the first direction


def test_gaussian_splits_repeat_and_family_seeds_are_disjoint():
    g1 = prepare_gaussian()
    g2 = prepare_gaussian()
    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(g1[split], g2[split])
    assert set(GAUSSIAN_SEEDS.values()) .isdisjoint(set(MIXTURE_SEEDS.values()))
    g, m = prepare_gaussian(), prepare_mixture()
    assert not np.array_equal(g["train"], m["train"])


def test_mixture_first_direction_variance():
    basis = sinusoidal_basis(192)
    m = prepare_mixture()
    projected = m["train"] @ basis[:, 0]
    # 12 (sign) + 4 (latent with s=2 -> 4) + noise 0.0625 = 16.0625
    assert abs(projected.var() - (12.0 + 4.0 + NOISE_STD**2)) < 0.5


def test_declared_mean_zero_scale_one():
    for prepared in (prepare_gaussian(), prepare_mixture()):
        assert prepared["mean"] == 0.0 and prepared["scale"] == 1.0


@pytest.mark.parametrize("intervening", [
    pytest.param("2018 01 01 23 52 30 2 4000\n", id="non_t1"),
    pytest.param("2018 01 01 23 52 30 1 9999\n", id="placeholder"),
    pytest.param("2018 13 01 23 52 30 1 4000\n", id="invalid_date"),
    pytest.param("2018 01 01 23 52 30 1 nan\n", id="invalid_height"),
    pytest.param("2018 01 01 23 52 30 1 4000\n" * 2, id="duplicate"),
])
def test_qualification_rejected_rows_break_support(tmp_path, intervening):
    start = datetime(2018, 1, 1)
    rows = [
        f"{start + timedelta(minutes=15 * i):%Y %m %d %H %M %S} 1 {i}\n"
        for i in range(192)
    ]
    rows.insert(96, intervening)
    path = tmp_path / "candidate.txt"
    path.write_text("#YY MM DD hh mm ss T HEIGHT\n#yr mo dy hr mn s - m\n" + "".join(rows))

    q = qualify_candidate(path, "2018-01-01", "2019-01-01", window_length=192)

    assert not q.eligible
    assert q.windows["train"] == 0
    assert q.window_grid["train"] == []
    assert q.mean is None
    assert q.scale is None


def test_qualification_reversal_breaks_support(tmp_path):
    start = datetime(2018, 1, 1)
    rows = [
        f"{start + timedelta(minutes=15 * i):%Y %m %d %H %M %S} 1 {i}\n"
        for i in range(192)
    ]
    rows = rows[96:] + rows[:96]
    path = tmp_path / "candidate.txt"
    path.write_text("#YY MM DD hh mm ss T HEIGHT\n#yr mo dy hr mn s - m\n" + "".join(rows))

    q = qualify_candidate(path, "2018-01-01", "2019-01-01", window_length=192)

    assert not q.eligible
    assert q.windows["train"] == 0
    assert q.window_grid["train"] == []
    assert q.mean is None
    assert q.scale is None


def test_qualification_normalizes_exact_union_of_reversed_segments(tmp_path):
    heights = np.arange(449, dtype=np.float64) ** 2 / 100
    heights[224] = 9999
    rows = []
    for start, raw_start in ((datetime(2018, 3, 1, 0, 15), 0),
                             (datetime(2018, 1, 1, 0, 15), 225)):
        rows.extend(
            f"{start + timedelta(minutes=15 * i):%Y %m %d %H %M %S} 1 {heights[raw_start + i]}\n"
            for i in range(224)
        )
    rows.insert(224, "2018 02 01 00 00 00 1 9999\n")
    path = tmp_path / "candidate.txt"
    path.write_text("#YY MM DD hh mm ss T HEIGHT\n#yr mo dy hr mn s - m\n" + "".join(rows))

    q = qualify_candidate(path, "2018-01-01", "2019-01-01", window_length=192)

    assert not q.eligible
    assert q.windows["train"] == 4
    assert q.window_grid["train"] == [
        (datetime(2018, 1, 1, 4, tzinfo=timezone.utc).timestamp(), 240),
        (datetime(2018, 1, 1, 8, tzinfo=timezone.utc).timestamp(), 256),
        (datetime(2018, 3, 1, 4, tzinfo=timezone.utc).timestamp(), 15),
        (datetime(2018, 3, 1, 8, tzinfo=timezone.utc).timestamp(), 31),
    ]
    covered = np.concatenate((heights[15:223], heights[240:448]))
    assert q.mean == pytest.approx(covered.mean())
    assert q.scale == pytest.approx(covered.std(ddof=0))


def test_prepare_dart_record_rejects_ineligible(tmp_path):
    start = datetime(2018, 1, 1)
    rows = [
        f"{start + timedelta(minutes=15 * i):%Y %m %d %H %M %S} 1 {i}\n"
        for i in range(192)
    ]
    path = tmp_path / "candidate.txt"
    path.write_text("#YY MM DD hh mm ss T HEIGHT\n#yr mo dy hr mn s - m\n" + "".join(rows))
    with pytest.raises(ValueError, match="ineligible"):
        prepare_dart_record(path, "2018-01-01", "2019-01-01")


@pytest.fixture
def complete_annual_record(tmp_path):
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    heights = 4000 + np.arange(365 * 96, dtype=np.float64) / 10000
    path = tmp_path / "44402t2018.txt"
    with path.open("w") as stream:
        stream.write("#YY MM DD hh mm ss T HEIGHT\n#yr mo dy hr mn s - m\n")
        for i, height in enumerate(heights):
            date = start + timedelta(minutes=15 * i)
            stream.write(f"{date:%Y %m %d %H %M %S} 1 {float(height)!r}\n")
    return path, heights, int(start.timestamp())


def test_complete_annual_record_has_expected_windows_weights_and_normalization(complete_annual_record):
    path, heights, origin = complete_annual_record
    prepared = prepare_dart_record(path, "2018-01-01", "2019-01-01")
    assert set(prepared) == {
        "train", "validation", "test", "train_weights", "validation_weights",
        "eval_weights", "starts_train", "starts_validation", "starts_test",
        "source_starts_train", "source_starts_validation", "source_starts_test",
        "mean", "scale", "dataset_name",
    }
    # Splits exclude two days on either side of days 219 and 292. At 15 minute
    # cadence, the three usable source-row intervals are [0,20832),
    # [21216,27840), and [28224,35040).
    expected_mean = heights[:20832].mean()
    expected_scale = heights[:20832].std(ddof=0)
    assert prepared["mean"] == expected_mean
    assert prepared["scale"] == expected_scale
    assert prepared["dataset_name"] == "dart_44402_2018"
    for split, first, last, count, weight_key in (
        ("train", 0, 20640, 1291, "train_weights"),
        ("validation", 21216, 27648, 403, "validation_weights"),
        ("test", 28224, 34848, 415, "eval_weights"),
    ):
        source_rows = np.arange(first, last + 1, 16, dtype=np.int64)
        assert len(source_rows) == count
        expected = (heights[source_rows[:, None] + np.arange(192)] - expected_mean) / expected_scale
        np.testing.assert_array_equal(prepared[split], expected)
        np.testing.assert_array_equal(prepared["source_starts_" + split], source_rows)
        np.testing.assert_array_equal(prepared["starts_" + split], origin + 900 * source_rows)
        hours = ((origin + 900 * source_rows) % 86400) // 3600
        for hour in PHASES:
            mask = hours == hour
            np.testing.assert_array_equal(prepared[weight_key][mask],
                                          np.full(mask.sum(), 1 / (6 * mask.sum())))
            assert prepared[weight_key][mask].sum() == pytest.approx(1 / 6)


def test_complete_annual_record_meets_stated_eligibility(complete_annual_record):
    path, _, _ = complete_annual_record
    result = qualify_candidate(path, "2018-01-01", "2019-01-01")
    assert result.eligible
    assert result.coverage == 1
    assert result.packs == {"train": 108, "validation": 34, "test": 35}
    assert result.windows == {"train": 1291, "validation": 403, "test": 415}
    assert all(count > 0 for counts in result.phases.values() for count in counts.values())


def test_annual_record_below_coverage_requirement_is_rejected(complete_annual_record):
    path, _, _ = complete_annual_record
    lines = path.read_text().splitlines(keepends=True)
    for i in range(2, len(lines), 4):
        fields = lines[i].split()
        fields[-1] = "9999"
        lines[i] = " ".join(fields) + "\n"
    path.write_text("".join(lines))
    result = qualify_candidate(path, "2018-01-01", "2019-01-01")
    assert not result.eligible
    assert result.coverage == 0.75
    assert any(reason.startswith("coverage ") for reason in result.reasons)


        # Archived donors were stored, not regenerated from this helper.


