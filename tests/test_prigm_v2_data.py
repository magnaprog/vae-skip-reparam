"""CPU tests for v2 synthetic families and DART qualification; no TensorFlow required."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prigm_v2_data import (GAUSSIAN_SEEDS, MIXTURE_SEEDS, MIXTURE_SIGN_SCALE,
                           NOISE_STD, PHASES, prepare_dart_record, prepare_gaussian,
                           prepare_mixture, qualify_candidate, same_phase_donors,
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


def test_same_phase_donors_have_disjoint_support_and_different_blocks():
    starts = np.array([h * 3600 + day * 86400 for day in range(40) for h in PHASES],
                      dtype=np.int64)
    source = starts // 900
    donors = same_phase_donors(starts, seed=66001, source_starts=source, block_origin=0)
    hours = ((starts % 86400) // 3600).astype(np.int64)
    blocks = starts // (14 * 86400)
    assert np.all(donors != np.arange(len(starts)))
    assert np.all(hours[donors] == hours)
    assert np.all(blocks[donors] != blocks)
    assert np.all(np.abs(source[donors] - source) >= 192)
    assert np.all(np.abs(starts[donors] - starts) > 191 * 900)
    again = same_phase_donors(starts, seed=66001, source_starts=source, block_origin=0)
    np.testing.assert_array_equal(donors, again)


def test_donors_use_declared_block_origin():
    starts = np.array([13, 15], dtype=np.int64) * 86400
    source = starts // 900
    np.testing.assert_array_equal(same_phase_donors(starts, 0, source, 0), [1, 0])
    assert same_phase_donors(starts, 0, source, 7 * 86400) is None


def test_donors_reject_overlap_across_block_boundary():
    starts = np.array([13, 14], dtype=np.int64) * 86400
    assert same_phase_donors(starts, 0, starts // 900, 0) is None


def test_donors_reject_overlapping_raw_support():
    starts = np.array([0, 15], dtype=np.int64) * 86400
    assert same_phase_donors(starts, 0, np.array([0, 32]), 0) is None


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("days,source", [
    ([0, 13, 14, 28], [0, 1248, 1344, 2688]),
    ([0, 15, 30], [0, 32, 5000]),
])
def test_donors_exclude_earlier_overlapping_support_for_integer_dtypes(dtype, days, source):
    starts = np.asarray(days, dtype=np.int64) * 86400
    source = np.asarray(source, dtype=np.int64)
    for seed in (0, 1):
        donors = same_phase_donors(starts.astype(dtype), seed, source.astype(dtype), 0)
        assert donors is not None
        assert np.all(np.abs(starts[donors] - starts) > 191 * 900)
        assert np.all(np.abs(source[donors] - source) >= 192)
        expected = same_phase_donors(starts, seed, source, 0)
        np.testing.assert_array_equal(donors, expected)


def test_no_donor_disables_whole_diagnostic_without_dropping_targets():
    starts = np.array([0, 15 * 86400, 30 * 86400 + 4 * 3600], dtype=np.int64)
    assert same_phase_donors(starts, 0, starts // 900, 0) is None


@pytest.mark.parametrize("source", [np.array([0.0, 192.5]), np.array([0, -1]),
                                    np.array([0])])
def test_donors_reject_invalid_source_indices(source):
    starts = np.array([0, 15 * 86400], dtype=np.int64)
    with pytest.raises(ValueError):
        same_phase_donors(starts, 0, source, 0)


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


def test_prepare_dart_record_matches_archived_44402_2018_except_donors():
    path = Path("datasets-v2-acquisition/raw/44402t2018.txt")
    if not path.is_file():
        pytest.skip("44402 2018 source not present")
    prepared = prepare_dart_record(path, "2018-01-01", "2019-01-01")
    archived = Path("runs-v2/dart_44402_2018.npz")
    if not archived.is_file():
        pytest.skip("archived 44402 2018 arrays not present")
    with np.load(archived) as store:
        for key in ("train", "validation", "test", "train_weights",
                    "validation_weights", "eval_weights", "starts_train",
                    "starts_validation", "starts_test"):
            np.testing.assert_allclose(prepared[key], store[key], rtol=0, atol=0)
        assert float(prepared["mean"]) == pytest.approx(float(store["mean"]))
        assert float(prepared["scale"]) == pytest.approx(float(store["scale"]))
        hours = ((prepared["starts_test"] % 86400) // 3600).astype(np.int64)
        assert prepared["reassignment_available"]
        donors = prepared["reassignment_donors"]
        source = prepared["source_starts_test"]
        blocks = ((prepared["starts_test"] - prepared["reassignment_block_origin"])
                  // (14 * 86400))
        assert np.all(donors != np.arange(len(donors)))
        assert np.all(hours[donors] == hours)
        assert np.all(blocks[donors] != blocks)
        assert np.all(np.abs(source[donors] - source) >= 192)
        # Archived donors were stored, not regenerated from this helper.


def test_three_ledger_candidates_match_recorded_eligibility():
    raw = Path("datasets-v2-acquisition/raw")
    cases = [
        ("44402t2017.txt", "2017-01-01", "2018-01-01", False, 0.9176940639269406, 108, 19, 35),
        ("46407t2012.txt", "2012-01-01", "2012-10-05", False, 0.43098021582733814, 53, 0, 0),
        ("44402t2018.txt", "2018-01-01", "2019-01-01", True, 0.990296803652968, 101, 33, 33),
    ]
    missing = [name for name, *_ in cases if not (raw / name).is_file()]
    if missing:
        pytest.skip("DART acquisition files absent: " + ", ".join(missing))
    for name, start, end, eligible, coverage, train_packs, val_packs, test_packs in cases:
        q = qualify_candidate(raw / name, start, end, station=name.split("t")[0])
        assert q.eligible is eligible
        assert q.coverage == pytest.approx(coverage)
        assert q.packs.get("train", 0) == train_packs
        assert q.packs.get("validation", 0) == val_packs
        assert q.packs.get("test", 0) == test_packs
