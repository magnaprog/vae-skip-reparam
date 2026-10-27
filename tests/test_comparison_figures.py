"""Comparisons retain individual fits, exact comparisons and dated intervals."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SPEC = importlib.util.spec_from_file_location("comparison_renderer", FIGURES / "render.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
values = renderer.figure_values
ORDER = ("no_skip_base", "skip_base", "no_skip_high", "skip_high")
SEEDS = (42, 1729, 31415)
SCORES = ("joint", "centered", "increment")


@pytest.fixture(scope="module")
def data():
    return renderer.load_comparisons(FIGURES)


@pytest.fixture(scope="module")
def evidence():
    # Public artist expectations read packaged absolute scores directly, without
    # invoking the production loader or per-fit reduction under test.
    return json.loads((FIGURES / "comparisons.json").read_text())["records"]


def expected_ppca(source, condition):
    level = {"no_skip_base": "low", "skip_base": "low", "no_skip_high": "high", "skip_high": "high"}[condition]
    return np.array([[source["neural_scores"][condition][score][seed]["mean"]
                      - source["baseline_scores"][level + "/ppca_matched"]["scores"][score]["mean"]
                      for score in SCORES] for seed in range(3)])


def test_per_fit_values_agree_with_independent_released_reductions(data):
    directory = ROOT / "results"
    with (directory / "ppca_means.csv").open() as stream:
        references = {(r["record"], r["variance_group"], r["geometry"]): float(r["ES"])
                      for r in csv.DictReader(stream)}
    with (directory / "energy_terms_fits.csv").open() as stream:
        fits = [r for r in csv.DictReader(stream) if r["variant"] == "original"]
    assert len(fits) == 252
    for fit in fits:
        reference = references[(fit["record"], fit["variance_group"], fit["geometry"])]
        expected = float(fit["ES"]) - reference
        actual = values.ppca_comparison_points(data["records"][fit["record"]], fit["condition"])[
            SEEDS.index(int(fit["seed"])), SCORES.index(fit["geometry"])]
        # The CSV recomputes energy terms. Its scores differ from the plotting
        # inputs by at most 1.21e-13 through floating-point rounding.
        assert abs(actual - expected) <= 1.3e-13


@pytest.mark.parametrize("change", ["missing_record", "extra_record", "wrong_station", "wrong_year",
                                   "seed_order", "boolean_schema", "old_schema", "invalid_condition", "missing_condition", "extra_score",
                                   "short_seed_axis", "nonfinite", "boolean_score", "condition_swap",
                                   "pooled_fits", "wrong_baseline", "bad_stream_mean"])
def test_bad_ppca_contract_is_rejected(data, change):
    broken = deepcopy(data)
    record = broken["records"]["23461t2022"]
    if change == "missing_record":
        del broken["records"]["dart_21418_2018"]
    elif change == "extra_record":
        broken["records"]["extra"] = deepcopy(record)
    elif change == "wrong_station":
        record["station"] = "44402"
    elif change == "wrong_year":
        record["year"] = 2018
    elif change == "seed_order":
        broken["training_seeds"].reverse()
    elif change == "boolean_schema":
        broken["schema_version"] = True
    elif change == "old_schema":
        broken["schema_version"] = 1
    elif change == "invalid_condition":
        record["ppca_comparison"]["A"] = record["ppca_comparison"].pop("no_skip_base")
    elif change == "missing_condition":
        del record["ppca_comparison"]["no_skip_high"]
    elif change == "extra_score":
        record["ppca_comparison"]["no_skip_base"]["extra"] = [0., 0., 0.]
    elif change == "short_seed_axis":
        record["ppca_comparison"]["no_skip_base"]["joint"].pop()
    elif change == "nonfinite":
        record["ppca_comparison"]["no_skip_base"]["joint"][0] = float("nan")
    elif change == "boolean_score":
        record["ppca_comparison"]["no_skip_base"]["joint"][0] = True
    elif change == "condition_swap":
        record["ppca_comparison"]["no_skip_base"], record["ppca_comparison"]["skip_base"] = (
            record["ppca_comparison"]["skip_base"], record["ppca_comparison"]["no_skip_base"])
    elif change == "pooled_fits":
        fits = record["ppca_comparison"]["no_skip_base"]["joint"]
        record["ppca_comparison"]["no_skip_base"]["joint"] = [float(np.mean(fits))] * 3
    elif change == "wrong_baseline":
        record["baseline_scores"]["low/ppca_matched"] = record["baseline_scores"]["high/ppca_matched"]
    else:
        record["neural_scores"]["no_skip_base"]["joint"][0]["mean"] += 1
    with pytest.raises(ValueError):
        values.validate_comparisons(broken, renderer.EVIDENCE_SHA)


def test_reader_authenticates_bytes_before_parsing(tmp_path):
    (tmp_path / "comparisons.json").write_text("not JSON")
    with pytest.raises(ValueError, match="data hash differs"):
        renderer.load_comparisons(tmp_path)


def test_reader_rejects_wrong_science_and_duplicate_keys(data, tmp_path, monkeypatch):
    wrong = deepcopy(data)
    wrong["scientific_manifest_sha256"] = "0" * 64
    for raw, message in [(json.dumps(wrong).encode(), "different evidence"),
                         (b'{"schema_version":1,"schema_version":1}', "Duplicate")]:
        (tmp_path / "comparisons.json").write_bytes(raw)
        monkeypatch.setattr(renderer, "COMPARISONS_SHA", sha256(raw).hexdigest())
        with pytest.raises(ValueError, match=message):
            renderer.load_comparisons(tmp_path)


def check_ppca_artists(fig, records, evidence):
    rows = [(name, condition) for name in values.RECORDS for condition in ORDER]
    assert len(fig.axes) == 3
    for panel, ax in enumerate(fig.axes):
        dots = [line for line in ax.lines if line.get_marker() in renderer.MARKERS.values()]
        assert len(dots) == 84 and not ax.collections  # no interval/CI artists
        for row, (name, condition) in enumerate(rows):
            expected = expected_ppca(evidence[name], condition)
            for seed, offset in enumerate((-.2, 0, .2)):
                point = dots[row * 3 + seed]
                np.testing.assert_array_equal(point.get_xdata(), [expected[seed, panel]])
                np.testing.assert_array_equal(point.get_ydata(), [row + offset])
                assert point.get_color() == renderer.COLORS[condition]
                assert point.get_marker() == renderer.MARKERS[condition]
                assert point.get_linestyle() == "-" and point.get_linewidth() == 0
        assert any(list(line.get_xdata()) == [0, 0] for line in ax.lines)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [renderer.CONDITION_LABELS[condition] for _, condition in rows]
    assert fig.get_figwidth() == 5.5


def test_ppca_plot_uses_every_fit_in_explicit_condition_order(data, evidence, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.ppca_comparison(tmp_path, data["records"])
    fig, = captured
    try:
        check_ppca_artists(fig, data["records"], evidence)
    finally:
        plt.close(fig)


def test_plot_oracle_rejects_wrong_condition_payload_with_correct_row(data, evidence, monkeypatch, tmp_path):
    original = values.ppca_comparison_points
    monkeypatch.setattr(values, "ppca_comparison_points",
                        lambda record, condition: original(record, {"no_skip_base": "skip_base", "skip_base": "no_skip_base"}.get(condition, condition)))
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.ppca_comparison(tmp_path, data["records"])
    fig, = captured
    try:
        with pytest.raises(AssertionError):
            check_ppca_artists(fig, data["records"], evidence)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("change", ["midpoints", "bad_end", "reverse", "duplicate", "empty",
                                   "nonfinite", "wrong_count", "zero_mass", "missing_seed", "pooled_seeds"])
def test_bad_interval_contract_is_rejected(data, change):
    broken = deepcopy(data)
    intervals = broken["records"]["dart_21415_2018"]["intervals"]["14"]
    if change == "midpoints":
        for item in intervals:
            item["start_unix"] += 7 * 86400
            item["end_unix"] += 7 * 86400
    elif change == "bad_end":
        intervals[0]["end_unix"] += 1
    elif change == "reverse":
        intervals.reverse()
    elif change == "duplicate":
        intervals.insert(1, deepcopy(intervals[0]))
    elif change == "empty":
        intervals.clear()
    elif change == "nonfinite":
        intervals[0]["high"][0] = float("inf")
    elif change == "wrong_count":
        intervals[0]["windows"] += 1
    elif change == "zero_mass":
        intervals[0]["mass"] = 0.
    elif change == "missing_seed":
        intervals[0]["base"].pop()
    else:
        for item in intervals:
            item["base"] = [float(np.mean(item["base"]))] * 3
    with pytest.raises(ValueError):
        values.validate_comparisons(broken, renderer.EVIDENCE_SHA)


def check_interval_artists(fig, names, records):
    assert len(fig.axes) == 4 * len(names) and fig.get_figwidth() == 5.5
    total = bins = 0
    for row, name in enumerate(names):
        all_intervals = [item for days in ("14", "28") for item in records[name]["intervals"][days]]
        scores = [0.] + [score for item in all_intervals for variance in ("base", "high")
                         for score in item[variance]]
        padding = (max(scores) - min(scores)) * .18 or .1
        expected_ylim = (min(scores) - padding, max(scores) + padding)
        dates = [mdates.date2num(datetime.fromtimestamp(item["start_unix"], timezone.utc))
                 for item in all_intervals]
        padding = max(3., (max(dates) - min(dates)) * .085)
        expected_xlim = (min(dates) - padding, max(dates) + padding)
        for col, days in enumerate(("14", "28")):
            ax, coverage = fig.axes[row * 4 + col * 2:row * 4 + col * 2 + 2]
            source = records[name]["intervals"][days]
            expected_x = [mdates.date2num(datetime.fromtimestamp(i["start_unix"], timezone.utc)) for i in source]
            lines = [line for line in ax.lines if line.get_marker() in ("o", "s", "^")]
            assert len(lines) == 6 and not ax.collections
            for variance_index, variance in enumerate(("base", "high")):
                for seed, marker in enumerate(("o", "s", "^")):
                    line = lines[variance_index * 3 + seed]
                    np.testing.assert_array_equal(line.get_xdata(), expected_x)
                    np.testing.assert_array_equal(line.get_ydata(), [i[variance][seed] for i in source])
                    lower, upper = ax.get_ylim()
                    assert lower < min(0., *line.get_ydata()) <= max(0., *line.get_ydata()) < upper
                    xy = np.column_stack((expected_x, line.get_ydata()))
                    shift = line.get_transform().transform(xy) - ax.transData.transform(xy)
                    np.testing.assert_allclose(shift[:, 0], (-4., 0., 4.)[seed] * fig.dpi / 72,
                                               rtol=0, atol=1e-9)
                    np.testing.assert_array_equal(shift[:, 1], np.zeros(len(expected_x)))
                    assert line.get_linestyle() == "None"
                    assert line.get_marker() == marker and line.get_markersize() == 3.5
                    assert line.get_color() == ("#CC3311" if variance == "base" else "#009988")
                    assert line.get_markerfacecolor() == ("#CC3311" if variance == "base" else "white")
                    total += len(expected_x)
            assert any(list(line.get_ydata()) == [0, 0] for line in ax.lines)
            assert ax.get_xlim() == expected_xlim and ax.get_ylim() == expected_ylim
            np.testing.assert_array_equal(coverage.get_xticks(), expected_x)
            text = [t for t in coverage.texts if t.get_transform() is coverage.transData]
            assert len(text) == 3 * len(source)
            for index, item in enumerate(source):
                start, n, mass = text[3 * index:3 * index + 3]
                assert start.get_text() == datetime.fromtimestamp(item["start_unix"], timezone.utc).strftime("%b\n%d")
                assert n.get_text() == str(item["windows"])
                rounded = float(f"{100 * item['mass']:.3g}")
                assert mass.get_text() == f"{rounded:.2f}".rstrip("0").rstrip(".")
                assert start.get_position() == (expected_x[index], .70)
                assert n.get_position() == (expected_x[index], .32)
                assert mass.get_position() == (expected_x[index], .06)
                assert all(t.get_visible() and t.get_rotation() == 0 for t in (start, n, mass))
            bins += len(source)
        left, right = fig.axes[row * 4], fig.axes[row * 4 + 2]
        assert left.get_ylim() == right.get_ylim() and left.get_xlim() == right.get_xlim()
        assert left.get_yscale() == right.get_yscale() == "linear"
    return total, bins


def check_interval_layout(fig, names, records):
    fig.canvas.draw()
    draw = fig.canvas.get_renderer()
    labels = list(fig.texts)
    plots = fig.axes[::2]
    for ax, coverage in zip(plots, fig.axes[1::2], strict=True):
        labels.extend(coverage.texts)
        labels.extend(t.label1 for t in ax.yaxis._update_ticks() if t.label1.get_visible())
        for text in coverage.texts:
            assert text.get_visible()
            box = text.get_window_extent(draw)
            assert box.y1 < ax.bbox.y0
            # All three metadata rows retain the physical start-date position.
            if text.get_transform() is coverage.transData:
                x = ax.transData.transform((text.get_position()[0], 0))[0]
                assert (box.x0 + box.x1) / 2 == pytest.approx(x)
            for tick in ax.xaxis.get_major_ticks():
                if tick.tick1line.get_visible():
                    assert not box.overlaps(tick.tick1line.get_window_extent(draw))
        assert not any(t.tick1line.get_visible() or t.label1.get_visible()
                       for t in coverage.xaxis.get_major_ticks())
    for row, name in enumerate(names):
        label, = [t for t in fig.texts if t.get_text() ==
                  f"{records[name]['station']} / {records[name]['year']}"]
        assert label.get_rotation() == 90 and label.get_fontsize() == 8.5
        # The station identifies the whole row, including its coverage panel.
        top = fig.axes[row * 4].bbox.y1
        bottom = fig.axes[row * 4 + 1].bbox.y0
        box = label.get_window_extent(draw)
        assert (box.y0 + box.y1) / 2 == pytest.approx((top + bottom) / 2)
    boxes = [text.get_window_extent(draw) for text in labels]
    boxes += [legend.get_window_extent(draw) for legend in fig.legends]
    for box in boxes:
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert all(not box.overlaps(ax.bbox) for ax in plots)
    # Includes adjacent dates, metadata rows, station names and the global ylabel.
    for first, second in itertools.combinations(boxes, 2):
        assert not first.overlaps(second)


@pytest.mark.parametrize("dpi", [100, 144])
def test_temporal_dates_counts_and_weights_are_separate_at_print_width(data, evidence, monkeypatch, tmp_path, dpi):
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.intervals(tmp_path, data["records"])
    try:
        for fig, names in zip(captured, (values.RECORDS[:4], values.RECORDS[4:]), strict=True):
            fig.set_dpi(dpi)
            check_interval_artists(fig, names, evidence)
            check_interval_layout(fig, names, evidence)
    finally:
        for fig in captured:
            plt.close(fig)


def test_temporal_artists_keep_seed_values_dates_and_coverage(data, evidence, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.intervals(tmp_path, data["records"])
    try:
        assert len(captured) == 2
        assert captured[0].get_figheight() <= 7.15
        assert check_interval_artists(captured[0], values.RECORDS[:4], evidence) == (180, 30)
        assert check_interval_artists(captured[1], values.RECORDS[4:], evidence) == (144, 24)
        for fig in captured:
            assert [t.get_text() for t in fig.legends[0].get_texts()] == [
                "Base variance", "High variance", "42", "1729", "31415"]
            for ax in fig.axes:
                assert all(t.get_fontsize() == 8.5 for t in ax.get_xticklabels() + ax.get_yticklabels()
                           if t.get_visible())
                assert all(t.get_fontsize() == 8.5 for t in ax.texts)
        # Explicitly guard the sparse positive bins and the future-end/date trap.
        last = evidence["44402t2016"]["intervals"]["28"][-1]
        assert datetime.fromtimestamp(last["start_unix"], timezone.utc).isoformat().startswith("2016-12-30")
        assert datetime.fromtimestamp(last["end_unix"], timezone.utc).year == 2017
        for row, expected in ((1, ("1", "0.44")), (2, ("10", "5.26"))):
            text = [t.get_text() for t in captured[1].axes[row * 4 + 1].texts]
            assert all(value in text for value in expected)
        annotations = [t.get_text() for a in captured[1].axes for t in a.texts]
        assert "+0.0133" in annotations
    finally:
        for fig in captured:
            plt.close(fig)


def test_temporal_oracle_rejects_midpoint_coordinates(data, evidence, monkeypatch, tmp_path):
    original = values.interval_points

    def wrong(record, days):
        result = original(record, days)
        result["start_unix"] = (result["start_unix"] + result["end_unix"]) // 2
        return result

    monkeypatch.setattr(values, "interval_points", wrong)
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.intervals(tmp_path, data["records"])
    try:
        with pytest.raises(AssertionError):
            check_interval_artists(captured[0], values.RECORDS[:4], evidence)
    finally:
        for fig in captured:
            plt.close(fig)
