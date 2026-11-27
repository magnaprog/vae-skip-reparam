"""Figure labels identify conditions and remain separate at print width."""
import importlib.util
import itertools
import json
from datetime import date
from pathlib import Path
import re

from matplotlib.collections import FillBetweenPolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
import pytest


FIGURES = Path(__file__).resolve().parents[1] / "figures"
spec = importlib.util.spec_from_file_location("label_renderer", FIGURES / "render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


DIAGNOSTIC_ROLES = (
    ("Training data", "#888888", ":", None), ("Test data", "#111111", "-", None),
    ("PPCA", "#555555", "-.", None), ("No skip, base variance", "#0077BB", "-", "^"),
    ("Skip, base variance", "#CC3311", "-", "s"), ("No skip, high variance", "#6A3D9A", "-", "^"),
    ("Skip, high variance", "#009988", "-", "s"))


def assert_separate(boxes):
    for first, second in itertools.combinations(boxes, 2):
        assert not first.overlaps(second)


def check_marker_clearance(ax, line, positions, min_horizontal_pt=0, min_vertical_pt=0):
    """Measure the actual outline against all four axes edges in print points."""
    px = ax.figure.dpi / 72
    style = line._marker  # get_marker() returns the symbol code without its join style.
    bounds = style.get_path().get_extents(style.get_transform())
    size = line.get_markersize()
    stroke = line.get_markeredgewidth() / 2
    x_stroke = bottom_stroke = top_stroke = stroke
    if line.get_marker() in ("^", "v") and style.get_joinstyle() == "miter":
        # The triangular marker has unit height and unit base. Its miter
        # joins extend farther than a half-stroke at the tips and corners.
        x_stroke = stroke * (1 + np.sqrt(5)) / 2
        if line.get_marker() == "^":
            top_stroke = stroke * np.sqrt(5)
        else:
            bottom_stroke = stroke * np.sqrt(5)
    x, y = positions.T
    assert np.min((x - ax.bbox.x0) / px + size * bounds.x0 - x_stroke) > min_horizontal_pt
    assert np.min((ax.bbox.x1 - x) / px - size * bounds.x1 - x_stroke) > min_horizontal_pt
    bottom = np.min(y / px + size * bounds.y0 - bottom_stroke)
    top = np.max(y / px + size * bounds.y1 + top_stroke)
    assert bottom - ax.bbox.y0 / px > min_vertical_pt
    assert ax.bbox.y1 / px - top > min_vertical_pt
    return bottom, top


def check_spectral_markers_and_bins(ax):
    frequency = np.arange(1, 97) / 2
    ticks = [.5, 1, 2, 6, 12, 24, 48]
    np.testing.assert_array_equal(ax.get_xticks(), ticks)
    assert [ax.xaxis.get_major_formatter()(x, i) for i, x in enumerate(ticks)] == [
        "0.5", "1", "2", "6", "12", "24", "48"]
    assert ax.get_xscale() == ax.get_yscale() == "log"
    symbols = 0
    for line in ax.lines:
        np.testing.assert_array_equal(line.get_xdata(), frequency)
        if line.get_marker() == "None":
            continue
        positions = line.get_transform().transform(line.get_xydata()[line.get_markevery()])
        check_marker_clearance(ax, line, positions)
        symbols += len(positions)
    assert symbols == 20


def check_seed_bands(ax, spectra):
    # Each panel has a shaded band for every role, including the reference curves.
    assert len(ax.collections) == 7
    for role, band in zip(renderer.ROLES, ax.collections, strict=True):
        assert isinstance(band, FillBetweenPolyCollection)
        assert to_hex(band.get_facecolors()[0]) == renderer.COLORS[role].lower()
        assert band.get_alpha() == (.12 if spectra else .13) and band.get_visible()
    for i, role in enumerate(renderer.CONDITION_ORDER):
        line, = [line for line in ax.lines if line.get_color() == renderer.COLORS[role]]
        assert line.get_markersize() == 3 and line.get_markeredgewidth() == .60
        if spectra:
            indices = ([0, 4, 13, 45, 88], [1, 6, 17, 52, 94],
                       [2, 8, 21, 60, 82], [3, 10, 27, 68, 77])[i]
        else:
            probabilities = np.array([.12, .36, .60, .84]) + .05 * i
            indices = np.searchsorted(line.get_ydata(), probabilities)
        np.testing.assert_array_equal(line.get_markevery(), indices)


def check_spectrum_pair(axes, case):
    frequency = np.arange(1, 97) / 2
    left, right = axes
    assert left.get_xlim() == right.get_xlim() and left.get_ylim() == right.get_ylim()
    for column, ax in enumerate(axes):
        for role, line, band in zip(renderer.ROLES, ax.lines, ax.collections, strict=True):
            neural = role in renderer.CONDITION_ORDER
            key = role + " means" if column == 1 and neural else role
            powers = np.array([np.mean([stream["power"] for stream in fit], axis=0)
                               for fit in case[key]])
            np.testing.assert_array_equal(line.get_xdata(), frequency)
            np.testing.assert_array_equal(line.get_ydata(), np.median(powers, axis=0))
            assert line.get_color() == renderer.COLORS[role]
            expected = FillBetweenPolyCollection("x", frequency, np.ma.masked_less_equal(powers.min(0), 0),
                                                 np.ma.masked_less_equal(powers.max(0), 0))
            for actual, reference in zip(band.get_paths(), expected.get_paths(), strict=True):
                np.testing.assert_array_equal(actual.vertices, reference.vertices)
                np.testing.assert_array_equal(actual.codes, reference.codes)


@pytest.mark.parametrize("height,spectra", [(7.03, False), (5.51, False),
                                            (7.07, True), (5.59, True)])
@pytest.mark.parametrize("dpi", [100, 144])
def test_condition_legends_fit_without_codes(height, spectra, dpi):
    with plt.rc_context(renderer.STYLE):
        fig = plt.figure(figsize=(5.5, height), dpi=dpi)
        try:
            renderer.diagnostic_legend(fig, spectra=spectra)
            fig.canvas.draw()
            labels = [text.get_text().replace("\n", " ") for text in fig.legends[1].get_texts()]
            assert labels == ["No skip, base variance", "Skip, base variance",
                              "No skip, high variance", "Skip, high variance"]
            handles = [h for legend in fig.legends for h in legend.legend_handles]
            assert len(handles) == len(DIAGNOSTIC_ROLES)
            for handle, (label, color, style, marker) in zip(handles, DIAGNOSTIC_ROLES, strict=True):
                assert handle.get_label().replace("\n", " ") == label
                assert handle.get_color() == color and handle.get_linestyle() == style
                assert handle.get_marker() == (marker or "None")
                if marker:
                    assert handle.get_markerfacecolor() == (color if "base" in label else "white")
            boxes = [legend.get_window_extent(fig.canvas.get_renderer()) for legend in fig.legends]
            for box in boxes:
                assert fig.bbox.contains(box.x0, box.y0)
                assert fig.bbox.contains(box.x1, box.y1)
            assert_separate(boxes)
        finally:
            plt.close(fig)


@pytest.mark.parametrize("dpi", [100, 144])
def test_diagnostic_station_labels_are_separate_from_axis_labels(monkeypatch, tmp_path, dpi):
    summary, reduced = renderer.load_inputs(FIGURES)
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, directory, name: captured.append((name, fig)))
    with plt.rc_context(renderer.STYLE):
        renderer.diagnostics(tmp_path, summary["records"], reduced)
        try:
            for name, fig in captured:
                fig.set_dpi(dpi)
                fig.canvas.draw()
                draw = fig.canvas.get_renderer()
                spectra = "spectra" in name
                columns = 2 if spectra else 3
                names = [n for n in renderer.figure_values.RECORDS
                         if (summary["records"][n]["target"] == "six_UTC_bins_equal") == name.endswith("_2018")]
                assert fig.get_figwidth() == 5.5
                labels = list(fig.texts)
                for row, record_name in enumerate(names):
                    ax = fig.axes[row * columns]
                    record = summary["records"][record_name]
                    if spectra:
                        pair = fig.axes[row * 2:row * 2 + 2]
                        check_spectrum_pair(pair, reduced[record_name])
                        if row == 0:
                            assert [a.get_title() for a in pair] == ["Neural samples with noise", "Decoder means"]
                    else:
                        for column, axis in enumerate(fig.axes[row * 3:row * 3 + 3]):
                            guides = [line for line in axis.lines if list(line.get_ydata()) == [.5, .5]]
                            assert len(guides) == int(column == 1)
                            assert len(axis.lines) == len(renderer.ROLES) + len(guides)
                    station, = [t for t in fig.texts if t.get_text() == f"{record['station']} / {record['year']}"]
                    assert station.get_rotation() == 90 and station.get_fontsize() == 8.5
                    box = station.get_window_extent(draw)
                    assert (box.y0 + box.y1) / 2 == pytest.approx((ax.bbox.y0 + ax.bbox.y1) / 2)
                    assert box.x1 < ax.yaxis.label.get_window_extent(draw).x0
                    assert ax.get_ylabel() == ("Bin power (m$^2$)" if spectra else "ECDF")
                    assert ax.yaxis.label.get_fontsize() == 9
                for ax in fig.axes:
                    check_seed_bands(ax, spectra)
                    for label, color, style, marker in DIAGNOSTIC_ROLES:
                        line, = [line for line in ax.lines if line.get_color() == color]
                        assert line.get_linestyle() == style and line.get_marker() == (marker or "None")
                        if marker:
                            assert line.get_markerfacecolor() == (color if "base" in label else "white")
                    if spectra:
                        check_spectral_markers_and_bins(ax)
                    if ax.get_ylabel():
                        labels.append(ax.yaxis.label)
                    ticks = [t.label1 for t in ax.yaxis._update_ticks() if t.label1.get_visible()]
                    assert all(t.get_fontsize() == 8.5 for t in ticks)
                    labels.extend(ticks)
                assert all(t.get_fontfamily() == ["DejaVu Serif"] for t in labels)
                boxes = [t.get_window_extent(draw) for t in labels]
                # Unframed legend padding is blank; compare the visible labels.
                boxes.extend(t.get_window_extent(draw) for legend in fig.legends for t in legend.get_texts())
                for box in boxes:
                    assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
                assert_separate(boxes)
        finally:
            for _, fig in captured:
                plt.close(fig)


def test_context_gap_matches_complete_utc_dates_between_observations(monkeypatch, tmp_path):
    summary = json.loads((FIGURES / "summary.json").read_text())
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        renderer.context(tmp_path, summary)
    fig, = captured
    try:
        assert len(fig.axes) == 4
        for panel in fig.axes:
            observations, = panel.collections
            np.testing.assert_array_equal(observations.get_sizes(), [.65])
            np.testing.assert_array_equal(observations.get_facecolors(), [renderer.matplotlib.colors.to_rgba("#008CFF")])
            assert observations.get_rasterized()
        ax = fig.axes[3]
        assert ax.get_title().replace(" ", "") == "52403/2020"
        assert ax.get_xscale() == "linear"
        points, = ax.collections
        days = np.unique(np.floor(points.get_offsets()[:, 0]).astype(int))
        start = (date(2020, 3, 28) - date(2020, 1, 1)).days + 1
        end = (date(2020, 8, 13) - date(2020, 1, 1)).days + 1
        assert start - 1 in days and end + 1 in days and not np.any((start <= days) & (days <= end))
        text, = ax.texts
        assert text.get_text() == "T=1 gap\nMar 28 to\nAug 13"
        fig.canvas.draw()
        box = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        assert start <= box.x0 < box.x1 <= end + 1
    finally:
        plt.close(fig)


def check_comparison_ranges(fig, records):
    for ax, score in zip(fig.axes, ("full_window", "without_window_mean", "time_differences"), strict=True):
        for container, pair, offset in zip(ax.containers, ("skip_base-no_skip_base", "skip_high-no_skip_high"),
                                          (-.15, .15), strict=True):
            stats = [records[n]["contrasts"][pair + "/original"]["scores"][score]["optimizer_seeds"]
                     for n in renderer.figure_values.RECORDS]
            mark, (left, right), (bars,) = container.lines
            medians = np.array([v["median"] for v in stats])
            minima = np.array([v["min"] for v in stats])
            maxima = np.array([v["max"] for v in stats])
            # Matplotlib reconstructs the two x coordinates from asymmetric errors.
            endpoints = (medians - (medians - minima), medians + (maxima - medians))
            rows = np.arange(len(stats)) + offset
            np.testing.assert_array_equal(mark.get_xydata(), np.column_stack((medians, rows)))
            for k, cap in enumerate((left, right)):
                expected = np.column_stack((endpoints[k], rows))
                np.testing.assert_array_equal(cap.get_xydata(), expected)
                np.testing.assert_array_equal(np.asarray(bars.get_segments())[:, k, :], expected)
                np.testing.assert_array_equal(cap.get_transform().transform(expected), ax.transData.transform(expected))
            np.testing.assert_array_equal(mark.get_transform().transform(mark.get_xydata()),
                                          ax.transData.transform(mark.get_xydata()))
            color, face = ("#CC3311", "#CC3311") if offset < 0 else ("#009988", "white")
            assert mark.get_marker() == "o" and mark.get_markeredgecolor() == color
            assert mark.get_markerfacecolor() == face
            assert mark.get_markersize() == 2.5 and mark.get_markeredgewidth() == .6
            assert left.get_markersize() == right.get_markersize() == 4
            assert left.get_marker() == right.get_marker() == "|"


def check_comparison_legend(legend):
    # The two keys must actually draw the median in the middle of a capped range.
    for key, color, face in zip(legend.findobj(DrawingArea), ("#CC3311", "#009988"), ("#CC3311", "white"), strict=True):
        rail, = key.findobj(LineCollection)
        segment, = rail.get_segments()
        lines = key.findobj(Line2D)
        median, = [p for p in lines if p.get_marker() == "o"]
        assert median.get_markeredgecolor() == color and median.get_markerfacecolor() == face
        caps = [p.get_xydata()[0] for p in lines if p.get_marker() == "|"]
        np.testing.assert_array_equal(caps, segment)
        np.testing.assert_array_equal(median.get_xydata()[0], segment.mean(axis=0))
        assert segment[0, 0] < segment[1, 0] and segment[0, 1] == segment[1, 1]


def check_refit_drawing(fig, records):
    names = renderer.figure_values.record_order(records)
    ratio, *score_axes = fig.axes
    assert len(score_axes) == 6 and ratio.get_yscale() == "log"
    assert ratio.get_ylabel() == "Variance\nratio"
    np.testing.assert_array_equal(ratio.get_yticks(), [.2, 1, 10])
    assert [t.get_text() for t in score_axes[-1].get_xticklabels()] == [
        f"{records[n]['station']}\n{records[n]['year']}" for n in names]
    np.testing.assert_array_equal(score_axes[-1].get_xticks(), np.arange(7) * 7.08 + (3.48 + 2) / 2)
    legend, = fig.legends
    for h, c in zip(legend.legend_handles, renderer.CONDITION_ORDER, strict=True):
        assert h.get_marker() == renderer.MARKERS[c] and h.get_color() == renderer.COLORS[c]
        assert h.get_markerfacecolor() == ("white" if c.endswith("_high") else renderer.COLORS[c])
        assert h.get_markeredgecolor() == renderer.COLORS[c] and h.get_markeredgewidth() == .7
        assert h.get_markersize() == 4
        assert h.get_label() == renderer.CONDITION_LABELS[c].replace(" variance", "")

    colors = [renderer.COLORS[c] for c in renderer.CONDITION_ORDER]
    def colored_lines(ax):
        return [line for line in ax.lines if line.get_color() in colors]

    ratios = colored_lines(ratio)
    panels = []
    for change, width in zip(score_axes[::2], score_axes[1::2], strict=True):
        assert change.get_yscale() == "linear" and width.get_yscale() == "log"
        assert [change.get_ylabel(), width.get_ylabel()] == ["Mean\nchange", "Range\nwidth"]
        assert change.get_position().y0 > width.get_position().y1
        for ax in (change, width):
            np.testing.assert_array_equal(ax.get_xlim(), ratio.get_xlim())
            ticks = [float(t.get_text().replace("−", "-")) for t in ax.get_yticklabels()]
            np.testing.assert_allclose(ticks, ax.get_yticks() * 1000, rtol=0, atol=1e-12)
        bars, whiskers = colored_lines(change), change.collections
        caps = [line for line in change.lines if line.get_marker() == "_"]
        widths = colored_lines(width)
        assert len(bars) == len(whiskers) == len(caps) == len(widths) == len(ratios) == 84
        panels.append((bars, whiskers, caps, widths))

    for group, (record, condition) in enumerate(itertools.product(names, renderer.CONDITION_ORDER)):
        values, streams = renderer.figure_values.refit_points(records[record], condition)
        symbol = "^" if condition.startswith("no_skip") else "s"
        color = renderer.COLORS[condition]
        is_high = condition.endswith("_high")
        face = "white" if is_high else color
        for seed in range(3):
            i = 3 * group + seed
            x = names.index(record) * 7.08 + condition.startswith("skip_") * 3.48 + seed
            point = ratios[i]
            assert point.get_markerfacecolor() == face and point.get_markeredgecolor() == color
            assert point.get_markersize() == 3.3 and point.get_markeredgewidth() == (.60 if is_high else .29)
            assert point._marker.get_joinstyle() == "miter"
            np.testing.assert_array_equal(point.get_xydata(), [[x, values[seed]]])
            assert point.get_marker() == symbol and point.get_color() == color
            for score, (bars, whiskers, caps, widths) in enumerate(panels):
                stream = streams[seed, score]
                low, high = stream.min(), stream.max()
                bar, whisker, cap, width = bars[i], whiskers[i], caps[i], widths[i]
                np.testing.assert_array_equal(bar.get_xydata(), [[x, 0], [x, stream.mean()]])
                assert bar.get_color() == color and bar.get_solid_capstyle() == "butt"
                endpoints = [[x, low], [x, high]]
                np.testing.assert_array_equal(whisker.get_segments(), [endpoints])
                np.testing.assert_array_equal(cap.get_xydata(), endpoints)
                assert cap.get_zorder() == whisker.get_zorder() > bar.get_zorder()
                assert to_hex(whisker.get_colors()[0]) == to_hex(cap.get_color()) == "#333333"
                offset = -.25 if condition.endswith("_base") else .25
                np.testing.assert_array_equal(width.get_xydata(), [[x + offset, high - low]])
                assert width.get_marker() == symbol and width.get_color() == color
                assert width.get_markerfacecolor() == face and width.get_markeredgecolor() == color
                assert width.get_markersize() == 2.55 and width.get_markeredgewidth() == (.40 if is_high else .14)
                assert width._marker.get_joinstyle() == "round"

    for ax in (ratio, *score_axes[1::2]):
        boxes = []
        for point in colored_lines(ax):
            position = point.get_transform().transform(point.get_xydata())
            check_marker_clearance(ax, point, position, .5, .5)
            # A quarter point on each outline enforces a half-point gap between neighbors.
            join = np.sqrt(5) if point.get_marker() == "^" and point._marker.get_joinstyle() == "miter" else 1
            padding = (.5 * point.get_markeredgewidth() * join + .25) * fig.dpi / 72
            boxes.append(point.get_window_extent(fig.canvas.get_renderer()).padded(padding))
        assert_separate(boxes)
    # Three regular-weight score titles separate the metrics. Each title belongs
    # to the mean-change axis, leaving the width row visibly paired with it.
    draw = fig.canvas.get_renderer()
    previous_axis = ratio
    headings = []
    expected = ("Full windows", "After mean subtraction", "Time increments")
    for label, change, width in zip(expected, score_axes[::2], score_axes[1::2], strict=True):
        title = change.title
        assert title.get_text() == label
        assert title.get_fontsize() == 9 and title.get_fontweight() == "normal"
        assert title.get_fontfamily() == ["DejaVu Serif"]
        box = title.get_window_extent(draw)
        assert (box.x0 + box.x1) / 2 == pytest.approx((change.bbox.x0 + change.bbox.x1) / 2)
        scale = 72 / fig.dpi
        anchor_y = title.get_transform().transform(title.get_position())[1]
        assert (anchor_y - change.bbox.y1) * scale == pytest.approx(4.5)
        own_title_gap = (box.y0 - change.bbox.y1) * scale
        previous_title_gap = (previous_axis.bbox.y0 - box.y1) * scale
        assert 0 < own_title_gap < previous_title_gap
        pair_gap = (change.bbox.y0 - width.bbox.y1) * scale
        assert 10 < pair_gap < (previous_axis.bbox.y0 - change.bbox.y1) * scale
        headings.append(title)
        previous_axis = width

    station_ticks = score_axes[-1].get_xticklabels()
    assert all(t.get_fontsize() == 8.5 for t in station_ticks)
    seed_note, = [t for t in fig.texts if "Seeds 42" in t.get_text()]
    assert seed_note.get_fontsize() == 8.5
    texts = [*fig.texts, *headings, *station_ticks, *legend.get_texts()]
    texts.extend(t for ax in fig.axes for t in [ax.yaxis.label, *ax.get_yticklabels()])
    assert_separate([t.get_window_extent(fig.canvas.get_renderer()) for t in texts if t.get_text()])


@pytest.mark.parametrize("name", ["comparison", "variance"])
def test_comparison_and_refit_draw_descriptive_labels(monkeypatch, tmp_path, name):
    records = json.loads((FIGURES / "summary.json").read_text())["records"]
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        getattr(renderer, name)(tmp_path, records)
        fig, = captured
        try:
            fig.canvas.draw()
            if name == "comparison":
                check_comparison_ranges(fig, records)
                assert any("Energy score difference" in text.get_text() and "skip minus no skip" in text.get_text()
                           for text in fig.texts)
                legend, = fig.legends
                assert [text.get_text().split(" (")[0] for text in legend.get_texts()] == [
                    "Base variance", "High variance"]
                check_comparison_legend(legend)
                for ax in fig.axes:
                    labels = [float(t.get_text().replace("−", "-")) for t in ax.get_xticklabels()]
                    np.testing.assert_allclose(labels, ax.get_xticks(), rtol=0, atol=1e-12)
                    assert ax.get_xscale() == "linear" and min(labels) < 0 < max(labels)
            else:
                check_refit_drawing(fig, records)
            for text in fig.findobj(match=renderer.matplotlib.text.Text):
                if text.get_visible():
                    assert not re.search(r"\b[ABCD]\b", text.get_text())
        finally:
            plt.close(fig)
