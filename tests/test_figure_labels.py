"""Figure labels identify conditions and remain separate at print width."""
import importlib.util
import itertools
import json
from datetime import date
from pathlib import Path
import re

from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pytest


FIGURES = Path(__file__).resolve().parents[1] / "figures"
spec = importlib.util.spec_from_file_location("label_renderer", FIGURES / "render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


DIAGNOSTIC_ROLES = (
    ("Training data", "#888888", ":", None), ("Test data", "#111111", "-", None),
    ("PPCA", "#555555", "-.", None), ("No skip, base variance", "#0077BB", "-", "^"),
    ("Skip, base variance", "#CC3311", "-", "s"), ("No skip, high variance", "#332288", "-", "v"),
    ("Skip, high variance", "#009988", "-", "o"))


def assert_separate(boxes):
    for first, second in itertools.combinations(boxes, 2):
        assert not first.overlaps(second)


def check_marker_clearance(ax, line, positions, min_horizontal_pt=0, min_vertical_pt=0):
    """Measure the actual outline against all four axes edges in print points."""
    px = ax.figure.dpi / 72
    style = MarkerStyle(line.get_marker())
    bounds = style.get_path().get_extents(style.get_transform())
    size = line.get_markersize()
    stroke = line.get_markeredgewidth() / 2
    x_stroke = bottom_stroke = top_stroke = stroke
    if line.get_marker() in ("^", "v"):
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


def check_printed_seed_ranges(ax, marks_per_role):
    stems = [c for c in ax.collections if isinstance(c, LineCollection)]
    caps = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(stems) == len(caps) == 4
    for (_, color, _, _), stem, cap in zip(DIAGNOSTIC_ROLES[3:], stems, caps, strict=True):
        line, = [line for line in ax.lines if line.get_color() == color]
        assert cap.get_label() == "_seed_range_caps"
        assert to_hex(stem.get_colors()[0]) == to_hex(cap.get_edgecolors()[0]) == color.lower()
        assert stem.get_visible() and cap.get_visible() and cap.get_clip_on()
        assert stem.get_zorder() < cap.get_zorder() < line.get_zorder()
        assert line.get_markersize() == 3 and line.get_markeredgewidth() == .60
        np.testing.assert_array_equal(stem.get_linewidths(), [.80])
        np.testing.assert_array_equal(cap.get_linewidths(), [.65])
        segments = np.asarray(stem.get_segments())
        assert segments.shape == (marks_per_role, 2, 2)
        expected = np.concatenate((segments[:, 0], segments[:, 1]))
        np.testing.assert_array_equal(cap.get_offsets(), expected)
        np.testing.assert_array_equal(cap.get_offset_transform().transform(expected),
                                      ax.transData.transform(expected))
        path, = cap.get_paths()
        # A horizontal stroke in physical points, anchored on the data endpoint.
        vertices = path.vertices
        np.testing.assert_allclose(vertices[:, 1], 0, atol=1e-15)
        np.testing.assert_array_equal(np.sort(vertices[:, 0]), [-.5, .5])
        width, = np.sqrt(cap.get_sizes())
        assert width == 4.15 and width > line.get_markersize() + line.get_markeredgewidth()
        px = ax.figure.dpi / 72
        x, y = ax.transData.transform(expected).T
        assert min(x) - width / 2 * px > ax.bbox.x0 and max(x) + width / 2 * px < ax.bbox.x1
        half_stroke = cap.get_linewidths()[0] / 2 * px
        assert min(y) - half_stroke > ax.bbox.y0 and max(y) + half_stroke < ax.bbox.y1


def check_spectrum_pair(axes, case):
    frequency = np.arange(1, 97) / 2
    left, right = axes
    assert left.get_xlim() == right.get_xlim() and left.get_ylim() == right.get_ylim()
    for column, ax in enumerate(axes):
        bars = [item for item in ax.collections if isinstance(item, LineCollection)]
        assert len(bars) == 4
        for role, line in zip(("train", "test", *renderer.CONDITION_ORDER, "PPCA"), ax.lines, strict=True):
            neural = role in renderer.CONDITION_ORDER
            key = role + " means" if column == 1 and neural else role
            powers = np.array([np.mean([stream["power"] for stream in fit], axis=0)
                               for fit in case[key]])
            np.testing.assert_array_equal(line.get_xdata(), frequency)
            np.testing.assert_array_equal(line.get_ydata(), np.median(powers, axis=0))
            assert line.get_color() == renderer.COLORS[role]
            if neural:
                bar = bars[renderer.CONDITION_ORDER.index(role)]
                for i, segment in zip(line.get_markevery(), bar.get_segments(), strict=True):
                    np.testing.assert_array_equal(segment, [[frequency[i], powers[:, i].min()],
                                                            [frequency[i], powers[:, i].max()]])


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
                    station, = [t for t in fig.texts if t.get_text() == f"{record['station']} / {record['year']}"]
                    assert station.get_rotation() == 90 and station.get_fontsize() == 8.5
                    box = station.get_window_extent(draw)
                    assert (box.y0 + box.y1) / 2 == pytest.approx((ax.bbox.y0 + ax.bbox.y1) / 2)
                    assert box.x1 < ax.yaxis.label.get_window_extent(draw).x0
                    assert ax.get_ylabel() == ("Bin power (m$^2$)" if spectra else "ECDF")
                    assert ax.yaxis.label.get_fontsize() == 9
                for ax in fig.axes:
                    check_printed_seed_ranges(ax, 5 if spectra else 4)
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
            np.testing.assert_array_equal(observations.get_facecolors(), [renderer.matplotlib.colors.to_rgba("#0072B2")])
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
                                          (-.115, .315), strict=True):
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
            xy = mark.get_xydata()
            shift = (mark.get_transform().transform(xy) - ax.transData.transform(xy)) * 72 / fig.dpi
            np.testing.assert_allclose(shift, np.tile([0, 3.35], (len(rows), 1)), rtol=0, atol=1e-10)
            assert 3.35 > (mark.get_markersize() + mark.get_markeredgewidth() +
                           left.get_markersize() + left.get_markeredgewidth()) / 2


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
            draw = fig.canvas.get_renderer()
            if name == "comparison":
                check_comparison_ranges(fig, records)
                assert any("Energy score difference" in text.get_text() and "skip minus no skip" in text.get_text()
                           for text in fig.texts)
                legend, = fig.legends
                assert [text.get_text().split(" (")[0] for text in legend.get_texts()] == [
                    "Base variance", "High variance"]
                for ax in fig.axes:
                    labels = [float(t.get_text().replace("−", "-")) for t in ax.get_xticklabels()]
                    np.testing.assert_allclose(labels, ax.get_xticks(), rtol=0, atol=1e-12)
                    assert ax.get_xscale() == "linear" and min(labels) < 0 < max(labels)
            else:
                labels = fig.axes[0].get_yticklabels()
                names = renderer.figure_values.record_order(records)
                assert [text.get_text() for text in labels] == [
                    f"{records[n]['station']}\n{records[n]['year']}" for n in names]
                np.testing.assert_array_equal(fig.axes[0].get_yticks(), np.arange(7) * 4 + 1.5)
                legend, = fig.legends
                assert [text.get_text() for text in legend.get_texts()] == [
                    "No skip, base", "Skip, base", "No skip, high", "Skip, high"]
                for condition, handle in zip(("no_skip_base", "skip_base", "no_skip_high", "skip_high"),
                                             legend.legend_handles, strict=True):
                    assert handle.get_marker() == renderer.MARKERS[condition]
                    assert handle.get_color() == renderer.COLORS[condition]
                assert fig.axes[0].get_xscale() == "log"
                assert len(fig.axes) == 7
                changes, ranges = fig.axes[1:4], fig.axes[4:]
                for col, (change, width) in enumerate(zip(changes, ranges, strict=True)):
                    assert change.get_title() == "Change" and width.get_title() == "Range"
                    assert change.get_xscale() == "linear" and width.get_xscale() == "log"
                    assert change.get_position().x1 < width.get_position().x0
                    if col < 2:
                        assert width.get_position().x1 < changes[col+1].get_position().x0
                    assert width.get_xlim() == (4.5e-6, 6e-3)
                    np.testing.assert_array_equal(width.get_xticks(), [1e-5, 1e-3])
                    for ax in (change, width):
                        ticks = [float(t.get_text().replace("−", "-")) for t in ax.get_xticklabels()]
                        np.testing.assert_allclose(ticks, ax.get_xticks() * 1000, rtol=0, atol=1e-12)
                        assert ax.get_ylim() == fig.axes[0].get_ylim()
                points = [[line for line in ax.lines if line.get_marker() not in ("None", None)] for ax in fig.axes]
                assert all(len(lines) == 84 for lines in points)
                for panel, (ax, lines) in enumerate(zip(fig.axes, points, strict=True)):
                    row_extents = []
                    for line in lines:
                        positions = line.get_transform().transform(line.get_xydata())
                        extent = check_marker_clearance(ax, line, positions, .5 if panel == 0 or panel >= 4 else 0,
                                                        .5 if panel >= 4 else 0)
                        row_extents.append(extent)
                    if panel >= 4:
                        # Rows descend on the page. Even identical widths must leave visible space.
                        assert all(upper[0] - lower[1] > .5 for upper, lower in itertools.pairwise(row_extents))
                np.testing.assert_array_equal(fig.axes[0].get_xticks(), [.2, 1, 5])
                for row, (record, condition) in enumerate(itertools.product(names, renderer.CONDITION_ORDER)):
                    ratios, deltas = renderer.figure_values.refit_points(records[record], condition)
                    for seed, offset in enumerate((-.29, 0, .29)):
                        index, y = row * 3 + seed, row + offset
                        fits = deltas[seed]
                        expected = [ratios[seed], *fits.mean(axis=1), *np.ptp(fits, axis=1)]
                        symbol = {"no_skip_base": "^", "skip_base": "s", "no_skip_high": "v", "skip_high": "o"}[condition]
                        symbols = [symbol, "|", "|", "|", symbol, symbol, symbol]
                        for lines, x, marker in zip(points, expected, symbols, strict=True):
                            point = lines[index]
                            np.testing.assert_array_equal(point.get_xydata(), [[x, y]])
                            assert point.get_color() == renderer.COLORS[condition] and point.get_marker() == marker
                        for change, width, stream in zip(changes, ranges, fits, strict=True):
                            low, high = stream.min(), stream.max()
                            segment, = change.collections[index].get_segments()
                            np.testing.assert_array_equal(segment, [[low, y], [high, y]])
                            assert 0 < width.get_xlim()[0] < high - low < width.get_xlim()[1]
                        for lines in points[4:]:
                            point = lines[index]
                            assert point.get_markersize() == {"^": 3.35, "s": 2.6, "v": 3.35, "o": 2.95}[symbol]
                            assert point.get_markeredgewidth() == .15
                            assert point.get_markerfacecolor() == point.get_markeredgecolor() == renderer.COLORS[condition]
                assert_separate([text.get_window_extent(draw) for text in labels + fig.axes[0].texts])
                footer = fig.texts[-1].get_window_extent(draw)
                for ax in fig.axes[1:]:
                    for tick in ax.xaxis._update_ticks():
                        assert not tick.label1.get_window_extent(draw).overlaps(footer)
            for text in fig.findobj(match=renderer.matplotlib.text.Text):
                if text.get_visible():
                    assert not re.search(r"\b[ABCD]\b", text.get_text())
        finally:
            plt.close(fig)
