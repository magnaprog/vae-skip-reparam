"""Condition names remain readable without exposing the retained data codes."""
import importlib.util
import itertools
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest


FIGURES = Path(__file__).resolve().parents[1] / "analysis/prigm/figures"
spec = importlib.util.spec_from_file_location("prigm_label_renderer", FIGURES / "render_figures.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def assert_separate(boxes):
    for first, second in itertools.combinations(boxes, 2):
        assert not first.overlaps(second)


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
                names = [n for n in renderer.plot_values.RECORDS
                         if (summary["records"][n]["target"] == "six_UTC_bins_equal") == name.endswith("_2018")]
                assert fig.get_figwidth() == 5.5
                labels = list(fig.texts)
                for row, record_name in enumerate(names):
                    ax = fig.axes[row * columns]
                    record = summary["records"][record_name]
                    station, = [t for t in fig.texts if t.get_text() == f"{record['station']} / {record['year']}"]
                    assert station.get_rotation() == 90 and station.get_fontsize() == 8.5
                    box = station.get_window_extent(draw)
                    assert (box.y0 + box.y1) / 2 == pytest.approx((ax.bbox.y0 + ax.bbox.y1) / 2)
                    assert box.x1 < ax.yaxis.label.get_window_extent(draw).x0
                    assert ax.get_ylabel() == ("Bin power (m$^2$)" if spectra else "ECDF")
                    assert ax.yaxis.label.get_fontsize() == 9
                for ax in fig.axes:
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


@pytest.mark.parametrize("name", ["comparison", "variance"])
def test_comparison_and_refit_draw_descriptive_labels(monkeypatch, tmp_path, name):
    records = json.loads((FIGURES / "plot_summary.json").read_text())["records"]
    captured = []
    monkeypatch.setattr(renderer, "save", lambda fig, *args: captured.append(fig))
    with plt.rc_context(renderer.STYLE):
        getattr(renderer, name)(tmp_path, records)
        fig, = captured
        try:
            fig.canvas.draw()
            draw = fig.canvas.get_renderer()
            if name == "comparison":
                assert [text.get_text() for text in fig.texts] == [
                    "Energy score difference: skip minus no skip\n(dimensionless coordinates)"]
                legend, = fig.legends
                assert legend.get_title().get_text() == "Skip minus no skip"
                assert [text.get_text().split(" (")[0] for text in legend.get_texts()] == [
                    "Base variance", "High variance"]
            else:
                labels = fig.axes[0].get_yticklabels()
                assert [text.get_text() for text in labels] == [
                    "No skip, base variance", "Skip, base variance",
                    "No skip, high variance", "Skip, high variance"] * 7
                points = [[line for line in ax.lines if line.get_marker() in renderer.MARKERS.values()]
                          for ax in fig.axes]
                assert all(len(lines) == 84 for lines in points)
                for row, (record, arm) in enumerate((record, arm)
                        for record in renderer.plot_values.record_order(records) for arm in ("A", "C", "D", "B")):
                    ratios, deltas = renderer.plot_values.refit_points(records[record], arm)
                    for seed, offset in enumerate((-.2, 0, .2)):
                        index = row * 3 + seed
                        for panel, lines in enumerate(points):
                            point = lines[index]
                            expected = ratios[seed] if panel == 0 else deltas[seed, panel - 1].mean()
                            np.testing.assert_array_equal(point.get_xdata(), [expected])
                            np.testing.assert_array_equal(point.get_ydata(), [row + offset])
                            assert point.get_color() == renderer.COLORS[arm]
                            assert point.get_marker() == renderer.MARKERS[arm]
                            if panel:
                                values = deltas[seed, panel - 1]
                                segment, = fig.axes[panel].collections[index].get_segments()
                                np.testing.assert_array_equal(segment, [[values.min(), row + offset],
                                                                       [values.max(), row + offset]])
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
