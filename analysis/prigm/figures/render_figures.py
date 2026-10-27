"""Render ten DART figures on CPU from the bundled summaries and plot inputs."""
from __future__ import annotations

import argparse
import calendar
from datetime import date, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import ScaledTranslation
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_values

EVIDENCE_SHA = "a73117bf43a049628455b72e27d55634c6e97d38d51a238fe679d55a0b4d0b6f"
SUMMARY_SHA = "91c0bf8adbca844f53aaf85e18db4cb36834e69c2d3e3676b5069fe3cf0c8696"
PLOT_MANIFEST_SHA = "5fbeaef79372cd88b870d19a613f38248ac5464362951df62d4daba1eb30a112"
TABLE_FIGURE_SHA = "e0d5c53b085630303435824a57217e896a97dbac53fd7ce1b1ee3903a1614d86"
FONT = "DejaVu Serif"
FONT_SHA = "107244956e9962b9e96faccdc551825e0ae0898ae13737133e1b921a2fd35ffa"
COLORS = {"A": "#0077BB", "C": "#CC3311", "B": "#009988", "D": "#6A3D9A",
          "train": "#888888", "test": "#111111", "PPCA": "#555555"}
CONDITION_ORDER = ("A", "C", "D", "B")
CONDITION_LABELS = {"A": "No skip, base variance", "C": "Skip, base variance",
                    "D": "No skip, high variance", "B": "Skip, high variance"}
MARKERS = {"A": "^", "B": "o", "C": "s", "D": "v"}
STYLES = {"train": ":", "test": "-", "PPCA": "-."}
ROLES = ("train", "test", *CONDITION_ORDER, "PPCA")
SCORES = ("joint", "centered", "increment")
STYLE = {"font.family": FONT, "mathtext.fontset": "stix", "font.size": 9,
         "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8.5,
         "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
         "pdf.fonttype": 42, "pdf.use14corefonts": False,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
         "axes.unicode_minus": False, "axes.formatter.useoffset": False,
         "path.simplify": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_table_inputs(directory):
    raw = (Path(directory) / "table_figure_data.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TABLE_FIGURE_SHA:
        raise ValueError("Table figure data hash differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate table figure key: " + key)
            result[key] = value
        return result

    return plot_values.validate_table_figure_data(
        json.loads(raw, object_pairs_hook=unique), EVIDENCE_SHA)


def load_inputs(directory):
    directory = Path(directory)
    summary_path = directory / "plot_summary.json"
    if sha(summary_path) != SUMMARY_SHA:
        raise ValueError("Plot summary hash differs")
    summary = json.loads(summary_path.read_text())
    if summary["schema_version"] != 1 or summary["scientific_manifest_sha256"] != EVIDENCE_SHA:
        raise ValueError("Plot summary belongs to different evidence")
    metadata_path = directory / "plot_data_manifest.json"
    if sha(metadata_path) != PLOT_MANIFEST_SHA:
        raise ValueError("Plot manifest hash differs")
    metadata = json.loads(metadata_path.read_text())
    if metadata["scientific_manifest_sha256"] != EVIDENCE_SHA:
        raise ValueError("Plot data belongs to different evidence")
    path = directory / "plot_data.npz"
    if sha(path) != metadata["cache_sha256"]:
        raise ValueError("Plot data hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(metadata["arrays"]):
            raise ValueError("Plot array inventory differs")
        arrays = {}
        for key, spec in metadata["arrays"].items():
            value = archive[key]
            if (list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]
                    or not np.isfinite(value).all()
                    or hashlib.sha256(value.tobytes(order="C")).hexdigest() != spec["sha256"]):
                raise ValueError("Invalid plot array: " + key)
            arrays[key] = value
    reduced = {}
    used = set()
    for name, case in metadata["distribution_index"].items():
        reduced[name] = {}
        for role, fits in case.items():
            result = []
            for fit in fits:
                streams = []
                for prefix in fit:
                    fields = ("power",) if role.endswith(" means") else ("features", "weights", "power")
                    stream = {field: arrays[prefix + "/" + field] for field in fields}
                    used.update(prefix + "/" + field for field in fields)
                    if stream["power"].shape != (96,) or np.any(stream["power"] < 0):
                        raise ValueError("Invalid spectral power")
                    if "features" in stream:
                        f, w = stream["features"], stream["weights"]
                        if (f.ndim != 2 or f.shape[1] != 3 or w.shape != (len(f),)
                                or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
                            raise ValueError("Invalid feature measure")
                    streams.append(stream)
                result.append(streams)
            reduced[name][role] = result
    if (used != set(arrays) or set(reduced) != set(summary["records"])
            or set(reduced) != set(plot_values.RECORDS)):
        raise ValueError("Incomplete plot coverage")
    for case in reduced.values():
        if set(case) != set(ROLES) | {a + " means" for a in "ABCD"}:
            raise ValueError("Incorrect distribution roles")
        for role, fits in case.items():
            expected = (1, 1) if role in ("train", "test") else (1, 3) if role == "PPCA" else (3, 3)
            if len(fits) != expected[0] or any(len(fit) != expected[1] for fit in fits):
                raise ValueError("Incorrect fit/stream coverage")
    return summary, reduced


def ecdf_arrays(case, feature):
    fits = [fit for role in ROLES for fit in case[role]]
    grid = plot_values.ecdf_grid(fits, feature)
    grid = np.r_[np.nextafter(grid[0], -np.inf), grid]
    return grid, {role: plot_values.ecdf_on_grid(case[role], feature, grid) for role in ROLES}


def step_support(grid, *curves):
    """Remove only redundant constant-segment vertices of a post-step path.

    No tolerance, rounding or smoothing is used. Keep the first/last point and
    every position where any supplied ordinate changes. The represented step
    functions on the full support are exactly unchanged.
    """
    keep = np.zeros(len(grid), dtype=bool)
    keep[0] = keep[-1] = True
    for curve in curves:
        keep[1:] |= np.diff(curve) != 0
    return keep


def legend_handles(roles=ROLES, multiline=False):
    labels = {"train": "Training data", "test": "Test data", "PPCA": "PPCA", **CONDITION_LABELS}
    if multiline:
        labels.update({key: value.replace(", ", ",\n") for key, value in CONDITION_LABELS.items()})
    return [Line2D([], [], color=COLORS[k], linestyle=STYLES.get(k, "-"),
                   marker=MARKERS.get(k), markersize=3, markerfacecolor="white",
                   label=labels.get(k, k), linewidth=1) for k in roles]



def diagnostic_legend(fig, spectra=False):
    # Keep measured/reference distributions separate from neural conditions.
    height = fig.get_figheight()
    fig.legend(handles=legend_handles(("train", "test", "PPCA")),
               loc="upper center", bbox_to_anchor=(.5, 1 - .06 / height), ncol=3,
               frameon=False, columnspacing=1.3, handlelength=2.7, borderaxespad=0)
    fig.legend(handles=legend_handles(CONDITION_ORDER, multiline=True),
               loc="upper center", bbox_to_anchor=(.5, 1 - .29 / height), ncol=4,
               frameon=False, columnspacing=1.1, handlelength=2, borderaxespad=0)
    if spectra:
        fig.legend(handles=[Line2D([], [], color=".35", linestyle="-", label="Neural samples with noise"),
                            Line2D([], [], color=".35", linestyle="--", label="Decoder means")],
                   loc="upper center", bbox_to_anchor=(.5, 1 - .70 / height), ncol=2,
                   frameon=False, columnspacing=1.1, handlelength=2.7, borderaxespad=0)

def save(fig, directory, name, digest):
    # Do not tight-crop: every output is 396 PDF points wide, matching the paper.
    fig.savefig(directory / (name + ".pdf"), dpi=250, metadata={
        "Title": name, "Author": "", "Creator": "DART retained-data figure renderer",
        "Subject": "Scientific evidence SHA256 " + digest, "CreationDate": None, "ModDate": None})
    plt.close(fig)


def comparison(directory, records):
    names = plot_values.record_order(records)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.75), sharey=True)
    for ax, score, title in zip(axes, SCORES, ("Full windows", "After mean\nsubtraction", "Time increments")):
        for pair, offset, color, marker, label in (
                ("C-A", -.12, COLORS["C"], "s", r"Base variance ($\sigma_0^2$)"),
                ("B-D", .12, COLORS["B"], "o", r"High variance ($25\sigma_0^2$)")):
            stats = [records[n]["contrasts"][pair + "/original"]["scores"][score]["optimizer_seeds"] for n in names]
            middle = np.array([s["median"] for s in stats])
            error = np.array([[s["median"] - s["min"] for s in stats], [s["max"] - s["median"] for s in stats]])
            ax.errorbar(middle, np.arange(7) + offset, xerr=error, fmt=marker,
                        color=color, markersize=3.8, linewidth=.9, capsize=2, label=label)
        ax.axvline(0, color=".55", linewidth=.65)
        ax.axhline(3.5, color=".7", linewidth=.6, linestyle=":")
        ax.grid(axis="x", color=".9", linewidth=.5)
        ax.set_title(title)

    axes[0].set_yticks(np.arange(7), [f"{records[n]['station']}/{records[n]['year']}" for n in names])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Skip minus no skip", title_fontsize=9,
               loc="upper center", bbox_to_anchor=(.58, 1.015), ncol=2, frameon=False)
    fig.text(.57, .02, "Energy score difference: skip minus no skip\n(dimensionless coordinates)",
             fontsize=9, ha="center")
    fig.tight_layout(rect=(0, .12, 1, .85), pad=.5, w_pad=.8)
    save(fig, directory, "reconstruction_generation", EVIDENCE_SHA)


def context(directory, summary):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6))
    records = summary["records"]
    sources = summary["annual_sources"]
    for ax, name in zip(axes.flat, plot_values.RECORDS[:4]):
        relative = next(path for path in sources if Path(path).name == name + ".txt")
        c = plot_values.read_observed_context(ROOT / relative, sources[relative])
        ax.scatter(c["days"], c["heights"], s=.35, linewidths=0, color="#008CFF", rasterized=True)
        ax.set_xlim(min(1, c["days"].min()), max(366 + calendar.isleap(c["year"]), c["days"].max()))
        ax.set_title(f"{records[name]['station']}/{records[name]['year']}")
        month_days = [(date(c["year"], m, 1) - date(c["year"], 1, 1)).days + 1
                      for m in (1, 4, 7, 10)]
        ax.set_xticks(month_days, ["Jan", "Apr", "Jul", "Oct"])
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("HEIGHT (m)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout(pad=.7, h_pad=1.2, w_pad=.8)
    save(fig, directory, "full_record_context", EVIDENCE_SHA)


def diagnostics(directory, records, reduced):
    for group, fresh in (("historical", False), ("fresh", True)):
        names = [n for n in plot_values.record_order(records) if (records[n]["target"] == "six_UTC_bins_equal") == fresh]
        fig, axes = plt.subplots(len(names), 3, figsize=(5.5, 1.52 * len(names) + .95), squeeze=False)
        for row, name in enumerate(names):
            for feature, ax in enumerate(axes[row]):
                grid, curves = ecdf_arrays(reduced[name], feature)
                for role in ROLES:
                    values = curves[role]
                    low, high, middle = values.min(0), values.max(0), np.median(values, axis=0)
                    envelope = step_support(grid, low, high)
                    line = step_support(grid, middle)
                    ax.fill_between(grid[envelope], low[envelope], high[envelope], step="post", color=COLORS[role], alpha=.13, linewidth=0)
                    # Marks identify curves without changing their exact step support.
                    fraction = {"A": .10, "C": .16, "D": .22, "B": .28}.get(role, .1)
                    marks = np.unique(((fraction + np.arange(4) * .2) * (int(line.sum()) - 1)).astype(int))
                    ax.step(grid[line], middle[line], where="post", color=COLORS[role],
                            linestyle=STYLES.get(role, "-"), linewidth=.85, marker=MARKERS.get(role),
                            markevery=marks, markersize=2.3, markeredgewidth=.5, markerfacecolor="white")
                ax.set_ylim(-.025, 1.025)
                ax.set_yticks([0, .5, 1])
                if feature:
                    ax.tick_params(labelleft=False)
                ax.locator_params(axis="x", nbins=3)
                if row == len(names)-1:
                    ax.set_xlabel(("Mean minus\ntraining mean (m)", "Standard deviation\nwithin a window (m)", "Time increment\nnorm (m)")[feature])
            axes[row, 0].set_ylabel("ECDF")
            axes[row, 0].text(0, 1.055, f"{records[name]['station']} / {records[name]['year']}",
                             transform=axes[row, 0].transAxes, ha="left", va="bottom", fontsize=9)
        diagnostic_legend(fig)
        fig.subplots_adjust(left=.115, right=.985, bottom=.105 if fresh else .08,
                            top=.8 if fresh else .837, hspace=.60, wspace=.24)
        for col, label in enumerate(("Window mean", "Standard deviation", "Time increment norm")):
            box = axes[0, col].get_position()
            fig.text((box.x0 + box.x1) / 2, .855 if fresh else .88,
                     label, ha="center", fontsize=9)
        save(fig, directory, "dart_ecdf_" + group, EVIDENCE_SHA)

        fig, axes = plt.subplots(len(names), 2, figsize=(5.5, 1.48 * len(names) + 1.15), squeeze=False)
        frequency = np.arange(1, 97) / 2
        for row, name in enumerate(names):
            for col, arms in enumerate((CONDITION_ORDER[:2], CONDITION_ORDER[2:])):
                ax = axes[row, col]
                for role in ("train", "test", "PPCA", *arms, *(a + " means" for a in arms)):
                    power = plot_values.power_arrays(reduced[name][role])
                    arm = role.split()[0]
                    positive = lambda x: np.ma.masked_less_equal(x, 0)
                    ax.fill_between(frequency, positive(power.min(0)), positive(power.max(0)), color=COLORS[arm], alpha=.12, linewidth=0)
                    ax.plot(frequency, positive(np.median(power, axis=0)), color=COLORS[arm],
                            linestyle="--" if role.endswith(" means") else STYLES.get(role, "-"),
                            linewidth=.85, marker=MARKERS.get(arm), markevery=([1, 5, 15, 47, 95] if arm in ("A", "B") else [0, 3, 11, 31, 79]),
                            markersize=2.3, markeredgewidth=.5, markerfacecolor="white")
                ax.set_xscale("log")
                ax.set_yscale("log", nonpositive="mask")
                ax.set_xlim(.5, 48)
                ax.set_xticks([.5, 1, 2, 6, 12, 24, 48], ["0.5", "1", "2", "6", "12", "24", "48"])
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_locator(LogLocator(numticks=4))
                ax.yaxis.set_minor_formatter(NullFormatter())
                if row == len(names)-1:
                    ax.set_xlabel("Frequency (cycles/day)")
                if row != len(names)-1:
                    ax.tick_params(labelbottom=False)
            limits = min(a.get_ylim()[0] for a in axes[row]), max(a.get_ylim()[1] for a in axes[row])
            for ax in axes[row]:
                ax.set_ylim(limits)
            axes[row, 0].set_ylabel("Bin power (m$^2$)")
            axes[row, 0].text(0, 1.055, f"{records[name]['station']} / {records[name]['year']}",
                             transform=axes[row, 0].transAxes, ha="left", va="bottom", fontsize=9)
        diagnostic_legend(fig, spectra=True)
        fig.subplots_adjust(left=.14, right=.985, bottom=.09 if fresh else .07,
                            top=.74 if fresh else .767, hspace=.32, wspace=.34)
        for col, label in enumerate(("Neural observation variance\n" + r"Base ($\sigma_0^2$)",
                                     "Neural observation variance\n" + r"High ($25\sigma_0^2$)")):
            box = axes[0, col].get_position()
            fig.text((box.x0 + box.x1) / 2, box.y1 + .046, label, ha="center", fontsize=9)
        save(fig, directory, "dart_spectra_" + group, EVIDENCE_SHA)


def variance(directory, records):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 7.15), sharey=True)
    labels = []
    for row, (name, arm) in enumerate((n, a) for n in plot_values.record_order(records) for a in CONDITION_ORDER):
        record = records[name]
        labels.append(CONDITION_LABELS[arm])
        if arm == CONDITION_ORDER[0]:
            axes[0].text(-2.75, row + 1.5, f"{record['station']}\n{record['year']}",
                         transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
        ratios, deltas = plot_values.refit_points(record, arm)
        for seed, offset in enumerate((-.2, 0, .2)):
            y = row + offset
            axes[0].plot(ratios[seed], y, marker=MARKERS[arm], color=COLORS[arm], markersize=3, linewidth=0)
            for col in range(3):
                values = deltas[seed, col]
                axes[col+1].hlines(y, values.min(), values.max(), color=COLORS[arm], linewidth=.65)
                axes[col+1].plot(values.mean(), y, marker=MARKERS[arm], color=COLORS[arm], markersize=3, linewidth=0)
    for col, ax in enumerate(axes):
        ax.axvline(1 if col == 0 else 0, color=".55", linewidth=.7, linestyle=":")
        for boundary in np.arange(3.5, 27, 4):
            ax.axhline(boundary, color=".6" if boundary == 15.5 else ".85", linewidth=.6,
                       linestyle=":" if boundary == 15.5 else "-")
        ax.set_title(("Observation\nvariance ratio", "Full\nwindows", "After mean\nsubtraction", "Time\nincrements")[col])
        if col:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value * 1e3:g}"))
            ax.locator_params(axis="x", nbins=3)
        if col == 0:
            ax.set_xlabel(r"$\hat v / v_{\mathrm{original}}$")
    axes[0].set_xscale("log")
    axes[0].set_xticks([.2, 1, 5], ["0.2", "1", "5"])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.42, right=.985, bottom=.09, top=.91, wspace=.35)
    fig.text(.5, .98, r"Original observation variance: base = $\sigma_0^2$; high = $25\sigma_0^2$",
             ha="center", fontsize=8.5)
    fig.text(.74, .012, "Energy score change ($10^{-3}$)\n(replacement minus original)",
             ha="center", fontsize=9)
    save(fig, directory, "dart_variance_refit", EVIDENCE_SHA)


def ppca_comparison(directory, records):
    """Show all 84 neural fits against PPCA at the same observation variance."""
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 7.15), sharey=True)
    labels = []
    for row, (name, arm) in enumerate((n, a) for n in plot_values.record_order(records)
                                    for a in CONDITION_ORDER):
        record = records[name]
        labels.append(CONDITION_LABELS[arm])
        if arm == CONDITION_ORDER[0]:
            axes[0].text(-1.95, row + 1.5, f"{record['station']}\n{record['year']}",
                         transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
        values = plot_values.ppca_comparison_points(record, arm)
        for seed, offset in enumerate((-.2, 0, .2)):
            for col, ax in enumerate(axes):
                ax.plot(values[seed, col], row + offset, marker=MARKERS[arm], color=COLORS[arm],
                        markersize=3, linewidth=0)
    for col, ax in enumerate(axes):
        ax.axvline(0, color=".55", linewidth=.7, linestyle=":")
        for boundary in np.arange(3.5, 27, 4):
            ax.axhline(boundary, color=".6" if boundary == 15.5 else ".85", linewidth=.6,
                       linestyle=":" if boundary == 15.5 else "-")
        ax.set_title(("Full\nwindows", "After mean\nsubtraction", "Time\nincrements")[col])
        ax.locator_params(axis="x", nbins=3)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.42, right=.985, bottom=.095, top=.94, wspace=.3)
    fig.text(.71, .018, "Energy score difference: neural minus PPCA\n(dimensionless coordinates)",
             ha="center", fontsize=9)
    save(fig, directory, "dart_ppca_comparison", EVIDENCE_SHA)


def intervals(directory, records):
    """Plot dated interval means, with counts and evaluation-weight shares."""
    seed_markers = ("o", "s", "^")
    contrasts = (("base", COLORS["C"]), ("high", COLORS["B"]))
    for group, names in (("original", plot_values.RECORDS[:4]),
                         ("additional", plot_values.RECORDS[4:])):
        original = group == "original"
        fig = plt.figure(figsize=(5.5, 7.15 if original else 6.25))
        grid = fig.add_gridspec(len(names), 2, left=.18, right=.985,
                               bottom=.095 if original else .12,
                               top=.86 if original else .83, hspace=.65, wspace=.12)
        for row, name in enumerate(names):
            record = records[name]
            points = {days: plot_values.interval_points(record, days) for days in (14, 28)}
            all_scores = np.concatenate([np.array([0.])] + [p[variance].ravel()
                                         for p in points.values() for variance, _ in contrasts])
            lower, upper = all_scores.min(), all_scores.max()
            score_padding = (upper - lower) * .18 or .1
            score_bounds = lower - score_padding, upper + score_padding
            all_starts = np.concatenate([p["start_unix"] for p in points.values()])
            dates = mdates.date2num(all_starts.astype("datetime64[s]"))
            padding = max(3., (dates.max() - dates.min()) * .085)
            bounds = dates.min() - padding, dates.max() + padding
            first_axis = None
            for col, days in enumerate((14, 28)):
                cell = grid[row, col].subgridspec(2, 1, height_ratios=(3.1, 1), hspace=.05)
                ax = fig.add_subplot(cell[0], sharey=first_axis)
                if first_axis is None:
                    first_axis = ax
                else:
                    ax.tick_params(labelleft=False)
                coverage = fig.add_subplot(cell[1], sharex=ax)
                values = points[days]
                starts = mdates.date2num(values["start_unix"].astype("datetime64[s]"))
                for variance, color in contrasts:
                    for seed, marker in enumerate(seed_markers):
                        # Separate glyphs in physical points, never in UTC dates or score values.
                        transform = ax.transData + ScaledTranslation(
                            (-4., 0., 4.)[seed] / 72, 0, fig.dpi_scale_trans)
                        ax.plot(starts, values[variance][:, seed], linestyle="none", marker=marker,
                                color=color, markerfacecolor=color if variance == "base" else "white",
                                markeredgewidth=.65, markersize=3.5, transform=transform)
                        if group == "additional":
                            for x, value in zip(starts, values[variance][:, seed], strict=True):
                                if value > 0:
                                    offset = 3 if value > .1 else (15 if variance == "base" else 1)
                                    ax.annotate(f"{value:+.3g}", (x, value), xytext=(-8, offset), xycoords=transform,
                                                textcoords="offset points", ha="right", color=color,
                                                fontsize=8.5, arrowprops={"arrowstyle": "-", "color": color,
                                                                         "linewidth": .45})
                ax.axhline(0, color=".5", linewidth=.6, linestyle=":", zorder=0)
                # Display-offset transforms do not reliably participate in autoscaling.
                # Bind both columns to the complete untransformed values, including zero.
                ax.set_ylim(score_bounds)
                ax.locator_params(axis="y", nbins=3)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.set_xlim(bounds)
                if col == 0:
                    ax.set_title(f"{record['station']} / {record['year']}", loc="left", pad=7, fontsize=8.5)
                if row == 0:
                    box = ax.get_position()
                    fig.text((box.x0 + box.x1) / 2, .915 if original else .90,
                             f"Intervals of {days} days", ha="center", fontsize=9)
                coverage.set_ylim(0, 1)
                coverage.set_yticks([])
                coverage.set_xticks(starts)
                coverage.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%d", tz=timezone.utc))
                coverage.tick_params(axis="x", labelsize=8.5, length=2, pad=2)
                for spine in coverage.spines.values():
                    spine.set_visible(False)
                for x, count, mass in zip(starts, values["windows"], values["mass"], strict=True):
                    coverage.text(x, .75, str(int(count)), ha="center", va="center", fontsize=8.5,
                                  fontweight="bold" if count == 1 else "normal")
                    weight = f"{float(f'{100 * mass:.3g}'):.2f}".rstrip("0").rstrip(".")
                    coverage.text(x, .15, weight, ha="center", va="center", fontsize=8.5)
                if col == 0:
                    for y, label in ((.75, "Windows"), (.15, "Weight (%)")):
                        coverage.text(-.03, y, label, transform=coverage.transAxes,
                                      ha="right", va="center", fontsize=8.5)
        handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=3.5,
                          markerfacecolor=color if variance == "base" else "white",
                          label=variance.capitalize() + " variance") for variance, color in contrasts]
        handles += [Line2D([], [], color=".25", marker=marker, linestyle="none", markersize=3.5,
                           label=str(seed)) for seed, marker in zip(plot_values.SEEDS, seed_markers, strict=True)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=5,
                   title="Observation variance / training seed", frameon=False,
                   columnspacing=1.0, handletextpad=.45)
        fig.text(.58, .025, "Interval start (UTC)", ha="center", fontsize=9)
        fig.text(.022, .48, "Energy score difference: skip minus no skip", rotation=90,
                 ha="center", va="center", fontsize=9)
        save(fig, directory, "dart_intervals_" + group, EVIDENCE_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures")
    args = parser.parse_args()
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    if sha(font_path) != FONT_SHA:
        raise ValueError("Expected Matplotlib's bundled DejaVu Serif 2.35; see README.md")
    summary, reduced = load_inputs(args.data_dir)
    tables = load_table_inputs(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        comparison(args.output_dir, summary["records"])
        context(args.output_dir, summary)
        diagnostics(args.output_dir, summary["records"], reduced)
        variance(args.output_dir, summary["records"])
        ppca_comparison(args.output_dir, tables["records"])
        intervals(args.output_dir, tables["records"])


if __name__ == "__main__":
    main()
