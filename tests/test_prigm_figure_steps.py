"""Renderer imports authenticate sources; ECDF paths retain exact step functions."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "prigm_figure_renderer", Path(__file__).resolve().parents[1] /
    "analysis/prigm/figures/render_figures.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.mark.parametrize("changed", [None, "src/build_prigm_v2_paper.py", "src/prigm_v2_metrics.py"])
def test_renderer_import_authenticates_sources_before_execution(tmp_path, changed):
    root = Path(__file__).resolve().parents[1]
    manifest = "paper/prigm/paper_evidence_manifest.json"
    sources = json.loads((root / manifest).read_text())["input_index"]["source_sha256"]
    files = [manifest, *sources, "analysis/prigm/figures/prepare_figure_data.py",
             "analysis/prigm/figures/render_figures.py"]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    if changed:
        with (tmp_path / changed).open("ab") as handle:
            handle.write(b"\nraise RuntimeError('Modified source executed before authentication')\n")
    # A fresh interpreter prevents the parent test process's cached modules from
    # masking either an import of changed code or an incorrect repository root.
    script = """
import runpy
import sys
from pathlib import Path
try:
    module = runpy.run_path(sys.argv[1])
except ValueError as error:
    assert sys.argv[2] and str(error) == 'Source hash differs: ' + sys.argv[2], str(error)
    assert 'build_prigm_v2_paper' not in sys.modules
else:
    assert not sys.argv[2], 'Modified source was accepted'
    assert Path(module['frozen'].__file__).resolve() == Path.cwd() / 'src/build_prigm_v2_paper.py'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / files[-1]), changed or ""],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def evaluate_post_step(x, y, queries):
    index = np.searchsorted(x, queries, side="right") - 1
    return y[np.clip(index, 0, len(y) - 1)]


def assert_same_functions(x, curves):
    keep = renderer.step_support(x, *curves)
    assert keep[0] and keep[-1]
    # Probe both sides of every breakpoint, the breakpoints themselves and
    # interior plateaus. Adjacent representable endpoints remain meaningful.
    queries = np.concatenate((x, np.nextafter(x, -np.inf),
                              np.nextafter(x, np.inf), (x[:-1] + x[1:]) / 2))
    queries = queries[(queries >= x[0]) & (queries <= x[-1])]
    for curve in curves:
        np.testing.assert_array_equal(evaluate_post_step(x, curve, queries),
                                      evaluate_post_step(x[keep], curve[keep], queries))
    return keep


def test_exact_step_preserves_zero_left_limit_ties_and_tiny_jump():
    x = np.r_[np.nextafter(-2., -np.inf), -2., -1., 0., 1., 2., 3., 4.]
    y = np.array([0, .25, .25, .25, np.nextafter(.25, np.inf), .75, .75, 1.])
    keep = assert_same_functions(x, [y])
    assert not keep[2] and not keep[3]
    assert keep[1] and keep[4] and keep[5]


def test_envelope_keeps_changes_in_either_boundary():
    x = np.arange(9, dtype=float)
    low = np.array([0, 0, .1, .1, .1, .4, .4, .8, 1.])
    high = np.array([0, .1, .1, .3, .3, .7, .7, 1., 1.])
    keep = assert_same_functions(x, [low, high])
    assert keep[1] and keep[2] and keep[3]
    assert not keep[4] and not keep[6]


def test_constant_curve_keeps_both_support_endpoints():
    x = np.linspace(-100, 200, 1001)
    y = np.ones_like(x)
    keep = assert_same_functions(x, [y])
    np.testing.assert_array_equal(np.flatnonzero(keep), [0, 1000])
