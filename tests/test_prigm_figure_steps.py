"""ECDF paths retain exact step functions without smoothing their jumps."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "prigm_figure_renderer", Path(__file__).resolve().parents[1] /
    "analysis/prigm/figures/render_figures.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


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
