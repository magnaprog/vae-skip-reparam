"""Check figure inputs, array dimensions and numerical reductions."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
spec = importlib.util.spec_from_file_location("figure_values", FIGURES / "values.py")
values = importlib.util.module_from_spec(spec)
spec.loader.exec_module(values)


def test_weighted_ecdf_ties_zero_weights_and_left_limit():
    points = np.array([2., -1., 2., 0.])
    weights = np.array([1., 2., 3., 0.])
    grid = np.array([-2., -1., 0., np.nextafter(2., -np.inf), 2., 3.])
    expected = np.array([0., 1 / 3, 1 / 3, 1 / 3, 1., 1.])
    np.testing.assert_array_equal(values.weighted_ecdf(points, weights, grid), expected)


@pytest.mark.parametrize("points,weights", [([], []), ([1], [-1]), ([1], [0]),
                                         ([np.nan], [1]), ([1], [np.inf]),
                                         ([1, 2], [1])])
def test_weighted_ecdf_rejects_invalid_measure(points, weights):
    with pytest.raises(ValueError, match="Invalid ECDF measure"):
        values.weighted_ecdf(points, weights, [0.])


def test_ecdf_averages_streams_without_pooling_samples_or_training_seeds():
    def stream(points):
        return {"features": np.column_stack((points, np.zeros((len(points), 2)))),
                "weights": np.ones(len(points))}

    # Unequal stream sizes distinguish stream means from pooled observations.
    fits = [[stream([0.]), stream([1., 1., 1.])],
            [stream([0., 0.]), stream([0.])]]
    grid = values.ecdf_grid(fits, 0)
    curves = values.ecdf_on_grid(fits, 0, grid)
    np.testing.assert_array_equal(grid, [0., 1.])
    np.testing.assert_array_equal(curves, [[.5, 1.], [1., 1.]])
    np.testing.assert_array_equal(values.ecdf_on_grid(fits, 0, [-1., 0., 1.]),
                                  [[0., .5, 1.], [0., 1., 1.]])


def test_record_order_ignores_mapping_insertion_order():
    records = dict.fromkeys(reversed(values.RECORDS))
    assert values.record_order(records) == list(values.RECORDS)


def test_spectral_stream_means_are_linear_and_keep_fits_separate():
    fits = [[{"power": np.array([1., 4.])}, {"power": np.array([9., 16.])}],
            [{"power": np.array([100., 400.])}, {"power": np.array([0., 0.])}]]
    np.testing.assert_array_equal(values.power_arrays(fits), [[5., 10.], [50., 200.]])


def test_refit_ratios_and_subtraction_keep_seed_score_and_stream_axes():
    record = {"models": {}}
    for index, seed in enumerate(values.SEEDS):
        original, refit = {"observation_variance": 2.}, {"variance_fit": {"pooled": 2. * (index + 1)}}
        original["scores"], refit["scores"] = {}, {}
        for score_index, score in enumerate(("full_window", "without_window_mean", "time_differences")):
            original["scores"][score] = {"weighted": {"values": [10., 20., 30.]}}
            delta = (index + 1) * (score_index + 1)
            refit["scores"][score] = {"weighted": {"values": [10. + delta, 20., 30. - delta]}}
        record["models"][f"no_skip_base/{seed}/original"] = original
        record["models"][f"no_skip_base/{seed}/variance_refit"] = refit
    ratios, deltas = values.refit_points(record, "no_skip_base")
    np.testing.assert_array_equal(ratios, [1., 2., 3.])
    expected = np.array([[[seed * score, 0., -seed * score] for score in (1, 2, 3)]
                         for seed in (1, 2, 3)])
    np.testing.assert_array_equal(deltas, expected)


def test_annual_context_retains_duplicates_source_order_and_leap_day(tmp_path):
    path = tmp_path / "44402t2016.txt"
    path.write_text("# header\n# units\n"
                    "2016 02 29 12 00 00 1 6000.001\n"
                    "2016 01 01 00 00 00 1 6000.002\n"
                    "2016 01 01 00 00 00 1 6000.003\n"
                    "2016 01 01 00 15 00 2 6000.004\n"
                    "2016 01 01 00 30 00 1 9999\n"
                    "2016 01 01 00 45 00 1 nan\n"
                    "2016 13 01 00 00 00 1 6000.005\n"
                    "2016 01 02 00 00 00 1 9000\n")
    result = values.read_observed_context(path, sha256(path.read_bytes()).hexdigest())
    assert result["year"] == 2016
    np.testing.assert_array_equal(result["days"], [60.5, 1., 1., 2.])
    np.testing.assert_array_equal(result["heights"], [6000.001, 6000.002, 6000.003, 9000.])
    path.write_text("changed annual bytes")
    with pytest.raises(ValueError, match="Public source hash differs"):
        values.read_observed_context(path, "0" * 64)


def copy_public_figures(destination, *, data=True):
    directory = destination / "figures"
    directory.mkdir(parents=True)
    files = ["render.py", "values.py"]
    if data:
        files += ["summary.json", "data_manifest.json", "data.npz", "comparisons.json"]
    for name in files:
        shutil.copyfile(FIGURES / name, directory / name)
    return directory


def run_python(root, arguments):
    return subprocess.run(
        [sys.executable, *arguments], cwd=root, capture_output=True, text=True,
        timeout=120, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                             MPLCONFIGDIR=str(root / "mpl-cache"),
                             OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                             CUDA_VISIBLE_DEVICES="-1"))


def test_help_works_without_plot_inputs_or_source_tree(tmp_path):
    directory = copy_public_figures(tmp_path, data=False)
    result = run_python(tmp_path, [str(directory / "render.py"), "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--data-dir" in result.stdout and "--output-dir" in result.stdout
    assert not (tmp_path / "runs").exists()


@pytest.fixture(scope="module")
def public_figures(tmp_path_factory):
    root = tmp_path_factory.mktemp("public-figures")
    directory = copy_public_figures(root)
    summary = json.loads((directory / "summary.json").read_text())
    for relative in summary["annual_sources"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root, directory


def test_ten_figure_cli_works_without_manuscript_or_experiment_modules(public_figures):
    root, directory = public_figures
    assert not (root / "paper").exists() and not (root / "src").exists()
    # Run the real default CLI; verify that it never imports scientific runners.
    script = """
import runpy, sys
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
assert not {'model', 'experiment', 'generative_evaluation', 'decoder_coordinates'} & set(sys.modules)
"""
    result = run_python(root, ["-c", script, str(directory / "render.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((root / "runs/figures").iterdir())
    assert {path.name for path in outputs} == {
        "dart_energy_score_differences.pdf", "dart_original_height_records.pdf",
        "dart_feature_distributions_original.pdf", "dart_feature_distributions_2018.pdf",
        "dart_spectra_original.pdf", "dart_spectra_2018.pdf",
        "dart_observation_variance.pdf", "dart_ppca_comparison.pdf",
        "dart_intervals_original.pdf", "dart_intervals_additional.pdf"}
    assert all(path.read_bytes().startswith(b"%PDF-") for path in outputs)
    assert not (root / "paper").exists()


@pytest.mark.parametrize("filename,error", [
    ("summary.json", "Plot summary hash differs"),
    ("data_manifest.json", "Plot manifest hash differs"),
    ("data.npz", "Plot data hash differs"),
])
def test_tampered_public_inputs_rejected_before_rendering(public_figures, tmp_path, filename, error):
    _, source = public_figures
    # Each negative fixture owns its copies; never edit repository inputs.
    for name in ("summary.json", "data_manifest.json", "data.npz"):
        shutil.copyfile(source / name, tmp_path / name)
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"\nchanged\n")
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import render
try:
    render.load_inputs(Path(sys.argv[2]))
except ValueError as error:
    assert str(error) == sys.argv[3], str(error)
else:
    raise AssertionError('Changed data accepted')
"""
    result = run_python(tmp_path, ["-c", script, str(source), str(tmp_path), error])
    assert result.returncode == 0, result.stdout + result.stderr


def test_changed_annual_record_rejected_with_public_summary(public_figures, tmp_path):
    _, directory = public_figures
    summary = json.loads((directory / "summary.json").read_text())
    relative, expected = next(iter(summary["annual_sources"].items()))
    path = tmp_path / Path(relative).name
    path.write_bytes((ROOT / relative).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Public source hash differs"):
        values.read_observed_context(path, expected)


def test_serialized_conditions_have_descriptive_names_and_fixed_order():
    expected = ("no_skip_base", "skip_base", "no_skip_high", "skip_high")
    assert values.CONDITIONS == expected
    assert values.SCORES == ("full_window", "without_window_mean", "time_differences")
    metadata = json.loads((FIGURES / "data_manifest.json").read_text())
    assert metadata["schema_version"] == 3
    for case in metadata["distribution_index"].values():
        assert tuple(role for role in case if role in expected) == expected
        assert set(case) == {"train", "test", "PPCA", *expected,
                             *(condition + " means" for condition in expected)}
    with np.load(FIGURES / "data.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(metadata["arrays"])
        assert {key.split("/")[1] for key in archive.files} == {
            "train", "test", "PPCA", *expected,
            *(condition + "_means" for condition in expected)}
    summary = json.loads((FIGURES / "summary.json").read_text())
    assert summary["schema_version"] == 3
    for record in summary["records"].values():
        assert tuple(dict.fromkeys(key.split("/")[0] for key in record["models"])) == expected
        assert set(record["contrasts"]) == {
            "skip_base-no_skip_base/original", "skip_high-no_skip_high/original"}


def test_unknown_condition_is_not_accepted():
    summary = json.loads((FIGURES / "summary.json").read_text())
    with pytest.raises(KeyError):
        values.refit_points(summary["records"][values.RECORDS[0]], "invalid_condition")
