"""CPU metrics tests using NumPy and SciPy: fixed-noise PPCA, increments and IW controls."""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prigm_metrics import energy_score, gaussian_log_prob, importance_statistics
from prigm_v2_metrics import (INCREMENT_LAGS, NOISE_GRID, centered_energy_score,
                              fixed_noise_ppca, increment_energy_score, increment_features,
                              linear_model_iw_control, noise_matched_diagonal,
                              posterior_score_components, refit_observation_variance,
                              select_noise_variance, unconditional_score_streams,
                              weighted_moments, window_spectrum_statistics)


@pytest.fixture(scope="module")
def low_rank_data():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((192, 8)) * 2
    X = rng.standard_normal((4000, 8)) @ A.T + 0.25 * rng.standard_normal((4000, 192))
    return X


def test_weighted_moments_match_unweighted(low_rank_data):
    mean, cov = weighted_moments(low_rank_data, np.ones(len(low_rank_data)))
    np.testing.assert_allclose(mean, low_rank_data.mean(0), atol=1e-10)
    np.testing.assert_allclose(cov, np.cov(low_rank_data.T, bias=True), atol=1e-8)


def test_fixed_noise_ppca_exact_posterior_iw(low_rank_data):
    c = np.cov(low_rank_data[:3000].T)
    control = linear_model_iw_control(c, 0.0625, low_rank_data[3000:3010], 256,
                                      seed=3, proposal_mode="exact")
    # With the exact posterior, weights are constant, so the IW score is a
    # deterministic evaluation of the control model's own log density.
    ess = control["ess"]
    assert np.allclose(ess, 256.0, rtol=1e-9)
    np.testing.assert_allclose(control["error"], 0.0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(control["iw_log_score"], control["exact_log_marginal"],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(control["log_weights"],
                               np.broadcast_to(control["exact_log_marginal"][:, None],
                                               control["log_weights"].shape),
                               rtol=0, atol=1e-8)


def test_fixed_noise_ppca_loading_rank_at_high_and_generating_noise(low_rank_data):
    mean, cov = weighted_moments(low_rank_data, np.ones(len(low_rank_data)))
    # A noise variance far above every eigenvalue must zero all loadings and
    # leave a pure v*I observation covariance (loading rank zero).
    model_hi = fixed_noise_ppca(mean, cov, variance=1e4, rank=16)
    assert model_hi["effective_rank"] == 0
    evals = np.linalg.eigvalsh(model_hi["covariance"])
    np.testing.assert_allclose(evals, 1e4, rtol=1e-9)
    # At the fixture's generating noise variance (the grid's upper endpoint),
    # the fitted loading rank is between eight and the requested cap.
    model_lo = fixed_noise_ppca(mean, cov, variance=0.0625, rank=16)
    assert 8 <= model_lo["effective_rank"] <= 16


def test_noise_selection_uses_highest_recorded_validation_score(low_rank_data):
    x, v = low_rank_data[:2000], low_rank_data[2000:2400]
    result = select_noise_variance(x, np.ones(2000), v, np.ones(400))
    assert len(result["grid_results"]) == len(NOISE_GRID)
    assert result["selection"]["variance"] in NOISE_GRID
    assert result["selection"]["weighted_validation_log_density"] == max(
        r["weighted_validation_log_density"] for r in result["grid_results"])


def test_noise_selection_breaks_an_actual_exact_tie_toward_larger_variance():
    # Both fixed-noise fits reconstruct covariance one exactly in binary arithmetic.
    train = np.array([[-1.0], [1.0]])
    validation = np.array([[0.0], [-0.5], [0.5]])
    result = select_noise_variance(train, np.ones(2), validation, np.ones(3),
                                   grid=(0.75, 0.9375), rank=1)
    first, second = result["grid_results"]
    assert first["weighted_validation_log_density"] == second["weighted_validation_log_density"]
    assert result["selection"]["variance"] == 0.9375


def test_increment_feature_width_and_scaling():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 192))
    features = increment_features(x)
    expected = sum(192 - lag for lag in INCREMENT_LAGS)
    assert features.shape == (5, expected)
    # lag-1 block equals diff/sqrt(191)
    np.testing.assert_allclose(features[:, :191], np.diff(x, axis=1) / np.sqrt(191),
                               atol=1e-12)


def test_noise_matched_diagonal_floors_at_v(low_rank_data):
    # max(weighted variance, v): with v far above every empirical variance the
    # diagonal collapses to v; with tiny v it equals the empirical variance.
    v_hi = 1e4
    baseline = noise_matched_diagonal(low_rank_data[:100], np.ones(100), v_hi)
    np.testing.assert_allclose(np.diag(baseline["covariance"]), v_hi, rtol=1e-12)
    _, cov = weighted_moments(low_rank_data[:100], np.ones(100))
    baseline_lo = noise_matched_diagonal(low_rank_data[:100], np.ones(100), 1e-8)
    np.testing.assert_allclose(np.diag(baseline_lo["covariance"]),
                               np.diag(cov), rtol=1e-9)


def test_inflated_and_shifted_proposals_degrade(low_rank_data):
    c = np.cov(low_rank_data[:3000].T)
    x = low_rank_data[3000:3010]
    exact = linear_model_iw_control(c, 0.0625, x, 256, seed=3, proposal_mode="exact")
    inflated = linear_model_iw_control(c, 0.0625, x, 256, seed=3, proposal_mode="inflated")
    shifted = linear_model_iw_control(c, 0.0625, x, 256, seed=3, proposal_mode="shifted")
    assert np.median(inflated["ess"]) < np.median(exact["ess"])
    assert np.median(shifted["ess"]) < np.median(inflated["ess"])
    # This comparison is specific to these fixed proposals, data and draws.
    # A single IW realization need not be below the true log density.
    assert np.median(np.abs(shifted["error"])) > np.median(np.abs(inflated["error"]))


@pytest.mark.parametrize("samples", [64, 256, 1024])
@pytest.mark.parametrize("noise_variance", [0.25, 1.0, 8.0])
@pytest.mark.parametrize("seed", [7, 11, 19])
def test_exact_weights_equal_fitted_model_density(samples, noise_variance, seed):
    from scipy.stats import multivariate_normal

    x = np.array([[0.0, 0.0], [1.0, -2.0], [-3.0, 0.5]])
    result = linear_model_iw_control(np.diag([4.0, 1.0]), noise_variance, x,
                                     samples, seed=seed, rank=1)
    fitted_covariance = np.diag([max(4.0, noise_variance), noise_variance])
    expected = multivariate_normal.logpdf(x, cov=fitted_covariance)
    np.testing.assert_allclose(result["exact_log_marginal"], expected,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["iw_log_score"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["log_weights"],
                               np.broadcast_to(expected[:, None], (len(x), samples)),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["ess"], samples, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["exact", "inflated", "shifted"])
def test_proposal_log_density_matches_scipy(mode):
    from scipy.special import logsumexp
    from scipy.stats import norm

    samples, seed = 64, 31
    x = np.array([[1.0, -2.0], [-3.0, 0.5]])
    result = linear_model_iw_control(np.diag([4.0, 1.0]), 1.0, x, samples,
                                     seed=seed, rank=1, proposal_mode=mode)
    rng = np.random.default_rng(seed)
    rng.standard_normal((samples, 1))  # Match the control's initial prior draws.
    expected_weights = []
    for xi in x:
        q_mean = np.sqrt(3.0) * xi[0] / 4.0 + (1.0 if mode == "shifted" else 0.0)
        q_std = 1.0 if mode == "inflated" else 0.5
        z = q_mean + q_std * rng.standard_normal((samples, 1))[:, 0]
        likelihood = norm.logpdf(xi[0], np.sqrt(3.0) * z) + norm.logpdf(xi[1])
        expected_weights.append(likelihood + norm.logpdf(z) - norm.logpdf(z, q_mean, q_std))
    expected_weights = np.asarray(expected_weights)
    np.testing.assert_allclose(result["log_weights"], expected_weights,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["iw_log_score"],
                               logsumexp(expected_weights, axis=1) - np.log(samples),
                               rtol=0, atol=1e-12)


class QuadraticDecoder:
    input_dim = latent_dim = 1
    b = observation_variance = residual_variance = 1.0

    def encode(self, x):
        return np.zeros_like(x), np.zeros_like(x)

    def decode_mean(self, z):
        return np.square(z)

    def decode_residual(self, z):
        return self.b * self.decode_mean(z)


@pytest.mark.parametrize("value,sign", [(0.0, 1), (2.0, -1)])
def test_nonlinear_posterior_mse_difference_has_no_fixed_sign(value, sign):
    model = QuadraticDecoder()
    result = posterior_score_components(model, np.array([[value]]), 16384, seed=17)
    assert result["posterior_mean_mse"][0] == value**2
    difference = result["posterior_expected_mse"][0] - result["posterior_mean_mse"][0]
    assert sign * difference > 0.5
    np.testing.assert_allclose(result["posterior_expected_nll"],
                               0.5 * (result["posterior_expected_mse"] + np.log(2 * np.pi)))
    np.testing.assert_array_equal(result["kl_by_latent"], [[0.0]])
    np.testing.assert_array_equal(result["kl"], [0.0])


@pytest.mark.parametrize("samples", [1, 8, 256])
def test_probability_components_reconstruct_same_draw_estimates(samples):
    class NonPriorEncoder(QuadraticDecoder):
        def encode(self, x):
            return np.full_like(x, 0.25), np.full_like(x, np.log(1.7))

    result = posterior_score_components(NonPriorEncoder(), np.array([[0.0], [1.0]]),
                                         samples, seed=19)
    np.testing.assert_allclose(result["log_weights"],
                               result["sample_log_likelihood"]
                               - result["sample_logq_minus_logprior"], atol=1e-12)
    np.testing.assert_allclose(result["negative_elbo_analytic_kl"],
                               result["posterior_expected_nll"] + result["kl"])
    np.testing.assert_allclose(result["sample_negative_elbo"],
                               result["posterior_expected_nll"]
                               + result["sample_logq_minus_logprior"].mean(axis=1))
    np.testing.assert_allclose(result["jensen_gap"],
                               result["log_score"] + result["sample_negative_elbo"])
    assert np.all(result["jensen_gap"] >= -1e-12)
    if samples == 1:
        np.testing.assert_allclose(result["jensen_gap"], 0.0, atol=1e-12)
    # Analytic and sampled KL are different estimators, even with shared draws.
    assert not np.allclose(result["sample_negative_elbo"],
                            result["negative_elbo_analytic_kl"], atol=1e-8, rtol=0)


def test_component_draws_do_not_depend_on_chunking():
    x = np.arange(7, dtype=np.float64)[:, None]
    first = posterior_score_components(QuadraticDecoder(), x, 17, seed=29, chunk=1)
    second = posterior_score_components(QuadraticDecoder(), x, 17, seed=29, chunk=4)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_residual_components_match_frozen_likelihood_arithmetic():
    class ResidualDecoder(QuadraticDecoder):
        b = 0.2
        observation_variance = 0.25
        residual_variance = b**2 * observation_variance

    model = ResidualDecoder()
    x = np.array([[0.3], [-0.8]])
    result = posterior_score_components(model, x, 19, seed=37,
                                         density_coordinates="residual")
    z = np.random.default_rng(37).standard_normal((2, 19, 1))
    residual = model.decode_residual(z.reshape(-1, 1).astype(np.float32))
    residual = residual.astype(np.float64).reshape(2, 19, 1)
    expected = (-0.5 * np.square(model.b * x[:, None, :] - residual).sum(axis=-1)
                / model.residual_variance
                - 0.5 * np.log(2 * np.pi * model.residual_variance) + np.log(model.b))
    np.testing.assert_array_equal(result["sample_log_likelihood"], expected)
    np.testing.assert_allclose(result["posterior_expected_nll"],
                               0.5 * np.log(2 * np.pi * model.observation_variance)
                               + result["posterior_expected_mse"]
                               / (2 * model.observation_variance), atol=1e-12)


@pytest.mark.parametrize("argument", [{"samples": 0}, {"samples": True},
                                      {"chunk": 0}, {"density_coordinates": "unknown"}])
def test_components_reject_invalid_protocol(argument):
    keywords = dict(samples=8, seed=0)
    keywords.update(argument)
    with pytest.raises(ValueError):
        posterior_score_components(QuadraticDecoder(), np.array([[1.0]]), **keywords)


@pytest.mark.parametrize("x", [np.array([]), np.empty((0, 1)), np.ones((1, 2)),
                               np.array([[np.nan]])])
def test_components_reject_invalid_observations(x):
    with pytest.raises(ValueError):
        posterior_score_components(QuadraticDecoder(), x, 8, seed=0)


@pytest.mark.parametrize("width", [2, 9, 192])
def test_spectrum_obeys_parseval_and_variance_partition(width):
    windows = np.random.default_rng(width).standard_normal((7, width))
    windows += np.arange(7)[:, None] * 3
    stats = window_spectrum_statistics(windows, np.arange(1, 8))
    np.testing.assert_allclose(stats["pooled_bin_power"].sum(),
                               stats["within_window_variance"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats["within_bin_fraction"].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(stats["pooled_entry_variance"],
                               stats["within_window_variance"]
                               + stats["between_window_mean_variance"], rtol=1e-12)


def test_dart_spectrum_distinguishes_bins_and_pooled_variance():
    time = np.arange(192)
    wave = 3 * np.sin(2 * np.pi * 4 * time / 192) + np.cos(2 * np.pi * 2 * time / 192)
    windows = np.stack([wave, wave + 10])
    stats = window_spectrum_statistics(windows, [1, 3])
    np.testing.assert_allclose(stats["frequency_cycles_per_day"][[2, 4, -1]], [1, 2, 48])
    np.testing.assert_allclose(stats["pooled_bin_power"][[2, 4]], [0.5, 4.5], atol=1e-12)
    np.testing.assert_allclose(stats["within_bin_fraction"][[2, 4]], [0.1, 0.9], atol=1e-12)
    assert stats["between_window_mean_variance"] == pytest.approx(18.75)
    assert stats["within_to_pooled_ratio"] == pytest.approx(5 / 23.75)


def test_spectrum_does_not_double_count_nyquist():
    wave = (-1.0)**np.arange(192)
    stats = window_spectrum_statistics(wave[None, :], [1])
    assert stats["pooled_bin_power"][-1] == pytest.approx(1.0)
    assert stats["within_window_variance"] == pytest.approx(1.0)
    np.testing.assert_allclose(stats["pooled_bin_power"][:-1], 0.0, atol=1e-12)


def test_spectrum_preserves_declared_affine_units():
    x = np.random.default_rng(42).standard_normal((4, 192))
    before = window_spectrum_statistics(x, [1, 2, 3, 4])
    after = window_spectrum_statistics(1.5 * x + 2600, [1, 2, 3, 4])
    np.testing.assert_allclose(after["pooled_bin_power"], 1.5**2 * before["pooled_bin_power"],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(after["within_bin_fraction"], before["within_bin_fraction"],
                               rtol=1e-10, atol=1e-12)


def test_constant_windows_have_no_within_variance_fraction():
    result = window_spectrum_statistics(np.array([[1.0] * 192, [3.0] * 192]), [1, 1])
    assert result["within_bin_fraction"] is None
    assert result["within_window_variance"] == 0
    assert result["within_to_pooled_ratio"] == 0
    identical = window_spectrum_statistics(np.ones((2, 192)), [1, 1])
    assert identical["within_to_pooled_ratio"] is None


def test_linear_complete_collapse_has_population_descent_direction():
    # W=t*u1 with an exact posterior; this is not a nonlinear-network test.
    variance = 1.5625
    eigenvalues = np.linspace(4.0, 1.0, 8)**2 + 0.25**2
    assert np.count_nonzero(eigenvalues > variance) == 7
    coefficient = (variance - eigenvalues[0]) / (2 * variance**2)
    assert coefficient == pytest.approx(-2.9696)
    t = 1e-4
    difference = 0.5 * (np.log1p(t**2 / variance)
                        - eigenvalues[0] * t**2 / (variance * (variance + t**2)))
    assert difference < 0
    assert difference / t**2 == pytest.approx(coefficient, rel=1e-7)


def test_centered_energy_removes_independent_draw_and_observation_offsets():
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((16, 192))
    observations = rng.standard_normal((3, 192))
    expected = energy_score(draws - draws.mean(axis=1, keepdims=True),
                            observations - observations.mean(axis=1, keepdims=True))
    np.testing.assert_array_equal(centered_energy_score(draws, observations), expected)
    shifted_draws = draws + np.arange(16)[:, None] * 7
    shifted_observations = observations + np.array([100, -200, 350])[:, None]
    for shifted in ((shifted_draws, observations), (draws, shifted_observations),
                    (shifted_draws, shifted_observations)):
        np.testing.assert_allclose(centered_energy_score(*shifted), expected, atol=1e-12)
        assert not np.allclose(energy_score(*shifted), energy_score(draws, observations))


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 2, 2)), np.empty((0, 3)),
                                 np.empty((2, 0)), [[np.nan] * 3], [[np.inf] * 3]])
@pytest.mark.parametrize("which", ["draws", "observations"])
def test_centered_energy_rejects_invalid_rows(bad, which):
    args = dict(draws=np.ones((2, 3)), observations=np.ones((2, 3)))
    args[which] = bad
    with pytest.raises(ValueError):
        centered_energy_score(**args)


def test_centered_energy_preserves_frozen_width_and_sample_count_rules():
    with pytest.raises(ValueError):
        centered_energy_score(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        centered_energy_score(np.ones((1, 3)), np.ones((2, 3)))


class LinearGaussianDecoder:
    """Analytical q and observation-space decoder, with no framework state."""
    latent_dim = 2
    b = 0.25
    alpha = 1 - b
    observation_variance = 0.75
    residual_variance = b**2 * observation_variance

    def __init__(self, input_dim=3):
        self.input_dim = input_dim
        self.loadings = np.column_stack((np.linspace(0.5, 1.5, input_dim),
                                         np.linspace(-0.25, 0.75, input_dim)))
        self.bias = np.linspace(-0.5, 0.5, input_dim)

    def encode(self, x):
        assert x.dtype == np.float32
        return (0.25 * x[:, :2] + 0.5,
                np.broadcast_to(np.log(np.array([0.5, 1.25], dtype=np.float32)),
                                (len(x), self.latent_dim)))

    def decode_mean(self, z):
        assert z.dtype == np.float32
        return z.astype(np.float64) @ self.loadings.T + self.bias

    def decode_residual(self, z):
        return self.b * self.decode_mean(z)


TRAINING = np.array([[0.25, -1.0, 2.0], [3.0, 2.0, -0.5], [-1.5, 0.5, 1.0]])


def _cached_posterior(model, x, samples, seed):
    mean, log_variance = (np.asarray(a, dtype=np.float64)
                          for a in model.encode(x.astype(np.float32)))
    epsilon = np.random.default_rng(seed).standard_normal((len(x), samples, model.latent_dim))
    z = mean[:, None, :] + np.exp(0.5 * log_variance[:, None, :]) * epsilon
    decoded = model.decode_mean(z.reshape(-1, model.latent_dim).astype(np.float32))
    return mean, log_variance, epsilon, z, decoded.reshape(len(x), samples, model.input_dim)


@pytest.mark.parametrize("coordinates", ["observation", "residual"])
def test_default_components_exact_full_arithmetic_regression(coordinates):
    model = LinearGaussianDecoder()
    x, samples, seed = TRAINING, 17, 33
    mean, log_variance, epsilon, z, decoded = _cached_posterior(model, x, samples, seed)
    if coordinates == "residual":
        difference = model.b * x[:, None, :] - model.b * decoded
        squared_error = np.square(difference).sum(axis=-1) / model.b**2
        likelihood = (-0.5 * np.square(difference).sum(axis=-1) / model.residual_variance
                      - 0.5 * model.input_dim * np.log(2 * np.pi * model.residual_variance)
                      + model.input_dim * np.log(model.b))
    else:
        squared_error = np.square(x[:, None, :] - decoded).sum(axis=-1)
        likelihood = (-0.5 * squared_error / model.observation_variance
                      - 0.5 * model.input_dim * np.log(2 * np.pi * model.observation_variance))
    log_prior = -0.5 * (np.square(z) + np.log(2 * np.pi)).sum(axis=-1)
    log_q = -0.5 * (np.square(epsilon) + log_variance[:, None, :]
                    + np.log(2 * np.pi)).sum(axis=-1)
    log_weights = likelihood + log_prior - log_q
    stats = importance_statistics(log_weights)
    kl_by_latent = 0.5 * (np.square(mean) + np.expm1(log_variance) - log_variance)
    kl = kl_by_latent.sum(axis=-1)
    nll = -likelihood.mean(axis=1)
    expected = {
        "log_score": stats["log_score"], "ess": stats["ess"],
        "log_weights": log_weights, "sample_log_likelihood": likelihood,
        "sample_logq_minus_logprior": log_q - log_prior,
        "posterior_expected_nll": nll,
        "posterior_expected_mse": squared_error.mean(axis=1) / model.input_dim,
        "posterior_mean_mse": np.square(x - model.decode_mean(mean.astype(np.float32))).mean(axis=-1),
        "kl": kl, "kl_by_latent": kl_by_latent,
        "negative_elbo_analytic_kl": nll + kl,
        "sample_negative_elbo": -log_weights.mean(axis=1),
        "jensen_gap": stats["log_score"] - log_weights.mean(axis=1),
    }
    omitted = posterior_score_components(model, x, samples, seed, density_coordinates=coordinates)
    explicit = posterior_score_components(model, x, samples, seed,
                                           density_coordinates=coordinates, observation_variance=None)
    assert omitted.keys() == explicit.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(omitted[key], expected[key])
        np.testing.assert_array_equal(explicit[key], expected[key])


@pytest.mark.parametrize("variance", [0.125, 2.5, np.array(7.0)])
@pytest.mark.parametrize("b", [1.0, 0.25, 0.2])
def test_override_density_coordinates_share_draws_and_preserve_functions(variance, b):
    model = LinearGaussianDecoder()
    model.b = b
    model.residual_variance = b**2 * model.observation_variance
    original = posterior_score_components(model, TRAINING, 19, seed=37)
    observed = posterior_score_components(model, TRAINING, 19, seed=37,
                                           observation_variance=variance)
    residual = posterior_score_components(model, TRAINING, 19, seed=37,
                                           observation_variance=variance,
                                           density_coordinates="residual")
    *_, decoded = _cached_posterior(model, TRAINING, 19, 37)
    sse = np.square(TRAINING[:, None, :] - decoded).sum(axis=-1)
    expected = -0.5 * sse / variance - 0.5 * model.input_dim * np.log(2 * np.pi * variance)
    np.testing.assert_array_equal(observed["sample_log_likelihood"], expected)
    for key in observed:
        np.testing.assert_allclose(observed[key], residual[key], rtol=1e-12, atol=1e-12)
    for key in ("posterior_expected_mse", "posterior_mean_mse", "kl", "kl_by_latent",
                "sample_logq_minus_logprior"):
        np.testing.assert_array_equal(observed[key], original[key])
    assert not np.array_equal(original["sample_log_likelihood"], observed["sample_log_likelihood"])


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False),
                                      [1.0], np.ones((1, 1)), "1.0", 1 + 2j])
def test_explicit_variance_rejects_nonpositive_nonfinite_or_nonscalar(variance):
    with pytest.raises(ValueError):
        posterior_score_components(LinearGaussianDecoder(), TRAINING, 8, 0,
                                   observation_variance=variance)
    with pytest.raises(ValueError):
        unconditional_score_streams(LinearGaussianDecoder(192), np.zeros((3, 192)),
                                     [1, 2, 3], variance, samples=16)


@pytest.mark.parametrize("attribute,value", [("observation_variance", 0),
                                             ("observation_variance", np.nan),
                                             ("b", 0), ("b", np.inf),
                                             ("residual_variance", -1),
                                             ("residual_variance", np.inf),
                                             ("residual_variance", 1.0)])
@pytest.mark.parametrize("override", [None, 1.25])
def test_override_cannot_bypass_original_model_variance_contract(attribute, value, override):
    model = LinearGaussianDecoder()
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        posterior_score_components(model, TRAINING, 8, 0, observation_variance=override)


def test_refit_matches_weighted_cached_errors_and_not_mean_code_or_raw_residual():
    model = LinearGaussianDecoder()
    weights = np.array([1, 7, 2])
    result = refit_observation_variance(model, TRAINING, weights, chunk=2)
    assert result.keys() == {"observation_variance", "stream_variances",
                             "posterior_expected_mse_per_window", "normalized_training_weights",
                             "samples", "seeds"}
    assert result["samples"] == 32 and type(result["samples"]) is int
    assert result["seeds"] == (72001, 72002, 72003)
    expected = []
    for seed in result["seeds"]:
        *_, decoded = _cached_posterior(model, TRAINING, result["samples"], seed)
        expected.append(np.square(TRAINING[:, None, :] - decoded).sum(axis=-1).mean(axis=1) / 3)
    expected = np.stack(expected)
    np.testing.assert_array_equal(result["posterior_expected_mse_per_window"], expected)
    np.testing.assert_array_equal(result["normalized_training_weights"], weights / weights.sum())
    np.testing.assert_array_equal(result["stream_variances"], expected @ (weights / weights.sum()))
    assert result["stream_variances"].dtype == np.float64
    assert result["stream_variances"].shape == (3,)
    assert result["observation_variance"] == result["stream_variances"].mean()
    assert len(set(result["stream_variances"])) == 3
    assert not np.isclose(result["observation_variance"], expected.mean())
    mean, _ = model.encode(TRAINING.astype(np.float32))
    mean_code_mse = np.square(TRAINING - model.decode_mean(mean)).mean(axis=1)
    assert not np.isclose(result["observation_variance"], weights @ mean_code_mse / weights.sum())
    assert not np.isclose(result["observation_variance"], model.b**2 * result["observation_variance"])
    rechunked = refit_observation_variance(model, TRAINING, weights * 5, chunk=1)
    for key in ("posterior_expected_mse_per_window", "stream_variances", "normalized_training_weights"):
        np.testing.assert_array_equal(rechunked[key], result[key])


def test_refit_matches_analytical_gaussian_expectation():
    model = LinearGaussianDecoder()
    mean, log_variance = model.encode(TRAINING.astype(np.float32))
    expected = (np.square(TRAINING - model.decode_mean(mean)).mean(axis=1)
                + (np.exp(log_variance.astype(np.float64)) @ np.square(model.loadings).sum(axis=0)) / 3)
    weights = np.array([1, 7, 2]) / 10
    result = refit_observation_variance(model, TRAINING, weights, samples=16384, chunk=1)
    np.testing.assert_allclose(result["posterior_expected_mse_per_window"].mean(axis=0),
                               expected, rtol=0.015)
    assert result["observation_variance"] == pytest.approx(weights @ expected, rel=0.01)


def test_refit_minimizes_reconstruction_nll_for_the_same_cached_sse():
    model = LinearGaussianDecoder()
    result = refit_observation_variance(model, TRAINING, [1, 7, 2], samples=64)
    variance = result["observation_variance"]
    weights = result["normalized_training_weights"]
    cached_mse = result["posterior_expected_mse_per_window"]

    def expected_nll(candidate):
        return 0.5 * model.input_dim * (np.log(2 * np.pi * candidate)
                                       + (cached_mse @ weights).mean() / candidate)

    for candidate in (variance * 0.5, variance * 0.9, variance * 1.1, variance * 2):
        assert expected_nll(candidate) > expected_nll(variance)
    # Re-evaluate the same streams, not fresh draws or validation/test windows.
    for candidate in (variance, variance * 2):
        measured = [posterior_score_components(model, TRAINING, 64, seed,
                                                observation_variance=candidate)[
                                                    "posterior_expected_nll"] @ weights
                    for seed in result["seeds"]]
        assert np.mean(measured) == pytest.approx(expected_nll(candidate), rel=1e-14)


@pytest.mark.parametrize("api", ["refit", "unconditional"])
@pytest.mark.parametrize("change", [
    {"weights": [0, 0, 0]}, {"weights": [1, -1, 1]}, {"weights": [1, np.nan, 1]},
    {"weights": [1, np.inf, 1]}, {"weights": [1e308, 1e308, 1e308]},
    {"weights": [1, 2]}, {"weights": [[1, 2, 3]]},
    {"seeds": (1, 1, 2)}, {"seeds": (1, 2)}, {"seeds": (1, 2, 3, 4)},
    {"seeds": (1, 2, 3.0)}, {"seeds": (1, 2, True)}, {"seeds": (1, 2, -3)},
    {"seeds": None}, {"seeds": 1}, {"samples": 0}, {"samples": 1.5}, {"samples": True},
])
def test_stream_apis_reject_invalid_weights_and_protocol(api, change):
    model = LinearGaussianDecoder(192)
    args = dict(weights=[1, 2, 3], samples=16, seeds=(1, 2, 3))
    args.update(change)
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        if api == "refit":
            refit_observation_variance(model, np.zeros((3, 192)), **args)
        else:
            unconditional_score_streams(model, np.zeros((3, 192)), observation_variance=1.0, **args)


@pytest.mark.parametrize("api", ["refit", "unconditional"])
@pytest.mark.parametrize("bad", [np.ones(192), np.zeros((0, 192)), np.zeros((3, 0)),
                                 np.ones((3, 191)), np.full((3, 192), np.nan),
                                 np.full((3, 192), np.inf)])
def test_stream_apis_reject_invalid_windows(api, bad):
    model = LinearGaussianDecoder(192)
    with pytest.raises(ValueError):
        if api == "refit":
            refit_observation_variance(model, bad, [1, 2, 3])
        else:
            unconditional_score_streams(model, bad, [1, 2, 3], 1.0, samples=16)


@pytest.mark.parametrize("attribute,value", [("input_dim", 0), ("input_dim", True),
                                             ("latent_dim", 0), ("latent_dim", 1.5)])
def test_stream_apis_validate_model_dimensions(attribute, value):
    model = LinearGaussianDecoder(192)
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        refit_observation_variance(model, np.zeros((3, 192)), [1, 2, 3])
    with pytest.raises(ValueError):
        unconditional_score_streams(model, np.zeros((3, 192)), [1, 2, 3], 1.0, samples=16)


@pytest.mark.parametrize("chunk", [0, True, 1.5])
def test_refit_rejects_invalid_chunk(chunk):
    with pytest.raises(ValueError):
        refit_observation_variance(LinearGaussianDecoder(), TRAINING, [1, 2, 3], chunk=chunk)


@pytest.mark.parametrize("decoded", [0.0, np.nan, np.inf])
def test_refit_rejects_nonpositive_or_nonfinite_estimated_variance(decoded):
    class ConstantDecoder(QuadraticDecoder):
        def decode_mean(self, z):
            return np.full_like(z, decoded)

    with np.errstate(invalid="ignore"), pytest.raises(ValueError):
        refit_observation_variance(ConstantDecoder(), np.zeros((3, 1)), [1, 2, 3])


def test_unconditional_streams_replay_explicit_noise_and_all_score_reductions():
    model = LinearGaussianDecoder(192)
    observations = np.random.default_rng(29).standard_normal((3, 192))
    observations += np.array([0, 4, -8])[:, None]
    weights = np.array([1, 7, 2])
    result = unconditional_score_streams(model, observations, weights, 2.5, samples=16)
    assert result["samples"] == 16 and type(result["samples"]) is int
    assert result["seeds"] == (71001, 71002, 71003)
    assert result["observation_variance"] == 2.5
    for key in ("prior_means", "prior_samples"):
        assert result[key].shape == (3, 16, 192)
        assert result[key].dtype == np.float64
    for stream, seed in enumerate(result["seeds"]):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((16, 2)).astype(np.float32)
        means = model.decode_mean(z)
        draws = means + np.sqrt(2.5) * rng.standard_normal((16, 192))
        np.testing.assert_array_equal(result["prior_means"][stream], means)
        np.testing.assert_array_equal(result["prior_samples"][stream], draws)
        for name, score in (("energy", energy_score), ("centered_energy", centered_energy_score),
                            ("increment_energy", increment_energy_score)):
            assert result[name + "_per_window"].shape == (3, 3)
            assert result[name + "_streams"].shape == (3,)
            assert result[name + "_uniform_streams"].shape == (3,)
            expected = score(draws, observations)
            np.testing.assert_array_equal(result[name + "_per_window"][stream], expected)
            assert result[name + "_streams"][stream] == pytest.approx(expected @ (weights / 10))
            assert result[name + "_uniform_streams"][stream] == expected.mean()
            assert not np.isclose(expected @ (weights / 10), expected.mean())
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not np.array_equal(result["prior_means"][first], result["prior_means"][second])
        assert not np.array_equal(result["prior_samples"][first], result["prior_samples"][second])
    reweighted = unconditional_score_streams(model, observations, [1, 1, 1], 2.5, samples=16)
    for key in result:
        if isinstance(result[key], np.ndarray):
            assert np.isfinite(result[key]).all()
            if not key.endswith("_streams") or key.endswith("_uniform_streams"):
                np.testing.assert_array_equal(result[key], reweighted[key])


def test_explicit_variance_is_shared_by_sampler_and_likelihood_without_model_mutation():
    import copy

    model = LinearGaussianDecoder(192)
    before = copy.deepcopy(vars(model))
    attributes = (model.alpha, model.b, model.observation_variance, model.residual_variance)
    observations = np.broadcast_to(np.arange(3)[:, None], (3, 192)).astype(np.float64)
    saved_observations = observations.copy()
    weights = np.array([1.0, 7.0, 2.0])
    fitted = refit_observation_variance(model, observations, weights, samples=16)
    first = unconditional_score_streams(model, observations, weights,
                                         fitted["observation_variance"], samples=16)
    second = unconditional_score_streams(model, observations, weights,
                                          4 * fitted["observation_variance"], samples=16)
    np.testing.assert_array_equal(first["prior_means"], second["prior_means"])
    np.testing.assert_allclose(second["prior_samples"] - second["prior_means"],
                               2 * (first["prior_samples"] - first["prior_means"]), atol=1e-14)
    for result in (first, second):
        variance = result["observation_variance"]
        likelihood = posterior_score_components(model, observations, 16, seed=2,
                                                 observation_variance=variance)
        np.testing.assert_allclose(likelihood["posterior_expected_nll"],
                                   0.5 * model.input_dim * (np.log(2 * np.pi * variance)
                                   + likelihood["posterior_expected_mse"] / variance), rtol=1e-14)
    assert vars(model).keys() == before.keys()
    for key in before:
        np.testing.assert_array_equal(vars(model)[key], before[key])
    assert attributes == (model.alpha, model.b, model.observation_variance, model.residual_variance)
    np.testing.assert_array_equal(observations, saved_observations)
    np.testing.assert_array_equal(weights, [1, 7, 2])


def test_unconditional_preserves_increment_width_and_energy_sample_rules():
    with pytest.raises(ValueError, match="lag"):
        unconditional_score_streams(LinearGaussianDecoder(96), np.zeros((3, 96)),
                                     [1, 2, 3], 1.0, samples=16)
    with pytest.raises(ValueError, match="two samples"):
        unconditional_score_streams(LinearGaussianDecoder(192), np.zeros((3, 192)),
                                     [1, 2, 3], 1.0, samples=1)
    # No hard-coded DART width beyond the existing lag constraint.
    result = unconditional_score_streams(LinearGaussianDecoder(97), np.zeros((3, 97)),
                                         [0, 2, 0], 1.0, samples=np.int64(2),
                                         seeds=np.array([3, 2, 1], dtype=np.int64))
    assert result["prior_samples"].shape == (3, 2, 97)
    assert result["seeds"] == (3, 2, 1)
    np.testing.assert_array_equal(result["energy_streams"], result["energy_per_window"][:, 1])


@pytest.mark.parametrize("bad", [np.zeros((16, 191)), np.zeros((16, 1, 192)),
                                 np.full((16, 192), np.nan), np.full((16, 192), np.inf)])
def test_unconditional_rejects_invalid_decoder_outputs(bad):
    class BadDecoder(LinearGaussianDecoder):
        def decode_mean(self, z):
            return bad

    with pytest.raises(ValueError, match="decoder means"):
        unconditional_score_streams(BadDecoder(192), np.zeros((3, 192)), [1, 2, 3],
                                     1.0, samples=16)
