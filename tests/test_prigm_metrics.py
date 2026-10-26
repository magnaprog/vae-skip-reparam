import os
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prigm_metrics import (
    energy_score,
    fit_ppca,
    gaussian_kl,
    gaussian_log_prob,
    importance_statistics,
)


def test_gaussian_log_prob_matches_scipy():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=3)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3) * 0.5
    x = rng.normal(size=(4, 3))
    expected = multivariate_normal(mean=mean, cov=cov).logpdf(x)
    actual = gaussian_log_prob(x, mean, cov)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_energy_score_small_known_case():
    samples = np.array([[0.0], [2.0]])
    observations = np.array([[1.0]])
    actual = energy_score(samples, observations)
    np.testing.assert_allclose(actual, np.array([0.0]))


def test_importance_statistics_equal_and_dominant_weights():
    equal = np.zeros((3, 4))
    stats = importance_statistics(equal)
    np.testing.assert_allclose(stats["ess"], np.full(3, 4.0))
    np.testing.assert_allclose(stats["normalized_ess"], np.ones(3))

    dominant = np.array([[0.0, -1000.0, -1000.0]])
    stats = importance_statistics(dominant)
    np.testing.assert_allclose(stats["ess"], np.array([1.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats["normalized_ess"], np.array([1.0 / 3.0]), rtol=1e-12, atol=1e-12)


def test_fit_ppca_covariance_positive_and_floor():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 5))
    fit = fit_ppca(x, rank=3, variance_floor=0.2)
    cov = fit["covariance"]
    np.testing.assert_allclose(cov, cov.T)
    eigs = np.linalg.eigvalsh(cov)
    assert np.all(eigs > 0)
    assert fit["variance"] >= 0.2
    assert fit["loadings"].shape == (5, 3)


def test_invalid_shapes_rejected():
    with pytest.raises(ValueError):
        gaussian_log_prob(np.array([1.0, 2.0]), np.zeros(2), np.eye(2))
    with pytest.raises(ValueError):
        energy_score(np.array([[1.0, 2.0]]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        importance_statistics(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        fit_ppca(np.ones((2, 3)), rank=2)


def test_gaussian_kl_sums_per_example():
    mean = np.array([[0.0, 1.0], [2.0, 0.0]])
    log_var = np.zeros_like(mean)
    actual = gaussian_kl(mean, log_var)
    expected = 0.5 * np.sum(mean ** 2, axis=-1)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("offset", [0.0, -1e12, 1e12, -1e16, 1e16])
def test_importance_statistics_ess_invariant_to_large_common_offsets(offset):
    # Even-integer differences survive float64 rounding at these offsets.
    weights = np.array([[0., 0., 0., 0.], [0., -2., -4., -8.]])
    normalized = np.exp(weights) / np.exp(weights).sum(axis=1, keepdims=True)
    expected_ess = 1.0 / np.sum(normalized ** 2, axis=1)
    stats = importance_statistics(weights + offset)
    np.testing.assert_allclose(stats["ess"], expected_ess, rtol=1e-12)
    np.testing.assert_allclose(stats["normalized_ess"], expected_ess / 4, rtol=1e-12)
    # Preserve the separate marginal-score calculation, including its units.
    from scipy.special import logsumexp
    np.testing.assert_array_equal(stats["log_score"], logsumexp(weights + offset, axis=1) - np.log(4))
