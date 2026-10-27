from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp


def _as_2d_float64(x, name: str):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def fit_ppca(x, rank: int = 16, variance_floor: float = 1e-4):
    x = _as_2d_float64(x, "x")
    n, d = x.shape
    if not (1 <= rank <= d):
        raise ValueError("rank must satisfy 1 <= rank <= n_features")
    if n <= rank:
        raise ValueError("n_samples must be greater than rank")
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / float(n)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    top = evals[:rank]
    noise = float(max(np.mean(evals[rank:]) if rank < d else 0.0, variance_floor))
    loading_scale = np.sqrt(np.maximum(top - noise, 0.0))
    loadings = evecs[:, :rank] * loading_scale
    covariance = loadings @ loadings.T + noise * np.eye(d, dtype=np.float64)
    return {"mean": mean, "loadings": loadings, "variance": noise, "covariance": covariance}


def gaussian_log_prob(x, mean, covariance):
    x = _as_2d_float64(x, "x")
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mean.shape[0] or x.shape[1] != mean.shape[0] or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("invalid Gaussian parameters")
    L = cholesky(cov, lower=True, check_finite=True)
    diff = (x - mean).T
    sol = solve_triangular(L, diff, lower=True, check_finite=True)
    maha = np.sum(sol * sol, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    d = mean.shape[0]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


def energy_score(samples, observations):
    samples = _as_2d_float64(samples, "samples")
    observations = _as_2d_float64(observations, "observations")
    if samples.shape[1] != observations.shape[1] or observations.shape[0] == 0:
        raise ValueError("shape mismatch or empty observations")
    obs_to_samples = cdist(observations, samples)
    mean_term = obs_to_samples.mean(axis=1)
    if samples.shape[0] < 2:
        raise ValueError("need at least two samples")
    pair_term = pdist(samples).mean()
    return mean_term - 0.5 * pair_term


def importance_statistics(log_weights):
    lw = np.asarray(log_weights, dtype=np.float64)
    if lw.ndim != 2 or not np.all(np.isfinite(lw)):
        raise ValueError("log_weights must be a finite 2D array")
    k = lw.shape[1]
    lse = logsumexp(lw, axis=1)
    log_score = lse - np.log(k)
    # Center both ESS reductions to avoid cancellation of a large common offset.
    centered = lw - np.max(lw, axis=1, keepdims=True)
    centered_lse = logsumexp(centered, axis=1)
    ess = np.exp(2.0 * centered_lse - logsumexp(2.0 * centered, axis=1))
    return {"log_score": log_score, "ess": ess, "normalized_ess": ess / k}


def gaussian_kl(mean, log_variance):
    mean = np.asarray(mean, dtype=np.float64)
    log_variance = np.asarray(log_variance, dtype=np.float64)
    if mean.shape != log_variance.shape or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(log_variance)):
        raise ValueError("mean and log_variance must match and be finite")
    return 0.5 * np.sum(np.exp(log_variance) + mean * mean - 1.0 - log_variance, axis=-1)
