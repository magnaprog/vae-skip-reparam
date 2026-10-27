"""Gaussian baselines, importance sampling, energy scores and window spectra."""
from __future__ import annotations

import numpy as np

from prigm_metrics import energy_score, gaussian_log_prob, importance_statistics


NOISE_GRID = (1e-4, 5e-4, 2.5e-3, 1.25e-2, 6.25e-2)
INCREMENT_LAGS = (1, 4, 16, 48, 96)


def fixed_noise_ppca(mean, covariance, variance, rank=16):
    """PPCA loadings at a fixed noise variance v: W = V sqrt(max(lam - v, 0))."""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    variance = _positive_variance(variance)
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    top = evals[:rank]
    loadings = evecs[:, :rank] * np.sqrt(np.maximum(top - variance, 0.0))
    model_covariance = loadings @ loadings.T + variance * np.eye(covariance.shape[0])
    effective_rank = int(np.sum(top > variance))
    return {"mean": mean, "loadings": loadings, "variance": variance,
            "covariance": model_covariance, "effective_rank": effective_rank}


def weighted_moments(x, weights):
    """Weighted mean and population covariance; weights need not sum to one."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite 2D array")
    if (weights.shape != (len(x),) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("weights must be finite nonnegative values matching the row count")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have finite positive mass")
    mean = weights @ x / total
    centered = x - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    return mean, covariance


def window_spectrum_statistics(windows, weights):
    """Within-window DFT power and between-window variance for 900-second data.

    Inputs are already in their declared coordinates. Bin fractions divide
    pooled within-window power; they are not estimates of tidal constituents.
    """
    windows = np.asarray(windows, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (windows.ndim != 2 or min(windows.shape) == 0 or windows.shape[1] < 2
            or weights.shape != (len(windows),) or not np.isfinite(windows).all()
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Expected finite windows and matching nonnegative weights with mass")
    weights = weights / weights.sum()
    width = windows.shape[1]
    means = windows.mean(axis=1)
    centered = windows - means[:, None]
    power = np.abs(np.fft.rfft(centered, axis=1))**2 / width**2
    power[:, 1:-1 if width % 2 == 0 else None] *= 2
    pooled_power = weights @ power
    within_variance = float(weights @ np.square(centered).mean(axis=1))
    overall_mean = float(weights @ means)
    between_variance = float(weights @ np.square(means - overall_mean))
    pooled_variance = float(weights @ np.square(windows - overall_mean).mean(axis=1))
    return {
        "frequency_cycles_per_day": np.fft.rfftfreq(width, d=900 / 86400),
        "pooled_bin_power": pooled_power,
        "within_bin_fraction": pooled_power / within_variance if within_variance > 0 else None,
        "within_window_variance": within_variance,
        "between_window_mean_variance": between_variance,
        "pooled_entry_variance": pooled_variance,
        "within_to_pooled_ratio": within_variance / pooled_variance if pooled_variance > 0 else None,
        "window_means": means,
    }


def exact_validation_score(x, weights, mean, covariance):
    """Weighted exact Gaussian log density, averaged over evaluation rows."""
    weights = np.asarray(weights, dtype=np.float64)
    log_prob = gaussian_log_prob(x, mean, covariance)
    if (weights.shape != log_prob.shape or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("weights must be finite nonnegative values matching the row count")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have finite positive mass")
    return float(weights @ log_prob / total)


def select_noise_variance(train_x, train_w, validation_x, validation_w,
                          grid=NOISE_GRID, rank=16):
    """Grid selection by weighted exact validation score; ties to larger v."""
    mean, covariance = weighted_moments(train_x, train_w)
    results = []
    for variance in grid:
        model = fixed_noise_ppca(mean, covariance, variance, rank=rank)
        score = exact_validation_score(validation_x, validation_w,
                                       model["mean"], model["covariance"])
        results.append({"variance": variance, "effective_rank": model["effective_rank"],
                        "weighted_validation_log_density": score})
    best = max(results, key=lambda r: (r["weighted_validation_log_density"],
                                       r["variance"]))
    at_boundary = best["variance"] in (min(grid), max(grid))
    return {"selection": best, "grid_results": results, "boundary_selection": at_boundary,
            "mean": mean, "covariance": covariance}


def increment_features(x, lags=INCREMENT_LAGS):
    """Concatenated lag differences, each block divided by sqrt(d - lag)."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    try:
        lags = tuple(lags)
    except TypeError as error:
        raise ValueError("lags must be positive integers below the window length") from error
    if not lags or any(
        isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer))
        or lag <= 0 or lag >= d for lag in lags
    ):
        raise ValueError("lags must be positive integers below the window length")
    blocks = []
    for lag in lags:
        blocks.append(np.diff(x, axis=1, n=1) if lag == 1 else x[:, lag:] - x[:, :-lag])
        blocks[-1] = blocks[-1] / np.sqrt(d - lag)
    return np.concatenate(blocks, axis=1)


def centered_energy_score(draws, observations):
    """Energy after removing each generated/observed window's own level."""
    draws = np.asarray(draws, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    for rows in (draws, observations):
        if rows.ndim != 2 or min(rows.shape) == 0 or not np.isfinite(rows).all():
            raise ValueError("expected nonempty finite 2D windows with positive width")
    return energy_score(draws - draws.mean(axis=1, keepdims=True),
                        observations - observations.mean(axis=1, keepdims=True))


def increment_energy_score(draws, observations):
    """Energy score on increment features; separate target from raw windows."""
    return energy_score(increment_features(draws), increment_features(observations))


def noise_matched_diagonal(train_x, train_w, variance):
    """Diagonal baseline with total variance max(weighted_var_j, v)."""
    variance = _positive_variance(variance)
    mean, covariance = weighted_moments(train_x, train_w)
    total = np.maximum(np.diag(covariance), variance)
    return {"mean": mean, "covariance": np.diag(total)}


def posterior_score_components(model, observations, samples, seed, chunk=32,
                               density_coordinates="observation", observation_variance=None):
    """Per-window probability components, sharing the same posterior draws.

    Probability arithmetic is float64; neural inputs and latent codes are
    float32. The residual likelihood includes the change-of-variables Jacobian.
    Expected NLL/MSE are Monte Carlo estimates; KL is analytic. The Jensen
    gap compares log-mean weights with mean log weights, not with true log p(x).
    An explicit observation variance changes only the likelihood, deriving the
    residual variance as b**2 times that scalar without changing model state.
    The original model variance contract is validated even with an override;
    None retains the original arithmetic.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if (observations.ndim != 2 or observations.shape[0] == 0
            or observations.shape[1] != model.input_dim
            or not np.isfinite(observations).all()):
        raise ValueError("observations must be nonempty finite model-width rows")
    for name, value in (("samples", samples), ("chunk", chunk)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if density_coordinates not in ("observation", "residual"):
        raise ValueError("unknown density coordinates")
    variance = float(model.observation_variance)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("observation variance must be finite and positive")
    b = float(model.b)
    residual_variance = float(model.residual_variance)
    if (not np.isfinite(b) or b <= 0 or not np.isfinite(residual_variance)
            or residual_variance <= 0
            or not np.isclose(residual_variance / b**2, variance, rtol=1e-12, atol=0)):
        raise ValueError("inconsistent residual and observation variances")
    if observation_variance is not None:
        variance = _positive_variance(observation_variance)
        residual_variance = _positive_variance(b**2 * variance)
    keys = ("log_score", "ess", "log_weights", "sample_log_likelihood",
            "sample_logq_minus_logprior", "posterior_expected_nll",
            "posterior_expected_mse", "posterior_mean_mse", "kl", "kl_by_latent",
            "negative_elbo_analytic_kl", "sample_negative_elbo", "jensen_gap")
    pieces = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    dimension = model.input_dim
    for start in range(0, len(observations), chunk):
        x = observations[start:start + chunk]
        mean, log_variance = model.encode(x.astype(np.float32))
        mean = np.asarray(mean, dtype=np.float64)
        log_variance = np.asarray(log_variance, dtype=np.float64)
        epsilon = rng.standard_normal((len(x), samples, model.latent_dim))
        z = mean[:, None, :] + np.exp(0.5 * log_variance[:, None, :]) * epsilon
        flat = z.reshape(-1, model.latent_dim).astype(np.float32)
        if density_coordinates == "residual":
            residual = np.asarray(model.decode_residual(flat), dtype=np.float64)
            difference = b * x[:, None, :] - residual.reshape(len(x), samples, dimension)
            squared_error = np.square(difference).sum(axis=-1) / b**2
            log_likelihood = (-0.5 * np.square(difference).sum(axis=-1) / residual_variance
                              - 0.5 * dimension * np.log(2 * np.pi * residual_variance)
                              + dimension * np.log(b))
            mean_decode = np.asarray(model.decode_residual(mean.astype(np.float32)),
                                     dtype=np.float64) / b
        else:
            decoded = np.asarray(model.decode_mean(flat), dtype=np.float64)
            difference = x[:, None, :] - decoded.reshape(len(x), samples, dimension)
            squared_error = np.square(difference).sum(axis=-1)
            log_likelihood = (-0.5 * squared_error / variance
                              - 0.5 * dimension * np.log(2 * np.pi * variance))
            mean_decode = np.asarray(model.decode_mean(mean.astype(np.float32)),
                                     dtype=np.float64)
        log_prior = -0.5 * (np.square(z) + np.log(2 * np.pi)).sum(axis=-1)
        log_q = -0.5 * (np.square(epsilon) + log_variance[:, None, :]
                        + np.log(2 * np.pi)).sum(axis=-1)
        log_weights = log_likelihood + log_prior - log_q
        stats = importance_statistics(log_weights)
        kl_by_latent = 0.5 * (np.square(mean) + np.expm1(log_variance) - log_variance)
        kl = kl_by_latent.sum(axis=-1)
        nll = -log_likelihood.mean(axis=1)
        values = {
            "log_score": stats["log_score"], "ess": stats["ess"],
            "log_weights": log_weights, "sample_log_likelihood": log_likelihood,
            "sample_logq_minus_logprior": log_q - log_prior,
            "posterior_expected_nll": nll,
            "posterior_expected_mse": squared_error.mean(axis=1) / dimension,
            "posterior_mean_mse": np.square(x - mean_decode).mean(axis=-1),
            "kl": kl, "kl_by_latent": kl_by_latent,
            "negative_elbo_analytic_kl": nll + kl,
            "sample_negative_elbo": -log_weights.mean(axis=1),
            "jensen_gap": stats["log_score"] - log_weights.mean(axis=1),
        }
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite probability component: {key}")
            pieces[key].append(value)
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _positive_variance(value):
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise ValueError("observation variance must be a finite positive scalar")
    variance = float(value)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("observation variance must be a finite positive scalar")
    return variance


def _score_stream_inputs(model, observations, weights, samples, seeds):
    """Validate the shared three-stream protocol, preserving window coordinates."""
    for name, value in (("input_dim", model.input_dim), ("latent_dim", model.latent_dim),
                        ("samples", samples)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    observations = np.asarray(observations, dtype=np.float64)
    if (observations.ndim != 2 or observations.shape[0] == 0
            or observations.shape[1] != model.input_dim
            or not np.isfinite(observations).all()):
        raise ValueError("observations must be nonempty finite model-width rows")
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (len(observations),) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("weights must be finite nonnegative values matching the window count")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have finite positive mass")
    try:
        seeds = tuple(seeds)
    except TypeError as error:
        raise ValueError("seeds must be three unique nonnegative integers") from error
    if (len(seeds) != 3
            or any(isinstance(seed, (bool, np.bool_))
                   or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds)
            or len(set(seeds)) != 3):
        raise ValueError("seeds must be three unique nonnegative integers")
    return observations, weights / total, int(samples), tuple(int(seed) for seed in seeds)


def refit_observation_variance(model, training, weights, samples=32,
                               seeds=(72001, 72002, 72003), chunk=32):
    """Training-only posterior-expected MSE with fixed encoder/decoder functions.

    The pooled scalar minimizes the Gaussian reconstruction term for these
    cached posterior draws, not marginal negative log likelihood. Callers own
    split selection and must supply only training windows and their weights.
    """
    training, weights, samples, seeds = _score_stream_inputs(
        model, training, weights, samples, seeds)
    per_window = np.stack([
        posterior_score_components(model, training, samples, seed, chunk=chunk)[
            "posterior_expected_mse"]
        for seed in seeds
    ])
    stream_variances = per_window @ weights
    for variance in stream_variances:
        _positive_variance(variance)
    variance = _positive_variance(stream_variances.mean())
    return {"observation_variance": variance, "stream_variances": stream_variances,
            "posterior_expected_mse_per_window": per_window,
            "normalized_training_weights": weights, "samples": samples, "seeds": seeds}


def unconditional_score_streams(model, observations, weights, observation_variance,
                                samples=1024, seeds=(71001, 71002, 71003)):
    """Three prior-predictive streams using one explicit observation variance.

    Each RNG draws float32 latent codes, then float64 observation noise. Reusing
    seeds across conditions pairs both draws; distinct seeds separate streams.
    Scores retain original window coordinates and both weighted/uniform means.
    """
    observations, weights, samples, seeds = _score_stream_inputs(
        model, observations, weights, samples, seeds)
    variance = _positive_variance(observation_variance)
    if samples < 2:
        raise ValueError("need at least two samples")
    if max(INCREMENT_LAGS) >= model.input_dim:
        raise ValueError("lag must be smaller than the window length")
    prior_means, prior_samples = [], []
    scores = {"energy": [], "centered_energy": [], "increment_energy": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((samples, model.latent_dim)).astype(np.float32)
        means = np.asarray(model.decode_mean(z), dtype=np.float64)
        if means.shape != (samples, model.input_dim) or not np.isfinite(means).all():
            raise ValueError("decoder means must be finite model-width rows")
        draws = means + np.sqrt(variance) * rng.standard_normal(means.shape)
        if not np.isfinite(draws).all():
            raise ValueError("prior samples must be finite")
        prior_means.append(means)
        prior_samples.append(draws)
        scores["energy"].append(energy_score(draws, observations))
        scores["centered_energy"].append(centered_energy_score(draws, observations))
        scores["increment_energy"].append(increment_energy_score(draws, observations))
    result = {"observation_variance": variance, "samples": samples, "seeds": seeds,
              "prior_means": np.stack(prior_means), "prior_samples": np.stack(prior_samples)}
    for name, values in scores.items():
        per_window = np.stack(values)
        result[name + "_per_window"] = per_window
        result[name + "_streams"] = per_window @ weights
        result[name + "_uniform_streams"] = per_window.mean(axis=1)
    for name, value in result.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite unconditional score component: {name}")
    return result


def linear_model_iw_control(covariance, noise_variance, x, prior_samples,
                            seed=0, proposal_mode="exact", rank=8):
    """IW calibration against a fitted linear model's own analytic density.

    The control model is z ~ N(0, I_rank) through the leading loadings of
    covariance - noise*I with isotropic noise v. Its marginal
    N(0, WW^T + vI) and exact posterior are analytic. importance_statistics
    is then compared with exact log p(x) under the exact posterior or
    controlled mean/variance misspecification.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    d = covariance.shape[0]
    evals, evecs = np.linalg.eigh(covariance - noise_variance * np.eye(d))
    order = np.argsort(evals)[::-1]
    top = evals[order][:rank]
    loadings = evecs[:, order][:, :rank] * np.sqrt(np.maximum(top, 0.0))
    marginal_covariance = loadings @ loadings.T + noise_variance * np.eye(d)
    exact_log_marginal = gaussian_log_prob(x, np.zeros(d), marginal_covariance)
    rng = np.random.default_rng(seed)
    # Preserve the initial draws so later importance samples retain their stream.
    rng.standard_normal((prior_samples, loadings.shape[1]))
    log_weights = []
    for xi in x:
        precision = loadings.T @ loadings / noise_variance + np.eye(loadings.shape[1])
        posterior_covariance = np.linalg.inv(precision)
        posterior_mean = posterior_covariance @ (loadings.T @ xi) / noise_variance
        if proposal_mode == "exact":
            q_mean, q_covariance = posterior_mean, posterior_covariance
        elif proposal_mode == "inflated":
            q_mean, q_covariance = posterior_mean, posterior_covariance * 4.0
        elif proposal_mode == "shifted":
            q_mean, q_covariance = posterior_mean + 1.0, posterior_covariance
        else:
            raise ValueError("unknown proposal mode")
        eps = rng.standard_normal((prior_samples, loadings.shape[1]))
        q_cholesky = np.linalg.cholesky(q_covariance)
        z = q_mean[None, :] + eps @ q_cholesky.T
        residual = xi - z @ loadings.T
        log_likelihood = -0.5 * (np.sum(residual ** 2, axis=1) / noise_variance
                                 + d * np.log(2 * np.pi * noise_variance))
        log_prior = -0.5 * np.sum(z ** 2 + np.log(2 * np.pi), axis=1)
        log_q = (-0.5 * np.sum(eps ** 2 + np.log(2 * np.pi), axis=1)
                 - np.log(np.diag(q_cholesky)).sum())
        log_weights.append(log_likelihood + log_prior - log_q)
    log_weights = np.asarray(log_weights)
    stats = importance_statistics(log_weights)
    return {"exact_log_marginal": exact_log_marginal, "log_weights": log_weights,
            "iw_log_score": stats["log_score"], "ess": stats["ess"],
            "error": stats["log_score"] - exact_log_marginal,
            "proposal_mode": proposal_mode}
