"""Gaussian VAE with observation or residual output coordinates.

Base, canonical and base_fast store the observation head w. Raw, raw_half
and head_rate store r = b*w, where b = 1 - alpha. All likelihoods use the
observation mean g(z) = output_head(z) / head_scale and the same declared
observation variance. Base and canonical use identical forward arithmetic.

The equivalent residual variance is b**2 * observation_variance. Residual
means b*g(z) and self-reconstructions alpha*x + b*g(z) are diagnostics only.
Weight files declare their output coordinates, scale and observation variance.
Loading requires these attributes to match the model.
"""
from __future__ import annotations

import math
import os
from numbers import Integral

import h5py
import tensorflow as tf


STORAGE_CONTRACT = 3
HEAD_RATE = "head_rate"  # raw coordinates with head learning rate scaled by b
BASE_FAST = "base_fast"  # observation coordinates with head learning rate scaled by 5
CONDITIONS = ("base", "raw", "canonical", "raw_half", HEAD_RATE, BASE_FAST)


def condition_settings(condition, observation_variance):
    """Return density, stored head coordinates (w or r), and optimizer settings."""
    if condition == "base":
        alpha, mode = 0.0, "plain"
    elif condition == "raw":
        alpha, mode = 0.8, "plain"
    elif condition == "canonical":
        alpha, mode = 0.8, "canonical"
    elif condition == "raw_half":
        alpha, mode = 0.5, "plain"
    elif condition == HEAD_RATE:
        alpha, mode = 0.8, "head_rate"
    elif condition == BASE_FAST:
        alpha, mode = 0.0, "head_rate"
    else:
        raise ValueError(f"unknown condition {condition!r}")
    try:
        observation_variance = float(observation_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_variance must be finite and positive") from exc
    if not math.isfinite(observation_variance) or observation_variance <= 0:
        raise ValueError("observation_variance must be finite and positive")
    b = 1.0 - alpha
    residual_variance = b**2 * observation_variance
    head_coordinates = "w" if condition in ("base", "canonical", BASE_FAST) else "r"
    head_scale = 1.0 if head_coordinates == "w" else b
    head_rate_multiplier = b if condition == HEAD_RATE else (5.0 if condition == BASE_FAST else 1.0)
    return {"alpha": alpha, "b": b, "residual_variance": residual_variance,
            "observation_variance": observation_variance, "optimizer_mode": mode,
            "storage_contract": STORAGE_CONTRACT, "head_coordinates": head_coordinates,
            "head_scale": head_scale, "head_rate_multiplier": head_rate_multiplier}


class BypassVAE(tf.keras.Model):
    """Gaussian encoder and nonlinear decoder with an affine output layer.

    The encoder has ReLU hidden layers of widths 128 and 64; the decoder
    has ReLU hidden layers of widths 64 and 128. The diagonal posterior has
    log variances limited to [-12, 12]. Sampling noise is supplied by the caller.
    Initialization multiplies the output kernel and bias by head_scale once.
    """

    storage_contract = STORAGE_CONTRACT

    def __init__(self, input_dim=192, latent_dim=16, condition="base",
                 observation_variance=1.0, seed=42, dtype=None):
        super().__init__(name="bypass_vae", dtype=dtype)
        for name, value in (("input_dim", input_dim), ("latent_dim", latent_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        settings = condition_settings(condition, observation_variance)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.condition = condition
        self.observation_variance = float(observation_variance)
        self.seed = int(seed)
        self.alpha = settings["alpha"]
        self.b = settings["b"]
        self.residual_variance = settings["residual_variance"]
        self.optimizer_mode = settings["optimizer_mode"]
        self.head_coordinates = settings["head_coordinates"]
        self.head_scale = settings["head_scale"]
        self.head_rate_multiplier = settings["head_rate_multiplier"]

        def dense(width, offset, name, activation=None):
            return tf.keras.layers.Dense(
                width, activation=activation, name=name, dtype=self.dtype_policy,
                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=self.seed + offset),
                bias_initializer=tf.keras.initializers.Zeros(),
            )

        self.encoder_hidden = (
            dense(128, 0, "encoder_128", "relu"),
            dense(64, 1, "encoder_64", "relu"),
        )
        self.mean_head = dense(self.latent_dim, 2, "posterior_mean")
        self.logvar_head = dense(self.latent_dim, 3, "posterior_log_variance")
        self.decoder_hidden = (
            dense(64, 4, "decoder_64", "relu"),
            dense(128, 5, "decoder_128", "relu"),
        )
        self.output_head = dense(self.input_dim, 6, "output_mean")

        self(tf.zeros((1, self.input_dim), dtype=self.compute_dtype))
        if self.head_scale != 1.0:
            self.output_head.kernel.assign(self.output_head.kernel * self.head_scale)
            self.output_head.bias.assign(self.output_head.bias * self.head_scale)

    # Weight storage excludes optimizer and RNG state.
    def _weight_contract(self):
        return {"storage_contract": self.storage_contract,
                "head_coordinates": self.head_coordinates,
                "head_scale": self.head_scale,
                "observation_variance": self.observation_variance}

    @staticmethod
    def _hdf5_path(filepath):
        filepath = os.fspath(filepath)
        if not filepath.endswith((".h5", ".hdf5")):
            raise ValueError("storage_contract=3 requires an HDF5 weight file (.h5 or .hdf5)")
        return filepath

    def save_weights(self, filepath, overwrite=True):
        """Save HDF5 weights with mandatory coordinate-contract attributes."""
        filepath = self._hdf5_path(filepath)
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(filepath)
        result = super().save_weights(filepath, overwrite=overwrite)
        with h5py.File(filepath, "r+") as checkpoint:
            checkpoint.attrs.update(self._weight_contract())
        return result

    def load_weights(self, filepath):
        """Reject missing/incompatible contracts before Keras assigns any weights."""
        filepath = self._hdf5_path(filepath)
        with h5py.File(filepath, "r") as checkpoint:
            for key, expected in self._weight_contract().items():
                if key not in checkpoint.attrs or checkpoint.attrs[key] != expected:
                    raise ValueError(f"incompatible weight storage contract: {key}; "
                                     "weights must declare matching output coordinates and observation variance")
        return super().load_weights(filepath)

    def _tensor(self, value, width):
        value = tf.cast(tf.convert_to_tensor(value), self.compute_dtype)
        tf.debugging.assert_rank_at_least(value, 1)
        tf.debugging.assert_equal(tf.shape(value)[-1], width,
                                  message="incorrect final dimension")
        return value

    def encode(self, x):
        x = self._tensor(x, self.input_dim)
        hidden = tf.reshape(x, (-1, self.input_dim))
        for layer in self.encoder_hidden:
            hidden = layer(hidden)
        shape = tf.concat([tf.shape(x)[:-1], [self.latent_dim]], axis=0)
        mean = tf.reshape(self.mean_head(hidden), shape)
        log_variance = tf.reshape(self.logvar_head(hidden), shape)
        return mean, tf.clip_by_value(log_variance, -12.0, 12.0)

    def decode_mean(self, z):
        z = self._tensor(z, self.latent_dim)
        hidden = tf.reshape(z, (-1, self.latent_dim))
        for layer in self.decoder_hidden:
            hidden = layer(hidden)
        shape = tf.concat([tf.shape(z)[:-1], [self.input_dim]], axis=0)
        return tf.reshape(self.output_head(hidden), shape) / self.head_scale

    def decode_residual(self, z):
        """Diagnostic residual mean; never used by the likelihood."""
        return self.b * self.decode_mean(z)

    def call(self, x, training=None):
        """Diagnostic self-reconstruction, not the observation-space decoder mean."""
        x = self._tensor(x, self.input_dim)
        mean, _ = self.encode(x)
        return self.alpha * x + self.b * self.decode_mean(mean)

    def _observation_log_density(self, difference):
        squared_error = tf.reduce_sum(tf.square(difference), axis=-1)
        log_normalizer = self.input_dim * (
            math.log(2.0 * math.pi) + math.log(self.observation_variance))
        return -0.5 * (squared_error / self.observation_variance + log_normalizer)

    def log_likelihood(self, x, z):
        x = self._tensor(x, self.input_dim)
        return self._observation_log_density(x - self.decode_mean(z))

    def loss_terms(self, x, epsilon):
        """Per-example one-sample negative ELBO; nll and kl sum dimensions."""
        x = self._tensor(x, self.input_dim)
        mean, log_variance = self.encode(x)
        epsilon = self._tensor(epsilon, self.latent_dim)
        tf.debugging.assert_equal(tf.rank(epsilon), tf.rank(mean),
                                  message="epsilon must match the encoder mean rank")
        tf.debugging.assert_equal(tf.shape(epsilon), tf.shape(mean),
                                  message="epsilon must match the encoder mean shape")
        z = mean + tf.exp(0.5 * log_variance) * epsilon
        difference = x - self.decode_mean(z)
        nll = -self._observation_log_density(difference)
        kl = 0.5 * tf.reduce_sum(
            tf.square(mean) + tf.exp(log_variance) - 1.0 - log_variance, axis=-1)
        corrected_mse = tf.reduce_mean(tf.square(difference), axis=-1)
        return {
            "negative_elbo": nll + kl,
            "nll": nll,
            "kl": kl,
            "self_mse": self.b**2 * corrected_mse,
            "corrected_mse": corrected_mse,
        }


class CanonicalBypassOptimizer:
    """Adam on stored variables, with one shared global clipping scalar.

    Defaults: lr=1e-3, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=False,
    global norm cap=10. Head learning rate alone is multiplied by the model's
    head_rate_multiplier. No shadow parameters or coordinate conversions are
    used. Adam's epsilon is added to sqrt(v), before bias correction is folded
    into the learning rate (TensorFlow 2.15), not to sqrt(v_hat).
    """

    def __init__(self, model, learning_rate=1e-3, global_norm=10.0,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-7, clip=True):
        if not isinstance(model, BypassVAE):
            raise TypeError("CanonicalBypassOptimizer requires a BypassVAE")
        self.model = model
        self.learning_rate = float(learning_rate)
        self.global_norm = float(global_norm)
        if not math.isfinite(self.global_norm) or self.global_norm <= 0:
            raise ValueError("global_norm must be finite and positive")
        self.clip = bool(clip)
        self.beta_1, self.beta_2, self.epsilon = float(beta_1), float(beta_2), float(epsilon)
        self.clipping_scale = tf.Variable(
            1.0, dtype=model.compute_dtype, trainable=False, name="clipping_scale")
        self.clipping_active = tf.Variable(
            False, dtype=tf.bool, trainable=False, name="clipping_active")
        head_lr = self.learning_rate * model.head_rate_multiplier
        self._optimizers = {
            "body": tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, beta_1=beta_1, beta_2=beta_2,
                epsilon=epsilon, amsgrad=False),
            "head": tf.keras.optimizers.Adam(
                learning_rate=head_lr, beta_1=beta_1, beta_2=beta_2,
                epsilon=epsilon, amsgrad=False),
        }
        self._head_weights = list(model.output_head.trainable_variables)
        self._head_refs = {v.ref() for v in self._head_weights}
        self._body_weights = [v for v in model.trainable_variables
                              if v.ref() not in self._head_refs]
        self._optimizers["body"].build(self._body_weights)
        self._optimizers["head"].build(self._head_weights)

    @property
    def steps(self):
        """Executed updates, including under tf.function, not Python trace calls."""
        return self._optimizers["body"].iterations

    def apply_gradients(self, grads_and_vars):
        """Apply ordered stored-variable gradients; return their pre-clip norm."""
        pairs = list(grads_and_vars)
        variables = [pair[1] for pair in pairs]
        expected = list(self.model.trainable_variables)
        if len(variables) != len(expected) or any(v is not e for v, e in zip(variables, expected)):
            raise ValueError("gradients must cover model.trainable_variables in order")
        gradients = [pair[0] for pair in pairs]
        if any(g is None for g in gradients):
            raise RuntimeError("Disconnected trainable variable")
        norm = tf.linalg.global_norm(gradients)
        threshold = tf.cast(self.global_norm, norm.dtype)
        # Exactly one below/at the cap: no epsilon perturbation of small gradients.
        scale = threshold / tf.maximum(norm, threshold) if self.clip else tf.ones_like(norm)
        self.clipping_scale.assign(scale)
        self.clipping_active.assign(scale < 1.0)
        clipped = [g * scale for g in gradients]
        grad_body, grad_head = [], []
        for g, v in zip(clipped, variables):
            if v.ref() in self._head_refs:
                grad_head.append(g)
            else:
                grad_body.append(g)
        self._optimizers["body"].apply_gradients(zip(grad_body, self._body_weights))
        self._optimizers["head"].apply_gradients(zip(grad_head, self._head_weights))
        return norm


def map_bypass(source, target):
    """Map encoder and decoder weights to the target parameterization.

    Requires equal input/latent dimensions and observation variance. Copies
    encoder hidden layers, posterior heads and decoder hidden layers. Output-head
    weights are scaled by target.head_scale/source.head_scale to preserve the
    observation-space mean function. Target configuration is retained; optimizer
    state is not copied. With the common prior and matched observation variance,
    this maps generative distributions without asserting equivalent optimizer
    trajectories.
    """
    if not isinstance(source, BypassVAE) or not isinstance(target, BypassVAE):
        raise TypeError("map_bypass requires two BypassVAE instances")
    if (source.input_dim, source.latent_dim) != (target.input_dim, target.latent_dim):
        raise ValueError("source and target dimensions must match")
    if not math.isclose(target.observation_variance, source.observation_variance,
                        rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("source and target observation variance must match")
    source_layers = (*source.encoder_hidden, source.mean_head, source.logvar_head,
                     *source.decoder_hidden)
    target_layers = (*target.encoder_hidden, target.mean_head, target.logvar_head,
                     *target.decoder_hidden)
    for target_layer, source_layer in zip(target_layers, source_layers):
        for target_weight, source_weight in zip(target_layer.weights, source_layer.weights):
            target_weight.assign(source_weight)
    ratio = target.head_scale / source.head_scale
    for target_weight, source_weight in zip(target.output_head.weights, source.output_head.weights):
        target_weight.assign(source_weight if ratio == 1.0 else source_weight * ratio)
    return target
