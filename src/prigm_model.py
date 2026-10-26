"""Single-pass Gaussian VAEs for the four controlled PriGM training arms.

The normalized conditional density is b**d * N(b*x; f(z), residual_variance*I),
where b = 1 - alpha. Arm settings are fixed Python scalars, not model weights.
There is no input clipping, regularization penalty, or extra KL multiplier.
"""

import math
from numbers import Integral

import tensorflow as tf


def arm_settings(arm, base_variance):
    """Return the fixed bypass and noise settings; base_variance is sigma_0**2.

    The caller fits/floors the base variance once. In particular, C's residual
    variance is not floored again after multiplying by b**2.
    """
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("arm must be one of A, B, C, D")
    try:
        base_variance = float(base_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError("base_variance must be finite and positive") from exc
    if not math.isfinite(base_variance) or base_variance <= 0:
        raise ValueError("base_variance must be finite and positive")

    alpha = 0.8 if arm in ("B", "C") else 0.0
    b = 1.0 - alpha
    residual_variance = base_variance
    if arm == "C":
        residual_variance *= b**2
    elif arm == "D":
        residual_variance /= (1.0 - 0.8)**2
    observation_variance = residual_variance / b**2
    if not all(math.isfinite(v) and v > 0 for v in
               (residual_variance, observation_variance)):
        raise ValueError("derived variances must be finite and positive")
    return {
        "alpha": alpha,
        "b": b,
        "residual_variance": residual_variance,
        "observation_variance": observation_variance,
    }


class GaussianVAE(tf.keras.Model):
    """Dense diagonal-Gaussian posterior and nonlinear residual-mean decoder.

    The decoder has two ReLU hidden layers and an affine output layer.

    Posterior log-variance is bounded to [-12, 12], defining a constrained
    Gaussian variational family. Neither observations nor posterior means are
    clipped. Sampling noise is supplied by the caller, so scoring/training
    does not consume an implicit RNG stream.

    Weights are built and the residual head scaled once in __init__. Save/load
    with a real .weights.h5 path and reconstruct the same arm/dimensions/noise
    settings before loading; a weights-only file does not store those settings.
    """

    def __init__(self, input_dim=192, latent_dim=16, arm="A",
                 base_variance=1.0, seed=42):
        super().__init__(name="gaussian_vae")
        for name, value in (("input_dim", input_dim), ("latent_dim", latent_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        settings = arm_settings(arm, base_variance)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.arm = arm
        self.base_variance = float(base_variance)
        self.seed = int(seed)
        self.alpha = settings["alpha"]
        self.b = settings["b"]
        self.residual_variance = settings["residual_variance"]
        self.observation_variance = settings["observation_variance"]

        def dense(width, offset, name, activation=None):
            return tf.keras.layers.Dense(
                width, activation=activation, name=name,
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
        self.residual_head = dense(self.input_dim, 6, "residual_mean")

        # Build all weights (and mark the Model built for weights-only I/O).
        self(tf.zeros((1, self.input_dim), dtype=self.compute_dtype))
        if self.b != 1.0:
            self.residual_head.kernel.assign(self.residual_head.kernel * self.b)
            self.residual_head.bias.assign(self.residual_head.bias * self.b)

    def _tensor(self, value, width):
        value = tf.cast(tf.convert_to_tensor(value), self.compute_dtype)
        tf.debugging.assert_rank_at_least(value, 1)
        tf.debugging.assert_equal(tf.shape(value)[-1], width,
                                  message="incorrect final dimension")
        return value

    def encode(self, x):
        """Return q(z|x) mean and bounded log-variance, preserving leading dims."""
        x = self._tensor(x, self.input_dim)
        hidden = tf.reshape(x, (-1, self.input_dim))
        for layer in self.encoder_hidden:
            hidden = layer(hidden)
        shape = tf.concat([tf.shape(x)[:-1], [self.latent_dim]], axis=0)
        mean = tf.reshape(self.mean_head(hidden), shape)
        log_variance = tf.reshape(self.logvar_head(hidden), shape)
        return mean, tf.clip_by_value(log_variance, -12.0, 12.0)

    def decode_residual(self, z):
        """Return f(z) for z shaped (..., latent_dim), including a single z."""
        z = self._tensor(z, self.latent_dim)
        hidden = tf.reshape(z, (-1, self.latent_dim))
        for layer in self.decoder_hidden:
            hidden = layer(hidden)
        shape = tf.concat([tf.shape(z)[:-1], [self.input_dim]], axis=0)
        return tf.reshape(self.residual_head(hidden), shape)

    def decode_mean(self, z):
        """Return the conditional mean in observation coordinates, f(z)/b."""
        return self.decode_residual(z) / self.b

    def call(self, x, training=None):
        """Deterministic self-reconstruction using the posterior mean code."""
        x = self._tensor(x, self.input_dim)
        mean, _ = self.encode(x)
        return self.alpha * x + self.decode_residual(mean)

    def _residual_log_density(self, difference):
        squared_error = tf.reduce_sum(tf.square(difference), axis=-1)
        log_normalizer = self.input_dim * (
            math.log(2.0 * math.pi) + math.log(self.residual_variance))
        log_jacobian = self.input_dim * math.log(self.b)
        return -0.5 * (squared_error / self.residual_variance + log_normalizer) + log_jacobian

    def log_likelihood(self, x, z):
        """Normalized log p(x|z), summed over dimensions in nats.

        Leading dimensions broadcast, e.g. x=(N, d), z=(K, N, latent_dim)
        returns (K, N). This conditional density is not a marginal estimate.
        """
        x = self._tensor(x, self.input_dim)
        return self._residual_log_density(self.b * x - self.decode_residual(z))

    def loss_terms(self, x, epsilon):
        """Per-example one-sample negative ELBO, with analytic standard-normal KL.

        epsilon must have the encoder mean's shape and supplies standard-normal
        noise externally. Both MSE diagnostics use the same sampled z as NLL;
        MSE averages dimensions, whereas NLL and KL sum their dimensions.
        negative_elbo = nll + kl (beta=1); corrected_mse = self_mse / b**2.
        """
        x = self._tensor(x, self.input_dim)
        mean, log_variance = self.encode(x)
        epsilon = self._tensor(epsilon, self.latent_dim)
        tf.debugging.assert_equal(tf.rank(epsilon), tf.rank(mean),
                                  message="epsilon must match the encoder mean rank")
        tf.debugging.assert_equal(tf.shape(epsilon), tf.shape(mean),
                                  message="epsilon must match the encoder mean shape")
        z = mean + tf.exp(0.5 * log_variance) * epsilon
        difference = self.b * x - self.decode_residual(z)
        nll = -self._residual_log_density(difference)
        kl = 0.5 * tf.reduce_sum(
            tf.square(mean) + tf.exp(log_variance) - 1.0 - log_variance, axis=-1)
        self_mse = tf.reduce_mean(tf.square(difference), axis=-1)
        return {
            "negative_elbo": nll + kl,
            "nll": nll,
            "kl": kl,
            "self_mse": self_mse,
            "corrected_mse": self_mse / self.b**2,
        }

    def map_from(self, source):
        """Copy a matched distribution, rescaling only the residual head.

        Requires equal dimensions and observation variance, up to float-roundoff
        tolerance. Assign real network weights, never model-wide set_weights:
        the target retains its own arm/noise settings. This is a checkpoint
        control, not a claim of matched optimizer states or training dynamics.
        """
        if not isinstance(source, GaussianVAE):
            raise TypeError("source must be a GaussianVAE")
        if (self.input_dim, self.latent_dim) != (source.input_dim, source.latent_dim):
            raise ValueError("source and target dimensions must match")
        if not math.isclose(self.observation_variance, source.observation_variance,
                            rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("source and target observation variance must match")
        target_layers = (*self.encoder_hidden, self.mean_head, self.logvar_head,
                         *self.decoder_hidden)
        source_layers = (*source.encoder_hidden, source.mean_head, source.logvar_head,
                         *source.decoder_hidden)
        for target_layer, source_layer in zip(target_layers, source_layers):
            for target_weight, source_weight in zip(target_layer.weights, source_layer.weights):
                target_weight.assign(source_weight)
        ratio = self.b / source.b
        self.residual_head.kernel.assign(source.residual_head.kernel * ratio)
        self.residual_head.bias.assign(source.residual_head.bias * ratio)
        return self
