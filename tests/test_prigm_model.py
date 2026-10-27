"""CPU checks of the PriGM probability model, mapping, and Keras weights I/O."""

import sys
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prigm_model import GaussianVAE, condition_settings


CONDITIONS = ("no_skip_base", "skip_base", "no_skip_high", "skip_high")


@pytest.fixture(autouse=True)
def cpu_device():
    with tf.device("/CPU:0"):
        yield


@pytest.mark.parametrize("condition,alpha,residual,observation", [
    ("no_skip_base", 0.0, 1.0, 1.0),
    ("skip_base", 0.8, 0.04, 1.0),
    ("no_skip_high", 0.0, 25.0, 25.0),
    ("skip_high", 0.8, 1.0, 25.0),
])
@pytest.mark.parametrize("base_variance", [1.0, 0.37, 0.0001])
def test_condition_settings(condition, alpha, residual, observation, base_variance):
    assert condition_settings(condition, base_variance) == pytest.approx({
        "alpha": alpha,
        "b": 1.0 - alpha,
        "residual_variance": residual * base_variance,
        "observation_variance": observation * base_variance,
    }, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("variance", [0.0, -1.0, np.nan, np.inf, -np.inf, None])
def test_invalid_noise_rejected(variance):
    with pytest.raises(ValueError, match="finite and positive"):
        condition_settings("no_skip_base", variance)


def test_invalid_condition_and_dimensions_rejected():
    with pytest.raises(ValueError, match="condition"):
        condition_settings("unknown", 1.0)
    with pytest.raises(ValueError, match="derived variances"):
        condition_settings("skip_high", 1e308)
    for arguments in ({"input_dim": 0}, {"latent_dim": -1}, {"input_dim": 2.5}):
        with pytest.raises(ValueError, match="positive integer"):
            GaussianVAE(**arguments)


@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_letter_conditions_are_rejected(condition):
    with pytest.raises(ValueError, match="condition must be one of no_skip_base"):
        condition_settings(condition, 1.0)
    with pytest.raises(ValueError, match="condition must be one of no_skip_base"):
        GaussianVAE(condition=condition)


def test_old_model_api_is_not_an_alias():
    import prigm_model

    assert not hasattr(prigm_model, "arm_settings")
    with pytest.raises(TypeError, match="arm"):
        GaussianVAE(arm="no_skip_base")


def test_default_architecture():
    model = GaussianVAE()
    assert isinstance(model, tf.keras.Model)
    assert (model.input_dim, model.latent_dim, model.condition) == (192, 16, "no_skip_base")
    assert not hasattr(model, "arm")
    assert [layer.units for layer in model.encoder_hidden] == [128, 64]
    assert [layer.units for layer in model.decoder_hidden] == [64, 128]
    for layer in (*model.encoder_hidden, *model.decoder_hidden):
        assert layer.activation == tf.keras.activations.relu
    for layer in (model.mean_head, model.logvar_head, model.residual_head):
        assert layer.activation == tf.keras.activations.linear
    assert len(model.trainable_weights) == 14
    assert not model.non_trainable_weights
    assert not model.losses
    assert model(tf.zeros((2, 192))).shape == (2, 192)


@pytest.mark.parametrize("seed", [42, 1729])
def test_all_conditions_share_initial_observation_means_and_other_weights(seed):
    models = {condition: GaussianVAE(input_dim=5, latent_dim=3, condition=condition, seed=seed)
              for condition in CONDITIONS}
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 5)).astype("float32")
    z = rng.normal(size=(4, 3)).astype("float32")
    reference = models["no_skip_base"]
    for model in models.values():
        np.testing.assert_allclose(model.decode_mean(z), reference.decode_mean(z),
                                   rtol=3e-6, atol=1e-6)
        for actual, expected in zip(model.encode(x), reference.encode(x)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(model.weights, reference.weights):
            is_head = (actual is model.residual_head.kernel
                       or actual is model.residual_head.bias)
            scale = model.b if is_head else 1.0
            np.testing.assert_allclose(actual, expected.numpy() * scale, rtol=1e-7, atol=1e-7)
        mean, _ = model.encode(x)
        np.testing.assert_allclose(model(x, training=True),
                                   model.alpha * x + model.decode_residual(mean), atol=1e-6)
        before = [weight.numpy().copy() for weight in model.weights]
        model(x)
        model(x)
        for actual, expected in zip(model.weights, before):
            np.testing.assert_array_equal(actual, expected)

    # Construction order and unrelated random draws cannot change seeded weights.
    GaussianVAE(input_dim=5, latent_dim=3, seed=seed + 1)
    tf.random.normal((100,))
    repeated = GaussianVAE(input_dim=5, latent_dim=3, seed=seed)
    for actual, expected in zip(repeated.weights, reference.weights):
        np.testing.assert_array_equal(actual, expected)
    different = GaussianVAE(input_dim=5, latent_dim=3, seed=seed + 1)
    assert not np.array_equal(different.mean_head.kernel, reference.mean_head.kernel)


def test_log_variance_bounds_constrain_posterior_not_data_or_mean():
    model = GaussianVAE(input_dim=3, latent_dim=2, condition="skip_high")
    model.mean_head.kernel.assign(tf.zeros_like(model.mean_head.kernel))
    model.mean_head.bias.assign([20.0, -20.0])
    model.logvar_head.kernel.assign(tf.zeros_like(model.logvar_head.kernel))
    model.logvar_head.bias.assign([-100.0, 100.0])
    model.residual_head.kernel.assign(tf.zeros_like(model.residual_head.kernel))
    model.residual_head.bias.assign(tf.zeros_like(model.residual_head.bias))
    x = tf.constant([[50.0, -40.0, 0.5]])
    mean, log_variance = tf.function(model.encode)(x)
    np.testing.assert_array_equal(mean, [[20.0, -20.0]])
    np.testing.assert_array_equal(log_variance, [[-12.0, 12.0]])
    np.testing.assert_allclose(model(x), 0.8 * x)
    for term in model.loss_terms(x, tf.ones((1, 2))).values():
        assert np.all(np.isfinite(term))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_likelihood_matches_manual_normalized_gaussian(condition):
    model = GaussianVAE(input_dim=3, latent_dim=2, condition=condition, base_variance=0.7)
    # Nonzero bias checks affine, not just linear, mean transformations.
    model.residual_head.bias.assign([0.1, -0.2, 0.3])
    x = np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.2]], dtype="float32")
    z = np.array([[[0.2, -0.3], [0.4, 0.6]],
                  [[-0.1, 0.7], [0.5, -0.8]]], dtype="float32")
    residual_mean = model.decode_residual(z).numpy().astype("float64")
    difference = model.b * x.astype("float64") - residual_mean
    expected = -0.5 * np.sum(
        difference**2 / model.residual_variance
        + np.log(2 * np.pi * model.residual_variance), axis=-1,
    ) + model.input_dim * np.log(model.b)
    observation_mean = residual_mean / model.b
    observation_expected = -0.5 * np.sum(
        (x.astype("float64") - observation_mean)**2 / model.observation_variance
        + np.log(2 * np.pi * model.observation_variance), axis=-1,
    )
    actual = tf.function(model.log_likelihood)(x, z)
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual, observation_expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("leading_shape", [(), (4,), (2, 4), (2, 3, 4)])
def test_decoder_preserves_arbitrary_leading_dimensions(leading_shape):
    model = GaussianVAE(input_dim=5, latent_dim=2, condition="skip_base")
    z = tf.reshape(tf.range(int(np.prod(leading_shape)) * 2, dtype=tf.float32),
                   leading_shape + (2,)) / 10.0
    actual = tf.function(model.decode_residual)(z)
    expected = model.decode_residual(tf.reshape(z, (-1, 2)))
    assert actual.shape == leading_shape + (5,)
    np.testing.assert_allclose(actual, tf.reshape(expected, actual.shape), atol=1e-7)
    np.testing.assert_allclose(model.decode_mean(z), actual / model.b, atol=1e-7)
    density = model.log_likelihood(tf.zeros((5,)), z)
    assert density.shape == leading_shape


@pytest.mark.parametrize("source_condition,target_condition", [
    ("no_skip_base", "skip_base"), ("skip_base", "no_skip_base"),
    ("no_skip_high", "skip_high"), ("skip_high", "no_skip_high"),
])
def test_mapping_copies_actual_weights_and_preserves_distribution(source_condition, target_condition):
    source = GaussianVAE(input_dim=5, latent_dim=3, condition=source_condition, base_variance=0.3)
    target = GaussianVAE(input_dim=5, latent_dim=3, condition=target_condition,
                         base_variance=0.3, seed=99)
    rng = np.random.default_rng(123)
    for weight in source.weights:
        weight.assign_add(rng.normal(scale=0.02, size=weight.shape).astype("float32"))
    source_before = [weight.numpy().copy() for weight in source.weights]
    settings_before = (target.condition, target.alpha, target.b, target.residual_variance,
                       target.observation_variance)
    assert target.map_from(source) is target
    assert settings_before == (target.condition, target.alpha, target.b, target.residual_variance,
                               target.observation_variance)
    for actual, expected in zip(source.weights, source_before):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(target.residual_head.bias,
                               source.residual_head.bias * (target.b / source.b), atol=1e-7)
    x = rng.normal(size=(4, 5)).astype("float32")
    z = rng.normal(size=(7, 4, 3)).astype("float32")
    epsilon = rng.normal(size=(4, 3)).astype("float32")
    for actual, expected in zip(target.encode(x), source.encode(x)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(target.decode_mean(z), source.decode_mean(z),
                               rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(target.log_likelihood(x, z), source.log_likelihood(x, z),
                               rtol=3e-6, atol=5e-6)
    source_terms = source.loss_terms(x, epsilon)
    target_terms = target.loss_terms(x, epsilon)
    for name in ("negative_elbo", "nll", "kl", "corrected_mse"):
        np.testing.assert_allclose(target_terms[name], source_terms[name],
                                   rtol=3e-6, atol=5e-6)
    np.testing.assert_allclose(target_terms["self_mse"],
                               source_terms["self_mse"] * (target.b / source.b)**2,
                               rtol=3e-6, atol=1e-6)


@pytest.mark.parametrize("arguments", [
    {"input_dim": 6, "latent_dim": 3, "condition": "skip_base"},
    {"input_dim": 5, "latent_dim": 2, "condition": "skip_base"},
    {"input_dim": 5, "latent_dim": 3, "condition": "skip_high"},
    {"input_dim": 5, "latent_dim": 3, "condition": "skip_base", "base_variance": 1.001},
])
def test_incompatible_mapping_rejected_before_assigning_weights(arguments):
    source = GaussianVAE(input_dim=5, latent_dim=3)
    target = GaussianVAE(**arguments)
    before = [weight.numpy().copy() for weight in target.weights]
    with pytest.raises(ValueError, match="dimensions|observation variance"):
        target.map_from(source)
    for actual, expected in zip(target.weights, before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_loss_uses_explicit_noise_analytic_kl_and_dimension_reductions(condition):
    model = GaussianVAE(input_dim=3, latent_dim=2, condition=condition, base_variance=0.4)
    model.mean_head.kernel.assign(tf.zeros_like(model.mean_head.kernel))
    model.mean_head.bias.assign([0.5, -0.25])
    model.logvar_head.kernel.assign(tf.zeros_like(model.logvar_head.kernel))
    model.logvar_head.bias.assign(np.log([0.5, 2.0]).astype("float32"))
    x = np.array([[0.4, -0.2, 0.7], [-0.3, 0.8, 0.2]], dtype="float32")
    epsilon = np.array([[0.2, -0.4], [0.7, 0.3]], dtype="float32")
    z = np.array([0.5, -0.25]) + np.sqrt([0.5, 2.0]) * epsilon
    residual = model.decode_residual(z).numpy().astype("float64")
    difference = model.b * x.astype("float64") - residual
    expected_mse = np.mean(difference**2, axis=-1)
    expected_kl = 0.5 * np.sum(np.square([0.5, -0.25]) + [0.5, 2.0]
                               - np.ones(2) - np.log([0.5, 2.0]))
    constant = 0.5 * 3 * np.log(2 * np.pi * model.residual_variance) - 3 * np.log(model.b)
    expected_nll = np.sum(difference**2, axis=-1) / (2 * model.residual_variance) + constant
    graph_loss = tf.function(model.loss_terms)
    terms = graph_loss(x, epsilon)
    repeated = graph_loss(x, epsilon)
    assert set(terms) == {"negative_elbo", "nll", "kl", "self_mse", "corrected_mse"}
    expected = {"nll": expected_nll, "kl": np.full(2, expected_kl),
                "negative_elbo": expected_nll + expected_kl, "self_mse": expected_mse,
                "corrected_mse": expected_mse / model.b**2}
    for name, value in terms.items():
        assert value.shape == (2,)
        np.testing.assert_allclose(value, expected[name], rtol=3e-6, atol=2e-6)
        np.testing.assert_array_equal(value, repeated[name])
    # Equivalent raw summed-MSE coefficient is 2*sigma_r**2, not a second beta.
    np.testing.assert_allclose(
        2 * model.residual_variance * (terms["negative_elbo"] - constant),
        3 * terms["self_mse"] + 2 * model.residual_variance * terms["kl"],
        rtol=3e-6, atol=3e-6,
    )
    assert not model.losses
    assert not np.allclose(terms["nll"], model.loss_terms(x, np.zeros_like(epsilon))["nll"])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_graph_gradient_update_and_real_hdf5_weights_roundtrip(condition, tmp_path):
    model = GaussianVAE(input_dim=5, latent_dim=3, condition=condition, base_variance=0.2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    x = tf.random.stateless_normal((4, 5), seed=(13, 1))
    epsilon = tf.random.stateless_normal((4, 3), seed=(13, 2))
    before = [weight.numpy().copy() for weight in model.trainable_weights]

    @tf.function(input_signature=[tf.TensorSpec((None, 5), tf.float32),
                                 tf.TensorSpec((None, 3), tf.float32)])
    def update(batch, noise):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model.loss_terms(batch, noise)["negative_elbo"])
        gradients = tape.gradient(loss, model.trainable_weights)
        for gradient in gradients:
            assert gradient is not None
            tf.debugging.assert_all_finite(gradient, "nonfinite gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss

    assert np.isfinite(update(x, epsilon))
    assert int(optimizer.iterations.numpy()) == 1
    assert any(not np.array_equal(old, new) for old, new in zip(before, model.trainable_weights))
    path = tmp_path / f"{condition}.weights.h5"
    model.save_weights(str(path))
    with path.open("rb") as handle:
        assert handle.read(8) == b"\x89HDF\r\n\x1a\n"
    restored = GaussianVAE(input_dim=5, latent_dim=3, condition=condition, base_variance=0.2, seed=99)
    restored.load_weights(str(path))
    for actual, expected in zip(restored.weights, model.weights):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored(x), model(x))
    np.testing.assert_array_equal(restored.decode_mean(epsilon), model.decode_mean(epsilon))
    np.testing.assert_array_equal(restored.log_likelihood(x, epsilon), model.log_likelihood(x, epsilon))
    for name, expected in model.loss_terms(x, epsilon).items():
        np.testing.assert_array_equal(restored.loss_terms(x, epsilon)[name], expected)
    assert (restored.b, restored.residual_variance) == (model.b, model.residual_variance)


def test_shapes_are_validated_in_graph_execution():
    model = GaussianVAE(input_dim=5, latent_dim=3)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(model.decode_residual)(tf.zeros((3, 2)))
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(model.loss_terms)(tf.zeros((4, 5)), tf.zeros((1, 3)))
    # Equal axis lengths must not let a rank-1 noise vector broadcast silently.
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(model.loss_terms)(tf.zeros((3, 5)), tf.zeros((3,)))
