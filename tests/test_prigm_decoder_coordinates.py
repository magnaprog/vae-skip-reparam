"""Storage-contract-3 model regressions; TensorFlow is required to execute them."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
import h5py

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prigm_decoder_coordinates import (BASE_FAST, HEAD_RATE, CONDITIONS, STORAGE_CONTRACT,
                            BypassVAE, CanonicalBypassOptimizer,
                            condition_settings, map_bypass)


OBS = 0.0625


def _model(condition, seed=42, dtype="float32"):
    return BypassVAE(input_dim=12, latent_dim=3, condition=condition,
                     observation_variance=OBS, seed=seed, dtype=dtype)


def _assert_exact_tensors(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.dtype == b.dtype
        assert np.isfinite(np.asarray(a)).all()
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _assert_adam_state_equal(left, right):
    for group in ("body", "head"):
        a, b = left._optimizers[group], right._optimizers[group]
        _assert_exact_tensors(a.variables(), b.variables())


@pytest.fixture
def common_draws():
    rng = np.random.default_rng(777)
    return (rng.standard_normal((6, 12)).astype(np.float32),
            rng.standard_normal((6, 3)).astype(np.float32),
            rng.standard_normal((4, 6, 3)).astype(np.float32))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition_settings_and_initial_storage(condition):
    settings = condition_settings(condition, OBS)
    model, base = _model(condition), _model("base")
    coordinates = "w" if condition in ("base", "canonical", BASE_FAST) else "r"
    scale = 1.0 if coordinates == "w" else model.b
    rate = model.b if condition == HEAD_RATE else (5.0 if condition == BASE_FAST else 1.0)
    assert STORAGE_CONTRACT == settings["storage_contract"] == model.storage_contract == 3
    assert settings["head_coordinates"] == model.head_coordinates == coordinates
    assert settings["head_scale"] == model.head_scale == scale
    assert settings["head_rate_multiplier"] == model.head_rate_multiplier == rate
    assert settings["residual_variance"] == model.b**2 * OBS
    assert model.alpha == (0.0 if condition in ("base", BASE_FAST) else
                           0.5 if condition == "raw_half" else 0.8)
    _assert_exact_tensors(model.trainable_variables[:-2], base.trainable_variables[:-2])
    _assert_exact_tensors(model.output_head.weights,
                         [v * scale for v in base.output_head.weights])
    before = [tf.identity(v) for v in model.weights]
    model(tf.zeros((2, 12)))
    model(tf.zeros((2, 12)))
    _assert_exact_tensors(model.weights, before)  # No repeated initialization scaling.


@pytest.mark.parametrize("source_condition,target_condition", [
    ("base", "canonical"), ("base", "raw"), ("raw", "base"),
    ("raw", "canonical"), ("canonical", "raw"), ("base", "raw_half"),
    ("raw_half", "canonical"), ("raw", HEAD_RATE), ("base", BASE_FAST),
])
def test_mapping_preserves_mean_density_and_log_weights(
        source_condition, target_condition, common_draws):
    source, target = _model(source_condition, seed=99), _model(target_condition, seed=1)
    # Nonzero bias makes a missing bias rescaling observable.
    source.output_head.bias.assign(tf.linspace(-0.3, 0.3, source.input_dim))
    assert map_bypass(source, target) is target
    ratio = target.head_scale / source.head_scale
    _assert_exact_tensors(target.output_head.weights,
                         [v * ratio for v in source.output_head.weights])
    x, epsilon, z = common_draws
    encoder = source.encode(x)
    _assert_exact_tensors(encoder, target.encode(x))
    mean, log_variance = encoder
    # Identical sampled z and q(z|x), not independently sampled importance weights.
    log_prior = -0.5 * tf.reduce_sum(np.log(2.0 * np.pi) + tf.square(z), axis=-1)
    log_q = -0.5 * tf.reduce_sum(
        np.log(2.0 * np.pi) + log_variance
        + tf.square(z - mean) * tf.exp(-log_variance), axis=-1)
    np.testing.assert_allclose(source.decode_mean(z), target.decode_mean(z),
                               rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(source.log_likelihood(x, z), target.log_likelihood(x, z),
                               rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(source.log_likelihood(x, z) + log_prior - log_q,
                               target.log_likelihood(x, z) + log_prior - log_q,
                               rtol=2e-6, atol=2e-5)
    a, b = source.loss_terms(x, epsilon), target.loss_terms(x, epsilon)
    for key in ("nll", "kl", "negative_elbo", "corrected_mse"):
        np.testing.assert_allclose(a[key], b[key], rtol=2e-6, atol=2e-5)
    for model, terms in ((source, a), (target, b)):
        _assert_exact_tensors([terms["self_mse"]],
                             [model.b**2 * terms["corrected_mse"]])
        _assert_exact_tensors([model.decode_residual(z)], [model.b * model.decode_mean(z)])
        _assert_exact_tensors([model(x)], [model.alpha * x + model.b * model.decode_mean(mean)])
        expected_ll = -0.5 * tf.reduce_sum(
            tf.square(x - model.decode_mean(z)) / OBS + np.log(2.0 * np.pi * OBS), axis=-1)
        np.testing.assert_allclose(model.log_likelihood(x, z), expected_ll,
                                   rtol=2e-6, atol=2e-5)
        sampled_z = mean + tf.exp(0.5 * log_variance) * epsilon
        _assert_exact_tensors([terms["nll"]], [-model.log_likelihood(x, sampled_z)])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("clip", [False, True])
def test_base_canonical_exact_twenty_step_trajectory(dtype, compiled, clip, common_draws):
    # No mapping/copying can hide incorrect canonical initialization here.
    base, canonical = _model("base", seed=31415, dtype=dtype), _model(
        "canonical", seed=31415, dtype=dtype)
    opt_b = CanonicalBypassOptimizer(base, clip=clip)
    opt_c = CanonicalBypassOptimizer(canonical, clip=clip)
    assert all(v.dtype == tf.as_dtype(dtype) for v in base.trainable_variables)
    _assert_exact_tensors(base.trainable_variables, canonical.trainable_variables)
    _assert_adam_state_equal(opt_b, opt_c)

    def make_step(model, optimizer):
        def step(x, epsilon):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(model.loss_terms(x, epsilon)["negative_elbo"])
            gradients = tape.gradient(loss, model.trainable_variables)
            norm = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return (loss, norm, tf.identity(optimizer.steps),
                    tf.identity(optimizer.clipping_scale), tf.identity(optimizer.clipping_active))
        return tf.function(step) if compiled else step

    step_b, step_c = make_step(base, opt_b), make_step(canonical, opt_c)
    x, epsilon, z = [tf.constant(draw, dtype=dtype) for draw in common_draws]
    for i in range(20):
        noise = epsilon * (1.0 + i / 100.0)
        _assert_exact_tensors(step_b(x, noise), step_c(x, noise))
        _assert_exact_tensors(base.trainable_variables, canonical.trainable_variables)
        _assert_adam_state_equal(opt_b, opt_c)
        for optimizer in (opt_b, opt_c):
            assert int(optimizer.steps.numpy()) == i + 1
            assert int(optimizer._optimizers["head"].iterations.numpy()) == i + 1
            if not clip:
                assert float(optimizer.clipping_scale.numpy()) == 1.0
                assert not bool(optimizer.clipping_active.numpy())
        _assert_exact_tensors([base.decode_mean(z), base.log_likelihood(x, z)],
                             [canonical.decode_mean(z), canonical.log_likelihood(x, z)])
        a, b = base.loss_terms(x, noise), canonical.loss_terms(x, noise)
        for key in ("nll", "kl", "negative_elbo", "corrected_mse"):
            _assert_exact_tensors([a[key]], [b[key]])
        _assert_exact_tensors([b["self_mse"]], [canonical.b**2 * a["corrected_mse"]])


@pytest.mark.parametrize("control_condition,rate_condition,multiplier", [
    ("raw", HEAD_RATE, 0.2), ("base", BASE_FAST, 5.0),
])
@pytest.mark.parametrize("clip", [False, True])
def test_head_rate_actual_body_and_head_deltas(control_condition, rate_condition, multiplier, clip):
    control, rate = _model(control_condition), _model(rate_condition)
    # Matched states, including both bias and kernel, without subtraction cancellation.
    for model in (control, rate):
        for v in model.output_head.weights:
            v.assign(tf.zeros_like(v))
    _assert_exact_tensors(control.trainable_variables, rate.trainable_variables)
    opt_control = CanonicalBypassOptimizer(control, clip=clip)
    opt_rate = CanonicalBypassOptimizer(rate, clip=clip)
    assert opt_rate.epsilon == opt_control.epsilon == 1e-7
    assert opt_rate.global_norm == opt_control.global_norm == 10.0
    assert float(opt_rate._optimizers["body"].learning_rate) == pytest.approx(1e-3)
    assert float(opt_rate._optimizers["head"].learning_rate) == pytest.approx(1e-3 * multiplier)
    gradients = [tf.fill(v.shape, tf.cast((-1.0)**i * (i + 1) / 8.0, v.dtype))
                 for i, v in enumerate(control.trainable_variables)]
    before = [tf.identity(v) for v in control.trainable_variables]
    expected_norm = tf.linalg.global_norm(gradients)
    assert float(expected_norm) > 10.0  # clip=True really exercises active clipping.
    for model, optimizer in ((control, opt_control), (rate, opt_rate)):
        norm = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        _assert_exact_tensors([norm], [expected_norm])
        assert bool(optimizer.clipping_active.numpy()) == clip
        assert int(optimizer.steps.numpy()) == int(optimizer._optimizers["head"].iterations.numpy()) == 1
    _assert_adam_state_equal(opt_control, opt_rate)  # Learning rate does not change moments.
    head_refs = {v.ref() for v in rate.output_head.weights}
    for initial, a, b in zip(before, control.trainable_variables, rate.trainable_variables):
        delta_control, delta_rate = a - initial, b - initial
        assert np.any(np.asarray(delta_control) != 0.0)
        if b.ref() in head_refs:
            np.testing.assert_allclose(delta_rate, multiplier * delta_control, rtol=2e-6, atol=0.0)
        else:
            _assert_exact_tensors([delta_control], [delta_rate])


@pytest.mark.parametrize("clip", [False, True])
def test_common_clipping_arithmetic_and_executed_steps(clip):
    model = _model("base", dtype="float64")
    optimizer = CanonicalBypassOptimizer(model, clip=clip)
    reference = [tf.Variable(v) for v in model.trainable_variables]
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-7, amsgrad=False)
    adam.build(reference)

    @tf.function
    def step(magnitude):
        gradients = [tf.zeros_like(v) for v in model.trainable_variables]
        # A 3-4-5 norm spanning body and head, with no per-group clipping.
        for index, fraction in ((0, 0.6), (len(gradients) - 1, 0.8)):
            variable = model.trainable_variables[index]
            flat = tf.concat([tf.reshape(magnitude * fraction, (1,)),
                              tf.zeros((variable.shape.num_elements() - 1,), dtype=variable.dtype)], 0)
            gradients[index] = tf.reshape(flat, variable.shape)
        norm = tf.linalg.global_norm(gradients)
        # Declared arithmetic: cap / max(norm, cap), exactly 1 below/at cap.
        scale = 10.0 / tf.maximum(norm, 10.0) if clip else tf.ones_like(norm)
        returned_norm = optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        adam.apply_gradients(zip([g * scale for g in gradients], reference))
        return returned_norm, norm, scale, tf.identity(optimizer.steps)

    for i, magnitude in enumerate((0.0, 5.0, 10.0, 50.0, 1e-14, 20.0, 0.0), start=1):
        returned, norm, expected_scale, count = step(tf.constant(magnitude, tf.float64))
        _assert_exact_tensors([returned, optimizer.clipping_scale], [norm, expected_scale])
        assert int(count.numpy()) == int(optimizer._optimizers["head"].iterations.numpy()) == i
        assert bool(optimizer.clipping_active.numpy()) == (clip and float(norm) > 10.0)
        if not clip or float(norm) <= 10.0:
            assert float(optimizer.clipping_scale.numpy()) == 1.0
        _assert_exact_tensors(model.trainable_variables, reference)


def _tf215_adam_first_delta(gradient, adam):
    """Keras v2.15.0 Adam.update_step, including its actual casts/epsilon hat.

    delta = -lr * sqrt(1-beta2**t)/(1-beta1**t) * m / (sqrt(v)+epsilon).
    At t=1 from zero slots, m=(1-beta1)*g and v=(1-beta2)*g**2.
    """
    dtype = gradient.dtype
    t = tf.constant(1, dtype=dtype)
    alpha = (tf.cast(adam.learning_rate, dtype)
             * tf.sqrt(1 - tf.pow(tf.cast(adam.beta_2, dtype), t))
             / (1 - tf.pow(tf.cast(adam.beta_1, dtype), t)))
    m = gradient * (1 - adam.beta_1)
    v = tf.square(gradient) * (1 - adam.beta_2)
    return -(m * alpha) / (tf.sqrt(v) + adam.epsilon)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_raw_fixed_head_gradient_one_step_epsilon_aware(dtype):
    base, raw = _model("base", dtype=dtype), _model("raw", dtype=dtype)
    optimizers = [CanonicalBypassOptimizer(model, clip=False) for model in (base, raw)]
    observation_deltas = []
    head_gradients = []
    for model, optimizer in zip((base, raw), optimizers):
        head_refs = {v.ref() for v in model.output_head.weights}
        gradients, expected = [], []
        for variable in model.trainable_variables:
            gradient = tf.zeros_like(variable)
            if variable.ref() in head_refs:
                variable.assign(tf.zeros_like(variable))
                pattern = tf.constant([1e-8, -1e-6, 1e-3, -0.25], dtype=dtype)
                gradient_w = tf.reshape(tf.tile(pattern, [variable.shape.num_elements() // 4]),
                                        variable.shape)
                gradient = gradient_w / model.head_scale
                expected.append(_tf215_adam_first_delta(gradient, optimizer._optimizers["head"]))
                if model is base:
                    head_gradients.append(gradient_w)
            gradients.append(gradient)
        body_before = [tf.identity(v) for v in optimizer._body_weights]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        _assert_exact_tensors(optimizer._body_weights, body_before)
        tolerance = 2e-6 if dtype == "float32" else 1e-12
        for actual, predicted in zip(model.output_head.weights, expected):
            np.testing.assert_allclose(actual, predicted, rtol=tolerance, atol=0.0)
        observation_deltas.append([v / model.head_scale for v in model.output_head.weights])
        assert float(optimizer.clipping_scale.numpy()) == 1.0
        assert not bool(optimizer.clipping_active.numpy())
    for g, delta_base, delta_raw in zip(head_gradients, *observation_deltas):
        sqrt_v = tf.sqrt(tf.square(g) * (1 - optimizers[0].beta_2))
        epsilon = optimizers[0].epsilon
        expected_ratio = (sqrt_v + epsilon) / (raw.b * (sqrt_v + raw.b * epsilon))
        np.testing.assert_allclose(delta_raw / delta_base, expected_ratio, rtol=tolerance, atol=0.0)
        # Small gradients must expose epsilon dependence, not a blanket 5x claim.
        assert float(tf.reshape(expected_ratio, (-1,))[0]) > 20.0


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("suffix", [".h5", ".weights.h5"])
def test_weight_contract_roundtrip(tmp_path, condition, suffix, common_draws):
    source, target = _model(condition, seed=99), _model(condition, seed=1)
    source.output_head.bias.assign(tf.linspace(-0.2, 0.2, source.input_dim))
    path = tmp_path / ("weights" + suffix)
    source.save_weights(path)
    with h5py.File(path, "r") as checkpoint:
        assert checkpoint.attrs["storage_contract"] == 3
        assert checkpoint.attrs["head_coordinates"] == source.head_coordinates
        assert checkpoint.attrs["head_scale"] == source.head_scale
        assert checkpoint.attrs["observation_variance"] == OBS
    target.load_weights(path)
    _assert_exact_tensors(source.weights, target.weights)
    x, _, z = common_draws
    _assert_exact_tensors([source.decode_mean(z), source.log_likelihood(x, z)],
                         [target.decode_mean(z), target.log_likelihood(x, z)])


@pytest.mark.parametrize("attribute,value", [
    ("storage_contract", None), ("storage_contract", 2),
    ("head_coordinates", "unknown_coordinates"), ("head_scale", 0.25),
    ("observation_variance", 2 * OBS),
])
def test_wrong_contract_rejected_before_any_assignment(tmp_path, attribute, value):
    source, target = _model("raw", seed=99), _model("raw", seed=1)
    path = tmp_path / "incompatible.h5"
    source.save_weights(path)
    with h5py.File(path, "r+") as checkpoint:
        if value is None:
            del checkpoint.attrs[attribute]
        else:
            checkpoint.attrs[attribute] = value
    before = [tf.identity(v) for v in target.weights]
    with pytest.raises(ValueError, match=attribute):
        target.load_weights(path)
    _assert_exact_tensors(target.weights, before)


@pytest.mark.parametrize("target_condition", ["canonical", "raw_half"])
def test_other_head_storage_requires_explicit_mapping(tmp_path, target_condition):
    source, target = _model("raw", seed=99), _model(target_condition, seed=1)
    path = tmp_path / "raw.h5"
    source.save_weights(path)
    before = [tf.identity(v) for v in target.weights]
    with pytest.raises(ValueError, match="incompatible weight storage contract"):
        target.load_weights(path)
    _assert_exact_tensors(target.weights, before)


def test_untagged_same_shape_weights_rejected(tmp_path):
    source, target = _model("raw", seed=99), _model("canonical", seed=1)
    path = tmp_path / "untagged.h5"
    # Deliberately use Keras directly to produce an untagged weight file.
    tf.keras.Model.save_weights(source, str(path))
    with h5py.File(path, "r") as checkpoint:
        assert "storage_contract" not in checkpoint.attrs
    # Prove this fixture really is accepted by unsafe Keras positional loading.
    unsafe = _model("canonical", seed=2)
    tf.keras.Model.load_weights(unsafe, str(path))
    _assert_exact_tensors(source.weights, unsafe.weights)
    before = [tf.identity(v) for v in target.weights]
    with pytest.raises(ValueError, match="storage_contract"):
        target.load_weights(path)
    _assert_exact_tensors(target.weights, before)
    # An overwrite refusal must not add coordinate attributes to an existing untagged file.
    with pytest.raises(FileExistsError):
        target.save_weights(path, overwrite=False)
    with h5py.File(path, "r") as checkpoint:
        assert "storage_contract" not in checkpoint.attrs


def test_tensorflow_checkpoint_prefix_is_not_a_contract_weight_file(tmp_path):
    model = _model("base")
    path = tmp_path / "weights.ckpt"
    with pytest.raises(ValueError, match="requires an HDF5 weight file"):
        model.save_weights(path)
    with pytest.raises(ValueError, match="requires an HDF5 weight file"):
        model.load_weights(path)
    assert list(tmp_path.iterdir()) == []
