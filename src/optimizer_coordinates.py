"""Train and score final models for the 21415/2018 optimizer follow-up."""
import argparse
import hashlib
from itertools import product
import os
from pathlib import Path
import platform

import numpy as np

from experiment import atomic_json, read_json, sha256
from generative_evaluation import unconditional_score_streams
from optimizer_comparisons import (MODES, ROOT, SEEDS, STREAMS, VARIANCES,
                                   comparison_rows, stream_columns, write_csv)
from optimizer_intervals import WINDOWS, window_rows


PREPARED = ROOT / "data" / "dart_21415_2018.npz"
PREPARED_SHA = "b1a5f4f8aa0fd6188e493cf4298c53ac5ae321ed70a337b5c6a4e0d5adb1945e"
STEPS = 10000
METRICS = ("loss", "kl", "observation_gradient_norm", "residual_gradient_norm",
           "clipping_scale", "body_step_norm", "observation_head_step_norm", "stored_head_step_norm")


def batches(seed, weights, steps):
    rng = np.random.default_rng(seed + 50000)
    probabilities = weights / weights.sum()
    for _ in range(steps):
        indices = rng.choice(len(weights), size=64, replace=True, p=probabilities).astype(np.int32)
        noise = rng.standard_normal((64, 16)).astype(np.float32)
        yield indices, noise


def build_update(model, mode, train_x):
    import tensorflow as tf
    from decoder_coordinates import CanonicalBypassOptimizer

    condition, transformed, norm_coordinates = MODES[mode]
    if model.condition != condition:
        raise ValueError("Mode and model condition disagree")
    variables = model.trainable_variables
    heads = {v.ref() for v in model.output_head.trainable_variables}
    is_head = [v.ref() in heads for v in variables]
    external_clip = model.head_coordinates == "w" and norm_coordinates == "residual"
    optimizer = CanonicalBypassOptimizer(
        model, global_norm=10.0, clip=not external_clip,
        head_learning_rate=0.001 / model.b if transformed else None,
        head_epsilon=1e-7 * model.b if transformed else None)

    @tf.function
    def update(indices, noise):
        with tf.GradientTape() as tape:
            terms = model.loss_terms(tf.gather(train_x, indices), noise)
            loss = tf.reduce_mean(terms["negative_elbo"])
        gradients = tape.gradient(loss, variables)
        before = [v.read_value() for v in variables]
        if model.head_coordinates == "r":
            obs_norm = tf.linalg.global_norm(
                [g * model.b if h else g for g, h in zip(gradients, is_head)])
            res_norm = optimizer.apply_gradients(zip(gradients, variables))
            scale = optimizer.clipping_scale.read_value()
        else:
            res_norm = tf.linalg.global_norm(
                [g / model.b if h else g for g, h in zip(gradients, is_head)])
            if external_clip:
                obs_norm = tf.linalg.global_norm(gradients)
                scale = 10.0 / tf.maximum(res_norm, 10.0)
                optimizer.apply_gradients(zip([scale * g for g in gradients], variables))
            else:
                obs_norm = optimizer.apply_gradients(zip(gradients, variables))
                scale = optimizer.clipping_scale.read_value()
        deltas = [v.read_value() - previous for v, previous in zip(variables, before)]
        body = tf.linalg.global_norm([d for d, h in zip(deltas, is_head) if not h])
        head = tf.linalg.global_norm([d for d, h in zip(deltas, is_head) if h])
        return tf.stack((loss, tf.reduce_mean(terms["kl"]), obs_norm, res_norm,
                         scale, body, head / model.head_scale, head))

    return update, optimizer


def make_model(variance, seed, mode):
    from decoder_coordinates import BypassVAE
    return BypassVAE(condition=MODES[mode][0], observation_variance=VARIANCES[variance], seed=seed)


def train(directory, train_x, weights, variance, seed, mode, steps, runtime):
    import tensorflow as tf

    directory.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(seed)
    model = make_model(variance, seed, mode)
    update, optimizer = build_update(model, mode, train_x)
    history = np.empty((steps, len(METRICS)), dtype=np.float32)
    indices_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    for step, (indices, noise) in enumerate(batches(seed, weights, steps)):
        indices_hash.update(indices.tobytes())
        noise_hash.update(noise.tobytes())
        history[step] = update(indices, noise).numpy()
        if not np.isfinite(history[step]).all():
            raise FloatingPointError(f"Nonfinite training metric at step {step + 1}")
    if int(optimizer.steps.numpy()) != steps:
        raise RuntimeError("Unexpected Adam step count")
    np.savez(directory / "history.npz", **dict(zip(METRICS, history.T)))
    checkpoint = directory / "final.weights.h5"
    model.save_weights(checkpoint, overwrite=False)
    sources = ("decoder_coordinates.py", "optimizer_coordinates.py", "optimizer_comparisons.py")
    atomic_json(directory / "complete.json", {
        "mode": mode, "seed": seed, "variance_label": variance, "steps": steps,
        "checkpoint": "final", "prepared_sha256": PREPARED_SHA, "weights_sha256": sha256(checkpoint),
        "indices_sha256": indices_hash.hexdigest(), "noise_sha256": noise_hash.hexdigest(),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in sources},
        "runtime": runtime})
    print(f"Completed {directory}", flush=True)


def evaluate_fit(checkpoint, data, variance, seed, mode):
    model = make_model(variance, seed, mode)
    model.load_weights(checkpoint)
    weights = data["eval_weights"]
    values = unconditional_score_streams(model, data["test"], weights / weights.sum(),
                                         VARIANCES[variance], samples=1024, seeds=STREAMS)
    digest = sha256(checkpoint)
    rows = []
    for geometry, key in (("joint", "energy"), ("centered", "centered_energy"),
                          ("increment", "increment_energy")):
        rows.append({"variance_label": variance, "observation_variance": VARIANCES[variance],
                     "mode": mode, "condition": model.condition, "seed": seed, "geometry": geometry,
                     **stream_columns(values[key + "_streams"]), "weights_sha256": digest})
    return rows, values["energy_per_window"]


def score(checkpoint, data, variance, seed, mode):
    rows, _ = evaluate_fit(checkpoint, data, variance, seed, mode)
    return rows


def summarize(fits, data, output):
    rows, training, per_window = [], {}, {}
    for variance, seed, mode in product(VARIANCES, SEEDS, MODES):
        directory = fits / f"{variance}-{seed}-{mode}"
        checkpoint = directory / "final.weights.h5"
        receipt = read_json(directory / "complete.json")
        expected = {"mode": mode, "seed": seed, "variance_label": variance, "steps": STEPS,
                    "checkpoint": "final", "prepared_sha256": PREPARED_SHA,
                    "weights_sha256": sha256(checkpoint)}
        if any(receipt[key] != value for key, value in expected.items()):
            raise ValueError(f"Incorrect final fit: {directory}")
        pairing = (receipt["indices_sha256"], receipt["noise_sha256"], receipt["runtime"])
        if training.setdefault(seed, pairing) != pairing:
            raise ValueError(f"Training streams or runtime differ within seed {seed}")
        fit_rows, window_scores = evaluate_fit(checkpoint, data, variance, seed, mode)
        per_window[variance, seed, mode] = window_scores
        rows.extend(fit_rows)
    comparisons = comparison_rows(rows, ROOT / "results" / "energy_terms_streams.csv")
    differences = window_rows(data["starts_test"], per_window)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / WINDOWS, differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", help="Use this NVIDIA GPU for this invocation (default CPU)")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train the grid, or select specific fits")
    fit.add_argument("--fits", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=STEPS, help="Shorter runs are diagnostic only")
    one = commands.add_parser("score", help="Evaluate one final weights file")
    one.add_argument("--weights", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    for cmd in (fit, one):
        cmd.add_argument("--seed", type=int, choices=SEEDS, required=cmd is one)
        cmd.add_argument("--variance", choices=VARIANCES, required=cmd is one)
        cmd.add_argument("--mode", choices=MODES, required=cmd is one)
    summary = commands.add_parser("summarize", help="Evaluate the complete grid and write scores and paired differences")
    summary.add_argument("--fits", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train" and not 1 <= args.steps <= STEPS:
        parser.error(f"--steps must be between 1 and {STEPS}")
    if sha256(PREPARED) != PREPARED_SHA:
        raise ValueError("Unexpected prepared 21415/2018 file")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid or "-1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import tensorflow as tf
    from experiment import gpu_runtime
    if args.gpu_uuid:
        device = gpu_runtime(args.gpu_uuid)["gpu_uuid"]
    else:
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        device = "CPU"
    runtime = {"device": device, "tensorflow": tf.__version__, "numpy": np.__version__,
               "python": platform.python_version(),
               "tf32": tf.config.experimental.tensor_float_32_execution_enabled()}
    with np.load(PREPARED, allow_pickle=False) as data:
        if args.command == "train":
            train_x = tf.constant(data["train"], dtype=tf.float32)
            weights = data["train_weights"]
            grid = product([args.variance] if args.variance else VARIANCES,
                           [args.seed] if args.seed else SEEDS, [args.mode] if args.mode else MODES)
            for variance, seed, mode in grid:
                tf.keras.backend.clear_session()
                train(args.fits / f"{variance}-{seed}-{mode}", train_x, weights,
                      variance, seed, mode, args.steps, runtime)
        elif args.command == "score":
            write_csv(args.output, score(args.weights, data, args.variance, args.seed, args.mode))
        else:
            summarize(args.fits, data, args.output)


if __name__ == "__main__":
    main()
