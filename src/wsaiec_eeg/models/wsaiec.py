"""WS-AIEC dynamic weighting and stacking primitives."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np


def softmax_weights(accuracies: Sequence[float] | np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(accuracies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("accuracies must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("accuracies must contain finite values in [0, 1]")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    logits = float(alpha) * values
    exponentials = np.exp(logits - np.max(logits))
    return exponentials / exponentials.sum()


def dynamic_weights(
    accuracies: Mapping[str, float],
    alpha: float,
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    names = list(order) if order is not None else list(accuracies)
    if not names or len(names) != len(set(names)):
        raise ValueError("order must contain at least one unique classifier name")
    missing = set(names) - set(accuracies)
    if missing:
        raise ValueError(f"Missing validation accuracies for {sorted(missing)}")
    values = softmax_weights([float(accuracies[name]) for name in names], alpha)
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def equation_9_weights(
    accuracies: Mapping[str, float],
    alpha: float,
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    return dynamic_weights(accuracies, alpha, order)


def validation_accuracies(
    predictions: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    order: Sequence[str],
) -> dict[str, float]:
    truth = np.asarray(y_true)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    missing = set(order) - set(predictions)
    if missing:
        raise ValueError(f"Missing predictions for {sorted(missing)}")
    output: dict[str, float] = {}
    for name in order:
        predicted = np.asarray(predictions[name])
        if predicted.ndim != 1 or predicted.shape != truth.shape:
            raise ValueError(f"Predictions for {name} have shape {predicted.shape}, expected {truth.shape}")
        output[name] = float(np.mean(predicted == truth))
    return output


def weighted_meta_features(
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    order: Sequence[str],
) -> np.ndarray:
    missing_predictions = set(order) - set(predictions)
    missing_weights = set(order) - set(weights)
    if missing_predictions or missing_weights:
        raise ValueError(
            "Predictions and weights must cover every selected classifier; "
            f"missing_predictions={sorted(missing_predictions)}, "
            f"missing_weights={sorted(missing_weights)}"
        )
    arrays = [np.asarray(predictions[name]) for name in order]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("Equation 10 requires one hard-prediction column per classifier")
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError("Base hard-prediction arrays are not aligned")
    weight_values = np.asarray([float(weights[name]) for name in order], dtype=np.float64)
    if not np.isfinite(weight_values).all() or np.any(weight_values < 0):
        raise ValueError("weights must be finite and non-negative")
    if weight_values.sum() <= 0:
        raise ValueError("At least one classifier weight must be positive")
    try:
        hard_predictions = np.column_stack(arrays).astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hard predictions must use numeric class labels") from exc
    result = hard_predictions * weight_values
    if not np.isfinite(result).all():
        raise ValueError("Weighted meta-features contain non-finite values")
    return result


def make_meta_classifier(settings: Any, seed: int) -> Any:
    from sklearn.svm import SVC

    if hasattr(settings, "section"):
        section = settings.section("wsaiec")
    else:
        section = settings
    if not isinstance(section, Mapping):
        raise TypeError("WS-AIEC settings must be a mapping or ExperimentConfig")
    parameters = dict(section.get("meta_svm", section))
    if str(parameters.get("kernel", "linear")) != "linear":
        raise ValueError("The WS-AIEC meta-classifier must use a linear SVM kernel")
    parameters["kernel"] = "linear"
    parameters["random_state"] = int(seed)
    return SVC(**parameters)


def optimize_alpha_gp(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    initial_points: int,
    iterations: int,
    candidates_per_iteration: int,
    kappa: float,
    seed: int,
) -> tuple[float, list[dict[str, float | int]]]:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    lower = float(lower)
    upper = float(upper)
    initial_points = int(initial_points)
    iterations = int(iterations)
    candidates_per_iteration = int(candidates_per_iteration)
    if not 0 < lower < upper:
        raise ValueError("alpha bounds must satisfy 0 < lower < upper")
    if initial_points < 1 or iterations < initial_points:
        raise ValueError("iterations must be at least initial_points, and both must be positive")
    if candidates_per_iteration < iterations:
        raise ValueError("candidates_per_iteration must be at least the total iteration count")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and non-negative")
    rng = np.random.default_rng(seed)
    if initial_points == 1:
        sampled = [float((lower + upper) / 2.0)]
    else:
        sampled = [lower, upper]
        sampled.extend(
            float(value)
            for value in rng.uniform(lower, upper, size=max(0, initial_points - 2))
        )

    def evaluate(alpha: float) -> float:
        score = float(objective(float(alpha)))
        if not np.isfinite(score):
            raise ValueError(f"Alpha objective returned a non-finite score at alpha={alpha}")
        return score

    scores = [evaluate(alpha) for alpha in sampled]
    candidates = np.linspace(lower, upper, candidates_per_iteration, dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
        noise_level=1e-6,
        noise_level_bounds=(1e-10, 1e-2),
    )
    while len(sampled) < iterations:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            random_state=int(seed),
            n_restarts_optimizer=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(np.asarray(sampled, dtype=np.float64).reshape(-1, 1), np.asarray(scores))
        mean, standard_deviation = gp.predict(candidates.reshape(-1, 1), return_std=True)
        acquisition = mean + float(kappa) * standard_deviation
        used = np.zeros(len(candidates), dtype=bool)
        for value in sampled:
            used[int(np.argmin(np.abs(candidates - value)))] = True
        acquisition[used] = -np.inf
        if not np.isfinite(acquisition).any():
            raise RuntimeError("Bayesian alpha candidate grid was exhausted")
        candidate = float(candidates[int(np.argmax(acquisition))])
        sampled.append(candidate)
        scores.append(evaluate(candidate))
    best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    history: list[dict[str, float | int]] = [
        {"iteration": index + 1, "alpha": float(alpha), "score": float(score)}
        for index, (alpha, score) in enumerate(zip(sampled, scores, strict=True))
    ]
    return float(sampled[best]), history


def optimize_alpha(
    objective: Callable[[float], float],
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[float, list[dict[str, float | int]]]:
    return optimize_alpha_gp(
        objective=objective,
        lower=float(settings["lower"]),
        upper=float(settings["upper"]),
        initial_points=int(settings["initial_points"]),
        iterations=int(settings["iterations"]),
        candidates_per_iteration=int(settings["candidates_per_iteration"]),
        kappa=float(settings["kappa"]),
        seed=int(seed),
    )


def optimize_shared_alpha(
    subject_keys: Sequence[Hashable],
    subject_objective: Callable[[Hashable, float], float],
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    keys = list(subject_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("subject_keys must contain at least one unique value")
    score_cache: dict[float, dict[Hashable, float]] = {}

    def mean_objective(alpha: float) -> float:
        scores = {key: float(subject_objective(key, alpha)) for key in keys}
        if not np.isfinite(np.asarray(list(scores.values()), dtype=np.float64)).all():
            raise ValueError(f"Subject alpha objective returned a non-finite score at alpha={alpha}")
        score_cache[float(alpha)] = scores
        return float(np.mean(list(scores.values())))

    best_alpha, base_history = optimize_alpha(mean_objective, settings, seed)
    history: list[dict[str, Any]] = []
    for record in base_history:
        alpha = float(record["alpha"])
        subject_scores = score_cache[alpha]
        history.append(
            {
                "iteration": int(record["iteration"]),
                "alpha": alpha,
                "score": float(record["score"]),
                "subject_scores": dict(subject_scores),
            }
        )
    return best_alpha, history


def weight_consistency_report(config: Any) -> dict[str, Any]:
    section = config.section("wsaiec") if hasattr(config, "section") else config
    static_weights = {
        str(name): float(value) for name, value in section.get("reported_static_weights", {}).items()
    }
    return {
        "dynamic_weight_equation": "exp(alpha * accuracy_i) / sum_j exp(alpha * accuracy_j)",
        "base_classifiers": list(section["base_classifiers"]),
        "meta_classifier": str(section["meta_classifier"]),
        "validation_fraction": float(section["validation_fraction"]),
        "alpha_optimization": dict(section["alpha_optimization"]),
        "reported_static_weights": static_weights,
        "reported_static_weight_sum": float(sum(static_weights.values())),
        "static_weight_role": "paper cluster-selection audit and Table 9 scenario 11 weighting",
        "dynamic_weight_role": "base-prediction weighting",
    }
