"""Fast no-download end-to-end check of CSP, base models, and stacking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.data.splits import chronological_holdout, make_outer_split
from wsaiec_eeg.evaluation.metrics import classification_metrics
from wsaiec_eeg.evaluation.stacking import (
    _base_holdout_predictions,
    _base_oof_predictions,
    _weighted_oof_features,
)
from wsaiec_eeg.features.csp import CSPFeatureTransformer, build_fold_features
from wsaiec_eeg.models.classifiers import aligned_probabilities, make_classifier
from wsaiec_eeg.models.wsaiec import (
    dynamic_weights,
    make_meta_classifier,
    optimize_alpha,
    validation_accuracies,
    weighted_meta_features,
)
from wsaiec_eeg.utils.io import ensure_directory, write_json


def _synthetic_eeg(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_trials, n_channels, n_samples = 180, 3, 96
    y = np.arange(n_trials) % 2
    X = rng.normal(scale=0.6, size=(n_trials, n_channels, n_samples))
    time = np.linspace(0, 1, n_samples, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 10 * time)
    pattern_b = np.sin(2 * np.pi * 18 * time)
    X[y == 0, 0] += 1.4 * pattern_a
    X[y == 0, 2] += 0.3 * pattern_b
    X[y == 1, 2] += 1.4 * pattern_b
    X[y == 1, 0] += 0.3 * pattern_a
    return X.astype(np.float64), y.astype(np.int64)


def run_smoke(config: ExperimentConfig, output: str | Path) -> Path:
    X, y = _synthetic_eeg(int(config.project["random_seed"]))
    split_settings = config.section("splitting")
    outer_split = make_outer_split(
        y,
        "chronological",
        float(split_settings["outer_test_fraction"]),
    )
    wsaiec_settings = config.section("wsaiec")
    validation_split = chronological_holdout(
        y[outer_split.train],
        float(wsaiec_settings["validation_fraction"]),
    )
    X_development = X[outer_split.train]
    y_development = y[outer_split.train]
    X_weight_train = X_development[validation_split.train]
    y_weight_train = y_development[validation_split.train]
    X_weight_validation = X_development[validation_split.test]
    y_weight_validation = y_development[validation_split.test]
    csp_settings = dict(config.section("features")["csp"])
    folds = build_fold_features(
        X_weight_train,
        y_weight_train,
        csp_settings,
        True,
        int(split_settings["time_series_splits"]),
    )
    selected = list(wsaiec_settings["base_classifiers"])
    classifier_config: dict[str, Any] = config.section("classifiers")
    classes = np.unique(y)
    oof_predictions, fold_for_row, fold_accuracies = _base_oof_predictions(
        folds,
        len(y_weight_train),
        selected,
        classifier_config,
        100,
        1,
        X=X_weight_train,
        y=y_weight_train,
        csp_settings=csp_settings,
        standardize=True,
        validation_fraction=float(wsaiec_settings["validation_fraction"]),
    )
    holdout_predictions = _base_holdout_predictions(
        X_weight_train,
        y_weight_train,
        X_weight_validation,
        csp_settings,
        True,
        selected,
        classifier_config,
        10_000,
        1,
    )
    holdout_accuracies = validation_accuracies(
        holdout_predictions,
        y_weight_validation,
        selected,
    )

    def objective(alpha: float) -> float:
        candidate_features, candidate_rows, _ = _weighted_oof_features(
            oof_predictions,
            fold_for_row,
            fold_accuracies,
            alpha,
            selected,
        )
        candidate_meta = make_meta_classifier(wsaiec_settings, 20_000)
        candidate_meta.fit(
            candidate_features[candidate_rows],
            y_weight_train[candidate_rows],
        )
        candidate_weights = dynamic_weights(holdout_accuracies, alpha, selected)
        holdout_features = weighted_meta_features(
            holdout_predictions,
            candidate_weights,
            selected,
        )
        return float(
            accuracy_score(y_weight_validation, candidate_meta.predict(holdout_features))
        )

    alpha, alpha_history = optimize_alpha(
        objective,
        wsaiec_settings["alpha_optimization"],
        20_000,
    )
    oof_features, meta_rows, _ = _weighted_oof_features(
        oof_predictions,
        fold_for_row,
        fold_accuracies,
        alpha,
        selected,
    )
    weights = dynamic_weights(holdout_accuracies, alpha, selected)
    holdout_features = weighted_meta_features(holdout_predictions, weights, selected)
    meta_X = np.vstack([oof_features[meta_rows], holdout_features])
    meta_y = np.concatenate([y_weight_train[meta_rows], y_weight_validation])
    meta = make_meta_classifier(wsaiec_settings, 30_000).fit(meta_X, meta_y)

    transformer = CSPFeatureTransformer(csp_settings, True)
    full_train = transformer.fit_transform(X_development, y_development)
    test = transformer.transform(X[outer_split.test])
    base_test_predictions: dict[str, np.ndarray] = {}
    for model_index, name in enumerate(selected):
        model = make_classifier(name, classifier_config, 500 + model_index, 1)
        model.fit(full_train, y_development)
        base_test_predictions[name] = np.asarray(model.predict(test), dtype=np.int64)
    meta_test = weighted_meta_features(base_test_predictions, weights, selected)
    probabilities = aligned_probabilities(meta, meta_test, classes)
    predicted = np.asarray(meta.predict(meta_test))
    metrics = classification_metrics(
        y[outer_split.test],
        predicted,
        probabilities,
        config.section("evaluation"),
    )
    payload = {
        "status": "ok",
        "n_trials": len(y),
        "n_meta_rows": len(meta_y),
        "alpha": alpha,
        "alpha_history": alpha_history,
        "validation_accuracies": holdout_accuracies,
        "weights": weights,
        "metrics": metrics,
    }
    target = ensure_directory(Path(output)) / "smoke_result.json"
    write_json(target, payload)
    return target
