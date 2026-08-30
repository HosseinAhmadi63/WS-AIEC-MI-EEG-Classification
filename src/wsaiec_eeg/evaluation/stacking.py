"""Leakage-safe out-of-fold construction and dataset-calibrated WS-AIEC evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.data.cache import load_cached_subject
from wsaiec_eeg.data.splits import chronological_holdout, make_outer_split
from wsaiec_eeg.evaluation.base_benchmark import _feature_settings, subject_result_directory
from wsaiec_eeg.evaluation.metrics import classification_metrics
from wsaiec_eeg.features.csp import CSPFeatureTransformer, FoldFeatures, build_fold_features
from wsaiec_eeg.models.classifiers import aligned_probabilities, make_classifier
from wsaiec_eeg.models.wsaiec import (
    dynamic_weights,
    make_meta_classifier,
    optimize_shared_alpha,
    validation_accuracies,
    weighted_meta_features,
)
from wsaiec_eeg.utils.io import ensure_directory, write_csv


@dataclass(frozen=True)
class SubjectCalibration:
    dataset_name: str
    subject: int
    classes: np.ndarray
    development_train_index: np.ndarray
    development_validation_index: np.ndarray
    y_weight_train: np.ndarray
    y_weight_validation: np.ndarray
    selected: tuple[str, ...]
    oof_predictions: dict[str, np.ndarray]
    fold_for_row: np.ndarray
    fold_accuracies: dict[int, dict[str, float]]
    fold_weight_validation_sizes: dict[int, int]
    holdout_predictions: dict[str, np.ndarray]
    holdout_accuracies: dict[str, float]
    base_seed: int


def _inner_weight_indices(
    outer_train_index: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(outer_train_index, dtype=np.int64)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    n_validation = int(math.ceil(len(source) * validation_fraction))
    boundary = len(source) - n_validation
    if boundary <= 0:
        raise ValueError("Outer fold is too small for inner weight validation")
    inner_train = source[:boundary]
    inner_valid = source[boundary:]
    labels = np.asarray(y)
    if not np.array_equal(np.unique(labels[inner_train]), np.unique(labels[source])):
        raise RuntimeError("Inner weight-training prefix is missing a class")
    return inner_train, inner_valid


def _base_oof_predictions(
    folds: list[FoldFeatures],
    n_samples: int,
    selected: list[str],
    classifier_config: dict[str, Any],
    seed: int,
    n_jobs: int,
    *,
    X: np.ndarray,
    y: np.ndarray,
    csp_settings: dict[str, Any],
    standardize: bool,
    validation_fraction: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, dict[str, float]]]:
    raw_X = np.asarray(X)
    labels = np.asarray(y)
    if len(raw_X) != n_samples or len(labels) != n_samples:
        raise ValueError("Raw epochs, labels, and n_samples must be aligned")
    predictions = {
        name: np.full(n_samples, np.nan, dtype=np.float64) for name in selected
    }
    fold_for_row = np.full(n_samples, -1, dtype=np.int64)
    accuracies: dict[int, dict[str, float]] = {}
    for fold in folds:
        fold_for_row[fold.valid_index] = fold.fold
        inner_train_index, inner_valid_index = _inner_weight_indices(
            fold.train_index,
            labels,
            validation_fraction,
        )
        if inner_valid_index.max() >= fold.valid_index.min():
            raise RuntimeError("Inner weight validation must precede outer-fold predictions")
        inner_transformer = CSPFeatureTransformer(csp_settings, standardize)
        inner_train_features = inner_transformer.fit_transform(
            raw_X[inner_train_index],
            labels[inner_train_index],
        )
        inner_valid_features = inner_transformer.transform(raw_X[inner_valid_index])
        inner_predictions: dict[str, np.ndarray] = {}
        for classifier_index, name in enumerate(selected):
            inner_model = make_classifier(
                name,
                classifier_config,
                seed + fold.fold * 10_000 + classifier_index,
                n_jobs,
            )
            inner_model.fit(inner_train_features, labels[inner_train_index])
            inner_predictions[name] = np.asarray(
                inner_model.predict(inner_valid_features),
                dtype=np.int64,
            )
            outer_model = make_classifier(
                name,
                classifier_config,
                seed + fold.fold * 10_000 + 5_000 + classifier_index,
                n_jobs,
            )
            outer_model.fit(fold.X_train, fold.y_train)
            predictions[name][fold.valid_index] = np.asarray(
                outer_model.predict(fold.X_valid),
                dtype=np.int64,
            )
        accuracies[fold.fold] = validation_accuracies(
            inner_predictions,
            labels[inner_valid_index],
            selected,
        )
    return predictions, fold_for_row, accuracies


def _base_holdout_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    csp_settings: dict[str, Any],
    standardize: bool,
    selected: list[str],
    classifier_config: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    transformer = CSPFeatureTransformer(csp_settings, standardize)
    train_features = transformer.fit_transform(X_train, y_train)
    valid_features = transformer.transform(X_valid)
    output: dict[str, np.ndarray] = {}
    for classifier_index, name in enumerate(selected):
        model = make_classifier(
            name,
            classifier_config,
            seed + classifier_index,
            n_jobs,
        )
        model.fit(train_features, y_train)
        output[name] = np.asarray(model.predict(valid_features), dtype=np.int64)
    return output


def _weighted_oof_features(
    predictions: dict[str, np.ndarray],
    fold_for_row: np.ndarray,
    fold_accuracies: dict[int, dict[str, float]],
    alpha: float,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
    output = np.full((len(fold_for_row), len(selected)), np.nan, dtype=np.float64)
    weights_by_fold: dict[int, dict[str, float]] = {}
    for fold, accuracies in fold_accuracies.items():
        rows = np.flatnonzero(fold_for_row == fold)
        weights = dynamic_weights(accuracies, alpha, selected)
        weights_by_fold[fold] = weights
        output[rows] = weighted_meta_features(
            {name: predictions[name][rows] for name in selected},
            weights,
            selected,
        )
    valid_rows = np.flatnonzero(np.isfinite(output).all(axis=1))
    if valid_rows.size == 0:
        raise RuntimeError("TimeSeriesSplit produced no out-of-fold rows for stacking")
    return output, valid_rows, weights_by_fold


def _fold_weight_validation_sizes(
    folds: list[FoldFeatures],
    y: np.ndarray,
    validation_fraction: float,
) -> dict[int, int]:
    return {
        fold.fold: len(_inner_weight_indices(fold.train_index, y, validation_fraction)[1])
        for fold in folds
    }


def _weight_history_rows(
    prepared: SubjectCalibration,
    alpha: float,
    fold_weights: dict[int, dict[str, float]],
    holdout_weights: dict[str, float],
    subject_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = list(prepared.selected)
    for fold in sorted(prepared.fold_accuracies):
        n_meta_predictions = int(np.sum(prepared.fold_for_row == fold))
        for name in selected:
            rows.append(
                {
                    "dataset": prepared.dataset_name,
                    "subject": prepared.subject,
                    "alpha_scope": "dataset",
                    "subject_mode": subject_mode,
                    "stage": "time_series_oof",
                    "weight_source": "inner_chronological_validation",
                    "epoch": fold,
                    "classifier": name,
                    "alpha": alpha,
                    "validation_accuracy": prepared.fold_accuracies[fold][name],
                    "weight": fold_weights[fold][name],
                    "n_validation": prepared.fold_weight_validation_sizes[fold],
                    "n_meta_predictions": n_meta_predictions,
                }
            )
    holdout_epoch = max(prepared.fold_accuracies, default=0) + 1
    for name in selected:
        rows.append(
            {
                "dataset": prepared.dataset_name,
                "subject": prepared.subject,
                "alpha_scope": "dataset",
                "subject_mode": subject_mode,
                "stage": "alpha_validation",
                "weight_source": "development_holdout",
                "epoch": holdout_epoch,
                "classifier": name,
                "alpha": alpha,
                "validation_accuracy": prepared.holdout_accuracies[name],
                "weight": holdout_weights[name],
                "n_validation": len(prepared.y_weight_validation),
                "n_meta_predictions": len(prepared.y_weight_validation),
            }
        )
    return rows


def _prepare_subject_predictions(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
    selected: list[str],
    oof_seed_offset: int,
    holdout_seed_offset: int,
) -> SubjectCalibration:
    cached = load_cached_subject(config, dataset_name, subject)
    split_settings = config.section("splitting")
    outer_split = make_outer_split(
        cached.y,
        str(split_settings["outer_strategy"]),
        float(split_settings["outer_test_fraction"]),
    )
    X_development = cached.X[outer_split.train]
    y_development = cached.y[outer_split.train]
    classes = np.unique(y_development)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("WS-AIEC requires contiguous integer class labels")
    wsaiec_settings = config.section("wsaiec")
    validation_fraction = float(wsaiec_settings["validation_fraction"])
    validation_split = chronological_holdout(y_development, validation_fraction)
    X_weight_train = X_development[validation_split.train]
    y_weight_train = y_development[validation_split.train]
    X_weight_validation = X_development[validation_split.test]
    y_weight_validation = y_development[validation_split.test]
    csp_settings, standardize = _feature_settings(config)
    folds = build_fold_features(
        X_weight_train,
        y_weight_train,
        csp_settings,
        standardize,
        int(split_settings["time_series_splits"]),
        int(split_settings["gap"]),
    )
    classifier_config = config.section("classifiers")
    n_jobs = int(config.project["n_jobs"])
    base_seed = int(config.project["random_seed"]) + subject * 10_000
    oof_predictions, fold_for_row, fold_accuracies = _base_oof_predictions(
        folds,
        len(y_weight_train),
        selected,
        classifier_config,
        base_seed + oof_seed_offset,
        n_jobs,
        X=X_weight_train,
        y=y_weight_train,
        csp_settings=csp_settings,
        standardize=standardize,
        validation_fraction=validation_fraction,
    )
    holdout_predictions = _base_holdout_predictions(
        X_weight_train,
        y_weight_train,
        X_weight_validation,
        csp_settings,
        standardize,
        selected,
        classifier_config,
        base_seed + holdout_seed_offset,
        n_jobs,
    )
    holdout_accuracies = validation_accuracies(
        holdout_predictions,
        y_weight_validation,
        selected,
    )
    return SubjectCalibration(
        dataset_name=dataset_name,
        subject=subject,
        classes=classes,
        development_train_index=validation_split.train,
        development_validation_index=validation_split.test,
        y_weight_train=y_weight_train,
        y_weight_validation=y_weight_validation,
        selected=tuple(selected),
        oof_predictions=oof_predictions,
        fold_for_row=fold_for_row,
        fold_accuracies=fold_accuracies,
        fold_weight_validation_sizes=_fold_weight_validation_sizes(
            folds,
            y_weight_train,
            validation_fraction,
        ),
        holdout_predictions=holdout_predictions,
        holdout_accuracies=holdout_accuracies,
        base_seed=base_seed,
    )


def _prepare_subject_calibration(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
) -> SubjectCalibration:
    return _prepare_subject_predictions(
        config,
        dataset_name,
        subject,
        list(config.section("wsaiec")["base_classifiers"]),
        100_000,
        200_000,
    )


def _subject_alpha_validation_accuracy(
    prepared: SubjectCalibration,
    alpha: float,
    wsaiec_settings: dict[str, Any],
) -> float:
    selected = list(prepared.selected)
    candidate_features, candidate_rows, _ = _weighted_oof_features(
        prepared.oof_predictions,
        prepared.fold_for_row,
        prepared.fold_accuracies,
        alpha,
        selected,
    )
    candidate_meta = make_meta_classifier(wsaiec_settings, prepared.base_seed + 300_000)
    candidate_meta.fit(
        candidate_features[candidate_rows],
        prepared.y_weight_train[candidate_rows],
    )
    candidate_weights = dynamic_weights(
        prepared.holdout_accuracies,
        alpha,
        selected,
    )
    validation_features = weighted_meta_features(
        prepared.holdout_predictions,
        candidate_weights,
        selected,
    )
    return float(
        accuracy_score(
            prepared.y_weight_validation,
            candidate_meta.predict(validation_features),
        )
    )


def _subject_output_paths(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
) -> tuple[Path, Path, Path, Path, Path]:
    directory = subject_result_directory(config, dataset_name, subject)
    return (
        directory / "wsaiec_metrics.csv",
        directory / "wsaiec_predictions.csv",
        directory / "wsaiec_oof.csv",
        directory / "wsaiec_alpha_history.csv",
        directory / "wsaiec_weight_history.csv",
    )


def _selected_history_record(
    alpha_history: list[dict[str, Any]],
    best_alpha: float,
) -> dict[str, Any]:
    matches = [
        record
        for record in alpha_history
        if float(record["alpha"]) == float(best_alpha)
    ]
    if len(matches) != 1:
        raise RuntimeError("Dataset alpha history does not contain one selected candidate")
    return matches[0]


def _finalize_subject_wsaiec(
    config: ExperimentConfig,
    prepared: SubjectCalibration,
    best_alpha: float,
    alpha_history: list[dict[str, Any]],
    configured_subjects: list[int],
    requested_subjects: list[int],
    subject_mode: str,
) -> tuple[Path, Path]:
    dataset_name = prepared.dataset_name
    subject = prepared.subject
    ensure_directory(subject_result_directory(config, dataset_name, subject))
    (
        metrics_path,
        predictions_path,
        oof_path,
        alpha_history_path,
        weight_history_path,
    ) = _subject_output_paths(config, dataset_name, subject)
    selected = list(prepared.selected)
    oof_features, meta_rows, fold_weights = _weighted_oof_features(
        prepared.oof_predictions,
        prepared.fold_for_row,
        prepared.fold_accuracies,
        best_alpha,
        selected,
    )
    final_weights = dynamic_weights(
        prepared.holdout_accuracies,
        best_alpha,
        selected,
    )
    validation_features = weighted_meta_features(
        prepared.holdout_predictions,
        final_weights,
        selected,
    )
    meta_X = np.vstack([oof_features[meta_rows], validation_features])
    meta_y = np.concatenate(
        [
            prepared.y_weight_train[meta_rows],
            prepared.y_weight_validation,
        ]
    )
    if not np.array_equal(np.unique(meta_y), prepared.classes):
        raise RuntimeError("Meta-classifier training rows do not contain every class")
    wsaiec_settings = config.section("wsaiec")
    meta = make_meta_classifier(wsaiec_settings, prepared.base_seed + 400_000)
    meta.fit(meta_X, meta_y)

    cached = load_cached_subject(config, dataset_name, subject)
    split_settings = config.section("splitting")
    outer_split = make_outer_split(
        cached.y,
        str(split_settings["outer_strategy"]),
        float(split_settings["outer_test_fraction"]),
    )
    X_development = cached.X[outer_split.train]
    y_development = cached.y[outer_split.train]
    X_test = cached.X[outer_split.test]
    y_test = cached.y[outer_split.test]
    if not np.array_equal(np.unique(y_test), prepared.classes):
        raise RuntimeError("Outer test partition does not contain every class")
    csp_settings, standardize = _feature_settings(config)
    full_transformer = CSPFeatureTransformer(csp_settings, standardize)
    full_train_features = full_transformer.fit_transform(X_development, y_development)
    test_features = full_transformer.transform(X_test)
    classifier_config = config.section("classifiers")
    n_jobs = int(config.project["n_jobs"])
    test_predictions: dict[str, np.ndarray] = {}
    for classifier_index, name in enumerate(selected):
        model = make_classifier(
            name,
            classifier_config,
            prepared.base_seed + 500_000 + classifier_index,
            n_jobs,
        )
        model.fit(full_train_features, y_development)
        test_predictions[name] = np.asarray(
            model.predict(test_features),
            dtype=np.int64,
        )
    meta_test = weighted_meta_features(test_predictions, final_weights, selected)
    probabilities = aligned_probabilities(meta, meta_test, prepared.classes)
    predicted = np.asarray(meta.predict(meta_test))
    metrics = classification_metrics(
        y_test,
        predicted,
        probabilities,
        config.section("evaluation"),
    )
    selected_record = _selected_history_record(alpha_history, best_alpha)
    subject_scores = {
        int(key): float(value)
        for key, value in selected_record["subject_scores"].items()
    }
    cohort_payload = {
        "configured_subjects": json.dumps(configured_subjects),
        "requested_subjects": json.dumps(requested_subjects),
        "n_alpha_subjects": len(requested_subjects),
    }
    metric_row = {
        "dataset": dataset_name,
        "subject": subject,
        "model": "WS-AIEC",
        "alpha_scope": "dataset",
        "subject_mode": subject_mode,
        **cohort_payload,
        "n_train": len(y_development),
        "n_weight_train": len(prepared.y_weight_train),
        "n_weight_validation": len(prepared.y_weight_validation),
        "n_meta_train": len(meta_y),
        "n_eval": len(y_test),
        "alpha": best_alpha,
        "alpha_validation_accuracy": subject_scores[subject],
        "dataset_alpha_validation_accuracy": float(selected_record["score"]),
        "validation_accuracies": json.dumps(prepared.holdout_accuracies, sort_keys=True),
        "weights": json.dumps(final_weights, sort_keys=True),
        **metrics,
    }
    write_csv(metrics_path, pd.DataFrame([metric_row]))

    prediction_rows = [
        {
            "sample_key": f"{dataset_name}:s{subject:02d}:trial{int(trial):05d}",
            "dataset": dataset_name,
            "subject": subject,
            "trial_index": int(trial),
            "model": "WS-AIEC",
            "alpha_scope": "dataset",
            "alpha": best_alpha,
            "y_true": int(true),
            "y_pred": int(prediction),
            "probabilities": json.dumps([float(value) for value in probability]),
            "base_predictions": json.dumps(
                {name: int(test_predictions[name][row]) for name in selected},
                sort_keys=True,
            ),
            "weights": json.dumps(final_weights, sort_keys=True),
            "meta_features": json.dumps([float(value) for value in meta_test[row]]),
        }
        for row, (trial, true, prediction, probability) in enumerate(
            zip(outer_split.test, y_test, predicted, probabilities, strict=True)
        )
    ]
    write_csv(predictions_path, pd.DataFrame(prediction_rows))

    oof_rows: list[dict[str, Any]] = []
    for index in meta_rows:
        fold = int(prepared.fold_for_row[index])
        development_index = int(prepared.development_train_index[index])
        for name in selected:
            oof_rows.append(
                {
                    "dataset": dataset_name,
                    "subject": subject,
                    "stage": "time_series_oof",
                    "weight_source": "inner_chronological_validation",
                    "epoch": fold,
                    "development_index": development_index,
                    "trial_index": int(outer_split.train[development_index]),
                    "classifier": name,
                    "y_true": int(prepared.y_weight_train[index]),
                    "hard_prediction": int(prepared.oof_predictions[name][index]),
                    "validation_accuracy": prepared.fold_accuracies[fold][name],
                    "n_weight_validation": prepared.fold_weight_validation_sizes[fold],
                    "weight": fold_weights[fold][name],
                    "weighted_prediction": float(
                        prepared.oof_predictions[name][index] * fold_weights[fold][name]
                    ),
                }
            )
    holdout_epoch = max(prepared.fold_accuracies) + 1
    for index, development_index in enumerate(prepared.development_validation_index):
        for name in selected:
            oof_rows.append(
                {
                    "dataset": dataset_name,
                    "subject": subject,
                    "stage": "alpha_validation",
                    "weight_source": "development_holdout",
                    "epoch": holdout_epoch,
                    "development_index": int(development_index),
                    "trial_index": int(outer_split.train[development_index]),
                    "classifier": name,
                    "y_true": int(prepared.y_weight_validation[index]),
                    "hard_prediction": int(prepared.holdout_predictions[name][index]),
                    "validation_accuracy": prepared.holdout_accuracies[name],
                    "n_weight_validation": len(prepared.y_weight_validation),
                    "weight": final_weights[name],
                    "weighted_prediction": float(
                        prepared.holdout_predictions[name][index] * final_weights[name]
                    ),
                }
            )
    write_csv(oof_path, pd.DataFrame(oof_rows))

    subject_alpha_rows = []
    for record in alpha_history:
        record_scores = {
            int(key): float(value)
            for key, value in record["subject_scores"].items()
        }
        subject_alpha_rows.append(
            {
                "dataset": dataset_name,
                "subject": subject,
                "alpha_scope": "dataset",
                "subject_mode": subject_mode,
                **cohort_payload,
                "iteration": int(record["iteration"]),
                "alpha": float(record["alpha"]),
                "subject_validation_accuracy": record_scores[subject],
                "mean_validation_accuracy": float(record["score"]),
                "subject_validation_accuracies": json.dumps(record_scores, sort_keys=True),
                "selected": float(record["alpha"]) == float(best_alpha),
            }
        )
    write_csv(alpha_history_path, pd.DataFrame(subject_alpha_rows))
    write_csv(
        weight_history_path,
        pd.DataFrame(
            _weight_history_rows(
                prepared,
                best_alpha,
                fold_weights,
                final_weights,
                subject_mode,
            )
        ),
    )
    return metrics_path, predictions_path


def _dataset_alpha_history_rows(
    dataset_name: str,
    best_alpha: float,
    alpha_history: list[dict[str, Any]],
    configured_subjects: list[int],
    requested_subjects: list[int],
    subject_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in alpha_history:
        subject_scores = {
            int(key): float(value)
            for key, value in record["subject_scores"].items()
        }
        rows.append(
            {
                "dataset": dataset_name,
                "alpha_scope": "dataset",
                "subject_mode": subject_mode,
                "configured_subjects": json.dumps(configured_subjects),
                "requested_subjects": json.dumps(requested_subjects),
                "n_subjects": len(requested_subjects),
                "iteration": int(record["iteration"]),
                "alpha": float(record["alpha"]),
                "mean_validation_accuracy": float(record["score"]),
                "subject_validation_accuracies": json.dumps(subject_scores, sort_keys=True),
                "selected": float(record["alpha"]) == float(best_alpha),
            }
        )
    return rows


def _dataset_outputs_reusable(
    config: ExperimentConfig,
    dataset_name: str,
    configured_subjects: list[int],
    requested_subjects: list[int],
    subject_mode: str,
) -> bool:
    history_path = config.run_root / dataset_name / "wsaiec_alpha_history.csv"
    required_paths = [
        path
        for subject in requested_subjects
        for path in _subject_output_paths(config, dataset_name, subject)
    ]
    if not history_path.exists() or not all(path.exists() for path in required_paths):
        return False
    try:
        history = pd.read_csv(history_path)
        if history.empty:
            return False
        first = history.iloc[0]
        return (
            str(first["dataset"]) == dataset_name
            and str(first["alpha_scope"]) == "dataset"
            and str(first["subject_mode"]) == subject_mode
            and json.loads(str(first["configured_subjects"])) == configured_subjects
            and json.loads(str(first["requested_subjects"])) == requested_subjects
            and int(first["n_subjects"]) == len(requested_subjects)
            and int(history["selected"].astype(bool).sum()) == 1
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_subject_wsaiec(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
    force: bool = False,
) -> tuple[Path, Path]:
    return run_dataset_wsaiec(
        config,
        dataset_name,
        subjects=[subject],
        force=force,
    )[0]


def run_dataset_wsaiec(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: list[int] | None = None,
    force: bool = False,
) -> list[tuple[Path, Path]]:
    if dataset_name not in config.datasets:
        raise ValueError(f"Unknown dataset {dataset_name!r}")
    configured_subjects = [int(subject) for subject in config.datasets[dataset_name]["subjects"]]
    requested_subjects = (
        list(configured_subjects)
        if subjects is None
        else [int(subject) for subject in subjects]
    )
    if not requested_subjects or len(requested_subjects) != len(set(requested_subjects)):
        raise ValueError("subjects must contain at least one unique configured subject")
    invalid = set(requested_subjects) - set(configured_subjects)
    if invalid:
        raise ValueError(f"Invalid subjects for {dataset_name}: {sorted(invalid)}")
    is_full_cohort = set(requested_subjects) == set(configured_subjects)
    if is_full_cohort:
        requested_subjects = list(configured_subjects)
    subject_mode = "paper_all_subjects" if is_full_cohort else "debug_subject_subset"
    if not force and _dataset_outputs_reusable(
        config,
        dataset_name,
        configured_subjects,
        requested_subjects,
        subject_mode,
    ):
        return [
            _subject_output_paths(config, dataset_name, subject)[:2]
            for subject in requested_subjects
        ]

    wsaiec_settings = config.section("wsaiec")
    alpha_settings = wsaiec_settings["alpha_optimization"]
    if str(alpha_settings["scoring"]) != "accuracy":
        raise ValueError("WS-AIEC alpha optimization currently requires accuracy scoring")
    prepared_by_subject = {
        subject: _prepare_subject_calibration(config, dataset_name, subject)
        for subject in requested_subjects
    }
    dataset_seed = (
        int(config.project["random_seed"])
        + list(config.datasets).index(dataset_name) * 1_000_000
        + 900_000
    )
    best_alpha, alpha_history = optimize_shared_alpha(
        requested_subjects,
        lambda key, alpha: _subject_alpha_validation_accuracy(
            prepared_by_subject[int(key)],
            alpha,
            wsaiec_settings,
        ),
        alpha_settings,
        dataset_seed,
    )
    outputs = [
        _finalize_subject_wsaiec(
            config,
            prepared_by_subject[subject],
            best_alpha,
            alpha_history,
            configured_subjects,
            requested_subjects,
            subject_mode,
        )
        for subject in requested_subjects
    ]
    dataset_history_path = (
        ensure_directory(config.run_root / dataset_name) / "wsaiec_alpha_history.csv"
    )
    write_csv(
        dataset_history_path,
        pd.DataFrame(
            _dataset_alpha_history_rows(
                dataset_name,
                best_alpha,
                alpha_history,
                configured_subjects,
                requested_subjects,
                subject_mode,
            )
        ),
    )
    return outputs
