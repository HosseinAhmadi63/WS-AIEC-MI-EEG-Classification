"""Executable, leakage-safe reconstruction of the eleven Table 9 ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.constants import CLASSIFIER_ORDER, METRIC_COLUMNS
from wsaiec_eeg.data.cache import load_cached_subject
from wsaiec_eeg.data.splits import make_outer_split
from wsaiec_eeg.evaluation.base_benchmark import _feature_settings, subject_result_directory
from wsaiec_eeg.evaluation.metrics import classification_metrics
from wsaiec_eeg.evaluation.stacking import (
    SubjectCalibration,
    _prepare_subject_predictions,
    _weighted_oof_features,
)
from wsaiec_eeg.features.csp import CSPFeatureTransformer
from wsaiec_eeg.models.classifiers import aligned_probabilities, make_classifier
from wsaiec_eeg.models.wsaiec import (
    dynamic_weights,
    make_meta_classifier,
    optimize_shared_alpha,
    weighted_meta_features,
)
from wsaiec_eeg.utils.io import ensure_directory, write_csv, write_json

ABLATION_DATASETS = ("BNCI2014_002", "Zhou2016")
ABLATION_SOURCE_COLUMNS = (
    "scenario_id",
    "scenario",
    "selected_classifiers",
    "accuracy_bnci2014_002",
    "accuracy_zhou2016",
)
EXPECTED_ABLATION_SCENARIOS = (
    (1, "CPA Only", ("SVM", "SVM_rbf", "MLP", "LR", "RC", "KN")),
    (2, "LCA Only", ("NB", "PC", "SGD", "RC", "SVM", "AB")),
    (
        3,
        "CPA + LCA (No Performance Stability)",
        ("NB", "PC", "SVM_rbf", "RF", "LDA", "SVM"),
    ),
    (
        4,
        "CPA + LCA (No Convergence Rate)",
        ("NB", "PC", "SVM_rbf", "GB", "RC", "SVM"),
    ),
    (
        5,
        "CPA + LCA (No AUC-CV)",
        ("SVM", "LDA", "SVM_rbf", "PC", "NB", "SGD"),
    ),
    (
        6,
        "CPA + LCA (Only AUC-CV)",
        ("NB", "MLP", "SVM_rbf", "AB", "SGD", "SVM"),
    ),
    (
        7,
        "CPA + LCA (Only Convergence Rate)",
        ("NB", "PC", "RF", "SGD", "SVM_rbf", "SVM"),
    ),
    (
        8,
        "CPA + LCA (Only Performance Stability)",
        ("MLP", "SVM", "RC", "PC", "NB", "SGD"),
    ),
    (9, "Without GB", ("NB", "PC", "SVM_rbf", "LDA", "SVM")),
    (10, "Without SVM-rbf", ("NB", "PC", "GB", "LDA", "SVM")),
    (11, "Static Weights", ("NB", "PC", "SVM_rbf", "GB", "LDA", "SVM")),
)


@dataclass(frozen=True)
class AblationScenario:
    scenario_id: int
    scenario: str
    selected_classifiers: tuple[str, ...]
    meta_classifier: str
    base_classifiers: tuple[str, ...]
    weighting_method: str
    paper_accuracy_bnci2014_002: float
    paper_accuracy_zhou2016: float

    @property
    def selected_text(self) -> str:
        return ";".join(self.selected_classifiers)


def static_rank_weights(
    reported_weights: dict[str, float],
    base_classifiers: tuple[str, ...] | list[str],
) -> dict[str, float]:
    names = list(base_classifiers)
    if not names or len(names) != len(set(names)):
        raise ValueError("base_classifiers must contain at least one unique classifier")
    missing = set(names) - set(reported_weights)
    if missing:
        raise ValueError(f"Missing printed Table 7 weights for {sorted(missing)}")
    values = np.asarray([float(reported_weights[name]) for name in names], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Printed Table 7 weights must be finite and positive")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def load_ablation_scenarios(config: ExperimentConfig) -> tuple[AblationScenario, ...]:
    source_path = config.publication_source / "ablation.csv"
    frame = pd.read_csv(source_path)
    if tuple(frame.columns) != ABLATION_SOURCE_COLUMNS:
        raise ValueError(
            f"Table 9 columns must be {list(ABLATION_SOURCE_COLUMNS)}, found {frame.columns.tolist()}"
        )
    observed = tuple(
        (
            int(row.scenario_id),
            str(row.scenario),
            tuple(str(row.selected_classifiers).split(";")),
        )
        for row in frame.itertuples(index=False)
    )
    if observed != EXPECTED_ABLATION_SCENARIOS:
        raise ValueError("Table 9 scenarios or selected classifiers differ from the frozen paper table")
    ranks = {
        str(name): int(rank)
        for name, rank in config.section("ranking")["reported_overall_ranks"].items()
    }
    scenarios: list[AblationScenario] = []
    for row in frame.itertuples(index=False):
        selected = tuple(str(row.selected_classifiers).split(";"))
        meta = min(selected, key=lambda name: (ranks[name], CLASSIFIER_ORDER.index(name)))
        if meta != "SVM":
            raise ValueError(f"Table 9 scenario {row.scenario_id} does not resolve to the SVM meta-classifier")
        bases = tuple(name for name in selected if name != meta)
        scenarios.append(
            AblationScenario(
                scenario_id=int(row.scenario_id),
                scenario=str(row.scenario),
                selected_classifiers=selected,
                meta_classifier=meta,
                base_classifiers=bases,
                weighting_method=(
                    "static_inverse_overall_rank"
                    if int(row.scenario_id) == 11
                    else "dynamic_softmax_validation_accuracy"
                ),
                paper_accuracy_bnci2014_002=float(row.accuracy_bnci2014_002),
                paper_accuracy_zhou2016=float(row.accuracy_zhou2016),
            )
        )
    return tuple(scenarios)


def _fixed_oof_features(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(predictions[name], dtype=np.float64) for name in selected]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Static out-of-fold predictions are not aligned")
    rows = np.flatnonzero(np.isfinite(np.column_stack(arrays)).all(axis=1))
    if rows.size == 0:
        raise RuntimeError("TimeSeriesSplit produced no out-of-fold rows for the ablation stack")
    features = np.full((len(arrays[0]), len(selected)), np.nan, dtype=np.float64)
    features[rows] = weighted_meta_features(
        {name: predictions[name][rows] for name in selected},
        weights,
        selected,
    )
    return features, rows


def _protocol_payload(
    config: ExperimentConfig,
    scenarios: tuple[AblationScenario, ...],
) -> dict[str, Any]:
    ranks = {
        str(name): int(rank)
        for name, rank in config.section("ranking")["reported_overall_ranks"].items()
    }
    reported_weights = {
        str(name): float(weight)
        for name, weight in config.section("wsaiec")["reported_static_weights"].items()
    }
    return {
        "datasets": list(ABLATION_DATASETS),
        "outer_split": "chronological 80% development and 20% untouched test",
        "weight_validation_split": "chronological final 20% of development",
        "meta_training": "eight-fold expanding-time-series out-of-fold rows plus weight-validation rows",
        "base_refit": "full development partition after all validation decisions",
        "meta_selection": "lowest frozen Table 6 overall rank among each Table 9 selection",
        "base_selection": "all selected classifiers except the resolved meta-classifier, preserving Table 9 order",
        "dynamic_weighting": "Equation 9 softmax of validation accuracy with dataset-and-scenario Bayesian alpha tuning over mean subject validation accuracy",
        "static_weighting": "printed Table 7 inverse-rank weights with the six-classifier normalization retained after omitting meta SVM",
        "overall_ranks": ranks,
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "scenario": scenario.scenario,
                "selected_classifiers": list(scenario.selected_classifiers),
                "meta_classifier": scenario.meta_classifier,
                "base_classifiers": list(scenario.base_classifiers),
                "weighting_method": scenario.weighting_method,
                "static_weights": (
                    static_rank_weights(reported_weights, scenario.base_classifiers)
                    if scenario.scenario_id == 11
                    else None
                ),
                "static_base_weight_sum": (
                    sum(static_rank_weights(reported_weights, scenario.base_classifiers).values())
                    if scenario.scenario_id == 11
                    else None
                ),
            }
            for scenario in scenarios
        ],
    }


def _scenario_weight_rows(
    prepared: SubjectCalibration,
    scenario: AblationScenario,
    alpha: float | None,
    fold_weights: dict[int, dict[str, float]],
    holdout_weights: dict[str, float],
    subject_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in sorted(prepared.fold_accuracies):
        n_meta_predictions = int(np.sum(prepared.fold_for_row == fold))
        for name in scenario.base_classifiers:
            rows.append(
                {
                    "dataset": prepared.dataset_name,
                    "subject": prepared.subject,
                    "scenario_id": scenario.scenario_id,
                    "scenario": scenario.scenario,
                    "alpha_scope": (
                        "not_applicable_static"
                        if scenario.scenario_id == 11
                        else "dataset_scenario"
                    ),
                    "subject_mode": subject_mode,
                    "stage": "time_series_oof",
                    "weight_source": (
                        "printed_table_7_static"
                        if scenario.scenario_id == 11
                        else "inner_chronological_validation"
                    ),
                    "epoch": fold,
                    "classifier": name,
                    "weighting_method": scenario.weighting_method,
                    "alpha": alpha,
                    "validation_accuracy": prepared.fold_accuracies[fold][name],
                    "weight": fold_weights[fold][name],
                    "n_validation": prepared.fold_weight_validation_sizes[fold],
                    "n_meta_predictions": n_meta_predictions,
                }
            )
    holdout_epoch = max(prepared.fold_accuracies, default=0) + 1
    for name in scenario.base_classifiers:
        rows.append(
            {
                "dataset": prepared.dataset_name,
                "subject": prepared.subject,
                "scenario_id": scenario.scenario_id,
                "scenario": scenario.scenario,
                "alpha_scope": (
                    "not_applicable_static"
                    if scenario.scenario_id == 11
                    else "dataset_scenario"
                ),
                "subject_mode": subject_mode,
                "stage": "alpha_validation",
                "weight_source": (
                    "printed_table_7_static"
                    if scenario.scenario_id == 11
                    else "development_holdout"
                ),
                "epoch": holdout_epoch,
                "classifier": name,
                "weighting_method": scenario.weighting_method,
                "alpha": alpha,
                "validation_accuracy": prepared.holdout_accuracies[name],
                "weight": holdout_weights[name],
                "n_validation": len(prepared.y_weight_validation),
                "n_meta_predictions": len(prepared.y_weight_validation),
            }
        )
    return rows


def _dynamic_alpha_objective(
    alpha: float,
    oof_predictions: dict[str, np.ndarray],
    holdout_predictions: dict[str, np.ndarray],
    fold_for_row: np.ndarray,
    fold_accuracies: dict[int, dict[str, float]],
    holdout_accuracies: dict[str, float],
    selected: tuple[str, ...],
    y_weight_train: np.ndarray,
    y_weight_validation: np.ndarray,
    wsaiec_settings: dict[str, Any],
    seed: int,
) -> float:
    selected_names = list(selected)
    candidate_features, candidate_rows, _ = _weighted_oof_features(
        {name: oof_predictions[name] for name in selected_names},
        fold_for_row,
        fold_accuracies,
        alpha,
        selected_names,
    )
    candidate_meta = make_meta_classifier(wsaiec_settings, seed)
    candidate_meta.fit(
        candidate_features[candidate_rows],
        y_weight_train[candidate_rows],
    )
    candidate_weights = dynamic_weights(
        holdout_accuracies,
        alpha,
        selected_names,
    )
    candidate_validation = weighted_meta_features(
        {name: holdout_predictions[name] for name in selected_names},
        candidate_weights,
        selected_names,
    )
    return float(
        accuracy_score(
            y_weight_validation,
            candidate_meta.predict(candidate_validation),
        )
    )


def _ablation_bases(scenarios: tuple[AblationScenario, ...]) -> list[str]:
    union = {name for scenario in scenarios for name in scenario.base_classifiers}
    return [name for name in CLASSIFIER_ORDER if name in union]


def _ablation_subject_paths(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
) -> tuple[Path, Path, Path, Path, Path]:
    directory = subject_result_directory(config, dataset_name, subject)
    return (
        directory / "ablation_metrics.csv",
        directory / "ablation_predictions.csv",
        directory / "ablation_alpha_history.csv",
        directory / "ablation_weight_history.csv",
        directory / "ablation_protocol.json",
    )


def _scenario_subject_score(
    prepared: SubjectCalibration,
    scenario: AblationScenario,
    alpha: float,
    wsaiec_settings: dict[str, Any],
) -> float:
    selected = list(scenario.base_classifiers)
    return _dynamic_alpha_objective(
        alpha,
        prepared.oof_predictions,
        prepared.holdout_predictions,
        prepared.fold_for_row,
        prepared.fold_accuracies,
        {name: prepared.holdout_accuracies[name] for name in selected},
        scenario.base_classifiers,
        prepared.y_weight_train,
        prepared.y_weight_validation,
        wsaiec_settings,
        prepared.base_seed + 900_000 + scenario.scenario_id * 10_000,
    )


def _static_scenario_subject_score(
    prepared: SubjectCalibration,
    scenario: AblationScenario,
    weights: dict[str, float],
    wsaiec_settings: dict[str, Any],
) -> float:
    selected = list(scenario.base_classifiers)
    oof_features, meta_rows = _fixed_oof_features(
        {name: prepared.oof_predictions[name] for name in selected},
        weights,
        selected,
    )
    validation_features = weighted_meta_features(
        {name: prepared.holdout_predictions[name] for name in selected},
        weights,
        selected,
    )
    meta = make_meta_classifier(
        wsaiec_settings,
        prepared.base_seed + 900_000 + scenario.scenario_id * 10_000,
    )
    meta.fit(oof_features[meta_rows], prepared.y_weight_train[meta_rows])
    return float(
        accuracy_score(
            prepared.y_weight_validation,
            meta.predict(validation_features),
        )
    )


def _optimize_dataset_scenario_alphas(
    config: ExperimentConfig,
    dataset_name: str,
    requested_subjects: list[int],
    prepared_by_subject: dict[int, SubjectCalibration],
    scenarios: tuple[AblationScenario, ...],
) -> tuple[dict[int, float | None], dict[int, list[dict[str, Any]]]]:
    wsaiec_settings = config.section("wsaiec")
    alpha_settings = wsaiec_settings["alpha_optimization"]
    if str(alpha_settings["scoring"]) != "accuracy":
        raise ValueError("Table 9 alpha optimization requires accuracy scoring")
    reported_weights = {
        str(name): float(value)
        for name, value in wsaiec_settings["reported_static_weights"].items()
    }
    alphas: dict[int, float | None] = {}
    histories: dict[int, list[dict[str, Any]]] = {}
    dataset_seed = (
        int(config.project["random_seed"])
        + list(config.datasets).index(dataset_name) * 1_000_000
        + 2_000_000
    )
    for scenario in scenarios:
        if scenario.scenario_id == 11:
            weights = static_rank_weights(reported_weights, scenario.base_classifiers)
            subject_scores = {
                subject: _static_scenario_subject_score(
                    prepared_by_subject[subject],
                    scenario,
                    weights,
                    wsaiec_settings,
                )
                for subject in requested_subjects
            }
            alphas[scenario.scenario_id] = None
            histories[scenario.scenario_id] = [
                {
                    "iteration": 1,
                    "alpha": float("nan"),
                    "score": float(np.mean(list(subject_scores.values()))),
                    "subject_scores": subject_scores,
                }
            ]
        else:
            best_alpha, history = optimize_shared_alpha(
                requested_subjects,
                lambda key, alpha, current=scenario: _scenario_subject_score(
                    prepared_by_subject[int(key)],
                    current,
                    alpha,
                    wsaiec_settings,
                ),
                alpha_settings,
                dataset_seed + scenario.scenario_id * 10_000,
            )
            alphas[scenario.scenario_id] = best_alpha
            histories[scenario.scenario_id] = history
    return alphas, histories


def _selected_scenario_history(
    scenario: AblationScenario,
    alpha: float | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if scenario.scenario_id == 11:
        if len(history) != 1:
            raise RuntimeError("Static-weight ablation must contain one audit record")
        return history[0]
    matches = [
        record
        for record in history
        if float(record["alpha"]) == float(alpha)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Scenario {scenario.scenario_id} has no unique selected alpha")
    return matches[0]


def _finalize_subject_ablations(
    config: ExperimentConfig,
    prepared: SubjectCalibration,
    scenarios: tuple[AblationScenario, ...],
    alphas: dict[int, float | None],
    histories: dict[int, list[dict[str, Any]]],
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
        alpha_history_path,
        weight_history_path,
        protocol_path,
    ) = _ablation_subject_paths(config, dataset_name, subject)
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
        raise RuntimeError("Ablation outer test partition does not contain every class")
    csp_settings, standardize = _feature_settings(config)
    full_transformer = CSPFeatureTransformer(csp_settings, standardize)
    full_train_features = full_transformer.fit_transform(X_development, y_development)
    test_features = full_transformer.transform(X_test)
    all_bases = _ablation_bases(scenarios)
    classifier_config = config.section("classifiers")
    n_jobs = int(config.project["n_jobs"])
    test_predictions: dict[str, np.ndarray] = {}
    for classifier_index, name in enumerate(all_bases):
        model = make_classifier(
            name,
            classifier_config,
            prepared.base_seed + 800_000 + classifier_index,
            n_jobs,
        )
        model.fit(full_train_features, y_development)
        test_predictions[name] = np.asarray(
            model.predict(test_features),
            dtype=np.int64,
        )
    wsaiec_settings = config.section("wsaiec")
    reported_weights = {
        str(name): float(value)
        for name, value in wsaiec_settings["reported_static_weights"].items()
    }
    cohort_payload = {
        "configured_subjects": json.dumps(configured_subjects),
        "requested_subjects": json.dumps(requested_subjects),
        "n_alpha_subjects": len(requested_subjects),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        selected = list(scenario.base_classifiers)
        best_alpha = alphas[scenario.scenario_id]
        history = histories[scenario.scenario_id]
        scenario_holdout_accuracies = {
            name: prepared.holdout_accuracies[name] for name in selected
        }
        if scenario.scenario_id == 11:
            final_weights = static_rank_weights(
                reported_weights,
                scenario.base_classifiers,
            )
            oof_features, meta_rows = _fixed_oof_features(
                {name: prepared.oof_predictions[name] for name in selected},
                final_weights,
                selected,
            )
            fold_weights = {
                fold: dict(final_weights) for fold in prepared.fold_accuracies
            }
        else:
            if best_alpha is None:
                raise RuntimeError(f"Scenario {scenario.scenario_id} lacks a dynamic alpha")
            oof_features, meta_rows, fold_weights = _weighted_oof_features(
                {name: prepared.oof_predictions[name] for name in selected},
                prepared.fold_for_row,
                prepared.fold_accuracies,
                best_alpha,
                selected,
            )
            final_weights = dynamic_weights(
                scenario_holdout_accuracies,
                best_alpha,
                selected,
            )
        validation_features = weighted_meta_features(
            {name: prepared.holdout_predictions[name] for name in selected},
            final_weights,
            selected,
        )
        meta_X = np.vstack([oof_features[meta_rows], validation_features])
        meta_y = np.concatenate(
            [prepared.y_weight_train[meta_rows], prepared.y_weight_validation]
        )
        if not np.array_equal(np.unique(meta_y), prepared.classes):
            raise RuntimeError(
                f"Scenario {scenario.scenario_id} meta-training rows omit a class"
            )
        scenario_seed = prepared.base_seed + 900_000 + scenario.scenario_id * 10_000
        meta = make_meta_classifier(wsaiec_settings, scenario_seed + 1_000)
        meta.fit(meta_X, meta_y)
        meta_test = weighted_meta_features(
            {name: test_predictions[name] for name in selected},
            final_weights,
            selected,
        )
        probabilities = aligned_probabilities(meta, meta_test, prepared.classes)
        predicted = np.asarray(meta.predict(meta_test), dtype=np.int64)
        metrics = classification_metrics(
            y_test,
            predicted,
            probabilities,
            config.section("evaluation"),
        )
        selected_record = _selected_scenario_history(scenario, best_alpha, history)
        subject_scores = {
            int(key): float(value)
            for key, value in selected_record["subject_scores"].items()
        }
        alpha_scope = (
            "not_applicable_static"
            if scenario.scenario_id == 11
            else "dataset_scenario"
        )
        metric_rows.append(
            {
                "dataset": dataset_name,
                "subject": subject,
                "scenario_id": scenario.scenario_id,
                "scenario": scenario.scenario,
                "selected_classifiers": scenario.selected_text,
                "meta_classifier": scenario.meta_classifier,
                "base_classifiers": ";".join(scenario.base_classifiers),
                "weighting_method": scenario.weighting_method,
                "alpha_scope": alpha_scope,
                "subject_mode": subject_mode,
                **cohort_payload,
                "n_train": len(y_development),
                "n_weight_train": len(prepared.y_weight_train),
                "n_weight_validation": len(prepared.y_weight_validation),
                "n_meta_train": len(meta_y),
                "n_eval": len(y_test),
                "alpha": best_alpha,
                "alpha_validation_accuracy": subject_scores[subject],
                "dataset_scenario_validation_accuracy": float(selected_record["score"]),
                "validation_accuracies": json.dumps(
                    scenario_holdout_accuracies,
                    sort_keys=True,
                ),
                "weights": json.dumps(final_weights, sort_keys=True),
                "weight_sum": float(sum(final_weights.values())),
                **metrics,
            }
        )
        for row_index, (trial, true, prediction, probability) in enumerate(
            zip(outer_split.test, y_test, predicted, probabilities, strict=True)
        ):
            prediction_rows.append(
                {
                    "sample_key": f"{dataset_name}:s{subject:02d}:trial{int(trial):05d}",
                    "dataset": dataset_name,
                    "subject": subject,
                    "trial_index": int(trial),
                    "scenario_id": scenario.scenario_id,
                    "scenario": scenario.scenario,
                    "alpha_scope": alpha_scope,
                    "alpha": best_alpha,
                    "y_true": int(true),
                    "y_pred": int(prediction),
                    "probabilities": json.dumps([float(value) for value in probability]),
                    "base_predictions": json.dumps(
                        {
                            name: int(test_predictions[name][row_index])
                            for name in selected
                        },
                        sort_keys=True,
                    ),
                    "weights": json.dumps(final_weights, sort_keys=True),
                    "meta_features": json.dumps(
                        [float(value) for value in meta_test[row_index]]
                    ),
                }
            )
        for record in history:
            record_scores = {
                int(key): float(value)
                for key, value in record["subject_scores"].items()
            }
            alpha_rows.append(
                {
                    "dataset": dataset_name,
                    "subject": subject,
                    "scenario_id": scenario.scenario_id,
                    "scenario": scenario.scenario,
                    "weighting_method": scenario.weighting_method,
                    "alpha_scope": alpha_scope,
                    "subject_mode": subject_mode,
                    **cohort_payload,
                    "iteration": int(record["iteration"]),
                    "alpha": float(record["alpha"]),
                    "subject_validation_accuracy": record_scores[subject],
                    "mean_validation_accuracy": float(record["score"]),
                    "subject_validation_accuracies": json.dumps(
                        record_scores,
                        sort_keys=True,
                    ),
                    "selected": (
                        scenario.scenario_id == 11
                        or float(record["alpha"]) == float(best_alpha)
                    ),
                }
            )
        weight_rows.extend(
            _scenario_weight_rows(
                prepared,
                scenario,
                best_alpha,
                fold_weights,
                final_weights,
                subject_mode,
            )
        )
    write_csv(metrics_path, pd.DataFrame(metric_rows))
    write_csv(predictions_path, pd.DataFrame(prediction_rows))
    write_csv(alpha_history_path, pd.DataFrame(alpha_rows))
    write_csv(weight_history_path, pd.DataFrame(weight_rows))
    write_json(protocol_path, _protocol_payload(config, scenarios))
    return metrics_path, predictions_path


def _dataset_ablation_history_rows(
    dataset_name: str,
    scenarios: tuple[AblationScenario, ...],
    alphas: dict[int, float | None],
    histories: dict[int, list[dict[str, Any]]],
    configured_subjects: list[int],
    requested_subjects: list[int],
    subject_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        best_alpha = alphas[scenario.scenario_id]
        for record in histories[scenario.scenario_id]:
            subject_scores = {
                int(key): float(value)
                for key, value in record["subject_scores"].items()
            }
            rows.append(
                {
                    "dataset": dataset_name,
                    "scenario_id": scenario.scenario_id,
                    "scenario": scenario.scenario,
                    "alpha_scope": (
                        "not_applicable_static"
                        if scenario.scenario_id == 11
                        else "dataset_scenario"
                    ),
                    "subject_mode": subject_mode,
                    "configured_subjects": json.dumps(configured_subjects),
                    "requested_subjects": json.dumps(requested_subjects),
                    "n_subjects": len(requested_subjects),
                    "iteration": int(record["iteration"]),
                    "alpha": float(record["alpha"]),
                    "mean_validation_accuracy": float(record["score"]),
                    "subject_validation_accuracies": json.dumps(
                        subject_scores,
                        sort_keys=True,
                    ),
                    "selected": (
                        scenario.scenario_id == 11
                        or float(record["alpha"]) == float(best_alpha)
                    ),
                }
            )
    return rows


def _ablation_outputs_reusable(
    config: ExperimentConfig,
    dataset_name: str,
    configured_subjects: list[int],
    requested_subjects: list[int],
    subject_mode: str,
) -> bool:
    history_path = config.run_root / dataset_name / "ablation_alpha_history.csv"
    required = [
        path
        for subject in requested_subjects
        for path in _ablation_subject_paths(config, dataset_name, subject)
    ]
    if not history_path.exists() or not all(path.exists() for path in required):
        return False
    try:
        history = pd.read_csv(history_path)
        if history.empty:
            return False
        first = history.iloc[0]
        selected_counts = history.groupby("scenario_id")["selected"].sum()
        return (
            str(first["dataset"]) == dataset_name
            and str(first["subject_mode"]) == subject_mode
            and json.loads(str(first["configured_subjects"])) == configured_subjects
            and json.loads(str(first["requested_subjects"])) == requested_subjects
            and int(first["n_subjects"]) == len(requested_subjects)
            and len(selected_counts) == 11
            and (selected_counts == 1).all()
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_subject_ablations(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
    force: bool = False,
) -> tuple[Path, Path]:
    return run_dataset_ablations(
        config,
        dataset_name,
        subjects=[subject],
        force=force,
    )[0]


def run_dataset_ablations(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: list[int] | None = None,
    force: bool = False,
) -> list[tuple[Path, Path]]:
    if dataset_name not in ABLATION_DATASETS:
        raise ValueError(f"Table 9 ablations are limited to {list(ABLATION_DATASETS)}")
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
    if not force and _ablation_outputs_reusable(
        config,
        dataset_name,
        configured_subjects,
        requested_subjects,
        subject_mode,
    ):
        return [
            _ablation_subject_paths(config, dataset_name, subject)[:2]
            for subject in requested_subjects
        ]
    scenarios = load_ablation_scenarios(config)
    all_bases = _ablation_bases(scenarios)
    prepared_by_subject = {
        subject: _prepare_subject_predictions(
            config,
            dataset_name,
            subject,
            all_bases,
            600_000,
            700_000,
        )
        for subject in requested_subjects
    }
    alphas, histories = _optimize_dataset_scenario_alphas(
        config,
        dataset_name,
        requested_subjects,
        prepared_by_subject,
        scenarios,
    )
    outputs = [
        _finalize_subject_ablations(
            config,
            prepared_by_subject[subject],
            scenarios,
            alphas,
            histories,
            configured_subjects,
            requested_subjects,
            subject_mode,
        )
        for subject in requested_subjects
    ]
    dataset_history_path = (
        ensure_directory(config.run_root / dataset_name) / "ablation_alpha_history.csv"
    )
    write_csv(
        dataset_history_path,
        pd.DataFrame(
            _dataset_ablation_history_rows(
                dataset_name,
                scenarios,
                alphas,
                histories,
                configured_subjects,
                requested_subjects,
                subject_mode,
            )
        ),
    )
    return outputs


def _collect_ablation_metrics(config: ExperimentConfig) -> pd.DataFrame:
    expected = {
        config.run_root / dataset / f"subject_{int(subject):02d}" / "ablation_metrics.csv": (
            dataset,
            int(subject),
        )
        for dataset in ABLATION_DATASETS
        for subject in config.datasets[dataset]["subjects"]
    }
    actual = {
        path
        for dataset in ABLATION_DATASETS
        for path in (config.run_root / dataset).glob("subject_*/ablation_metrics.csv")
    }
    if actual != set(expected):
        missing = sorted(str(path.relative_to(config.run_root)) for path in set(expected) - actual)
        extra = sorted(str(path.relative_to(config.run_root)) for path in actual - set(expected))
        raise FileNotFoundError(
            f"Incomplete Table 9 cohort below {config.run_root}; missing={missing}, extra={extra}"
        )
    frames: list[pd.DataFrame] = []
    for path, (dataset, subject) in expected.items():
        frame = pd.read_csv(path)
        identities = set(
            map(tuple, frame[["dataset", "subject"]].itertuples(index=False, name=None))
        )
        if identities != {(dataset, subject)}:
            raise ValueError(f"Rows in {path} do not match its directory")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ablation_metrics_exist(config: ExperimentConfig) -> bool:
    return any(
        path
        for dataset in ABLATION_DATASETS
        for path in (config.run_root / dataset).glob("subject_*/ablation_metrics.csv")
    )


def aggregate_ablations(config: ExperimentConfig) -> Path:
    scenarios = load_ablation_scenarios(config)
    metrics = _collect_ablation_metrics(config)
    expected = {
        (dataset, int(subject), scenario.scenario_id)
        for dataset in ABLATION_DATASETS
        for subject in config.datasets[dataset]["subjects"]
        for scenario in scenarios
    }
    if metrics.duplicated(["dataset", "subject", "scenario_id"]).any():
        raise ValueError("Ablation metrics contain duplicate subject-scenario rows")
    actual = set(
        map(
            tuple,
            metrics[["dataset", "subject", "scenario_id"]].itertuples(
                index=False,
                name=None,
            ),
        )
    )
    if actual != expected:
        raise ValueError(
            f"Ablation metrics are incomplete; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    numeric = metrics[[*METRIC_COLUMNS, "score"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Ablation metrics contain missing or non-finite scores")
    weight_sums = pd.to_numeric(metrics["weight_sum"], errors="coerce")
    expected_weight_sums = np.where(metrics["scenario_id"].astype(int) == 11, 0.4588, 1.0)
    if weight_sums.isna().any() or not np.allclose(
        weight_sums.to_numpy(dtype=float),
        expected_weight_sums,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Ablation metrics contain invalid dynamic or static weight sums")
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    for row in metrics.itertuples(index=False):
        scenario = scenario_map[int(row.scenario_id)]
        if (
            str(row.scenario) != scenario.scenario
            or str(row.selected_classifiers) != scenario.selected_text
            or str(row.meta_classifier) != scenario.meta_classifier
            or str(row.base_classifiers) != ";".join(scenario.base_classifiers)
            or str(row.weighting_method) != scenario.weighting_method
        ):
            raise ValueError(f"Ablation scenario metadata differs for scenario {row.scenario_id}")
    means = metrics.groupby(["scenario_id", "dataset"], as_index=False)["accuracy"].mean()
    pivoted = means.pivot(index="scenario_id", columns="dataset", values="accuracy")
    rows = [
        {
            "scenario_id": scenario.scenario_id,
            "scenario": scenario.scenario,
            "selected_classifiers": scenario.selected_text,
            "accuracy_bnci2014_002": float(
                pivoted.loc[scenario.scenario_id, "BNCI2014_002"]
            ),
            "accuracy_zhou2016": float(pivoted.loc[scenario.scenario_id, "Zhou2016"]),
        }
        for scenario in scenarios
    ]
    output = ensure_directory(config.publication_generated)
    path = output / "ablation.csv"
    write_csv(path, pd.DataFrame(rows, columns=ABLATION_SOURCE_COLUMNS))
    write_csv(
        output / "ablation_subject_metrics.csv",
        metrics.sort_values(["scenario_id", "dataset", "subject"]),
    )
    write_json(output / "ablation_protocol.json", _protocol_payload(config, scenarios))
    return path
