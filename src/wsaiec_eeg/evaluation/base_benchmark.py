"""Subject-wise benchmark of all 16 classifiers and the training-size analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.constants import DATASET_ORDER
from wsaiec_eeg.data.cache import load_cached_subject
from wsaiec_eeg.data.splits import make_outer_split, time_series_splits
from wsaiec_eeg.evaluation.metrics import classification_metrics
from wsaiec_eeg.features.csp import CSPFeatureTransformer, build_fold_features
from wsaiec_eeg.models.classifiers import aligned_probabilities, make_classifier
from wsaiec_eeg.models.tuning import tune_classifier
from wsaiec_eeg.utils.io import ensure_directory, write_csv, write_json


def subject_result_directory(config: ExperimentConfig, dataset: str, subject: int) -> Path:
    return config.run_root / dataset / f"subject_{subject:02d}"


def dataset_result_directory(config: ExperimentConfig, dataset: str) -> Path:
    return config.run_root / dataset


def _feature_settings(config: ExperimentConfig) -> tuple[dict[str, Any], bool]:
    features = config.section("features")
    return dict(features["csp"]), bool(features["standardize"])


def _classes(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Expected contiguous integer class labels")
    return classes


def _prediction_rows(
    dataset: str,
    subject: int,
    classifier: str,
    trial_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_key": f"{dataset}:s{subject:02d}:trial{int(trial):05d}",
            "dataset": dataset,
            "subject": subject,
            "trial_index": int(trial),
            "classifier": classifier,
            "y_true": int(true),
            "y_pred": int(predicted),
            "probabilities": json.dumps([float(value) for value in probability]),
        }
        for trial, true, predicted, probability in zip(
            trial_indices, y_true, y_pred, probabilities, strict=True
        )
    ]


def run_subject_benchmark(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
    force: bool = False,
) -> tuple[Path, Path]:
    """Run all base models for one participant and write atomic result tables."""

    directory = ensure_directory(subject_result_directory(config, dataset_name, subject))
    metrics_path = directory / "base_metrics.csv"
    predictions_path = directory / "base_predictions.csv"
    tuning_path = directory / "tuning.json"
    if metrics_path.exists() and predictions_path.exists() and not force:
        return metrics_path, predictions_path

    cached = load_cached_subject(config, dataset_name, subject)
    split_settings = config.section("splitting")
    split = make_outer_split(
        cached.y,
        str(split_settings["outer_strategy"]),
        float(split_settings["outer_test_fraction"]),
    )
    X_train, y_train = cached.X[split.train], cached.y[split.train]
    X_test, y_test = cached.X[split.test], cached.y[split.test]
    classes = _classes(cached.y)
    csp_settings, standardize = _feature_settings(config)
    folds = build_fold_features(
        X_train,
        y_train,
        csp_settings,
        standardize,
        int(split_settings["time_series_splits"]),
        int(split_settings["gap"]),
    )
    full_transformer = CSPFeatureTransformer(csp_settings, standardize)
    full_train_features = full_transformer.fit_transform(X_train, y_train)
    test_features = full_transformer.transform(X_test)

    project = config.project
    seed = int(project["random_seed"]) + subject * 10_000
    n_jobs = int(project["n_jobs"])
    classifier_config = config.section("classifiers")
    tuning_config = config.section("tuning")
    evaluation = config.section("evaluation")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    tuning_records: dict[str, Any] = {}

    for classifier_index, name in enumerate(classifier_config["order"]):
        effective_config = classifier_config
        if tuning_config["enabled"]:
            best, records = tune_classifier(
                name,
                folds,
                classifier_config,
                tuning_config,
                seed + classifier_index * 100,
                n_jobs,
            )
            effective_config = {**classifier_config, name: best}
            tuning_records[name] = {"best": best, "trials": records}

        for fold in folds:
            model = make_classifier(
                name,
                effective_config,
                seed + classifier_index * 100 + fold.fold,
                n_jobs,
            )
            model.fit(fold.X_train, fold.y_train)
            probabilities = aligned_probabilities(model, fold.X_valid, classes)
            predicted = np.asarray(model.predict(fold.X_valid))
            row = {
                "dataset": dataset_name,
                "subject": subject,
                "classifier": name,
                "scope": "time_series_validation",
                "fold": fold.fold,
                "training_fraction": 1.0,
                "n_train": len(fold.y_train),
                "n_eval": len(fold.y_valid),
                **classification_metrics(fold.y_valid, predicted, probabilities, evaluation),
            }
            metric_rows.append(row)

        final_model = make_classifier(
            name, effective_config, seed + classifier_index * 100 + 99, n_jobs
        )
        final_model.fit(full_train_features, y_train)
        probabilities = aligned_probabilities(final_model, test_features, classes)
        predicted = np.asarray(final_model.predict(test_features))
        metric_rows.append(
            {
                "dataset": dataset_name,
                "subject": subject,
                "classifier": name,
                "scope": "outer_test",
                "fold": 0,
                "training_fraction": 1.0,
                "n_train": len(y_train),
                "n_eval": len(y_test),
                **classification_metrics(y_test, predicted, probabilities, evaluation),
            }
        )
        prediction_rows.extend(
            _prediction_rows(
                dataset_name,
                subject,
                name,
                split.test,
                y_test,
                predicted,
                probabilities,
            )
        )

    write_csv(metrics_path, pd.DataFrame(metric_rows))
    write_csv(predictions_path, pd.DataFrame(prediction_rows))
    write_json(tuning_path, tuning_records)
    return metrics_path, predictions_path


def run_dataset_benchmark(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: list[int] | None = None,
    force: bool = False,
) -> list[tuple[Path, Path]]:
    requested = subjects or list(config.datasets[dataset_name]["subjects"])
    return [run_subject_benchmark(config, dataset_name, subject, force) for subject in requested]


def expected_full_training_volume(config: ExperimentConfig, dataset_name: str) -> int:
    specification = config.datasets[dataset_name]
    trials_per_subject = int(specification["trials_per_class"]) * len(
        specification["events"]
    )
    test_count = int(
        math.ceil(
            trials_per_subject
            * float(config.section("splitting")["outer_test_fraction"])
        )
    )
    return (trials_per_subject - test_count) * len(specification["subjects"])


def expected_training_volumes(
    config: ExperimentConfig, dataset_name: str
) -> dict[float, int]:
    specification = config.datasets[dataset_name]
    subject_count = len(specification["subjects"])
    full_volume = expected_full_training_volume(config, dataset_name)
    development_per_subject = full_volume // subject_count
    return {
        float(fraction): _prefix_count(development_per_subject, float(fraction))
        * subject_count
        for fraction in config.section("splitting")["training_fractions"]
    }


def _prefix_count(length: int, fraction: float) -> int:
    if fraction >= 1.0:
        return length
    return max(1, int(math.ceil(length * fraction)))


def _learning_curve_complete(
    frame: pd.DataFrame,
    dataset_name: str,
    classifiers: list[str],
    fractions: list[float],
    folds: int,
    training_volumes: dict[float, int],
    subject_count: int,
) -> bool:
    required = {
        "dataset",
        "classifier",
        "training_fraction",
        "training_volume",
        "fold",
        "fold_fit_count",
        "fold_validation_count",
        "subject_count",
        "train_score",
        "validation_score",
    }
    if frame.empty or required - set(frame):
        return False
    keys = ["dataset", "classifier", "training_fraction", "fold"]
    if frame.duplicated(keys).any():
        return False
    expected = {
        (dataset_name, classifier, fraction, fold)
        for classifier in classifiers
        for fraction in fractions
        for fold in range(1, folds + 1)
    }
    actual = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
    if actual != expected:
        return False
    if set(frame["subject_count"].astype(int)) != {subject_count}:
        return False
    for fraction, expected_volume in training_volumes.items():
        rows = frame.loc[np.isclose(frame["training_fraction"], fraction)]
        if set(rows["training_volume"].astype(int)) != {expected_volume}:
            return False
    counts = frame[["fold_fit_count", "fold_validation_count"]].apply(
        pd.to_numeric, errors="coerce"
    )
    scores = frame[["train_score", "validation_score"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return bool(
        counts.notna().all().all()
        and (counts > 0).all().all()
        and scores.notna().all().all()
        and np.isfinite(scores.to_numpy()).all()
        and ((scores >= 0.0) & (scores <= 1.0)).all().all()
    )


def run_dataset_learning_curves(
    config: ExperimentConfig,
    dataset_name: str,
    force: bool = False,
) -> Path:
    """Evaluate pooled dataset learning curves from chronological subject prefixes."""

    directory = ensure_directory(dataset_result_directory(config, dataset_name))
    output_path = directory / "learning_curve.csv"
    metadata_path = directory / "learning_curve_metadata.json"
    split_settings = config.section("splitting")
    subjects = [int(subject) for subject in config.datasets[dataset_name]["subjects"]]
    development_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    development_counts: dict[str, int] = {}
    for subject in subjects:
        cached = load_cached_subject(config, dataset_name, subject)
        split = make_outer_split(
            cached.y,
            str(split_settings["outer_strategy"]),
            float(split_settings["outer_test_fraction"]),
        )
        development_X = cached.X[split.train]
        development_y = cached.y[split.train]
        development_blocks.append((development_X, development_y))
        development_counts[str(subject)] = len(development_y)

    expected_volume = expected_full_training_volume(config, dataset_name)
    observed_volume = sum(development_counts.values())
    if observed_volume != expected_volume:
        raise RuntimeError(
            f"{dataset_name} pooled development volume is {observed_volume}, "
            f"expected {expected_volume} from the executable dataset specification"
        )

    fractions = [float(value) for value in split_settings["training_fractions"]]
    prefix_counts = {
        fraction: {
            str(subject): _prefix_count(development_counts[str(subject)], fraction)
            for subject in subjects
        }
        for fraction in fractions
    }
    training_volumes = {
        fraction: sum(counts.values()) for fraction, counts in prefix_counts.items()
    }
    configured_volumes = expected_training_volumes(config, dataset_name)
    if training_volumes != configured_volumes:
        raise RuntimeError(
            f"{dataset_name} pooled prefix volumes {training_volumes} do not match "
            f"the executable specification {configured_volumes}"
        )
    classifiers = [str(name) for name in config.section("classifiers")["order"]]
    n_splits = int(split_settings["time_series_splits"])
    expected_metadata = {
        "dataset": dataset_name,
        "subjects": subjects,
        "development_counts": development_counts,
        "training_fractions": fractions,
        "subject_prefix_counts": {
            str(fraction): counts for fraction, counts in prefix_counts.items()
        },
        "training_volumes": {
            str(fraction): volume for fraction, volume in training_volumes.items()
        },
        "expected_full_training_volume": expected_volume,
        "row_count": len(classifiers) * len(fractions) * n_splits,
    }
    if output_path.exists() and metadata_path.exists() and not force:
        existing = pd.read_csv(output_path)
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_metadata = None
        if existing_metadata == expected_metadata and _learning_curve_complete(
            existing,
            dataset_name,
            classifiers,
            fractions,
            n_splits,
            training_volumes,
            len(subjects),
        ):
            return output_path

    csp_settings, standardize = _feature_settings(config)
    classifier_config = config.section("classifiers")
    dataset_index = DATASET_ORDER.index(dataset_name)
    base_seed = int(config.project["random_seed"]) + dataset_index * 1_000_000
    n_jobs = int(config.project["n_jobs"])
    expected_classes = np.arange(len(config.datasets[dataset_name]["events"]))
    rows: list[dict[str, Any]] = []
    for fraction_index, fraction in enumerate(fractions):
        pooled_X = np.concatenate(
            [
                X[: prefix_counts[fraction][str(subject)]]
                for subject, (X, _) in zip(subjects, development_blocks, strict=True)
            ],
            axis=0,
        )
        pooled_y = np.concatenate(
            [
                y[: prefix_counts[fraction][str(subject)]]
                for subject, (_, y) in zip(subjects, development_blocks, strict=True)
            ]
        )
        if len(pooled_y) != training_volumes[fraction]:
            raise RuntimeError("Pooled prefix volume does not match its recorded training volume")
        if not np.array_equal(np.unique(pooled_y), expected_classes):
            raise ValueError(
                f"{dataset_name} pooled prefix at fraction {fraction} omits a class"
            )
        folds = time_series_splits(
            len(pooled_y),
            n_splits,
            int(split_settings["gap"]),
        )
        for fold_index, (fold_train, fold_valid) in enumerate(folds, start=1):
            if not np.array_equal(np.unique(pooled_y[fold_train]), expected_classes):
                raise ValueError(
                    f"{dataset_name} fraction {fraction} fold {fold_index} "
                    "training window omits a class"
                )
            transformer = CSPFeatureTransformer(csp_settings, standardize)
            train_features = transformer.fit_transform(pooled_X[fold_train], pooled_y[fold_train])
            valid_features = transformer.transform(pooled_X[fold_valid])
            for classifier_index, name in enumerate(classifiers):
                model = make_classifier(
                    name,
                    classifier_config,
                    base_seed
                    + fraction_index * 10_000
                    + fold_index * 100
                    + classifier_index,
                    n_jobs,
                )
                model.fit(train_features, pooled_y[fold_train])
                train_predicted = np.asarray(model.predict(train_features))
                valid_predicted = np.asarray(model.predict(valid_features))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "classifier": name,
                        "training_fraction": fraction,
                        "training_volume": training_volumes[fraction],
                        "fold": fold_index,
                        "fold_fit_count": len(fold_train),
                        "fold_validation_count": len(fold_valid),
                        "subject_count": len(subjects),
                        "train_score": float(
                            np.mean(train_predicted == pooled_y[fold_train])
                        ),
                        "validation_score": float(
                            np.mean(valid_predicted == pooled_y[fold_valid])
                        ),
                    }
                )
    write_csv(output_path, pd.DataFrame(rows))
    if len(rows) != expected_metadata["row_count"]:
        raise RuntimeError("Learning-curve row count does not match the complete experiment grid")
    write_json(metadata_path, expected_metadata)
    return output_path
