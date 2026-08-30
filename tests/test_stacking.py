from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from wsaiec_eeg.evaluation import stacking
from wsaiec_eeg.features.csp import FoldFeatures


class _IdentityTransformer:
    fit_sizes: list[int] = []

    def __init__(self, settings: dict[str, Any], standardize: bool) -> None:
        self.settings = settings
        self.standardize = standardize

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit_sizes.append(len(y))
        return np.asarray(X).reshape(len(X), -1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X).reshape(len(X), -1)


class _FeaturePredictionClassifier:
    fit_sizes: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> _FeaturePredictionClassifier:
        self.fit_sizes.append(len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, 0].astype(np.int64)


def test_oof_weights_use_inner_earlier_validation_labels(monkeypatch) -> None:
    y = np.arange(12, dtype=np.int64) % 2
    X = y.astype(np.float64).reshape(-1, 1, 1)
    X[8:, 0, 0] = 1 - y[8:]
    train = np.arange(8, dtype=np.int64)
    valid = np.arange(8, 12, dtype=np.int64)
    fold = FoldFeatures(
        fold=1,
        train_index=train,
        valid_index=valid,
        X_train=X[train].reshape(len(train), -1),
        y_train=y[train],
        X_valid=X[valid].reshape(len(valid), -1),
        y_valid=y[valid],
    )
    _IdentityTransformer.fit_sizes = []
    _FeaturePredictionClassifier.fit_sizes = []
    monkeypatch.setattr(stacking, "CSPFeatureTransformer", _IdentityTransformer)
    monkeypatch.setattr(
        stacking,
        "make_classifier",
        lambda name, settings, seed, n_jobs: _FeaturePredictionClassifier(),
    )

    predictions, fold_for_row, accuracies = stacking._base_oof_predictions(
        [fold],
        len(y),
        ["A"],
        {},
        10,
        1,
        X=X,
        y=y,
        csp_settings={},
        standardize=False,
        validation_fraction=0.20,
    )

    assert accuracies == {1: {"A": 1.0}}
    np.testing.assert_array_equal(predictions["A"][valid], 1 - y[valid])
    np.testing.assert_array_equal(fold_for_row[:8], np.full(8, -1))
    np.testing.assert_array_equal(fold_for_row[valid], np.ones(4))
    assert _IdentityTransformer.fit_sizes == [6]
    assert _FeaturePredictionClassifier.fit_sizes == [6, 8]


def test_inner_weight_validation_need_not_contain_every_class() -> None:
    y = np.asarray([0, 1, 2, 0, 1, 2])
    inner_train, inner_valid = stacking._inner_weight_indices(
        np.arange(len(y)),
        y,
        0.20,
    )
    np.testing.assert_array_equal(inner_train, [0, 1, 2, 3])
    np.testing.assert_array_equal(inner_valid, [4, 5])
    np.testing.assert_array_equal(np.unique(y[inner_train]), [0, 1, 2])


def test_dataset_orchestration_calibrates_once_for_requested_cohort(
    monkeypatch,
    paper_config,
) -> None:
    prepared: list[int] = []
    finalized: list[tuple[int, float, str]] = []
    written: list[tuple[Path, Any]] = []

    def prepare(config, dataset_name: str, subject: int) -> SimpleNamespace:
        prepared.append(subject)
        return SimpleNamespace(subject=subject)

    def optimize(keys, objective, settings, seed):
        scores = {key: objective(key, 2.5) for key in keys}
        return 2.5, [
            {
                "iteration": 1,
                "alpha": 2.5,
                "score": float(np.mean(list(scores.values()))),
                "subject_scores": scores,
            }
        ]

    def finalize(
        config,
        subject_data,
        alpha,
        history,
        configured_subjects,
        requested_subjects,
        subject_mode,
    ):
        finalized.append((subject_data.subject, alpha, subject_mode))
        return Path(f"metrics-{subject_data.subject}"), Path(f"predictions-{subject_data.subject}")

    monkeypatch.setattr(stacking, "_dataset_outputs_reusable", lambda *args: False)
    monkeypatch.setattr(stacking, "_prepare_subject_calibration", prepare)
    monkeypatch.setattr(
        stacking,
        "_subject_alpha_validation_accuracy",
        lambda subject_data, alpha, settings: subject_data.subject / 10 + alpha / 100,
    )
    monkeypatch.setattr(stacking, "optimize_shared_alpha", optimize)
    monkeypatch.setattr(stacking, "_finalize_subject_wsaiec", finalize)
    monkeypatch.setattr(stacking, "ensure_directory", lambda path: Path(path))
    monkeypatch.setattr(stacking, "write_csv", lambda path, frame: written.append((Path(path), frame)))

    outputs = stacking.run_dataset_wsaiec(
        paper_config,
        "BNCI2014_001",
        subjects=[1, 2],
        force=True,
    )

    assert prepared == [1, 2]
    assert finalized == [
        (1, 2.5, "debug_subject_subset"),
        (2, 2.5, "debug_subject_subset"),
    ]
    assert outputs == [
        (Path("metrics-1"), Path("predictions-1")),
        (Path("metrics-2"), Path("predictions-2")),
    ]
    history_path, history = written[-1]
    assert history_path.name == "wsaiec_alpha_history.csv"
    assert history.loc[0, "alpha_scope"] == "dataset"
    assert history.loc[0, "subject_mode"] == "debug_subject_subset"
    assert history.loc[0, "requested_subjects"] == "[1, 2]"
    assert history.loc[0, "n_subjects"] == 2
    assert np.isclose(history.loc[0, "mean_validation_accuracy"], 0.175)


def test_full_configured_cohort_is_recorded_as_paper_mode(monkeypatch, paper_config) -> None:
    modes: list[str] = []
    monkeypatch.setattr(stacking, "_dataset_outputs_reusable", lambda *args: False)
    monkeypatch.setattr(
        stacking,
        "_prepare_subject_calibration",
        lambda config, dataset_name, subject: SimpleNamespace(subject=subject),
    )
    monkeypatch.setattr(
        stacking,
        "_subject_alpha_validation_accuracy",
        lambda subject_data, alpha, settings: 0.5,
    )
    monkeypatch.setattr(
        stacking,
        "optimize_shared_alpha",
        lambda keys, objective, settings, seed: (
            1.0,
            [
                {
                    "iteration": 1,
                    "alpha": 1.0,
                    "score": 0.5,
                    "subject_scores": {key: objective(key, 1.0) for key in keys},
                }
            ],
        ),
    )
    monkeypatch.setattr(
        stacking,
        "_finalize_subject_wsaiec",
        lambda config, data, alpha, history, configured, requested, mode: (
            modes.append(mode) or Path(f"m-{data.subject}"),
            Path(f"p-{data.subject}"),
        ),
    )
    monkeypatch.setattr(stacking, "ensure_directory", lambda path: Path(path))
    monkeypatch.setattr(stacking, "write_csv", lambda path, frame: None)

    stacking.run_dataset_wsaiec(
        paper_config,
        "Zhou2016",
        subjects=None,
        force=True,
    )

    assert modes == ["paper_all_subjects"] * 4
