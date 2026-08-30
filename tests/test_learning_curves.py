from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from wsaiec_eeg.constants import PAPER_FULL_TRAINING_VOLUMES
from wsaiec_eeg.evaluation import base_benchmark
from wsaiec_eeg.evaluation.base_benchmark import (
    expected_full_training_volume,
    run_dataset_learning_curves,
)


class IdentityTransformer:
    def __init__(self, settings, standardize) -> None:
        self.settings = settings
        self.standardize = standardize

    def fit_transform(self, X, y):
        return X.reshape(len(X), -1)

    def transform(self, X):
        return X.reshape(len(X), -1)


class EncodedLabelClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return X[:, 0].astype(int)


def test_expected_pooled_full_training_volumes(paper_config) -> None:
    observed = {
        dataset: expected_full_training_volume(paper_config, dataset)
        for dataset in paper_config.datasets
    }
    assert observed == PAPER_FULL_TRAINING_VOLUMES


def test_dataset_learning_curve_pools_subject_prefixes_and_resumes(
    paper_config,
    tmp_path,
    monkeypatch,
) -> None:
    raw = deepcopy(paper_config.raw)
    raw["project"]["results_root"] = str(tmp_path / "results")
    raw["project"]["n_jobs"] = 1
    raw["datasets"]["BNCI2014_001"]["subjects"] = [1, 2]
    raw["datasets"]["BNCI2014_001"]["events"] = ["left", "right"]
    raw["datasets"]["BNCI2014_001"]["trials_per_class"] = 10
    raw["splitting"]["training_fractions"] = [0.5, 1.0]
    raw["splitting"]["time_series_splits"] = 2
    raw["classifiers"] = {"order": ["A", "B"], "A": {}, "B": {}}
    config = replace(paper_config, raw=raw)

    def load_cached_subject(config, dataset_name, subject):
        labels = np.arange(20, dtype=np.int64) % 2
        epochs = np.zeros((20, 1, 2), dtype=np.float32)
        epochs[:, 0, 0] = labels
        epochs[:, 0, 1] = subject
        return SimpleNamespace(X=epochs, y=labels)

    monkeypatch.setattr(base_benchmark, "load_cached_subject", load_cached_subject)
    monkeypatch.setattr(base_benchmark, "CSPFeatureTransformer", IdentityTransformer)
    monkeypatch.setattr(
        base_benchmark,
        "make_classifier",
        lambda name, classifier_config, seed, n_jobs: EncodedLabelClassifier(),
    )

    path = run_dataset_learning_curves(config, "BNCI2014_001")
    frame = pd.read_csv(path)
    assert len(frame) == 8
    assert set(frame["subject_count"]) == {2}
    assert set(frame.loc[frame["training_fraction"] == 0.5, "training_volume"]) == {16}
    assert set(frame.loc[frame["training_fraction"] == 1.0, "training_volume"]) == {32}
    half = frame.loc[
        (frame["training_fraction"] == 0.5) & (frame["classifier"] == "A")
    ].sort_values("fold")
    assert half["fold_fit_count"].tolist() == [6, 11]
    assert half["fold_validation_count"].tolist() == [5, 5]
    assert np.allclose(frame["train_score"], 1.0)
    assert np.allclose(frame["validation_score"], 1.0)

    metadata = json.loads(path.with_name("learning_curve_metadata.json").read_text())
    assert metadata["subjects"] == [1, 2]
    assert metadata["development_counts"] == {"1": 16, "2": 16}
    assert metadata["subject_prefix_counts"]["0.5"] == {"1": 8, "2": 8}

    def fail_classifier(*args, **kwargs):
        raise AssertionError("A complete dataset learning curve must resume without refitting")

    monkeypatch.setattr(base_benchmark, "make_classifier", fail_classifier)
    assert run_dataset_learning_curves(config, "BNCI2014_001") == path
