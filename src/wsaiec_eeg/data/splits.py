"""Ordered outer holdout and expanding-window time-series validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class OuterSplit:
    train: np.ndarray
    test: np.ndarray


def chronological_holdout(y: np.ndarray, test_fraction: float) -> OuterSplit:
    """Use the first 80% for development and final 20% for unseen testing."""

    labels = np.asarray(y)
    n_trials = len(labels)
    n_test = int(math.ceil(n_trials * test_fraction))
    boundary = n_trials - n_test
    if boundary <= 0:
        raise ValueError("Not enough trials for the requested holdout")
    train = np.arange(boundary, dtype=np.int64)
    test = np.arange(boundary, n_trials, dtype=np.int64)
    expected = np.unique(labels)
    if not np.array_equal(np.unique(labels[train]), expected):
        raise ValueError("Chronological training partition does not contain every class")
    if not np.array_equal(np.unique(labels[test]), expected):
        raise ValueError("Chronological testing partition does not contain every class")
    return OuterSplit(train=train, test=test)


def make_outer_split(y: np.ndarray, strategy: str, test_fraction: float) -> OuterSplit:
    if strategy != "chronological":
        raise ValueError(f"Unsupported outer strategy {strategy!r}; paper.yaml uses 'chronological'")
    return chronological_holdout(y, test_fraction)


def time_series_splits(n_samples: int, n_splits: int, gap: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return scikit-learn's exact expanding-window TimeSeriesSplit indices."""

    indices = np.arange(n_samples)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    folds = [(train.astype(np.int64), valid.astype(np.int64)) for train, valid in splitter.split(indices)]
    for train, valid in folds:
        if len(train) == 0 or len(valid) == 0 or train.max() >= valid.min():
            raise RuntimeError("Invalid time-series split: validation must strictly follow training")
    return folds


def chronological_fraction(indices: np.ndarray, fraction: float, y: np.ndarray) -> np.ndarray:
    """Take an ordered training prefix, extending only enough to include every class."""

    source = np.asarray(indices, dtype=np.int64)
    if fraction >= 1:
        return source.copy()
    count = max(1, int(math.ceil(len(source) * fraction)))
    expected = set(np.unique(y[source]).tolist())
    while count < len(source) and set(np.unique(y[source[:count]]).tolist()) != expected:
        count += 1
    return source[:count]
