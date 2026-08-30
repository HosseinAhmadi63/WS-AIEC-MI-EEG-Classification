from __future__ import annotations

import numpy as np

from wsaiec_eeg.data.splits import chronological_holdout, time_series_splits


def test_chronological_outer_split() -> None:
    y = np.arange(100) % 2
    split = chronological_holdout(y, 0.2)
    np.testing.assert_array_equal(split.train, np.arange(80))
    np.testing.assert_array_equal(split.test, np.arange(80, 100))


def test_time_series_split_has_eight_ordered_validation_blocks() -> None:
    folds = time_series_splits(90, 8)
    assert len(folds) == 8
    for train, valid in folds:
        assert train.max() < valid.min()
        assert set(train).isdisjoint(valid)
    assert folds[0][0].tolist() == list(range(10))
    assert folds[0][1].tolist() == list(range(10, 20))
    assert folds[-1][1].tolist() == list(range(80, 90))
