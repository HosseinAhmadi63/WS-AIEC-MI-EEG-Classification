from __future__ import annotations

import numpy as np

from wsaiec_eeg.features.csp import build_fold_features


def test_csp_is_refitted_with_ordered_nonoverlapping_folds(paper_config) -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(90, 3, 80))
    y = np.arange(90) % 2
    X[y == 0, 0] *= 2.0
    X[y == 1, 2] *= 2.0
    settings = dict(paper_config.section("features")["csp"])
    folds = build_fold_features(X, y, settings, True, 8)
    assert len(folds) == 8
    for fold in folds:
        assert fold.train_index.max() < fold.valid_index.min()
        assert fold.X_train.shape[1] == 3
        assert fold.X_valid.shape[1] == 3
        assert np.isfinite(fold.X_train).all()
