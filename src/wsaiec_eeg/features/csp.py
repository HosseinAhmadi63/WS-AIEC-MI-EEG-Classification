"""CSP fitting isolated inside every development fold and outer train split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from wsaiec_eeg.data.splits import time_series_splits


class CSPFeatureTransformer:
    """Fit MNE CSP followed by an optional z-score scaler on training data only."""

    def __init__(self, settings: dict[str, Any], standardize: bool = True) -> None:
        self.settings = dict(settings)
        self.standardize = bool(standardize)
        self.csp_: Any | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPFeatureTransformer:
        try:
            from mne.decoding import CSP
        except ImportError as exc:
            raise RuntimeError("MNE is required for Common Spatial Pattern features") from exc
        parameters = dict(self.settings)
        parameters["n_components"] = min(int(parameters["n_components"]), int(X.shape[1]))
        self.csp_ = CSP(**parameters)
        transformed = self.csp_.fit_transform(X, y)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(transformed)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.csp_ is None:
            raise RuntimeError("CSPFeatureTransformer must be fitted before transform")
        transformed = np.asarray(self.csp_.transform(X), dtype=np.float64)
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        if not np.isfinite(transformed).all():
            raise RuntimeError("CSP produced non-finite features")
        return transformed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


@dataclass(frozen=True)
class FoldFeatures:
    fold: int
    train_index: np.ndarray
    valid_index: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray


def build_fold_features(
    X: np.ndarray,
    y: np.ndarray,
    csp_settings: dict[str, Any],
    standardize: bool,
    n_splits: int,
    gap: int = 0,
) -> list[FoldFeatures]:
    """Fit a fresh CSP/scaler pair in every expanding time-series fold."""

    output: list[FoldFeatures] = []
    all_classes = np.unique(y)
    for fold, (train, valid) in enumerate(time_series_splits(len(y), n_splits, gap), start=1):
        if not np.array_equal(np.unique(y[train]), all_classes):
            raise RuntimeError(f"Time-series fold {fold} training segment is missing a class")
        transformer = CSPFeatureTransformer(csp_settings, standardize)
        X_train = transformer.fit_transform(X[train], y[train])
        X_valid = transformer.transform(X[valid])
        output.append(
            FoldFeatures(
                fold=fold,
                train_index=train,
                valid_index=valid,
                X_train=X_train,
                y_train=y[train],
                X_valid=X_valid,
                y_valid=y[valid],
            )
        )
    return output
