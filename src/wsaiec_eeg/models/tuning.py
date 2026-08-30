"""Manual grid/random search over leakage-safe fold-specific CSP features."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from wsaiec_eeg.features.csp import FoldFeatures
from wsaiec_eeg.models.classifiers import make_classifier


def tune_classifier(
    name: str,
    folds: list[FoldFeatures],
    classifier_config: dict[str, Any],
    tuning_config: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the highest mean-validation-accuracy parameter mapping and audit trail."""

    if name not in tuning_config:
        return dict(classifier_config[name]), []
    specification = tuning_config[name]
    method = specification["method"]
    space = specification["parameters"]
    if method == "grid":
        candidates = list(ParameterGrid(space))
    elif method == "random":
        candidates = list(
            ParameterSampler(
                space,
                n_iter=min(int(tuning_config["random_search_iterations"]), _space_size(space)),
                random_state=seed,
            )
        )
    else:
        raise ValueError(f"Unknown tuning method {method!r} for {name}")

    records: list[dict[str, Any]] = []
    base = dict(classifier_config[name])
    for candidate_index, candidate in enumerate(candidates):
        merged = {**base, **candidate}
        trial_config = {**classifier_config, name: merged}
        scores: list[float] = []
        for fold in folds:
            model = make_classifier(
                name, trial_config, seed + 1000 * candidate_index + fold.fold, n_jobs
            )
            model.fit(fold.X_train, fold.y_train)
            scores.append(float(np.mean(model.predict(fold.X_valid) == fold.y_valid)))
        records.append({"parameters": merged, "mean_accuracy": float(np.mean(scores))})
    best = max(records, key=lambda record: record["mean_accuracy"])
    return dict(best["parameters"]), records


def _space_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size
