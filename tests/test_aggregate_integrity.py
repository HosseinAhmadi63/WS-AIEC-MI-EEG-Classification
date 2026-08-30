from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from wsaiec_eeg.evaluation.aggregate import (
    _assert_exact_grid,
    _learning_summary,
    aggregate_learning_curves,
    aggregate_ranking,
)
from wsaiec_eeg.evaluation.base_benchmark import expected_training_volumes


def test_aggregate_grid_rejects_partial_and_duplicate_results() -> None:
    expected = {("dataset_a", 1), ("dataset_a", 2)}
    with pytest.raises(ValueError, match="incomplete"):
        _assert_exact_grid(
            pd.DataFrame({"dataset": ["dataset_a"], "subject": [1]}),
            ["dataset", "subject"],
            expected,
            "test",
        )
    with pytest.raises(ValueError, match="duplicate"):
        _assert_exact_grid(
            pd.DataFrame({"dataset": ["dataset_a", "dataset_a"], "subject": [1, 1]}),
            ["dataset", "subject"],
            {("dataset_a", 1)},
            "test",
        )


def test_learning_summary_uses_numpy_trapezoid_integration() -> None:
    curves = pd.DataFrame(
        {
            "dataset": ["A", "A", "A", "A"],
            "classifier": ["X", "X", "Y", "Y"],
            "training_fraction": [0.5, 1.0, 0.5, 1.0],
            "training_volume": [1.0, 2.0, 1.0, 2.0],
            "train_score": [0.8, 0.9, 0.7, 0.8],
            "validation_score": [0.5, 1.0, 0.4, 0.8],
            "validation_score_std": [0.1, 0.2, 0.2, 0.3],
        }
    )
    summary = _learning_summary(curves).set_index("classifier")
    assert summary.loc["X", "auc_cv"] == pytest.approx(0.75)
    assert summary.loc["Y", "auc_cv"] == pytest.approx(0.6)


def test_computed_and_paper_learning_rankings_are_separate(
    paper_config,
    tmp_path,
) -> None:
    raw = deepcopy(paper_config.raw)
    raw["project"]["results_root"] = str(tmp_path / "results")
    raw["publication"]["generated_outputs"] = str(tmp_path / "generated")
    raw["publication"]["reference_tables"] = str(paper_config.publication_source)
    config = replace(paper_config, raw=raw)
    classifiers = list(config.section("classifiers")["order"])
    fractions = [float(value) for value in config.section("splitting")["training_fractions"]]
    folds = int(config.section("splitting")["time_series_splits"])
    for dataset, specification in config.datasets.items():
        rows = []
        volumes = expected_training_volumes(config, dataset)
        for classifier_index, classifier in enumerate(classifiers):
            validation_score = 0.5 + classifier_index / 100.0
            for fraction in fractions:
                for fold in range(1, folds + 1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "classifier": classifier,
                            "training_fraction": fraction,
                            "training_volume": volumes[fraction],
                            "fold": fold,
                            "fold_fit_count": max(1, volumes[fraction] - fold),
                            "fold_validation_count": fold,
                            "subject_count": len(specification["subjects"]),
                            "train_score": validation_score + 0.1,
                            "validation_score": validation_score,
                        }
                    )
        path = config.run_root / dataset / "learning_curve.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    paths = aggregate_learning_curves(config)
    computed = pd.read_csv(paths["computed_learning_ranking"])
    legacy = pd.read_csv(paths["learning_ranking"])
    paper = pd.read_csv(paths["paper_learning_ranking"])
    frozen = pd.read_csv(config.publication_source / "learning_ranking.csv")
    assert computed.equals(legacy)
    assert paper.equals(frozen)
    assert not computed.equals(paper)


def test_cluster_selection_uses_frozen_table_seven_groups(
    paper_config,
    tmp_path,
) -> None:
    raw = deepcopy(paper_config.raw)
    raw["publication"]["generated_outputs"] = str(tmp_path / "generated")
    config = replace(paper_config, raw=raw)
    generated = config.publication_generated
    generated.mkdir(parents=True)
    pd.read_csv(
        paper_config.publication_source / "aggregated_performance.csv"
    ).to_csv(generated / "aggregated_performance.csv", index=False)
    pd.read_csv(paper_config.publication_source / "learning_ranking.csv").to_csv(
        generated / "learning_ranking.csv",
        index=False,
    )

    aggregate_ranking(config)
    selection = pd.read_csv(generated / "cluster_selection.csv")
    reference = pd.read_csv(paper_config.publication_source / "clusters.csv")
    assert selection[["cluster", "classifiers", "best_classifier", "overall_rank"]].equals(
        reference[["cluster", "classifiers", "best_classifier", "overall_rank"]]
    )
    computed = pd.read_csv(generated / "cluster_assignments.csv")
    assert "computed_ward_cluster" in computed
    assert "computed_cluster_winner" in computed
