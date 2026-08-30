from __future__ import annotations

import numpy as np
import pandas as pd

from wsaiec_eeg.constants import CLASSIFIER_ORDER, CLUSTER_WINNERS, DATASET_ORDER, METRIC_COLUMNS


def test_tables_three_and_five_reproduce_table_six_ranking(repository_root) -> None:
    source = repository_root / "results" / "publication" / "source"
    performance = pd.read_csv(source / "aggregated_performance.csv")
    learning = pd.read_csv(source / "learning_ranking.csv")
    reported = pd.read_csv(source / "overall_ranking.csv")
    derived = performance[["classifier", "rank"]].rename(
        columns={"rank": "performance_rank"}
    ).merge(
        learning[["classifier", "rank"]].rename(columns={"rank": "learning_rank"}),
        on="classifier",
        validate="one_to_one",
    )
    derived["average_rank"] = (
        derived["performance_rank"] + derived["learning_rank"]
    ) / 2.0
    derived = derived.sort_values(["average_rank", "learning_rank", "classifier"])
    assert derived["classifier"].tolist() == reported["classifier"].tolist()
    assert reported["rank"].tolist() == list(range(1, 17))
    inverse = 1.0 / reported["rank"].to_numpy(dtype=float)
    expected_weights = inverse / inverse.sum()
    np.testing.assert_allclose(reported["weight"], expected_weights, atol=0.0006, rtol=0.0)


def test_table_seven_partitions_classifiers_and_selects_ranked_winners(repository_root) -> None:
    source = repository_root / "results" / "publication" / "source"
    ranking = pd.read_csv(source / "overall_ranking.csv").set_index("classifier")
    clusters = pd.read_csv(source / "clusters.csv")
    members: list[str] = []
    for row in clusters.itertuples(index=False):
        current = row.classifiers.split(";")
        members.extend(current)
        expected_winner = min(current, key=lambda classifier: int(ranking.loc[classifier, "rank"]))
        assert row.best_classifier == expected_winner
        assert int(row.overall_rank) == int(ranking.loc[row.best_classifier, "rank"])
    assert len(members) == len(set(members)) == 16
    assert set(members) == set(CLASSIFIER_ORDER)
    assert clusters["best_classifier"].tolist() == list(CLUSTER_WINNERS)
    selected_weights = ranking.loc[clusters["best_classifier"], "weight"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        clusters["normalized_weight"],
        selected_weights / selected_weights.sum(),
        atol=0.00006,
        rtol=0.0,
    )


def test_tables_two_four_and_eight_have_complete_six_dataset_grids(repository_root) -> None:
    source = repository_root / "results" / "publication" / "source"
    base = pd.read_csv(source / "base_classifier_metrics.csv")
    learning = pd.read_csv(source / "learning_curve_metrics.csv")
    comparison = pd.read_csv(source / "wsaiec_metrics.csv")
    expected_base = {
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in CLASSIFIER_ORDER
    }
    assert set(map(tuple, base[["dataset", "classifier"]].itertuples(index=False, name=None))) == expected_base
    assert set(map(tuple, learning[["dataset", "classifier"]].itertuples(index=False, name=None))) == expected_base
    assert base.groupby("dataset").size().to_dict() == {
        dataset: 16 for dataset in DATASET_ORDER
    }
    assert learning.groupby("dataset").size().to_dict() == {
        dataset: 16 for dataset in DATASET_ORDER
    }
    assert comparison.groupby("dataset").size().to_dict() == {
        dataset: 6 for dataset in DATASET_ORDER
    }


def test_table_eight_comparator_metrics_are_exact_table_two_values(repository_root) -> None:
    source = repository_root / "results" / "publication" / "source"
    base = pd.read_csv(source / "base_classifier_metrics.csv")
    comparison = pd.read_csv(source / "wsaiec_metrics.csv")
    individual = comparison.loc[comparison["classifier"] != "WS_AIEC"]
    joined = individual.merge(
        base,
        on=["dataset", "classifier"],
        suffixes=("_table8", "_table2"),
        validate="one_to_one",
    )
    for metric in METRIC_COLUMNS:
        np.testing.assert_array_equal(joined[f"{metric}_table8"], joined[f"{metric}_table2"])
    assert comparison.loc[comparison["classifier"] == "WS_AIEC"].shape[0] == 6
