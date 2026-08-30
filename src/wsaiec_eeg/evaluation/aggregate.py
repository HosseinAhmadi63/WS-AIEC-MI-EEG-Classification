"""Aggregate participant outputs into the WS-AIEC analysis tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.constants import CLASSIFIER_ORDER, DATASET_ORDER, METRIC_COLUMNS
from wsaiec_eeg.evaluation.base_benchmark import expected_training_volumes
from wsaiec_eeg.utils.io import ensure_directory, write_csv


def _collect(config: ExperimentConfig, filename: str) -> pd.DataFrame:
    expected = {
        config.run_root / dataset / f"subject_{int(subject):02d}" / filename: (
            dataset,
            int(subject),
        )
        for dataset, specification in config.datasets.items()
        for subject in specification["subjects"]
    }
    actual = set(config.run_root.glob(f"*/subject_*/{filename}"))
    if actual != set(expected):
        missing = sorted(str(path.relative_to(config.run_root)) for path in set(expected) - actual)
        extra = sorted(str(path.relative_to(config.run_root)) for path in actual - set(expected))
        raise FileNotFoundError(
            f"Incomplete {filename} cohort below {config.run_root}; "
            f"missing={missing}, extra={extra}"
        )
    frames: list[pd.DataFrame] = []
    for path, (dataset, subject) in expected.items():
        frame = pd.read_csv(path)
        if frame.empty or {"dataset", "subject"} - set(frame):
            raise ValueError(f"{path} is empty or lacks identity columns")
        identities = set(
            map(tuple, frame[["dataset", "subject"]].itertuples(index=False, name=None))
        )
        if identities != {(dataset, subject)}:
            raise ValueError(f"Rows in {path} do not match its directory")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _subject_pairs(config: ExperimentConfig) -> set[tuple[object, ...]]:
    return {
        (dataset, int(subject))
        for dataset, specification in config.datasets.items()
        for subject in specification["subjects"]
    }


def _collect_learning_curves(config: ExperimentConfig) -> pd.DataFrame:
    expected = {
        config.run_root / dataset / "learning_curve.csv": dataset
        for dataset in config.datasets
    }
    actual = set(config.run_root.glob("*/learning_curve.csv"))
    if actual != set(expected):
        missing = sorted(str(path.relative_to(config.run_root)) for path in set(expected) - actual)
        extra = sorted(str(path.relative_to(config.run_root)) for path in actual - set(expected))
        raise FileNotFoundError(
            f"Incomplete dataset learning curves below {config.run_root}; "
            f"missing={missing}, extra={extra}"
        )
    frames: list[pd.DataFrame] = []
    for path, dataset in expected.items():
        frame = pd.read_csv(path)
        if frame.empty or "dataset" not in frame:
            raise ValueError(f"{path} is empty or lacks its dataset identity")
        if set(frame["dataset"].astype(str)) != {dataset}:
            raise ValueError(f"Rows in {path} do not match its directory")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _assert_exact_grid(
    frame: pd.DataFrame,
    columns: list[str],
    expected: set[tuple[object, ...]],
    label: str,
) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{label} contains duplicate keys {columns}")
    actual = set(map(tuple, frame[columns].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(
            f"{label} is incomplete; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _ordered(frame: pd.DataFrame, rank_column: str) -> pd.DataFrame:
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    classifier_order = {name: index for index, name in enumerate(CLASSIFIER_ORDER)}
    output = frame.copy()
    if "dataset" in output:
        output["_dataset_order"] = output["dataset"].map(dataset_order)
    output["_classifier_order"] = output["classifier"].map(classifier_order)
    keys = (["_dataset_order"] if "_dataset_order" in output else []) + [
        rank_column,
        "_classifier_order",
    ]
    return output.sort_values(keys).drop(
        columns=[
            column
            for column in ("_dataset_order", "_classifier_order")
            if column in output
        ]
    )


def _weighted_mean(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    columns = list(weights)
    values = frame[columns].to_numpy(dtype=float)
    vector = np.asarray([float(weights[column]) for column in columns], dtype=float)
    return pd.Series(values @ vector / vector.sum(), index=frame.index)


def aggregate_base(config: ExperimentConfig) -> dict[str, Path]:
    output = ensure_directory(config.publication_generated)
    metrics = _collect(config, "base_metrics.csv")
    required = {"time_series_validation", "outer_test"}
    if set(metrics["scope"]) != required:
        raise ValueError(f"Base metrics scopes must be {sorted(required)}")
    values = metrics[list(METRIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Base metrics contain missing or non-numeric values")
    outer = metrics.loc[metrics["scope"] == "outer_test"].copy()
    expected = {
        (*pair, classifier)
        for pair in _subject_pairs(config)
        for classifier in CLASSIFIER_ORDER
    }
    _assert_exact_grid(outer, ["dataset", "subject", "classifier"], expected, "Outer tests")
    aggregate = outer.groupby(["dataset", "classifier"], as_index=False)[
        list(METRIC_COLUMNS)
    ].mean()
    performance_weights = config.section("ranking")["performance_metric_weights"]
    aggregate["score"] = _weighted_mean(aggregate, performance_weights)
    aggregate["rank"] = (
        aggregate.groupby("dataset")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    aggregate = _ordered(aggregate, "rank")
    base_path = output / "base_classifier_metrics.csv"
    write_csv(base_path, aggregate)

    global_performance = aggregate.groupby("classifier", as_index=False)[
        list(METRIC_COLUMNS)
    ].mean()
    global_performance["overall_score"] = _weighted_mean(
        global_performance, performance_weights
    )
    global_performance = global_performance.sort_values(
        ["overall_score", "classifier"], ascending=[False, True]
    )
    global_performance["rank"] = range(1, len(global_performance) + 1)
    performance_path = output / "aggregated_performance.csv"
    write_csv(performance_path, global_performance)
    return {
        "base_classifier_metrics": base_path,
        "aggregated_performance": performance_path,
    }


def _cost_normalize(values: pd.Series, higher_is_better: bool) -> pd.Series:
    source = values.to_numpy(dtype=float)
    minimum = float(np.min(source))
    maximum = float(np.max(source))
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(source)), index=values.index)
    if higher_is_better:
        normalized = (maximum - source) / (maximum - minimum)
    else:
        normalized = (source - minimum) / (maximum - minimum)
    return pd.Series(normalized, index=values.index)


def _learning_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, classifier), group in curves.groupby(
        ["dataset", "classifier"], sort=False
    ):
        ordered = group.sort_values("training_fraction")
        x = ordered["training_volume"].to_numpy(dtype=float)
        validation = ordered["validation_score"].to_numpy(dtype=float)
        auc_cv = float(np.trapezoid(validation, x))
        final = ordered.iloc[-1]
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "auc_cv": auc_cv,
                "convergence_rate": abs(
                    float(final["train_score"]) - float(final["validation_score"])
                ),
                "performance_stability": float(ordered["validation_score_std"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    normalized: list[pd.DataFrame] = []
    for _, group in summary.groupby("dataset", sort=False):
        current = group.copy()
        current["auc_cv_normalized"] = _cost_normalize(current["auc_cv"], True)
        current["convergence_rate_normalized"] = _cost_normalize(
            current["convergence_rate"], False
        )
        current["performance_stability_normalized"] = _cost_normalize(
            current["performance_stability"], False
        )
        normalized.append(current)
    return pd.concat(normalized, ignore_index=True)


def aggregate_learning_curves(config: ExperimentConfig) -> dict[str, Path]:
    output = ensure_directory(config.publication_generated)
    metrics = _collect_learning_curves(config)
    numeric_columns = [
        "training_volume",
        "fold_fit_count",
        "fold_validation_count",
        "subject_count",
        "train_score",
        "validation_score",
    ]
    missing_columns = set(numeric_columns) - set(metrics)
    if missing_columns:
        raise ValueError(f"Learning curves lack columns {sorted(missing_columns)}")
    numeric = metrics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Learning curves contain invalid numeric values")
    scores = numeric[["train_score", "validation_score"]]
    if ((scores < 0.0) | (scores > 1.0)).any().any():
        raise ValueError("Learning-curve scores must be in [0, 1]")
    counts = numeric[
        ["training_volume", "fold_fit_count", "fold_validation_count", "subject_count"]
    ]
    if (counts <= 0).any().any():
        raise ValueError("Learning-curve counts must be positive")
    expected = {
        (dataset, classifier, float(fraction), fold)
        for dataset in config.datasets
        for classifier in CLASSIFIER_ORDER
        for fraction in config.section("splitting")["training_fractions"]
        for fold in range(1, int(config.section("splitting")["time_series_splits"]) + 1)
    }
    _assert_exact_grid(
        metrics,
        ["dataset", "classifier", "training_fraction", "fold"],
        expected,
        "Learning curves",
    )
    for dataset in config.datasets:
        expected_volumes = expected_training_volumes(config, dataset)
        dataset_rows = metrics.loc[metrics["dataset"] == dataset]
        expected_subject_count = len(config.datasets[dataset]["subjects"])
        if set(dataset_rows["subject_count"].astype(int)) != {expected_subject_count}:
            raise ValueError(f"{dataset} learning curves have an incorrect subject count")
        for fraction, expected_volume in expected_volumes.items():
            fraction_rows = dataset_rows.loc[
                np.isclose(dataset_rows["training_fraction"], fraction)
            ]
            if set(fraction_rows["training_volume"].astype(int)) != {expected_volume}:
                raise ValueError(
                    f"{dataset} fraction {fraction} must use pooled volume {expected_volume}"
                )
    curves = (
        metrics.groupby(["dataset", "classifier", "training_fraction"], as_index=False)
        .agg(
            training_volume=("training_volume", "first"),
            fold_fit_count_mean=("fold_fit_count", "mean"),
            fold_validation_count_mean=("fold_validation_count", "mean"),
            train_score=("train_score", "mean"),
            train_score_std=("train_score", lambda values: float(np.std(values, ddof=0))),
            validation_score=("validation_score", "mean"),
            validation_score_std=(
                "validation_score",
                lambda values: float(np.std(values, ddof=0)),
            ),
        )
    )
    curves_path = output / "learning_curves.csv"
    write_csv(curves_path, curves)

    summary = _learning_summary(curves)
    learning_weights = config.section("ranking")["learning_metric_weights"]
    normalized_weights = {
        f"{name}_normalized": float(value) for name, value in learning_weights.items()
    }
    summary["learning_score"] = _weighted_mean(summary, normalized_weights)
    summary["learning_rank"] = (
        summary.groupby("dataset")["learning_score"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    summary = _ordered(summary, "learning_rank")
    summary_path = output / "learning_curve_summary.csv"
    write_csv(summary_path, summary)

    ranking = summary.groupby("classifier", as_index=False)["learning_score"].mean()
    ranking = ranking.sort_values(["learning_score", "classifier"])
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking_path = output / "learning_ranking.csv"
    computed_ranking_path = output / "computed_learning_ranking.csv"
    write_csv(ranking_path, ranking)
    write_csv(computed_ranking_path, ranking)
    paths = {
        "learning_curves": curves_path,
        "learning_curve_summary": summary_path,
        "learning_ranking": ranking_path,
        "computed_learning_ranking": computed_ranking_path,
    }
    replay_mode = str(config.section("ranking")["learning_ranking_replay_mode"])
    if replay_mode == "paper_reported":
        paper_ranking = pd.read_csv(config.publication_source / "learning_ranking.csv")
        _assert_exact_grid(
            paper_ranking,
            ["classifier"],
            {(classifier,) for classifier in CLASSIFIER_ORDER},
            "Paper learning ranking",
        )
        ranks = paper_ranking["rank"].astype(int).tolist()
        if sorted(ranks) != list(range(1, len(CLASSIFIER_ORDER) + 1)):
            raise ValueError("Paper learning ranking must assign ranks 1 through 16")
        paper_ranking_path = output / "paper_learning_ranking.csv"
        write_csv(paper_ranking_path, paper_ranking)
        paths["paper_learning_ranking"] = paper_ranking_path
    return paths


def aggregate_ranking(config: ExperimentConfig) -> dict[str, Path]:
    output = ensure_directory(config.publication_generated)
    performance = pd.read_csv(output / "aggregated_performance.csv")
    learning = pd.read_csv(output / "learning_ranking.csv")
    ranking = performance[["classifier", "rank"]].rename(
        columns={"rank": "performance_rank"}
    ).merge(
        learning[["classifier", "rank"]].rename(columns={"rank": "learning_rank"}),
        on="classifier",
        validate="one_to_one",
    )
    weights = config.section("ranking")["composite_weights"]
    numerator = (
        float(weights["performance_rank"]) * ranking["performance_rank"]
        + float(weights["learning_rank"]) * ranking["learning_rank"]
    )
    ranking["average_rank"] = numerator / sum(float(value) for value in weights.values())
    ranking = ranking.sort_values(["average_rank", "learning_rank", "classifier"])
    ranking["rank"] = range(1, len(ranking) + 1)
    inverse = 1.0 / ranking["rank"].to_numpy(dtype=float)
    ranking["weight"] = inverse / inverse.sum()
    ranking_path = output / "overall_ranking.csv"
    write_csv(ranking_path, ranking)

    features = ranking.set_index("classifier")[["performance_rank", "learning_rank"]]
    values = features.to_numpy(dtype=float)
    if config.section("ranking")["standardize_rank_features"]:
        from sklearn.preprocessing import StandardScaler

        values = StandardScaler().fit_transform(values)
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans

    linkage_matrix = linkage(
        values,
        method=str(config.section("ranking")["clustering_method"]),
        metric=str(config.section("ranking")["clustering_metric"]),
    )
    raw_labels = fcluster(
        linkage_matrix,
        t=int(config.section("ranking")["number_of_clusters"]),
        criterion="maxclust",
    )
    raw_members = {
        int(label): sorted(
            ranking.loc[raw_labels == label, "classifier"].astype(str).tolist()
        )
        for label in np.unique(raw_labels)
    }
    canonical_order = sorted(raw_members, key=lambda label: tuple(raw_members[label]))
    label_map = {label: index + 1 for index, label in enumerate(canonical_order)}
    ranking["cluster"] = [label_map[int(label)] for label in raw_labels]
    cluster_rows: list[dict[str, Any]] = []
    for cluster, group in ranking.groupby("cluster", sort=True):
        ordered = group.sort_values("rank")
        winner = str(ordered.iloc[0]["classifier"])
        for row in group.itertuples(index=False):
            cluster_rows.append(
                {
                    "classifier": row.classifier,
                    "computed_ward_cluster": int(cluster),
                    "performance_rank": int(row.performance_rank),
                    "learning_rank": int(row.learning_rank),
                    "average_rank": float(row.average_rank),
                    "overall_rank": int(row.rank),
                    "computed_cluster_winner": row.classifier == winner,
                }
            )
    ranking_settings = config.section("ranking")
    if str(ranking_settings["cluster_selection_mode"]) != "paper_reported":
        raise ValueError("The frozen paper run requires paper_reported cluster selection")
    reported_clusters = {
        int(cluster): [str(name) for name in members]
        for cluster, members in ranking_settings["reported_clusters"].items()
    }
    reported_winners = [str(name) for name in ranking_settings["reported_cluster_winners"]]
    reported_ranks = {
        str(name): int(rank)
        for name, rank in ranking_settings["reported_overall_ranks"].items()
    }
    selection = pd.DataFrame(
        [
            {
                "cluster": cluster,
                "classifiers": ";".join(reported_clusters[cluster]),
                "best_classifier": reported_winners[cluster - 1],
                "overall_rank": reported_ranks[reported_winners[cluster - 1]],
            }
            for cluster in sorted(reported_clusters)
        ]
    )
    selected_inverse = 1.0 / selection["overall_rank"].to_numpy(dtype=float)
    selection["normalized_weight"] = selected_inverse / selected_inverse.sum()
    clusters_path = output / "cluster_assignments.csv"
    selection_path = output / "cluster_selection.csv"
    write_csv(clusters_path, pd.DataFrame(cluster_rows))
    write_csv(selection_path, selection)

    linkage_path = output / "cluster_linkage.csv"
    write_csv(
        linkage_path,
        pd.DataFrame(
            linkage_matrix,
            columns=["left", "right", "distance", "sample_count"],
        ),
    )
    elbow_rows = []
    for number in range(1, len(CLASSIFIER_ORDER)):
        model = KMeans(n_clusters=number, random_state=int(config.project["random_seed"]), n_init=50)
        model.fit(values)
        elbow_rows.append({"number_of_clusters": number, "inertia": float(model.inertia_)})
    elbow_path = output / "cluster_elbow.csv"
    write_csv(elbow_path, pd.DataFrame(elbow_rows))
    return {
        "overall_ranking": ranking_path,
        "cluster_assignments": clusters_path,
        "cluster_selection": selection_path,
        "cluster_linkage": linkage_path,
        "cluster_elbow": elbow_path,
    }


def aggregate_wsaiec(config: ExperimentConfig) -> Path:
    output = ensure_directory(config.publication_generated)
    metrics = _collect(config, "wsaiec_metrics.csv")
    values = metrics[list(METRIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("WS-AIEC metrics contain invalid values")
    _assert_exact_grid(metrics, ["dataset", "subject"], _subject_pairs(config), "WS-AIEC")
    aggregate = metrics.groupby("dataset", as_index=False)[list(METRIC_COLUMNS)].mean()
    aggregate["score"] = aggregate[list(METRIC_COLUMNS)].mean(axis=1)
    aggregate["_order"] = aggregate["dataset"].map(
        {name: index for index, name in enumerate(DATASET_ORDER)}
    )
    aggregate = aggregate.sort_values("_order").drop(columns="_order")
    path = output / "wsaiec_metrics.csv"
    write_csv(path, aggregate)
    return path


def aggregate_pre_ensemble(config: ExperimentConfig) -> dict[str, Path]:
    paths = aggregate_base(config)
    paths.update(aggregate_learning_curves(config))
    paths.update(aggregate_ranking(config))
    return paths


def aggregate_all(config: ExperimentConfig, include_learning_curves: bool = True) -> dict[str, Path]:
    paths = aggregate_base(config)
    if include_learning_curves:
        paths.update(aggregate_learning_curves(config))
        paths.update(aggregate_ranking(config))
    paths["wsaiec_metrics"] = aggregate_wsaiec(config)
    from wsaiec_eeg.evaluation.ablation import ablation_metrics_exist, aggregate_ablations

    if ablation_metrics_exist(config):
        paths["ablation"] = aggregate_ablations(config)
    return paths
