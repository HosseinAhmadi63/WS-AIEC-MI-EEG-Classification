"""Validate the transcribed paper tables and compare completed reruns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.constants import (
    CLASSIFIER_ORDER,
    CLUSTER_WINNERS,
    DATASET_ORDER,
    METRIC_COLUMNS,
    PAPER_DOI,
)
from wsaiec_eeg.models.wsaiec import weight_consistency_report
from wsaiec_eeg.utils.io import ensure_directory, write_csv, write_json

SOURCE_TABLE_ROWS = {
    "datasets.csv": 6,
    "base_classifier_metrics.csv": 96,
    "aggregated_performance.csv": 16,
    "learning_curve_metrics.csv": 96,
    "learning_ranking.csv": 16,
    "overall_ranking.csv": 16,
    "clusters.csv": 6,
    "wsaiec_metrics.csv": 36,
    "ablation.csv": 11,
    "related_works.csv": 26,
}

TABLE_1_COLUMNS = [
    "dataset",
    "paper_reference",
    "subjects",
    "channels",
    "classes",
    "trials_per_class",
    "trial_duration_s",
    "sampling_rate_hz",
    "sessions",
]
TABLE_2_COLUMNS = ["dataset", "classifier", *METRIC_COLUMNS, "rank"]
TABLE_3_COLUMNS = ["classifier", *METRIC_COLUMNS, "overall_score", "rank"]
TABLE_4_COLUMNS = [
    "dataset",
    "classifier",
    "auc_cv_raw",
    "convergence_rate_raw",
    "performance_stability_raw",
    "auc_cv_normalized",
    "convergence_rate_normalized",
    "performance_stability_normalized",
    "rank",
]
TABLE_5_COLUMNS = ["classifier", "rank"]
TABLE_6_COLUMNS = ["classifier", "rank", "weight"]
TABLE_7_COLUMNS = [
    "cluster",
    "classifiers",
    "best_classifier",
    "overall_rank",
    "normalized_weight",
]
TABLE_8_COLUMNS = ["dataset", "classifier", *METRIC_COLUMNS]
TABLE_9_COLUMNS = [
    "scenario_id",
    "scenario",
    "selected_classifiers",
    "accuracy_bnci2014_002",
    "accuracy_zhou2016",
]
TABLE_10_COLUMNS = [
    "reference",
    "classifiers",
    "datasets",
    "eme",
    "wau",
    "sau",
    "ame",
    "cst",
    "accuracy_percent",
]
TABLE_10_REFERENCES = (
    "Nicolas et al., 2014 [34]",
    "Rahimi et al., 2016 [35]",
    "Mohammadpour et al., 2016 [36]",
    "Datta et al., 2017 [37]",
    "Ramos et al., 2017 [38]",
    "Chatterjee et al., 2018 [39]",
    "Raza et al., 2019 [40]",
    "Salimi et al., 2019 [14]",
    "Zhang et al., 2020 [41]",
    "Tyagi et al., 2020 [12]",
    "Norizadeh et al., 2021 [42]",
    "Nugroho et al., 2021 [43]",
    "Rashid et al., 2021 [44]",
    "Wei et al., 2021 [15]",
    "Zheng et al., 2021 [45]",
    "Sun et al., 2021 [46]",
    "Du et al., 2021 [18]",
    "Kanhi et al., 2022 [47]",
    "Dolzhikova et al., 2022 [17]",
    "Quanyu et al., 2023 [48]",
    "Mehtiyev et al., 2023 [11]",
    "Almohammadi et al., 2023 [13]",
    "Esfahani et al., 2023 [16]",
    "Shin et al., 2024 [49]",
    "Hossein et al., 2024 [20]",
    "This Study",
)
TABLE_8_CLASSIFIERS = ("WS_AIEC", "SVM", "RC", "SVM_rbf", "LR", "NB")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_manifest(source: Path) -> dict[str, Any]:
    manifest_path = source / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing immutable source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    article = manifest.get("article")
    if not isinstance(article, dict) or article.get("doi") != PAPER_DOI:
        raise ValueError("Publication source manifest DOI does not match the configured paper")
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError("Publication source manifest must declare its CSV files")
    expected_names = set(SOURCE_TABLE_ROWS)
    actual_names = {path.name for path in source.glob("*.csv")}
    if set(declared) != expected_names or actual_names != expected_names:
        raise ValueError(
            "Publication source file set differs from SOURCE_MANIFEST.json: "
            f"expected={sorted(expected_names)}, declared={sorted(declared)}, "
            f"actual={sorted(actual_names)}"
        )

    verified: dict[str, Any] = {}
    for filename, expected_rows in SOURCE_TABLE_ROWS.items():
        specification = declared[filename]
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid manifest entry for {filename}")
        path = source / filename
        actual_hash = _sha256(path)
        expected_hash = str(specification.get("sha256", ""))
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for immutable publication source {filename}")
        actual_rows = len(pd.read_csv(path))
        manifest_rows = int(specification.get("rows", -1))
        if actual_rows != manifest_rows or actual_rows != expected_rows:
            raise ValueError(
                f"Row-count mismatch for {filename}: expected {expected_rows}, "
                f"manifest declares {manifest_rows}, found {actual_rows}"
            )
        verified[filename] = {
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
            "rows": actual_rows,
            "article_location": specification.get("source"),
        }
    return {
        "manifest_sha256": _sha256(manifest_path),
        "files": verified,
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.columns.tolist() != columns:
        raise ValueError(
            f"{label} columns are invalid: expected {columns}, found {frame.columns.tolist()}"
        )


def _numeric(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains missing or non-finite numeric values")
    return values


def _unit_interval(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    values = _numeric(frame, columns, label)
    if ((values < 0.0) | (values > 1.0)).any().any():
        raise ValueError(f"{label} contains a value outside [0, 1]")
    return values


def _rank_sequence(values: pd.Series, label: str, size: int = 16) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or sorted(numeric.astype(int).tolist()) != list(range(1, size + 1)):
        raise ValueError(f"{label} does not contain ranks 1 through {size}")
    if not np.allclose(numeric.to_numpy(dtype=float), numeric.astype(int).to_numpy(dtype=float)):
        raise ValueError(f"{label} contains a non-integral rank")


def _exact_key_grid(
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
            f"{label} has an incomplete key grid: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_table_1(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    _require_columns(frame, TABLE_1_COLUMNS, "Table 1")
    if frame["dataset"].tolist() != list(DATASET_ORDER):
        raise ValueError("Table 1 datasets are not in paper order")
    expected_references = [23, 24, 25, 26, 27, 28]
    if frame["paper_reference"].astype(int).tolist() != expected_references:
        raise ValueError("Table 1 dataset references are invalid")
    numeric_columns = TABLE_1_COLUMNS[1:]
    values = _numeric(frame, numeric_columns, "Table 1")
    if (values <= 0).any().any():
        raise ValueError("Table 1 acquisition values must be positive")
    for row in frame.itertuples(index=False):
        specification = config.datasets[str(row.dataset)]
        expected = {
            "subjects": len(specification["subjects"]),
            "channels": int(specification["channels"]),
            "classes": len(specification["events"]),
            "trials_per_class": int(specification["paper_trials_per_class"]),
            "trial_duration_s": float(specification["epoch_duration_seconds"]),
            "sampling_rate_hz": int(specification["sampling_rate"]),
            "sessions": int(specification["max_sessions"]),
        }
        for column, expected_value in expected.items():
            if not np.isclose(float(getattr(row, column)), float(expected_value)):
                raise ValueError(
                    f"Table 1 {row.dataset}.{column} differs from the frozen configuration"
                )
    return {"rows": len(frame), "datasets": frame["dataset"].tolist()}


def _validate_table_2(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_2_COLUMNS, "Table 2")
    expected = {
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in CLASSIFIER_ORDER
    }
    _exact_key_grid(frame, ["dataset", "classifier"], expected, "Table 2")
    _unit_interval(frame, list(METRIC_COLUMNS), "Table 2")
    expected_order = [dataset for dataset in DATASET_ORDER for _ in CLASSIFIER_ORDER]
    if frame["dataset"].tolist() != expected_order:
        raise ValueError("Table 2 datasets are not grouped in paper order")
    for dataset, group in frame.groupby("dataset", sort=False):
        _rank_sequence(group["rank"], f"Table 2 {dataset}")
        if group["rank"].astype(int).tolist() != list(range(1, 17)):
            raise ValueError(f"Table 2 {dataset} rows are not in rank order")
    if not np.allclose(frame["accuracy"], frame["recall"], atol=1e-12, rtol=0.0):
        raise ValueError("Table 2 weighted recall must equal accuracy")
    return {"rows": len(frame), "datasets": len(DATASET_ORDER), "classifiers": 16}


def _validate_table_3(
    frame: pd.DataFrame,
    table_2: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    _require_columns(frame, TABLE_3_COLUMNS, "Table 3")
    _exact_key_grid(
        frame,
        ["classifier"],
        {(classifier,) for classifier in CLASSIFIER_ORDER},
        "Table 3",
    )
    _unit_interval(frame, [*METRIC_COLUMNS, "overall_score"], "Table 3")
    _rank_sequence(frame["rank"], "Table 3")
    if frame["rank"].astype(int).tolist() != list(range(1, 17)):
        raise ValueError("Table 3 rows are not in rank order")
    scores = frame["overall_score"].to_numpy(dtype=float)
    if np.any(np.diff(scores) > 0.0):
        raise ValueError("Table 3 overall scores are not descending by rank")
    score_delta = np.abs(frame[list(METRIC_COLUMNS)].mean(axis=1) - frame["overall_score"])
    if float(score_delta.max()) > tolerance:
        raise ValueError("Table 3 overall score is not the printed six-metric mean")
    table_2_means = table_2.groupby("classifier", as_index=False)[list(METRIC_COLUMNS)].mean()
    joined = frame.merge(table_2_means, on="classifier", suffixes=("_table3", "_table2"))
    aggregation_deltas = [
        np.abs(joined[f"{metric}_table3"] - joined[f"{metric}_table2"])
        for metric in METRIC_COLUMNS
    ]
    maximum_aggregation_delta = float(np.max(np.column_stack(aggregation_deltas)))
    if maximum_aggregation_delta > 0.0005:
        raise ValueError("Table 3 is inconsistent with the rounded Table 2 dataset means")
    return {
        "rows": len(frame),
        "maximum_score_rounding_difference": float(score_delta.max()),
        "maximum_table_2_aggregation_difference": maximum_aggregation_delta,
    }


def _validate_table_4(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_4_COLUMNS, "Table 4")
    expected = {
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in CLASSIFIER_ORDER
    }
    _exact_key_grid(frame, ["dataset", "classifier"], expected, "Table 4")
    raw_columns = [
        "auc_cv_raw",
        "convergence_rate_raw",
        "performance_stability_raw",
    ]
    raw = _numeric(frame, raw_columns, "Table 4 raw metrics")
    if (raw < 0.0).any().any():
        raise ValueError("Table 4 raw learning metrics must be non-negative")
    normalized_columns = [
        "auc_cv_normalized",
        "convergence_rate_normalized",
        "performance_stability_normalized",
    ]
    _unit_interval(frame, normalized_columns, "Table 4 normalized metrics")
    expected_order = [dataset for dataset in DATASET_ORDER for _ in CLASSIFIER_ORDER]
    if frame["dataset"].tolist() != expected_order:
        raise ValueError("Table 4 datasets are not grouped in paper order")
    for dataset, group in frame.groupby("dataset", sort=False):
        _rank_sequence(group["rank"], f"Table 4 {dataset}")
        if group["rank"].astype(int).tolist() != list(range(1, 17)):
            raise ValueError(f"Table 4 {dataset} rows are not in rank order")
    return {"rows": len(frame), "datasets": len(DATASET_ORDER), "classifiers": 16}


def _validate_table_5(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_5_COLUMNS, "Table 5")
    _exact_key_grid(
        frame,
        ["classifier"],
        {(classifier,) for classifier in CLASSIFIER_ORDER},
        "Table 5",
    )
    _rank_sequence(frame["rank"], "Table 5")
    if frame["rank"].astype(int).tolist() != list(range(1, 17)):
        raise ValueError("Table 5 rows are not in rank order")
    return {"rows": len(frame), "best_classifier": str(frame.iloc[0]["classifier"])}


def _validate_table_6(
    frame: pd.DataFrame,
    table_3: pd.DataFrame,
    table_5: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    _require_columns(frame, TABLE_6_COLUMNS, "Table 6")
    _exact_key_grid(
        frame,
        ["classifier"],
        {(classifier,) for classifier in CLASSIFIER_ORDER},
        "Table 6",
    )
    _rank_sequence(frame["rank"], "Table 6")
    if frame["rank"].astype(int).tolist() != list(range(1, 17)):
        raise ValueError("Table 6 rows are not in rank order")
    weights = _unit_interval(frame, ["weight"], "Table 6")["weight"].to_numpy(dtype=float)
    if np.any(np.diff(weights) > 0.0):
        raise ValueError("Table 6 weights are not descending by rank")
    expected_weights = 1.0 / np.arange(1, 17, dtype=float)
    expected_weights /= expected_weights.sum()
    maximum_weight_delta = float(np.max(np.abs(weights - expected_weights)))
    if maximum_weight_delta > 0.0006 or abs(float(weights.sum()) - 1.0) > 0.0011:
        raise ValueError("Table 6 weights do not reproduce normalized inverse ranks")

    performance = table_3[["classifier", "rank"]].rename(
        columns={"rank": "performance_rank"}
    )
    learning = table_5[["classifier", "rank"]].rename(columns={"rank": "learning_rank"})
    derived = performance.merge(learning, on="classifier", validate="one_to_one")
    derived["average_rank"] = (
        derived["performance_rank"] + derived["learning_rank"]
    ) / 2.0
    derived = derived.sort_values(["average_rank", "learning_rank", "classifier"])
    if derived["classifier"].tolist() != frame["classifier"].tolist():
        raise ValueError("Table 6 does not reproduce the combined Tables 3 and 5 ranking")

    configured = {
        str(classifier): int(rank)
        for classifier, rank in config.section("ranking")["reported_overall_ranks"].items()
    }
    printed = dict(zip(frame["classifier"], frame["rank"].astype(int), strict=True))
    if configured != printed:
        raise ValueError("Table 6 ranks differ from the frozen configuration")
    return {
        "rows": len(frame),
        "weight_sum": float(weights.sum()),
        "maximum_inverse_rank_rounding_difference": maximum_weight_delta,
    }


def _validate_table_7(
    frame: pd.DataFrame,
    table_6: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    _require_columns(frame, TABLE_7_COLUMNS, "Table 7")
    _rank_sequence(frame["cluster"], "Table 7 clusters", size=6)
    if frame["cluster"].astype(int).tolist() != list(range(1, 7)):
        raise ValueError("Table 7 clusters are not in paper order")
    overall_ranks = dict(
        zip(table_6["classifier"], table_6["rank"].astype(int), strict=True)
    )
    members: list[str] = []
    for row in frame.itertuples(index=False):
        current = str(row.classifiers).split(";")
        if len(current) != len(set(current)) or str(row.best_classifier) not in current:
            raise ValueError(f"Table 7 cluster {row.cluster} has invalid members or winner")
        if int(row.overall_rank) != overall_ranks[str(row.best_classifier)]:
            raise ValueError(f"Table 7 cluster {row.cluster} winner rank is invalid")
        if str(row.best_classifier) != min(current, key=overall_ranks.__getitem__):
            raise ValueError(f"Table 7 cluster {row.cluster} winner is not its top-ranked member")
        members.extend(current)
    if len(members) != 16 or set(members) != set(CLASSIFIER_ORDER):
        raise ValueError("Table 7 clusters do not partition all 16 classifiers")
    if frame["best_classifier"].tolist() != list(CLUSTER_WINNERS):
        raise ValueError("Table 7 cluster winners are not in paper order")

    weights = _unit_interval(frame, ["normalized_weight"], "Table 7")[
        "normalized_weight"
    ].to_numpy(dtype=float)
    table_6_weights = table_6.set_index("classifier").loc[
        frame["best_classifier"], "weight"
    ].to_numpy(dtype=float)
    expected_weights = table_6_weights / table_6_weights.sum()
    maximum_weight_delta = float(np.max(np.abs(weights - expected_weights)))
    if maximum_weight_delta > 0.00006 or abs(float(weights.sum()) - 0.9999) > 1e-9:
        raise ValueError("Table 7 weights do not reproduce renormalized Table 6 weights")
    configured = {
        str(classifier): float(weight)
        for classifier, weight in config.section("wsaiec")["reported_static_weights"].items()
    }
    printed = dict(zip(frame["best_classifier"], weights, strict=True))
    if configured != printed:
        raise ValueError("Table 7 weights differ from the frozen configuration")
    return {
        "rows": len(frame),
        "members": len(members),
        "weight_sum": float(weights.sum()),
        "maximum_table_6_renormalization_difference": maximum_weight_delta,
    }


def _validate_table_8(frame: pd.DataFrame, table_2: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_8_COLUMNS, "Table 8")
    expected = {
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in TABLE_8_CLASSIFIERS
    }
    _exact_key_grid(frame, ["dataset", "classifier"], expected, "Table 8")
    _unit_interval(frame, list(METRIC_COLUMNS), "Table 8")
    expected_order = [
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in TABLE_8_CLASSIFIERS
    ]
    actual_order = list(frame[["dataset", "classifier"]].itertuples(index=False, name=None))
    if actual_order != expected_order:
        raise ValueError("Table 8 rows are not in paper order")
    comparison = frame.loc[frame["classifier"] != "WS_AIEC"].merge(
        table_2,
        on=["dataset", "classifier"],
        suffixes=("_table8", "_table2"),
        validate="one_to_one",
    )
    maximum_comparison_delta = max(
        float(
            np.abs(comparison[f"{metric}_table8"] - comparison[f"{metric}_table2"]).max()
        )
        for metric in METRIC_COLUMNS
    )
    if maximum_comparison_delta > 1e-12:
        raise ValueError("Table 8 individual-classifier rows differ from Table 2")
    wsaiec = frame.loc[frame["classifier"] == "WS_AIEC", ["dataset", "accuracy"]]
    individual = (
        frame.loc[frame["classifier"] != "WS_AIEC"]
        .groupby("dataset", as_index=False)["accuracy"]
        .max()
    )
    superiority = wsaiec.merge(individual, on="dataset", suffixes=("_wsaiec", "_best"))
    if not (superiority["accuracy_wsaiec"] > superiority["accuracy_best"]).all():
        raise ValueError("Table 8 WS-AIEC accuracy must exceed every printed comparator")
    return {
        "rows": len(frame),
        "wsaiec_rows": len(wsaiec),
        "maximum_table_2_difference": maximum_comparison_delta,
        "mean_wsaiec_accuracy": float(wsaiec["accuracy"].mean()),
    }


def _validate_table_9(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_9_COLUMNS, "Table 9")
    _rank_sequence(frame["scenario_id"], "Table 9 scenario IDs", size=11)
    if frame["scenario_id"].astype(int).tolist() != list(range(1, 12)):
        raise ValueError("Table 9 scenarios are not in paper order")
    accuracy_columns = ["accuracy_bnci2014_002", "accuracy_zhou2016"]
    _unit_interval(frame, accuracy_columns, "Table 9")
    if frame["scenario"].astype(str).str.strip().eq("").any():
        raise ValueError("Table 9 contains an empty scenario label")
    selections: list[list[str]] = []
    for row in frame.itertuples(index=False):
        classifiers = str(row.selected_classifiers).split(";")
        if (
            len(classifiers) != len(set(classifiers))
            or not set(classifiers).issubset(CLASSIFIER_ORDER)
            or len(classifiers) not in {5, 6}
        ):
            raise ValueError(f"Table 9 scenario {row.scenario_id} has invalid classifiers")
        selections.append(classifiers)
    if selections[-1] != list(CLUSTER_WINNERS):
        raise ValueError("Table 9 static-weight scenario does not use the Table 7 winners")
    return {
        "rows": len(frame),
        "scenarios": frame["scenario"].tolist(),
        "minimum_accuracy": float(frame[accuracy_columns].min().min()),
        "maximum_accuracy": float(frame[accuracy_columns].max().max()),
    }


def _validate_table_10(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, TABLE_10_COLUMNS, "Table 10")
    if frame["reference"].tolist() != list(TABLE_10_REFERENCES):
        raise ValueError("Table 10 references are not in paper order")
    if frame["reference"].duplicated().any():
        raise ValueError("Table 10 contains duplicate references")
    numeric = _numeric(
        frame,
        ["classifiers", "datasets", "accuracy_percent"],
        "Table 10",
    )
    counts = numeric[["classifiers", "datasets"]].to_numpy(dtype=float)
    if (counts <= 0).any() or not np.allclose(counts, np.rint(counts)):
        raise ValueError("Table 10 classifier and dataset counts must be positive integers")
    accuracy = numeric["accuracy_percent"].to_numpy(dtype=float)
    if ((accuracy <= 0.0) | (accuracy > 100.0)).any():
        raise ValueError("Table 10 accuracy percentages must lie in (0, 100]")
    boolean_columns = ["eme", "wau", "sau", "ame", "cst"]
    if not all(pd.api.types.is_bool_dtype(frame[column]) for column in boolean_columns):
        raise ValueError("Table 10 method indicators must be booleans")
    if not frame["eme"].all():
        raise ValueError("Table 10 must mark ensemble methodology for every row")
    if frame.loc[frame["reference"] != "This Study", "cst"].any():
        raise ValueError("Table 10 CST is reported only for This Study")
    this_study = frame.iloc[-1]
    if not this_study[boolean_columns].all():
        raise ValueError("Table 10 This Study row must include all five method indicators")
    if int(this_study["classifiers"]) != 16 or int(this_study["datasets"]) != 6:
        raise ValueError("Table 10 This Study classifier or dataset count is invalid")
    if not np.isclose(float(this_study["accuracy_percent"]), 99.58):
        raise ValueError("Table 10 This Study accuracy is invalid")
    return {
        "rows": len(frame),
        "prior_studies": len(frame) - 1,
        "external_prior_studies": len(frame) - 2,
        "maximum_accuracy_percent": float(accuracy.max()),
    }


def _validate_source_tables(
    source: Path,
    config: ExperimentConfig,
    tolerance: float,
) -> dict[str, Any]:
    datasets = pd.read_csv(source / "datasets.csv")
    table_2 = pd.read_csv(source / "base_classifier_metrics.csv")
    table_3 = pd.read_csv(source / "aggregated_performance.csv")
    table_4 = pd.read_csv(source / "learning_curve_metrics.csv")
    table_5 = pd.read_csv(source / "learning_ranking.csv")
    table_6 = pd.read_csv(source / "overall_ranking.csv")
    table_7 = pd.read_csv(source / "clusters.csv")
    table_8 = pd.read_csv(source / "wsaiec_metrics.csv")
    table_9 = pd.read_csv(source / "ablation.csv")
    table_10 = pd.read_csv(source / "related_works.csv")
    return {
        "table_1": _validate_table_1(datasets, config),
        "table_2": _validate_table_2(table_2),
        "table_3": _validate_table_3(table_3, table_2, tolerance),
        "table_4": _validate_table_4(table_4),
        "table_5": _validate_table_5(table_5),
        "table_6": _validate_table_6(table_6, table_3, table_5, config),
        "table_7": _validate_table_7(table_7, table_6, config),
        "table_8": _validate_table_8(table_8, table_2),
        "table_9": _validate_table_9(table_9),
        "table_10": _validate_table_10(table_10),
    }


def _compare_frames(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
    label: str,
) -> pd.DataFrame:
    for source_label, frame in (("reference", reference), ("generated", generated)):
        missing_columns = set(keys + metrics) - set(frame)
        if missing_columns:
            raise ValueError(f"{source_label} {label} is missing columns {sorted(missing_columns)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{source_label} {label} contains duplicate result keys {keys}")
    reference_keys = set(map(tuple, reference[keys].itertuples(index=False, name=None)))
    generated_keys = set(map(tuple, generated[keys].itertuples(index=False, name=None)))
    if reference_keys != generated_keys:
        raise ValueError(
            f"Generated {label} has incomplete result keys: "
            f"missing={sorted(reference_keys - generated_keys)}, "
            f"extra={sorted(generated_keys - reference_keys)}"
        )
    joined = reference[keys + metrics].merge(
        generated[keys + metrics],
        on=keys,
        suffixes=("_paper", "_generated"),
        how="inner",
        validate="one_to_one",
    )
    for metric in metrics:
        paper = pd.to_numeric(joined[f"{metric}_paper"], errors="coerce")
        rerun = pd.to_numeric(joined[f"{metric}_generated"], errors="coerce")
        joined[f"{metric}_absolute_error"] = np.abs(paper - rerun)
    error_columns = [f"{metric}_absolute_error" for metric in metrics]
    if not np.isfinite(joined[error_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Generated comparison for {label} contains non-finite errors")
    return joined


def _compare_table(
    reference_path: Path,
    generated_path: Path,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    return _compare_frames(
        pd.read_csv(reference_path),
        pd.read_csv(generated_path),
        keys,
        metrics,
        generated_path.name,
    )


def _write_comparison(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
    filename: str,
    output: Path,
) -> dict[str, Any]:
    comparison = _compare_frames(reference, generated, keys, metrics, filename)
    output_path = output / f"comparison_{filename}"
    write_csv(output_path, comparison)
    maximum_by_metric = {
        metric: float(comparison[f"{metric}_absolute_error"].max()) for metric in metrics
    }
    return {
        "path": str(output_path),
        "rows": len(comparison),
        "maximum_absolute_error": max(maximum_by_metric.values()),
        "maximum_absolute_error_by_metric": maximum_by_metric,
    }


def _compare_generated_tables(source: Path, generated: Path) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    specifications = [
        (
            "base_classifier_metrics.csv",
            ["dataset", "classifier"],
            [*METRIC_COLUMNS, "rank"],
            None,
        ),
        (
            "aggregated_performance.csv",
            ["classifier"],
            [*METRIC_COLUMNS, "overall_score", "rank"],
            None,
        ),
        ("learning_ranking.csv", ["classifier"], ["rank"], None),
        ("overall_ranking.csv", ["classifier"], ["rank", "weight"], None),
        (
            "wsaiec_metrics.csv",
            ["dataset"],
            list(METRIC_COLUMNS),
            "WS_AIEC",
        ),
    ]
    for filename, keys, metrics, reference_classifier in specifications:
        generated_path = generated / filename
        if not generated_path.exists():
            continue
        reference = pd.read_csv(source / filename)
        rerun = pd.read_csv(generated_path)
        if reference_classifier is not None:
            reference = reference.loc[
                reference["classifier"] == reference_classifier
            ].drop(columns="classifier")
            if "classifier" in rerun:
                rerun = rerun.loc[rerun["classifier"] == reference_classifier].drop(
                    columns="classifier"
                )
        comparisons[filename] = _write_comparison(
            reference,
            rerun,
            keys,
            metrics,
            filename,
            generated,
        )
    learning_summary_path = generated / "learning_curve_summary.csv"
    if learning_summary_path.exists():
        learning_summary = pd.read_csv(learning_summary_path).rename(
            columns={
                "auc_cv": "auc_cv_raw",
                "convergence_rate": "convergence_rate_raw",
                "performance_stability": "performance_stability_raw",
                "learning_rank": "rank",
            }
        )
        learning_metrics = [
            "auc_cv_raw",
            "convergence_rate_raw",
            "performance_stability_raw",
            "auc_cv_normalized",
            "convergence_rate_normalized",
            "performance_stability_normalized",
            "rank",
        ]
        comparisons["learning_curve_metrics.csv"] = _write_comparison(
            pd.read_csv(source / "learning_curve_metrics.csv"),
            learning_summary,
            ["dataset", "classifier"],
            learning_metrics,
            "learning_curve_metrics.csv",
            generated,
        )
    cluster_selection_path = generated / "cluster_selection.csv"
    if cluster_selection_path.exists():
        reference_clusters = pd.read_csv(source / "clusters.csv")
        generated_clusters = pd.read_csv(cluster_selection_path)

        def canonical_members(value: object) -> str:
            return ";".join(sorted(str(value).replace("|", ";").split(";")))

        reference_clusters["cluster_members"] = reference_clusters["classifiers"].map(
            canonical_members
        )
        generated_clusters["cluster_members"] = generated_clusters["classifiers"].map(
            canonical_members
        )
        comparisons["clusters.csv"] = _write_comparison(
            reference_clusters,
            generated_clusters,
            ["cluster_members", "best_classifier"],
            ["overall_rank", "normalized_weight"],
            "clusters.csv",
            generated,
        )
    ablation_path = generated / "ablation.csv"
    if ablation_path.exists():
        comparisons["ablation.csv"] = _write_comparison(
            pd.read_csv(source / "ablation.csv"),
            pd.read_csv(ablation_path),
            ["scenario_id", "scenario", "selected_classifiers"],
            ["accuracy_bnci2014_002", "accuracy_zhou2016"],
            "ablation.csv",
            generated,
        )
    return comparisons


def verify_publication(config: ExperimentConfig, compare_generated: bool = True) -> dict[str, Any]:
    """Validate immutable paper references and write a machine-readable audit."""

    configured_doi = str(config.section("publication").get("doi", ""))
    if configured_doi != PAPER_DOI:
        raise ValueError("Publication configuration DOI does not match the paper")
    source = config.publication_source
    generated = ensure_directory(config.publication_generated)
    tolerance = float(config.section("publication")["comparison_tolerance"])
    manifest_validation = _validate_source_manifest(source)
    table_validation = _validate_source_tables(source, config, tolerance)
    report_path = generated / "paper_consistency.json"
    report: dict[str, Any] = {
        "paper_doi": PAPER_DOI,
        "config_sha256": config.sha256,
        "source_manifest_sha256": manifest_validation["manifest_sha256"],
        "reference_files": manifest_validation["files"],
        "table_validation": table_validation,
        "weight_consistency": weight_consistency_report(config),
        "generated_comparisons": (
            _compare_generated_tables(source, generated) if compare_generated else {}
        ),
        "report_path": str(report_path),
    }
    write_json(report_path, report)
    return report
