from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from wsaiec_eeg.constants import PAPER_DOI
from wsaiec_eeg.evaluation.publication import (
    SOURCE_TABLE_ROWS,
    _compare_generated_tables,
    _compare_table,
    _validate_source_tables,
    _validate_table_10,
    verify_publication,
)


def test_tables_one_through_ten_are_self_consistent(paper_config) -> None:
    validation = _validate_source_tables(
        paper_config.publication_source,
        paper_config,
        float(paper_config.section("publication")["comparison_tolerance"]),
    )
    assert validation["table_1"]["rows"] == 6
    assert validation["table_2"] == {
        "rows": 96,
        "datasets": 6,
        "classifiers": 16,
    }
    assert validation["table_4"] == {
        "rows": 96,
        "datasets": 6,
        "classifiers": 16,
    }
    assert validation["table_7"]["members"] == 16
    assert validation["table_8"]["rows"] == 36
    assert validation["table_8"]["wsaiec_rows"] == 6
    assert validation["table_9"]["rows"] == 11
    assert validation["table_10"] == {
        "rows": 26,
        "prior_studies": 25,
        "external_prior_studies": 24,
        "maximum_accuracy_percent": 99.58,
    }


def test_manifest_and_transcribed_tables_are_verified_together(paper_config, tmp_path) -> None:
    raw = deepcopy(paper_config.raw)
    raw["publication"]["generated_outputs"] = str(tmp_path / "generated")
    report = verify_publication(replace(paper_config, raw=raw), compare_generated=False)
    assert report["paper_doi"] == PAPER_DOI
    assert set(report["reference_files"]) == set(SOURCE_TABLE_ROWS)
    assert {
        filename: specification["rows"]
        for filename, specification in report["reference_files"].items()
    } == SOURCE_TABLE_ROWS
    assert report["table_validation"]["table_2"]["rows"] == 96
    assert report["table_validation"]["table_3"]["rows"] == 16
    assert report["table_validation"]["table_4"]["rows"] == 96
    assert report["table_validation"]["table_5"]["rows"] == 16
    assert report["table_validation"]["table_6"]["rows"] == 16
    assert report["table_validation"]["table_7"]["rows"] == 6
    assert report["table_validation"]["table_8"]["rows"] == 36
    assert report["table_validation"]["table_9"]["rows"] == 11
    assert report["table_validation"]["table_10"]["rows"] == 26


def test_table_ten_rejects_a_non_boolean_method_indicator(paper_config) -> None:
    frame = pd.read_csv(paper_config.publication_source / "related_works.csv")
    frame["cst"] = frame["cst"].astype(object)
    frame.loc[0, "cst"] = "false"
    with pytest.raises(ValueError, match="method indicators must be booleans"):
        _validate_table_10(frame)


def test_generated_comparison_reports_numeric_errors(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    pd.DataFrame(
        {
            "dataset": ["A", "B"],
            "accuracy": [0.8, 0.9],
            "rank": [2, 1],
        }
    ).to_csv(reference, index=False)
    pd.DataFrame(
        {
            "dataset": ["A", "B"],
            "accuracy": [0.75, 0.91],
            "rank": [1, 2],
        }
    ).to_csv(generated, index=False)
    comparison = _compare_table(
        reference,
        generated,
        ["dataset"],
        ["accuracy", "rank"],
    )
    assert comparison["accuracy_absolute_error"].tolist() == pytest.approx([0.05, 0.01])
    assert comparison["rank_absolute_error"].tolist() == [1, 1]


def test_generated_comparison_rejects_an_incomplete_key_grid(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    pd.DataFrame({"dataset": ["A", "B"], "accuracy": [0.8, 0.9]}).to_csv(
        reference, index=False
    )
    pd.DataFrame({"dataset": ["A"], "accuracy": [0.8]}).to_csv(generated, index=False)
    with pytest.raises(ValueError, match="incomplete result keys"):
        _compare_table(reference, generated, ["dataset"], ["accuracy"])


def test_generated_table_four_seven_and_nine_mappings(paper_config, tmp_path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    table_4 = pd.read_csv(paper_config.publication_source / "learning_curve_metrics.csv").rename(
        columns={
            "auc_cv_raw": "auc_cv",
            "convergence_rate_raw": "convergence_rate",
            "performance_stability_raw": "performance_stability",
            "rank": "learning_rank",
        }
    )
    table_4["learning_score"] = 0.0
    table_4.to_csv(generated / "learning_curve_summary.csv", index=False)
    table_7 = pd.read_csv(paper_config.publication_source / "clusters.csv")
    table_7["cluster"] = list(reversed(table_7["cluster"].tolist()))
    table_7["classifiers"] = table_7["classifiers"].str.replace(";", "|", regex=False)
    table_7.to_csv(generated / "cluster_selection.csv", index=False)
    table_9 = pd.read_csv(paper_config.publication_source / "ablation.csv")
    table_9.to_csv(generated / "ablation.csv", index=False)

    comparisons = _compare_generated_tables(paper_config.publication_source, generated)
    assert set(comparisons) == {
        "learning_curve_metrics.csv",
        "clusters.csv",
        "ablation.csv",
    }
    assert all(result["maximum_absolute_error"] == 0.0 for result in comparisons.values())
    assert (generated / "comparison_learning_curve_metrics.csv").is_file()
    assert (generated / "comparison_clusters.csv").is_file()
    assert (generated / "comparison_ablation.csv").is_file()
