from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wsaiec_eeg.evaluation.ablation import (
    ABLATION_DATASETS,
    EXPECTED_ABLATION_SCENARIOS,
    _dataset_ablation_history_rows,
    _fixed_oof_features,
    _optimize_dataset_scenario_alphas,
    ablation_metrics_exist,
    aggregate_ablations,
    load_ablation_scenarios,
    static_rank_weights,
)


def test_table_9_scenarios_resolve_deterministic_meta_and_bases(paper_config) -> None:
    scenarios = load_ablation_scenarios(paper_config)
    assert tuple(
        (scenario.scenario_id, scenario.scenario, scenario.selected_classifiers)
        for scenario in scenarios
    ) == EXPECTED_ABLATION_SCENARIOS
    assert {scenario.meta_classifier for scenario in scenarios} == {"SVM"}
    assert all("SVM" not in scenario.base_classifiers for scenario in scenarios)
    assert scenarios[8].base_classifiers == ("NB", "PC", "SVM_rbf", "LDA")
    assert scenarios[9].base_classifiers == ("NB", "PC", "GB", "LDA")
    assert scenarios[10].weighting_method == "static_inverse_overall_rank"
    assert all(
        scenario.weighting_method == "dynamic_softmax_validation_accuracy"
        for scenario in scenarios[:10]
    )


def test_static_weights_are_the_printed_table_seven_base_weights(paper_config) -> None:
    reported = paper_config.section("wsaiec")["reported_static_weights"]
    bases = ("NB", "PC", "SVM_rbf", "GB", "LDA")
    weights = static_rank_weights(reported, bases)
    assert weights == {
        "NB": 0.1079,
        "PC": 0.0676,
        "SVM_rbf": 0.1810,
        "GB": 0.0420,
        "LDA": 0.0603,
    }
    assert sum(weights.values()) == pytest.approx(0.4588)
    assert weights["SVM_rbf"] > weights["NB"] > weights["PC"] > weights["LDA"] > weights["GB"]


def test_static_oof_features_discard_unpredicted_prefix() -> None:
    predictions = {
        "A": np.asarray([np.nan, np.nan, 0.0, 1.0]),
        "B": np.asarray([np.nan, np.nan, 1.0, 0.0]),
    }
    features, rows = _fixed_oof_features(
        predictions,
        {"A": 0.75, "B": 0.25},
        ["A", "B"],
    )
    np.testing.assert_array_equal(rows, [2, 3])
    np.testing.assert_allclose(features[rows], [[0.0, 0.25], [0.75, 0.0]])


def test_dynamic_ablation_alpha_is_shared_by_dataset_and_scenario(
    monkeypatch,
    paper_config,
) -> None:
    scenarios = load_ablation_scenarios(paper_config)
    subjects = [1, 2]
    prepared = {subject: object() for subject in subjects}
    calls: list[tuple[list[int], int]] = []

    def optimize(keys, objective, settings, seed):
        current = list(keys)
        calls.append((current, seed))
        scores = {key: objective(key, 1.5) for key in current}
        return 1.5, [
            {
                "iteration": 1,
                "alpha": 1.5,
                "score": float(np.mean(list(scores.values()))),
                "subject_scores": scores,
            }
        ]

    monkeypatch.setattr(
        "wsaiec_eeg.evaluation.ablation.optimize_shared_alpha",
        optimize,
    )
    monkeypatch.setattr(
        "wsaiec_eeg.evaluation.ablation._scenario_subject_score",
        lambda subject_data, scenario, alpha, settings: scenario.scenario_id / 100 + alpha / 10,
    )
    monkeypatch.setattr(
        "wsaiec_eeg.evaluation.ablation._static_scenario_subject_score",
        lambda subject_data, scenario, weights, settings: 0.75,
    )

    alphas, histories = _optimize_dataset_scenario_alphas(
        paper_config,
        "BNCI2014_002",
        subjects,
        prepared,
        scenarios,
    )

    assert len(calls) == 10
    assert all(keys == subjects for keys, _ in calls)
    assert all(alphas[scenario_id] == 1.5 for scenario_id in range(1, 11))
    assert alphas[11] is None
    assert histories[3][0]["subject_scores"] == {1: 0.18, 2: 0.18}
    assert histories[11][0]["subject_scores"] == {1: 0.75, 2: 0.75}
    assert histories[11][0]["score"] == 0.75


def test_dataset_ablation_history_records_scope_and_debug_cohort(paper_config) -> None:
    scenarios = load_ablation_scenarios(paper_config)
    alphas = {scenario.scenario_id: 2.0 for scenario in scenarios}
    alphas[11] = None
    histories = {
        scenario.scenario_id: [
            {
                "iteration": 1,
                "alpha": float("nan") if scenario.scenario_id == 11 else 2.0,
                "score": 0.8,
                "subject_scores": {1: 0.7, 2: 0.9},
            }
        ]
        for scenario in scenarios
    }
    rows = _dataset_ablation_history_rows(
        "BNCI2014_002",
        scenarios,
        alphas,
        histories,
        list(paper_config.datasets["BNCI2014_002"]["subjects"]),
        [1, 2],
        "debug_subject_subset",
    )
    frame = pd.DataFrame(rows)
    assert len(frame) == 11
    assert set(frame.loc[frame["scenario_id"] <= 10, "alpha_scope"]) == {
        "dataset_scenario"
    }
    assert frame.loc[frame["scenario_id"] == 11, "alpha_scope"].item() == (
        "not_applicable_static"
    )
    assert set(frame["subject_mode"]) == {"debug_subject_subset"}
    assert set(frame["requested_subjects"]) == {"[1, 2]"}
    assert frame["selected"].all()


def test_ablation_aggregation_requires_and_aggregates_the_exact_cohort(
    paper_config,
    tmp_path,
) -> None:
    raw = deepcopy(paper_config.raw)
    raw["project"]["results_root"] = str(tmp_path / "results")
    raw["publication"]["generated_outputs"] = str(tmp_path / "generated")
    config = replace(paper_config, raw=raw)
    scenarios = load_ablation_scenarios(config)
    assert not ablation_metrics_exist(config)
    with pytest.raises(FileNotFoundError, match="Incomplete Table 9 cohort"):
        aggregate_ablations(config)

    expected: dict[tuple[int, str], float] = {}
    for dataset_index, dataset in enumerate(ABLATION_DATASETS):
        subject_values = []
        for subject in config.datasets[dataset]["subjects"]:
            directory = config.run_root / dataset / f"subject_{int(subject):02d}"
            directory.mkdir(parents=True, exist_ok=True)
            rows = []
            for scenario in scenarios:
                accuracy = 0.6 + dataset_index * 0.05 + scenario.scenario_id * 0.01 + subject * 0.0001
                subject_values.append((scenario.scenario_id, accuracy))
                rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "scenario_id": scenario.scenario_id,
                        "scenario": scenario.scenario,
                        "selected_classifiers": scenario.selected_text,
                        "meta_classifier": scenario.meta_classifier,
                        "base_classifiers": ";".join(scenario.base_classifiers),
                        "weighting_method": scenario.weighting_method,
                        "weight_sum": 0.4588 if scenario.scenario_id == 11 else 1.0,
                        "accuracy": accuracy,
                        "precision": accuracy,
                        "recall": accuracy,
                        "f1": accuracy,
                        "auc_roc": accuracy,
                        "kappa": accuracy,
                        "score": accuracy,
                    }
                )
            pd.DataFrame(rows).to_csv(directory / "ablation_metrics.csv", index=False)
        for scenario in scenarios:
            expected[(scenario.scenario_id, dataset)] = float(
                np.mean(
                    [
                        accuracy
                        for scenario_id, accuracy in subject_values
                        if scenario_id == scenario.scenario_id
                    ]
                )
            )

    assert ablation_metrics_exist(config)
    path = aggregate_ablations(config)
    result = pd.read_csv(path)
    assert result.columns.tolist() == [
        "scenario_id",
        "scenario",
        "selected_classifiers",
        "accuracy_bnci2014_002",
        "accuracy_zhou2016",
    ]
    assert result["scenario_id"].tolist() == list(range(1, 12))
    for row in result.itertuples(index=False):
        assert row.accuracy_bnci2014_002 == pytest.approx(
            expected[(row.scenario_id, "BNCI2014_002")]
        )
        assert row.accuracy_zhou2016 == pytest.approx(
            expected[(row.scenario_id, "Zhou2016")]
        )
    assert (config.publication_generated / "ablation_subject_metrics.csv").is_file()
    assert (config.publication_generated / "ablation_protocol.json").is_file()
