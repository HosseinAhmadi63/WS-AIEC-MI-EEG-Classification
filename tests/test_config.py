from __future__ import annotations

from wsaiec_eeg.constants import CLASSIFIER_ORDER, DATASET_ORDER
from wsaiec_eeg.data.registry import create_dataset


def test_frozen_paper_protocol(paper_config) -> None:
    assert tuple(paper_config.datasets) == DATASET_ORDER
    assert tuple(paper_config.section("classifiers")["order"]) == CLASSIFIER_ORDER
    assert paper_config.section("preprocessing")["fmin_hz"] == 7.0
    assert paper_config.section("preprocessing")["fmax_hz"] == 30.0
    assert paper_config.section("splitting")["time_series_splits"] == 8
    assert paper_config.section("ranking")["reported_cluster_winners"] == [
        "NB",
        "PC",
        "SVM_rbf",
        "GB",
        "LDA",
        "SVM",
    ]
    assert paper_config.section("wsaiec")["base_classifiers"] == [
        "NB",
        "PC",
        "SVM_rbf",
        "GB",
        "LDA",
    ]


def test_table_one_counts(paper_config) -> None:
    expected = {
        "BNCI2014_001": (9, 22, 144, 4.0, 250),
        "BNCI2014_002": (14, 15, 80, 5.0, 512),
        "BNCI2014_004": (9, 3, 360, 4.5, 250),
        "BNCI2015_001": (12, 13, 200, 5.0, 512),
        "Zhou2016": (4, 14, 150, 5.0, 250),
        "AlexMI": (8, 16, 20, 3.0, 512),
    }
    for name, (subjects, channels, trials_per_class, duration, sfreq) in expected.items():
        spec = paper_config.datasets[name]
        assert len(spec["subjects"]) == subjects
        assert spec["channels"] == channels
        assert spec["trials_per_class"] == trials_per_class
        assert spec["epoch_duration_seconds"] == duration
        assert spec["sampling_rate"] == sfreq


def test_config_hash_isolates_caches_runs_and_generated_outputs(paper_config) -> None:
    assert paper_config.cache_root.name == paper_config.run_key
    assert paper_config.run_root.name == paper_config.run_key
    assert paper_config.publication_generated.name == paper_config.run_key


def test_all_pinned_moabb_dataset_constructors() -> None:
    for name in DATASET_ORDER:
        dataset = create_dataset(name)
        assert dataset.subject_list
