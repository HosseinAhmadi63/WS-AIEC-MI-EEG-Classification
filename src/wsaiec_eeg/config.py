"""Strict loading and validation of the frozen experiment configuration."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wsaiec_eeg.constants import (
    BASE_CLASSIFIERS,
    CLASSIFIER_ORDER,
    CLUSTER_WINNERS,
    DATASET_ORDER,
    META_CLASSIFIER,
    METRIC_COLUMNS,
    PAPER_FULL_TRAINING_VOLUMES,
)


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration with repository-relative path handling."""

    path: Path
    root: Path
    raw: dict[str, Any]
    sha256: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw[name]
        if not isinstance(value, dict):
            raise ConfigurationError(f"Configuration section {name!r} must be a mapping")
        return value

    @property
    def run_key(self) -> str:
        return self.sha256[:12]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def project(self) -> dict[str, Any]:
        return self.section("project")

    @property
    def datasets(self) -> dict[str, dict[str, Any]]:
        return self.section("datasets")

    @property
    def results_root(self) -> Path:
        return self.resolve(self.project["results_root"])

    @property
    def run_root(self) -> Path:
        return self.results_root / "runs" / self.run_key

    @property
    def cache_root(self) -> Path:
        return self.resolve(self.project["cache_root"]) / self.run_key

    @property
    def moabb_root(self) -> Path:
        return self.resolve(self.project["moabb_root"])

    @property
    def publication_source(self) -> Path:
        return self.resolve(self.section("publication")["reference_tables"])

    @property
    def publication_generated(self) -> Path:
        return self.resolve(self.section("publication")["generated_outputs"]) / self.run_key


REQUIRED_TOP_LEVEL = {
    "project",
    "datasets",
    "preprocessing",
    "features",
    "splitting",
    "evaluation",
    "classifiers",
    "tuning",
    "ranking",
    "wsaiec",
    "publication",
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML, reject malformed paper settings, and compute its run identity."""

    config_path = Path(path).expanduser().resolve()
    payload = config_path.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    missing = REQUIRED_TOP_LEVEL - set(raw)
    unknown = set(raw) - REQUIRED_TOP_LEVEL
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown configuration sections: {sorted(unknown)}")

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    config = ExperimentConfig(
        path=config_path,
        root=root.resolve(),
        raw=raw,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate all publication-defining invariants before any expensive work."""

    if tuple(config.datasets) != DATASET_ORDER:
        raise ConfigurationError(
            f"datasets must appear in paper order {list(DATASET_ORDER)}, got {list(config.datasets)}"
        )
    for name, spec in config.datasets.items():
        subjects = spec.get("subjects")
        if not isinstance(subjects, list) or not subjects or len(subjects) != len(set(subjects)):
            raise ConfigurationError(f"{name}.subjects must be a non-empty unique list")
        if spec.get("epoch_duration_seconds", 0) <= 0:
            raise ConfigurationError(f"{name}.epoch_duration_seconds must be positive")
        if spec.get("channels", 0) <= 0 or spec.get("sampling_rate", 0) <= 0:
            raise ConfigurationError(f"{name} has invalid acquisition metadata")
        if spec.get("trials_per_class", 0) <= 0 or spec.get("paper_trials_per_class", 0) <= 0:
            raise ConfigurationError(f"{name}.trials_per_class must be positive")
        if len(spec.get("events", [])) not in {2, 3, 4}:
            raise ConfigurationError(f"{name}.events must contain two, three, or four classes")

    preprocessing = config.section("preprocessing")
    if preprocessing["fmin_hz"] != 7.0 or preprocessing["fmax_hz"] != 30.0:
        raise ConfigurationError("The paper configuration must use the reported 7-30 Hz band")
    if preprocessing["filter_method"] != "fir" or preprocessing["fir_window"] != "hamming":
        raise ConfigurationError("The paper configuration requires FIR filtering with a Hamming window")
    if (
        float(preprocessing["l_trans_bandwidth"]) != 2.0
        or float(preprocessing["h_trans_bandwidth"]) != 2.0
    ):
        raise ConfigurationError("The reported stopbands require 2 Hz transition bands")
    if preprocessing["resample_hz"] is not None or preprocessing["baseline"] is not None:
        raise ConfigurationError("The frozen executable protocol does not resample or baseline-correct epochs")
    if bool(preprocessing["reject_artifacts"]):
        raise ConfigurationError("The frozen executable protocol retains the nominal trial cohort")

    splitting = config.section("splitting")
    if not 0 < float(splitting["outer_test_fraction"]) < 1:
        raise ConfigurationError("outer_test_fraction must be in (0, 1)")
    if int(splitting["time_series_splits"]) != 8:
        raise ConfigurationError("The reported protocol uses exactly eight time-series splits")
    fractions = [float(value) for value in splitting["training_fractions"]]
    if any(not 0 < value <= 1 for value in fractions) or fractions != sorted(fractions):
        raise ConfigurationError("training_fractions must be sorted values in (0, 1]")
    for name, spec in config.datasets.items():
        trials_per_subject = int(spec["trials_per_class"]) * len(spec["events"])
        development_per_subject = trials_per_subject - int(
            math.ceil(trials_per_subject * float(splitting["outer_test_fraction"]))
        )
        pooled_volume = development_per_subject * len(spec["subjects"])
        if pooled_volume != PAPER_FULL_TRAINING_VOLUMES[name]:
            raise ConfigurationError(
                f"{name} full pooled learning-curve volume must be "
                f"{PAPER_FULL_TRAINING_VOLUMES[name]}, got {pooled_volume}"
            )

    classifier_section = config.section("classifiers")
    if tuple(classifier_section["order"]) != CLASSIFIER_ORDER:
        raise ConfigurationError("Classifier order does not match the 16 classifiers in the paper")
    if set(classifier_section) - {"order"} != set(CLASSIFIER_ORDER):
        raise ConfigurationError("Every classifier must have exactly one parameter mapping")

    evaluation = config.section("evaluation")
    if tuple(evaluation["metrics"]) != METRIC_COLUMNS:
        raise ConfigurationError(f"Evaluation metrics must be {list(METRIC_COLUMNS)}")
    if evaluation["subject_aggregation"] != "mean":
        raise ConfigurationError("Paper tables average metrics across participants")
    if evaluation["composite_score"] != "arithmetic_mean":
        raise ConfigurationError("The classifier score is the arithmetic mean of six metrics")

    tuning = config.section("tuning")
    if tuning["scoring"] != "accuracy":
        raise ConfigurationError("The executable tuning protocol uses validation accuracy")

    ranking = config.section("ranking")
    if int(ranking["number_of_clusters"]) != 6:
        raise ConfigurationError("The paper selects six hierarchical clusters")
    if ranking["clustering_method"] != "ward" or ranking["clustering_metric"] != "euclidean":
        raise ConfigurationError("The frozen clustering protocol uses Ward linkage and Euclidean distance")
    if ranking["learning_ranking_replay_mode"] not in {"paper_reported", "disabled"}:
        raise ConfigurationError(
            "learning_ranking_replay_mode must be paper_reported or disabled"
        )
    winners = tuple(ranking["reported_cluster_winners"])
    if winners != CLUSTER_WINNERS:
        raise ConfigurationError(f"Reported cluster winners must be {list(CLUSTER_WINNERS)}")
    overall_ranks = {str(name): int(rank) for name, rank in ranking["reported_overall_ranks"].items()}
    if set(overall_ranks) != set(CLASSIFIER_ORDER) or sorted(overall_ranks.values()) != list(
        range(1, 17)
    ):
        raise ConfigurationError("reported_overall_ranks must assign ranks 1 through 16")

    wsaiec = config.section("wsaiec")
    if wsaiec["meta_features"] != "hard_predictions":
        raise ConfigurationError("Equation 10 requires one weighted prediction per base classifier")
    if wsaiec["meta_classifier"] != META_CLASSIFIER:
        raise ConfigurationError("The paper designates linear SVM as the meta-classifier")
    if tuple(wsaiec["base_classifiers"]) != BASE_CLASSIFIERS:
        raise ConfigurationError(f"WS-AIEC base classifiers must be {list(BASE_CLASSIFIERS)}")
    if float(wsaiec["validation_fraction"]) != 0.20:
        raise ConfigurationError("Dynamic weights use the reported 20% validation subset")
    if wsaiec["dynamic_weight_formula"] != "softmax_accuracy":
        raise ConfigurationError("Dynamic weights must implement Equation 9")
    alpha = wsaiec["alpha_optimization"]
    if not 0 < float(alpha["lower"]) < float(alpha["upper"]):
        raise ConfigurationError("Bayesian alpha bounds must be positive and ordered")
    static_weights = {str(name): float(value) for name, value in wsaiec["reported_static_weights"].items()}
    if set(static_weights) != set(CLUSTER_WINNERS):
        raise ConfigurationError("Table 7 static weights must cover all six cluster winners")
    if abs(sum(static_weights.values()) - 0.9999) > 1e-9:
        raise ConfigurationError("Rounded Table 7 static weights must sum to 0.9999")


def config_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    """Return serializable configuration identity metadata."""

    return {
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "run_key": config.run_key,
        "configuration": config.raw,
    }
