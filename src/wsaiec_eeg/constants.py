"""Publication constants that are identities, not tunable defaults."""

from __future__ import annotations

PAPER_TITLE = (
    "Enhancing MI EEG Signal Classification With a Novel Weighted and Stacked "
    "Adaptive Integrated Ensemble Model: A Multi-Dataset Approach"
)
PAPER_DOI = "10.1109/ACCESS.2024.3434654"

DATASET_ORDER = (
    "BNCI2014_001",
    "BNCI2014_002",
    "BNCI2014_004",
    "BNCI2015_001",
    "Zhou2016",
    "AlexMI",
)

PAPER_FULL_TRAINING_VOLUMES = {
    "BNCI2014_001": 4140,
    "BNCI2014_002": 1792,
    "BNCI2014_004": 5184,
    "BNCI2015_001": 3840,
    "Zhou2016": 1440,
    "AlexMI": 384,
}

CLASSIFIER_ORDER = (
    "LDA",
    "LR",
    "PC",
    "SGD",
    "RC",
    "SVM",
    "SVM_rbf",
    "KN",
    "NB",
    "DT",
    "RF",
    "ET",
    "GB",
    "AB",
    "QDA",
    "MLP",
)

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "auc_roc", "kappa")
CLUSTER_WINNERS = ("NB", "PC", "SVM_rbf", "GB", "LDA", "SVM")
BASE_CLASSIFIERS = ("NB", "PC", "SVM_rbf", "GB", "LDA")
META_CLASSIFIER = "SVM"
