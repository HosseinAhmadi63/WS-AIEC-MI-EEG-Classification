from __future__ import annotations

from wsaiec_eeg.constants import CLASSIFIER_ORDER
from wsaiec_eeg.models.classifiers import make_classifier
from wsaiec_eeg.models.mlp import TorchMLPClassifier


def test_all_sixteen_classifier_factories(paper_config) -> None:
    settings = paper_config.section("classifiers")
    models = {name: make_classifier(name, settings, seed=42, n_jobs=1) for name in CLASSIFIER_ORDER}
    assert len(models) == 16
    assert isinstance(models["MLP"], TorchMLPClassifier)
    assert models["SVM_rbf"].kernel == "rbf"
    assert models["RF"].n_estimators == 100
    assert models["ET"].bootstrap is False
