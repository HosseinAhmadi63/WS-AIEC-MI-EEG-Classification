from __future__ import annotations

import numpy as np

from wsaiec_eeg.models.wsaiec import (
    dynamic_weights,
    make_meta_classifier,
    optimize_alpha_gp,
    optimize_shared_alpha,
    softmax_weights,
)


def test_equation_9_softmax_weights() -> None:
    accuracies = {"A": 0.9, "B": 0.7, "C": 0.5}
    resolved = dynamic_weights(accuracies, alpha=2.0, order=["A", "B", "C"])
    expected = np.exp(2.0 * np.asarray([0.9, 0.7, 0.5]))
    expected /= expected.sum()
    np.testing.assert_allclose(
        [resolved[name] for name in ["A", "B", "C"]],
        expected,
    )
    assert resolved["A"] > resolved["B"] > resolved["C"]
    assert np.isclose(sum(resolved.values()), 1.0)


def test_equation_9_is_uniform_at_zero_and_numerically_stable() -> None:
    np.testing.assert_allclose(softmax_weights([0.1, 0.5, 0.9], 0.0), np.full(3, 1 / 3))
    concentrated = softmax_weights([0.0, 0.5, 1.0], 10_000.0)
    assert np.isfinite(concentrated).all()
    assert concentrated[-1] == 1.0


def test_bayesian_alpha_history_and_linear_meta_factory(paper_config) -> None:
    best, history = optimize_alpha_gp(
        objective=lambda alpha: -(alpha - 1.4) ** 2,
        lower=0.1,
        upper=3.0,
        initial_points=3,
        iterations=7,
        candidates_per_iteration=64,
        kappa=2.0,
        seed=9,
    )
    assert len(history) == 7
    assert best in {float(record["alpha"]) for record in history}
    assert max(float(record["score"]) for record in history) == next(
        float(record["score"]) for record in history if float(record["alpha"]) == best
    )
    meta = make_meta_classifier(paper_config, seed=42)
    assert meta.kernel == "linear"
    assert meta.C == 1.0
    assert meta.random_state == 42


def test_shared_alpha_uses_mean_subject_validation_accuracy() -> None:
    best, history = optimize_shared_alpha(
        [1, 2, 3],
        lambda subject, alpha: 1.0 - alpha if subject == 1 else alpha,
        {
            "lower": 0.1,
            "upper": 1.0,
            "initial_points": 2,
            "iterations": 2,
            "candidates_per_iteration": 2,
            "kappa": 2.0,
        },
        seed=7,
    )
    assert best == 1.0
    selected = next(record for record in history if record["alpha"] == best)
    assert selected["subject_scores"] == {1: 0.0, 2: 1.0, 3: 1.0}
    assert np.isclose(float(selected["score"]), 2 / 3)
