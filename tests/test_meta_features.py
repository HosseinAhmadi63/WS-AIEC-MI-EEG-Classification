from __future__ import annotations

import numpy as np
import pytest

from wsaiec_eeg.models.wsaiec import weighted_meta_features


def test_weighted_hard_predictions_form_one_column_per_base() -> None:
    predictions = {
        "A": np.asarray([1, 0, 2]),
        "B": np.asarray([0, 2, 1]),
    }
    result = weighted_meta_features(predictions, {"A": 0.75, "B": 0.25}, ["A", "B"])
    np.testing.assert_allclose(
        result,
        [[0.75, 0.0], [0.0, 0.5], [1.5, 0.25]],
    )
    assert result.shape == (3, 2)


def test_probability_blocks_are_rejected_by_equation_10() -> None:
    probabilities = {
        "A": np.asarray([[0.2, 0.8], [0.7, 0.3]]),
        "B": np.asarray([[0.9, 0.1], [0.4, 0.6]]),
    }
    with pytest.raises(ValueError, match="one hard-prediction column"):
        weighted_meta_features(probabilities, {"A": 0.75, "B": 0.25}, ["A", "B"])
