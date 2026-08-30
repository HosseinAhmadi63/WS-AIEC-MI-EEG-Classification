from __future__ import annotations

import numpy as np
import pandas as pd

from wsaiec_eeg.constants import CLASSIFIER_ORDER
from wsaiec_eeg.plotting import figures


def test_figure_three_uses_training_volume_axis(tmp_path, monkeypatch) -> None:
    curves = pd.DataFrame(
        [
            {
                "dataset": "BNCI2014_002",
                "classifier": classifier,
                "training_fraction": fraction,
                "training_volume": volume,
                "train_score": 0.8,
                "validation_score": 0.7,
            }
            for classifier in CLASSIFIER_ORDER
            for fraction, volume in [(0.1, 182), (1.0, 1792)]
        ]
    )
    captured = {}

    def capture(figure, path):
        axis = figure.axes[0]
        captured["x"] = np.asarray(axis.lines[0].get_xdata(), dtype=float)
        captured["label"] = axis.get_xlabel()
        figures.plt.close(figure)
        return path

    monkeypatch.setattr(figures, "_save", capture)
    path = figures.plot_learning_curves(curves, tmp_path)
    np.testing.assert_array_equal(captured["x"], np.asarray([182.0, 1792.0]))
    assert captured["label"] == "Training examples"
    assert path == tmp_path / "figure_3_learning_curves_bnci2014_002.png"
