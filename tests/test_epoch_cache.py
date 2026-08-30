from __future__ import annotations

import numpy as np
import pytest

from wsaiec_eeg.data.cache import _validate_against_paper
from wsaiec_eeg.data.preprocessing import iter_subject_trials
from wsaiec_eeg.data.types import EpochDataset


def test_epoch_cache_round_trip(tmp_path) -> None:
    cached = EpochDataset(
        dataset="synthetic",
        subject=1,
        X=np.ones((8, 3, 20), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        class_names=("left", "right"),
        channel_names=("C3", "Cz", "C4"),
        sfreq=100.0,
        session=np.asarray(["0"] * 8),
        run=np.asarray(["0"] * 8),
        source_event_index=np.arange(8),
    )
    path = cached.save(tmp_path / "subject.npz")
    loaded = EpochDataset.load(path)
    assert loaded.dataset == cached.dataset
    assert loaded.class_names == cached.class_names
    np.testing.assert_array_equal(loaded.X, cached.X)
    np.testing.assert_array_equal(loaded.y, cached.y)


def test_paper_cache_validation_rejects_swapped_class_mapping() -> None:
    cached = EpochDataset(
        dataset="synthetic",
        subject=1,
        X=np.ones((4, 3, 20), dtype=np.float32),
        y=np.asarray([0, 1, 0, 1]),
        class_names=("right", "left"),
        channel_names=("C3", "Cz", "C4"),
        sfreq=100.0,
        session=np.asarray(["0"] * 4),
        run=np.asarray(["0"] * 4),
        source_event_index=np.arange(4),
    )
    specification = {
        "events": ["left", "right"],
        "channels": 3,
        "sampling_rate": 100,
        "epoch_duration_seconds": 0.2,
        "trials_per_class": 2,
    }
    with pytest.raises(RuntimeError, match="Expected class order"):
        _validate_against_paper(cached, specification, "synthetic", 1)


def test_annotation_only_moabb_run_uses_annotation_as_epoch_start() -> None:
    import mne

    info = mne.create_info(["C3", "C4"], 100.0, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(np.ones((2, 1000)), info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[8.0], duration=[1.0], description=["left"]))

    class Dataset:
        code = "SyntheticAnnotations"
        event_id = {"left": 1}
        interval = [2.0, 3.0]

        def get_data(self, subjects):
            return {subjects[0]: {"0": {"0": raw}}}

    records = list(
        iter_subject_trials(
            Dataset(),
            1,
            {
                "events": ["left"],
                "epoch_duration_seconds": 1.0,
                "max_sessions": 1,
            },
            {
                "fmin_hz": 7.0,
                "fmax_hz": 30.0,
                "filter_method": "fir",
                "phase": "zero",
                "fir_window": "hamming",
                "fir_design": "firwin",
                "l_trans_bandwidth": 2.0,
                "h_trans_bandwidth": 2.0,
            },
        )
    )
    assert len(records) == 1
    assert records[0].data.shape == (2, 100)
