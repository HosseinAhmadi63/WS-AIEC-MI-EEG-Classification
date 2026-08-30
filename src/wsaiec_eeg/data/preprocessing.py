"""Paper-faithful ordered FIR filtering and half-open MI epoch extraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    data: np.ndarray
    label: int
    channel_names: tuple[str, ...]
    sfreq: float
    session: str
    run: str
    event_index: int


def _session_items(subject_data: dict[str, Any], max_sessions: int) -> list[tuple[str, Any]]:
    """Retain insertion/source order and the first Table-1 sessions only."""

    return list(subject_data.items())[:max_sessions]


def iter_subject_trials(
    dataset: Any,
    subject: int,
    dataset_spec: dict[str, Any],
    preprocessing: dict[str, Any],
) -> Iterator[TrialRecord]:
    """Yield filtered trials in source order with exact duration x sfreq samples."""

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to preprocess the public EEG recordings") from exc

    subject_data = dataset.get_data(subjects=[subject])[subject]
    event_names = tuple(dataset_spec["events"])
    missing_events = set(event_names) - set(dataset.event_id)
    if missing_events:
        raise RuntimeError(f"{dataset.code} is missing configured events {sorted(missing_events)}")
    code_to_label = {int(dataset.event_id[name]): index for index, name in enumerate(event_names)}

    configured_duration = float(dataset_spec["epoch_duration_seconds"])
    interval_start = float(dataset.interval[0])
    source_interval_stop = float(dataset.interval[1])
    source_duration = source_interval_stop - interval_start
    if not np.isclose(source_duration, configured_duration, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"{dataset.code} interval duration changed from the paper protocol: "
            f"configured={configured_duration}, MOABB={source_duration}"
        )
    for session_name, runs in _session_items(subject_data, int(dataset_spec["max_sessions"])):
        for run_name, raw in runs.items():
            annotation_event_id = {
                name: int(dataset.event_id[name]) for name in event_names
            }
            events, _ = mne.events_from_annotations(
                raw,
                event_id=annotation_event_id,
                verbose=False,
            )
            events_are_epoch_starts = len(events) > 0
            if not events_are_epoch_starts:
                stim_indices = mne.pick_types(raw.info, stim=True)
                if len(stim_indices) != 1:
                    raise RuntimeError(
                        f"No usable annotations and {len(stim_indices)} stimulus channels for "
                        f"{dataset.code} subject={subject} session={session_name} run={run_name}"
                    )
                stim_name = raw.ch_names[int(stim_indices[0])]
                events = mne.find_events(
                    raw,
                    stim_channel=stim_name,
                    shortest_event=1,
                    consecutive=True,
                    verbose=False,
                )
                valid = np.isin(events[:, 2], np.fromiter(code_to_label, dtype=int))
                events = events[valid]
            if len(events) == 0:
                continue

            filtered = raw.copy().load_data()
            filtered.filter(
                l_freq=float(preprocessing["fmin_hz"]),
                h_freq=float(preprocessing["fmax_hz"]),
                picks="eeg",
                method=str(preprocessing["filter_method"]),
                phase=str(preprocessing["phase"]),
                fir_window=str(preprocessing["fir_window"]),
                fir_design=str(preprocessing["fir_design"]),
                l_trans_bandwidth=preprocessing["l_trans_bandwidth"],
                h_trans_bandwidth=preprocessing["h_trans_bandwidth"],
                verbose=False,
            )
            sfreq = float(filtered.info["sfreq"])
            expected_samples = int(round(configured_duration * sfreq))
            eeg_picks = mne.pick_types(filtered.info, eeg=True, eog=False, stim=False)
            channel_names = tuple(filtered.ch_names[int(index)] for index in eeg_picks)

            for event_index, event in enumerate(events):
                onset = int(event[0] - filtered.first_samp)
                start = (
                    onset
                    if events_are_epoch_starts
                    else onset + int(round(interval_start * sfreq))
                )
                stop = start + expected_samples
                trial = filtered.get_data(picks=eeg_picks, start=start, stop=stop)
                if trial.shape != (len(channel_names), expected_samples):
                    raise RuntimeError(
                        f"Unexpected epoch shape {trial.shape}; expected "
                        f"({len(channel_names)}, {expected_samples}) for {dataset.code} "
                        f"subject={subject} session={session_name} run={run_name} event={event_index}"
                    )
                yield TrialRecord(
                    data=trial,
                    label=code_to_label[int(event[2])],
                    channel_names=channel_names,
                    sfreq=sfreq,
                    session=str(session_name),
                    run=str(run_name),
                    event_index=event_index,
                )
