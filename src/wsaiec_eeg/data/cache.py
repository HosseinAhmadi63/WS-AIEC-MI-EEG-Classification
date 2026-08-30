"""Download and cache each participant as an ordered NumPy epoch tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.data.preprocessing import iter_subject_trials
from wsaiec_eeg.data.registry import create_dataset
from wsaiec_eeg.data.types import EpochDataset


def subject_cache_path(config: ExperimentConfig, dataset_name: str, subject: int) -> Path:
    return config.cache_root / dataset_name / f"subject_{subject:02d}.npz"


def configure_moabb_download_root(config: ExperimentConfig) -> None:
    config.moabb_root.mkdir(parents=True, exist_ok=True)
    try:
        import moabb
    except ImportError as exc:
        raise RuntimeError("MOABB is required to download the BNCI datasets") from exc
    moabb.set_download_dir(str(config.moabb_root))


def download_dataset(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: list[int] | None = None,
    force: bool = False,
) -> None:
    configure_moabb_download_root(config)
    dataset = create_dataset(dataset_name)
    requested = subjects or list(config.datasets[dataset_name]["subjects"])
    dataset.download(
        subject_list=requested,
        path=str(config.moabb_root),
        force_update=force,
        update_path=False,
        accept=True,
        verbose=False,
    )


def cache_subject(
    config: ExperimentConfig,
    dataset_name: str,
    subject: int,
    force: bool = False,
) -> Path:
    """Create and strictly validate one participant cache."""

    target = subject_cache_path(config, dataset_name, subject)
    if target.exists() and not force:
        load_cached_subject(config, dataset_name, subject)
        return target

    configure_moabb_download_root(config)
    dataset = create_dataset(dataset_name)
    spec = config.datasets[dataset_name]
    preprocessing = config.section("preprocessing")
    records = list(iter_subject_trials(dataset, subject, spec, preprocessing))
    if not records:
        raise RuntimeError(f"No motor-imagery trials found for {dataset_name} subject {subject}")

    channel_names = records[0].channel_names
    sfreq = records[0].sfreq
    if any(record.channel_names != channel_names for record in records):
        raise RuntimeError(f"Channel order changes within {dataset_name} subject {subject}")
    if any(record.sfreq != sfreq for record in records):
        raise RuntimeError(f"Sampling rate changes within {dataset_name} subject {subject}")

    cached = EpochDataset(
        dataset=dataset_name,
        subject=subject,
        X=np.stack([record.data for record in records]).astype(preprocessing["output_dtype"]),
        y=np.asarray([record.label for record in records], dtype=np.int64),
        class_names=tuple(spec["events"]),
        channel_names=channel_names,
        sfreq=sfreq,
        session=np.asarray([record.session for record in records]),
        run=np.asarray([record.run for record in records]),
        source_event_index=np.asarray([record.event_index for record in records], dtype=np.int64),
    )
    _validate_against_paper(cached, spec, dataset_name, subject)
    return cached.save(target)


def _validate_against_paper(
    cached: EpochDataset,
    spec: dict[str, object],
    expected_dataset: str | None = None,
    expected_subject: int | None = None,
) -> None:
    cached.validate()
    if expected_dataset is not None and cached.dataset != expected_dataset:
        raise RuntimeError(f"Expected dataset {expected_dataset}, found {cached.dataset}")
    if expected_subject is not None and cached.subject != expected_subject:
        raise RuntimeError(f"Expected subject {expected_subject}, found {cached.subject}")
    expected_classes = tuple(str(name) for name in spec["events"])
    if cached.class_names != expected_classes:
        raise RuntimeError(
            f"Expected class order {expected_classes}, found {cached.class_names}"
        )
    if cached.X.shape[1] != int(spec["channels"]):
        raise RuntimeError(f"Expected {spec['channels']} EEG channels, found {cached.X.shape[1]}")
    if cached.sfreq != float(spec["sampling_rate"]):
        raise RuntimeError(f"Expected {spec['sampling_rate']} Hz, found {cached.sfreq}")
    expected_samples = int(round(float(spec["epoch_duration_seconds"]) * cached.sfreq))
    if cached.X.shape[2] != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} samples per trial, found {cached.X.shape[2]}")
    counts = np.bincount(cached.y, minlength=len(cached.class_names))
    expected_trials = int(spec["trials_per_class"])
    if not np.all(counts == expected_trials):
        raise RuntimeError(
            f"Paper protocol expects {expected_trials} trials/class, found {counts.tolist()} "
            f"for {cached.dataset} subject {cached.subject}"
        )


def cache_dataset(
    config: ExperimentConfig,
    dataset_name: str,
    subjects: list[int] | None = None,
    force: bool = False,
) -> list[Path]:
    requested = subjects or list(config.datasets[dataset_name]["subjects"])
    download_dataset(config, dataset_name, requested, force=False)
    return [cache_subject(config, dataset_name, subject, force=force) for subject in requested]


def load_cached_subject(
    config: ExperimentConfig, dataset_name: str, subject: int
) -> EpochDataset:
    path = subject_cache_path(config, dataset_name, subject)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: wsaiec-eeg cache --config {config.path} "
            f"--dataset {dataset_name} --subject {subject}"
        )
    cached = EpochDataset.load(path)
    _validate_against_paper(cached, config.datasets[dataset_name], dataset_name, subject)
    return cached
