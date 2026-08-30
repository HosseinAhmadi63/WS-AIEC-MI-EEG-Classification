"""Typed EEG epoch container and stable on-disk representation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpochDataset:
    """One participant's ordered, preprocessed motor-imagery trials."""

    dataset: str
    subject: int
    X: np.ndarray
    y: np.ndarray
    class_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    sfreq: float
    session: np.ndarray
    run: np.ndarray
    source_event_index: np.ndarray

    def validate(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must have shape trials x channels x samples, got {self.X.shape}")
        n_trials = len(self.X)
        for name, array in {
            "y": self.y,
            "session": self.session,
            "run": self.run,
            "source_event_index": self.source_event_index,
        }.items():
            if len(array) != n_trials:
                raise ValueError(f"{name} length {len(array)} does not match {n_trials} trials")
        if self.X.shape[1] != len(self.channel_names):
            raise ValueError("Channel names do not match X")
        if not np.isfinite(self.X).all():
            raise ValueError("EEG cache contains NaN or infinite values")
        if not np.array_equal(np.unique(self.y), np.arange(len(self.class_names))):
            raise ValueError("Labels must be contiguous integers aligned with class_names")
        if self.sfreq <= 0:
            raise ValueError("Sampling frequency must be positive")

    def save(self, path: str | Path) -> Path:
        """Atomically save a compressed cache file."""

        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".npz", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary,
                dataset=np.asarray(self.dataset),
                subject=np.asarray(self.subject, dtype=np.int64),
                X=np.asarray(self.X, dtype=np.float32),
                y=np.asarray(self.y, dtype=np.int64),
                class_names=np.asarray(self.class_names, dtype=str),
                channel_names=np.asarray(self.channel_names, dtype=str),
                sfreq=np.asarray(self.sfreq, dtype=np.float64),
                session=np.asarray(self.session, dtype=str),
                run=np.asarray(self.run, dtype=str),
                source_event_index=np.asarray(self.source_event_index, dtype=np.int64),
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path) -> EpochDataset:
        with np.load(Path(path), allow_pickle=False) as payload:
            result = cls(
                dataset=str(payload["dataset"].item()),
                subject=int(payload["subject"].item()),
                X=payload["X"],
                y=payload["y"],
                class_names=tuple(str(value) for value in payload["class_names"]),
                channel_names=tuple(str(value) for value in payload["channel_names"]),
                sfreq=float(payload["sfreq"].item()),
                session=payload["session"],
                run=payload["run"],
                source_event_index=payload["source_event_index"],
            )
        result.validate()
        return result
