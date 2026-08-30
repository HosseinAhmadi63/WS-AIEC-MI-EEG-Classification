"""Shared test configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wsaiec_eeg.config import ExperimentConfig, load_config

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def paper_config(repository_root: Path) -> ExperimentConfig:
    return load_config(repository_root / "configs" / "paper.yaml")
