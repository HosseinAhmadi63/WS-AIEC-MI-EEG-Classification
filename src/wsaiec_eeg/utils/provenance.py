"""Capture sufficient provenance to audit an experiment run."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wsaiec_eeg.config import ExperimentConfig, config_snapshot
from wsaiec_eeg.utils.io import write_json
from wsaiec_eeg.version import __version__


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_manifest(config: ExperimentConfig, command: str) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(config.root),
        "dependencies": {
            name: _version(name)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "mne",
                "moabb",
                "matplotlib",
                "PyYAML",
                "joblib",
                "torch",
            ]
        },
        **config_snapshot(config),
    }


def write_manifest(config: ExperimentConfig, command: str) -> Path:
    path = config.run_root / "manifest.json"
    write_json(path, build_manifest(config, command))
    return path
