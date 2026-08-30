"""Exact MOABB dataset registry for Table 1 of the paper."""

from __future__ import annotations

from typing import Any

from wsaiec_eeg.constants import DATASET_ORDER


def dataset_names() -> tuple[str, ...]:
    return DATASET_ORDER


def create_dataset(name: str) -> Any:
    """Construct a pinned MOABB dataset without importing MOABB at package import."""

    try:
        from moabb.datasets import (
            BNCI2014_001,
            BNCI2014_002,
            BNCI2014_004,
            BNCI2015_001,
            AlexMI,
            Zhou2016,
        )
    except ImportError as exc:
        raise RuntimeError("MOABB is required for public EEG download and caching") from exc

    factories = {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_002": BNCI2014_002,
        "BNCI2014_004": BNCI2014_004,
        "BNCI2015_001": BNCI2015_001,
        "Zhou2016": Zhou2016,
        "AlexMI": AlexMI,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASET_ORDER)}") from exc
