"""PyCharm entry point: download and cache EEG data."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["cache", *sys.argv[1:]]))
