"""PyCharm entry point: run all 16 base classifiers."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
