"""PyCharm entry point: run the 10%-100% training-volume analysis."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["learning-curves", *sys.argv[1:]]))
