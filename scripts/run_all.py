"""PyCharm entry point: complete paper pipeline."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
