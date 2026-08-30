"""PyCharm entry point: train and test WS-AIEC."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["wsaiec", *sys.argv[1:]]))
