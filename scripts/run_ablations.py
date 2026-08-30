"""PyCharm entry point: reproduce the eleven Table 9 ablation scenarios."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ablations", *sys.argv[1:]]))
