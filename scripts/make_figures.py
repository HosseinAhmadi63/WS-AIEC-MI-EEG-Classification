"""PyCharm entry point: generate article-style figures."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["figures", *sys.argv[1:]]))
