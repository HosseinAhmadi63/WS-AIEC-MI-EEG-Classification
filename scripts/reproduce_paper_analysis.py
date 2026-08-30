"""PyCharm entry point: aggregate, plot, and compare with the article."""

from __future__ import annotations

import sys

from wsaiec_eeg.cli import main


def run() -> int:
    aggregate_status = main(["aggregate", *sys.argv[1:]])
    if aggregate_status:
        return aggregate_status
    figures_status = main(["figures", "--source", "generated", *sys.argv[1:]])
    if figures_status:
        return figures_status
    return main(["verify-paper", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(run())
