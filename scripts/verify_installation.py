"""PyCharm entry point: verify paper references and run synthetic smoke test."""

from __future__ import annotations

import argparse

from wsaiec_eeg.cli import main


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default="results/smoke")
    arguments = parser.parse_args()
    status = main(
        ["verify-paper", "--config", arguments.config, "--no-generated-comparison"]
    )
    if status:
        return status
    return main(["smoke", "--config", arguments.config, "--output", arguments.output])


if __name__ == "__main__":
    raise SystemExit(run())
