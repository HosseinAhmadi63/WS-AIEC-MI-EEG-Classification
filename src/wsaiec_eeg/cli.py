"""Command-line interface used directly and by every PyCharm wrapper script."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from wsaiec_eeg.config import ExperimentConfig, load_config
from wsaiec_eeg.constants import DATASET_ORDER
from wsaiec_eeg.utils.provenance import write_manifest

LOGGER = logging.getLogger("wsaiec_eeg")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/paper.yaml", help="Experiment YAML file")


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show progress for long-running stages",
    )


def _add_selection(
    parser: argparse.ArgumentParser,
    dataset_choices: Sequence[str] = DATASET_ORDER,
) -> None:
    parser.add_argument(
        "--dataset",
        action="append",
        choices=dataset_choices,
        help="Dataset to process; repeat the flag or omit it for every supported dataset",
    )
    parser.add_argument(
        "--subject",
        action="append",
        type=int,
        help="Participant number; repeat the flag or omit it for all configured subjects",
    )
    parser.add_argument("--force", action="store_true", help="Recompute completed artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsaiec-eeg",
        description="Reproduce the WS-AIEC motor-imagery EEG experiments",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("cache", "download, filter, epoch, and cache the public EEG data"),
        ("benchmark", "evaluate all 16 base classifiers"),
        ("learning-curves", "evaluate training volumes from 10 through 100 percent"),
        ("wsaiec", "train and evaluate the adaptive five-base weighted stack"),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        _add_verbose(command)
        _add_config(command)
        _add_selection(command)

    from wsaiec_eeg.evaluation.ablation import ABLATION_DATASETS

    ablations = subparsers.add_parser(
        "ablations",
        help="run the eleven Table 9 ensemble ablations on the two reported datasets",
    )
    _add_verbose(ablations)
    _add_config(ablations)
    _add_selection(ablations, ABLATION_DATASETS)

    aggregate = subparsers.add_parser("aggregate", help="build cross-subject paper tables")
    _add_verbose(aggregate)
    _add_config(aggregate)
    aggregate.add_argument(
        "--without-learning-curves", action="store_true", help="aggregate before learning curves exist"
    )

    figures = subparsers.add_parser("figures", help="generate equivalents of article Figures 2-6")
    _add_verbose(figures)
    _add_config(figures)
    figures.add_argument("--source", choices=["paper", "generated"], default="paper")

    verify = subparsers.add_parser(
        "verify-paper", help="verify frozen paper values and compare generated tables when present"
    )
    _add_verbose(verify)
    _add_config(verify)
    verify.add_argument("--no-generated-comparison", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run a no-download synthetic end-to-end check")
    _add_verbose(smoke)
    _add_config(smoke)
    smoke.add_argument("--output", default="results/smoke")

    run_all = subparsers.add_parser("all", help="run the complete ordered reproduction pipeline")
    _add_verbose(run_all)
    _add_config(run_all)
    run_all.add_argument("--force", action="store_true")
    return parser


def _datasets(config: ExperimentConfig, requested: list[str] | None) -> list[str]:
    return requested or list(config.datasets)


def _subjects(config: ExperimentConfig, dataset: str, requested: list[int] | None) -> list[int]:
    configured = list(config.datasets[dataset]["subjects"])
    if requested is None:
        return configured
    invalid = set(requested) - set(configured)
    if invalid:
        raise ValueError(f"Invalid subjects for {dataset}: {sorted(invalid)}")
    return requested


def _run_selected(args: argparse.Namespace, config: ExperimentConfig) -> None:
    from wsaiec_eeg.data.cache import cache_dataset
    from wsaiec_eeg.evaluation.base_benchmark import (
        run_dataset_benchmark,
        run_dataset_learning_curves,
    )
    from wsaiec_eeg.evaluation.stacking import run_dataset_wsaiec

    if args.command == "learning-curves" and args.subject is not None:
        raise ValueError(
            "Paper learning curves are dataset-scoped and require every configured subject; "
            "remove --subject"
        )
    write_manifest(config, args.command)
    for dataset in _datasets(config, args.dataset):
        if args.command == "learning-curves":
            subjects = list(config.datasets[dataset]["subjects"])
            LOGGER.info("%s: pooled learning curves for %d participant(s)", dataset, len(subjects))
            run_dataset_learning_curves(config, dataset, args.force)
        else:
            subjects = _subjects(config, dataset, args.subject)
            LOGGER.info("%s: %d participant(s)", dataset, len(subjects))
        if args.command == "cache":
            cache_dataset(config, dataset, subjects, args.force)
        elif args.command == "benchmark":
            run_dataset_benchmark(config, dataset, subjects, args.force)
        elif args.command == "wsaiec":
            run_dataset_wsaiec(config, dataset, subjects, args.force)


def _run_ablations(args: argparse.Namespace, config: ExperimentConfig) -> None:
    from wsaiec_eeg.evaluation.ablation import (
        ABLATION_DATASETS,
        aggregate_ablations,
        run_dataset_ablations,
    )

    write_manifest(config, args.command)
    datasets = list(args.dataset) if args.dataset else list(ABLATION_DATASETS)
    for dataset in datasets:
        subjects = _subjects(config, dataset, args.subject)
        LOGGER.info("Table 9 ablations for %s: %d participant(s)", dataset, len(subjects))
        run_dataset_ablations(config, dataset, subjects, args.force)
    if args.dataset is None and args.subject is None:
        print(aggregate_ablations(config))


def _run_all(args: argparse.Namespace, config: ExperimentConfig) -> None:
    from wsaiec_eeg.data.cache import cache_dataset
    from wsaiec_eeg.evaluation.ablation import (
        ABLATION_DATASETS,
        run_dataset_ablations,
    )
    from wsaiec_eeg.evaluation.aggregate import aggregate_all, aggregate_pre_ensemble
    from wsaiec_eeg.evaluation.base_benchmark import (
        run_dataset_benchmark,
        run_dataset_learning_curves,
    )
    from wsaiec_eeg.evaluation.publication import verify_publication
    from wsaiec_eeg.evaluation.stacking import run_dataset_wsaiec
    from wsaiec_eeg.plotting.figures import make_all_figures

    write_manifest(config, "all")
    for dataset in config.datasets:
        subjects = list(config.datasets[dataset]["subjects"])
        LOGGER.info("Caching %s", dataset)
        cache_dataset(config, dataset, subjects, args.force)
        LOGGER.info("Benchmarking %s", dataset)
        run_dataset_benchmark(config, dataset, subjects, args.force)
        LOGGER.info("Training-volume analysis for %s", dataset)
        run_dataset_learning_curves(config, dataset, args.force)
    aggregate_pre_ensemble(config)
    for dataset in config.datasets:
        subjects = list(config.datasets[dataset]["subjects"])
        LOGGER.info("WS-AIEC evaluation for %s", dataset)
        run_dataset_wsaiec(config, dataset, subjects, args.force)
    for dataset in ABLATION_DATASETS:
        subjects = list(config.datasets[dataset]["subjects"])
        LOGGER.info("Table 9 ablation evaluation for %s", dataset)
        run_dataset_ablations(config, dataset, subjects, args.force)
    aggregate_all(config, include_learning_curves=True)
    make_all_figures(config, source="generated")
    verify_publication(config, compare_generated=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config(args.config)

    if args.command in {"cache", "benchmark", "learning-curves", "wsaiec"}:
        _run_selected(args, config)
    elif args.command == "ablations":
        _run_ablations(args, config)
    elif args.command == "aggregate":
        from wsaiec_eeg.evaluation.aggregate import aggregate_all

        paths = aggregate_all(config, include_learning_curves=not args.without_learning_curves)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    elif args.command == "figures":
        from wsaiec_eeg.plotting.figures import make_all_figures

        print("\n".join(str(path) for path in make_all_figures(config, args.source)))
    elif args.command == "verify-paper":
        from wsaiec_eeg.evaluation.publication import verify_publication

        report = verify_publication(config, not args.no_generated_comparison)
        print(json.dumps(report, indent=2))
    elif args.command == "smoke":
        from wsaiec_eeg.synthetic import run_smoke

        print(run_smoke(config, args.output))
    elif args.command == "all":
        _run_all(args, config)
    else:
        parser.error(f"Unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
