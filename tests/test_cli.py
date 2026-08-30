from __future__ import annotations

import pytest

from wsaiec_eeg.cli import build_parser, main


def test_cli_help_renders_without_loading_a_run(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "learning-curves" in capsys.readouterr().out


def test_ablation_cli_accepts_only_the_two_table_9_datasets() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["ablations", "--dataset", "BNCI2014_002", "--subject", "1"]
    )
    assert args.command == "ablations"
    assert args.dataset == ["BNCI2014_002"]
    with pytest.raises(SystemExit):
        parser.parse_args(["ablations", "--dataset", "AlexMI"])


def test_learning_curve_cli_rejects_subject_subsets(repository_root, monkeypatch) -> None:
    monkeypatch.chdir(repository_root)
    with pytest.raises(ValueError, match="require every configured subject"):
        main(
            [
                "learning-curves",
                "--config",
                "configs/paper.yaml",
                "--dataset",
                "AlexMI",
                "--subject",
                "1",
            ]
        )


def test_verify_cli(repository_root, monkeypatch) -> None:
    monkeypatch.chdir(repository_root)
    status = main(["verify-paper", "--config", "configs/paper.yaml", "--no-generated-comparison"])
    assert status == 0
