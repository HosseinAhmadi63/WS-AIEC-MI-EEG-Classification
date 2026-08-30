# PyCharm configuration

## Interpreter

1. Open the repository root.
2. Open Settings, Project, Python Interpreter.
3. Add a local virtual environment.
4. Select Python 3.11.
5. Set the environment location to .venv.
6. Run these commands in the PyCharm terminal:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
~~~

## Complete pipeline

Create a Python run configuration with:

| Field | Value |
|---|---|
| Script path | scripts/run_all.py |
| Parameters | --config configs/paper.yaml --verbose |
| Working directory | Repository root |
| Interpreter | .venv |

## Individual stages

| Name | Script | Parameters |
|---|---|---|
| 01 Cache data | scripts/cache_data.py | --config configs/paper.yaml --verbose |
| 02 Base classifiers | scripts/run_base_classifiers.py | --config configs/paper.yaml --verbose |
| 03 Learning curves | scripts/run_learning_curves.py | --config configs/paper.yaml --verbose |
| 04 WS-AIEC | scripts/run_wsaiec.py | --config configs/paper.yaml --verbose |
| 05 Ablations | scripts/run_ablations.py | --config configs/paper.yaml --verbose |
| 06 Paper analysis | scripts/reproduce_paper_analysis.py | --config configs/paper.yaml --verbose |
| 07 Figures | scripts/make_figures.py | --config configs/paper.yaml --source generated |
| Verify installation | scripts/verify_installation.py | --config configs/paper.yaml |

Add --dataset BNCI2014_002 --subject 1 to stages 01, 02, 04, and 05 for a
diagnostic single-participant run. Stage 03 always uses all participants. Full
paper-mode WS-AIEC alpha tuning and Table 9 aggregation require all configured
participants.

## Tests

Create a pytest configuration with:

| Field | Value |
|---|---|
| Target | tests |
| Working directory | Repository root |
| Additional arguments | -ra |
