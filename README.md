# WS-AIEC MI-EEG Classification

This repository is the complete Python and PyCharm reproduction pipeline for:

> **Enhancing MI EEG Signal Classification With a Novel Weighted and Stacked Adaptive Integrated Ensemble Model: A Multi-Dataset Approach**  
> Hossein Ahmadi and Luca Mesin  
> *IEEE Access*, volume 12, pages 103626-103646, 2024  
> [DOI](https://doi.org/10.1109/ACCESS.2024.3434654) ·
> [IEEE document 10613776](https://ieeexplore.ieee.org/document/10613776)

The pipeline downloads all six public motor-imagery EEG datasets, applies the
paper's 7-30 Hz FIR preprocessing and Common Spatial Pattern feature extraction,
evaluates all 16 classifiers, produces the eight-fold time-series learning
curves, recreates both ranking systems, performs six-cluster hierarchical
selection, constructs the Weighted and Stacked Adaptive Integrated Ensemble
Classifier, runs all eleven ablation scenarios, calculates every reported
metric, generates equivalents of Figures 2-6, validates the exact values
transcribed from Tables 1-10, and writes key-aligned comparisons wherever a
completed run has an executable counterpart.

If you use this repository, its code, or its results, cite the article above.
Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).

## Paper-aligned frozen pipeline

| Stage | Frozen implementation |
|---|---|
| Datasets | BNCI2014-001, BNCI2014-002, BNCI2014-004, BNCI2015-001, Zhou2016, AlexMI |
| Participants | 9, 14, 9, 12, 4, and 8 |
| Classes | 4, 2, 2, 2, 3, and 3 |
| Trial windows | 4.0 s, 5.0 s, 4.5 s, 5.0 s, 5.0 s, and 3.0 s |
| Preprocessing | Reported 7-30 Hz FIR/Hamming filter; frozen zero-phase MNE implementation with 2 Hz transitions and EEG channels only |
| Features | Four-component CSP, limited by channel count, followed by fold-local standardization |
| Outer evaluation | First 80% of each participant's ordered trials for development and final 20% for testing |
| Validation | Eight expanding-window TimeSeriesSplit folds |
| Learning volume | 10%, 20%, ..., 100% chronological participant prefixes, pooled in participant order at dataset scope before eight-fold validation |
| Base models | LDA, LR, PC, SGD, RC, linear SVM, SVM-RBF, KN, NB, DT, RF, ET, GB, AB, QDA, MLP |
| Direct score | Equal mean of accuracy, weighted precision, weighted recall, weighted F1, hard-prediction ROC-AUC, and Cohen's kappa |
| Learning score | Equal mean of normalized AUC-CV cost, convergence rate, and performance stability |
| Overall rank | Equal mean of direct-performance rank and learning-curve rank |
| Clustering | Computed audit: standardized two-rank matrix, Euclidean distance, Ward linkage, six clusters; ensemble replay: the six groups printed in Table 7 |
| Paper cluster winners | NB, PC, SVM-RBF, GB, LDA, SVM |
| Meta-classifier | Linear SVM |
| Base classifiers | NB, PC, SVM-RBF, GB, LDA |
| Dynamic weights | Softmax of one dataset-level Bayesian-tuned alpha multiplied by validation accuracy |
| Stacking input | One dynamically weighted hard prediction per base classifier |

## Repository structure

~~~text
configs/paper.yaml                   Complete frozen experiment definition
scripts/cache_data.py                Download, filter, epoch, validate, and cache data
scripts/run_base_classifiers.py      Run all 16 participant-wise classifiers
scripts/run_learning_curves.py       Run the 10%-100% TSCV learning experiment
scripts/run_wsaiec.py                Tune dynamic weights and evaluate WS-AIEC
scripts/run_ablations.py             Run all eleven Table 9 ablation scenarios
scripts/reproduce_paper_analysis.py  Aggregate, verify, and compare all paper tables
scripts/make_figures.py              Generate Figures 2-6
scripts/run_all.py                   Run the complete ordered pipeline
src/wsaiec_eeg/                      Installable implementation
tests/                               Scientific, configuration, and CLI checks
results/publication/source/          Exact values transcribed from Tables 1-10
results/publication/source/related_works.csv  Table 10 literature context
results/publication/generated/<run-key>/  Aggregated tables, comparisons, and figures
results/runs/<run-key>/              Participant-level metrics and predictions
~~~

The detailed article-to-code map is in
[docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md).

## Installation

Python 3.11 is the reference interpreter.

~~~bash
git clone https://github.com/HosseinAhmadi63/WS-AIEC-MI-EEG-Classification.git
cd WS-AIEC-MI-EEG-Classification
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
~~~

On Windows:

~~~powershell
git clone https://github.com/HosseinAhmadi63/WS-AIEC-MI-EEG-Classification.git
cd WS-AIEC-MI-EEG-Classification
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
~~~

Verify the installation without downloading EEG data:

~~~bash
pytest
wsaiec-eeg verify-paper --config configs/paper.yaml
wsaiec-eeg smoke --config configs/paper.yaml --output results/smoke
~~~

## PyCharm

1. Open the repository root in PyCharm.
2. Select Python 3.11 and create the interpreter at .venv.
3. Open the PyCharm terminal and run the installation commands above.
4. Create a Python run configuration for scripts/run_all.py.
5. Set the working directory to the repository root.
6. Set parameters to --config configs/paper.yaml --verbose.
7. Run the configuration.

All stage-specific PyCharm configurations are listed in
[docs/PYCHARM.md](docs/PYCHARM.md).

## Complete reproduction

Run every stage:

~~~bash
python scripts/run_all.py --config configs/paper.yaml --verbose
~~~

The pipeline is resumable. EEG caches, run artifacts, and publication aggregates
are isolated by the same 12-character configuration hash. Use `--force` on a
stage after changing source code or pinned dependencies without changing the
configuration.

Run the stages separately:

~~~bash
python scripts/cache_data.py --config configs/paper.yaml --verbose
python scripts/run_base_classifiers.py --config configs/paper.yaml --verbose
python scripts/run_learning_curves.py --config configs/paper.yaml --verbose
python scripts/run_wsaiec.py --config configs/paper.yaml --verbose
python scripts/run_ablations.py --config configs/paper.yaml --verbose
python scripts/reproduce_paper_analysis.py --config configs/paper.yaml --verbose
python scripts/make_figures.py --config configs/paper.yaml --source generated
~~~

Add `--force` to a cache, benchmark, learning-curve, WS-AIEC, ablation, or full
pipeline command to recompute its completed artifacts.

Run one participant:

~~~bash
python scripts/cache_data.py --config configs/paper.yaml --dataset BNCI2014_002 --subject 1 --verbose
python scripts/run_base_classifiers.py --config configs/paper.yaml --dataset BNCI2014_002 --subject 1 --verbose
python scripts/run_wsaiec.py --config configs/paper.yaml --dataset BNCI2014_002 --subject 1 --verbose
python scripts/run_ablations.py --config configs/paper.yaml --dataset BNCI2014_002 --subject 1 --verbose
~~~

Learning curves are dataset-scoped and therefore always use every configured
participant. A participant-filtered WS-AIEC or ablation run is diagnostic mode;
paper-mode alpha tuning uses the complete dataset cohort.

## Outputs

~~~text
results/runs/<run-key>/<dataset>/subject_<id>/base_metrics.csv
results/runs/<run-key>/<dataset>/subject_<id>/base_predictions.csv
results/runs/<run-key>/<dataset>/learning_curve.csv
results/runs/<run-key>/<dataset>/learning_curve_metadata.json
results/runs/<run-key>/<dataset>/wsaiec_alpha_history.csv
results/runs/<run-key>/<dataset>/subject_<id>/wsaiec_metrics.csv
results/runs/<run-key>/<dataset>/subject_<id>/wsaiec_predictions.csv
results/runs/<run-key>/<dataset>/subject_<id>/wsaiec_oof.csv
results/runs/<run-key>/<dataset>/subject_<id>/wsaiec_alpha_history.csv
results/runs/<run-key>/<dataset>/subject_<id>/wsaiec_weight_history.csv
results/runs/<run-key>/<dataset>/subject_<id>/ablation_metrics.csv
results/runs/<run-key>/<dataset>/subject_<id>/ablation_predictions.csv
results/runs/<run-key>/<dataset>/subject_<id>/ablation_alpha_history.csv
results/runs/<run-key>/<dataset>/subject_<id>/ablation_weight_history.csv
results/runs/<run-key>/<dataset>/subject_<id>/ablation_protocol.json
results/runs/<run-key>/<dataset>/ablation_alpha_history.csv
results/publication/generated/<run-key>/base_classifier_metrics.csv
results/publication/generated/<run-key>/aggregated_performance.csv
results/publication/generated/<run-key>/learning_curves.csv
results/publication/generated/<run-key>/learning_curve_summary.csv
results/publication/generated/<run-key>/learning_ranking.csv
results/publication/generated/<run-key>/computed_learning_ranking.csv
results/publication/generated/<run-key>/paper_learning_ranking.csv
results/publication/generated/<run-key>/overall_ranking.csv
results/publication/generated/<run-key>/cluster_selection.csv
results/publication/generated/<run-key>/wsaiec_metrics.csv
results/publication/generated/<run-key>/ablation.csv
results/publication/generated/<run-key>/ablation_subject_metrics.csv
results/publication/generated/<run-key>/ablation_protocol.json
results/publication/generated/<run-key>/comparison_*.csv
results/publication/generated/<run-key>/paper_consistency.json
results/publication/generated/<run-key>/figures/<source>/*.png
~~~

## Data

No EEG recordings are committed. MOABB downloads the official files to
data/moabb/; filtered participant caches are written below
data/cache/<run-key>/. Both
directories are excluded from Git. The original dataset terms remain
controlling.

## Citation

~~~bibtex
@article{ahmadi2024wsaiec,
  author  = {Ahmadi, Hossein and Mesin, Luca},
  title   = {Enhancing MI EEG Signal Classification With a Novel Weighted and Stacked Adaptive Integrated Ensemble Model: A Multi-Dataset Approach},
  journal = {IEEE Access},
  year    = {2024},
  volume  = {12},
  pages   = {103626--103646},
  doi     = {10.1109/ACCESS.2024.3434654}
}
~~~

## License

The source code is released under the [MIT License](LICENSE). The article and
dataset licenses are separate from the software license.
