# Result schema

## Participant base metrics

Path: `results/runs/<run-key>/<dataset>/subject_<id>/base_metrics.csv`

Primary row key: `(dataset, subject, classifier, scope, fold, training_fraction)`.

`scope=time_series_validation` produces eight rows per classifier. `scope=outer_test`
produces one final row. Metric columns are `accuracy`, `precision`, `recall`,
`f1`, `auc_roc`, `kappa`, and their arithmetic mean `score`.

## Participant base predictions

Path: `.../base_predictions.csv`

Primary row key: `(sample_key, classifier)`. `sample_key` contains dataset,
subject, and source-order trial index. `probabilities` is a JSON array aligned
with the class order in `configs/paper.yaml`.

## Learning curves

Path: `results/runs/<run-key>/<dataset>/learning_curve.csv`

Primary row key: `(dataset, classifier, training_fraction, fold)`. Each of the
ten pooled training volumes contains eight expanding-window folds.
`training_volume` is the sum of chronological participant prefixes and is the
Figure 3/Equation (2) x-coordinate. `fold_fit_count` and
`fold_validation_count` record the actual pooled TimeSeriesSplit row counts.
`subject_count` records the participant cohort. The sibling
`learning_curve_metadata.json` records the exact requested participants and
full pooled volume.

## WS-AIEC metrics

Path: `.../wsaiec_metrics.csv`

One row per participant. `n_meta_train` is the total number of rows fitted by
the final meta-classifier: true out-of-fold rows plus the chronological
alpha-validation block included after alpha selection. The initial
expanding-window training block is intentionally excluded.
`validation_accuracies` and `weights` are sorted JSON mappings.

## WS-AIEC predictions

Path: `.../wsaiec_predictions.csv`

Primary row key: `sample_key`. Each row contains the untouched outer-test
label, final prediction, aligned class probabilities, the five base hard
predictions, and the five weighted Equation (10) meta-features.

## WS-AIEC OOF data

Path: `.../wsaiec_oof.csv`

Primary row key: `(dataset, subject, stage, development_index, classifier)`.
`stage=time_series_oof` contains expanding-window validation predictions;
`stage=alpha_validation` contains the chronological development holdout. The
table records hard predictions, validation accuracies, dynamic weights, and
weighted predictions.

## Adaptive-weight histories

- Dataset-level `wsaiec_alpha_history.csv`: one Bayesian-optimization
  evaluation per row, including shared alpha, mean and participant validation
  accuracies, cohort mode, and the selected evaluation.
- Participant `wsaiec_alpha_history.csv`: the same dataset search with the
  current participant's validation accuracy.
- `wsaiec_weight_history.csv`: one classifier row for each time-series fold and
  the final alpha-validation stage, containing validation accuracy and the
  Equation (9) dynamic weight.

## Ablations

- Dataset-level `ablation_alpha_history.csv`: one row per
  dataset/scenario/optimizer evaluation. Scenarios 1-10 use
  `alpha_scope=dataset_scenario`; static scenario 11 uses
  `alpha_scope=not_applicable_static`.
- Participant `ablation_metrics.csv`: one final outer-test row for each of the
  eleven Table 9 scenarios, including classifier selection, cohort identity,
  alpha, final weights, and all six metrics.
- Participant `ablation_predictions.csv`: one outer-test prediction per
  scenario and sample, with base hard predictions and weighted meta-features.
- Participant `ablation_alpha_history.csv`: the shared dataset/scenario search
  with the current participant's validation accuracy.
- Participant `ablation_weight_history.csv`: time-series and final-validation
  weights per scenario and base classifier.
- Participant `ablation_protocol.json`: the frozen Table 9 classifier sets,
  weighting modes, ranks, and static weights.

## Publication aggregates

All generated paths are below
`results/publication/generated/<run-key>/`.

- `base_classifier_metrics.csv`: mean of participant outer-test metrics by
  `(dataset, classifier)`, composite score, dataset rank.
- `aggregated_performance.csv`: six-dataset mean metrics, direct-performance
  score, and rank.
- `learning_curves.csv`: mean train/cross-validation curves by dataset,
  classifier, and training fraction.
- `learning_curve_summary.csv`: AUC-CV, convergence rate, performance
  stability, normalized costs, score, and dataset rank.
- `learning_ranking.csv` and `computed_learning_ranking.csv`: the computed
  six-dataset Equation (5) learning score and rank.
- `paper_learning_ranking.csv`: explicitly named Table 5 replay used only when
  `paper_reported` replay mode is configured.
- `overall_ranking.csv`: equal mean of direct-performance and learning ranks,
  final rank, and inverse-rank static weight.
- `cluster_assignments.csv`, `cluster_linkage.csv`, `cluster_elbow.csv`: the
  computed Ward audit.
- `cluster_selection.csv`: explicitly configured Table 7 group/winner replay
  in `paper_reported` mode.
- `wsaiec_metrics.csv`: mean participant WS-AIEC metrics by dataset.
- `ablation.csv`: generated Table 9 accuracies for BNCI2014-002 and Zhou2016.
- `comparison_*.csv`: paper/generated values and absolute error per compared
  metric.
- `paper_consistency.json`: reference hashes, table invariants, implementation
  choices, and comparison summaries.
