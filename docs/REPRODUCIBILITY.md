# Reproducibility contract

## Determinism

- The experiment seed is 42.
- Estimator seeds are derived deterministically from dataset stage, participant,
  training fraction, fold, and classifier.
- Participant trials remain in source session, run, and event order.
- The outer test set is always the chronologically last 20%.
- Time-series validation uses eight expanding folds without shuffling.
- The configuration SHA-256 determines the result directory.

## Leakage controls

- FIR filtering has fixed coefficients and does not use labels.
- CSP and StandardScaler are fitted separately inside every training fold.
- Base-model out-of-fold predictions are generated only from earlier training
  samples, and their weights are calculated from an earlier inner validation
  block rather than their own labels.
- One alpha per dataset is tuned from the final 20% development blocks and the
  mean participant validation accuracy.
- The meta-classifier receives no outer-test labels or fitted test features.
- Final base models and CSP are refitted on the complete development partition
  only after alpha selection.

## Paper references

The ten CSV files in results/publication/source are immutable transcriptions
of Tables 1-10. SOURCE_MANIFEST.json records their SHA-256 hashes, row counts,
and paper locations. The verify-paper command validates the manifest before
performing any generated comparison.

## Result identity

Each run, filtered cache, and generated publication bundle is isolated below:

~~~text
results/runs/<first-12-characters-of-config-sha256>/
data/cache/<first-12-characters-of-config-sha256>/
results/publication/generated/<first-12-characters-of-config-sha256>/
~~~

The run manifest records the full configuration, package versions, Python
version, platform, command, timestamp, and Git revision when available.
After changing source code or a pinned dependency without changing the YAML,
rerun affected stages with `--force`.

## Comparison policy

Generated metrics are compared with printed paper values by exact keys.
Differences are reported as absolute errors. Computed learning ranks and Ward
clusters remain separate from the explicitly named Table 5 and Table 7 paper
replays; generated values are never silently replaced by reference values.
