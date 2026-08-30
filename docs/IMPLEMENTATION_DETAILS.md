# Implementation details

## Ordered data path

For each participant, the loader preserves MOABB session order, run order, and
event order. EEG channels are filtered on the continuous recording before
half-open epoch extraction. Each cache stores the epoch tensor, numeric class
labels, class names, channel names, sampling rate, session, run, and original
event index.

## Filtering

MNE applies a zero-phase FIR filter with a Hamming window and firwin design.
The passband is 7-30 Hz. Both transition bands are 2 Hz, placing the reported
stopband edges at 5 and 32 Hz.

## CSP

CSP and feature standardization are trained only on the current fold's
training samples. Four components are requested and limited to the number of
available EEG channels. The transform returns logarithmic average power.

## Direct classifier score

Participant outer-test metrics are averaged by dataset and classifier. The
global table then averages each metric across the six datasets. Accuracy,
weighted precision, weighted recall, weighted F1, hard-prediction ROC-AUC, and
kappa receive equal weight.

## Learning-curve score

For every dataset and training fraction, the pipeline takes the corresponding
chronological prefix from each participant's outer-development block and
concatenates those prefixes in participant order. A fresh eight-fold
TimeSeriesSplit then produces fold-local CSP, training accuracy, and validation
accuracy. `training_volume` is the pooled prefix count; `fold_fit_count` records
the actual number fitted in a fold.

This cross-subject construction is an explicit completion. Figure 3 and the
Table 4 AUC-CV scales require a pooled example-count axis, but the article does
not specify how subject blocks are combined. A literal eight-fold TSCV on every
subject's 10% prefix is infeasible for AlexMI because that prefix has five
trials. The repository therefore records the configured participant order and
all effective counts instead of presenting the completion as a reported step.

- AUC-CV is the trapezoidal integral of mean validation accuracy over pooled
  `training_volume`.
- Convergence rate is the absolute final difference between mean training and
  validation accuracy.
- Performance stability is the mean validation-score standard deviation across
  training sizes.

AUC-CV is converted to a descending min-max cost. Convergence rate and
stability use ascending min-max costs. Their equal mean produces the learning
score.

## Ranking and clustering

Direct-performance rank and computed Equation (5) learning rank receive equal
weight. Ties are resolved by learning rank and classifier name, matching the
printed Table 6 order. The printed Table 5 order is also retained as a separate
paper replay because it is not derivable from the printed normalized Table 4
values.

The two computed rank columns are standardized before Ward hierarchical
clustering with Euclidean distance. Its assignments, linkage, and elbow curve
are retained as a computed audit. They do not reproduce the published Table 7
groups. In `paper_reported` selection mode, the ensemble therefore uses the six
Table 7 groups and winners frozen in `configs/paper.yaml`.

## Dynamic weighting and stacking

SVM is the highest-ranked cluster winner and becomes the linear meta-classifier.
NB, PC, SVM-RBF, GB, and LDA are the base classifiers.

For accuracy a_i and alpha greater than zero, each base weight is:

~~~text
w_i = exp(alpha * a_i) / sum_j exp(alpha * a_j)
~~~

One alpha per dataset is selected with Gaussian-process upper-confidence
optimization over mean participant validation accuracy. Inside each outer
time-series fold, base accuracy and Equation (9) weights are learned from an
earlier inner chronological validation block; those weights transform the
later outer-fold predictions. Each weighted hard prediction forms one
meta-feature. The chronological dataset-alpha validation block is then included
in final meta fitting. Final base estimators are refitted on the full
development set and evaluated on the untouched chronological holdout.

The eleven ablations use the same construction. Dynamic scenarios 1-10 select
one alpha per dataset and scenario from mean participant validation accuracy;
scenario 11 applies the five printed Table 7 base weights without
renormalization after meta SVM is removed.
