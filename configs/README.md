# Configuration

`paper.yaml` is the frozen, executable interpretation of the experiment
described in the article. It contains every user-facing scientific setting;
algorithm mechanics, deterministic seed offsets, and pinned-library defaults
remain in the referenced source modules.

The file also freezes the executable definitions that the article does not
numerically specify: CSP component count, classifier parameters, dynamic epoch,
Bayesian alpha search, stacking construction, and the explicit Table 5/Table 7
paper-replay modes used alongside computed audits.

Experiment folders, filtered caches, and generated publication bundles are
keyed by the SHA-256 hash of this file. Editing it creates a new protocol
namespace. Source-code or dependency changes with unchanged YAML require
`--force` for affected stages.
