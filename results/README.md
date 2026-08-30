# Results

- `publication/source/` contains exact article values transcribed from Tables
  1-10. They remain tracked in Git and are protected by a SHA-256 manifest.
- `publication/source/related_works.csv` preserves Table 10 as literature
  context and is not compared with an executable rerun.
- `publication/generated/<config-hash>/` receives aggregate rerun tables,
  comparisons, consistency reports, and figures. Generated content is ignored
  by Git.
- `runs/<config-hash>/` receives participant-level metrics and predictions,
  dataset-level learning curves and alpha searches, OOF meta-features, dynamic
  weight histories, eleven-scenario ablation outputs, and provenance. Generated
  content is ignored by Git.

Run `wsaiec-eeg verify-paper --config configs/paper.yaml` to verify the tracked
reference tables without downloading or training. Run `scripts/run_all.py` to
produce a complete generated result bundle.
