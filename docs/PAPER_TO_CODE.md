# Paper-to-code map

| Article component | Implementation |
|---|---|
| Section II-A and Table 1 | configs/paper.yaml, wsaiec_eeg.data.registry, wsaiec_eeg.data.cache |
| Section II-B | wsaiec_eeg.models.classifiers |
| Section II-C.1 | wsaiec_eeg.data.preprocessing.iter_subject_trials |
| Section II-C.2 | wsaiec_eeg.features.csp.CSPFeatureTransformer |
| Section II-C.3 | wsaiec_eeg.data.splits |
| Section II-D | wsaiec_eeg.evaluation.base_benchmark |
| Equation 1 | wsaiec_eeg.evaluation.aggregate.aggregate_base |
| Equations 2-5 | wsaiec_eeg.evaluation.aggregate._learning_summary and aggregate_learning_curves |
| Equation 6 | wsaiec_eeg.evaluation.aggregate.aggregate_ranking |
| Equation 7 | wsaiec_eeg.evaluation.aggregate.aggregate_ranking |
| Equation 8 | wsaiec_eeg.evaluation.aggregate.aggregate_ranking |
| Equation 9 | wsaiec_eeg.models.wsaiec.dynamic_weights |
| Equation 10 | wsaiec_eeg.models.wsaiec.weighted_meta_features |
| Equation 11 | wsaiec_eeg.evaluation.stacking.run_dataset_wsaiec |
| Algorithm 1 | wsaiec_eeg.cli._run_all and scripts/run_all.py |
| Section III-G and Table 9 | wsaiec_eeg.evaluation.ablation and scripts/run_ablations.py |
| Figure 2 | wsaiec_eeg.plotting.figures.plot_base_accuracy |
| Figure 3 | wsaiec_eeg.plotting.figures.plot_learning_curves |
| Figure 4 | wsaiec_eeg.plotting.figures.plot_clustering |
| Figure 5 | wsaiec_eeg.plotting.figures.plot_wsaiec_comparison |
| Figure 6 | wsaiec_eeg.plotting.figures.plot_ablation |
| Table 2 | results/publication/source/base_classifier_metrics.csv |
| Table 3 | results/publication/source/aggregated_performance.csv |
| Table 4 | results/publication/source/learning_curve_metrics.csv |
| Table 5 | results/publication/source/learning_ranking.csv |
| Table 6 | results/publication/source/overall_ranking.csv |
| Table 7 | results/publication/source/clusters.csv |
| Table 8 | results/publication/source/wsaiec_metrics.csv |
| Table 9 reference | results/publication/source/ablation.csv |
| Table 9 rerun | results/publication/generated/<run-key>/ablation.csv |
| Table 10 literature context | results/publication/source/related_works.csv |
| Publication audit | wsaiec_eeg.evaluation.publication.verify_publication |

Every estimator, CSP transform, scaler, weight optimization, and
meta-classifier fit is restricted to the relevant development partition. Inner
chronological validation labels define OOF weights before the later OOF rows.
The final 20% outer-test partition is used once for evaluation.
