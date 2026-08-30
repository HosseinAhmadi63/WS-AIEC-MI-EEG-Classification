"""Generate paper-equivalent Figures 2 through 6 from tabular results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wsaiec_eeg.config import ExperimentConfig
from wsaiec_eeg.constants import CLASSIFIER_ORDER, DATASET_ORDER
from wsaiec_eeg.utils.io import ensure_directory

TOP_FIVE_CLASSIFIERS = ("SVM", "RC", "SVM_rbf", "LR", "NB")
TABLE_8_ORDER = ("WS_AIEC", *TOP_FIVE_CLASSIFIERS)


def _source_directory(config: ExperimentConfig, source: str) -> Path:
    if source == "paper":
        return config.publication_source
    if source == "generated":
        return config.publication_generated
    raise ValueError("source must be 'paper' or 'generated'")


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _classifier_label(classifier: str) -> str:
    return {"SVM_rbf": "SVM-rbf", "WS_AIEC": "WS-AIEC"}.get(classifier, classifier)


def _dataset_label(dataset: str) -> str:
    if dataset.startswith("BNCI"):
        prefix, suffix = dataset.rsplit("_", maxsplit=1)
        return f"{prefix}-{suffix}"
    return dataset


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")


def plot_base_accuracy(base: pd.DataFrame, output: Path) -> Path:
    """Recreate Figure 2: classifier accuracy for each of the six datasets."""

    _require_columns(base, {"dataset", "classifier", "accuracy"}, "Base accuracy table")
    figure, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER, strict=True):
        group = base.loc[base["dataset"] == dataset, ["classifier", "accuracy"]]
        if set(group["classifier"]) != set(CLASSIFIER_ORDER):
            raise ValueError(f"Base accuracy table is incomplete for {dataset}")
        group = group.set_index("classifier").loc[list(CLASSIFIER_ORDER)].reset_index()
        positions = np.arange(len(group))
        axis.bar(
            positions,
            group["accuracy"],
            color="#77b7c5",
            edgecolor="#1f4e5f",
            linewidth=0.6,
        )
        axis.axhline(
            float(group["accuracy"].mean()),
            color="#c53030",
            linestyle="--",
            linewidth=1.4,
            label="Mean accuracy",
        )
        axis.set_title(_dataset_label(dataset), fontweight="bold")
        axis.set_xticks(positions, [_classifier_label(value) for value in group["classifier"]])
        axis.tick_params(axis="x", rotation=55, labelsize=7)
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("Accuracy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle(
        "Figure 2. Comparative accuracy of 16 classifiers across six datasets",
        fontsize=15,
        fontweight="bold",
    )
    return _save(figure, output / "figure_2_classifier_accuracy.png")


def plot_learning_curves(curves: pd.DataFrame, output: Path) -> Path:
    """Recreate Figure 3 for BNCI2014-002 as a 16-panel learning-curve grid."""

    required = {
        "dataset",
        "classifier",
        "training_fraction",
        "training_volume",
        "train_score",
        "validation_score",
    }
    _require_columns(curves, required, "Learning-curve table")
    dataset = "BNCI2014_002"
    selected = curves.loc[curves["dataset"] == dataset].copy()
    if set(selected["classifier"]) != set(CLASSIFIER_ORDER):
        raise ValueError(f"Learning-curve table is incomplete for {dataset}")
    figure, axes = plt.subplots(
        4,
        4,
        figsize=(17, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, classifier in zip(axes.flat, CLASSIFIER_ORDER, strict=True):
        group = selected.loc[selected["classifier"] == classifier].sort_values(
            "training_fraction"
        )
        x = group["training_volume"].to_numpy(dtype=float)
        train = group["train_score"].to_numpy(dtype=float)
        validation = group["validation_score"].to_numpy(dtype=float)
        axis.plot(x, train, color="#1f77b4", marker="o", markersize=3, label="Training")
        axis.plot(
            x,
            validation,
            color="#d62728",
            marker="s",
            markersize=3,
            label="Cross-validation",
        )
        if "train_score_std" in group:
            spread = group["train_score_std"].to_numpy(dtype=float)
            axis.fill_between(x, train - spread, train + spread, color="#1f77b4", alpha=0.12)
        if "validation_score_std" in group:
            spread = group["validation_score_std"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                validation - spread,
                validation + spread,
                color="#d62728",
                alpha=0.12,
            )
        axis.set_title(_classifier_label(classifier), fontsize=10, fontweight="bold")
        axis.set_ylim(0.0, 1.03)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Training examples", fontsize=8)
        axis.set_ylabel("Score", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle(
        "Figure 3. Learning curves for BNCI2014-002",
        fontsize=15,
        fontweight="bold",
    )
    return _save(figure, output / "figure_3_learning_curves_bnci2014_002.png")


def plot_clustering(
    performance: pd.DataFrame,
    learning: pd.DataFrame,
    output: Path,
    random_seed: int,
    standardize: bool,
) -> Path:
    """Recreate Figure 4: hierarchical dendrogram and six-cluster elbow analysis."""

    _require_columns(performance, {"classifier", "rank"}, "Performance ranking")
    _require_columns(learning, {"classifier", "rank"}, "Learning ranking")
    ranking = performance[["classifier", "rank"]].rename(
        columns={"rank": "performance_rank"}
    ).merge(
        learning[["classifier", "rank"]].rename(columns={"rank": "learning_rank"}),
        on="classifier",
        validate="one_to_one",
    )
    if set(ranking["classifier"]) != set(CLASSIFIER_ORDER):
        raise ValueError("Clustering rankings do not cover all 16 classifiers")
    ranking = ranking.set_index("classifier").loc[list(CLASSIFIER_ORDER)].reset_index()
    values = ranking[["performance_rank", "learning_rank"]].to_numpy(dtype=float)
    if standardize:
        from sklearn.preprocessing import StandardScaler

        values = StandardScaler().fit_transform(values)

    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import KMeans

    linkage_matrix = linkage(values, method="ward", metric="euclidean")
    figure, (dendrogram_axis, elbow_axis) = plt.subplots(
        1, 2, figsize=(17, 7), constrained_layout=True
    )
    dendrogram(
        linkage_matrix,
        labels=[_classifier_label(value) for value in ranking["classifier"]],
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=None,
        ax=dendrogram_axis,
    )
    dendrogram_axis.set_title("Hierarchical clustering", fontweight="bold")
    dendrogram_axis.set_xlabel("Classifier")
    dendrogram_axis.set_ylabel("Ward linkage distance")
    dendrogram_axis.grid(axis="y", alpha=0.2)

    cluster_numbers = np.arange(1, len(CLASSIFIER_ORDER), dtype=int)
    inertia = [
        float(
            KMeans(n_clusters=int(number), random_state=random_seed, n_init=50)
            .fit(values)
            .inertia_
        )
        for number in cluster_numbers
    ]
    elbow_axis.plot(cluster_numbers, inertia, color="#1f77b4", marker="o")
    elbow_axis.axvline(6, color="#c53030", linestyle="--", label="Selected: 6 clusters")
    elbow_axis.set_xticks(cluster_numbers)
    elbow_axis.set_title("Elbow method", fontweight="bold")
    elbow_axis.set_xlabel("Number of clusters")
    elbow_axis.set_ylabel("Within-cluster sum of squares")
    elbow_axis.grid(alpha=0.2)
    elbow_axis.legend(frameon=False)
    figure.suptitle(
        "Figure 4. Classifier cluster analysis",
        fontsize=15,
        fontweight="bold",
    )
    return _save(figure, output / "figure_4_classifier_clustering.png")


def _table_8_frame(base: pd.DataFrame, wsaiec: pd.DataFrame) -> pd.DataFrame:
    if "classifier" in wsaiec:
        table = wsaiec.loc[wsaiec["classifier"].isin(TABLE_8_ORDER)].copy()
    else:
        _require_columns(wsaiec, {"dataset", "accuracy"}, "WS-AIEC table")
        ensemble = wsaiec.copy()
        ensemble["classifier"] = "WS_AIEC"
        individual = base.loc[base["classifier"].isin(TOP_FIVE_CLASSIFIERS)].copy()
        table = pd.concat([ensemble, individual], ignore_index=True, sort=False)
    expected = {
        (dataset, classifier)
        for dataset in DATASET_ORDER
        for classifier in TABLE_8_ORDER
    }
    actual = set(map(tuple, table[["dataset", "classifier"]].itertuples(index=False, name=None)))
    if table.duplicated(["dataset", "classifier"]).any() or actual != expected:
        raise ValueError("WS-AIEC comparison table does not contain the paper top-five grid")
    return table


def plot_wsaiec_comparison(base: pd.DataFrame, wsaiec: pd.DataFrame, output: Path) -> Path:
    """Recreate Figure 5: WS-AIEC versus the paper's top five classifiers."""

    _require_columns(base, {"dataset", "classifier", "accuracy"}, "Base accuracy table")
    table = _table_8_frame(base, wsaiec)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASET_ORDER, strict=True):
        group = (
            table.loc[table["dataset"] == dataset, ["classifier", "accuracy"]]
            .set_index("classifier")
            .loc[list(TABLE_8_ORDER)]
            .reset_index()
        )
        positions = np.arange(len(group))
        colors = ["#dd6b20", *(["#77b7c5"] * len(TOP_FIVE_CLASSIFIERS))]
        bars = axis.bar(positions, group["accuracy"], color=colors, edgecolor="#2d3748")
        axis.set_xticks(positions, [_classifier_label(value) for value in group["classifier"]])
        axis.tick_params(axis="x", rotation=40, labelsize=8)
        axis.set_title(_dataset_label(dataset), fontweight="bold")
        axis.set_ylim(0.0, 1.04)
        axis.set_ylabel("Accuracy")
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, group["accuracy"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.012,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    figure.suptitle(
        "Figure 5. WS-AIEC versus the top five individual classifiers",
        fontsize=15,
        fontweight="bold",
    )
    return _save(figure, output / "figure_5_wsaiec_comparison.png")


def _benchmark_accuracies(wsaiec: pd.DataFrame) -> tuple[float, float] | None:
    frame = wsaiec
    if "classifier" in frame:
        frame = frame.loc[frame["classifier"] == "WS_AIEC"]
    indexed = frame.set_index("dataset")
    required = {"BNCI2014_002", "Zhou2016"}
    if "accuracy" not in indexed or not required.issubset(indexed.index):
        return None
    return (
        float(indexed.loc["BNCI2014_002", "accuracy"]),
        float(indexed.loc["Zhou2016", "accuracy"]),
    )


def plot_ablation(
    ablation: pd.DataFrame,
    output: Path,
    benchmarks: tuple[float, float] | None,
) -> Path:
    """Recreate Figure 6: grouped accuracies for the eleven ablation scenarios."""

    required = {
        "scenario_id",
        "scenario",
        "accuracy_bnci2014_002",
        "accuracy_zhou2016",
    }
    _require_columns(ablation, required, "Ablation table")
    ordered = ablation.sort_values("scenario_id")
    if ordered["scenario_id"].astype(int).tolist() != list(range(1, 12)):
        raise ValueError("Ablation table must contain scenarios 1 through 11")
    positions = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(17, 7), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        ordered["accuracy_bnci2014_002"],
        width,
        label="BNCI2014-002",
        color="#4c78a8",
    )
    axis.bar(
        positions + width / 2,
        ordered["accuracy_zhou2016"],
        width,
        label="Zhou2016",
        color="#f28e2b",
    )
    if benchmarks is not None:
        axis.axhline(
            benchmarks[0],
            color="#2f5597",
            linestyle="--",
            linewidth=1.2,
            label="BNCI2014-002 benchmark",
        )
        axis.axhline(
            benchmarks[1],
            color="#b45f06",
            linestyle=":",
            linewidth=1.5,
            label="Zhou2016 benchmark",
        )
    axis.set_xticks(positions, [f"S{int(value)}" for value in ordered["scenario_id"]])
    axis.set_xlabel("Ablation scenario")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.75, 1.01)
    axis.set_title("Figure 6. Accuracy by ablation scenario", fontweight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    return _save(figure, output / "figure_6_ablation.png")


def make_all_figures(config: ExperimentConfig, source: str = "paper") -> list[Path]:
    """Create every paper figure supported by the selected result directory."""

    directory = _source_directory(config, source)
    output = ensure_directory(config.publication_generated / "figures" / source)
    base = pd.read_csv(directory / "base_classifier_metrics.csv")
    wsaiec = pd.read_csv(directory / "wsaiec_metrics.csv")
    performance = pd.read_csv(directory / "aggregated_performance.csv")
    learning = pd.read_csv(directory / "learning_ranking.csv")
    paths = [plot_base_accuracy(base, output)]

    learning_curves_path = directory / "learning_curves.csv"
    if learning_curves_path.exists():
        paths.append(plot_learning_curves(pd.read_csv(learning_curves_path), output))

    paths.append(
        plot_clustering(
            performance,
            learning,
            output,
            random_seed=int(config.project["random_seed"]),
            standardize=bool(config.section("ranking")["standardize_rank_features"]),
        )
    )
    paths.append(plot_wsaiec_comparison(base, wsaiec, output))

    ablation_path = directory / "ablation.csv"
    if not ablation_path.exists():
        ablation_path = config.publication_source / "ablation.csv"
    paths.append(
        plot_ablation(
            pd.read_csv(ablation_path),
            output,
            _benchmark_accuracies(wsaiec),
        )
    )
    return paths
