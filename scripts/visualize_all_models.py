#!/usr/bin/env python3
"""
Generate curated benchmark visualizations from the committed comparison CSV.

The public repository keeps a small set of figures that are useful in a
portfolio setting:
    - classification_metrics_comparison.png
    - performance_efficiency_tradeoff.png
    - comprehensive_table.png

These figures are derived directly from
results/comparison/model_comparison_results.csv.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT_DIR / "results" / "comparison" / "model_comparison_results.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "visualization"

DISPLAY_NAMES = {
    "CNN1D": "CNN1D",
    "LSTM": "LSTM",
    "RESNET1D": "ResNet1D",
    "HYBRID_CNN_LSTM": "Hybrid CNN-LSTM",
}

MODEL_COLORS = {
    "CNN1D": "#1b9e77",
    "LSTM": "#d95f02",
    "ResNet1D": "#7570b3",
    "Hybrid CNN-LSTM": "#e7298a",
}

METRIC_COLORS = {
    "Accuracy": "#1b9e77",
    "F1 Score": "#7570b3",
    "AUC Score": "#d95f02",
}


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10


def load_model_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    if "Training Time (s)" in df.columns:
        placeholder_mask = df["Training Time (s)"].eq(3000.0) & df["Model"].isin(["CNN1D", "LSTM"])
        df.loc[placeholder_mask, "Training Time (s)"] = np.nan

    df["Display Model"] = df["Model"].map(DISPLAY_NAMES).fillna(df["Model"])
    return df


def annotate_metric_points(ax: plt.Axes, plot_df: pd.DataFrame, metric_order: list[str]) -> None:
    x_positions = {label: idx for idx, label in enumerate(plot_df["Display Model"].drop_duplicates())}
    metric_offsets = {
        "Accuracy": -0.16,
        "F1 Score": 0.0,
        "AUC Score": 0.16,
    }

    for metric in metric_order:
        metric_rows = plot_df[plot_df["Metric"] == metric]
        for _, row in metric_rows.iterrows():
            ax.text(
                x_positions[row["Display Model"]] + metric_offsets[metric],
                row["Score"] + 0.0015,
                f"{row['Score']:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )


def create_classification_metrics_comparison(df: pd.DataFrame, save_dir: Path) -> None:
    metric_order = ["Accuracy", "F1 Score", "AUC Score"]
    plot_df = (
        df[["Display Model", *metric_order]]
        .sort_values("Accuracy", ascending=False)
        .melt(id_vars="Display Model", value_vars=metric_order, var_name="Metric", value_name="Score")
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.pointplot(
        data=plot_df,
        x="Display Model",
        y="Score",
        hue="Metric",
        hue_order=metric_order,
        palette=METRIC_COLORS,
        dodge=0.35,
        markers=["o", "s", "D"],
        linestyles="",
        scale=1.1,
        errorbar=None,
        ax=ax,
    )

    annotate_metric_points(ax, plot_df, metric_order)
    ax.set_title("Classification Metrics Across Baseline Models", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0.88, 0.995)
    ax.legend(title="Metric", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_path = save_dir / "classification_metrics_comparison.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def annotate_points(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series, labels: pd.Series) -> None:
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )


def create_performance_efficiency_tradeoff(df: pd.DataFrame, save_dir: Path) -> None:
    plot_df = df.sort_values("Accuracy", ascending=False).copy()
    colors = [MODEL_COLORS[name] for name in plot_df["Display Model"]]
    params_m = plot_df["Parameters"] / 1_000_000

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Performance-Efficiency Trade-offs", fontsize=17, fontweight="bold")

    axes[0].scatter(params_m, plot_df["Accuracy"], s=160, c=colors, edgecolors="black", linewidths=0.8)
    annotate_points(axes[0], params_m, plot_df["Accuracy"], plot_df["Display Model"])
    axes[0].set_title("Model Size vs Accuracy")
    axes[0].set_xlabel("Parameters (millions)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.88, 0.95)

    axes[1].scatter(
        plot_df["Inference Time (s)"],
        plot_df["Accuracy"],
        s=160,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
    )
    annotate_points(axes[1], plot_df["Inference Time (s)"], plot_df["Accuracy"], plot_df["Display Model"])
    axes[1].set_title("Inference Time vs Accuracy")
    axes[1].set_xlabel("Inference time (seconds)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.88, 0.95)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    save_path = save_dir / "performance_efficiency_tradeoff.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def create_comprehensive_table(df: pd.DataFrame, save_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("tight")
    ax.axis("off")

    table_df = df.copy()
    table_df["Display Model"] = table_df["Display Model"]
    table_df["Parameters (M)"] = (table_df["Parameters"] / 1_000_000).round(2)
    table_df["Inference Time (s)"] = table_df["Inference Time (s)"].round(3)
    table_df["Accuracy"] = table_df["Accuracy"].round(4)
    table_df["F1 Score"] = table_df["F1 Score"].round(4)
    table_df["AUC Score"] = table_df["AUC Score"].round(4)

    display_columns = [
        "Display Model",
        "Accuracy",
        "F1 Score",
        "AUC Score",
        "Parameters (M)",
        "Inference Time (s)",
    ]
    table_df = table_df[display_columns].rename(columns={"Display Model": "Model"})

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 1.8)

    header_color = "#264653"
    highlight_color = "#d8f3dc"

    for col_idx in range(len(table_df.columns)):
        table[(0, col_idx)].set_facecolor(header_color)
        table[(0, col_idx)].set_text_props(weight="bold", color="white")

    best_accuracy = table_df["Accuracy"].max()
    best_f1 = table_df["F1 Score"].max()
    best_auc = table_df["AUC Score"].max()

    for row_idx in range(len(table_df)):
        if table_df.iloc[row_idx]["Accuracy"] == best_accuracy:
            table[(row_idx + 1, 1)].set_facecolor(highlight_color)
        if table_df.iloc[row_idx]["F1 Score"] == best_f1:
            table[(row_idx + 1, 2)].set_facecolor(highlight_color)
        if table_df.iloc[row_idx]["AUC Score"] == best_auc:
            table[(row_idx + 1, 3)].set_facecolor(highlight_color)

    ax.set_title("Committed Benchmark Snapshot", fontsize=18, fontweight="bold", pad=24)

    fig.tight_layout()
    save_path = save_dir / "comprehensive_table.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    print("Generating benchmark visualizations")
    print(f"Reading results from: {RESULTS_PATH}")

    df = load_model_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    create_classification_metrics_comparison(df, OUTPUT_DIR)
    create_performance_efficiency_tradeoff(df, OUTPUT_DIR)
    create_comprehensive_table(df, OUTPUT_DIR)

    print(f"Visualization files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
