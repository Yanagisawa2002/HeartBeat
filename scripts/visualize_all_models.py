#!/usr/bin/env python3
"""
Generate benchmark-ready visualizations from a model comparison CSV.

This script is intentionally lightweight and report-oriented. It reads one
comparison table, creates a small set of high-value figures, and writes a short
text summary to the requested output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = ROOT_DIR / "results" / "comparison" / "model_comparison_results.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "visualization"

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECG benchmark visualizations.")
    parser.add_argument(
        "--results-csv",
        default=str(DEFAULT_RESULTS_CSV),
        help="Path to model comparison results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where figures and summary report will be written.",
    )
    return parser.parse_args()


def load_results(results_csv: str) -> pd.DataFrame:
    csv_path = Path(results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {
        "Model",
        "Accuracy",
        "F1 Score",
        "AUC Score",
        "Training Time (s)",
        "Inference Time (s)",
        "Parameters",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {', '.join(sorted(missing))}")

    return df


def build_model_colors(df: pd.DataFrame) -> tuple[list, dict[str, tuple]]:
    colors = sns.color_palette("husl", n_colors=len(df))
    color_lookup = {
        model_name: colors[index]
        for index, model_name in enumerate(df["Model"].tolist())
    }
    return colors, color_lookup


def create_classification_metrics_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    metric_columns = ["Accuracy", "F1 Score", "AUC Score"]
    model_names = df["Model"].tolist()
    x_positions = np.arange(len(model_names))
    offsets = np.linspace(-0.22, 0.22, num=len(metric_columns))
    colors = sns.color_palette("Set2", n_colors=len(metric_columns))

    for metric_index, metric_name in enumerate(metric_columns):
        metric_values = df[metric_name].to_numpy()
        ax.scatter(
            x_positions + offsets[metric_index],
            metric_values,
            s=180,
            color=colors[metric_index],
            label=metric_name,
            zorder=3,
        )
        for point_x, value in zip(x_positions + offsets[metric_index], metric_values):
            ax.text(
                point_x,
                value + 0.0018,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim(max(0.80, df[metric_columns].min().min() - 0.02), min(1.0, df[metric_columns].max().max() + 0.03))
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()

    output_path = output_dir / "classification_metrics_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def create_performance_efficiency_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors, _ = build_model_colors(df)
    params_m = df["Parameters"] / 1_000_000

    scatter = ax.scatter(
        params_m,
        df["Accuracy"],
        s=220,
        c=colors,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
    )
    _ = scatter  # Keep linter quiet when edgecolors triggers a style warning.

    for index, model_name in enumerate(df["Model"]):
        ax.annotate(
            model_name,
            (params_m.iloc[index], df["Accuracy"].iloc[index]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance-Efficiency Trade-off", fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    output_path = output_dir / "performance_efficiency_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_comprehensive_table(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    table_data = df.copy()
    table_data["Training Time (h)"] = (table_data["Training Time (s)"] / 3600).round(2)
    table_data["Inference Time (s)"] = table_data["Inference Time (s)"].round(3)
    table_data["Parameters (M)"] = (table_data["Parameters"] / 1_000_000).round(2)

    display_columns = [
        "Model",
        "Accuracy",
        "F1 Score",
        "AUC Score",
        "Training Time (h)",
        "Inference Time (s)",
        "Parameters (M)",
    ]
    table_data = table_data[display_columns].copy()
    for column in ["Accuracy", "F1 Score", "AUC Score"]:
        table_data[column] = table_data[column].round(4)

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 2.0)

    for column_index in range(len(table_data.columns)):
        table[(0, column_index)].set_facecolor("#1b6a68")
        table[(0, column_index)].set_text_props(weight="bold", color="white")

    max_accuracy = table_data["Accuracy"].max()
    max_f1 = table_data["F1 Score"].max()
    max_auc = table_data["AUC Score"].max()
    min_params = table_data["Parameters (M)"].min()

    for row_index in range(len(table_data)):
        if table_data.iloc[row_index]["Accuracy"] == max_accuracy:
            table[(row_index + 1, table_data.columns.get_loc("Accuracy"))].set_facecolor("#c7f1d5")
        if table_data.iloc[row_index]["F1 Score"] == max_f1:
            table[(row_index + 1, table_data.columns.get_loc("F1 Score"))].set_facecolor("#c7f1d5")
        if table_data.iloc[row_index]["AUC Score"] == max_auc:
            table[(row_index + 1, table_data.columns.get_loc("AUC Score"))].set_facecolor("#c7f1d5")
        if table_data.iloc[row_index]["Parameters (M)"] == min_params:
            table[(row_index + 1, table_data.columns.get_loc("Parameters (M)"))].set_facecolor("#e3ecff")

    ax.set_title("ECG Benchmark Comparison Table", fontsize=18, fontweight="bold", pad=22)
    plt.tight_layout()

    output_path = output_dir / "comprehensive_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _best_practical_model(df: pd.DataFrame) -> pd.Series:
    accuracy_gap = 0.01
    max_accuracy = df["Accuracy"].max()
    candidates = df[df["Accuracy"] >= max_accuracy - accuracy_gap].copy()
    if candidates.empty:
        candidates = df.copy()
    return candidates.sort_values(
        by=["Parameters", "Inference Time (s)"],
        ascending=[True, True],
    ).iloc[0]


def generate_summary_report(df: pd.DataFrame, output_dir: Path) -> Path:
    best_accuracy = df.loc[df["Accuracy"].idxmax()]
    best_f1 = df.loc[df["F1 Score"].idxmax()]
    best_auc = df.loc[df["AUC Score"].idxmax()]
    fastest_inference = df.loc[df["Inference Time (s)"].idxmin()]
    smallest_model = df.loc[df["Parameters"].idxmin()]
    practical_model = _best_practical_model(df)

    lines = [
        "ECG BENCHMARK SUMMARY",
        "=" * 40,
        "",
        "Leaders",
        f"- Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})",
        f"- Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})",
        f"- Best AUC Score: {best_auc['Model']} ({best_auc['AUC Score']:.4f})",
        f"- Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference Time (s)']:.3f}s)",
        f"- Smallest Model: {smallest_model['Model']} ({smallest_model['Parameters'] / 1_000_000:.2f}M params)",
        f"- Best Practical Baseline: {practical_model['Model']}",
        "",
        "Interpretation",
        f"- {best_accuracy['Model']} led the current run on the main discrimination metrics.",
        f"- {practical_model['Model']} offers the strongest performance-efficiency trade-off in this run.",
        "- These results describe supervised binary ECG classification on PTB-XL, not clinical diagnosis.",
        "",
        "Per-model details",
    ]

    for _, row in df.iterrows():
        lines.extend(
            [
                f"- {row['Model']}: "
                f"acc={row['Accuracy']:.4f}, "
                f"f1={row['F1 Score']:.4f}, "
                f"auc={row['AUC Score']:.4f}, "
                f"params={row['Parameters'] / 1_000_000:.2f}M, "
                f"inference={row['Inference Time (s)']:.3f}s, "
                f"train={row['Training Time (s)'] / 3600:.2f}h"
            ]
        )

    output_path = output_dir / "evaluation_summary_report.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    df = load_results(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_plot = create_classification_metrics_plot(df, output_dir)
    tradeoff_plot = create_performance_efficiency_plot(df, output_dir)
    table_plot = create_comprehensive_table(df, output_dir)
    summary_report = generate_summary_report(df, output_dir)

    print("Saved benchmark visualization artifacts:")
    for path in [metrics_plot, tradeoff_plot, table_plot, summary_report]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
