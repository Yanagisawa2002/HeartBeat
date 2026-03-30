#!/usr/bin/env python3
"""
Generate benchmark visualizations from the committed comparison CSV.

This script is intentionally report-oriented. It reads the committed model
comparison table, treats known placeholder training times as missing, and
writes a consistent set of figures and a short text summary under
results/visualization/.
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

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11


def load_model_results() -> pd.DataFrame | None:
    results_path = ROOT_DIR / "results" / "comparison" / "model_comparison_results.csv"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None

    df = pd.read_csv(results_path)

    # The committed snapshot uses 3000.0 as a placeholder for some training times.
    placeholder_mask = df["Training Time (s)"].eq(3000.0) & df["Model"].isin(["CNN1D", "LSTM"])
    df.loc[placeholder_mask, "Training Time (s)"] = np.nan

    print("Loaded model results:")
    print(df)
    return df


def create_performance_comparison(df: pd.DataFrame, save_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Normal-vs-Abnormal ECG Classification Model Comparison", fontsize=18, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    metrics = [
        ("Accuracy", axes[0, 0], (0.85, 0.95)),
        ("F1 Score", axes[0, 1], (0.85, 0.95)),
        ("AUC Score", axes[1, 0], (0.95, 0.99)),
    ]

    for column, axis, y_limits in metrics:
        bars = axis.bar(df["Model"], df[column], color=colors, alpha=0.8)
        axis.set_title(f"{column} Comparison", fontweight="bold")
        axis.set_ylabel(column)
        axis.set_ylim(*y_limits)
        axis.tick_params(axis="x", rotation=45)
        for bar, value in zip(bars, df[column]):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.002,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    training_hours = df["Training Time (s)"] / 3600
    bars = axes[1, 1].bar(df["Model"], training_hours, color=colors, alpha=0.8)
    axes[1, 1].set_title("Training Time Comparison", fontweight="bold")
    axes[1, 1].set_ylabel("Training Time (hours)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    for bar, value in zip(bars, training_hours):
        if pd.notna(value):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.02,
                f"{value:.2f}h",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                0.02,
                "NA",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    save_path = save_dir / "performance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Performance comparison saved to: {save_path}")


def create_radar_chart(df: pd.DataFrame, save_dir: Path) -> None:
    metrics = ["Accuracy", "F1 Score", "AUC Score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for color, (_, row) in zip(colors, df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Classification Metric Radar Chart", size=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    save_path = save_dir / "radar_chart_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar chart saved to: {save_path}")


def create_efficiency_analysis(df: pd.DataFrame, save_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Efficiency Analysis", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    params_m = df["Parameters"] / 1_000_000
    ax1.scatter(params_m, df["Accuracy"], c=colors, s=150, alpha=0.7, edgecolors="black")
    ax1.set_xlabel("Parameters (Millions)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Parameter Efficiency\n(Accuracy vs Model Size)")
    ax1.grid(True, alpha=0.3)
    for index, model in enumerate(df["Model"]):
        ax1.annotate(
            model,
            (params_m.iloc[index], df["Accuracy"].iloc[index]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color="black",
        )

    training_df = df.dropna(subset=["Training Time (s)"]).copy()
    if not training_df.empty:
        training_hours = training_df["Training Time (s)"] / 3600
        training_colors = [colors[i] for i in training_df.index]
        ax2.scatter(training_hours, training_df["Accuracy"], c=training_colors, s=150, alpha=0.7, edgecolors="black")
        ax2.set_xlabel("Training Time (Hours)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Efficiency\n(Accuracy vs Training Time)")
        ax2.grid(True, alpha=0.3)
        for idx, (_, row) in enumerate(training_df.iterrows()):
            ax2.annotate(
                row["Model"],
                (training_hours.iloc[idx], row["Accuracy"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                color="black",
            )
    else:
        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            "Training-time comparison unavailable\nfor this result snapshot",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = save_dir / "efficiency_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Efficiency analysis saved to: {save_path}")


def create_comprehensive_table(df: pd.DataFrame, save_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = df.copy()
    table_data["Training Time (h)"] = (table_data["Training Time (s)"] / 3600).round(2)
    table_data["Training Time (h)"] = table_data["Training Time (h)"].where(table_data["Training Time (h)"].notna(), "NA")
    table_data["Parameters (M)"] = (table_data["Parameters"] / 1_000_000).round(2)
    table_data["Inference Time (s)"] = table_data["Inference Time (s)"].round(3)

    display_columns = [
        "Model",
        "Accuracy",
        "F1 Score",
        "AUC Score",
        "Training Time (h)",
        "Inference Time (s)",
        "Parameters (M)",
    ]
    table_data = table_data[display_columns]
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
    table.scale(1.2, 2)

    for column_index in range(len(table_data.columns)):
        table[(0, column_index)].set_facecolor("#4CAF50")
        table[(0, column_index)].set_text_props(weight="bold", color="white")

    ax.set_title("Normal-vs-Abnormal ECG Classification Comparison", fontsize=18, fontweight="bold", pad=30)

    for row_index in range(len(table_data)):
        if table_data.iloc[row_index, 1] == table_data["Accuracy"].max():
            table[(row_index + 1, 1)].set_facecolor("#90EE90")
        if table_data.iloc[row_index, 2] == table_data["F1 Score"].max():
            table[(row_index + 1, 2)].set_facecolor("#90EE90")
        if table_data.iloc[row_index, 3] == table_data["AUC Score"].max():
            table[(row_index + 1, 3)].set_facecolor("#90EE90")

    ax.text(
        0.02,
        0.02,
        "Legend:\nBest performance",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    save_path = save_dir / "comprehensive_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Comprehensive comparison table saved to: {save_path}")


def create_inference_speed_analysis(df: pd.DataFrame, save_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Inference Time Analysis", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax1.bar(df["Model"], df["Inference Time (s)"], color=colors, alpha=0.8)
    ax1.set_title("Inference Time Comparison")
    ax1.set_ylabel("Inference Time (seconds)")
    ax1.tick_params(axis="x", rotation=45)
    for bar, value in zip(bars, df["Inference Time (s)"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.1,
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.scatter(df["Inference Time (s)"], df["Accuracy"], c=colors, s=150, alpha=0.7, edgecolors="black")
    ax2.set_xlabel("Inference Time (seconds)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Inference Time Trade-off")
    ax2.grid(True, alpha=0.3)
    for index, model in enumerate(df["Model"]):
        ax2.annotate(
            model,
            (df["Inference Time (s)"].iloc[index], df["Accuracy"].iloc[index]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()
    save_path = save_dir / "inference_speed_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Inference speed analysis saved to: {save_path}")


def generate_summary_report(df: pd.DataFrame, save_dir: Path) -> None:
    report: list[str] = []
    report.append("NORMAL-VS-ABNORMAL ECG CLASSIFICATION SUMMARY")
    report.append("=" * 50)
    report.append("")

    best_accuracy = df.loc[df["Accuracy"].idxmax()]
    best_f1 = df.loc[df["F1 Score"].idxmax()]
    best_auc = df.loc[df["AUC Score"].idxmax()]
    training_df = df.dropna(subset=["Training Time (s)"])
    fastest_inference = df.loc[df["Inference Time (s)"].idxmin()]
    most_efficient = df.loc[(df["Accuracy"] / (df["Parameters"] / 1_000_000)).idxmax()]

    report.append("PERFORMANCE LEADERS:")
    report.append(f"- Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    report.append(f"- Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    report.append(f"- Best AUC Score: {best_auc['Model']} ({best_auc['AUC Score']:.4f})")
    if not training_df.empty:
        fastest_training = training_df.loc[training_df["Training Time (s)"].idxmin()]
        report.append(
            f"- Fastest Training (available runs): {fastest_training['Model']} "
            f"({fastest_training['Training Time (s)'] / 3600:.2f} hours)"
        )
    else:
        report.append("- Fastest Training: not reported in this result snapshot")
    report.append(f"- Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference Time (s)']:.3f}s)")
    report.append(
        f"- Most Parameter Efficient: {most_efficient['Model']} "
        f"(Accuracy/M-params: {most_efficient['Accuracy'] / (most_efficient['Parameters'] / 1_000_000):.3f})"
    )
    report.append("")

    report.append("DETAILED ANALYSIS:")
    for _, row in df.iterrows():
        report.append(f"\n{row['Model']}:")
        report.append(f"  - Accuracy: {row['Accuracy']:.4f}")
        report.append(f"  - F1 Score: {row['F1 Score']:.4f}")
        report.append(f"  - AUC Score: {row['AUC Score']:.4f}")
        if pd.notna(row["Training Time (s)"]):
            report.append(f"  - Training Time: {row['Training Time (s)'] / 3600:.2f} hours")
        else:
            report.append("  - Training Time: not reported")
        report.append(f"  - Inference Time: {row['Inference Time (s)']:.3f} seconds")
        report.append(f"  - Parameters: {row['Parameters'] / 1_000_000:.2f}M")

    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("Interpretation notes:")
    report.append("- Highest reported accuracy/F1/AUC in the committed results: LSTM")
    report.append("- Strongest size-latency trade-off in the committed results: CNN1D")
    report.append("- Lower reported inference time in the committed results: CNN1D and Hybrid CNN-LSTM")
    report.append("- These results describe supervised binary ECG classification, not clinical diagnosis.")

    report_text = "\n".join(report)
    save_path = save_dir / "evaluation_summary_report.txt"
    save_path.write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 70)
    print(report_text)
    print("\n" + "=" * 70)
    print(f"Summary report saved to: {save_path}")


def main() -> None:
    print("Normal-vs-Abnormal ECG Classification Visualization")
    print("=" * 50)

    df = load_model_results()
    if df is None:
        return

    save_dir = ROOT_DIR / "results" / "visualization"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")
    print(f"Results will be saved to: {save_dir}")

    create_performance_comparison(df, save_dir)
    create_radar_chart(df, save_dir)
    create_efficiency_analysis(df, save_dir)
    create_comprehensive_table(df, save_dir)
    create_inference_speed_analysis(df, save_dir)
    generate_summary_report(df, save_dir)

    print("\nAll visualizations completed successfully.")
    print(f"Files saved in: {save_dir}")


if __name__ == "__main__":
    main()
