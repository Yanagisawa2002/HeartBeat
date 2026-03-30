from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


CLASS_NAMES = {0: "Normal", 1: "Abnormal"}


def _load_prediction_artifact(prediction_csv_path: Path) -> pd.DataFrame:
    prediction_df = pd.read_csv(prediction_csv_path)
    required_columns = {
        "true_label",
        "predicted_class",
        "predicted_probability_abnormal",
        "model_name",
        "split",
    }
    missing_columns = required_columns - set(prediction_df.columns)
    if missing_columns:
        raise ValueError(
            "Prediction artifact is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    return prediction_df


def _save_confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    labels = [0, 1]
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{CLASS_NAMES[label].lower()}" for label in labels],
        columns=[f"pred_{CLASS_NAMES[label].lower()}" for label in labels],
    )
    confusion_path = output_dir / "confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion.astype(float),
        row_sums,
        out=np.zeros_like(confusion, dtype=float),
        where=row_sums != 0,
    )
    normalized_df = pd.DataFrame(
        normalized,
        index=confusion_df.index,
        columns=confusion_df.columns,
    )
    normalized_path = output_dir / "confusion_matrix_normalized.csv"
    normalized_df.to_csv(normalized_path)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([CLASS_NAMES[label] for label in labels])
    ax.set_yticklabels([CLASS_NAMES[label] for label in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = confusion[row_index, col_index]
            percent = normalized[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value}\n{percent:.1%}",
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    figure_path = output_dir / "confusion_matrix.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "confusion_matrix_csv": str(confusion_path),
        "confusion_matrix_normalized_csv": str(normalized_path),
        "confusion_matrix_figure": str(figure_path),
    }


def _save_roc_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> Dict[str, str | float | None]:
    result: Dict[str, str | float | None] = {
        "roc_curve_csv": None,
        "roc_curve_figure": None,
        "roc_auc": None,
    }
    if len(np.unique(y_true)) < 2:
        return result

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    roc_path = output_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figure_path = output_dir / "roc_curve.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result.update(
        {
            "roc_curve_csv": str(roc_path),
            "roc_curve_figure": str(figure_path),
            "roc_auc": float(roc_auc),
        }
    )
    return result


def _save_precision_recall_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> Dict[str, str | float | None]:
    result: Dict[str, str | float | None] = {
        "precision_recall_curve_csv": None,
        "precision_recall_curve_figure": None,
        "average_precision": None,
    }
    if len(np.unique(y_true)) < 2:
        return result

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    threshold_column = np.append(thresholds, np.nan)
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": threshold_column,
        }
    )
    pr_path = output_dir / "precision_recall_curve.csv"
    pr_df.to_csv(pr_path, index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {average_precision:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figure_path = output_dir / "precision_recall_curve.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result.update(
        {
            "precision_recall_curve_csv": str(pr_path),
            "precision_recall_curve_figure": str(figure_path),
            "average_precision": float(average_precision),
        }
    )
    return result


def _compute_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, num=101)
    rows = []

    for threshold in thresholds:
        y_pred_threshold = (y_score >= threshold).astype(int)
        confusion = confusion_matrix(y_true, y_pred_threshold, labels=[0, 1])
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append(
            {
                "threshold": threshold,
                "accuracy": float((y_pred_threshold == y_true).mean()),
                "precision": precision_score(
                    y_true, y_pred_threshold, zero_division=0
                ),
                "recall": recall_score(y_true, y_pred_threshold, zero_division=0),
                "f1": f1_score(y_true, y_pred_threshold, zero_division=0),
                "specificity": specificity,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


def _save_threshold_sweep_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    threshold_df = _compute_threshold_sweep(y_true, y_score)
    threshold_path = output_dir / "threshold_sweep.csv"
    threshold_df.to_csv(threshold_path, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    ax.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity")
    ax.set_xlabel("Abnormal-class threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold Sweep")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    figure_path = output_dir / "threshold_sweep.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "threshold_sweep_csv": str(threshold_path),
        "threshold_sweep_figure": str(figure_path),
    }


def _save_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    metrics_path = output_dir / "per_class_metrics.csv"
    report_df.to_csv(metrics_path, index=False)
    return {"per_class_metrics_csv": str(metrics_path)}


def generate_evaluation_artifacts(
    prediction_csv_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, str | float | None]:
    """
    Generate richer evaluation artifacts from a prediction-level CSV.

    The prediction CSV is treated as the source of truth for downstream
    evaluation reporting so that plots and machine-readable summaries are
    directly tied to the saved prediction outputs.
    """
    prediction_csv_path = Path(prediction_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = _load_prediction_artifact(prediction_csv_path)
    y_true = prediction_df["true_label"].astype(int).to_numpy()
    y_pred = prediction_df["predicted_class"].astype(int).to_numpy()
    y_score = prediction_df["predicted_probability_abnormal"].astype(float).to_numpy()

    artifacts: Dict[str, str | float | None] = {
        "prediction_csv": str(prediction_csv_path),
        "model_name": str(prediction_df["model_name"].iloc[0]),
        "split": str(prediction_df["split"].iloc[0]),
        "num_samples": int(len(prediction_df)),
    }

    artifacts.update(_save_confusion_matrix_artifacts(y_true, y_pred, output_dir))
    artifacts.update(_save_precision_recall_artifacts(y_true, y_score, output_dir))
    artifacts.update(_save_roc_artifacts(y_true, y_score, output_dir))
    artifacts.update(_save_threshold_sweep_artifacts(y_true, y_score, output_dir))
    artifacts.update(_save_per_class_metrics(y_true, y_pred, output_dir))

    summary_path = output_dir / "artifact_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)
    artifacts["artifact_summary_json"] = str(summary_path)

    return artifacts
