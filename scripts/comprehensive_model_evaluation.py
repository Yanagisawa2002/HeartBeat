#!/usr/bin/env python3
"""
Evaluate the committed baseline checkpoints on the processed PTB-XL test split.

This helper is intended for the portfolio/reporting layer. It reads the saved
test arrays from data/processed/, evaluates the committed comparison models,
and writes a refreshed summary table and figures under results/.
"""

from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.comparison_models import create_comparison_model


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_test_arrays() -> tuple[np.ndarray, np.ndarray]:
    x_test_path = ROOT_DIR / "data" / "processed" / "X_test.npy"
    y_test_path = ROOT_DIR / "data" / "processed" / "y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "Processed test arrays were not found. Expected "
            f"{x_test_path} and {y_test_path}."
        )

    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    return x_test, y_test


def create_test_loader(x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> DataLoader:
    x_tensor = torch.FloatTensor(x_test)
    if model_name.lower() == "lstm":
        x_tensor = x_tensor.transpose(1, 2)
    y_tensor = torch.LongTensor(y_test)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    inference_times: list[float] = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            start_time = time.time()
            output = model(data)
            inference_times.append(time.time() - start_time)

            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "precision_per_class": precision_score(
            all_labels, all_preds, average=None, zero_division=0
        ),
        "recall_per_class": recall_score(
            all_labels, all_preds, average=None, zero_division=0
        ),
        "f1_per_class": f1_score(all_labels, all_preds, average=None, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "inference_time": float(sum(inference_times)),
        "predictions": all_preds,
        "probabilities": all_probs,
        "true_labels": all_labels,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def load_checkpoint_results() -> dict[str, float]:
    comparison_csv = ROOT_DIR / "results" / "comparison" / "model_comparison_results.csv"
    if not comparison_csv.exists():
        return {}

    df = pd.read_csv(comparison_csv)
    if "Model" not in df.columns or "Training Time (s)" not in df.columns:
        return {}

    return dict(zip(df["Model"].astype(str), df["Training Time (s)"]))


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict | None:
    if not checkpoint_path.exists():
        print(f"Missing checkpoint: {checkpoint_path}")
        return None

    model = create_comparison_model(
        model_name=model_name,
        input_dim=12,
        seq_len=x_test.shape[2],
        num_classes=2,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)

    test_loader = create_test_loader(x_test, y_test, model_name)
    return evaluate_model(model, test_loader, device, model_name.upper())


def save_results(all_results: list[dict]) -> None:
    training_times = load_checkpoint_results()
    rows = []
    for result in all_results:
        rows.append(
            {
                "Model": result["model_name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1"],
                "AUC Score": result["auc"],
                "Training Time (s)": training_times.get(result["model_name"], np.nan),
                "Inference Time (s)": result["inference_time"],
                "Parameters": result["parameters"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT_DIR / "results" / "comprehensive_model_evaluation.csv", index=False)

    with open(ROOT_DIR / "results" / "detailed_evaluation_results.pkl", "wb") as handle:
        pickle.dump(all_results, handle)


def create_visualizations(all_results: list[dict]) -> None:
    output_dir = ROOT_DIR / "results" / "comprehensive_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [result["model_name"] for result in all_results]
    colors = ["#C84C31", "#2A7F62", "#3A6EA5", "#8E7DBE"][: len(models)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Comprehensive Model Comparison", fontsize=16, fontweight="bold")

    accuracies = [result["accuracy"] for result in all_results]
    aucs = [result["auc"] for result in all_results]
    inference_times = [result["inference_time"] for result in all_results]
    parameters_m = [result["parameters"] / 1e6 for result in all_results]

    axes[0, 0].bar(models, accuracies, color=colors)
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(models, aucs, color=colors)
    axes[0, 1].set_title("ROC-AUC")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(models, inference_times, color=colors)
    axes[1, 0].set_title("Inference Time (s)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(models, parameters_m, color=colors)
    axes[1, 1].set_title("Parameters (Millions)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 8))
    for color, result in zip(colors, all_results):
        plt.plot(
            result["fpr"],
            result["tpr"],
            color=color,
            linewidth=2,
            label=f"{result['model_name']} (AUC = {result['auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]
    for axis, color, result in zip(axes, colors, all_results):
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
        )
        axis.set_title(result["model_name"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(all_results: list[dict]) -> None:
    print("=" * 80)
    print("Detailed model evaluation report")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models evaluated: {len(all_results)}")

    sorted_results = sorted(all_results, key=lambda item: item["auc"], reverse=True)
    for index, result in enumerate(sorted_results, start=1):
        print(f"\n{index}. {result['model_name']}")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        print(f"   F1 Score: {result['f1']:.4f}")
        print(f"   AUC Score: {result['auc']:.4f}")
        print(f"   Inference Time: {result['inference_time']:.4f}s")
        print(f"   Parameters: {result['parameters']:,}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_test, y_test = load_test_arrays()
    checkpoints = [
        (ROOT_DIR / "results" / "comparison" / "models" / "cnn1d_epoch_100.pth", "cnn1d"),
        (ROOT_DIR / "results" / "comparison" / "models" / "lstm_epoch_100.pth", "lstm"),
        (ROOT_DIR / "results" / "comparison" / "models" / "resnet1d_epoch_100.pth", "resnet1d"),
        (
            ROOT_DIR / "results" / "comparison" / "models" / "hybrid_cnn_lstm_epoch_100.pth",
            "hybrid_cnn_lstm",
        ),
    ]

    all_results: list[dict] = []
    for checkpoint_path, model_name in checkpoints:
        print(f"\nEvaluating {model_name} from {checkpoint_path}")
        result = evaluate_checkpoint(checkpoint_path, model_name, x_test, y_test, device)
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No checkpoints were evaluated successfully.")

    print_report(all_results)
    save_results(all_results)
    create_visualizations(all_results)
    print("\nSaved refreshed evaluation artifacts under results/.")


if __name__ == "__main__":
    main()
