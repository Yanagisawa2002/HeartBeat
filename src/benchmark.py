from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .comparison_models import (
    BENCHMARK_MODEL_NAMES,
    SEQUENCE_FIRST_MODEL_NAMES,
    create_comparison_model,
)
from .config_utils import load_config, resolve_config_path
from .data_loader import PTBDataLoader
from .evaluation_artifacts import generate_evaluation_artifacts


SUPPORTED_MODELS = BENCHMARK_MODEL_NAMES


def normalize_model_names(model_names: Sequence[str] | None) -> List[str]:
    if not model_names:
        return list(SUPPORTED_MODELS)

    normalized = [name.lower() for name in model_names]
    unknown = [name for name in normalized if name not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(
            f"Unsupported model(s): {', '.join(unknown)}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return normalized


def select_device(config: dict) -> torch.device:
    requested_device = config.get("device", "cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_processed_data(config_path: str, max_samples: int | None = None) -> PTBDataLoader:
    loader = PTBDataLoader(config_path)
    try:
        loader.load_processed_data("train")
        loader.load_processed_data("val")
        loader.load_processed_data("test")
    except FileNotFoundError:
        loader.process_and_save_data(max_samples=max_samples)
    return loader


def load_processed_context(
    config_path: str,
    ensure_data: bool = True,
    max_samples: int | None = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    PTBDataLoader,
]:
    loader = (
        ensure_processed_data(config_path, max_samples=max_samples)
        if ensure_data
        else PTBDataLoader(config_path)
    )
    X_train, y_train = loader.load_processed_data("train")
    X_val, y_val = loader.load_processed_data("val")
    X_test, y_test = loader.load_processed_data("test")
    return (X_train, X_val, X_test), (y_train, y_val, y_test), loader


def load_processed_splits(
    config_path: str,
    ensure_data: bool = True,
    max_samples: int | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    X_data, y_data, _ = load_processed_context(
        config_path,
        ensure_data=ensure_data,
        max_samples=max_samples,
    )
    return X_data, y_data


def _reshape_inputs(data: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in SEQUENCE_FIRST_MODEL_NAMES:
        return np.transpose(data, (0, 2, 1))
    return data


def create_dataloaders(
    X_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    model_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train, X_val, X_test = (_reshape_inputs(split, model_name) for split in X_data)
    y_train, y_val, y_test = y_data

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training split only.

    This preserves label semantics: the data loader does not relabel records,
    and imbalance is handled during optimization instead of by contaminating
    the abnormal class with relabeled normal examples.
    """
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)

    nonzero_mask = class_counts > 0
    if nonzero_mask.any():
        weights[nonzero_mask] = len(y_train) / (num_classes * class_counts[nonzero_mask])
    else:
        weights[:] = 1.0

    return torch.tensor(weights, dtype=torch.float32)


def _collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[int] = []
    labels: List[int] = []
    probabilities: List[np.ndarray] = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(target.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return np.array(predictions), np.array(probabilities), np.array(labels)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def _prepare_prediction_frame(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sample_manifest: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a stable prediction-level artifact aligned to the evaluated split."""
    num_samples = len(y_true)

    if sample_manifest is None or sample_manifest.empty:
        prediction_df = pd.DataFrame(
            {
                "window_id": [f"{split_name}_{index:07d}" for index in range(num_samples)],
                "sample_index_in_split": np.arange(num_samples, dtype=int),
                "source_record_id": pd.Series([pd.NA] * num_samples, dtype="object"),
                "patient_id": pd.Series([pd.NA] * num_samples, dtype="object"),
                "split": [split_name] * num_samples,
                "manifest_source": ["missing_window_manifest"] * num_samples,
            }
        )
    else:
        prediction_df = sample_manifest.copy().reset_index(drop=True)
        if len(prediction_df) != num_samples:
            raise ValueError(
                f"Prediction/sample-manifest length mismatch for split '{split_name}': "
                f"{num_samples} predictions vs {len(prediction_df)} manifest rows"
            )

        if "source_record_id" not in prediction_df.columns and "ecg_id" in prediction_df.columns:
            prediction_df["source_record_id"] = prediction_df["ecg_id"]
        if "window_id" not in prediction_df.columns:
            prediction_df["window_id"] = [
                f"{split_name}_{index:07d}" for index in range(num_samples)
            ]
        if "sample_index_in_split" not in prediction_df.columns:
            prediction_df["sample_index_in_split"] = np.arange(num_samples, dtype=int)
        prediction_df["split"] = split_name

    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        prob_normal = y_prob[:, 0]
        prob_abnormal = y_prob[:, 1]
    else:
        prob_abnormal = np.asarray(y_prob).reshape(-1)
        prob_normal = 1.0 - prob_abnormal

    prediction_df["model_name"] = model_name
    prediction_df["true_label"] = y_true.astype(int)
    prediction_df["predicted_class"] = y_pred.astype(int)
    prediction_df["predicted_probability_normal"] = prob_normal
    prediction_df["predicted_probability_abnormal"] = prob_abnormal
    prediction_df["correct"] = (
        prediction_df["true_label"] == prediction_df["predicted_class"]
    )

    preferred_columns = [
        "window_id",
        "sample_index_in_split",
        "source_record_id",
        "patient_id",
        "split",
        "model_name",
        "true_label",
        "predicted_class",
        "predicted_probability_abnormal",
        "predicted_probability_normal",
        "correct",
        "window_index_within_record",
        "segment_start_sample",
        "segment_end_sample",
        "filename_lr",
        "filename_hr",
        "record_path",
        "scp_codes",
        "manifest_source",
    ]
    available_columns = [
        column for column in preferred_columns if column in prediction_df.columns
    ]
    return prediction_df[available_columns]


def save_prediction_artifact(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sample_manifest: Optional[pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Save one prediction-level CSV per model and split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df = _prepare_prediction_frame(
        model_name=model_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sample_manifest=sample_manifest,
    )
    output_path = output_dir / f"{model_name}_{split_name}_predictions.csv"
    prediction_df.to_csv(output_path, index=False)
    print(f"Saved prediction artifact: {output_path}")
    return output_path


def train_single_model(
    model_name: str,
    X_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    checkpoint_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    test_sample_manifest: Optional[pd.DataFrame] = None,
    predictions_dir: Optional[Path] = None,
    evaluation_output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    train_loader, val_loader, _ = create_dataloaders(X_data, y_data, batch_size, model_name)
    input_dim = X_data[0].shape[1]
    seq_len = X_data[0].shape[2]
    y_train = y_data[0]

    model = create_comparison_model(
        model_name=model_name,
        input_dim=input_dim,
        seq_len=seq_len,
        num_classes=2,
    ).to(device)

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"Using class-weighted loss for {model_name}: "
        f"{class_weights.detach().cpu().tolist()}"
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        val_pred, val_prob, val_true = _collect_predictions(model, val_loader, device)
        val_metrics = compute_classification_metrics(val_true, val_pred, val_prob)

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "seq_len": seq_len,
                    "num_classes": 2,
                    "epoch": epoch + 1,
                    "best_val_accuracy": best_val_accuracy,
                    "seed": seed,
                },
                checkpoint_path,
            )

    training_time = time.time() - start_time
    metrics = evaluate_checkpoint(
        model_name,
        checkpoint_path,
        X_data,
        y_data,
        batch_size,
        device,
        split_name="test",
        sample_manifest=test_sample_manifest,
        predictions_dir=predictions_dir,
        evaluation_output_dir=evaluation_output_dir,
    )
    metrics["training_time"] = training_time
    metrics["best_epoch"] = best_epoch
    metrics["checkpoint_path"] = str(checkpoint_path)
    return metrics


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    X_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
    split_name: str = "test",
    sample_manifest: Optional[pd.DataFrame] = None,
    predictions_dir: Optional[Path] = None,
    evaluation_output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_loader = create_dataloaders(X_data, y_data, batch_size, model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_comparison_model(
        model_name=model_name,
        input_dim=checkpoint["input_dim"],
        seq_len=checkpoint["seq_len"],
        num_classes=checkpoint.get("num_classes", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    inference_start = time.time()
    y_pred, y_prob, y_true = _collect_predictions(model, test_loader, device)
    inference_time = time.time() - inference_start

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics["inference_time"] = inference_time
    metrics["parameters"] = sum(p.numel() for p in model.parameters())

    if predictions_dir is not None:
        artifact_path = save_prediction_artifact(
            model_name=model_name,
            split_name=split_name,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            sample_manifest=sample_manifest,
            output_dir=predictions_dir,
        )
        metrics["prediction_artifact_path"] = str(artifact_path)
        if evaluation_output_dir is not None:
            evaluation_artifacts = generate_evaluation_artifacts(
                prediction_csv_path=artifact_path,
                output_dir=evaluation_output_dir,
            )
            if evaluation_artifacts.get("artifact_summary_json") is not None:
                metrics["evaluation_artifact_summary_path"] = str(
                    evaluation_artifacts["artifact_summary_json"]
                )

    return metrics


def metrics_to_dataframe(rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": row["model_name"].upper(),
                "Accuracy": row["accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1 Score": row["f1"],
                "AUC Score": row["auc"],
                "Training Time (s)": row.get("training_time", 0.0),
                "Inference Time (s)": row["inference_time"],
                "Parameters": row["parameters"],
                "Checkpoint": row["checkpoint_path"],
            }
            for row in rows
        ]
    )


def resolve_output_paths(config: dict, results_dir: str | None) -> Tuple[Path, Path]:
    if results_dir:
        base_results_dir = Path(results_dir)
        checkpoint_dir = base_results_dir / "models"
    else:
        paths_config = config.get("paths", {})
        base_results_dir = Path(paths_config.get("results_path", "results/comparison"))
        checkpoint_dir = Path(paths_config.get("model_save_path", base_results_dir / "models"))
    return base_results_dir, checkpoint_dir


def train_models(
    config_path: str,
    model_names: Sequence[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    results_dir: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    resolved_config_path = resolve_config_path(config_path)
    config = load_config(str(resolved_config_path))
    set_random_seed(config.get("seed", 42))
    device = select_device(config)

    X_data, y_data, loader = load_processed_context(
        str(resolved_config_path),
        ensure_data=True,
        max_samples=max_samples,
    )

    normalized_models = normalize_model_names(model_names)
    training_config = config.get("training", {})
    batch_size = batch_size or training_config.get("batch_size", 32)
    epochs = epochs or training_config.get("num_epochs", 100)
    learning_rate = learning_rate or training_config.get("learning_rate", 1e-3)

    base_results_dir, checkpoint_dir = resolve_output_paths(config, results_dir)
    prediction_dir = base_results_dir / "predictions"
    evaluation_dir = base_results_dir / "evaluation"
    try:
        test_sample_manifest = loader.load_window_manifest(
            "test", expected_samples=len(y_data[2])
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Warning: could not load test window manifest: {exc}")
        test_sample_manifest = None

    rows = []
    for model_name in normalized_models:
        metrics = train_single_model(
            model_name=model_name,
            X_data=X_data,
            y_data=y_data,
            checkpoint_dir=checkpoint_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=config.get("seed", 42),
            test_sample_manifest=test_sample_manifest,
            predictions_dir=prediction_dir,
            evaluation_output_dir=evaluation_dir / model_name,
        )
        metrics["model_name"] = model_name
        rows.append(metrics)

    df = metrics_to_dataframe(rows)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_results_dir / "model_comparison_results.csv", index=False)
    return df


def evaluate_models(
    config_path: str,
    model_names: Sequence[str] | None = None,
    batch_size: int | None = None,
    results_dir: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    resolved_config_path = resolve_config_path(config_path)
    config = load_config(str(resolved_config_path))
    set_random_seed(config.get("seed", 42))
    device = select_device(config)

    X_data, y_data, loader = load_processed_context(
        str(resolved_config_path),
        ensure_data=True,
        max_samples=max_samples,
    )

    normalized_models = normalize_model_names(model_names)
    batch_size = batch_size or config.get("training", {}).get("batch_size", 32)
    base_results_dir, checkpoint_dir = resolve_output_paths(config, results_dir)
    prediction_dir = base_results_dir / "predictions"
    evaluation_dir = base_results_dir / "evaluation"
    try:
        test_sample_manifest = loader.load_window_manifest(
            "test", expected_samples=len(y_data[2])
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Warning: could not load test window manifest: {exc}")
        test_sample_manifest = None

    rows = []
    for model_name in normalized_models:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        metrics = evaluate_checkpoint(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            X_data=X_data,
            y_data=y_data,
            batch_size=batch_size,
            device=device,
            split_name="test",
            sample_manifest=test_sample_manifest,
            predictions_dir=prediction_dir,
            evaluation_output_dir=evaluation_dir / model_name,
        )
        metrics["model_name"] = model_name
        metrics["checkpoint_path"] = str(checkpoint_path)
        rows.append(metrics)

    df = metrics_to_dataframe(rows)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_results_dir / "model_comparison_results.csv", index=False)
    return df
