from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import os

import numpy as np
import torch

from .comparison_models import (
    BENCHMARK_MODEL_NAMES,
    SEQUENCE_FIRST_MODEL_NAMES,
    create_comparison_model,
)
from .config_utils import PROJECT_ROOT, load_config, resolve_config_path
from .signal_preprocessing import ensure_lead_first_window, parse_ecg_csv_text, preprocess_ecg_signals


DEFAULT_CHECKPOINT_DIRS = (
    PROJECT_ROOT / "artifacts" / "checkpoints",
    PROJECT_ROOT / "results" / "comparison" / "models",
)
_MODEL_CACHE: Dict[tuple[str, str, str], tuple[torch.nn.Module, dict, torch.device]] = {}


@dataclass(frozen=True)
class AvailableModel:
    name: str
    checkpoint_path: str
    source_dir: str


def load_runtime_config(config_path: Optional[str] = None) -> dict:
    resolved_config = resolve_config_path(config_path)
    return load_config(str(resolved_config))


def get_signal_spec(config_path: Optional[str] = None) -> dict:
    config = load_runtime_config(config_path)
    data_config = config.get("data", {})
    return {
        "leads": data_config.get("leads", []),
        "num_leads": len(data_config.get("leads", [])) or 12,
        "signal_length": data_config.get("signal_length", 1000),
        "sampling_rate": data_config.get("sampling_rate", 100),
    }


def _resolve_checkpoint_dirs(
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
) -> List[Path]:
    if checkpoint_dirs:
        candidates = [Path(path).resolve() for path in checkpoint_dirs]
    else:
        configured_dir = os.getenv("HEARTBEAT_CHECKPOINT_DIR")
        if configured_dir:
            candidates = [Path(configured_dir).resolve()]
        else:
            candidates = [path.resolve() for path in DEFAULT_CHECKPOINT_DIRS]

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def list_available_models(
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
) -> List[AvailableModel]:
    available: list[AvailableModel] = []

    for checkpoint_dir in _resolve_checkpoint_dirs(checkpoint_dirs):
        if not checkpoint_dir.exists():
            continue

        for checkpoint_path in sorted(checkpoint_dir.glob("*_best.pth")):
            model_name = checkpoint_path.stem.removesuffix("_best").lower()
            if model_name not in BENCHMARK_MODEL_NAMES:
                continue
            available.append(
                AvailableModel(
                    name=model_name,
                    checkpoint_path=str(checkpoint_path),
                    source_dir=str(checkpoint_dir),
                )
            )

    # Keep the first discovered checkpoint for each registered model name.
    deduplicated: dict[str, AvailableModel] = {}
    for model in available:
        deduplicated.setdefault(model.name, model)

    return [deduplicated[name] for name in sorted(deduplicated)]


def resolve_checkpoint_path(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
) -> Path:
    normalized_name = model_name.lower()
    if normalized_name not in BENCHMARK_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {', '.join(BENCHMARK_MODEL_NAMES)}"
        )

    if checkpoint_path:
        resolved_path = Path(checkpoint_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
        return resolved_path

    for model_entry in list_available_models(checkpoint_dirs=checkpoint_dirs):
        if model_entry.name == normalized_name:
            return Path(model_entry.checkpoint_path)

    searched_dirs = ", ".join(str(path) for path in _resolve_checkpoint_dirs(checkpoint_dirs))
    raise FileNotFoundError(
        f"No checkpoint found for model '{normalized_name}'. "
        f"Searched: {searched_dirs}"
    )


def _normalize_device(device: Optional[str]) -> torch.device:
    requested = (device or os.getenv("HEARTBEAT_DEVICE", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_inference_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, dict, torch.device]:
    resolved_checkpoint = resolve_checkpoint_path(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_dirs=checkpoint_dirs,
    )
    torch_device = _normalize_device(device)
    cache_key = (model_name.lower(), str(resolved_checkpoint), str(torch_device))

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = torch.load(resolved_checkpoint, map_location=torch_device)
    model = create_comparison_model(
        model_name=model_name,
        input_dim=checkpoint["input_dim"],
        seq_len=checkpoint["seq_len"],
        num_classes=checkpoint.get("num_classes", 2),
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaded = (model, checkpoint, torch_device)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def prepare_single_window(
    ecg: np.ndarray,
    model_name: str,
    config_path: Optional[str] = None,
    preprocess: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Validate, optionally preprocess, and shape a single ECG window for inference.
    """
    signal_spec = get_signal_spec(config_path)
    normalized = ensure_lead_first_window(
        ecg,
        expected_leads=signal_spec["num_leads"],
        expected_length=signal_spec["signal_length"],
    )
    prepared = (
        preprocess_ecg_signals(normalized, sampling_rate=signal_spec["sampling_rate"])
        if preprocess
        else normalized.astype(np.float32, copy=False)
    )

    if model_name.lower() in SEQUENCE_FIRST_MODEL_NAMES:
        batched = np.transpose(prepared, (1, 0))[None, :, :]
    else:
        batched = prepared[None, :, :]

    tensor = torch.from_numpy(batched.astype(np.float32, copy=False))
    return prepared, tensor


def predict_single_window(
    ecg: np.ndarray,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    preprocess: bool = True,
) -> dict:
    """
    Run inference for one fixed-length 12-lead ECG window.
    """
    normalized_name = model_name.lower()
    prepared_window, model_input = prepare_single_window(
        ecg=ecg,
        model_name=normalized_name,
        config_path=config_path,
        preprocess=preprocess,
    )
    model, _, torch_device = load_inference_model(
        model_name=normalized_name,
        checkpoint_path=checkpoint_path,
        checkpoint_dirs=checkpoint_dirs,
        device=device,
    )

    with torch.no_grad():
        outputs = model(model_input.to(torch_device))
        probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))

    resolved_checkpoint = resolve_checkpoint_path(
        model_name=normalized_name,
        checkpoint_path=checkpoint_path,
        checkpoint_dirs=checkpoint_dirs,
    )
    return {
        "model_name": normalized_name,
        "predicted_class": predicted_class,
        "predicted_label": "abnormal" if predicted_class == 1 else "normal",
        "probability_normal": float(probabilities[0]),
        "probability_abnormal": float(probabilities[1]),
        "input_shape": list(prepared_window.shape),
        "preprocessing_applied": bool(preprocess),
        "checkpoint_path": str(resolved_checkpoint),
    }


def predict_from_csv_text(
    csv_text: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_dirs: Optional[Sequence[str | Path]] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    preprocess: bool = True,
) -> dict:
    signal_spec = get_signal_spec(config_path)
    ecg = parse_ecg_csv_text(
        csv_text,
        expected_leads=signal_spec["num_leads"],
        expected_length=signal_spec["signal_length"],
    )
    return predict_single_window(
        ecg=ecg,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_dirs=checkpoint_dirs,
        config_path=config_path,
        device=device,
        preprocess=preprocess,
    )
