#!/usr/bin/env python3
"""
Lightweight workflow helper for the ECG classification benchmark.

This script is intentionally read-only and informational. The authoritative
workflow is the module-based CLI:

    python -m src preprocess
    python -m src train
    python -m src evaluate
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def print_status() -> None:
    config_path = ROOT_DIR / "configs" / "config.yaml"
    raw_data_dir = ROOT_DIR / "data" / "raw"
    processed_data_dir = ROOT_DIR / "data" / "processed"
    results_dir = ROOT_DIR / "results" / "comparison"

    print("ECG Classification Benchmark Workflow")
    print("=" * 50)
    print(f"Config file: {'FOUND' if config_path.exists() else 'MISSING'} - {config_path}")
    print(f"Raw data dir: {'FOUND' if raw_data_dir.exists() else 'MISSING'} - {raw_data_dir}")
    print(f"Processed data dir: {'FOUND' if processed_data_dir.exists() else 'MISSING'} - {processed_data_dir}")
    print(f"Results dir: {'FOUND' if results_dir.exists() else 'MISSING'} - {results_dir}")
    print()


def print_commands() -> None:
    print("Recommended commands")
    print("-" * 50)
    print("pip install -r requirements.txt")
    print("python -m src preprocess --config configs/config.yaml")
    print("python -m src train --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm")
    print("python -m src evaluate --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm")
    print("python scripts/visualize_all_models.py")
    print()
    print("Quick test run")
    print("-" * 50)
    print("python -m src preprocess --config configs/config.yaml --max-samples 200")
    print("python -m src train --config configs/config.yaml --models cnn1d lstm --epochs 5 --batch-size 16")
    print("python -m src evaluate --config configs/config.yaml --models cnn1d lstm")
    print()
    print("Notes")
    print("-" * 50)
    print("- This repository is a supervised normal-vs-abnormal ECG classification benchmark.")
    print("- The maintained public workflow is the module-based CLI in src/.")
    print("- Results should be interpreted as benchmark artifacts, not as clinical deployment evidence.")


def main() -> None:
    print_status()
    print_commands()


if __name__ == "__main__":
    main()
