#!/usr/bin/env python3
"""
Project setup helper for the PTB-XL ECG classification benchmark.

This script prepares a local working tree for the documented CLI workflow.
It creates the expected folders, can install dependencies, and writes a
short raw-data note under data/raw/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "configs",
    "data",
    "data/raw",
    "data/processed",
    "docs",
    "logs",
    "artifacts",
    "artifacts/checkpoints",
    "results",
    "scripts",
]


def print_banner() -> None:
    print("PTB-XL ECG Classification Benchmark Setup")
    print("=" * 50)


def check_python_version() -> None:
    version = sys.version_info
    if version < (3, 8):
        raise RuntimeError(
            f"Python 3.8+ is required. Detected {version.major}.{version.minor}.{version.micro}."
        )
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")


def create_directories() -> None:
    print("\nCreating repository directories...")
    for relative_path in REQUIRED_DIRS:
        path = ROOT_DIR / relative_path
        path.mkdir(parents=True, exist_ok=True)
        print(f"  created: {path}")


def install_dependencies() -> None:
    requirements_path = ROOT_DIR / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing dependency file: {requirements_path}")

    print("\nInstalling Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )
    print("Dependencies installed.")


def write_data_note() -> None:
    readme_path = ROOT_DIR / "data" / "raw" / "README.md"
    content = """# PTB-XL Raw Data

Place the PTB-XL dataset under:

```text
data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
```

The repository expects the raw PhysioNet release to be downloaded manually.
It is not redistributed here.

Reference:
- https://physionet.org/content/ptb-xl/
"""
    readme_path.write_text(content, encoding="utf-8")
    print(f"Raw-data note written to {readme_path}")


def print_next_steps() -> None:
    print("\nNext steps")
    print("-" * 50)
    print("1. Download PTB-XL into data/raw/.")
    print("2. Review the current workflow:")
    print("   python scripts/example_usage.py")
    print("3. Preprocess data:")
    print("   python -m src preprocess --config configs/config.yaml")
    print("4. Train models:")
    print("   python -m src train --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm inception1d")
    print("5. Evaluate saved checkpoints:")
    print("   python -m src evaluate --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm inception1d")
    print("6. Generate comparison plots:")
    print("   python scripts/visualize_all_models.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the PTB-XL ECG benchmark repository.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip dependency installation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_banner()
    check_python_version()
    create_directories()
    write_data_note()

    if not args.skip_install:
        install_dependencies()
    else:
        print("\nSkipping dependency installation.")

    print_next_steps()


if __name__ == "__main__":
    main()
