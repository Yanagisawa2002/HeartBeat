from __future__ import annotations

import argparse

from .comparison_models import BENCHMARK_MODEL_NAMES
from .config_utils import resolve_config_path


SUPPORTED_MODELS = BENCHMARK_MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECG classification benchmark CLI")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the runtime config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess PTB-XL and save train/val/test splits.")
    preprocess_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick test runs.")

    train_parser = subparsers.add_parser("train", help="Train baseline models on the processed splits.")
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="Models to train.",
    )
    train_parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=None, help="Override the learning rate.")
    train_parser.add_argument("--results-dir", default=None, help="Directory for checkpoints and result tables.")
    train_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick test runs.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved checkpoints on the processed test split.")
    evaluate_parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="Models to evaluate.",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=None, help="Override the evaluation batch size.")
    evaluate_parser.add_argument("--results-dir", default=None, help="Directory containing checkpoints and output CSV.")
    evaluate_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick test runs.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolved_config = resolve_config_path(args.config)

    if args.command == "preprocess":
        from .data_loader import PTBDataLoader

        loader = PTBDataLoader(str(resolved_config))
        loader.process_and_save_data(max_samples=args.max_samples)
        print(f"Processed data written to: {loader.processed_data_path}")
        return

    if args.command == "train":
        from .benchmark import train_models

        df = train_models(
            config_path=str(resolved_config),
            model_names=args.models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            results_dir=args.results_dir,
            max_samples=args.max_samples,
        )
        print(df.to_string(index=False, float_format="%.4f"))
        return

    if args.command == "evaluate":
        from .benchmark import evaluate_models

        df = evaluate_models(
            config_path=str(resolved_config),
            model_names=args.models,
            batch_size=args.batch_size,
            results_dir=args.results_dir,
            max_samples=args.max_samples,
        )
        print(df.to_string(index=False, float_format="%.4f"))
        return
