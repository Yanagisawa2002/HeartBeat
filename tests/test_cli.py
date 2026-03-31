import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src import cli


class TestCLI(unittest.TestCase):
    def test_build_parser_parses_train_command(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/config.yaml",
                "train",
                "--models",
                "cnn1d",
                "lstm",
                "--epochs",
                "3",
                "--batch-size",
                "8",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(args.models, ["cnn1d", "lstm"])
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 8)

    def test_build_parser_accepts_inception1d(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/config.yaml",
                "train",
                "--models",
                "inception1d",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(args.models, ["inception1d"])

    def test_main_dispatches_preprocess(self) -> None:
        mock_loader_cls = MagicMock()
        loader = mock_loader_cls.return_value
        loader.processed_data_path = "data/processed"
        fake_module = types.ModuleType("src.data_loader")
        fake_module.PTBDataLoader = mock_loader_cls

        with patch.dict(sys.modules, {"src.data_loader": fake_module}):
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    ["prog", "--config", "configs/config.yaml", "preprocess", "--max-samples", "5"],
                ):
                    cli.main()

        loader.process_and_save_data.assert_called_once_with(max_samples=5)

    def test_main_dispatches_train(self) -> None:
        mock_train_models = MagicMock()
        mock_train_models.return_value = pd.DataFrame(
            [{"Model": "CNN1D", "Accuracy": 0.9}]
        )
        fake_module = types.ModuleType("src.benchmark")
        fake_module.train_models = mock_train_models

        with patch.dict(sys.modules, {"src.benchmark": fake_module}):
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "prog",
                        "--config",
                        "configs/config.yaml",
                        "train",
                        "--models",
                        "cnn1d",
                        "--epochs",
                        "2",
                    ],
                ):
                    cli.main()

        kwargs = mock_train_models.call_args.kwargs
        self.assertEqual(kwargs["model_names"], ["cnn1d"])
        self.assertEqual(kwargs["epochs"], 2)
        self.assertTrue(str(kwargs["config_path"]).endswith("config.yaml"))

    def test_main_dispatches_evaluate(self) -> None:
        mock_evaluate_models = MagicMock()
        mock_evaluate_models.return_value = pd.DataFrame(
            [{"Model": "CNN1D", "Accuracy": 0.9}]
        )
        fake_module = types.ModuleType("src.benchmark")
        fake_module.evaluate_models = mock_evaluate_models

        with patch.dict(sys.modules, {"src.benchmark": fake_module}):
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "prog",
                        "--config",
                        "configs/config.yaml",
                        "evaluate",
                        "--models",
                        "cnn1d",
                        "--batch-size",
                        "4",
                    ],
                ):
                    cli.main()

        kwargs = mock_evaluate_models.call_args.kwargs
        self.assertEqual(kwargs["model_names"], ["cnn1d"])
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertTrue(str(kwargs["config_path"]).endswith("config.yaml"))
