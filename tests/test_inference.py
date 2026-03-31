import unittest
from pathlib import Path

import numpy as np
import torch

from src.comparison_models import create_comparison_model
from src.inference import list_available_models, predict_single_window


TEST_TMP_ROOT = Path(".tmp_tests") / "inference"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write_dummy_checkpoint(path: Path, model_name: str = "cnn1d") -> None:
    model = create_comparison_model(
        model_name=model_name,
        input_dim=12,
        seq_len=1000,
        num_classes=2,
    )
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "input_dim": 12,
            "seq_len": 1000,
            "num_classes": 2,
            "epoch": 1,
            "best_val_accuracy": 0.0,
            "seed": 42,
        },
        path,
    )


class TestInferenceUtilities(unittest.TestCase):
    def test_list_available_models_discovers_checkpoint(self) -> None:
        case_dir = TEST_TMP_ROOT / "discover"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = case_dir / "cnn1d_best.pth"
        _write_dummy_checkpoint(checkpoint_path)

        models = list_available_models(checkpoint_dirs=[case_dir])

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "cnn1d")
        self.assertEqual(Path(models[0].checkpoint_path), checkpoint_path.resolve())

    def test_predict_single_window_accepts_transposed_input(self) -> None:
        case_dir = TEST_TMP_ROOT / "predict"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = case_dir / "cnn1d_best.pth"
        _write_dummy_checkpoint(checkpoint_path)

        synthetic_ecg = np.random.randn(1000, 12).astype(np.float32)
        prediction = predict_single_window(
            ecg=synthetic_ecg,
            model_name="cnn1d",
            checkpoint_dirs=[case_dir],
            device="cpu",
        )

        self.assertEqual(prediction["model_name"], "cnn1d")
        self.assertEqual(prediction["input_shape"], [12, 1000])
        self.assertIn(prediction["predicted_class"], [0, 1])
        self.assertAlmostEqual(
            prediction["probability_normal"] + prediction["probability_abnormal"],
            1.0,
            places=5,
        )

    def test_predict_single_window_rejects_invalid_shape(self) -> None:
        case_dir = TEST_TMP_ROOT / "reject_invalid"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = case_dir / "cnn1d_best.pth"
        _write_dummy_checkpoint(checkpoint_path)

        with self.assertRaises(ValueError):
            predict_single_window(
                ecg=np.random.randn(8, 500).astype(np.float32),
                model_name="cnn1d",
                checkpoint_dirs=[case_dir],
                device="cpu",
            )
