import os
import unittest
from pathlib import Path

import numpy as np
import torch

from src.comparison_models import create_comparison_model

TEST_TMP_ROOT = Path(".tmp_tests") / "api"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

try:
    from fastapi.testclient import TestClient
    from app.main import app
except Exception:  # pragma: no cover - optional dependency in local smoke tests
    TestClient = None
    app = None


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


@unittest.skipIf(TestClient is None or app is None, "FastAPI test dependencies unavailable.")
class TestWebDemoAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.case_dir = TEST_TMP_ROOT / self._testMethodName
        self.case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.case_dir / "cnn1d_best.pth"
        _write_dummy_checkpoint(checkpoint_path)
        self._previous_checkpoint_dir = os.environ.get("HEARTBEAT_CHECKPOINT_DIR")
        self._previous_device = os.environ.get("HEARTBEAT_DEVICE")
        os.environ["HEARTBEAT_CHECKPOINT_DIR"] = str(self.case_dir)
        os.environ["HEARTBEAT_DEVICE"] = "cpu"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self._previous_checkpoint_dir is None:
            os.environ.pop("HEARTBEAT_CHECKPOINT_DIR", None)
        else:
            os.environ["HEARTBEAT_CHECKPOINT_DIR"] = self._previous_checkpoint_dir

        if self._previous_device is None:
            os.environ.pop("HEARTBEAT_DEVICE", None)
        else:
            os.environ["HEARTBEAT_DEVICE"] = self._previous_device

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_models_endpoint_lists_available_checkpoint(self) -> None:
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["name"], "cnn1d")

    def test_samples_endpoint_lists_bundled_inputs(self) -> None:
        response = self.client.get("/samples")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload), 2)
        sources = {record["source"] for record in payload}
        self.assertIn("ptb-xl-v1.0.1", sources)
        self.assertIn("synthetic-demo", sources)

    def test_predict_file_endpoint(self) -> None:
        synthetic_ecg = np.random.randn(12, 1000).astype(np.float32)
        csv_text = "\n".join(
            ",".join(f"{value:.6f}" for value in row)
            for row in synthetic_ecg
        )

        response = self.client.post(
            "/predict-file",
            data={"model_name": "cnn1d"},
            files={"file": ("synthetic.csv", csv_text, "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_name"], "cnn1d")
        self.assertEqual(payload["input_shape"], [12, 1000])
        self.assertIn(payload["predicted_class"], [0, 1])

    def test_predict_rejects_bad_shape(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "model_name": "cnn1d",
                "ecg": np.random.randn(3, 9).tolist(),
            },
        )

        self.assertEqual(response.status_code, 400)
