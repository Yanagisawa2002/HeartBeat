import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from src.data_loader import PTBDataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    PTBDataLoader = None
    DATA_LOADER_IMPORT_ERROR = exc
else:
    DATA_LOADER_IMPORT_ERROR = None


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


if PTBDataLoader is not None:
    class FakePTBDataLoader(PTBDataLoader):
        def __init__(self, config_path: str, metadata: pd.DataFrame, signals_by_record: dict[int, np.ndarray]):
            self._test_metadata = metadata
            self._signals_by_record = signals_by_record
            super().__init__(config_path)

        def load_ptbxl_database(self) -> pd.DataFrame:
            return self._test_metadata

        def _resolve_record_path(self, row: pd.Series) -> str:
            return str(int(row["ecg_id"]))

        def load_ptb_record(self, record_path: str):
            record_id = int(record_path)
            return self._signals_by_record[record_id], {"fs": self.sampling_rate}


@unittest.skipIf(
    PTBDataLoader is None,
    f"Optional data loader dependency unavailable: {DATA_LOADER_IMPORT_ERROR}",
)
class TestDataPipeline(unittest.TestCase):
    def _get_temp_root(self) -> Path:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "data_pipeline"
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root

    def _write_test_config(self, root: Path) -> Path:
        config = {
            "device": "cpu",
            "seed": 42,
            "data": {
                "raw_data_path": str(root / "raw"),
                "processed_data_path": str(root / "processed"),
                "sampling_rate": 100,
                "signal_length": 100,
                "leads": LEADS,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "num_epochs": 1,
                "weight_decay": 0.0001,
                "early_stopping_patience": 2,
            },
            "paths": {
                "model_save_path": str(root / "results" / "models"),
                "results_path": str(root / "results" / "comparison"),
                "logs_path": str(root / "logs"),
            },
            "evaluation": {
                "save_confusion_matrix": True,
                "save_roc_curve": True,
            },
            "graph": {
                "frequency_bands": [0.0, 0.5, 4.0, 8.0, 13.0, 30.0, 40.0],
            },
        }
        config_path = root / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        return config_path

    def _build_fake_metadata(self) -> pd.DataFrame:
        rows = [
            {"ecg_id": 1, "patient_id": 101, "scp_codes": "{'NORM': 100.0}"},
            {"ecg_id": 2, "patient_id": 102, "scp_codes": "{'MI': 100.0}"},
            {"ecg_id": 3, "patient_id": 103, "scp_codes": "{'NORM': 100.0}"},
            {"ecg_id": 4, "patient_id": 104, "scp_codes": "{'STTC': 100.0}"},
            {"ecg_id": 5, "patient_id": 105, "scp_codes": "{'NORM': 100.0}"},
            {"ecg_id": 6, "patient_id": 106, "scp_codes": "{'HYP': 100.0}"},
        ]
        return pd.DataFrame(rows).set_index("ecg_id")

    def _build_fake_signals(self) -> dict[int, np.ndarray]:
        signals_by_record = {}
        t = np.linspace(0.0, 4.0 * np.pi, 200)
        lead_offsets = np.linspace(0.0, 0.5, len(LEADS))

        for record_id in range(1, 7):
            signal = np.vstack(
                [
                    np.sin(t + lead_offset + record_id * 0.05)
                    + 0.01 * record_id
                    for lead_offset in lead_offsets
                ]
            )
            signals_by_record[record_id] = signal.astype(np.float32)

        return signals_by_record

    def test_process_and_save_data_writes_arrays_and_manifests(self) -> None:
        root = self._get_temp_root()
        config_path = self._write_test_config(root)
        metadata = self._build_fake_metadata()
        signals_by_record = self._build_fake_signals()

        loader = FakePTBDataLoader(str(config_path), metadata, signals_by_record)
        loader.process_and_save_data()

        processed_dir = root / "processed"
        results_dir = root / "results" / "comparison" / "preprocessing"

        for split in ("train", "val", "test"):
            self.assertTrue((processed_dir / f"X_{split}.npy").exists())
            self.assertTrue((processed_dir / f"y_{split}.npy").exists())
            self.assertTrue((processed_dir / f"{split}_manifest.csv").exists())
            self.assertTrue((processed_dir / f"{split}_window_manifest.csv").exists())

        combined_manifest_path = results_dir / "split_manifest_latest.csv"
        self.assertTrue(combined_manifest_path.exists())
        self.assertEqual(loader.last_split_level, "patient")

        train_x = np.load(processed_dir / "X_train.npy")
        train_window_manifest = pd.read_csv(processed_dir / "train_window_manifest.csv")
        self.assertEqual(len(train_window_manifest), train_x.shape[0])
        self.assertIn("source_record_id", train_window_manifest.columns)

        loaded_window_manifest = loader.load_window_manifest(
            "train", expected_samples=train_x.shape[0]
        )
        self.assertEqual(len(loaded_window_manifest), train_x.shape[0])

        combined_manifest = pd.read_csv(combined_manifest_path)
        source_split_counts = combined_manifest.groupby("ecg_id")["split"].nunique()
        self.assertTrue((source_split_counts == 1).all())
