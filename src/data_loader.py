import ast
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy import signal
from sklearn.model_selection import train_test_split

try:
    from config_utils import load_config, resolve_config_path
    from signal_preprocessing import preprocess_ecg_signals
except ImportError:
    from src.config_utils import load_config, resolve_config_path
    from src.signal_preprocessing import preprocess_ecg_signals


class PTBDataLoader:
    """
    PTB-XL data loader for supervised normal-vs-abnormal ECG classification.

    Protocol note:
    This loader now assigns train/validation/test splits at the source-record
    level before any window segmentation. If PTB-XL patient identifiers are
    present in the metadata, the split is performed at the patient level so
    that every record from the same patient stays in exactly one split.
    Window segmentation happens only after split assignment.
    """

    def __init__(self, config_path: str = "config.yaml"):
        resolved_config_path = resolve_config_path(config_path)
        self.config = load_config(str(resolved_config_path))
        self.project_root = Path(resolved_config_path).resolve().parents[1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(
                "CUDA memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )

        self.raw_data_path = self.config["data"]["raw_data_path"]
        self.processed_data_path = self.config["data"]["processed_data_path"]
        self.sampling_rate = self.config["data"]["sampling_rate"]
        self.signal_length = self.config["data"]["signal_length"]
        self.leads = self.config["data"]["leads"]
        self.random_seed = 42
        self.last_split_level: Optional[str] = None
        self.last_split_manifest_path: Optional[str] = None

        os.makedirs(self.processed_data_path, exist_ok=True)
        self.results_path = self.config.get("paths", {}).get(
            "results_path", self.processed_data_path
        )

        self.ptbxl_path = self._resolve_ptbxl_path()
        self.database_path = os.path.join(self.ptbxl_path, "ptbxl_database.csv")

    def _resolve_ptbxl_path(self) -> str:
        """
        Resolve the PTB-XL dataset root from a few practical local layouts.

        Preferred order:
        1. Official extracted folder name under ``data/raw``
        2. A shorter ``ptb-xl`` folder under ``data/raw``
        3. A repository-root ``ptb-xl`` folder
        """
        raw_data_root = Path(self.raw_data_path).resolve()
        candidates = [
            raw_data_root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
            raw_data_root / "ptb-xl",
            self.project_root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
            self.project_root / "ptb-xl",
        ]

        for candidate in candidates:
            if (
                candidate.exists()
                and (candidate / "ptbxl_database.csv").exists()
                and (candidate / "records100").exists()
            ):
                print(f"Using PTB-XL dataset root: {candidate}")
                return str(candidate)

        return str(
            raw_data_root
            / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
        )

    def load_ptb_record(self, record_path: str) -> Tuple[np.ndarray, Dict]:
        """Load one PTB-XL record from a WFDB path without file extension."""
        try:
            record = wfdb.rdrecord(record_path)
            signals = record.p_signal.T
            metadata = {
                "fs": record.fs,
                "sig_len": record.sig_len,
                "sig_name": record.sig_name,
                "units": record.units,
                "comments": record.comments,
            }
            return signals, metadata
        except Exception as exc:
            print(f"Error loading record {record_path}: {exc}")
            return None, None

    def preprocess_signal(self, signals: np.ndarray) -> np.ndarray:
        """Apply per-lead filtering and standardization."""
        return preprocess_ecg_signals(signals, sampling_rate=self.sampling_rate)

    def extract_features(self, signals: np.ndarray) -> np.ndarray:
        """Retained for compatibility with older exploratory code."""
        features = []

        for lead_signal in signals:
            lead_features = [
                np.mean(lead_signal),
                np.std(lead_signal),
                np.var(lead_signal),
                np.max(lead_signal),
                np.min(lead_signal),
                np.ptp(lead_signal),
                np.mean(np.abs(lead_signal)),
                np.sqrt(np.mean(lead_signal**2)),
            ]

            freqs, psd = signal.welch(lead_signal, fs=self.sampling_rate, nperseg=1024)
            freq_bands = self.config["graph"]["frequency_bands"]
            for index in range(len(freq_bands) - 1):
                band_mask = (freqs >= freq_bands[index]) & (freqs < freq_bands[index + 1])
                lead_features.append(np.sum(psd[band_mask]))

            dominant_freq = freqs[np.argmax(psd)]
            lead_features.append(dominant_freq)

            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            lead_features.append(spectral_entropy)
            features.append(lead_features)

        return np.array(features)

    def segment_signal(
        self, signals: np.ndarray, segment_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Segment a recording into fixed-length overlapping windows.

        This method is intentionally split-agnostic. The important protocol
        constraint is enforced by calling it only after each source record has
        already been assigned to train/validation/test.
        """
        if segment_length is None:
            segment_length = self.signal_length

        _, total_samples = signals.shape
        segments = []
        step_size = segment_length // 2  # 50% overlap

        for start in range(0, total_samples - segment_length + 1, step_size):
            end = start + segment_length
            segments.append(signals[:, start:end])

        return segments

    def load_ptbxl_database(self) -> pd.DataFrame:
        """Load PTB-XL metadata indexed by ecg_id."""
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"PTB-XL metadata not found: {self.database_path}")
        return pd.read_csv(self.database_path, index_col="ecg_id")

    def get_record_path(self, ecg_id: int, sampling_rate: int = 100) -> str:
        """Fallback path builder when filename columns are unavailable."""
        folder = f"{ecg_id:05d}"[:-3] + "000"
        filename = f"{ecg_id:05d}_lr"
        return os.path.join(self.ptbxl_path, f"records{sampling_rate}", folder, filename)

    def parse_scp_codes(self, scp_codes_str: str) -> dict:
        """Parse the PTB-XL SCP-code dictionary stored as text."""
        try:
            return ast.literal_eval(scp_codes_str)
        except Exception:
            return {}

    def is_normal_record(self, scp_codes_str: str) -> bool:
        """
        Define the binary label at the source-record level.

        A record is normal only if:
        - NORM is present with confidence >= 50
        - no other non-SR code has confidence >= 50
        """
        scp_dict = self.parse_scp_codes(scp_codes_str)

        if "NORM" in scp_dict and scp_dict["NORM"] >= 50.0:
            for code, confidence in scp_dict.items():
                if code != "NORM" and code != "SR" and confidence >= 50.0:
                    return False
            return True

        return False

    def _resolve_record_path(self, row: pd.Series) -> str:
        """Use PTB-XL filename metadata when available; fall back to ecg_id logic."""
        filename_lr = row.get("filename_lr")
        if isinstance(filename_lr, str) and filename_lr:
            return os.path.join(self.ptbxl_path, filename_lr)
        return self.get_record_path(int(row["ecg_id"]), sampling_rate=self.sampling_rate)

    def _build_record_metadata(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Build one row per source record before any segmentation.

        The resulting table is the authoritative source for:
        - source-record labels
        - split assignment
        - patient grouping when available
        """
        print("Loading PTB-XL metadata...")
        df = self.load_ptbxl_database()

        records = []
        print("Deriving record-level labels from SCP codes...")
        for ecg_id, row in df.iterrows():
            if pd.isna(row.get("scp_codes")):
                continue

            label = 0 if self.is_normal_record(row["scp_codes"]) else 1
            records.append(
                {
                    "ecg_id": int(ecg_id),
                    "label": label,
                    "patient_id": row.get("patient_id", np.nan),
                    "filename_lr": row.get("filename_lr", None),
                    "filename_hr": row.get("filename_hr", None),
                    "scp_codes": row.get("scp_codes", None),
                }
            )

        records_df = pd.DataFrame(records).sort_values("ecg_id").reset_index(drop=True)
        normal_count = int((records_df["label"] == 0).sum())
        abnormal_count = int((records_df["label"] == 1).sum())
        print(f"Found {normal_count} normal records")
        print(f"Found {abnormal_count} abnormal records")

        # Do not alter source-record labels to "balance" the dataset.
        # Label-preserving imbalance handling belongs in the training stage,
        # e.g. class-weighted loss or a weighted sampler.
        if normal_count > 0 and abnormal_count > 0:
            imbalance_ratio = max(normal_count, abnormal_count) / min(
                normal_count, abnormal_count
            )
            print(f"Record-level class imbalance ratio: {imbalance_ratio:.2f}:1")
        else:
            print("Warning: only one class is present in the current record metadata.")

        if max_samples:
            normal_records = records_df[records_df["label"] == 0].head(
                min(max_samples // 2, (records_df["label"] == 0).sum())
            )
            abnormal_records = records_df[records_df["label"] == 1].head(
                min(max_samples - len(normal_records), (records_df["label"] == 1).sum())
            )
            records_df = (
                pd.concat([normal_records, abnormal_records])
                .sort_values("ecg_id")
                .reset_index(drop=True)
            )

        return records_df

    def _maybe_stratify(self, labels: pd.Series) -> Optional[pd.Series]:
        """Use stratification only when both classes have enough samples."""
        label_counts = labels.value_counts()
        if len(label_counts) < 2:
            return None
        if (label_counts < 2).any():
            return None
        return labels

    def _split_ids(
        self,
        ids: pd.Series,
        labels: pd.Series,
        train_fraction: float,
    ) -> Tuple[List, List]:
        """Split IDs with stratification when feasible, else plain random split."""
        test_fraction = 1.0 - train_fraction
        stratify = self._maybe_stratify(labels)
        try:
            train_ids, test_ids = train_test_split(
                ids,
                test_size=test_fraction,
                random_state=self.random_seed,
                stratify=stratify,
            )
        except ValueError as exc:
            print(f"Stratified split failed, falling back to random split: {exc}")
            train_ids, test_ids = train_test_split(
                ids,
                test_size=test_fraction,
                random_state=self.random_seed,
            )
        return list(train_ids), list(test_ids)

    def _split_record_metadata(self, records_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split source records before any windowing.

        Preferred behavior:
        - patient-level split if patient_id is available
        - otherwise record-level split

        In both cases, every source record belongs to exactly one split.
        """
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]
        test_ratio = self.config["data"]["test_ratio"]
        temp_train_fraction = val_ratio / (val_ratio + test_ratio)

        use_patient_level = (
            "patient_id" in records_df.columns
            and records_df["patient_id"].notna().all()
            and records_df["patient_id"].nunique() > 1
        )

        if use_patient_level:
            patient_df = (
                records_df.groupby("patient_id", as_index=False)
                .agg(patient_label=("label", "max"))
                .sort_values("patient_id")
                .reset_index(drop=True)
            )

            train_patients, temp_patients = self._split_ids(
                patient_df["patient_id"], patient_df["patient_label"], train_ratio
            )

            temp_patient_df = patient_df[patient_df["patient_id"].isin(temp_patients)]
            val_patients, test_patients = self._split_ids(
                temp_patient_df["patient_id"],
                temp_patient_df["patient_label"],
                temp_train_fraction,
            )

            split_tables = {
                "train": records_df[records_df["patient_id"].isin(train_patients)].copy(),
                "val": records_df[records_df["patient_id"].isin(val_patients)].copy(),
                "test": records_df[records_df["patient_id"].isin(test_patients)].copy(),
            }
            self.last_split_level = "patient"
        else:
            try:
                train_df, temp_df = train_test_split(
                    records_df,
                    test_size=(1.0 - train_ratio),
                    random_state=self.random_seed,
                    stratify=self._maybe_stratify(records_df["label"]),
                )
            except ValueError as exc:
                print(f"Stratified split failed, falling back to random split: {exc}")
                train_df, temp_df = train_test_split(
                    records_df,
                    test_size=(1.0 - train_ratio),
                    random_state=self.random_seed,
                )

            temp_stratify = self._maybe_stratify(temp_df["label"])
            try:
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=(1.0 - temp_train_fraction),
                    random_state=self.random_seed,
                    stratify=temp_stratify,
                )
            except ValueError as exc:
                print(f"Stratified split failed, falling back to random split: {exc}")
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=(1.0 - temp_train_fraction),
                    random_state=self.random_seed,
                )

            split_tables = {
                "train": train_df.copy(),
                "val": val_df.copy(),
                "test": test_df.copy(),
            }
            self.last_split_level = "record"

        for split_name, split_df in split_tables.items():
            split_tables[split_name] = (
                split_df.sort_values("ecg_id").reset_index(drop=True)
            )

        print(f"Split level: {self.last_split_level}")
        for split_name, split_df in split_tables.items():
            normal_count = int((split_df["label"] == 0).sum())
            abnormal_count = int((split_df["label"] == 1).sum())
            print(
                f"{split_name}: {len(split_df)} records "
                f"(normal={normal_count}, abnormal={abnormal_count})"
            )

        return split_tables

    def _process_split_records(
        self, split_df: pd.DataFrame, split_name: str
    ) -> Tuple[List[np.ndarray], List[int], pd.DataFrame, pd.DataFrame]:
        """
        Preprocess and segment only the records assigned to one split.

        Because split assignment already happened at the record or patient level,
        every window produced here is guaranteed to stay inside a single split.

        Returns:
            processed_data:
                Window tensors in the exact order later written to X_<split>.npy.
            labels:
                Window labels aligned to processed_data.
            record_manifest:
                One row per source record with processing status and segment counts.
            window_manifest:
                One row per saved window. This is the provenance table used later
                by evaluation to map prediction rows back to source records.
        """
        processed_data: List[np.ndarray] = []
        labels: List[int] = []
        manifest_rows = []
        window_rows = []
        step_size = self.signal_length // 2
        split_sample_index = 0

        print(f"Processing {len(split_df)} source records for split '{split_name}'...")
        for index, row in split_df.iterrows():
            if index % 100 == 0:
                print(f"  Progress: {index}/{len(split_df)}")

            record_path = self._resolve_record_path(row)
            signals, _ = self.load_ptb_record(record_path)
            if signals is None:
                manifest_rows.append(
                    {
                        "ecg_id": int(row["ecg_id"]),
                        "patient_id": row.get("patient_id", np.nan),
                        "label": int(row["label"]),
                        "filename_lr": row.get("filename_lr", None),
                        "filename_hr": row.get("filename_hr", None),
                        "scp_codes": row.get("scp_codes", None),
                        "record_path": record_path,
                        "num_segments": 0,
                        "processing_status": "load_failed",
                        "error": "wfdb_load_failed",
                    }
                )
                continue

            try:
                processed_signals = self.preprocess_signal(signals)
                segments = self.segment_signal(processed_signals)
                label = int(row["label"])

                for window_index_within_record, segment in enumerate(segments):
                    processed_data.append(segment)
                    labels.append(label)
                    segment_start_sample = window_index_within_record * step_size
                    window_rows.append(
                        {
                            "window_id": f"{split_name}_{split_sample_index:07d}",
                            "sample_index_in_split": split_sample_index,
                            "window_index_within_record": window_index_within_record,
                            "source_record_id": int(row["ecg_id"]),
                            "patient_id": row.get("patient_id", np.nan),
                            "split": split_name,
                            "label": label,
                            "filename_lr": row.get("filename_lr", None),
                            "filename_hr": row.get("filename_hr", None),
                            "scp_codes": row.get("scp_codes", None),
                            "record_path": record_path,
                            "segment_start_sample": segment_start_sample,
                            "segment_end_sample": segment_start_sample
                            + self.signal_length,
                            "manifest_source": "generated_during_preprocessing",
                        }
                    )
                    split_sample_index += 1

                manifest_rows.append(
                    {
                        "ecg_id": int(row["ecg_id"]),
                        "patient_id": row.get("patient_id", np.nan),
                        "label": label,
                        "filename_lr": row.get("filename_lr", None),
                        "filename_hr": row.get("filename_hr", None),
                        "scp_codes": row.get("scp_codes", None),
                        "num_segments": len(segments),
                        "record_path": record_path,
                        "processing_status": "processed",
                        "error": "",
                    }
                )
            except Exception as exc:
                print(f"Error processing record {row['ecg_id']}: {exc}")
                manifest_rows.append(
                    {
                        "ecg_id": int(row["ecg_id"]),
                        "patient_id": row.get("patient_id", np.nan),
                        "label": int(row["label"]),
                        "filename_lr": row.get("filename_lr", None),
                        "filename_hr": row.get("filename_hr", None),
                        "scp_codes": row.get("scp_codes", None),
                        "record_path": record_path,
                        "num_segments": 0,
                        "processing_status": "processing_failed",
                        "error": str(exc),
                    }
                )
                continue

        print(
            f"Finished split '{split_name}': "
            f"{len(processed_data)} windows from {len(manifest_rows)} records"
        )
        return (
            processed_data,
            labels,
            pd.DataFrame(manifest_rows),
            pd.DataFrame(window_rows),
        )

    def _save_split_manifest(
        self, split_manifests: Dict[str, pd.DataFrame]
    ) -> Optional[str]:
        """
        Save a combined record-level split manifest for the current preprocessing run.

        The manifest is written twice:
        - a timestamped per-run CSV
        - a stable `split_manifest_latest.csv` path for documentation and inspection
        """
        manifest_frames = []
        for split_name, manifest_df in split_manifests.items():
            if manifest_df is None or manifest_df.empty:
                continue

            manifest_copy = manifest_df.copy()
            manifest_copy["split"] = split_name
            manifest_copy["split_level"] = self.last_split_level
            manifest_copy["sampling_rate"] = self.sampling_rate
            manifest_copy["segment_length"] = self.signal_length
            manifest_copy["window_step_size"] = self.signal_length // 2
            manifest_frames.append(manifest_copy)

        if not manifest_frames:
            return None

        combined_manifest = (
            pd.concat(manifest_frames, ignore_index=True)
            .sort_values(["split", "ecg_id"])
            .reset_index(drop=True)
        )

        manifest_dir = os.path.join(self.results_path, "preprocessing")
        os.makedirs(manifest_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_manifest_path = os.path.join(manifest_dir, f"split_manifest_{timestamp}.csv")
        latest_manifest_path = os.path.join(manifest_dir, "split_manifest_latest.csv")

        combined_manifest.to_csv(run_manifest_path, index=False)
        combined_manifest.to_csv(latest_manifest_path, index=False)

        self.last_split_manifest_path = latest_manifest_path
        print(f"Saved run split manifest: {run_manifest_path}")
        print(f"Saved latest split manifest: {latest_manifest_path}")
        return latest_manifest_path

    def load_and_process_dataset(
        self, max_samples: Optional[int] = None
    ) -> Tuple[
        Dict[str, Tuple[List[np.ndarray], List[int]]],
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
    ]:
        """
        End-to-end preprocessing with split-before-segmentation protocol.

        Returns:
            split_datasets:
                {'train': (windows, labels), 'val': (...), 'test': (...)}
            split_manifests:
                record-level manifest for each split
            split_window_manifests:
                window-level manifest for each split, aligned to the saved arrays
        """
        records_df = self._build_record_metadata(max_samples=max_samples)
        split_tables = self._split_record_metadata(records_df)

        split_datasets: Dict[str, Tuple[List[np.ndarray], List[int]]] = {}
        split_manifests: Dict[str, pd.DataFrame] = {}
        split_window_manifests: Dict[str, pd.DataFrame] = {}

        for split_name, split_df in split_tables.items():
            (
                split_data,
                split_labels,
                manifest_df,
                window_manifest_df,
            ) = self._process_split_records(split_df, split_name)
            split_datasets[split_name] = (split_data, split_labels)
            split_manifests[split_name] = manifest_df
            split_window_manifests[split_name] = window_manifest_df

        return split_datasets, split_manifests, split_window_manifests

    def save_processed_data(
        self,
        split_datasets: Dict[str, Tuple[List[np.ndarray], List[int]]],
        split_manifests: Optional[Dict[str, pd.DataFrame]] = None,
        split_window_manifests: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Save already-split arrays and optional record/window manifests."""
        for split_name, (data, labels) in split_datasets.items():
            if len(data) == 0:
                raise ValueError(f"No data available for split '{split_name}'")

            data_array = np.array(data)
            labels_array = np.array(labels)

            print(
                f"{split_name} data shape: {data_array.shape} "
                f"(normal={np.sum(labels_array == 0)}, abnormal={np.sum(labels_array == 1)})"
            )

            np.save(os.path.join(self.processed_data_path, f"X_{split_name}.npy"), data_array)
            np.save(os.path.join(self.processed_data_path, f"y_{split_name}.npy"), labels_array)

            if split_manifests and split_name in split_manifests:
                manifest_path = os.path.join(
                    self.processed_data_path, f"{split_name}_manifest.csv"
                )
                split_manifests[split_name].to_csv(manifest_path, index=False)
                print(f"Saved {split_name} manifest: {manifest_path}")

            if split_window_manifests and split_name in split_window_manifests:
                window_manifest_path = os.path.join(
                    self.processed_data_path, f"{split_name}_window_manifest.csv"
                )
                split_window_manifests[split_name].to_csv(
                    window_manifest_path, index=False
                )
                print(
                    f"Saved {split_name} window manifest: {window_manifest_path}"
                )

        self._save_split_manifest(split_manifests or {})

        protocol_path = os.path.join(self.processed_data_path, "split_protocol.txt")
        with open(protocol_path, "w", encoding="utf-8") as handle:
            handle.write(
                "Split-before-segmentation protocol\n"
                f"Split level: {self.last_split_level}\n"
                "All windows from a source record stay in exactly one split.\n"
            )
            if self.last_split_manifest_path:
                handle.write(f"Combined split manifest: {self.last_split_manifest_path}\n")
        print(f"Saved protocol note: {protocol_path}")

    def _reconstruct_window_manifest(
        self, split: str, record_manifest: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reconstruct window-level provenance from a record-level manifest.

        This supports backward compatibility for older processed splits that
        saved `num_segments` per record but did not persist one row per window.
        """
        processed_records = record_manifest.copy()
        if "processing_status" in processed_records.columns:
            processed_records = processed_records[
                processed_records["processing_status"] == "processed"
            ]
        elif "num_segments" in processed_records.columns:
            processed_records = processed_records[
                processed_records["num_segments"].fillna(0).astype(int) > 0
            ]

        window_rows = []
        sample_index = 0
        step_size = self.signal_length // 2

        for _, row in processed_records.iterrows():
            num_segments = int(row.get("num_segments", 0) or 0)
            for window_index_within_record in range(num_segments):
                segment_start_sample = window_index_within_record * step_size
                window_rows.append(
                    {
                        "window_id": f"{split}_{sample_index:07d}",
                        "sample_index_in_split": sample_index,
                        "window_index_within_record": window_index_within_record,
                        "source_record_id": int(row["ecg_id"]),
                        "patient_id": row.get("patient_id", np.nan),
                        "split": split,
                        "label": int(row["label"]),
                        "filename_lr": row.get("filename_lr", None),
                        "filename_hr": row.get("filename_hr", None),
                        "scp_codes": row.get("scp_codes", None),
                        "record_path": row.get("record_path", None),
                        "segment_start_sample": segment_start_sample,
                        "segment_end_sample": segment_start_sample
                        + self.signal_length,
                        "manifest_source": "reconstructed_from_record_manifest",
                    }
                )
                sample_index += 1

        return pd.DataFrame(window_rows)

    def load_window_manifest(
        self, split: str, expected_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load per-window provenance for one split.

        If a window-level manifest is missing but the older record-level
        manifest is available, reconstruct a compatible table from
        `num_segments` so evaluation can still write prediction artifacts.
        """
        window_manifest_path = os.path.join(
            self.processed_data_path, f"{split}_window_manifest.csv"
        )
        if os.path.exists(window_manifest_path):
            window_manifest = pd.read_csv(window_manifest_path)
        else:
            record_manifest_path = os.path.join(
                self.processed_data_path, f"{split}_manifest.csv"
            )
            if not os.path.exists(record_manifest_path):
                raise FileNotFoundError(
                    "Window manifest not found and record-level manifest is unavailable: "
                    f"{window_manifest_path}"
                )

            record_manifest = pd.read_csv(record_manifest_path)
            window_manifest = self._reconstruct_window_manifest(split, record_manifest)
            if not window_manifest.empty:
                window_manifest.to_csv(window_manifest_path, index=False)
                print(
                    "Reconstructed window manifest from record manifest: "
                    f"{window_manifest_path}"
                )

        if expected_samples is not None and len(window_manifest) != expected_samples:
            raise ValueError(
                f"Window manifest length mismatch for split '{split}': "
                f"expected {expected_samples}, found {len(window_manifest)}"
            )

        return window_manifest

    def load_processed_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed arrays for one split."""
        x_path = os.path.join(self.processed_data_path, f"X_{split}.npy")
        y_path = os.path.join(self.processed_data_path, f"y_{split}.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            old_path = os.path.join(self.processed_data_path, f"{split}_data.pkl")
            if os.path.exists(old_path):
                with open(old_path, "rb") as handle:
                    data, labels = pickle.load(handle)
                return np.array(data), np.array(labels)
            raise FileNotFoundError(
                f"Processed data files not found: {x_path} or {y_path}"
            )

        data = np.load(x_path)
        labels = np.load(y_path)
        return data, labels

    def process_and_save_data(
        self, output_dir: Optional[str] = None, max_samples: Optional[int] = None
    ):
        """Run the full split-before-segmentation preprocessing pipeline."""
        if output_dir is not None and output_dir != self.processed_data_path:
            self.processed_data_path = output_dir
        os.makedirs(self.processed_data_path, exist_ok=True)

        split_datasets, split_manifests, split_window_manifests = (
            self.load_and_process_dataset(max_samples=max_samples)
        )

        if not split_datasets:
            raise ValueError("No records were processed successfully")

        self.save_processed_data(
            split_datasets,
            split_manifests,
            split_window_manifests,
        )
        print(
            "Data preprocessing completed using "
            f"{self.last_split_level}-level split-before-segmentation."
        )


def main():
    """CLI entry point for PTB-XL preprocessing."""
    print("Starting PTB-XL preprocessing...")
    data_loader = PTBDataLoader()

    if not os.path.exists(data_loader.raw_data_path):
        print(f"Warning: raw data path does not exist: {data_loader.raw_data_path}")
        print("Please place the PTB-XL files under that directory first.")
        return

    try:
        data_loader.process_and_save_data(max_samples=None)
    except Exception as exc:
        print(f"Data processing failed: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
