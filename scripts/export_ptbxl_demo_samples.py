from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "sample_inputs"
DEFAULT_EXPORTS = [
    {
        "ecg_id": 1,
        "output_name": "ptbxl_normal_ecg_00001.csv",
        "label": "normal",
    },
    {
        "ecg_id": 17,
        "output_name": "ptbxl_abnormal_ecg_00017.csv",
        "label": "abnormal",
    },
]


def resolve_ptbxl_root() -> Path:
    candidates = [
        PROJECT_ROOT / "ptb-xl",
        PROJECT_ROOT / "data" / "raw" / "ptb-xl",
        PROJECT_ROOT / "data" / "raw" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    ]
    for candidate in candidates:
        if (candidate / "ptbxl_database.csv").exists():
            return candidate
    raise FileNotFoundError("Unable to locate PTB-XL data root.")


def parse_scp_codes(value: str) -> dict:
    if not isinstance(value, str):
        return {}
    try:
        return ast.literal_eval(value)
    except Exception:
        return {}


def export_demo_samples() -> None:
    data_root = resolve_ptbxl_root()
    metadata = pd.read_csv(data_root / "ptbxl_database.csv")
    metadata = metadata.set_index("ecg_id")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for item in DEFAULT_EXPORTS:
        row = metadata.loc[item["ecg_id"]]
        record_path = data_root / str(row["filename_lr"])
        signal, _ = wfdb.rdsamp(str(record_path))
        output_path = OUTPUT_DIR / item["output_name"]
        pd.DataFrame(signal.T).to_csv(output_path, header=False, index=False, float_format="%.6f")
        print(f"Exported ecg_id={item['ecg_id']} -> {output_path}")


if __name__ == "__main__":
    export_demo_samples()
