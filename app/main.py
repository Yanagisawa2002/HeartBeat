from __future__ import annotations

from pathlib import Path
import os
import json

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .schemas import (
    AvailableModelResponse,
    DemoConfigResponse,
    PredictRequest,
    PredictResponse,
    SampleInputResponse,
)
from src.inference import (
    get_signal_spec,
    list_available_models,
    predict_from_csv_text,
    predict_single_window,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "sample_inputs"
SAMPLE_MANIFEST_PATH = SAMPLE_INPUTS_DIR / "manifest.json"
DEFAULT_RESULTS_CSV_CANDIDATES = [
    PROJECT_ROOT / "results" / "full_benchmark_all_models_20260331" / "model_comparison_results.csv",
    PROJECT_ROOT / "results" / "comparison" / "model_comparison_results.csv",
]
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title="HeartBeat Inference Demo",
    description=(
        "Web demo for fixed-window 12-lead ECG inference on the HeartBeat "
        "benchmark models."
    ),
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount(
    "/sample-inputs",
    StaticFiles(directory=str(SAMPLE_INPUTS_DIR)),
    name="sample-inputs",
)


def _default_model_name() -> str | None:
    available_models = list_available_models()
    available_names = {model.name for model in available_models}

    configured_default = os.getenv("HEARTBEAT_DEFAULT_MODEL")
    if configured_default and configured_default.lower() in available_names:
        return configured_default.lower()

    benchmark_summary = _load_benchmark_summary()
    if benchmark_summary and benchmark_summary["best_model"] in available_names:
        return benchmark_summary["best_model"]

    if available_models:
        return available_models[0].name
    return None


def _prediction_error_response(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def _load_sample_manifest() -> list[dict]:
    if not SAMPLE_MANIFEST_PATH.exists():
        return []

    with open(SAMPLE_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Sample manifest must be a JSON list.")

    records = []
    for record in payload:
        file_name = record.get("file_name")
        if not file_name:
            continue
        if not (SAMPLE_INPUTS_DIR / file_name).exists():
            continue
        records.append(record)
    return records


def _resolve_results_csv_path() -> Path | None:
    configured = os.getenv("HEARTBEAT_RESULTS_CSV")
    if configured:
        path = Path(configured).resolve()
        if path.exists():
            return path

    for candidate in DEFAULT_RESULTS_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_benchmark_summary() -> dict | None:
    results_csv = _resolve_results_csv_path()
    if results_csv is None:
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    df = pd.read_csv(results_csv)
    required_columns = {"Model", "Accuracy", "F1 Score", "AUC Score", "Parameters", "Inference Time (s)"}
    if not required_columns.issubset(df.columns):
        return None

    best_row = df.loc[df["Accuracy"].idxmax()]
    practical_candidates = df[df["Accuracy"] >= df["Accuracy"].max() - 0.01].copy()
    if practical_candidates.empty:
        practical_candidates = df.copy()
    practical_row = practical_candidates.sort_values(
        by=["Parameters", "Inference Time (s)"],
        ascending=[True, True],
    ).iloc[0]

    return {
        "results_csv": str(results_csv),
        "best_model": str(best_row["Model"]).lower(),
        "best_accuracy": float(best_row["Accuracy"]),
        "best_auc": float(best_row["AUC Score"]),
        "practical_model": str(practical_row["Model"]).lower(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    signal_spec = get_signal_spec(os.getenv("HEARTBEAT_CONFIG_PATH"))
    benchmark_summary = _load_benchmark_summary()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "project_name": "HeartBeat",
            "signal_length": signal_spec["signal_length"],
            "sampling_rate": signal_spec["sampling_rate"],
            "leads": signal_spec["leads"] or [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ],
            "benchmark_summary": benchmark_summary,
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", response_model=list[AvailableModelResponse])
def models() -> list[AvailableModelResponse]:
    return [AvailableModelResponse(**model.__dict__) for model in list_available_models()]


@app.get("/samples", response_model=list[SampleInputResponse])
def samples() -> list[SampleInputResponse]:
    return [SampleInputResponse(**record) for record in _load_sample_manifest()]


@app.get("/demo-config", response_model=DemoConfigResponse)
def demo_config() -> DemoConfigResponse:
    signal_spec = get_signal_spec(os.getenv("HEARTBEAT_CONFIG_PATH"))
    return DemoConfigResponse(
        num_leads=signal_spec["num_leads"],
        signal_length=signal_spec["signal_length"],
        sampling_rate=signal_spec["sampling_rate"],
        leads=signal_spec["leads"] or [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        default_model=_default_model_name(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_single_window(
            ecg=np.asarray(payload.ecg, dtype=np.float32),
            model_name=payload.model_name,
            config_path=os.getenv("HEARTBEAT_CONFIG_PATH"),
            device=os.getenv("HEARTBEAT_DEVICE", "cpu"),
        )
    except Exception as exc:
        raise _prediction_error_response(exc) from exc

    return PredictResponse(**result)


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(
    model_name: str = Form(...),
    file: UploadFile = File(...),
) -> PredictResponse:
    try:
        csv_text = (await file.read()).decode("utf-8")
        result = predict_from_csv_text(
            csv_text=csv_text,
            model_name=model_name,
            config_path=os.getenv("HEARTBEAT_CONFIG_PATH"),
            device=os.getenv("HEARTBEAT_DEVICE", "cpu"),
        )
    except Exception as exc:
        raise _prediction_error_response(exc) from exc

    return PredictResponse(**result)
