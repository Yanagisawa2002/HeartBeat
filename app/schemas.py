from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AvailableModelResponse(BaseModel):
    name: str
    checkpoint_path: str
    source_dir: str


class DemoConfigResponse(BaseModel):
    num_leads: int
    signal_length: int
    sampling_rate: int
    leads: List[str]
    default_model: Optional[str] = None


class SampleInputResponse(BaseModel):
    name: str
    file_name: str
    description: str
    source: str


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Registered benchmark model name.")
    ecg: List[List[float]] = Field(
        ...,
        description="Numeric ECG matrix in 12 x length or length x 12 format.",
    )


class PredictResponse(BaseModel):
    model_name: str
    predicted_class: int
    predicted_label: str
    probability_normal: float
    probability_abnormal: float
    input_shape: List[int]
    preprocessing_applied: bool
    checkpoint_path: str


class PredictAllRequest(BaseModel):
    ecg: List[List[float]] = Field(
        ...,
        description="Numeric ECG matrix in 12 x length or length x 12 format.",
    )
    model_names: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of registered model names to compare.",
    )


class PredictAllResponse(BaseModel):
    predictions: List[PredictResponse]
