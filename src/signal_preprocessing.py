from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np
from scipy import signal


def ensure_lead_first_window(
    ecg: np.ndarray,
    expected_leads: int = 12,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize a single ECG window into lead-first shape ``(leads, length)``.

    The inference demo accepts either ``(12, length)`` or ``(length, 12)``
    numeric input. Any other layout is rejected to keep the deployment scope
    aligned with the benchmark task.
    """
    ecg_array = np.asarray(ecg, dtype=np.float32)

    if ecg_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D ECG array, got shape {tuple(ecg_array.shape)}"
        )

    if ecg_array.shape[0] == expected_leads:
        normalized = ecg_array
    elif ecg_array.shape[1] == expected_leads:
        normalized = ecg_array.T
    else:
        raise ValueError(
            "ECG input must contain exactly "
            f"{expected_leads} leads in either rows or columns; "
            f"received shape {tuple(ecg_array.shape)}"
        )

    if expected_length is not None and normalized.shape[1] != expected_length:
        raise ValueError(
            f"Expected ECG window length {expected_length}, "
            f"received {normalized.shape[1]}"
        )

    return normalized.astype(np.float32, copy=False)


def preprocess_ecg_signals(
    signals: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    """
    Apply the repository's per-lead ECG preprocessing pipeline.

    The steps mirror the benchmark loader:
    - high-pass filter for baseline drift
    - low-pass filter for high-frequency noise
    - optional notch-style band-stop filtering when sampling rate permits
    - per-lead standardization
    """
    normalized = ensure_lead_first_window(signals)
    processed_signals = []

    for lead_signal in normalized:
        sos_hp = signal.butter(
            4,
            0.5,
            btype="high",
            fs=sampling_rate,
            output="sos",
        )
        filtered_signal = signal.sosfilt(sos_hp, lead_signal)

        nyquist = sampling_rate / 2
        cutoff_freq = min(40, nyquist - 1)
        sos_lp = signal.butter(
            4,
            cutoff_freq,
            btype="low",
            fs=sampling_rate,
            output="sos",
        )
        filtered_signal = signal.sosfilt(sos_lp, filtered_signal)

        if sampling_rate > 120:
            sos_notch = signal.butter(
                4,
                [49, 51],
                btype="bandstop",
                fs=sampling_rate,
                output="sos",
            )
            filtered_signal = signal.sosfilt(sos_notch, filtered_signal)

        signal_std = np.std(filtered_signal)
        if signal_std > 1e-8:
            filtered_signal = (
                filtered_signal - np.mean(filtered_signal)
            ) / signal_std
        else:
            filtered_signal = filtered_signal - np.mean(filtered_signal)

        processed_signals.append(filtered_signal)

    return np.array(processed_signals, dtype=np.float32)


def parse_ecg_csv_text(
    csv_text: str,
    expected_leads: int = 12,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """
    Parse a numeric CSV string into a validated ECG window.

    The CSV must contain plain numeric values only. The resulting array is
    returned in lead-first shape ``(leads, length)``.
    """
    try:
        ecg_array = np.loadtxt(StringIO(csv_text), delimiter=",", dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Unable to parse ECG CSV as numeric data.") from exc

    if ecg_array.ndim != 2:
        raise ValueError(
            "ECG CSV must contain a 2D numeric matrix with leads and samples."
        )

    return ensure_lead_first_window(
        ecg_array,
        expected_leads=expected_leads,
        expected_length=expected_length,
    )
