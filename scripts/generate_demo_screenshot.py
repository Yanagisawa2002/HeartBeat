from __future__ import annotations

from pathlib import Path
import textwrap
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import list_available_models, predict_single_window

OUTPUT_PATH = PROJECT_ROOT / "docs" / "images" / "web_demo_homepage.png"
RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "full_benchmark_all_models_20260331"
    / "model_comparison_results.csv"
)
SAMPLE_PATH = PROJECT_ROOT / "sample_inputs" / "ptbxl_normal_ecg_00001.csv"
SAMPLE_MANIFEST_PATH = PROJECT_ROOT / "sample_inputs" / "manifest.json"
MODEL_NAME = "inception1d"

WIDTH = 1600
HEIGHT = 1280

BG = "#f3efe6"
PANEL = "#fffdf8"
BORDER = "#d9d4c8"
TEXT = "#1a2230"
MUTED = "#607080"
ACCENT = "#1b6a68"
ACCENT_SOFT = "#d5ece6"
WARNING = "#b45a35"
WARNING_SOFT = "#f4dfd7"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


FONT_H1 = load_font(54, bold=True)
FONT_H2 = load_font(28, bold=True)
FONT_H3 = load_font(22, bold=True)
FONT_BODY = load_font(20)
FONT_SMALL = load_font(16)
FONT_LABEL = load_font(15, bold=True)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1, radius: int = 26) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, fill: str, width_chars: int, line_spacing: int = 6) -> None:
    lines = textwrap.wrap(text, width=width_chars)
    draw.multiline_text(xy, "\n".join(lines), font=font, fill=fill, spacing=line_spacing)


def read_benchmark_summary() -> dict[str, str | float]:
    df = pd.read_csv(RESULTS_CSV)
    best = df.loc[df["Accuracy"].idxmax()]
    return {
        "best_model": str(best["Model"]).upper(),
        "accuracy": float(best["Accuracy"]),
        "auc": float(best["AUC Score"]),
    }


def read_sample_metadata(file_name: str) -> dict[str, str]:
    with open(SAMPLE_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    for sample in samples:
        if sample.get("file_name") == file_name:
            return sample
    return {
        "name": file_name,
        "description": "Bundled demo input",
        "source": "sample-inputs",
    }


def load_demo_state() -> tuple[np.ndarray, dict, list[str], dict[str, str | float], dict[str, str]]:
    ecg = np.loadtxt(SAMPLE_PATH, delimiter=",", dtype=np.float32)
    prediction = predict_single_window(ecg=ecg, model_name=MODEL_NAME, device="cpu")
    models = [model.name for model in list_available_models()]
    summary = read_benchmark_summary()
    sample_meta = read_sample_metadata(SAMPLE_PATH.name)
    return ecg, prediction, models, summary, sample_meta


def draw_waveform(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ecg: np.ndarray) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    rows = ecg.shape[0]
    row_h = height / rows
    plot_left = left + 70
    plot_right = right - 24
    plot_top = top + 10
    plot_bottom = bottom - 10
    plot_height = plot_bottom - plot_top
    row_h = plot_height / rows

    draw.rounded_rectangle((left, top, right, bottom), radius=24, fill="#fffdf9", outline="#ece7de", width=1)

    minor_vertical = 24
    major_vertical = minor_vertical * 5
    x = plot_left
    while x <= plot_right:
        is_major = (x - plot_left) % major_vertical == 0
        color = "#d6dee6" if is_major else "#eef3f7"
        draw.line((x, plot_top, x, plot_bottom), fill=color, width=1)
        x += minor_vertical

    for idx in range(rows):
        row_top = plot_top + row_h * idx
        center_y = row_top + row_h / 2
        minor_horizontal = row_h / 5

        for step in range(6):
            y = row_top + step * minor_horizontal
            color = "#d6dee6" if step in (0, 5) else "#eef3f7"
            draw.line((plot_left, y, plot_right, y), fill=color, width=1)

        draw.line((plot_left, center_y, plot_right, center_y), fill="#d8e2df", width=1)
        draw.text((left + 14, center_y - 10), leads[idx], font=FONT_SMALL, fill=MUTED)

        signal = ecg[idx]
        max_abs = max(float(np.max(np.abs(signal))), 1e-6)
        y_scale = (row_h * 0.32) / max_abs

        points: list[tuple[float, float]] = []
        for sample_idx, value in enumerate(signal):
            x = plot_left + (sample_idx / max(len(signal) - 1, 1)) * (plot_right - plot_left)
            y = center_y - float(value) * y_scale
            points.append((x, y))
        draw.line(points, fill=ACCENT, width=2)


def create_demo_image() -> Path:
    ecg, prediction, models, summary, sample_meta = load_demo_state()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    # Subtle background glow
    draw.ellipse((-220, -160, 520, 420), fill="#e0eee8")
    draw.ellipse((1080, -120, 1760, 360), fill="#e4ebf4")

    hero = (40, 36, 1560, 336)
    left_panel = (40, 368, 760, 784)
    right_panel = (800, 368, 1560, 784)
    waveform_panel = (40, 816, 1560, 1236)

    rounded(draw, hero, PANEL, outline=BORDER, width=2, radius=32)
    rounded(draw, left_panel, PANEL, outline=BORDER, width=2)
    rounded(draw, right_panel, PANEL, outline=BORDER, width=2)
    rounded(draw, waveform_panel, PANEL, outline=BORDER, width=2)

    # Hero content
    draw.text((76, 72), "Dockerized Inference Demo", font=FONT_LABEL, fill=ACCENT)
    draw.text((76, 104), "HeartBeat ECG Benchmark Demo", font=FONT_H1, fill=TEXT)
    wrapped(
        draw,
        "Browser-based inference demo for the repository's fixed-window 12-lead ECG classification models. This preview uses a bundled PTB-XL example window and the latest Inception1D checkpoint.",
        (76, 182),
        FONT_BODY,
        MUTED,
        width_chars=52,
    )

    pill_y = 270
    pills = ["12 leads", "1000 samples per window", "100 Hz preprocessing config"]
    pill_x = 76
    for pill in pills:
        bbox = draw.textbbox((0, 0), pill, font=FONT_LABEL)
        pill_w = bbox[2] - bbox[0] + 28
        rounded(draw, (pill_x, pill_y, pill_x + pill_w, pill_y + 38), ACCENT_SOFT, radius=18)
        draw.text((pill_x + 14, pill_y + 10), pill, font=FONT_LABEL, fill=ACCENT)
        pill_x += pill_w + 10

    note_box = (1030, 72, 1516, 300)
    rounded(draw, note_box, "#ffffff", outline="#e3e0d7", width=1, radius=24)
    draw.text((1060, 104), "Latest Full Benchmark", font=FONT_H3, fill=TEXT)
    wrapped(
        draw,
        f"{summary['best_model']} reached accuracy {summary['accuracy']:.4f} and AUC {summary['auc']:.4f}. The demo defaults to this model when its checkpoint is available.",
        (1060, 142),
        FONT_BODY,
        MUTED,
        width_chars=28,
    )
    rounded(draw, (1058, 232, 1488, 284), "#eef7f4", outline=None, radius=18)
    draw.text((1080, 248), "All five benchmark models are bundled into the demo image.", font=FONT_SMALL, fill=ACCENT)

    # Input panel
    draw.text((68, 392), "Input", font=FONT_LABEL, fill=ACCENT)
    draw.text((68, 420), "Select or Upload ECG Window", font=FONT_H2, fill=TEXT)

    field_y = 492
    for label, value in [
        ("Model", prediction["model_name"]),
        ("Built-in sample", sample_meta["name"]),
        ("CSV file upload", "No file selected"),
    ]:
        draw.text((68, field_y), label, font=FONT_LABEL, fill=TEXT)
        rounded(draw, (68, field_y + 28, 732, field_y + 84), "#ffffff", outline="#d7d7d7", radius=18)
        draw.text((86, field_y + 46), value, font=FONT_BODY, fill=TEXT)
        field_y += 106

    rounded(draw, (68, 744, 252, 788), ACCENT, radius=18)
    draw.text((104, 756), "Run Inference", font=FONT_LABEL, fill="#ffffff")
    draw.text((280, 756), f"Ready. {len(models)} models and 4 bundled samples available.", font=FONT_SMALL, fill=MUTED)

    # Output panel
    draw.text((828, 392), "Output", font=FONT_LABEL, fill=ACCENT)
    draw.text((828, 420), "Prediction Summary", font=FONT_H2, fill=TEXT)

    rounded(draw, (828, 486, 1532, 752), "#fffaf5", outline="#eee1d5", width=1, radius=24)
    draw.text((858, 520), "Predicted label", font=FONT_SMALL, fill=MUTED)
    draw.text((858, 548), str(prediction["predicted_label"]).upper(), font=FONT_H1, fill=TEXT)

    chip_fill = WARNING_SOFT if prediction["predicted_class"] == 1 else ACCENT_SOFT
    chip_text = WARNING if prediction["predicted_class"] == 1 else ACCENT
    rounded(draw, (1280, 534, 1498, 582), chip_fill, radius=20)
    draw.text(
        (1304, 550),
        f"{prediction['probability_abnormal'] * 100:.1f}% abnormal",
        font=FONT_LABEL,
        fill=chip_text,
    )

    metrics = [
        ("Model", str(prediction["model_name"])),
        ("Abnormal probability", f"{prediction['probability_abnormal']:.4f}"),
        ("Normal probability", f"{prediction['probability_normal']:.4f}"),
        ("Input shape", " x ".join(str(v) for v in prediction["input_shape"])),
    ]
    mx, my = 858, 620
    card_w, card_h = 314, 86
    for idx, (label, value) in enumerate(metrics):
        col = idx % 2
        row = idx // 2
        x0 = mx + col * (card_w + 18)
        y0 = my + row * (card_h + 16)
        rounded(draw, (x0, y0, x0 + card_w, y0 + card_h), "#ffffff", outline="#ece7de", width=1, radius=18)
        draw.text((x0 + 18, y0 + 16), label, font=FONT_SMALL, fill=MUTED)
        draw.text((x0 + 18, y0 + 42), value, font=FONT_H3, fill=TEXT)

    # Waveform panel
    draw.text((68, 840), "Visualization", font=FONT_LABEL, fill=ACCENT)
    draw.text((68, 868), "Waveform Preview", font=FONT_H2, fill=TEXT)
    draw.text(
        (68, 910),
        "Real PTB-XL example waveform shown as it appears in the demo input preview. The inference above uses the repository's standard preprocessing pipeline.",
        font=FONT_BODY,
        fill=MUTED,
    )
    draw_waveform(draw, (60, 954, 1540, 1214), ecg)

    image.save(OUTPUT_PATH, format="PNG")
    return OUTPUT_PATH


if __name__ == "__main__":
    output = create_demo_image()
    print(f"Saved demo screenshot to {output}")
