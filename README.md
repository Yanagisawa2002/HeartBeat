# Normal-vs-Abnormal 12-Lead ECG Classification on PTB-XL

This repository documents an exploratory medical machine learning benchmark on a focused supervised ECG classification task: distinguishing **normal** from **abnormal** 12-lead ECG recordings from **PTB-XL** using four baseline deep learning models.

The project is intended as a **transparent benchmark comparison**, not as a deployable clinical diagnostic system. Its value is in the problem framing, preprocessing pipeline, baseline modeling, reproducible command-line workflow, committed result artifacts, and explicit discussion of what the current experiment does and does not support.

Useful result artifacts included in this repository:

- [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)
- [results/comparison/model_comparison_results.csv](results/comparison/model_comparison_results.csv)
- [results/visualization/comprehensive_table.png](results/visualization/comprehensive_table.png)

## Problem Statement

**Task:** given a preprocessed fixed-length **12-lead ECG window** derived from a PTB-XL recording, predict whether it should be treated as **normal** or **abnormal** under the repository's binary labeling rule.

- **Input:** 12-lead ECG signal window
- **Output:** binary label
  - `0`: normal
  - `1`: abnormal
- **Models compared:** CNN1D, LSTM, ResNet1D, Hybrid CNN-LSTM
- **Primary goal:** compare discrimination performance and computational trade-offs across standard neural sequence models on a public biomedical signal dataset

This is a **binary ECG classification benchmark**, not an unsupervised anomaly detector, not a multi-label ECG interpretation system, and not a claim of clinical readiness.

## Why This Task Matters

ECGs are one of the most common cardiac tests in routine care. Even a coarse normal-versus-abnormal classification task is useful for studying how model architecture, preprocessing, and computational cost affect performance on real biomedical waveforms.

At the same time, this task is much simpler than real clinical ECG interpretation. A model that performs well here still does **not** provide a diagnosis, estimate uncertainty, or replace clinician review.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── configs/
├── data/
├── docs/
├── notebooks/
├── results/
├── scripts/
└── src/
```

- [src/](src): core data loading, model definitions, benchmark workflow, and CLI
- [scripts/](scripts): setup, visualization, and supplementary evaluation helpers
- [docs/](docs): supplementary benchmark notes
- [notebooks/](notebooks): exploratory notebook material
- [configs/](configs): committed runtime configuration
- [results/](results): selected committed summary artifacts

## Core Files

- [src/data_loader.py](src/data_loader.py): PTB-XL loading, preprocessing, labeling, segmentation, and split generation
- [src/comparison_models.py](src/comparison_models.py): model definitions for CNN1D, LSTM, ResNet1D, and Hybrid CNN-LSTM
- [src/benchmark.py](src/benchmark.py): training and evaluation workflow for the baseline models
- [src/cli.py](src/cli.py): command-line entry point for preprocessing, training, and evaluation
- [scripts/visualize_all_models.py](scripts/visualize_all_models.py): visualization script for the comparison CSV
- [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md): supplementary benchmark note based on the committed summary results

## Command-Line Workflow

The repository now exposes a simple module-based CLI:

```bash
python -m src --help
```

Main entry points:

- `python -m src preprocess`: preprocess PTB-XL and save `train` / `val` / `test` arrays
- `python -m src train`: train one or more baseline models and save checkpoints
- `python -m src evaluate`: evaluate saved checkpoints on the processed test split

Exact example commands:

```bash
pip install -r requirements.txt
python -m src preprocess --config configs/config.yaml
python -m src train --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
python -m src evaluate --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
python scripts/visualize_all_models.py
```

Useful optional overrides:

```bash
python -m src preprocess --config configs/config.yaml --max-samples 200
python -m src train --config configs/config.yaml --models cnn1d lstm --epochs 20 --batch-size 16
python -m src evaluate --config configs/config.yaml --models cnn1d lstm --results-dir results/comparison
```

## Experimental Protocol

### Protocol Revision

Earlier versions of this repository used a weaker preprocessing protocol. The current pipeline now assigns train/validation/test splits at the source-record level before any window segmentation, so overlapping windows from the same ECG recording cannot appear in multiple splits. The previous label-contaminating imbalance heuristic has also been removed; class imbalance is now handled with label-preserving weighting during training. This revision improves the rigor and auditability of the benchmark, but it does not remove broader limitations such as single-dataset evaluation and the absence of external validation.

### Dataset and Task

- **Dataset source:** PTB-XL from PhysioNet
- **Signal type:** 12-lead ECG
- **Implemented input path:** the loader reads the low-resolution PTB-XL waveform files through the `records100` directory
- **Task definition:** supervised binary classification of fixed-length ECG windows derived from PTB-XL recordings
- **Prediction target:** `0 = normal`, `1 = abnormal`

Signals are filtered with high-pass and low-pass filtering, optional notch filtering when the sampling rate is high enough, and per-lead standardization before any window extraction.

### Label Definition

The repository implements a record-level binary label derived from PTB-XL `scp_codes`:

- A record is labeled **normal** if `NORM >= 50` and there is no other non-`SR` code with confidence `>= 50`
- Otherwise the record is labeled **abnormal**

All windows from a source record inherit that record-level binary label. This creates a coarse normal-vs-abnormal endpoint rather than a disease-specific ECG diagnosis task.

### Split Strategy

- **Split order:** split first, then segment
- **Preferred grouping:** patient-level splitting when `patient_id` is available in the PTB-XL metadata
- **Fallback:** record-level splitting when patient identifiers are unavailable or incomplete
- **Windowing:** segmentation happens only after split assignment, using fixed-length windows with 50% overlap
- **Leakage control:** all windows from the same source record remain in exactly one split

This is intended to prevent leakage from correlated windows originating from the same recording.

### Imbalance Handling

Class imbalance is handled during training with **inverse-frequency class weighting** computed from the training split only and applied through weighted cross-entropy loss. The current pipeline does **not** relabel samples or move normal records into the abnormal class.

### Evaluation Metrics

The benchmark workflow reports:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1 score
- ROC-AUC based on the abnormal-class probability
- Parameter count
- Inference time
- Training time when models are trained through the current CLI workflow
- Prediction-level outputs for the test split
- Confusion matrix, precision-recall curve, ROC curve, threshold sweep, and per-class metrics

### Saved Reproducibility Artifacts

The current code saves several artifacts intended to make the benchmark auditable:

- processed arrays under `data/processed/`, including `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, and `y_test.npy`
- per-split record manifests and window manifests under `data/processed/`
- a combined split manifest under `results/comparison/preprocessing/`, including `split_manifest_latest.csv` and timestamped per-run manifests
- a benchmark summary table in [results/comparison/model_comparison_results.csv](results/comparison/model_comparison_results.csv)
- prediction-level test outputs under [results/comparison/predictions/](results/comparison/predictions), for example `cnn1d_test_predictions.csv`
- per-model evaluation artifacts under [results/comparison/evaluation/](results/comparison/evaluation), including machine-readable CSVs and figures for confusion matrices, ROC/PR curves, threshold sweeps, and per-class metrics

The canonical command-line workflow is:

```bash
python -m src preprocess --config configs/config.yaml
python -m src train --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
python -m src evaluate --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
```

## Key Results

The committed comparison artifact in [results/comparison/model_comparison_results.csv](results/comparison/model_comparison_results.csv) reports the following model-level results:

| Model | Accuracy | F1 Score | AUC Score | Parameters | Inference Time (s) |
|-------|----------|----------|-----------|------------|--------------------|
| CNN1D | 0.9369 | 0.9369 | 0.9808 | 705,218 | 0.796 |
| LSTM | **0.9412** | **0.9414** | **0.9849** | 903,298 | 5.514 |
| ResNet1D | 0.8937 | 0.8943 | 0.9579 | 3,849,858 | 1.769 |
| Hybrid CNN-LSTM | 0.9151 | 0.9152 | 0.9716 | 1,035,458 | 0.867 |

In the committed benchmark snapshot, **LSTM** achieved the strongest discrimination performance across the reported classification metrics, while **CNN1D** provided the best practical efficiency trade-off in terms of parameter count and reported inference time. **ResNet1D** was the largest model and the weakest performer in this comparison, and **Hybrid CNN-LSTM** occupied a middle position without clearly outperforming the simpler CNN1D baseline.

Visual summary:

![](./results/visualization/comprehensive_table.png)

I treat the discrimination metrics above as the most useful summary. Some training-time fields in the repository are inconsistent across scripts, so they should be interpreted more cautiously than accuracy, F1, or AUC.

## Limitations

This repository is best read as a **baseline comparison study**, with several important limitations:

- **Single-dataset evaluation:** all benchmarks are derived from PTB-XL only, so robustness across institutions, devices, acquisition settings, and patient populations is not established.
- **Simplified binary framing:** the implemented endpoint is a coarse normal-versus-abnormal classification task based on SCP-code rules, not disease-specific ECG diagnosis or full clinical interpretation.
- **Window labels inherit record labels:** each extracted window receives the binary label of its source record, which is practical for benchmarking but does not guarantee that abnormal morphology is present in every labeled abnormal window.
- **No external validation:** there is no independent hospital cohort, temporal validation, or cross-dataset replication in the current repository.
- **Generalization remains uncertain:** the revised split-before-windowing protocol improves internal validity, but it does not by itself demonstrate out-of-distribution performance or stability under dataset shift.
- **No clinical deployment claim:** the repository does not evaluate calibration, uncertainty, subgroup performance, clinical workflow integration, or prospective utility, and should be interpreted as a research benchmark rather than a deployable medical system.

Because of these limitations, the current results should be interpreted as **exploratory benchmark results**, not as evidence of clinical deployment readiness.

## Reproducibility

### What is included

- dependency list in [requirements.txt](requirements.txt)
- a default runtime config in [configs/config.yaml](configs/config.yaml)
- model definitions in [src/comparison_models.py](src/comparison_models.py)
- data loading and preprocessing code in [src/data_loader.py](src/data_loader.py)
- a module-based workflow in [src/cli.py](src/cli.py) and [src/benchmark.py](src/benchmark.py)
- visualization and reporting helpers in [scripts/](scripts)
- committed metrics in [results/comparison/model_comparison_results.csv](results/comparison/model_comparison_results.csv)
- a committed summary figure in [results/visualization/comprehensive_table.png](results/visualization/comprehensive_table.png)

### Intended rerun path

```bash
pip install -r requirements.txt
python -m src preprocess --config configs/config.yaml
python -m src train --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
python -m src evaluate --config configs/config.yaml --models cnn1d lstm resnet1d hybrid_cnn_lstm
python scripts/visualize_all_models.py
```

### Configuration

- Runtime configuration defaults to [configs/config.yaml](configs/config.yaml).
- For backward compatibility, the code also falls back to a legacy root-level `config.yaml`.
- The committed config is intended as a reasonable default, not as proof that the published snapshot was generated under exactly the same settings.
- After preprocessing, the loader writes a combined audit manifest to `results/comparison/preprocessing/split_manifest_latest.csv`.
- After evaluation, the benchmark writes one prediction CSV per model to `results/comparison/predictions/<model>_test_predictions.csv`.
- After evaluation, the benchmark also writes per-model artifacts such as `results/comparison/evaluation/<model>/confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `threshold_sweep.csv`, and `per_class_metrics.csv`.
- Large checkpoints and generated run artifacts are intentionally excluded from version control in this public repository.

### Current gaps

- The processed split files used for the committed runs are not included.
- Exact regeneration of the published numbers from a fresh checkout is still **not guaranteed**, because the committed results are a repository snapshot and some artifacts may predate the current preprocessing protocol.

### Testing

The repository includes a small unit-test layer under [tests/](tests) covering CLI dispatch, config loading, model instantiation, and a mocked preprocessing sanity check that writes arrays and manifests without running full training.

Run the full test suite with:

```bash
python -m unittest discover -s tests -v
```

### Data Setup

The loader expects PTB-XL under:

```text
data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
```

The raw data itself is not redistributed in this repository.

## Additional Documentation

- Supplementary evaluation summary: [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)
- Exploratory notebook: [notebooks/model_training_evaluation.ipynb](notebooks/model_training_evaluation.ipynb)

## Summary

This project is strongest when presented as:

- a focused **normal-vs-abnormal ECG classification benchmark**
- built on a real public biomedical dataset
- comparing multiple neural baselines
- with a simple command-line workflow
- with a small set of committed summary artifacts
- and with clear acknowledgment of the current experimental and reproducibility limitations

That framing is more credible than presenting the repository as a polished framework or as a clinically validated medical AI system.
