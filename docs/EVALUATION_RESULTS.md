# ECG Benchmark Result Note

This note summarizes the **authoritative committed benchmark snapshot** kept in
the public repository for portfolio and interview use.

## Scope

HeartBeat presents a **supervised normal-vs-abnormal ECG classification
benchmark** on **PTB-XL**. It should be interpreted as a benchmark study on a
public dataset, not as a clinical diagnostic system.

## Committed Result Snapshot

The current public result snapshot is:

- [../results/full_benchmark_all_models_20260331/model_comparison_results.csv](../results/full_benchmark_all_models_20260331/model_comparison_results.csv)
- [../results/full_benchmark_all_models_20260331/visualization/comprehensive_table.png](../results/full_benchmark_all_models_20260331/visualization/comprehensive_table.png)

## Summary Table

| Model | Accuracy | F1 Score | AUC Score | Parameters | Inference Time (s) |
|------|----------|----------|-----------|-----------:|-------------------:|
| CNN1D | 0.8700 | 0.8709 | 0.9470 | 705,218 | **0.275** |
| LSTM | 0.8721 | 0.8726 | 0.9445 | 903,298 | 3.117 |
| ResNet1D | 0.8715 | 0.8721 | 0.9461 | 3,849,858 | 0.378 |
| Hybrid CNN-LSTM | 0.8641 | 0.8650 | 0.9472 | 1,035,458 | 0.420 |
| Inception1D | **0.8767** | **0.8774** | **0.9495** | **460,226** | 0.285 |

## Short Interpretation

- **Strongest overall model:** Inception1D has the best accuracy, F1 score, and ROC-AUC in the committed full benchmark rerun.
- **Fastest inference baseline:** CNN1D remains the lowest-latency model in this snapshot.
- **Best practical default:** Inception1D combines the strongest discrimination performance with the smallest parameter count in this run.

## Figure

![Committed Summary Table](../results/full_benchmark_all_models_20260331/visualization/comprehensive_table.png)

## Caveat

These numbers should be interpreted as **benchmark evidence under the current
repository protocol**, not as evidence of external validity or clinical
deployment readiness.
