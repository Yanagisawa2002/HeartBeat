# Demo Artifacts

This directory stores lightweight inference artifacts used by the Dockerized
web demo and the published container image.

## Bundled checkpoint layout

The web app discovers model checkpoints from:

- `artifacts/checkpoints/`
- `results/comparison/models/`

The intended demo layout is:

```text
artifacts/
`-- checkpoints/
    |-- cnn1d_best.pth
    |-- lstm_best.pth
    |-- resnet1d_best.pth
    |-- hybrid_cnn_lstm_best.pth
    `-- inception1d_best.pth
```

These checkpoint files are committed specifically so the published GHCR image
can be built as a true one-click demo. When they are present during
`docker build`, the resulting image bundles all benchmark models and the web UI
exposes them in the model selector.

If you want to replace them with your own trained checkpoints, keep the same
file naming convention or point the container to a different directory with
`HEARTBEAT_CHECKPOINT_DIR`.
