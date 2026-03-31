# Demo Artifacts

This directory stores lightweight inference artifacts used by the Dockerized
web demo.

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

When these files are present during `docker build`, the resulting image bundles
all benchmark models and the web UI exposes them in the model selector.

The public repository may still omit checkpoint binaries from version control.
If you build from a checkout that does not contain them, add the `.pth` files
under `artifacts/checkpoints/` before building the image, or point the
container to a different directory with `HEARTBEAT_CHECKPOINT_DIR`.
