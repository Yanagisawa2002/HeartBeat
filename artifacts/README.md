# Demo Artifacts

This directory is reserved for lightweight inference artifacts used by the
Dockerized web demo.

## Expected checkpoint layout

The web app discovers model checkpoints from:

- `artifacts/checkpoints/`
- `results/comparison/models/`

The recommended deployable layout is:

```text
artifacts/
└── checkpoints/
    └── cnn1d_best.pth
```

The public repository does **not** commit trained checkpoints by default.
Mount a checkpoint into `artifacts/checkpoints/` or point the container to a
different directory with the `HEARTBEAT_CHECKPOINT_DIR` environment variable.
