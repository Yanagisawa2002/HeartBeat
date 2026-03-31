# Sample Input Format

This directory contains **bundled demo inputs** for the Dockerized web UI.

Important:

- The committed CSV files are **synthetic demo waveforms**
- They are included only to make the browser demo and Docker deployment testable
- They are **not** PTB-XL samples and **not** benchmark artifacts

Supported CSV shapes:

- `12 x 1000` (lead-first)
- `1000 x 12` (sample-first; the demo will transpose it)

Notes:

- Use plain numeric CSV only. Header rows are not supported.
- The expected length comes from `configs/config.yaml`.
- Real inference still requires a compatible trained checkpoint under `artifacts/checkpoints/` or another directory referenced by `HEARTBEAT_CHECKPOINT_DIR`.
