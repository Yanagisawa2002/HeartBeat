# Sample Input Format

This directory contains **bundled demo inputs** for the Dockerized web UI.

The committed sample set now includes two kinds of files:

- **PTB-XL example windows:** a small number of real public ECG examples used to
  make the demo preview look like actual ECG data
- **Synthetic fallback waveforms:** deterministic numeric inputs kept for UI and
  deployment testing

Important:

- the full PTB-XL dataset is **not** redistributed here
- only a very small number of individual example windows are bundled for demo use
- these sample files are for **inference UI preview**, not for benchmark training

Supported CSV shapes:

- `12 x 1000` (lead-first)
- `1000 x 12` (sample-first; the demo will transpose it)

Notes:

- Use plain numeric CSV only. Header rows are not supported.
- The expected length comes from `configs/config.yaml`.
- Real inference still requires a compatible trained checkpoint under `artifacts/checkpoints/` or another directory referenced by `HEARTBEAT_CHECKPOINT_DIR`.
