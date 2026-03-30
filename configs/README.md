# Configuration

The repository now includes a default runtime config at `configs/config.yaml`.

For backward compatibility, the codebase also falls back to a legacy root-level
`config.yaml` if `configs/config.yaml` is not present.

If you need machine-specific or experiment-specific overrides, edit
`configs/config.yaml` or point the CLI to an alternate file with `--config`.
