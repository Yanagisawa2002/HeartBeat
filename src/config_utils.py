from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    """Resolve a config file path while preserving backward compatibility."""
    candidates = []

    if config_path:
        requested = Path(config_path)
        if requested.is_absolute():
            candidates.append(requested)
        else:
            candidates.extend(
                [
                    requested,
                    PROJECT_ROOT / requested,
                    PROJECT_ROOT / "configs" / requested.name,
                ]
            )
    else:
        candidates.extend(
            [
                PROJECT_ROOT / "configs" / "config.yaml",
                PROJECT_ROOT / "configs" / "config.example.yaml",
                PROJECT_ROOT / "config.yaml",
            ]
        )

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    if ordered_candidates:
        return ordered_candidates[0]
    return PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
