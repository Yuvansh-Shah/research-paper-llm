from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    extends = payload.pop("extends", None)
    if extends is None:
        return payload

    base_path = (path.parent / extends).resolve()
    base_payload = _load_json_config(base_path)
    return _deep_merge(base_payload, payload)


def detect_config_path(argv: Optional[list[str]] = None) -> Optional[Path]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    return known.config


def load_config_defaults(
    argv: Optional[list[str]] = None,
    *,
    section: Optional[str] = None,
) -> tuple[Optional[Path], dict[str, Any]]:
    config_path = detect_config_path(argv)
    if config_path is None:
        return None, {}

    resolved = config_path.resolve()
    payload = _load_json_config(resolved)
    if section is not None and section in payload:
        section_payload = payload[section]
    else:
        section_payload = payload

    if not isinstance(section_payload, dict):
        raise ValueError(
            f"Config section '{section}' must be a JSON object in {resolved}"
        )
    return resolved, dict(section_payload)
