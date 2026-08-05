from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override {key}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form: {override}")
        key, raw_value = override.split("=", 1)
        _set_dotted(config, key, yaml.safe_load(raw_value))
    return config


def parse_config_args(description: str) -> tuple[dict[str, Any], argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-run", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    return load_config(args.config, args.overrides), args


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

