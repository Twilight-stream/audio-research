"""Safe YAML loading, relative recursive inheritance and typed CLI overrides."""

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml


def _merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load(path: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"base_config inheritance cycle: {(*ancestors, path)}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config is None:
        config = {}
    if not isinstance(config, dict) or not all(isinstance(k, str) for k in config):
        raise ValueError(f"config must be a mapping with string keys: {path}")
    bases = config.pop("base_config", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(p, str) for p in bases):
        raise ValueError("base_config must be a path string or list of path strings")
    result: dict[str, Any] = {}
    for base in bases:
        result = _merge(result, _load(path.parent / base, (*ancestors, path)))
    return _merge(result, config)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Load YAML; bases merge left-to-right, then child, then key=value overrides.

    Base paths are relative to their declaring file. Overrides use safe YAML
    values and allow new keys; traversing an existing scalar raises ValueError.
    """
    config = _load(Path(path), ())
    for override in overrides or ():
        key, separator, raw = override.partition("=")
        parts = key.split(".")
        if not separator or any(not part.strip() or part != part.strip() for part in parts):
            raise ValueError(f"invalid override {override!r}; expected dotted.key=value")
        value = yaml.safe_load(raw)
        # PyYAML treats bare 1e-4 as a string; CLI scientific notation is numeric.
        if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", raw):
            value = float(raw)
        node = config
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise ValueError(f"cannot traverse non-mapping key {part!r} in {override!r}")
            node = node[part]
        node[parts[-1]] = value
    return config
