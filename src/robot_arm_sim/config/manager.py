from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def get(self, *keys: str, default: Any = None) -> Any:
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def save_to(self, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / self.config_path.name
        shutil.copy2(self.config_path, dest)
        return dest
