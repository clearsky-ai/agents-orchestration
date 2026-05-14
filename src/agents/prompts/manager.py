from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Prompt:
    """The system_message an agent fetches at startup."""

    name: str
    version: str
    system_message: str


class PromptManager:
    """Lazy loader for the YAML prompt registry.

    Reads all ``*.yaml`` files under ``registry_path`` once on first use and
    merges their top-level keys into a single name -> entry map. Each entry
    must have ``default_version`` and a ``versions`` dict with at least one
    version containing ``system_message``.
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        if registry_path is None:
            registry_path = Path(__file__).parent / "registry"
        self.registry_path = Path(registry_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Prompt registry not found: {self.registry_path}"
            )
        for path in sorted(self.registry_path.glob("*.yaml")):
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(
                    f"{path} must contain a top-level mapping of agent names."
                )
            self._cache.update(data)
        self._loaded = True

    def get(self, name: str, version: Optional[str] = None) -> Prompt:
        self._ensure_loaded()

        if name not in self._cache:
            raise KeyError(
                f"Prompt '{name}' not found. Available: {sorted(self._cache.keys())}"
            )

        entry = self._cache[name]

        if version is None:
            version = entry.get("default_version")
            if version is None:
                raise ValueError(f"No default_version for prompt '{name}'")

        versions = entry.get("versions") or {}
        if version not in versions:
            raise KeyError(
                f"Version '{version}' not found for '{name}'. "
                f"Available: {sorted(versions.keys())}"
            )

        v = versions[version]
        if "system_message" not in v:
            raise ValueError(
                f"Missing 'system_message' in {name} v{version}"
            )

        return Prompt(
            name=name,
            version=version,
            system_message=v["system_message"],
        )

    def reload(self) -> None:
        """Clear the cache and reread the registry on next access."""
        self._cache.clear()
        self._loaded = False


_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Process-wide singleton accessor."""
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager
