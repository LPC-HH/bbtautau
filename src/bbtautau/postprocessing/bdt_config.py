from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path

"""
BDT Configuration Registry

Model configurations are stored in bdt_configs/ as config_{modelname}.py files,
organized into subfolders (e.g. standard/, key_pars_k2v0/, legacy/).
Configs are located by scanning the directory tree and loaded on-demand.
"""

_BDT_CONFIGS_DIR = Path(__file__).parent / "bdt_configs"


def _find_config_file(modelname: str) -> Path:
    """Find config_{modelname}.py anywhere under bdt_configs/."""
    target = f"config_{modelname}.py"
    matches = list(_BDT_CONFIGS_DIR.rglob(target))
    if not matches:
        raise KeyError(
            f"Model '{modelname}' not found. " f"No file named '{target}' under {_BDT_CONFIGS_DIR}"
        )
    if len(matches) > 1:
        locs = ", ".join(str(m.relative_to(_BDT_CONFIGS_DIR)) for m in matches)
        raise KeyError(f"Ambiguous model '{modelname}': found in multiple locations: {locs}")
    return matches[0]


def _load_config_from_file(path: Path, modelname: str) -> dict:
    """Import a config file by path and return its CONFIG dict."""
    spec = importlib.util.spec_from_file_location(f"config_{modelname}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise KeyError(f"{path.name} does not define a CONFIG dictionary")

    config = module.CONFIG
    if config.get("modelname") != modelname:
        raise ValueError(
            f"Config modelname mismatch in {path}: "
            f"has '{config.get('modelname')}', expected '{modelname}'"
        )
    return config


class _ConfigDict(dict):
    """Dict-like interface for loading BDT model configs on-demand.

    Scans bdt_configs/ and its subfolders for config_{modelname}.py files.
    Results are cached after first access.
    """

    def __init__(self):
        super().__init__()
        self._cache: dict[str, dict] = {}

    def __getitem__(self, modelname: str):
        if modelname not in self._cache:
            path = _find_config_file(modelname)
            self._cache[modelname] = _load_config_from_file(path, modelname)
        return self._cache[modelname]

    def __contains__(self, modelname: str) -> bool:
        if modelname in self._cache:
            return True
        try:
            self[modelname]
            return True
        except (KeyError, ValueError):
            return False

    def keys(self):
        """Discover all available model names by scanning the directory tree."""
        self._scan_all()
        return self._cache.keys()

    def __iter__(self):
        self._scan_all()
        return iter(self._cache)

    def __len__(self):
        self._scan_all()
        return len(self._cache)

    def _scan_all(self):
        """Load every config_{*}.py found under bdt_configs/."""
        for path in _BDT_CONFIGS_DIR.rglob("config_*.py"):
            modelname = path.stem.removeprefix("config_")
            if modelname not in self._cache:
                with contextlib.suppress(KeyError, ValueError):
                    self._cache[modelname] = _load_config_from_file(path, modelname)


# Public API
BDT_CONFIG = _ConfigDict()
