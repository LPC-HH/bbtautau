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
    module = _load_config_module(path)

    if not hasattr(module, "CONFIG"):
        raise KeyError(f"{path.name} does not define a CONFIG dictionary")

    config = module.CONFIG
    if config.get("modelname") != modelname:
        raise ValueError(
            f"Config modelname mismatch in {path}: "
            f"has '{config.get('modelname')}', expected '{modelname}'"
        )
    return config


def _load_config_module(path: Path):
    """Import a config module by path."""
    spec = importlib.util.spec_from_file_location(f"config_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_explicit_config(path: str | Path, modelname: str | None = None) -> tuple[str, dict]:
    """Load a config file from an explicit path and validate its model name.

    Args:
        path: Path to a Python config file defining ``CONFIG``.
        modelname: Optional explicit model name to validate against. If omitted,
            the model name is read from ``CONFIG["modelname"]``.

    Returns:
        Tuple of ``(resolved_modelname, config_dict)``.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Explicit config file not found: {path}")

    if modelname is not None:
        config = _load_config_from_file(path, modelname)
        return modelname, config

    module = _load_config_module(path)
    if not hasattr(module, "CONFIG"):
        raise KeyError(f"{path.name} does not define a CONFIG dictionary")

    config = module.CONFIG
    target_model = config.get("modelname")
    if not target_model:
        raise KeyError(f"{path.name} does not define CONFIG['modelname']")

    if config.get("modelname") != target_model:
        raise ValueError(
            f"Config modelname mismatch in {path}: "
            f"has '{config.get('modelname')}', expected '{target_model}'"
        )
    return target_model, config


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

    def register(self, modelname: str, config: dict) -> dict:
        """Register an explicit config in the in-memory cache."""
        if config.get("modelname") != modelname:
            raise ValueError(
                f"Config modelname mismatch in explicit config: "
                f"has '{config.get('modelname')}', expected '{modelname}'"
            )
        self._cache[modelname] = config
        return config

    def register_from_file(self, path: str | Path, modelname: str | None = None) -> str:
        """Load a config from a file and register it in the cache."""
        resolved_modelname, config = load_explicit_config(path, modelname=modelname)
        self.register(resolved_modelname, config)
        return resolved_modelname

    def _scan_all(self):
        """Load every config_{*}.py found under bdt_configs/."""
        for path in _BDT_CONFIGS_DIR.rglob("config_*.py"):
            modelname = path.stem.removeprefix("config_")
            if modelname not in self._cache:
                with contextlib.suppress(KeyError, ValueError):
                    self._cache[modelname] = _load_config_from_file(path, modelname)


# Public API
BDT_CONFIG = _ConfigDict()
