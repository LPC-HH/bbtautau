from __future__ import annotations

import importlib

"""
BDT Configuration Registry

Model configurations are stored in bdt_configs/ as config_{modelname}.py files.
Configs are loaded on-demand by directly importing the corresponding module.
"""


class _ConfigDict(dict):
    """Dict-like interface for loading BDT model configs on-demand.

    Configs are loaded by importing bdt_configs.config_{modelname} and
    accessing the CONFIG dictionary. Results are cached after first access.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    def _load_config(self, modelname: str):
        """Load config for a given modelname by importing the config module."""
        # Convert modelname to module name: config_{modelname}
        module_name = f"bbtautau.postprocessing.bdt_configs.config_{modelname}"

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise KeyError(
                f"Model '{modelname}' not found. "
                f"Expected file: bdt_configs/config_{modelname}.py"
            ) from e

        if not hasattr(module, "CONFIG"):
            raise KeyError(f"Module config_{modelname} does not define CONFIG dictionary")

        config = module.CONFIG

        # Validate modelname matches
        if config.get("modelname") != modelname:
            raise ValueError(
                f"Config modelname mismatch: file has '{config.get('modelname')}', "
                f"expected '{modelname}'"
            )

        return config

    def __getitem__(self, modelname: str):
        """Load and return config for modelname, caching the result."""
        if modelname not in self._cache:
            self._cache[modelname] = self._load_config(modelname)
        return self._cache[modelname]

    def __contains__(self, modelname: str) -> bool:
        """Check if config exists by attempting to load it (caches on success)."""
        if modelname in self._cache:
            return True
        try:
            config = self._load_config(modelname)
            self._cache[modelname] = config  # Cache on successful load
            return True
        except (KeyError, ImportError, ValueError):
            return False

    def keys(self):
        """Not implemented - requires scanning all files."""
        raise NotImplementedError(
            "keys() not supported. Use BDT_CONFIG[modelname] to access specific configs."
        )

    def __iter__(self):
        """Not implemented - requires scanning all files."""
        raise NotImplementedError(
            "Iteration not supported. Use BDT_CONFIG[modelname] to access specific configs."
        )

    def __len__(self):
        """Not implemented - requires scanning all files."""
        raise NotImplementedError(
            "len() not supported. Use BDT_CONFIG[modelname] to access specific configs."
        )


# Public API: BDT_CONFIG dict
BDT_CONFIG = _ConfigDict()
