"""
Configuration file for the bbtautau package.

Authors: Ludovico Mori, Haoyang Billy Li

Paths under the git checkout are resolved from this file's location (not the process cwd).
Default skimmer data path points to Billy's primary ceph area because it has BSM samples; override if needed:

  export BBTAUTAU_DATA_DIR=/path/to/skimmer/output
  export BBTAUTAU_BDT_EVAL_DIR=/path/to/BDT_predictions
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def _username() -> str:
    return os.environ.get("USER") or getpass.getuser()


# .../src/bbtautau/userConfig.py -> repo root (parent of src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PACKAGE_ROOT = Path(__file__).resolve().parent


def path_dict(path: str, path_2022: str = None):
    return {
        "2022": {
            "data": Path(path_2022 if path_2022 else path),
            "bg": Path(path_2022 if path_2022 else path),
            "signal": Path(path_2022 if path_2022 else path),
        },
        "2022EE": {
            "data": Path(path),
            "bg": Path(path),
            "signal": Path(path),
        },
        "2023": {
            "data": Path(path),
            "bg": Path(path),
            "signal": Path(path),
        },
        "2023BPix": {
            "data": Path(path),
            "bg": Path(path),
            "signal": Path(path),
        },
    }


MAIN_DIR = _REPO_ROOT
MODEL_DIR = _PACKAGE_ROOT / "postprocessing" / "classifier" / "trained_models"
CLASSIFIER_DIR = _PACKAGE_ROOT / "postprocessing" / "classifier"

_user = _username()
_repo_name = _REPO_ROOT.name
# Skimmer ntuple path (shared bbtautau skimmer tag on ceph)
_default_data_dir = (
    "/ceph/cms/store/user/haoyang/bbtautau/skimmer/25Sep23AddVars_v12_private_signal"
)
DATA_DIR = os.environ.get("BBTAUTAU_DATA_DIR", _default_data_dir)
DATA_PATHS = path_dict(DATA_DIR)

_default_bdt_eval = f"/ceph/cms/store/user/{_user}/{_repo_name}/BDT_predictions/"
BDT_EVAL_DIR = Path(os.environ.get("BBTAUTAU_BDT_EVAL_DIR", _default_bdt_eval))

PLOT_DIR = _REPO_ROOT / "plots"

# backwards compatibility
# data_dir_2022 = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal"
# data_dir_otheryears = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
# DATA_PATHS = path_dict(data_dir_2022, data_dir_otheryears)

# Probably could make a file just to configure the fit
SHAPE_VAR = {
    "name": "bbFatJetParTmassResApplied",
    "range": [60, 220],
    "nbins": 16,
    "blind_window": [110, 150],
}

PT_CUTS = {
    "bb": 250,
    "tt": 200,
}

# usually will go (hh,ggf)->(hh,vbf)->(hm,ggf), etc.
CHANNEL_ORDERING = ["hh", "hm", "he"]  # order of applying selection and vetoes
SIGNAL_ORDERING = ["ggfbbtt", "vbfbbtt"]  # order of applying selection and vetoes
