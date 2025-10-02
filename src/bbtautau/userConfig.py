"""
Configuration file for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path


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


MAIN_DIR = Path("../../")
MODEL_DIR = Path(
    "/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models"
)
CLASSIFIER_DIR = Path("/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/")
DATA_DIR = "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Sep23AddVars_v12_private_signal"
DATA_PATHS = path_dict(DATA_DIR)

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
