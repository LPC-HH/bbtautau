"""
Configuration file for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
BDT_EVAL_DIR = Path("/ceph/cms/store/user/lumori/bbtautau/BDT_predictions/")
DATA_DIR = "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Sep23AddVars_v12_private_signal"
# DATA_DIR = "/ceph/cms/store/user/lumori/bbtautau/skimmer/26Jan22_PNetPresbb0p3_v12_private_signal"
DATA_PATHS = path_dict(DATA_DIR)

PLOT_DIR = Path("/home/users/lumori/bbtautau/plots")

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


# Working point bin edges for tt GloParT discriminants
# Determine using the plot_cuts_hist.py script
# # Example:
# WPS_TTPART = {
#     "ttFatJetParTXtauhtauhvsQCDTop": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
#     "ttFatJetParTXtauhtauevsQCDTop": np.array([0.15, 0.25, 0.35, 0.45]),
#     "ttFatJetParTXtauhtaumvsQCDTop": np.array([0.12, 0.22, 0.32, 0.42, 0.52]),
# }
WPS_TTPART: dict[str, np.ndarray] = {
    "ttFatJetParTXtauhtauhvsQCDTop": np.array(
        [0.4, 0.7722271084785461, 0.9961739778518677, 0.9997739195823669, 0.9999425411224365]
    ),
    "ttFatJetParTXtauhtaumvsQCDTop": np.array(
        [0.4, 0.8451746702194214, 0.9981048107147217, 0.9997354745864868, 0.9999304413795471]
    ),
    "ttFatJetParTXtauhtauevsQCDTop": np.array(
        [0.4, 0.806851863861084, 0.9941548109054565, 0.9995246529579163, 0.9998924732208252]
    ),
}
