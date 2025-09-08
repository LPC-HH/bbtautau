"""
Configuration file for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path
from boostedhh import hh_vars
from boostedhh.utils import Sample

MAIN_DIR = Path("../../")
MODEL_DIR = Path(
    "/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models"
)
CLASSIFIER_DIR = Path("/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/")

data_dir_2022 = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal"
data_dir_otheryears = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"

DATA_PATHS = {
    "2022": {
        "data": Path(data_dir_2022),
        "bg": Path(data_dir_2022),
        "signal": Path(data_dir_2022),
    },
    "2022EE": {
        "data": Path(data_dir_otheryears),
        "bg": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
    "2023": {
        "data": Path(data_dir_otheryears),
        "bg": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
    "2023BPix": {
        "data": Path(data_dir_otheryears),
        "bg": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
}

ABCD_SAMPLES = {
    "jetmet": Sample(
        selector="^(JetHT|JetMET)",
        label="JetMET",
        isData=True,
    ),
    "tau": Sample(
        selector="^Tau_Run",
        label="Tau",
        isData=True,
    ),
    "muon": Sample(
        selector="^Muon_Run",
        label="Muon",
        isData=True,
    ),
    "egamma": Sample(
        selector="^EGamma_Run",
        label="EGamma",
        isData=True,
    ),
    "ttbarhad": Sample(
        selector="^TTto4Q",
        label="TT Had",
        isSignal=False,
    ),
    "ttbarsl": Sample(
        selector="^TTtoLNu2Q",
        label="TT SL",
        isSignal=False,
    ),
    "ttbarll": Sample(
        selector="^TTto2L2Nu",
        label="TT LL",
        isSignal=False,
    ),
    "dyjets": Sample(
        selector="^DYto2L",
        label="DY+Jets",
        isSignal=False,
    ),
    "bbtt": Sample(
        selector=hh_vars.bbtt_sigs["bbtt"],
        label=r"ggF HHbb$\tau\tau$",
        isSignal=True,
    ),
}

# Probably could make a file just to configure the fit
SHAPE_VAR = {
    "name": "bbFatJetParTmassResApplied",
    "range": [60, 220],
    "nbins": 16,
    "blind_window": [110, 150],
}


# SAMPLE_MAPPING = {
#     "he": 0,
#     "hm": 1,
#     "hh": 2,
#     "QCD": 3,
#     "all": 4,
# }
