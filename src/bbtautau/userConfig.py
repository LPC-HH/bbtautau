"""
Configuration file for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path

MAIN_DIR = Path("../../")

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

SAMPLE_MAPPING = {
    "he": 0,
    "hm": 1,
    "hh": 2,
    "QCD": 3,
    "all": 4,
}
