from __future__ import annotations

import numpy as np
import pandas as pd
import Samples
from boostedhh import utils
from boostedhh.utils import Channel

from bbtautau import bbtautau_vars

filters = [
    [
        ("('ak8FatJetPt', '0')", ">=", 250),
        ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
        ("('ak8FatJetPt', '1')", ">=", 200),
        # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
        # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
        # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
    ],
]


def load_samples(year: str, channel: Channel, paths: dict[str]):
    events_dict = {}

    samples = Samples.SAMPLES.copy()

    # remove unnecessary data samples
    for key in Samples.DATASETS:
        if (key in samples) and (key not in channel.data_samples):
            del samples[key]

    # load samples
    for key, sample in samples.items():
        if sample.selector is not None:
            events_dict[key] = utils.load_sample(sample, year, paths, filters)

    # keep only the specified bbtt channel
    for signal in Samples.SIGNALS:
        events_dict[f"{signal}{channel.key}"] = events_dict[signal][
            events_dict[signal][f"GenTau{channel.key}"][0]
        ]
        del events_dict[signal]

    return events_dict


def apply_triggers(
    events_dict: dict[str, pd.DataFrame], year: str, channel: Channel  # noqa: ARG001
):
    """Apply triggers in MC and data, and remove overlap between datasets."""
    # MC
    for skey, events in events_dict.items():
        if skey not in Samples.DATASETS:
            triggered = np.sum([events[hlt][0] for hlt in channel.triggers], axis=0).astype(bool)
            events_dict[skey] = events[triggered]

    # data + overlap removal
    ldataset = channel.lepton_dataset

    trigdict = {"jetmet": {}, "tau": {}}
    if channel.isLepton:
        trigdict[ldataset] = {}
        lepton_triggers = utils.list_intersection(channel.lepton_triggers, channel.triggers)

    jet_triggers = utils.list_intersection(bbtautau_vars.HLT_jets, channel.triggers)
    tau_triggers = utils.list_intersection(bbtautau_vars.HLT_taus, channel.triggers)

    for key, d in trigdict.items():
        d["jets"] = np.sum([events_dict[key][hlt][0] for hlt in jet_triggers], axis=0).astype(bool)

        if key == "jetmet":
            continue

        d["taus"] = np.sum([events_dict[key][hlt][0] for hlt in tau_triggers], axis=0).astype(bool)
        d["taunojets"] = ~d["jets"] & d["taus"]

        if key == "tau":
            continue

        if channel.isLepton:
            d[ldataset] = np.sum(
                [events_dict[key][hlt][0] for hlt in lepton_triggers], axis=0
            ).astype(bool)

            d[f"{ldataset}noothers"] = ~d["jets"] & ~d["taus"] & d[ldataset]

    # remove overlap
    events_dict["jetmet"] = events_dict["jetmet"][trigdict["jetmet"]["jets"]]
    events_dict["tau"] = events_dict["tau"][trigdict["tau"]["taunojets"]]
    events_dict[ldataset] = events_dict[ldataset][trigdict[ldataset][f"{ldataset}noothers"]]
