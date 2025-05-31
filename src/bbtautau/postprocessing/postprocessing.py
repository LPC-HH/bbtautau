"""
Postprocessing functions for bbtautau.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import hist
import matplotlib as mpl
import numpy as np
import pandas as pd
from boostedhh import hh_vars, utils
from boostedhh.hh_vars import data_key
from boostedhh.utils import Sample, ShapeVar, add_bool_arg
from hist import Hist

import bbtautau.postprocessing.utils as putils
from bbtautau.bbtautau_utils import Channel
from bbtautau.HLTs import HLTs
from bbtautau.postprocessing import Regions, Samples, plotting
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES, SIGNALS
from bbtautau.postprocessing.utils import LoadedSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")


base_filters_default = [
    [
        ("('ak8FatJetPt', '0')", ">=", 250),
        ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
        ("('ak8FatJetPt', '1')", ">=", 200),
        # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
        # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
        # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
    ]
]


control_plot_vars = (
    [
        ShapeVar(var=f"{jet}FatJetPt", label=rf"$p_T^{{{jlabel}}}$ [GeV]", bins=[20, 250, 1250])
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
    ]
    + [
        ShapeVar(
            var="METPt", label=r"$p^{miss}_T$ [GeV]", bins=[20, 0, 300]
        ),  # METPt is used for resel samples
        # ShapeVar(var="MET_phi", label=r"$\phi^{miss}$", bins=[20, -3.2, 3.2]),
    ]
    + [
        ShapeVar(var=f"ak8FatJetPt{i}", label=rf"$p_T^{{j{i + 1}}}$ [GeV]", bins=[20, 250, 1250])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetMsd{i}", label=rf"$m_{{SD}}^{{j{i + 1}}}$ [GeV]", bins=[20, 0, 300])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetEta{i}", label=rf"$\eta^{{j{i + 1}}}$", bins=[20, -2.5, 2.5])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetPhi{i}", label=rf"$\phi^{{j{i + 1}}}$", bins=[20, -3.2, 3.2])
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetPNetmassLegacy{i}",
            label=rf"PNet Legacy $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetParTmassResApplied{i}",
            label=rf"ParT Resonance $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetParTmassVisApplied{i}",
            label=rf"ParT Visable $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    # ak8FatJetParTXbbvsQCD
    + [
        ShapeVar(
            var=f"ak8FatJetParTXbbvsQCD{i}",
            label=rf"ParT XbbvsQCD j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetParTXbbvsQCDTop
    + [
        ShapeVar(
            var=f"ak8FatJetParTXbbvsQCDTop{i}",
            label=rf"ParT XbbvsQCDTop j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetPNetXbbvsQCDLegacy
    + [
        ShapeVar(
            var=f"ak8FatJetPNetXbbvsQCDLegacy{i}",
            label=rf"PNet Legacy XbbvsQCD j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    #  nElectrons
    + [ShapeVar(var="nElectrons", label=r"Number of Electrons", bins=[3, 0, 3])]
    #  nMuons
    + [ShapeVar(var="nMuons", label=r"Number of Muons", bins=[3, 0, 3])]
    #  nTaus
    + [ShapeVar(var="nTaus", label=r"Number of Taus", bins=[3, 0, 3])]
    #  nBoostedTaus
    + [ShapeVar(var="nBoostedTaus", label=r"Number of Boosted Taus", bins=[3, 0, 3])]
)

# fitting on bb regressed mass
shape_vars = [
    ShapeVar(
        "bbFatJetPNetmassLegacy",
        r"$m^{bb}_\mathrm{Reg}$ [GeV]",
        [16, 60, 220],
        reg=True,
        blind_window=[110, 140],
    )
]


def main(args: argparse.Namespace):
    CHANNEL = CHANNELS[args.channel]

    data_paths = {
        "signal": args.signal_data_dirs,
        "data": args.data_dir,
        "bg": args.bg_data_dirs,
    }

    if args.sigs is None:
        args.sigs = {s + CHANNEL.key: SAMPLES[s + CHANNEL.key] for s in SIGNALS}

    if args.bgs is None:
        args.bgs = {bkey: b for bkey, b in SAMPLES.items() if b.get_type() == "bg"}

    if args.templates:
        filters = bb_filters(num_fatjets=3, bb_cut=0.3)
        filters = tt_filters(CHANNEL, filters, num_fatjets=3, tt_cut=0.3)
    else:
        filters = None

    print("Loading samples")
    print("Filters:")
    pprint(filters)
    print()
    # dictionary that will contain all information (from all samples)
    events_dict = load_samples(
        args.year,
        CHANNEL,
        data_paths,
        load_data=True,
        load_bgs=True,
        filters_dict=filters,
        loaded_samples=True,
    )

    cutflow = utils.Cutflow(samples=events_dict)
    cutflow.add_cut(events_dict, "Preselection", "finalWeight")
    print(cutflow.cutflow)

    print("\nTriggers")
    apply_triggers(events_dict, args.year, CHANNEL)
    cutflow.add_cut(events_dict, "Triggers", "finalWeight")
    print(cutflow.cutflow)

    derive_variables(events_dict, CHANNEL)

    print("\nbbtautau assignment")
    bbtautau_assignment(events_dict, CHANNEL)

    print("\nTemplates")
    templates = get_templates(
        events_dict,
        args.year,
        args.sigs,
        args.bgs,
        CHANNEL,
        shape_vars,
        {},  # TODO: systematics
        # pass_ylim=150,
        # fail_ylim=1e5,
        sig_scale_dict={f"bbtt{CHANNEL.key}": 300, f"vbfbbtt-k2v0{CHANNEL.key}": 40},
        template_dir=args.template_dir,
        plot_dir=args.plot_dir,
        show=False,
    )

    print("\nSaving templates")
    save_templates(
        templates, args.template_dir / f"{args.year}_templates.pkl", args.blinded, shape_vars
    )


def bb_filters(in_filters: list[tuple] = None, num_fatjets: int = 3, bb_cut: float = 0.3):
    """
    0.3 corresponds to roughly, 85% signal efficiency, 2% QCD efficiency (pT: 250-400, mSD:0-250, mRegLegacy:40-250)
    """
    if in_filters is None:
        in_filters = base_filters_default

    filters = [
        ifilter + [(f"('ak8FatJetParTXbbvsQCD', '{n}')", ">=", bb_cut)]
        for n in range(num_fatjets)
        for ifilter in in_filters
    ]
    return filters


def tt_filters(
    channel: Channel, in_filters: list[tuple] = None, num_fatjets: int = 3, tt_cut: float = 0.9
):
    if in_filters is None:
        in_filters = base_filters_default

    if channel.key == "hm":
        warnings.warn(
            "Temporarily applying vsQCD filter only for tauhtaum due to missing keys!", stacklevel=2
        )
        tt_cut = 0.05
        vslabel = "vsQCD"
    else:
        vslabel = "vsQCDTop"

    filters = [
        ifilter + [(f"('ak8FatJetParTX{channel.tagger_label}{vslabel}', '{n}')", ">=", tt_cut)]
        for n in range(num_fatjets)
        for ifilter in in_filters
    ]
    return filters


def trigger_filter(
    triggers: dict[str, list[str]],
    year: str,
    base_filters: list[tuple] = None,
    fast_mode: bool = False,
    PNetXbb_cut: float = None,
    num_fatjets: int = 3,
) -> dict[str, dict[str, list[list[tuple]]]]:
    """
    creates a list of filters for each trigger in the list of triggers. It is granular to triggers = {"data": { [...] , ...}, "signal": { [...]}.
    """
    if base_filters is None:
        base_filters = copy.deepcopy(base_filters_default)

    if fast_mode:
        for i in range(len(base_filters)):
            base_filters[i] += [("('ak8FatJetPNetXbbLegacy', '0')", ">=", 0.95)]

    filters_dict = {}

    if year == "2023":
        skip_names = ["PNet", "Parking", "Quadjet"]
        skip = []
        for name in skip_names:
            skip += HLTs.hlts_by_type(year, name)

        # exclude from filtering since they change mid-2023 and have dype as bool instead of int
        triggers["data"] = [trigger for trigger in triggers["data"] if trigger not in skip]

    if not isinstance(triggers, dict):
        print(triggers, year, "triggers should be a dictionary")

    for dtype, trigger_list in triggers.items():
        filters_dict[dtype] = [
            ifilter + [(f"('{trigger}', '0')", "==", 1)]
            for trigger in trigger_list
            for ifilter in base_filters
        ]

    if PNetXbb_cut is not None:
        extras = [
            (f"('ak8FatJetPNetXbbLegacy', '{i}')", ">=", PNetXbb_cut) for i in range(num_fatjets)
        ]
        for dtype, filters in filters_dict.items():
            filters_dict[dtype] = [branch + [extra] for branch in filters for extra in extras]

    return filters_dict


def get_columns(
    year: str,
    triggers_in_channel: Channel = None,
    legacy_taggers: bool = True,
    ParT_taggers: bool = True,
    other: bool = True,
    num_fatjets: int = 3,
):

    columns_data = [
        ("weight", 1),
        ("ak8FatJetPt", num_fatjets),
        ("ak8FatJetEta", num_fatjets),
        ("ak8FatJetPhi", num_fatjets),
    ]

    # common columns
    if legacy_taggers:
        columns_data += [
            ("ak8FatJetPNetXbbLegacy", num_fatjets),
            ("ak8FatJetPNetQCDLegacy", num_fatjets),
            ("ak8FatJetPNetmassLegacy", num_fatjets),
            ("ak8FatJetParTmassResApplied", num_fatjets),
            ("ak8FatJetParTmassVisApplied", num_fatjets),
            ("ak8FatJetMsd", num_fatjets),
        ]

    if ParT_taggers:
        for branch in (
            [f"ak8FatJetParT{key}" for key in Samples.qcdouts + Samples.topouts + Samples.sigouts]
            + [f"ak8FatJetParT{key}vsQCD" for key in Samples.sigouts if key != "Xtauhtaum"]
            + [f"ak8FatJetParT{key}vsQCDTop" for key in Samples.sigouts if key != "Xtauhtaum"]
        ):
            columns_data.append((branch, num_fatjets))

    if other:
        columns_data += [
            ("nFatJets", 1),
            ("METPt", 1),
            ("METPhi", 1),
        ]

    columns_mc = copy.deepcopy(columns_data)

    if triggers_in_channel is not None:
        for branch in triggers_in_channel.triggers(year, data_only=True):
            columns_data.append((branch, 1))
        for branch in triggers_in_channel.triggers(year, mc_only=True):
            columns_mc.append((branch, 1))

    # signal-only columns
    columns_signal = copy.deepcopy(columns_mc)

    columns_signal += [
        ("GenTauhh", 1),
        ("GenTauhm", 1),
        ("GenTauhe", 1),
    ]

    columns = {
        "data": utils.format_columns(columns_data),
        "signal": utils.format_columns(columns_signal),
        "bg": utils.format_columns(columns_mc),
    }

    return columns


def load_samples(
    year: str,
    channel: Channel,
    paths: dict[str],
    filters_dict: dict[str, list[list[tuple]]] = None,
    load_columns: dict[str, list[tuple]] = None,
    load_bgs: bool = False,
    load_data: bool = True,
    load_just_bbtt: bool = False,
    loaded_samples: bool = False,
) -> dict[str, LoadedSample | pd.DataFrame]:
    if not loaded_samples:
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future, by setting loaded_samples=True!",
            stacklevel=1,
        )

    events_dict = {}

    samples = Samples.SAMPLES.copy()
    signals = Samples.SIGNALS.copy()

    if load_just_bbtt:  # quite ad hoc but should become obsolete
        del samples["vbfbbtt-k2v0"]
        del samples["vbfbbtt"]
        signals.remove("vbfbbtt-k2v0")
        signals.remove("vbfbbtt")

    # remove unnecessary data samples
    for key in Samples.DATASETS + (not load_bgs) * Samples.BGS:
        if (key in samples) and (key not in channel.data_samples or not load_data):
            del samples[key]

    # load only the specified columns
    if load_columns is not None:
        for sample in samples.values():
            sample.load_columns = load_columns[sample.get_type()]

    # load samples
    for key, sample in samples.items():
        if isinstance(filters_dict, dict):
            filters = filters_dict[sample.get_type()]
        else:
            filters = filters_dict  # this should not be used since the triggers change in data and MC on a given year

        if sample.selector is not None:

            events = utils.load_sample(
                sample,
                year,
                paths,
                filters,
            )

            if not loaded_samples:
                events_dict[key] = events
            else:
                events_dict[key] = LoadedSample(sample=sample, events=events)

    # keep only the specified bbtt channel
    for signal in signals:
        if not loaded_samples:
            # quick fix due to old naming still in samples
            events_dict[f"{signal}{channel.key}"] = events_dict[signal][
                events_dict[signal][f"GenTau{channel.key}"][0]
            ]
            del events_dict[signal]
        else:
            events_dict[f"{signal}{channel.key}"] = LoadedSample(
                sample=Samples.SAMPLES[f"{signal}{channel.key}"],
                events=events_dict[signal].events[
                    events_dict[signal].get_var(f"GenTau{channel.key}")
                ],
            )
            del events_dict[signal]

    return events_dict


def apply_triggers_data_old(events_dict: dict[str, pd.DataFrame], year: str, channel: Channel):
    """Apply triggers to data and remove overlap between datasets due to multiple triggers fired in an event."""
    ldataset = channel.lepton_dataset

    # storing triggers fired per dataset
    trigdict = {"jetmet": {}, "tau": {}}
    if channel.isLepton:
        trigdict[ldataset] = {}
        lepton_triggers = utils.list_intersection(
            channel.lepton_triggers(year), channel.triggers(year, data_only=True)
        )

    # JetMET triggers considered in this channel
    jet_triggers = utils.list_intersection(
        HLTs.hlts_by_dataset(year, "JetMET", data_only=True), channel.triggers(year, data_only=True)
    )

    # Tau triggers considered in this channel
    tau_triggers = utils.list_intersection(
        HLTs.hlts_by_dataset(year, "Tau", data_only=True), channel.triggers(year, data_only=True)
    )

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

            events_dict[ldataset] = events_dict[ldataset][trigdict[ldataset][f"{ldataset}noothers"]]

    # remove overlap
    # print(trigdict["jetmet"])

    events_dict["jetmet"] = events_dict["jetmet"][trigdict["jetmet"]["jets"]]
    events_dict["tau"] = events_dict["tau"][trigdict["tau"]["taunojets"]]

    return events_dict


def apply_triggers_old(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    channel: Channel,
):
    """Apply triggers in MC and data, and remove overlap between datasets, for old version of events_dict.

    Deprecation warning: Should switch to using the LoadedSample class in the future!
    """
    for skey, events in events_dict.items():
        if not Samples.SAMPLES[skey].isData:
            triggered = np.sum(
                [events[hlt][0] for hlt in channel.triggers(year, mc_only=True)], axis=0
            ).astype(bool)
            events_dict[skey] = events[triggered]

    if any(Samples.SAMPLES[skey].isData for skey in events_dict):
        apply_triggers_data_old(events_dict, year, channel)

    return events_dict


def apply_triggers_data(events_dict: dict[str, LoadedSample], year: str, channel: Channel):
    """Apply triggers to data and remove overlap between datasets due to multiple triggers fired in an event."""
    ldataset = channel.lepton_dataset

    # storing triggers fired per dataset
    trigdict = {"jetmet": {}, "tau": {}}
    if channel.isLepton:
        trigdict[ldataset] = {}
        lepton_triggers = utils.list_intersection(
            channel.lepton_triggers(year), channel.triggers(year, data_only=True)
        )

    # JetMET triggers considered in this channel
    jet_triggers = utils.list_intersection(
        HLTs.hlts_by_dataset(year, "JetMET", data_only=True), channel.triggers(year, data_only=True)
    )

    # Tau triggers considered in this channel
    tau_triggers = utils.list_intersection(
        HLTs.hlts_by_dataset(year, "Tau", data_only=True), channel.triggers(year, data_only=True)
    )

    for key, d in trigdict.items():
        d["jets"] = np.sum([events_dict[key].get_var(hlt) for hlt in jet_triggers], axis=0).astype(
            bool
        )
        if key == "jetmet":
            continue

        d["taus"] = np.sum([events_dict[key].get_var(hlt) for hlt in tau_triggers], axis=0).astype(
            bool
        )
        d["taunojets"] = ~d["jets"] & d["taus"]

        if key == "tau":
            continue

        if channel.isLepton:
            d[ldataset] = np.sum(
                [events_dict[key].get_var(hlt) for hlt in lepton_triggers], axis=0
            ).astype(bool)

            d[f"{ldataset}noothers"] = ~d["jets"] & ~d["taus"] & d[ldataset]

            events_dict[ldataset].apply_selection(trigdict[ldataset][f"{ldataset}noothers"])

    # remove overlap
    # print(trigdict["jetmet"])

    events_dict["jetmet"].apply_selection(trigdict["jetmet"]["jets"])
    events_dict["tau"].apply_selection(trigdict["tau"]["taunojets"])

    return events_dict


def apply_triggers(
    events_dict: dict[str, pd.DataFrame | LoadedSample],
    year: str,
    channel: Channel,
):
    """Apply triggers in MC and data, and remove overlap between datasets."""

    if not isinstance(next(iter(events_dict.values())), LoadedSample):
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future!",
            stacklevel=1,
        )
        return apply_triggers_old(events_dict, year, channel)

    # MC
    for _skey, sample in events_dict.items():
        if not sample.sample.isData:
            triggered = np.sum(
                [sample.get_var(hlt) for hlt in channel.triggers(year, mc_only=True)], axis=0
            ).astype(bool)
            sample.events = sample.events[triggered]

    if any(sample.sample.isData for sample in events_dict.values()):
        apply_triggers_data(events_dict, year, channel)

    return events_dict


def delete_columns(
    events_dict: dict[str, LoadedSample | pd.DataFrame], year: str, channel: Channel, triggers=True
):
    if not isinstance(next(iter(events_dict.values())), LoadedSample):
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future!",
            stacklevel=1,
        )
        print("No action taken, events_dict is not a LoadedSample")
        return events_dict

    for sample in events_dict.values():
        isData = sample.sample.isData
        if triggers:
            sample.events.drop(
                columns=list(
                    set(sample.events.columns)
                    - set(channel.triggers(year, data_only=isData, mc_only=not isData))
                )
            )
    return events_dict


def derive_variables(events_dict: dict[str, LoadedSample], channel: Channel, num_fatjets: int = 3):
    """Derive variables for each event."""
    for sample in events_dict.values():
        if "ak8FatJetPNetXbbvsQCDLegacy" not in sample.events:
            Xbb = sample.get_var("ak8FatJetPNetXbbLegacy")
            QCD = sample.get_var("ak8FatJetPNetQCDLegacy")
            Xbb_vs_QCD = np.divide(Xbb, Xbb + QCD, out=np.zeros_like(Xbb), where=(Xbb + QCD) != 0)

            for n in range(num_fatjets):
                sample.events[("ak8FatJetPNetXbbvsQCDLegacy", str(n))] = Xbb_vs_QCD[:, n]

        if channel.key == "hm" and "ak8FatJetParTXtauhtaumvsQCDTop" not in sample.events:
            tauhtaum = sample.get_var("ak8FatJetParTXtauhtaum")
            qcd = sample.get_var("ak8FatJetParTQCD")
            top = sample.get_var("ak8FatJetParTTop")
            tauhtaum_vs_QCDTop = np.divide(
                tauhtaum,
                tauhtaum + qcd + top,
                out=np.zeros_like(tauhtaum),
                where=(tauhtaum + qcd + top) != 0,
            )

            for n in range(num_fatjets):
                sample.events[("ak8FatJetParTXtauhtaumvsQCDTop", str(n))] = tauhtaum_vs_QCDTop[:, n]

        if channel.key == "hm" and "ak8FatJetParTXtauhtaumvsQCD" not in sample.events:
            tauhtaum_vs_QCD = np.divide(
                tauhtaum, tauhtaum + qcd, out=np.zeros_like(tauhtaum), where=(tauhtaum + qcd) != 0
            )
            for n in range(num_fatjets):
                sample.events[("ak8FatJetParTXtauhtaumvsQCD", str(n))] = tauhtaum_vs_QCD[:, n]


def bbtautau_assignment_old(events_dict: dict[str, pd.DataFrame], channel: Channel):
    """Assign bb and tautau jets per each event.

    Deprecation warning: Should switch to using the LoadedSample class in the future!
    """
    bbtt_masks = {}
    for sample_key, sample_events in events_dict.items():
        print(sample_key)
        bbtt_masks[sample_key] = {
            "bb": np.zeros_like(sample_events["ak8FatJetPt"].to_numpy(), dtype=bool),
            "tt": np.zeros_like(sample_events["ak8FatJetPt"].to_numpy(), dtype=bool),
        }

        # assign tautau jet as the one with the highest ParTtautauvsQCD score
        tautau_pick = np.argmax(
            sample_events[f"ak8FatJetParTX{channel.tagger_label}vsQCD"].to_numpy(), axis=1
        )

        # assign bb jet as the one with the highest ParTXbbvsQCD score, but prioritize tautau
        bb_sorted = np.argsort(sample_events["ak8FatJetParTXbbvsQCD"].to_numpy(), axis=1)
        bb_highest = bb_sorted[:, -1]
        bb_second_highest = bb_sorted[:, -2]
        bb_pick = np.where(bb_highest == tautau_pick, bb_second_highest, bb_highest)

        # now convert into boolean masks
        bbtt_masks[sample_key]["bb"][range(len(bb_pick)), bb_pick] = True
        bbtt_masks[sample_key]["tt"][range(len(tautau_pick)), tautau_pick] = True

    return bbtt_masks


def bbtautau_assignment(
    events_dict: dict[str, pd.DataFrame | LoadedSample],
    channel: Channel = None,
    agnostic: bool = False,
):
    """Assign bb and tautau jets per each event."""

    # if channel is none but agnostic is false raise an error
    if channel is None and not agnostic:
        raise ValueError("Channel is required if agnostic is False")

    if not isinstance(next(iter(events_dict.values())), LoadedSample):
        if agnostic:
            raise ValueError("Need to work with LoadedSample if agnostic is True")
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future!",
            stacklevel=1,
        )
        return bbtautau_assignment_old(events_dict, channel)

    for _skey, sample in events_dict.items():
        bbtt_masks = {
            "bb": np.zeros_like(sample.get_var("ak8FatJetPt"), dtype=bool),
            "tt": np.zeros_like(sample.get_var("ak8FatJetPt"), dtype=bool),
        }

        # assign tautau jet as the one with the highest ParTtautauvsQCD score
        if agnostic:
            sig_labels = [ch.tagger_label for ch in CHANNELS.values()]
            num = (
                sample.get_var(f"ak8FatJetParTX{sig_labels[0]}")
                + sample.get_var(f"ak8FatJetParTX{sig_labels[1]}")
                + sample.get_var(f"ak8FatJetParTX{sig_labels[2]}")
            )
            denom = num + sample.get_var("ak8FatJetParTQCD") + sample.get_var("ak8FatJetParTTop")
            combined_score = np.divide(num, denom, out=np.zeros_like(num), where=(num + denom) != 0)
            tautau_pick = np.argmax(combined_score, axis=1)
        else:
            tautau_pick = np.argmax(
                sample.get_var(f"ak8FatJetParTX{channel.tagger_label}vsQCD"), axis=1
            )

        # assign bb jet as the one with the highest ParTXbbvsQCD score, but prioritize tautau
        bb_sorted = np.argsort(sample.get_var("ak8FatJetParTXbbvsQCD"), axis=1)
        bb_highest = bb_sorted[:, -1]
        bb_second_highest = bb_sorted[:, -2]
        bb_pick = np.where(bb_highest == tautau_pick, bb_second_highest, bb_highest)

        # now convert into boolean masks
        bbtt_masks["bb"][range(len(bb_pick)), bb_pick] = True
        bbtt_masks["tt"][range(len(tautau_pick)), tautau_pick] = True

        sample.bb_mask = bbtt_masks["bb"]
        sample.tt_mask = bbtt_masks["tt"]


def _add_bdt_scores(
    events: pd.DataFrame,
    sample_bdt_preds: np.ndarray,
    multiclass: bool,
    all_outs: bool = True,
    jshift: str = "",
):
    """
    Assumes map to be
    Class 0: bbtthe
    Class 1: bbtthh
    Class 2: bbtthm
    Class 3: dyjets
    Class 4: qcd
    Class 5: ttbarhad
    Class 6: ttbarll
    Class 7: ttbarsl
    """

    if jshift != "":
        jshift = "_" + jshift

    if not multiclass:
        events[f"BDTScore{jshift}"] = sample_bdt_preds
    else:
        bg_tot = np.sum(sample_bdt_preds[:, 3:], axis=1)
        hh_score = sample_bdt_preds[:, 0]  # TODO could be de-hardcoded
        hm_score = sample_bdt_preds[:, 1]
        he_score = sample_bdt_preds[:, 2]

        events[f"BDTScorehh{jshift}"] = hh_score / (hh_score + bg_tot)
        events[f"BDTScorehm{jshift}"] = hm_score / (hm_score + bg_tot)
        events[f"BDTScorehe{jshift}"] = he_score / (he_score + bg_tot)

        if all_outs:
            events[f"BDTScoreDY{jshift}"] = sample_bdt_preds[:, 3]
            events[f"BDTScoreQCD{jshift}"] = sample_bdt_preds[:, 4]
            events[f"BDTScoreTThad{jshift}"] = sample_bdt_preds[:, 5]
            events[f"BDTScoreTTll{jshift}"] = sample_bdt_preds[:, 6]
            events[f"BDTScoreTTSL{jshift}"] = sample_bdt_preds[:, 7]


def load_bdt_preds(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    bdt_preds_dir: Path,
    # jec_jmsr_shifts: bool = False,
    all_outs: bool = False,
):
    """
    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.
    If ``jec_jmsr_shifts``, also loads BDT preds for every JEC / JMSR shift in MC.

    Args:
        bdt_preds (str): Path to the bdt_preds .npy file.
        bdt_sample_order (List[str]): Order of samples in the predictions file.

    """
    with (bdt_preds_dir / year / "sample_order.txt").open() as f:
        sample_order_dict = eval(f.read())

    bdt_preds = np.load(f"{bdt_preds_dir}/{year}/preds.npy")

    multiclass = len(bdt_preds.shape) > 1

    # if jec_jmsr_shifts:
    #     shift_preds = {
    #         jshift: np.load(f"{bdt_preds_dir}/{year}/preds_{jshift}.npy")
    #         for jshift in jec_shifts + jmsr_shifts
    #     }

    i = 0
    for sample, num_events in sample_order_dict.items():
        if sample in events_dict:
            events = events_dict[sample]
            assert num_events == len(
                events
            ), f"# of BDT predictions does not match # of events for sample {sample}"

            sample_bdt_preds = bdt_preds[i : i + num_events]
            _add_bdt_scores(events, sample_bdt_preds, multiclass, all_outs)

            # if jec_jmsr_shifts and sample != data_key:
            #     for jshift in jec_shifts + jmsr_shifts:
            #         sample_bdt_preds = shift_preds[jshift][i : i + num_events]
            #         _add_bdt_scores(
            #             events, sample_bdt_preds, multiclass, multisig, all_outs, jshift=jshift
            #         )

        i += num_events

    assert i == len(bdt_preds), f"# events {i} != # of BDT preds {len(bdt_preds)}"


def control_plots(
    events_dict: dict[str, pd.DataFrame],
    channel: Channel,
    sigs: dict[str, Sample],
    bgs: dict[str, Sample],
    control_plot_vars: list[ShapeVar],
    plot_dir: Path,
    year: str,
    bbtt_masks: dict[str, pd.DataFrame] = None,
    weight_key: str = "finalWeight",
    hists: dict = None,
    cutstr: str = "",
    cutlabel: str = "",
    title: str = None,
    selection: dict[str, np.ndarray] = None,
    sig_scale_dict: dict[str, float] = None,
    combine_pdf: bool = True,
    plot_ratio: bool = True,
    plot_significance: bool = False,
    same_ylim: bool = False,
    show: bool = False,
    log: tuple[bool, str] = "both",
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.
        sig_splits: split up signals into different plots (in case there are too many for one)
        HEM2d: whether to plot 2D hists of FatJet phi vs eta for bb and VV jets as a check for HEM cleaning.
        plot_ratio: whether to plot the data/MC ratio.
        plot_significance: whether to plot the significance as well as the ratio plot.
        same_ylim: whether to use the same y-axis limits for all plots.
        log: True or False if plot on log scale or not - or "both" if both.
    """

    from PyPDF2 import PdfMerger

    if hists is None:
        hists = {}
    if sig_scale_dict is None:
        sig_scale_dict = {sig_key: 2e5 for sig_key in sigs}

    print(control_plot_vars)
    print(selection)
    print(list(events_dict.keys()))

    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            hists[shape_var.var] = putils.singleVarHist(
                events_dict,
                shape_var,
                channel,
                bbtt_masks=bbtt_masks,
                weight_key=weight_key,
                selection=selection,
            )

    print(hists)

    ylim = (np.max([h.values() for h in hists.values()]) * 1.05) if same_ylim else None

    with (plot_dir / "hists.pkl").open("wb") as f:
        pickle.dump(hists, f)

    do_log = [True, False] if log == "both" else [log]

    for log, logstr in [(False, ""), (True, "_log")]:
        if log not in do_log:
            continue

        merger_control_plots = PdfMerger()

        for shape_var in control_plot_vars:
            pylim = np.max(hists[shape_var.var].values()) * 1.4 if ylim is None else ylim

            name = f"{plot_dir}/{cutstr}{shape_var.var}{logstr}.pdf"
            plotting.ratioHistPlot(
                hists[shape_var.var],
                year,
                channel,
                list(sigs.keys()),
                list(bgs.keys()),
                name=name,
                title=title,
                sig_scale_dict=sig_scale_dict if not log else None,
                plot_significance=plot_significance,
                significance_dir=shape_var.significance_dir,
                cutlabel=cutlabel,
                show=show,
                log=log,
                ylim=pylim if not log else 1e15,
                plot_ratio=plot_ratio,
                cmslabel="Work in progress",
                leg_args={"fontsize": 18},
            )
            merger_control_plots.append(name)

        if combine_pdf:
            merger_control_plots.write(f"{plot_dir}/{cutstr}{year}{logstr}_ControlPlots.pdf")

        merger_control_plots.close()

    return hists


def get_templates(
    events_dict: dict[str, LoadedSample],
    year: str,
    sig_keys: list[str],
    bg_keys: list[str],
    channel: Channel,
    shape_vars: list[ShapeVar],
    systematics: dict,  # noqa: ARG001
    template_dir: Path = "",
    plot_dir: Path = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    plot_sig_keys: list[str] = None,
    sig_scale_dict: dict = None,
    weight_shifts: dict = None,
    jshift: str = "",
    plot_shifts: bool = False,
    pass_ylim: int = None,
    fail_ylim: int = None,
    blind: bool = True,
    blind_pass: bool = False,
    plot_data: bool = True,
    show: bool = False,
) -> dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) TODO: Applies the Txbb scale factor in the pass region,
    (3) TODO: Calculates trigger uncertainty,
    (4) TODO: Calculates weight variations if ``weight_shifts`` is not empty (and ``jshift`` is ""),
    (5) TODO: Takes JEC / JSMR shift into account if ``jshift`` is not empty,
    (6) Saves a plot of each (if ``plot_dir`` is not "").

    Args:
        selection_region (Dict[str, Dict]): Dictionary of ``Region``s including cuts and labels.
        bg_keys (list[str]): background keys to plot.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """
    import time

    start = time.time()

    if weight_shifts is None:
        weight_shifts = {}

    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates = {}

    # do TXbb SFs + uncs. for signals and Hbb samples only
    # txbb_samples = sig_keys + [key for key in bg_keys if key in hbb_bg_keys]

    selection_regions = Regions.get_selection_regions(channel)

    for rname, region in selection_regions.items():
        pass_region = rname.startswith("pass")

        print(f"{rname} Region: {time.time() - start:.2f}")

        if not do_jshift:
            print(rname)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region.cuts,
            events_dict,
            prev_cutflow=prev_cutflow,
            jshift=jshift,
            weight_key=weight_key,
        )
        print(f"Selection: {time.time() - start:.2f}")

        if template_dir != "":
            cf.to_csv(f"{template_dir}/cutflows/{year}/{rname}_cutflow{jlabel}.csv")

        # trigger uncertainties
        # if not do_jshift:
        #     systematics[year][rname] = {}
        #     total, total_err = corrections.get_uncorr_trig_eff_unc(events_dict, bb_masks, year, sel)
        #     systematics[year][rname]["trig_total"] = total
        #     systematics[year][rname]["trig_total_err"] = total_err
        #     print(f"Trigger SF Unc.: {total_err / total:.3f}\n")

        # ParticleNetMD Txbb and ParT LP SFs
        sig_events = {}
        for sig_key in sig_keys:
            lsample = events_dict[sig_key]
            sig_events[sig_key] = lsample.copy_from_selection(sel[sig_key], do_deepcopy=True)

            # if region.signal:
            #     corrections.apply_txbb_sfs(
            #         sig_events[sig_key], sig_bb_mask, year, weight_key, do_shifts=not do_jshift
            #     )

            #     print(f"Txbb SFs: {time.time() - start:.2f}")

        # set up samples
        hist_samples = list(events_dict.keys())

        # if not do_jshift:
        #     # add all weight-based variations to histogram axis
        #     for shift in ["down", "up"]:
        #         if region.signal:
        #             for sig_key in sig_keys:
        #                 hist_samples.append(f"{sig_key}_txbb_{shift}")

        #         for wshift, wsyst in weight_shifts.items():
        #             # if year in wsyst.years:
        #             # add to the axis even if not applied to this year to make it easier to sum later
        #             for wsample in wsyst.samples:
        #                 if wsample in events_dict:
        #                     hist_samples.append(f"{wsample}_{wshift}_{shift}")

        # histograms
        h = Hist(
            hist.axis.StrCategory(hist_samples + [data_key], name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="weight",
        )

        # fill histograms
        for skey, lsample in events_dict.items():
            if skey in sig_keys:
                sample = sig_events[skey]
            else:
                sample = lsample.copy_from_selection(sel[skey])

            if not len(sample.events):
                continue

            fill_data = utils.get_fill_data(
                sample, shape_vars, jshift=jshift if sample.sample.isData else None
            )
            weight = sample.get_var(weight_key)

            # breakpoint()
            h.fill(Sample=skey, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if skey in wsyst.samples and year in wsyst.years:
                        if wshift not in ["scale", "pdf"]:
                            # fill histogram with weight variations
                            for shift_key, shift in [("Down", "down"), ("Up", "up")]:
                                h.fill(
                                    Sample=f"{skey}_{wshift}_{shift}",
                                    **fill_data,
                                    weight=sample.get_var(f"weight_{wshift}{shift_key}"),
                                )
                        else:
                            # get histograms for all QCD scale and PDF variations
                            whists = utils.get_qcdvar_hists(sample, shape_vars, fill_data, wshift)

                            if wshift == "scale":
                                # renormalization / factorization scale uncertainty is the max/min envelope of the variations
                                shape_up = np.max(whists.values(), axis=0)
                                shape_down = np.min(whists.values(), axis=0)
                            else:
                                # pdf uncertainty is the norm of each variation (corresponding to 103 eigenvectors) - nominal
                                nom_vals = h[sample, ...].values()
                                abs_unc = np.linalg.norm(
                                    (whists.values() - nom_vals), axis=0
                                )  # / np.sqrt(103)
                                # cap at 100% uncertainty
                                rel_unc = np.clip(abs_unc / nom_vals, 0, 1)
                                shape_up = nom_vals * (1 + rel_unc)
                                shape_down = nom_vals * (1 - rel_unc)

                            h.values()[
                                utils.get_key_index(h, f"{skey}_{wshift}_up"), ...
                            ] = shape_up
                            h.values()[
                                utils.get_key_index(h, f"{skey}_{wshift}_down"), ...
                            ] = shape_down

        print(f"Histograms: {time.time() - start:.2f}")

        # sum data histograms
        data_hist = sum(h[skey, ...] for skey in channel.data_samples)
        h.view(flow=True)[utils.get_key_index(h, data_key)].value = data_hist.values(flow=True)
        h.view(flow=True)[utils.get_key_index(h, data_key)].variance = data_hist.variances(
            flow=True
        )

        print(h)

        if region.signal and blind:
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        # if region.signal and not do_jshift:
        #     for sig_key in sig_keys:
        #         if not len(sig_events[sig_key].events):
        #             continue

        #         # ParticleNetMD Txbb SFs
        #         fill_data = utils.get_fill_data(sig_events[sig_key], shape_vars)
        #         for shift in ["down", "up"]:
        #             h.fill(
        #                 Sample=f"{sig_key}_txbb_{shift}",
        #                 **fill_data,
        #                 weight=sig_events[sig_key].get_var(f"{weight_key}_txbb_{shift}"),
        #             )

        templates[rname + jlabel] = h

        ################################
        # Plot templates incl variations
        ################################

        if plot_dir != "" and (not do_jshift or plot_shifts):
            print(f"Plotting templates: {time.time() - start:.2f}")
            if plot_sig_keys is None:
                plot_sig_keys = sig_keys

            if sig_scale_dict is None:
                sig_scale_dict = {skey: 10 for skey in plot_sig_keys}

            title = (
                f"{region.label} Region Pre-Fit Shapes"
                if not do_jshift
                else f"{region.label} Region {jshift} Shapes"
            )

            # don't plot qcd in the pass regions
            # if pass_region:
            #     p_bg_keys = [key for key in bg_keys if key != qcd_key]
            # else:
            p_bg_keys = bg_keys

            for i, shape_var in enumerate(shape_vars):
                plot_params = {
                    "hists": h.project(0, i + 1),
                    "sig_keys": plot_sig_keys,
                    "sig_scale_dict": (
                        {key: sig_scale_dict[key] for key in plot_sig_keys}
                        if region.signal
                        else None
                    ),
                    "channel": channel,
                    "show": show,
                    "year": year,
                    "ylim": pass_ylim if pass_region else fail_ylim,
                    "plot_data": (not (rname == "pass" and blind_pass)) and plot_data,
                    "leg_args": {"fontsize": 22, "ncol": 2},
                }

                plot_name = (
                    f"{plot_dir}/"
                    f"{'jshifts/' if do_jshift else ''}"
                    f"{rname}_region_{shape_var.var}"
                )

                plotting.ratioHistPlot(
                    **plot_params,
                    bg_keys=p_bg_keys,
                    title=title,
                    name=f"{plot_name}{jlabel}.pdf",
                    plot_ratio=plot_data,
                )

                if not do_jshift and plot_shifts:
                    plot_name = f"{plot_dir}/wshifts/" f"{rname}_region_{shape_var.var}"

                    for wshift, wsyst in weight_shifts.items():
                        plotting.ratioHistPlot(
                            **plot_params,
                            bg_keys=p_bg_keys,
                            syst=(wshift, wsyst.samples),
                            title=f"{region.label} Region {wsyst.label} Unc.",
                            name=f"{plot_name}_{wshift}.pdf",
                            plot_ratio=False,
                            reorder_legend=False,
                        )

                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            plotting.ratioHistPlot(
                                **plot_params,
                                bg_keys=p_bg_keys,  # don't plot QCD
                                syst=(wshift, wsyst.samples),
                                variation=shift,
                                title=f"{region.label} Region {wsyst.label} Unc. {skey} Shapes",
                                name=f"{plot_name}_{wshift}_{shift}.pdf",
                                plot_ratio=False,
                            )

                    if region.signal:
                        plotting.ratioHistPlot(
                            **plot_params,
                            bg_keys=p_bg_keys,
                            sig_err="txbb",
                            title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                            name=f"{plot_name}_txbb.pdf",
                        )

    return templates


def save_templates(
    templates: dict[str, Hist],
    template_file: Path,
    blind: bool,
    shape_vars: list[ShapeVar],
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    if blind:
        from copy import deepcopy

        blind_window = shape_vars[0].blind_window

        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}MCBlinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channel",
        required=True,
        choices=list(Samples.CHANNELS.keys()),
        help="channel",
        type=str,
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--bg-data-dirs",
        default=[],
        help="path to skimmed background parquets, if different from other data",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--signal-data-dirs",
        default=[],
        help="path to skimmed signal parquets, if different from other data",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--year",
        required=True,
        choices=hh_vars.years,
        type=str,
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    parser.add_argument(
        "--template-dir",
        help="If saving templates, path to file to save them in. If scanning, directory to save in.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--templates-name",
        help="If saving templates, optional name for folder (comes under cuts directory if scanning).",
        default="",
        type=str,
    )

    add_bool_arg(parser, "control-plots", "make control plots", default=False)

    add_bool_arg(parser, "blinded", "blind the data in the Higgs mass window", default=True)
    add_bool_arg(parser, "templates", "save m_bb templates", default=False)
    add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    add_bool_arg(parser, "do-jshifts", "Do JEC/JMC variations", default=True)
    add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)
    add_bool_arg(
        parser, "override-systs", "Override saved systematics file if it exists", default=False
    )

    parser.add_argument(
        "--sigs",
        help="specify signal samples. By default, will use the samples defined in `hh_vars`.",
        nargs="*",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--bgs",
        help="specify background samples",
        nargs="*",
        default=None,
        type=str,
    )

    add_bool_arg(parser, "read-sig-samples", "read signal samples from directory", default=False)
    add_bool_arg(parser, "data", "include data", default=True)
    add_bool_arg(parser, "filters", "apply filters", default=True)

    parser.add_argument(
        "--control-plot-vars",
        help="Specify control plot variables to plot. By default plots all.",
        default=[],
        nargs="*",
        type=str,
    )

    args = parser.parse_args()

    if args.control_plots:
        raise NotImplementedError("Control plots not implemented")

    if not args.signal_data_dirs and args.data_dir:
        args.signal_data_dirs = [args.data_dir]

    if not args.bg_data_dirs and args.data_dir:
        args.bg_data_dirs = [args.data_dir]

    save_args = deepcopy(args)

    # save args in args.plot_dir and args.template_dir if they exit
    if args.plot_dir:
        args.plot_dir = Path(args.plot_dir) / args.channel / args.year
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        with (args.plot_dir / "args.json").open("w") as f:
            json.dump(save_args.__dict__, f, indent=4)

    if args.template_dir:
        args.template_dir = Path(args.template_dir) / args.channel
        (args.template_dir / "cutflows" / args.year).mkdir(parents=True, exist_ok=True)
        with (args.template_dir / "args.json").open("w") as f:
            json.dump(save_args.__dict__, f, indent=4)

    print(args)
    return args


if __name__ == "__main__":
    mpl.use("Agg")
    args = parse_args()
    main(args)
