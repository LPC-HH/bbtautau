"""
Postprocessing functions for bbtautau.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotting
import Samples
import utils as putils
from boostedhh import utils
from boostedhh.utils import Sample, ShapeVar

from bbtautau.bbtautau_utils import Channel
from bbtautau.HLTs import HLTs

base_filters = [
    ("('ak8FatJetPt', '0')", ">=", 250),
    ("('ak8FatJetPNetmassLegacy', '0')", ">=", 50),
    ("('ak8FatJetPt', '1')", ">=", 200),
    # ("('ak8FatJetMsd', '0')", ">=", msd_cut),
    # ("('ak8FatJetMsd', '1')", ">=", msd_cut),
    # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
]


control_plot_vars = (
    [
        ShapeVar(var="MET_pt", label=r"$p^{miss}_T$ [GeV]", bins=[20, 0, 300]),
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
            label=r"ParT XbbvsQCD$",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetParTXbbvsQCDTop
    + [
        ShapeVar(
            var=f"ak8FatJetParTXbbvsQCDTop{i}",
            label=r"ParT XbbvsQCDTop$",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetPNetXbbvsQCDLegacy
    + [
        ShapeVar(
            var=f"ak8FatJetPNetXbbvsQCDLegacy{i}",
            label=r"PNet Legacy XbbvsQCD$",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    #  nElectrons
    + [ShapeVar(var="nElectrons", label=r"Number of Electrons", bins=[3, 0, 2])]
    #  nMuons
    + [ShapeVar(var="nMuons", label=r"Number of Muons", bins=[3, 0, 2])]
    #  nTaus
    + [ShapeVar(var="nTaus", label=r"Number of Taus", bins=[3, 0, 2])]
    #  nBoostedTaus
    + [ShapeVar(var="nBoostedTaus", label=r"Number of Boosted Taus", bins=[3, 0, 2])]
)


def bb_filters(num_fatjets: int = 3):
    filters = [
        # roughly, 85% signal efficiency, 2% QCD efficiency (pT: 250-400, mSD:0-250, mRegLegacy:40-250)
        base_filters + [(f"('ak8FatJetPNetXbbLegacy', '{n}')", ">=", 0.3)]
        for n in range(num_fatjets)
    ]
    return filters


def trigger_filter(
    triggers: dict[str, list[str]],
    base_filters: list[tuple] = base_filters,
    fast_mode: bool = False,
) -> dict[str, dict[str, list[list[tuple]]]]:
    """
    creates a list of filters for each trigger in the list of triggers. It is granular to the usual {"data" / "signal" : {year : [triggers],...} structure: triggers = {"data": {"2022" : [...] , ...}, "signal": { [...]}.
    """
    if fast_mode:
        base_filters += [("('ak8FatJetPNetXbbLegacy', '0')", ">=", 0.95)]

    filters_dict = {}
    for dtype, years in triggers.items():
        filters_dict[dtype] = {}
        for year, trigger_list in years.items():
            filters_dict[dtype][year] = [
                base_filters + [(f"('{trigger}', '0')", "==", 1)] for trigger in trigger_list
            ]
    # print(f"\n\nTrigger filters_dict for data: {filters_dict['data']['2022'][0]}")
    return filters_dict


def get_columns(
    year: str,
    channel: Channel,
    triggers: bool = True,
    legacy_taggers: bool = True,
    ParT_taggers: bool = True,
    num_fatjets: int = 3,
):

    columns_data = [("weight", 1), ("ak8FatJetPt", num_fatjets)]

    # common columns
    if legacy_taggers:
        columns_data += [
            ("ak8FatJetPNetXbbLegacy", num_fatjets),
            ("ak8FatJetPNetQCDLegacy", num_fatjets),
            ("ak8FatJetPNetmassLegacy", num_fatjets),
            ("ak8FatJetParTmassResApplied", num_fatjets),
            ("ak8FatJetParTmassVisApplied", num_fatjets),
        ]

    if ParT_taggers:
        for branch in [
            f"ak8FatJetParT{key}" for key in Samples.qcdouts + Samples.topouts + Samples.sigouts
        ]:
            columns_data.append((branch, num_fatjets))

    columns_mc = copy.deepcopy(columns_data)

    if triggers:
        for branch in channel.triggers(year, data_only=True):
            columns_data.append((branch, 1))
        for branch in channel.triggers(year, mc_only=True):
            columns_mc.append((branch, 1))

    # signal-only columns
    columns_signal = copy.deepcopy(columns_mc)

    columns_signal += [
        ("GenTauhh", 1),
        ("GenTauhmu", 1),  # TODO this will need to be changed to GenTauhm
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
    filters_dict: dict[str, dict[str, list[list[tuple]]]] = None,
    load_columns: dict[str, list[tuple]] = None,
    load_bgs: bool = False,
    load_just_bbtt: bool = False,
):
    events_dict = {}

    samples = Samples.SAMPLES.copy()
    signals = Samples.SIGNALS.copy()

    if load_just_bbtt:  # quite ad hoc but should become obsolete
        del samples["vbfbbtt-k2v0"]
        signals.remove("vbfbbtt-k2v0")

    # remove unnecessary data samples
    for key in Samples.DATASETS + (not load_bgs) * Samples.BGS:
        if (key in samples) and (key not in channel.data_samples):
            del samples[key]

    # load only the specified columns
    if load_columns is not None:
        for sample in samples.values():
            sample.load_columns = load_columns[sample.get_type()]

    # load samples
    for key, sample in samples.items():
        if isinstance(filters_dict, dict):
            filters = filters_dict[sample.get_type()][year]
        else:
            filters = filters_dict

        if sample.selector is not None:
            events_dict[key] = utils.load_sample(
                sample,
                year,
                paths,
                filters,
            )

    # keep only the specified bbtt channel
    for signal in signals:
        # quick fix due to old naming still in samples
        events_dict[f"{signal}{channel.key}"] = events_dict[signal][
            events_dict[signal][f"GenTau{channel.key}" + "u" * (channel.key == "hm")][0]
        ]
        del events_dict[signal]

    return events_dict


def apply_triggers_data(events_dict: dict[str, pd.DataFrame], year: str, channel: Channel):
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


def apply_triggers(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    channel: Channel,
):
    """Apply triggers in MC and data, and remove overlap between datasets."""
    # MC
    for skey, events in events_dict.items():
        if not Samples.SAMPLES[skey].isData:
            triggered = np.sum(
                [events[hlt][0] for hlt in channel.triggers(year, mc_only=True)], axis=0
            ).astype(bool)
            events_dict[skey] = events[triggered]

    if any(Samples.SAMPLES[skey].isData for skey in events_dict):
        apply_triggers_data(events_dict, year, channel)

    return events_dict


def delete_columns(
    events_dict: dict[str, pd.DataFrame], year: str, channel: Channel, triggers=True
):
    for sample_key, sample_events in events_dict.items():
        print(sample_key, len(sample_events))
        isData = Samples.SAMPLES[sample_key].isData
        if triggers:
            sample_events.drop(
                columns=list(
                    set(sample_events.columns)
                    - set(channel.triggers(year, data_only=isData, mc_only=not isData))
                )
            )
    return events_dict


def control_plots(
    events_dict: dict[str, pd.DataFrame],
    channel: Channel,
    # bb_masks: dict[str, pd.DataFrame],
    sigs: dict[str, Sample],
    bgs: dict[str, Sample],
    control_plot_vars: list[ShapeVar],
    plot_dir: Path,
    year: str,
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
                events_dict, shape_var, channel, weight_key=weight_key, selection=selection
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
