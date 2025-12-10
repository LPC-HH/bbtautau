"""
General utilities for postprocessing.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import hist
import numpy as np
import pandas as pd
import vector
from boostedhh import utils
from boostedhh.hh_vars import data_key
from boostedhh.utils import PAD_VAL, Sample, ShapeVar
from hist import Hist
from joblib import Parallel, delayed

from bbtautau.HLTs import HLTs
from bbtautau.postprocessing import Samples
from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.postprocessing.bdt_utils import compute_or_load_bdt_preds
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.userConfig import BDT_EVAL_DIR, DATA_PATHS, MODEL_DIR

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


def concatenate_loaded_samples(
    loaded_samples: list[LoadedSample], out_sample: Sample = None
) -> LoadedSample:
    """Concatenate multiple LoadedSample objects. Checks if the Samples are the same, or raises an error."""

    # check if the samples are the same
    if out_sample is None and not all(
        ls.sample.label == loaded_samples[0].sample.label for ls in loaded_samples
    ):
        raise ValueError(
            "Samples are not all the same, need to indicate the Sample object for the output"
        )

    # check if the masks exist
    if not all(ls.bb_mask is not None for ls in loaded_samples):
        raise ValueError("bb_masks are not set for some samples")
    if not all(ls.tt_mask is not None for ls in loaded_samples):
        raise ValueError("tt_masks are not set for some samples")

    new_events = pd.concat([ls.events for ls in loaded_samples])
    new_bb_mask = np.concatenate([ls.bb_mask for ls in loaded_samples])
    new_tt_mask = np.concatenate([ls.tt_mask for ls in loaded_samples])

    return LoadedSample(
        sample=out_sample if out_sample is not None else loaded_samples[0].sample,
        events=new_events,
        bb_mask=new_bb_mask,
        tt_mask=new_tt_mask,
    )


def concatenate_years(
    events_dict: dict[str, dict[str, LoadedSample]], years: list[str] = None
) -> dict[str, dict[str, LoadedSample]]:
    """Collapse typical events_dict[year][sample] = LoadedSample into events_dict[sample] = LoadedSample."""

    if years is None:
        years = list(events_dict.keys())

    # Check that all years have the same set of samples
    sample_sets = [set(events_dict[year].keys()) for year in years]
    first_set = sample_sets[0]
    for i, sset in enumerate(sample_sets[1:], 1):
        if sset != first_set:
            raise ValueError(
                f"Sample sets differ between years: {years[0]} has {first_set}, {years[i]} has {sset}"
            )

    new_events_dict = {}

    # Get all sample keys (can take from any year)
    all_samples = list(sample_sets[0])

    for sample in all_samples:
        loaded_samples = []
        for year in years:
            loaded_samples.append(events_dict[year][sample])
        new_events_dict[sample] = concatenate_loaded_samples(loaded_samples)

    return new_events_dict


def extract_base_signal_key(sig_key_with_channel: str) -> str:
    """Extract base signal key from signal key with channel suffix.

    Examples:
        "ggfbbtthh" -> "ggfbbtt"
        "vbfbbtthm" -> "vbfbbtt"
    """
    for base_sig in ["ggfbbtt", "vbfbbtt", "vbfbbtt-k2v0"]:
        if sig_key_with_channel.startswith(base_sig):
            return base_sig
    # If no match, assume it's already a base key
    return sig_key_with_channel


def base_filter(test_mode: bool = False):
    """
    Returns the base filters for the data, signal, and background samples.
    """

    base_filters = copy.deepcopy(base_filters_default)
    if test_mode:
        for i in range(len(base_filters)):
            base_filters[i] += [
                ("('ak8FatJetPhi', '0')", ">=", 2.9),
                # ("('ak8FatJetParTXbbvsQCD', '0')", ">=", 0.7),
            ]

    return {"data": base_filters, "signal": base_filters, "bg": base_filters}


def bb_filters(
    in_filters: dict[str, list[tuple]] = None, num_fatjets: int = 3, bb_cut: float = 0.3
):
    """
    0.3 corresponds to roughly, 85% signal efficiency, 2% QCD efficiency (pT: 250-400, mSD:0-250, mRegLegacy:40-250)
    """
    if in_filters is None:
        in_filters = base_filter()

    filters = {}
    for dtype, ifilters_bydtype in in_filters.items():
        filters[dtype] = [
            ifilter + [(f"('ak8FatJetParTXbbvsQCDTop', '{n}')", ">=", bb_cut)]
            for n in range(num_fatjets)
            for ifilter in ifilters_bydtype
        ]

    return filters


def tt_filters(
    # if channel is None, it is agnostic selection
    channel: Channel = None,
    in_filters: dict[str, list[tuple]] = None,
    num_fatjets: int = 3,
    tt_cut: float = 0.3,
):
    if in_filters is None:
        in_filters = base_filter()

    filters = {}

    # Agnostic selection: at least one jet with ParT score in any channel is required. Note that cannot sum scores together at this stage.
    if channel is None:
        for dtype, ifilters_bydtype in in_filters.items():
            filters[dtype] = [
                ifilter + [(f"('ak8FatJetParTX{ch.tagger_label}vsQCDTop', '{n}')", ">=", tt_cut)]
                for n in range(num_fatjets)
                for ifilter in ifilters_bydtype
                for ch in CHANNELS.values()
            ]
    else:
        for dtype, ifilters_bydtype in in_filters.items():
            filters[dtype] = [
                ifilter
                + [(f"('ak8FatJetParTX{channel.tagger_label}vsQCDTop', '{n}')", ">=", tt_cut)]
                for n in range(num_fatjets)
                for ifilter in ifilters_bydtype
            ]

    return filters


def get_columns(
    year: str,
    triggers_in_channel: Channel = None,
    legacy_taggers: bool = True,
    ParT_taggers: bool = True,
    leptons: bool = True,
    other: bool = True,
    lowercase_jetaway: bool = False,
):

    columns_data = [
        ("weight", 1),
        ("ak8FatJetPt", 3),
        ("ak8FatJetEta", 3),
        ("ak8FatJetPhi", 3),
        ("ak4JetPt", 3),
        ("ak4JetEta", 3),
        ("ak4JetPhi", 3),
        ("ak4JetMass", 3),
    ]
    if lowercase_jetaway:
        columns_data += [
            ("ak4JetAwayPt", 2),
            ("ak4JetAwayEta", 2),
            ("ak4JetAwayPhi", 2),
            ("ak4JetAwayMass", 2),
        ]
    else:
        columns_data += [
            ("AK4JetAwayPt", 2),
            ("AK4JetAwayEta", 2),
            ("AK4JetAwayPhi", 2),
            ("AK4JetAwayMass", 2),
        ]

    # common columns
    if legacy_taggers:
        columns_data += [
            ("ak8FatJetPNetXbbLegacy", 3),
            ("ak8FatJetPNetQCDLegacy", 3),
            ("ak8FatJetPNetmassLegacy", 3),
            ("ak8FatJetParTmassResApplied", 3),
            ("ak8FatJetParTmassVisApplied", 3),
            ("ak8FatJetMsd", 3),
        ]

    if ParT_taggers:
        for branch in (
            [f"ak8FatJetParT{key}" for key in Samples.qcdouts + Samples.topouts + Samples.sigouts]
            + [
                f"ak8FatJetParT{key}vsQCD" for key in Samples.sigouts
            ]  # remove exception in muon channel, was only necessary in old ntuples
            + [f"ak8FatJetParT{key}vsQCDTop" for key in Samples.sigouts]
        ):
            columns_data.append((branch, 3))

    if leptons:
        columns_data += [
            ("ElectronPt", 2),
            ("ElectronEta", 2),
            ("ElectronPhi", 2),
            ("Electroncharge", 2),
            ("ElectronMass", 2),
            ("MuonPt", 2),
            ("MuonEta", 2),
            ("MuonPhi", 2),
            ("Muoncharge", 2),
            ("MuonMass", 2),
        ]
    if other:
        columns_data += [
            ("nFatJets", 1),
            ("METPt", 1),
            ("METPhi", 1),
            ("ht", 1),
            ("nElectrons", 1),
            ("nMuons", 1),
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
    paths: dict[str],
    signals: list[str],
    channels: list[Channel],
    samples: dict[str, Sample] = None,
    additional_samples: dict[str, Sample] = None,
    restrict_data_to_channel: bool = True,
    filters_dict: dict[str, list[list[tuple]]] = None,
    load_columns: dict[str, list[tuple]] = None,
    load_bgs: bool = False,
    load_data: bool = True,
    loaded_samples: bool = True,
    multithread: bool = True,
    loader_n_jobs: int | None = None,
) -> dict[str, LoadedSample | pd.DataFrame]:
    """
    Loads and preprocesses physics event samples for a given year and analysis channel.

    This function is designed to flexibly load datasets for different physics samples (signal, background, data)
    according to user-specified filters, columns, and channel restrictions. It supports returning results either
    as plain DataFrames (legacy behavior) or as `LoadedSample` objects for enhanced data encapsulation.

    Parameters
    ----------
    year : str
        The data-taking year for which to load the samples (e.g., "2018").
    channels : list[Channel]
        The analysis channel definition, containing metadata and configuration. Indicates which signal channel samples are desired in the output. Can affect the data loading if restrict_data_to_channel is True.
    paths : dict[str]
        Dictionary mapping sample names to input file paths. First level keys are "data", "signal", "bg". Second level keys are years.
    samples : dict[str, Sample], optional
        Dictionary of sample objects to load. If None, uses the default `Samples.SAMPLES`. If samples are provided, the flags load_bgs, load_data, restrict_data_to_channel are ignored.
    additional_samples : dict[str, Sample], optional
        Dictionary of additional samples to load, without overriding the default samples. Useful when not using `samples` arg, and taking channel-specific samples from `CHANNELS`.
    restrict_data_to_channel : bool, optional
        If True, restricts data loading to samples associated only with the specified channels (default: True). Practical to be used without providing samples.
    filters_dict : dict[str, list[list[tuple]]], optional
        Dictionary of filter definitions to be applied per sample type. If not provided or not a dictionary, no filters are applied.
    load_columns : dict[str, list[tuple]], optional
        Specifies which columns to load for each sample type. If provided, limits data loading to these columns.
    load_bgs : bool, optional
        If True, includes background samples in the loading process (default: False).
    load_data : bool, optional
        If True, includes data samples in the loading process (default: True).
    load_just_ggf : bool, optional
        If True, loads only the "ggf" signal samples, excluding "vbf" (default: False). Used for specialized studies.
    loaded_samples : bool, optional
        If True, returns results as `LoadedSample` objects. If False, returns plain DataFrames (deprecated).

    Returns
    -------
    dict[str, LoadedSample | pd.DataFrame]
        Dictionary mapping sample names (or signal-channel keys) to either `LoadedSample` objects
        or pandas DataFrames, depending on `loaded_samples`.

    Notes
    -----
    - **Deprecation warning:** The legacy DataFrame-based output (when `loaded_samples=False`) is deprecated and will be removed in the future. It is recommended to use `LoadedSample` outputs for better compatibility.
    - If filtering or column selection is not provided, the function loads all available data for the relevant samples.
    - When `restrict_to_channel` is True, only channel-specific samples are loaded.
    - The function automatically adjusts for specific study types, such as restricting to GGF-only signals.
    - Signal samples are remapped by channel at the end of the function for downstream compatibility.


    """

    # Legacy check from when samples was a single channel
    if not isinstance(channels, list):
        warnings.warn(
            "Deprecation warning: Should switch to using a list of channels in the future!",
            stacklevel=1,
        )
        channels = [channels]

    if not loaded_samples:
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future, by setting loaded_samples=True!",
            stacklevel=1,
        )

    if not isinstance(filters_dict, dict):
        print("Warning: filters_dict is not a dictionary. Not applying filters.")
        filters_dict = None

    events_dict = {}

    if samples is None:
        samples = Samples.SAMPLES.copy()

        if not load_bgs:
            for key in Samples.BGS:
                if key in samples:
                    del samples[key]

        if not load_data:
            for key in Samples.DATASETS:
                if key in samples:
                    del samples[key]

        for key, sample in copy.deepcopy(samples).items():
            if sample.isSignal and key not in signals:
                del samples[key]

    else:
        print("Note: samples are provided. Ignoring load_samples kwargs load_bgs, load_data.")

    if restrict_data_to_channel:
        # remove unnecessary data samples
        for channel in channels:
            for key in Samples.DATASETS:
                if (key in samples) and (key not in channel.data_samples):
                    del samples[key]

    if additional_samples is not None:
        for key, sample in additional_samples.items():
            samples[key] = sample

    print(
        f"Loading year {year}, samples",
        [key for key in samples if samples[key].selector is not None],
    )

    # load only the specified columns
    if load_columns is not None:
        for sample in samples.values():
            sample.load_columns = load_columns[sample.get_type()]

    if multithread:
        if not loaded_samples:
            warnings.warn(
                "Deprecation warning: Switch to using the LoadedSample class. Loading failed.",
                stacklevel=1,
            )
            return None

        # Threaded loading avoids process pickling overhead; cap workers sensibly
        import os

        n_jobs = loader_n_jobs
        if n_jobs is None:
            n_jobs = min(max(1, 2 * len(samples)), os.cpu_count() or 1)

        data_ = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(LoadedSample)(
                sample=sample,
                events=utils.load_sample(
                    sample,
                    year,
                    paths,
                    filters_dict[sample.get_type()] if filters_dict is not None else None,
                ),
            )
            for sample in samples.values()
            if sample.selector is not None  # this line is key to only load bbtt once
        )

        keys = [key for key, sample in samples.items() if sample.selector is not None]
        events_dict = dict(zip(keys, data_))

    else:
        # load samples (legacy)
        for key, sample in samples.items():

            filters = filters_dict[sample.get_type()] if filters_dict is not None else None

            if sample.selector is not None:

                events = utils.load_sample(
                    sample,
                    year,
                    paths,
                    filters,
                )
                print("Loaded sample", key, "in year", year)

                if not loaded_samples:
                    events_dict[key] = events
                else:
                    events_dict[key] = LoadedSample(sample=sample, events=events)

    # keep only the specified signal channels
    for signal in signals:
        for channel in channels:
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
    events_dict: dict[str, LoadedSample | pd.DataFrame],
    year: str,
    channels: list[Channel],
    triggers=True,
):
    if not isinstance(next(iter(events_dict.values())), LoadedSample):
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future!",
            stacklevel=1,
        )
        print("No action taken, events_dict is not a LoadedSample")
        return events_dict

    if not isinstance(channels, list):
        warnings.warn(
            "Deprecation warning: Should switch to using a list of channels in the future!",
            stacklevel=1,
        )
        channels = [channels]

    for sample in events_dict.values():
        isData = sample.sample.isData
        if triggers:
            sample.events.drop(
                columns=(
                    set(sample.events.columns)
                    - {
                        trigger
                        for channel in channels
                        for trigger in channel.triggers(year, data_only=isData, mc_only=not isData)
                    }
                )
            )
    return events_dict


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
            ) / 3
            denom = num + sample.get_var("ak8FatJetParTQCD")
            combined_score = np.divide(
                num, denom, out=np.zeros_like(num), where=((num != PAD_VAL) & (denom != 0))
            )
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


# decorators give problems
# @numba.vectorize(
#     [
#         numba.float32(numba.float32, numba.float32),
#         numba.float64(numba.float64, numba.float64),
#     ]
# )
def delta_phi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


# @numba.vectorize(
#     [
#         numba.float32(numba.float32, numba.float32),
#         numba.float64(numba.float64, numba.float64),
#     ]
# )
def delta_eta(a, b):
    return a - b


def _get_lepton_mask(sample, lepton_type: str, dR_cut: float) -> np.ndarray:
    """
    Calculates a boolean mask to select the highest-pT lepton of a given type
    that is within a dR_cut of the ttFatJet.

    Returns a boolean array with at most one 'True' per row (event).
    """
    # Use .to_numpy() for efficient computation.
    # The jet's (N,) arrays are reshaped to (N, 1) to broadcast correctly
    # against the lepton's (N, M) arrays, where M is the number of leptons.

    lepton_eta = sample.get_var(f"{lepton_type}Eta")
    lepton_phi = sample.get_var(f"{lepton_type}Phi")
    jet_eta = sample.get_var("ttFatJetEta")[:, np.newaxis]
    jet_phi = sample.get_var("ttFatJetPhi")[:, np.newaxis]

    # print(sample.sample.label)
    # print(np.sum(lepton_eta != PAD_VAL, axis=0)/len(lepton_eta))

    # 1. Calculate dR for all leptons and create an initial mask
    dR = np.sqrt(delta_eta(lepton_eta, jet_eta) ** 2 + delta_phi(lepton_phi, jet_phi) ** 2)
    initial_mask = dR < dR_cut

    # 2. Find events that have at least one passing lepton
    events_with_any_pass = initial_mask.any(axis=1)

    # 3. Find the index of the *first* (highest-pT) passing lepton in each event
    # For events where none pass, np.argmax incorrectly returns 0.
    first_pass_idx = np.argmax(initial_mask, axis=1)

    # 4. Create the final mask, initialized to all False
    final_mask = np.zeros_like(initial_mask, dtype=bool)

    # 5. Use advanced indexing to set a single 'True' in the correct spot.
    # This clever line only sets final_mask[i, j] to True if events_with_any_pass[i]
    # is True, effectively ignoring the incorrect argmax result for non-passing events.
    row_indices = np.arange(len(final_mask))
    final_mask[row_indices, first_pass_idx] = events_with_any_pass

    return final_mask


def _get_ak4_mask(sample, fatjet_dR_cut: float = 0.9, lepton_dR_cut: float = 0.4) -> np.ndarray:
    # Parameters from https://github.com/LPC-HH/HH4b/blob/9d8038bcc31bf1872352e332eff84a4602934b3e/src/HH4b/processors/bbbbSkimmer.py#L170
    """
    Calculates a boolean mask to select the closest ak4 jet that is:
    1. Outside the ttFatJet cone (dR > fatjet_dR_cut)
    2. Not near any electrons (dR > lepton_dR_cut from all electrons)
    3. Not near any muons (dR > lepton_dR_cut from all muons)

    Among jets passing all criteria, selects the one closest to the ttFatJet.
    Returns a boolean array of shape (N_events, N_ak4jets) with at most one 'True' per event.
    """
    # Get ak4 jet coordinates
    ak4_eta = sample.get_var("ak4JetEta")  # Shape: (N_events, N_ak4jets)
    ak4_phi = sample.get_var("ak4JetPhi")  # Shape: (N_events, N_ak4jets)

    # Create mask for valid (non-padded) ak4 jets
    valid_ak4_mask = ak4_eta != PAD_VAL

    # Get fatjet coordinates
    fatjet_eta = sample.get_var("ttFatJetEta")[:, np.newaxis]  # Shape: (N_events, 1)
    fatjet_phi = sample.get_var("ttFatJetPhi")[:, np.newaxis]  # Shape: (N_events, 1)

    # 1. Calculate dR from fatjet and require ak4 jets to be OUTSIDE the cone
    dR_fatjet = np.sqrt(delta_eta(ak4_eta, fatjet_eta) ** 2 + delta_phi(ak4_phi, fatjet_phi) ** 2)
    outside_fatjet_mask = dR_fatjet > fatjet_dR_cut

    # 2. Check distance from electrons
    electron_eta = sample.get_var("ElectronEta")  # Shape: (N_events, N_electrons)
    electron_phi = sample.get_var("ElectronPhi")  # Shape: (N_events, N_electrons)
    valid_electron_mask = electron_eta != PAD_VAL  # Shape: (N_events, N_electrons)

    # Calculate dR between all ak4 jets and all electrons
    # Need to reshape for broadcasting: ak4 (N, M, 1) vs electrons (N, 1, L)
    ak4_eta_expanded = ak4_eta[:, :, np.newaxis]  # (N_events, N_ak4jets, 1)
    ak4_phi_expanded = ak4_phi[:, :, np.newaxis]  # (N_events, N_ak4jets, 1)
    electron_eta_expanded = electron_eta[:, np.newaxis, :]  # (N_events, 1, N_electrons)
    electron_phi_expanded = electron_phi[:, np.newaxis, :]  # (N_events, 1, N_electrons)

    dR_electrons = np.sqrt(
        delta_eta(ak4_eta_expanded, electron_eta_expanded) ** 2
        + delta_phi(ak4_phi_expanded, electron_phi_expanded) ** 2
    )  # Shape: (N_events, N_ak4jets, N_electrons)

    # Only consider valid electrons; set invalid electron distances to large value
    valid_electron_expanded = valid_electron_mask[:, np.newaxis, :]  # (N_events, 1, N_electrons)
    dR_electrons = np.where(valid_electron_expanded, dR_electrons, np.inf)

    # For each ak4 jet, check if it's far from ALL valid electrons (min distance > threshold)
    away_from_electrons_mask = np.all(dR_electrons > lepton_dR_cut, axis=2)  # (N_events, N_ak4jets)

    # 3. Check distance from muons
    muon_eta = sample.get_var("MuonEta")  # Shape: (N_events, N_muons)
    muon_phi = sample.get_var("MuonPhi")  # Shape: (N_events, N_muons)
    valid_muon_mask = muon_eta != PAD_VAL  # Shape: (N_events, N_muons)

    muon_eta_expanded = muon_eta[:, np.newaxis, :]  # (N_events, 1, N_muons)
    muon_phi_expanded = muon_phi[:, np.newaxis, :]  # (N_events, 1, N_muons)

    dR_muons = np.sqrt(
        delta_eta(ak4_eta_expanded, muon_eta_expanded) ** 2
        + delta_phi(ak4_phi_expanded, muon_phi_expanded) ** 2
    )  # Shape: (N_events, N_ak4jets, N_muons)

    # Only consider valid muons; set invalid muon distances to large value
    valid_muon_expanded = valid_muon_mask[:, np.newaxis, :]  # (N_events, 1, N_muons)
    dR_muons = np.where(valid_muon_expanded, dR_muons, np.inf)

    # For each ak4 jet, check if it's far from ALL valid muons (min distance > threshold)
    away_from_muons_mask = np.all(dR_muons > lepton_dR_cut, axis=2)  # (N_events, N_ak4jets)

    # 4. Combine all conditions, including validity of ak4 jets
    passing_mask = (
        valid_ak4_mask & outside_fatjet_mask & away_from_electrons_mask & away_from_muons_mask
    )

    # 5. Among passing jets, find the one closest to the fatjet (smallest dR)
    # For events where no jets pass, np.argmin incorrectly returns 0
    events_with_any_pass = passing_mask.any(axis=1)

    # Set dR to infinity for jets that don't pass, so they won't be selected as minimum
    dR_for_selection = np.where(passing_mask, dR_fatjet, np.inf)
    closest_jet_idx = np.argmin(dR_for_selection, axis=1)

    # 6. Create final mask with at most one True per event
    final_mask = np.zeros_like(passing_mask, dtype=bool)

    # Use advanced indexing to set only the closest jet to True, only for events that have passing jets
    row_indices = np.arange(len(final_mask))
    final_mask[row_indices, closest_jet_idx] = events_with_any_pass

    return final_mask


def leptons_assignment(
    events_dict: dict[str, LoadedSample],
    dR_cut: float = 1.5,
):
    """
    Assigns electrons and muons to the tt system.

    For each event, it identifies the highest-pT electron and muon within
    a cone of dR < 1.5 around the ttFatJet. The resulting boolean masks
    (e.g., sample.e_mask) will have at most one 'True' per event.
    """

    for sample in events_dict.values():
        sample.e_mask = _get_lepton_mask(sample, "Electron", dR_cut)
        sample.m_mask = _get_lepton_mask(sample, "Muon", dR_cut)


def derive_variables(
    events_dict: dict[str, LoadedSample], channel: Channel = None, num_fatjets: int = 3
):
    """Derive variables for each event."""

    if channel is not None:
        print(
            "Warning (postprocessing.derive_variables): indicating the channel is deprecated and needed only for data with tag < 25Sep23AddVars_v12_private_signal. Consider switching to new data in userConfig.py"
        )

    for sample in events_dict.values():
        if "ak8FatJetPNetXbbvsQCDLegacy" not in sample.events:
            Xbb = sample.get_var("ak8FatJetPNetXbbLegacy")
            QCD = sample.get_var("ak8FatJetPNetQCDLegacy")
            Xbb_vs_QCD = np.divide(Xbb, Xbb + QCD, out=np.zeros_like(Xbb), where=(Xbb + QCD) != 0)

            for n in range(num_fatjets):
                sample.events[("ak8FatJetPNetXbbvsQCDLegacy", str(n))] = Xbb_vs_QCD[:, n]

        if channel is not None:
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
                    sample.events[("ak8FatJetParTXtauhtaumvsQCDTop", str(n))] = tauhtaum_vs_QCDTop[
                        :, n
                    ]

            if channel.key == "hm" and "ak8FatJetParTXtauhtaumvsQCD" not in sample.events:
                tauhtaum_vs_QCD = np.divide(
                    tauhtaum,
                    tauhtaum + qcd,
                    out=np.zeros_like(tauhtaum),
                    where=(tauhtaum + qcd) != 0,
                )
                for n in range(num_fatjets):
                    sample.events[("ak8FatJetParTXtauhtaumvsQCD", str(n))] = tauhtaum_vs_QCD[:, n]


def derive_lepton_variables(events_dict: dict[str, LoadedSample]):
    for sample in events_dict.values():

        for n in range(sample.get_var("ElectronEta").shape[-1]):

            sample.events[("ElectronDeltaEta", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("ElectronPhi"))[:, n]
            )
            sample.events.loc[sample.e_mask[:, n], ("ElectronDeltaEta", str(n))] = delta_eta(
                sample.get_var("ElectronEta")[:, n], sample.get_var("ttFatJetEta")
            )[sample.e_mask[:, n]]

            sample.events[("ElectronDeltaPhi", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("ElectronPhi"))[:, n]
            )
            sample.events.loc[sample.e_mask[:, n], ("ElectronDeltaPhi", str(n))] = delta_phi(
                sample.get_var("ElectronPhi")[:, n], sample.get_var("ttFatJetPhi")
            )[sample.e_mask[:, n]]

        # need first to create the full branch before slicing
        for n in range(sample.get_var("ElectronEta").shape[-1]):

            sample.events[("Electron_dRak8Jet", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("ElectronPhi"))[:, n]
            )
            sample.events.loc[sample.e_mask[:, n], ("Electron_dRak8Jet", str(n))] = np.sqrt(
                sample.get_var("ElectronDeltaEta")[:, n] ** 2
                + sample.get_var("ElectronDeltaPhi")[:, n] ** 2
            )[sample.e_mask[:, n]]

        for n in range(sample.get_var("MuonEta").shape[-1]):

            sample.events[("MuonDeltaEta", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("MuonPhi"))[:, n]
            )
            sample.events.loc[sample.m_mask[:, n], ("MuonDeltaEta", str(n))] = delta_eta(
                sample.get_var("MuonEta")[:, n], sample.get_var("ttFatJetEta")
            )[sample.m_mask[:, n]]

            sample.events[("MuonDeltaPhi", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("MuonPhi"))[:, n]
            )
            sample.events.loc[sample.m_mask[:, n], ("MuonDeltaPhi", str(n))] = delta_phi(
                sample.get_var("MuonPhi")[:, n], sample.get_var("ttFatJetPhi")
            )[sample.m_mask[:, n]]

        for n in range(sample.get_var("MuonEta").shape[-1]):

            sample.events[("Muon_dRak8Jet", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("MuonPhi"))[:, n]
            )
            sample.events.loc[sample.m_mask[:, n], ("Muon_dRak8Jet", str(n))] = np.sqrt(
                sample.get_var("MuonDeltaEta")[:, n] ** 2
                + sample.get_var("MuonDeltaPhi")[:, n] ** 2
            )[sample.m_mask[:, n]]

    return


def load_data_channel(
    years: list[str],
    signals: list[str],
    channel: Channel,
    test_mode: bool,
    tt_pres: bool,
    models: list[str] | None = None,
    model_dir: Path = MODEL_DIR,
    bdt_eval_dir: Path = BDT_EVAL_DIR,
    at_inference: bool = False,
    cutflow: bool = False,
    **kwargs,
):
    """Load data for all years and signals for a given channel."""
    events_dict = {}
    for year in years:
        filters_dict = base_filter(test_mode)

        # Prefilters already applied in skimmer
        # filters_dict = bb_filters(filters_dict, num_fatjets=3, bb_cut=0.3)

        if tt_pres:
            filters_dict = tt_filters(
                channel=channel, in_filters=filters_dict, num_fatjets=3, tt_cut=0.3
            )

        columns = get_columns(year, triggers_in_channel=channel)

        events_dict[year] = load_samples(
            year=year,
            paths=DATA_PATHS[year],
            signals=signals,
            channels=[channel],
            filters_dict=filters_dict,
            load_columns=columns,
            restrict_data_to_channel=True,
            loaded_samples=True,
            multithread=True,
            **kwargs,
        )

        if cutflow:
            cutflow = utils.Cutflow(samples=events_dict[year])
            cutflow.add_cut(events_dict[year], "Preselection", "finalWeight")
            print(cutflow.cutflow)
            print("\nTriggers")

        apply_triggers(events_dict[year], year, channel)

        if cutflow:
            cutflow.add_cut(events_dict[year], "Triggers", "finalWeight")
            print(cutflow.cutflow)

        delete_columns(events_dict[year], year, channels=[channel])

        derive_variables(events_dict[year])
        bbtautau_assignment(events_dict[year], agnostic=True)
        leptons_assignment(events_dict[year], dR_cut=1.5)
        derive_lepton_variables(events_dict[year])

    # Load or compute BDT predictions for all signals in a single pass
    if models is not None:
        for model in models:
            for sig in signals:
                compute_or_load_bdt_preds(
                    events_dict=events_dict,
                    modelname=model,
                    model_dir=model_dir,
                    signal_objective=sig,
                    channel=channel,
                    bdt_preds_dir=bdt_eval_dir,
                    tt_pres=tt_pres,
                    test_mode=test_mode,
                    at_inference=at_inference,
                    all_outs=True,
                )

    if cutflow:
        return events_dict, cutflow
    else:
        return events_dict


def singleVarHist(
    events_dict: dict[str, pd.DataFrame | LoadedSample],
    shape_var: ShapeVar,
    channel: Channel,
    bbtt_masks: dict[str, pd.DataFrame] = None,
    weight_key: str = "finalWeight",
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """

    if not isinstance(next(iter(events_dict.values())), LoadedSample):
        warnings.warn(
            "Deprecation warning: Should switch to using the LoadedSample class in the future!",
            stacklevel=1,
        )
        return singleVarHistOld(events_dict, bbtt_masks, shape_var, channel, weight_key, selection)

    samples = list(events_dict.keys())

    h = Hist(
        hist.axis.StrCategory(samples + [data_key], name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for skey in samples:
        sample = events_dict[skey]

        if sample.sample.isData and shape_var.isVariation:
            fill_var = shape_var.var_no_variation()  # remove _up/_down
        else:
            fill_var = var

        fill_data = {var: sample.get_var(fill_var)}
        weight = sample.get_var(weight_key)

        if selection is not None:
            sel = selection[skey]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        if fill_data[var] is not None:
            h.fill(Sample=skey, **fill_data, weight=weight)

    data_hist = sum(h[skey, ...] for skey in channel.data_samples)
    h.view(flow=True)[utils.get_key_index(h, data_key)].value = data_hist.values(flow=True)
    h.view(flow=True)[utils.get_key_index(h, data_key)].variance = data_hist.variances(flow=True)

    if shape_var.blind_window is not None:
        utils.blindBins(h, shape_var.blind_window, data_key)

    return h


def label_transform(classes: list[str], labels: list[str]) -> list[int]:
    """Transform labels to integers."""
    return [classes.index(label) for label in labels]


#### Legacy functions


def derive_httak4away(
    events_dict: dict[str, LoadedSample],
    fatjet_dR_cut: float = 0.9,
    lepton_dR_cut: float = 0.4,
):
    # Keep this for possible future use, but is already implementedin skimmer

    for sample in events_dict.values():
        ak4_mask = _get_ak4_mask(sample, fatjet_dR_cut, lepton_dR_cut)

        ak4away = vector.array(
            {
                "pt": sample.get_var("ak4JetPt")[ak4_mask].squeeze(),
                "eta": sample.get_var("ak4JetEta")[ak4_mask].squeeze(),
                "phi": sample.get_var("ak4JetPhi")[ak4_mask].squeeze(),
                "mass": sample.get_var("ak4JetMass")[ak4_mask].squeeze(),
            }
        )

        htt = vector.array(
            {
                "pt": sample.get_var("ttFatJetPt")[ak4_mask.any(axis=1)].squeeze(),
                "eta": sample.get_var("ttFatJetEta")[ak4_mask.any(axis=1)].squeeze(),
                "phi": sample.get_var("ttFatJetPhi")[ak4_mask.any(axis=1)].squeeze(),
                "mass": sample.get_var("ttFatJetParTmassVisApplied")[
                    ak4_mask.any(axis=1)
                ].squeeze(),
            }
        )

        httak4away = htt + ak4away

        for n in range(sample.get_var("ak4JetEta").shape[-1]):

            sample.events[("httak4away_dR", str(n))] = (
                PAD_VAL * np.ones_like(sample.get_var("ak4JetPt"))[:, n]
            )
            sample.events.loc[ak4_mask[:, n], ("httak4away_dR", str(n))] = httak4away.deltaR()


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


# Leave to keep track but depends on deprecated standalone get_var that we removed from the framework


def get_var_old(events: pd.DataFrame, bbtt_mask: pd.DataFrame, feat: str):
    warnings.warn(
        "Deprecation warning: Should switch to using the LoadedSample class in the future!",
        stacklevel=1,
    )
    if feat in events:
        return events[feat].to_numpy().squeeze()
    elif feat.startswith(("bb", "tt")):
        jkey = feat[:2]
        return events[feat.replace(jkey, "ak8")].to_numpy()[bbtt_mask[jkey]]
    elif utils.is_int(feat[-1]):
        return events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()


def singleVarHistOld(
    events_dict: dict[str, pd.DataFrame | LoadedSample],
    bbtt_masks: dict[str, pd.DataFrame],
    shape_var: ShapeVar,
    channel: Channel,
    weight_key: str = "finalWeight",
    selection: dict | None = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Deprecated: use singleVarHist() with LoadedSample objects instead.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist(
        hist.axis.StrCategory(samples + [data_key], name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for sample in samples:
        events = events_dict[sample]
        if Samples.SAMPLES[sample].isData and var.endswith(("_up", "_down")):
            fill_var = "_".join(var.split("_")[:-2])  # remove _up/_down
        else:
            fill_var = var

        fill_data = {var: get_var_old(events, bbtt_masks[sample], fill_var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        if fill_data[var] is not None:
            h.fill(Sample=sample, **fill_data, weight=weight)

    data_hist = sum(h[skey, ...] for skey in channel.data_samples)
    h.view(flow=True)[utils.get_key_index(h, data_key)].value = data_hist.values(flow=True)
    h.view(flow=True)[utils.get_key_index(h, data_key)].variance = data_hist.variances(flow=True)

    if shape_var.blind_window is not None:
        utils.blindBins(h, shape_var.blind_window, data_key)

    return h
