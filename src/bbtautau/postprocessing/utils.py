"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import hist
import numpy as np
import pandas as pd
from boostedhh import utils
from boostedhh.hh_vars import data_key
from boostedhh.utils import PAD_VAL, Sample, ShapeVar
from hist import Hist

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing import Samples


def get_var(events: pd.DataFrame, bbtt_mask: pd.DataFrame, feat: str):
    if feat in events:
        return events[feat].to_numpy().squeeze()
    elif feat.startswith(("bb", "tt")):
        jkey = feat[:2]
        return events[feat.replace(jkey, "ak8")].to_numpy()[bbtt_mask[jkey]]
    elif utils.is_int(feat[-1]):
        return events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()


def rename_jetbranch_ak8(name: str) -> str:
    """
    Rename a branch/discriminator name to the ak8FatJet version.
    Used for plotting.
    """
    if name.startswith("bbFatJet"):
        return name.replace("bbFatJet", "ak8FatJet")
    elif name.startswith("ttFatJet"):
        return name.replace("ttFatJet", "ak8FatJet")
    else:
        return name


def concatenate_loaded_samples(loaded_samples: list[LoadedSample]) -> LoadedSample:
    """Concatenate multiple LoadedSample objects. Checks if the Samples are the same, or raises an error."""

    # check if the samples are the same
    if not all(ls.sample.label == loaded_samples[0].sample.label for ls in loaded_samples):
        raise ValueError("Samples are not the same")

    # check if the masks exist
    if not all(ls.bb_mask is not None for ls in loaded_samples):
        raise ValueError("bb_masks are not set for some samples")
    if not all(ls.tt_mask is not None for ls in loaded_samples):
        raise ValueError("tt_masks are not set for some samples")

    new_events = pd.concat([ls.events for ls in loaded_samples])
    new_bb_mask = np.concatenate([ls.bb_mask for ls in loaded_samples])
    new_tt_mask = np.concatenate([ls.tt_mask for ls in loaded_samples])

    return LoadedSample(
        sample=loaded_samples[0].sample,
        events=new_events,
        bb_mask=new_bb_mask,
        tt_mask=new_tt_mask,
    )


def concatenate_years(
    events_dict: dict[str, dict[str, LoadedSample]], years: list[str]
) -> dict[str, dict[str, LoadedSample]]:
    """Collapse typical events_dict[year][sample] = LoadedSample into events_dict[sample] = LoadedSample."""

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


@dataclass
class LoadedSample(utils.LoadedSampleABC):
    """Loaded sample."""

    sample: Sample
    events: pd.DataFrame = None
    bb_mask: np.ndarray = None
    tt_mask: np.ndarray = None
    m_mask: np.ndarray = None
    e_mask: np.ndarray = None

    def get_var(self, feat: str, pad_nan=False):
        if feat.startswith("ttMuon"):
            if self.m_mask is None:
                raise ValueError(f"m_mask is not set for {self.sample}")
            padded_array = np.full(len(self.events), PAD_VAL)
            padded_array[np.any(self.m_mask, axis=1)] = (
                self.events[feat.replace("ttMuon", "Muon")].to_numpy()[self.m_mask].squeeze()
            )
            return padded_array
        elif feat.startswith("ttElectron"):
            if self.e_mask is None:
                raise ValueError(f"e_mask is not set for {self.sample}")
            padded_array = np.full(len(self.events), PAD_VAL)
            padded_array[np.any(self.e_mask, axis=1)] = (
                self.events[feat.replace("ttElectron", "Electron")]
                .to_numpy()[self.e_mask]
                .squeeze()
            )
            return padded_array
        elif feat in self.events:
            return self.events[feat].to_numpy().squeeze()

        elif feat.startswith("bbFatJet"):
            if self.bb_mask is None:
                raise ValueError(f"bb_mask is not set for {self.sample}")

            if pad_nan:
                return np.nan_to_num(
                    self.events[feat.replace("bbFatJet", "ak8FatJet")]
                    .to_numpy()[self.bb_mask]
                    .squeeze(),
                    nan=PAD_VAL,
                )
            else:
                return (
                    self.events[feat.replace("bbFatJet", "ak8FatJet")]
                    .to_numpy()[self.bb_mask]
                    .squeeze()
                )
        elif feat.startswith("ttFatJet"):
            if self.tt_mask is None:
                raise ValueError(f"tt_mask is not set for {self.sample}")

            if pad_nan:
                return np.nan_to_num(
                    self.events[feat.replace("ttFatJet", "ak8FatJet")]
                    .to_numpy()[self.tt_mask]
                    .squeeze(),
                    nan=PAD_VAL,
                )
            return (
                self.events[feat.replace("ttFatJet", "ak8FatJet")]
                .to_numpy()[self.tt_mask]
                .squeeze()
            )

        # Not sure if should pad also this case.
        elif utils.is_int(feat[-1]):
            return self.events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()

    def copy_from_selection(
        self, selection: np.ndarray[bool], do_deepcopy: bool = False
    ) -> LoadedSample:
        """Copy of LoadedSample after applying a selection."""
        return LoadedSample(
            sample=self.sample,
            events=deepcopy(self.events[selection]) if do_deepcopy else self.events[selection],
            bb_mask=self.bb_mask[selection] if self.bb_mask is not None else None,
            tt_mask=self.tt_mask[selection] if self.tt_mask is not None else None,
        )

    def get_mask(self, jet: str) -> np.ndarray:
        if jet == "bb":
            return self.bb_mask
        elif jet == "tt":
            return self.tt_mask
        else:
            raise ValueError(f"Invalid jet: {jet}")


@dataclass
class Region:
    cuts: dict = None
    label: str = None
    signal: bool = False  # is this a signal region?
    cutstr: str = None  # optional label for the region's cuts e.g. when scanning cuts


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

        fill_data = {var: get_var(events, bbtt_masks[sample], fill_var)}
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
