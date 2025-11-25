"""
Shared types for postprocessing.

Author: Ludovico Mori
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from boostedhh import utils
from boostedhh.utils import PAD_VAL, Sample

from bbtautau.HLTs import HLTs


@dataclass
class Channel:
    """Channel."""

    key: str  # key in dictionaries etc.
    label: str  # label for plotting
    data_samples: list[str]  # datasets for this channel
    hlt_types: list[str]  # list of HLT types
    isLepton: bool  # lepton channel or fully hadronic
    tagger_label: str  # label for tagger score used
    txbb_cut: float  # cut on bb tagger score
    txtt_cut: float  # cut on tt tagger score
    txtt_BDT_cut: float  # cut on tt BDT score
    tt_mass_cut: tuple[str, list[float]]  # cut on tt mass
    lepton_dataset: str = None  # lepton dataset (if applicable)

    def triggers(
        self,
        year: str,
        **hlt_kwargs,
    ):
        """Get triggers for a given year for this channel."""
        return HLTs.hlts_by_type(year, self.hlt_types, **hlt_kwargs)

    def lepton_triggers(
        self,
        year: str,
        **hlt_kwargs,
    ):
        """Get lepton triggers for a given year for this channel."""
        if self.lepton_dataset is None:
            return None

        return HLTs.hlts_by_dataset(year, self.lepton_dataset, **hlt_kwargs)


@dataclass
class LoadedSample(utils.LoadedSampleABC):
    """Loaded sample with events and jet masks."""

    sample: Sample
    events: pd.DataFrame = None
    bb_mask: np.ndarray = None
    tt_mask: np.ndarray = None
    m_mask: np.ndarray = None
    e_mask: np.ndarray = None

    def get_var(self, feat: str, pad_nan=False):
        """Get a variable from the events DataFrame, applying appropriate masks for jet-specific features."""
        if feat in self.events:
            if pad_nan:
                return np.nan_to_num(self.events[feat].to_numpy().squeeze(), nan=PAD_VAL)
            else:
                return self.events[feat].to_numpy().squeeze()
        elif feat.startswith("ttMuon"):
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

        elif feat.startswith("bbFatJet"):
            if self.bb_mask is None:
                raise ValueError(f"bb_mask is not set for {self.sample}")

            ak8_feat = self.rename_jetbranch_ak8(feat)
            if pad_nan:
                return np.nan_to_num(
                    self.events[ak8_feat].to_numpy()[self.bb_mask].squeeze(),
                    nan=PAD_VAL,
                )
            else:
                return self.events[ak8_feat].to_numpy()[self.bb_mask].squeeze()
        elif feat.startswith("ttFatJet"):
            if self.tt_mask is None:
                raise ValueError(f"tt_mask is not set for {self.sample}")

            ak8_feat = self.rename_jetbranch_ak8(feat)
            if pad_nan:
                return np.nan_to_num(
                    self.events[ak8_feat].to_numpy()[self.tt_mask].squeeze(),
                    nan=PAD_VAL,
                )
            return self.events[ak8_feat].to_numpy()[self.tt_mask].squeeze()

        elif feat.startswith("bbJetAway"):
            return self.events[feat.replace("bbJetAway", "AK4JetAway")].to_numpy()[
                :, 0
            ]  # first col is bb away, second is tt away

        elif feat.startswith("ttJetAway"):
            return self.events[feat.replace("ttJetAway", "AK4JetAway")].to_numpy()[
                :, 1
            ]  # first col is bb away, second is tt away

        # Not sure if should pad also this case.
        elif utils.is_int(feat[-1]):
            return self.events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()

        else:
            raise ValueError(f"Feature {feat} not found in events")

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
        """Get the mask for a specific jet type."""
        if jet == "bb":
            return self.bb_mask
        elif jet == "tt":
            return self.tt_mask
        else:
            raise ValueError(f"Invalid jet: {jet}")

    @staticmethod
    def rename_jetbranch_ak8(name: str) -> str:
        """
        Rename a branch/discriminator name to the ak8FatJet version.
        Used for plotting and internal variable lookups.
        """
        if name.startswith("bbFatJet"):
            return name.replace("bbFatJet", "ak8FatJet")
        elif name.startswith("ttFatJet"):
            return name.replace("ttFatJet", "ak8FatJet")
        else:
            return name


@dataclass
class Region:
    """Analysis region definition."""

    cuts: dict = None
    label: str = None
    signal: bool = False  # is this a signal region?
    cutstr: str = None  # optional label for the region's cuts e.g. when scanning cuts
