"""
Shared types for postprocessing.

Author: Ludovico Mori
"""

from __future__ import annotations

import warnings
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


class FOM:
    def __init__(self, fom_func, label, name):
        self.fom_func = fom_func
        self.label = label
        self.name = name


@dataclass
class ABCD:
    pass_sideband: float
    fail_resonant: float
    fail_sideband: float
    pass_resonant: float | None = None  # should never be initialized on data!
    isData: bool = False  # decide whether to determine the pass resonant using abcd method

    pass_resonant_offset: float | None = (
        None  # if isData, can use to add MC to data-driven estimate of pass resonant
    )

    def __post_init__(self):
        # compute the transfer factor
        self.tf = self.fail_resonant / self.fail_sideband if self.fail_sideband > 0 else 0

        if self.isData:
            if self.pass_resonant is not None:
                warnings.warn(
                    "Pass resonant should not be initialized from data (would be unblinding)! Setting to data driven value",
                    stacklevel=2,
                )

            self.pass_resonant = self.fail_resonant / self.tf if self.tf > 0 else 0

    def subtract_MC(self, other: ABCD):
        """
        Subtract the MC from the data-driven estimate of the pass resonant.
        """
        if not self.isData or other.isData:
            warnings.warn("Must subtract MC from data. Returning initial object.", stacklevel=2)
            return self

        return ABCD(
            pass_sideband=self.pass_sideband - other.pass_sideband,
            fail_resonant=self.fail_resonant - other.fail_resonant,
            fail_sideband=self.fail_sideband - other.fail_sideband,
            isData=True,
            pass_resonant_offset=other.pass_resonant,
        )


@dataclass
class SRConfig:
    """Configuration for a signal region optimization.
    This supports defining orthogonal signal regions (e.g., VBF-enriched and GGF-enriched).
    When optimizing one signal, you can exclude events selected by other signals to avoid overlap.
    Result stored in optima dictionary. Vetos reference other SRConfig objects, vetoing the cuts for the given b_min_for_exclusion.
    """

    name: str
    signals: list[str]
    channel: str
    bb_disc_name: str
    tt_disc_name: str
    optima: dict[str, dict] = None
    veto_cuts: dict[str, tuple[float, float, str, str]] = (
        None  # veto_name -> (bb_cut, tt_cut, bb_disc, tt_disc)
    )
    bmin_for_exclusion: float = 10.0

    def add_veto_from_optimum(self, veto_sr_config: SRConfig, bmin: float = None):
        """Extract veto cuts from another SRConfig's optima and add to this region's vetoes.

        Args:
            veto_sr_config: The SRConfig of the veto region (must have optima populated)
            bmin: B_min value to extract cuts from (uses veto's bmin_for_exclusion if None)
        """
        if veto_sr_config.optima is None:
            raise ValueError(
                f"Veto region '{veto_sr_config.name}' has no optima to extract cuts from"
            )

        bmin = bmin or veto_sr_config.bmin_for_exclusion
        bmin_key = f"Bmin={bmin}"

        # Extract cuts from the default FOM
        if (
            "2sqrtB_S_var" not in veto_sr_config.optima
            or bmin_key not in veto_sr_config.optima["2sqrtB_S_var"]
        ):
            raise ValueError(f"Veto region '{veto_sr_config.name}' missing optima for {bmin_key}")

        optimum = veto_sr_config.optima["2sqrtB_S_var"][bmin_key]
        bb_cut, tt_cut = optimum["cuts"]

        # Initialize veto structures if needed
        if self.veto_cuts is None:
            self.veto_cuts = {}

        self.veto_cuts[veto_sr_config.name] = (
            bb_cut,
            tt_cut,
            veto_sr_config.bb_disc_name,
            veto_sr_config.tt_disc_name[self.channel],
        )
