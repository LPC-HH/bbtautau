from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import hh_vars
from boostedhh.utils import PAD_VAL, Sample
from joblib import Parallel, delayed

from bbtautau.postprocessing import utils
from bbtautau.postprocessing.plotting import (
    plot_optimization_sig_eff,
    plot_optimization_thresholds,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.Samples import CHANNELS, SM_SIGNALS
from bbtautau.postprocessing.utils import load_data_channel
from bbtautau.userConfig import (
    CHANNEL_ORDERING,
    PLOT_DIR,
    PT_CUTS,
    SHAPE_VAR,
    SIGNAL_ORDERING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# Global variables
TODAY = date.today()  # to name output folder

# Default B_min values for optimization
DEFAULT_BMIN_VALUES = [1, 5, 10, 12]  # Used for non-adaptive mode and final adaptive stage
DEFAULT_ADAPTIVE_BMIN = 10  # Single B_min used during adaptive refinement iterations

# Adaptive optimization defaults (to avoid too many CLI args)
ADAPTIVE_STAGES = 2
ADAPTIVE_TOP_FRAC = 0.1
ADAPTIVE_PADDING = 1.0
ADAPTIVE_REFINE_GRIDSIZE = None
ADAPTIVE_BMIN = DEFAULT_ADAPTIVE_BMIN

adaptive_kwargs = {
    "adaptive_stages": ADAPTIVE_STAGES,
    "adaptive_top_frac": ADAPTIVE_TOP_FRAC,
    "adaptive_padding": ADAPTIVE_PADDING,
    "adaptive_refine_gridsize": ADAPTIVE_REFINE_GRIDSIZE,
    "adaptive_bmin": ADAPTIVE_BMIN,
}

# Define discriminant names
BB_DISC_NAME = "bbFatJetParTXbbvsQCDTop"
TT_DISC_NAMES_PART = {  # ParT does not distinguish between signals
    sig_key: {
        channel_key: f"ttFatJetParTX{CHANNELS[channel_key].tagger_label}vsQCDTop"
        for channel_key in CHANNELS
    }
    for sig_key in SM_SIGNALS
}
TT_DISC_NAMES_BDT = {
    sig_key: {
        channel_key: f"BDT{sig_key.removesuffix('tt')}{CHANNELS[channel_key].tagger_label}vsAll"
        for channel_key in CHANNELS
    }
    for sig_key in SM_SIGNALS
}


class FOM:
    def __init__(self, fom_func, label, name):
        self.fom_func = fom_func
        self.label = label
        self.name = name


def fom_2sqrtB_S(b, s, _tf):
    return np.where(s > 0, 2 * np.sqrt(b * _tf) / s, np.nan)


def fom_2sqrtB_S_var(b, s, _tf):
    return np.where((b > 0) & (s > 0), 2 * np.sqrt(b * _tf * (1 + _tf)) / s, np.nan)


def fom_punzi(b, s, _tf, a=3):
    """
    a is the number of sigmas of the test significance
    """
    return np.where(s > 0, (np.sqrt(b * _tf) + a / 2) / s, np.nan)


FOMS = {
    "2sqrtB_S": FOM(fom_2sqrtB_S, "$2\\sqrt{B}/S$", "2sqrtB_S"),
    "2sqrtB_S_var": FOM(fom_2sqrtB_S_var, "$2\\sqrt{B+B^2/\\tilde{B}}/S$", "2sqrtB_S_var"),
    "punzi": FOM(fom_punzi, "$(\\sqrt{B}+a/2)/S$", "punzi"),
}

FOMS_TO_OPTIMIZE = [FOMS["2sqrtB_S_var"]]


@dataclass
class CutOptimum:
    limit: float
    signal_yield: float
    signal_eff: float
    bkg_yield: float
    hmass_fail: float
    sideband_fail: float
    transfer_factor: float
    cuts: tuple[float, float]

    # Optional fields for plotting
    BBcut: np.ndarray | None = None
    TTcut: np.ndarray | None = None
    sig_map: np.ndarray | None = None
    bg_map: np.ndarray | None = None
    sel_B_min: np.ndarray | None = None
    fom: FOM | None = None

    # Signal efficiency specific fields (for grid_search_opt_sig_eff)
    sig_eff_cuts: tuple[float, float] | None = None  # Signal efficiency values at optimum
    BBcut_sig_eff: np.ndarray | None = None  # Signal efficiency grid for bb tagger
    TTcut_sig_eff: np.ndarray | None = None  # Signal efficiency grid for tt tagger
    fom_map: np.ndarray | None = None  # FOM values grid for plotting
    bb_disc_name: str | None = None  # Name of bb discriminant used
    tt_disc_name: str | None = None  # Name of tt discriminant used


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
    optima: dict[str, CutOptimum] = None
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
        bb_cut, tt_cut = optimum.cuts

        # Initialize veto structures if needed
        if self.veto_cuts is None:
            self.veto_cuts = {}

        self.veto_cuts[veto_sr_config.name] = (
            bb_cut,
            tt_cut,
            veto_sr_config.bb_disc_name,
            veto_sr_config.tt_disc_name[self.channel],
        )


class Analyser:
    def __init__(
        self,
        events_dict,
        sr_config: SRConfig,
        test_mode: bool | None = None,
        use_ParT: bool | None = None,
        at_inference=False,
        tt_pres=False,
        adaptive=False,
        bdt_dir=None,
        plot_dir: Path | None = None,
    ):
        """Initialize Analyser with support for multiple signals."""
        self.sr_config = sr_config
        self.events_dict = events_dict
        self.years = list(events_dict.keys())
        self.test_mode = test_mode
        self.tt_pres = tt_pres
        self.at_inference = at_inference
        self.use_ParT = use_ParT
        self.adaptive = adaptive
        self.bdt_dir = bdt_dir

        # Unpack some handy variables
        self.signals = sr_config.signals
        self.channel = CHANNELS[sr_config.channel]
        self.region_name = sr_config.name

        self.sig_keys_channel = [sig + sr_config.channel for sig in sr_config.signals]
        self.all_keys = self.sig_keys_channel + self.channel.data_samples

        print("All keys: ", self.all_keys)

        self.bb_disc_name = sr_config.bb_disc_name
        self.tt_disc_name = sr_config.tt_disc_name
        self.tt_disc_name_channel = self.tt_disc_name[self.channel.key]

        self.plot_dir = plot_dir

    # Can remove in next PR, leave just for record. was only used in plot_mass, but that was updated

    # @staticmethod
    # def get_jet_vals(vals, mask, nan_to_pad=True):

    #     # TODO: Deprecate this (just need to use get_var properly)

    #     # check if vals is a numpy array
    #     if not isinstance(vals, np.ndarray):
    #         vals = vals.to_numpy()
    #     if len(vals.shape) == 1:
    #         warnings.warn(
    #             f"vals is a numpy array of shape {vals.shape}. Ignoring mask.", stacklevel=2
    #         )
    #         return vals if not nan_to_pad else np.nan_to_num(vals, nan=PAD_VAL)
    #     if nan_to_pad:
    #         return np.nan_to_num(vals[mask], nan=PAD_VAL)
    #     else:
    #         return vals[mask]

    def compute_and_plot_rocs(self, discs=None):
        years = list(self.events_dict.keys())
        events_dict_allyears = utils.concatenate_years(self.events_dict)

        background_names = self.channel.data_samples

        # Get channel-specific discriminant names
        channel_label = self.channel.tagger_label
        discs_tt = [
            f"ttFatJetParTX{channel_label}vsQCD",
            f"ttFatJetParTX{channel_label}vsQCDTop",
        ]

        discs_bb = ["bbFatJetParTXbbvsQCD", "bbFatJetParTXbbvsQCDTop", "bbFatJetPNetXbbvsQCDLegacy"]

        # Add BDT discriminants if using BDT
        if not self.use_ParT:
            # For multiple signals, add BDT discriminants for each
            for sig in self.signals:
                signal_base = sig.removesuffix("tt") if sig.endswith("tt") else sig
                discs_tt += [
                    # f"BDT{signal_base}{channel_label}vsQCD", #can add if needed
                    f"BDT{signal_base}{channel_label}vsAll",
                ]

        discs_all = discs or (discs_bb + discs_tt)

        print(f"Computing ROCs for signals: {self.sig_keys_channel}")

        # Create a combined signal if multiple signals are present. ROCAnalyzer requires a single signal.
        # This allows ROC computation for "SM signal" = union of ggF and VBF.
        if len(self.sig_keys_channel) > 1:
            combined_signal_name = f"SM_{self.sr_config.channel}"
            print(
                f"Combining {len(self.sig_keys_channel)} signals into '{combined_signal_name}' for ROC analysis"
            )

            # Collect SM signals and concatenate them
            SM_sample = Sample(label=f"SM bbtt{self.sr_config.channel}", isSignal=True)
            combined_sample = utils.concatenate_loaded_samples(
                [events_dict_allyears[key] for key in self.sig_keys_channel], out_sample=SM_sample
            )

            signals_for_roc = {combined_signal_name: combined_sample}
            signal_name_for_fill = combined_signal_name
        else:
            # Single signal: use as-is
            signals_for_roc = {
                sig_key: events_dict_allyears[sig_key] for sig_key in self.sig_keys_channel
            }
            signal_name_for_fill = self.sig_keys_channel[0]

        self.rocAnalyzer = ROCAnalyzer(
            years=years,
            signals=signals_for_roc,
            backgrounds={bkg: events_dict_allyears[bkg] for bkg in background_names},
        )

        # Fill discriminants for the signal (combined or single)
        self.rocAnalyzer.fill_discriminants(
            discs_all, signal_name=signal_name_for_fill, background_names=background_names
        )

        self.rocAnalyzer.compute_rocs()

        # Plot bb discriminants
        self.rocAnalyzer.plot_rocs(title="bbFatJet", disc_names=discs_bb, plot_dir=self.plot_dir)

        # Plot tt discriminants
        self.rocAnalyzer.plot_rocs(title="ttFatJet", disc_names=discs_tt, plot_dir=self.plot_dir)

        # Plot discriminant scores and confusion matrices
        for disc in discs_all:
            self.rocAnalyzer.plot_disc_scores(disc, background_names, self.plot_dir)
            self.rocAnalyzer.plot_disc_scores(
                disc, [[bkg] for bkg in background_names], self.plot_dir
            )
            self.rocAnalyzer.compute_confusion_matrix(disc, plot_dir=self.plot_dir)

        print(f"ROCs computed and plotted for {signal_name_for_fill} vs backgrounds.")

    def plot_mass(self):
        """Plot mass distributions for signal and data samples."""
        # Determine which samples to plot
        plot_configs = [
            ("signal", self.sig_keys_channel, f"Signal: {'+'.join(self.signals)}"),
            ("data", self.channel.data_samples, "Data"),
        ]
        events_dict_allyears = utils.concatenate_years(self.events_dict)

        for plot_key, sample_keys, label in plot_configs:
            print(f"Plotting mass for {label}")

            sample_info = Sample(label=label, isSignal=plot_key == "signal")
            combined_sample = utils.concatenate_loaded_samples(
                [events_dict_allyears[key] for key in sample_keys], out_sample=sample_info
            )

            weights = combined_sample.get_var("finalWeight")

            bins = np.linspace(0, 250, 50)
            fig, axs = plt.subplots(1, 2, figsize=(24, 10))

            for i, (jet, jlabel) in enumerate(
                zip(["bb", "tt"], ["bb FatJet", rf"{self.channel.label} FatJet"])
            ):
                ax = axs[i]
                jet_prefix = f"{jet}FatJet"

                print(f"Plotting mass for {jet} jet")
                print(f"Number of events: {len(combined_sample.events)}")

                for j, (var_suffix, mlabel) in enumerate(
                    zip(
                        ["Msd", "PNetmassLegacy", "ParTmassResApplied", "ParTmassVisApplied"],
                        ["SoftDrop", "PNetLegacy", "ParT Res", "ParT Vis"],
                    )
                ):
                    # Use LoadedSample.get_var with jet prefix - it automatically applies masks
                    var_name = f"{jet_prefix}{var_suffix}"
                    jet_masses = combined_sample.get_var(var_name, pad_nan=True)

                    # Filter out PAD_VAL
                    valid = jet_masses != PAD_VAL

                    ax.hist(
                        jet_masses[valid],
                        bins=bins,
                        histtype="step",
                        weights=weights[valid],
                        label=mlabel,
                        linewidth=2,
                        color=plt.cm.tab10.colors[j],
                    )

                ax.vlines(125, 0, ax.get_ylim()[1], linestyle="--", color="k", alpha=0.1)
                ax.set_xlabel("Mass [GeV]")
                ax.set_ylabel("Events")
                ax.legend()
                ax.set_ylim(0)

                # CMS label
                hep.cms.label(
                    ax=ax,
                    label="Preliminary",
                    data=(plot_key == "data"),
                    year="2022-23" if self.years == hh_vars.years else "+".join(self.years),
                    com="13.6",
                    fontsize=20,
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                )

                ax.text(
                    0.03,
                    0.92,
                    jlabel,
                    transform=ax.transAxes,
                    fontsize=24,
                )

            # Create mass directory if it doesn't exist
            (self.plot_dir / "mass").mkdir(parents=True, exist_ok=True)

            # Save plots
            years_str = "_".join(self.years)
            plt.savefig(
                self.plot_dir / f"mass/{plot_key}_{years_str}.png",
                bbox_inches="tight",
            )
            plt.savefig(
                self.plot_dir / f"mass/{plot_key}_{years_str}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    def _concat_years(self, var_name: str, key: str) -> np.ndarray:
        """Concatenate a variable across all years for a given sample key."""
        return np.concatenate(
            [self.events_dict[year][key].get_var(var_name) for year in self.years], axis=0
        )

    def _extract_var_with_cuts(
        self, var_name: str, pt_cuts: dict, concatenate_samples: dict[str, list[str]] = None
    ) -> dict[str, np.ndarray]:
        """Extract a variable for all samples with pt cuts applied.
        If concatenate_samples is provided, concatenate the samples in the given groups.
        e.g. if concatenate_samples = {"signal": ["ggfbbtthh", "vbfbbtthh"], "background": ["qcd", "ttbar"]}, then the output will be a dictionary with keys "signal" and "background", and values will be the concatenation of the samples in the given groups.
        """

        if concatenate_samples is None:
            return {key: self._concat_years(var_name, key)[pt_cuts[key]] for key in self.all_keys}
        else:
            samples = {
                key: self._concat_years(var_name, key)[pt_cuts[key]] for key in self.all_keys
            }
            return {
                group_name: np.concatenate(
                    [samples[key] for key in concatenate_samples[group_name]], axis=0
                )
                for group_name in concatenate_samples
            }

    def prepare_sensitivity(self):
        """Prepare discriminants and kinematic variables for optimization."""

        mtt1, mtt2 = self.channel.tt_mass_cut[1]

        # just a safety check
        if not set(self.all_keys) <= set(self.events_dict[self.years[0]].keys()):
            raise ValueError(
                f"All keys {self.all_keys} do not match the keys in the events dictionary {list(self.events_dict[self.years[0]].keys())}"
            )

        # Compute pt cuts and safety bbmass cuts once for all samples
        pt_veto_cuts = {
            key: (self._concat_years("bbFatJetPt", key) > PT_CUTS["bb"])
            & (self._concat_years("ttFatJetPt", key) > PT_CUTS["tt"])
            & (self._concat_years(SHAPE_VAR["name"], key) > SHAPE_VAR["range"][0])
            & (self._concat_years(SHAPE_VAR["name"], key) < SHAPE_VAR["range"][1])
            for key in self.all_keys
        }

        if self.use_ParT:
            # apply the tt mass cuts in the preprocessing stage if we do not use the BDT
            for sig_key in self.sig_keys_channel:
                pt_veto_cuts[sig_key] &= (
                    self._concat_years(self.channel.tt_mass_cut[0], sig_key) > mtt1
                ) & (self._concat_years(self.channel.tt_mass_cut[0], sig_key) < mtt2)
            for key in self.channel.data_samples:
                pt_veto_cuts[key] &= (
                    self._concat_years(self.channel.tt_mass_cut[0], key) < mtt1
                ) | (self._concat_years(self.channel.tt_mass_cut[0], key) > mtt2)

        # Apply veto cuts: exclude events passing veto region selections
        if self.sr_config.veto_cuts:
            for veto_key in self.sr_config.veto_cuts:
                veto_bb_cut, veto_tt_cut, veto_bb_disc, veto_tt_disc = self.sr_config.veto_cuts[
                    veto_key
                ]
                # Create veto mask
                print(veto_key, veto_bb_cut, veto_tt_cut, veto_bb_disc, veto_tt_disc)
                veto_mask = {
                    key: ~(
                        (self._concat_years(veto_bb_disc, key) > veto_bb_cut)
                        & (self._concat_years(veto_tt_disc, key) > veto_tt_cut)
                    )
                    for key in self.all_keys
                }
                # Combine veto mask with pt_cuts
                pt_veto_cuts = {key: pt_veto_cuts[key] & veto_mask[key] for key in self.all_keys}

        # Extract main discriminants and kinematics
        sig_vs_bkg_groups = {
            "signal": self.sig_keys_channel,
            "background": self.channel.data_samples,
        }
        self.sensitivity_vars = {
            "txbbs": self._extract_var_with_cuts(
                self.bb_disc_name, pt_veto_cuts, concatenate_samples=sig_vs_bkg_groups
            ),
            "txtts": self._extract_var_with_cuts(
                self.tt_disc_name_channel, pt_veto_cuts, concatenate_samples=sig_vs_bkg_groups
            ),
            "massbb": self._extract_var_with_cuts(
                SHAPE_VAR["name"], pt_veto_cuts, concatenate_samples=sig_vs_bkg_groups
            ),
            "masstt": self._extract_var_with_cuts(
                self.channel.tt_mass_cut[0], pt_veto_cuts, concatenate_samples=sig_vs_bkg_groups
            ),
            "finalWeight": self._extract_var_with_cuts(
                "finalWeight", pt_veto_cuts, concatenate_samples=sig_vs_bkg_groups
            ),
        }

    def compute_sig_bkg_abcd(self, txbbcut, txttcut, mbb1, mbb2):
        """Compute signal and background in ABCD regions.

        Args:
            years: Years to process
            txbbcut: TxBB discriminant cut
            txttcut: TxTT discriminant cut
            mbb1, mbb2: bb mass window boundaries

            TODO: This function can be optimized by passing the sideband cuts as args
        """

        # build the selections for the signals in the current channel
        cut_sigs_pass = (
            (self.sensitivity_vars["txbbs"]["signal"] > txbbcut)
            & (self.sensitivity_vars["txtts"]["signal"] > txttcut)
            & (self.sensitivity_vars["massbb"]["signal"] > mbb1)
            & (self.sensitivity_vars["massbb"]["signal"] < mbb2)
        )

        # Background in mass sidebands
        cut_sb = (self.sensitivity_vars["massbb"]["background"] < mbb1) | (
            self.sensitivity_vars["massbb"]["background"] > mbb2
        )
        # Background passing discriminant cuts
        cut_pass = (self.sensitivity_vars["txbbs"]["background"] > txbbcut) & (
            self.sensitivity_vars["txtts"]["background"] > txttcut
        )

        # Compute signal yield in the resonant region
        sig_pass = np.sum(self.sensitivity_vars["finalWeight"]["signal"][cut_sigs_pass])

        # Background passing discriminant cuts in mass sidebands
        bg_pass_sb = np.sum(self.sensitivity_vars["finalWeight"]["background"][cut_pass & cut_sb])
        # Background failing discriminant cuts in mass sidebands
        bg_fail_sb = np.sum(
            self.sensitivity_vars["finalWeight"]["background"][(~cut_pass) & cut_sb]
        )
        # Background failing discriminant cuts in resonant region
        bg_fail_res = np.sum(
            self.sensitivity_vars["finalWeight"]["background"][(~cut_pass) & (~cut_sb)]
        )
        # signal, B, C, D, TF = C/D
        tf = bg_fail_res / bg_fail_sb if bg_fail_sb > 0 else 0

        return sig_pass, bg_pass_sb, bg_fail_res, bg_fail_sb, tf

    def grid_search_opt(
        self, gridsize, gridlims, B_min_vals, foms, use_thresholds=False
    ) -> CutOptimum:
        """
        Grid search optimization for signal/background discrimination.

        Args:
            gridsize: Size of the grid (gridsize x gridsize points)
            gridlims: Either a single tuple (min, max) used for both axes,
                     or a dict with keys 'x' and 'y': {'x': (min, max), 'y': (min, max)}
            B_min_vals: List of minimum background values to test
            foms: List of figure-of-merit functions
            normalize_sig: Whether to normalize signal yields
            use_sig_eff: If True, use signal efficiency coordinates; if False, use raw thresholds

        Returns:
            Optimum: Results dictionary containing optimization results

        Note:
            When use_sig_eff=True, gridlims should be efficiency values (0.0 to 1.0).
            When use_sig_eff=False, gridlims should be raw threshold values.
        """

        mbb1, mbb2 = SHAPE_VAR["blind_window"]

        # Parse gridlims: support both single tuple and dict format
        if isinstance(gridlims, dict):
            gridlims_x = gridlims["x"]
            gridlims_y = gridlims["y"]
        else:
            # Backward compatibility: single tuple used for both axes
            gridlims_x = gridlims
            gridlims_y = gridlims

        if not use_thresholds:
            # Signal efficiency coordinate system
            if not hasattr(self, "rocAnalyzer"):
                print("ROC analyzer not found. Running compute_and_plot_rocs first.")
                self.compute_and_plot_rocs()
                print("ROC analyzer computed and plotted.")

            # Check discriminants exist
            if self.bb_disc_name not in self.rocAnalyzer.discriminants:
                raise ValueError(
                    f"BB discriminant '{self.bb_disc_name}' not found in ROC analyzer. Available: {list(self.rocAnalyzer.discriminants.keys())}"
                )
            if self.tt_disc_name_channel not in self.rocAnalyzer.discriminants:
                raise ValueError(
                    f"TT discriminant '{self.tt_disc_name_channel}' not found in ROC analyzer. Available: {list(self.rocAnalyzer.discriminants.keys())}"
                )

            # Create signal efficiency grid with separate limits for each axis
            bbcutSigEff = np.linspace(*gridlims_x, gridsize)
            ttcutSigEff = np.linspace(*gridlims_y, gridsize)
            BBcutSigEff, TTcutSigEff = np.meshgrid(bbcutSigEff, ttcutSigEff)

            # Get discriminant objects for threshold lookup
            bb_discriminant = self.rocAnalyzer.discriminants[self.bb_disc_name]
            tt_discriminant = self.rocAnalyzer.discriminants[self.tt_disc_name_channel]

            # VECTORIZED threshold computation
            bbcut_thresholds = bb_discriminant.get_cut_from_sig_eff(bbcutSigEff)
            ttcut_thresholds = tt_discriminant.get_cut_from_sig_eff(ttcutSigEff)

            # Create threshold meshgrids for actual cuts
            BBcut, TTcut = np.meshgrid(bbcut_thresholds, ttcut_thresholds)

        else:
            # Raw threshold coordinate system with separate limits for each axis
            bbcut = np.linspace(*gridlims_x, gridsize)
            ttcut = np.linspace(*gridlims_y, gridsize)
            BBcut, TTcut = np.meshgrid(bbcut, ttcut)

            # For consistency, set efficiency grids to None when not using signal efficiency
            BBcutSigEff = None
            TTcutSigEff = None

        # Flatten the grid for parallel evaluation
        bbcut_flat = BBcut.ravel()
        ttcut_flat = TTcut.ravel()

        sig_bkg_f = self.compute_sig_bkg_abcd

        def sig_bg(bbcut, ttcut):
            return sig_bkg_f(
                txbbcut=bbcut,
                txttcut=ttcut,
                mbb1=mbb1,
                mbb2=mbb2,
            )

        print(f"Running grid search on {len(bbcut_flat)} points...")
        results = Parallel(n_jobs=-10, verbose=1)(  # reducing a bit njobs
            delayed(sig_bg)(_b, _t) for _b, _t in zip(bbcut_flat, ttcut_flat)
        )

        # Reshape results to match grid shape
        sigs, bgs_sb, bg_fails_res, bg_fails_sb, tfs = zip(*results)
        grid_shape = BBcut.shape

        sigs = np.array(sigs).reshape(grid_shape)
        bgs_sb = np.array(bgs_sb).reshape(grid_shape)
        bg_fails_res = np.array(bg_fails_res).reshape(grid_shape)
        bg_fails_sb = np.array(bg_fails_sb).reshape(grid_shape)
        tfs = np.array(tfs).reshape(grid_shape)

        tot_sig_weight = np.sum(self.sensitivity_vars["finalWeight"]["signal"])

        sigs_eff = sigs / tot_sig_weight

        bgs_scaled = bgs_sb * tfs

        results = {}
        for fom in foms:
            results[fom.name] = {}
            for B_min in B_min_vals:

                results[fom.name][f"Bmin={B_min}"] = {}

                sel_B_min = bgs_sb >= B_min
                if not np.any(sel_B_min):
                    print(f"Warning: No points satisfy B_min>{B_min} for FOM={fom.name}. Skipping.")
                    continue

                limits = fom.fom_func(bgs_sb, sigs, tfs)
                sel_indices = np.argwhere(sel_B_min)
                selected_limits = limits[sel_B_min]
                min_idx_in_selected = np.argmin(selected_limits)
                idx_opt = tuple(sel_indices[min_idx_in_selected])
                limit_opt = selected_limits[min_idx_in_selected]

                # Get optimal cuts (always in threshold space)
                bbcut_opt, ttcut_opt = BBcut[idx_opt], TTcut[idx_opt]

                # Prepare Optimum object with appropriate fields
                optimum_args = {
                    "limit": limit_opt,
                    "signal_yield": sigs[idx_opt],
                    "signal_eff": sigs_eff[idx_opt],
                    "bkg_yield": bgs_sb[idx_opt],
                    "hmass_fail": bg_fails_res[idx_opt],
                    "sideband_fail": bg_fails_sb[idx_opt],
                    "transfer_factor": tfs[idx_opt],
                    "cuts": (bbcut_opt, ttcut_opt),  # Always actual thresholds
                    "BBcut": BBcut,
                    "TTcut": TTcut,
                    "bb_disc_name": self.bb_disc_name,
                    "tt_disc_name": self.tt_disc_name_channel,
                    "sig_map": sigs,
                    "bg_map": bgs_scaled,
                    "sel_B_min": sel_B_min,
                    "fom": fom,
                }

                # Add signal efficiency specific fields if using signal efficiency coordinates
                if not use_thresholds:
                    bbcut_sig_eff_opt = BBcutSigEff[idx_opt]
                    ttcut_sig_eff_opt = TTcutSigEff[idx_opt]

                    optimum_args.update(
                        {
                            "sig_eff_cuts": (bbcut_sig_eff_opt, ttcut_sig_eff_opt),
                            "BBcut_sig_eff": BBcutSigEff,
                            "TTcut_sig_eff": TTcutSigEff,
                            "fom_map": limits,
                        }
                    )

                results[fom.name][f"Bmin={B_min}"] = CutOptimum(**optimum_args)

        return results

    def _adaptive_refine_limits(
        self, optimum: CutOptimum, use_thresholds: bool, top_fraction: float, padding: float
    ) -> dict:
        """Compute refined grid limits for the next stage from the union of padded regions around best points.

        This method:
        1. Identifies the top N% best grid points (lowest FOM values)
        2. Creates a padded region around each of these points
        3. Computes the union of all these regions
        4. Returns separate (min, max) limits for X and Y axes

        Args:
            optimum: CutOptimum object containing FOM map and grid coordinates
            use_thresholds: Whether using threshold or signal efficiency coordinates
            top_fraction: Fraction of best points to consider (e.g., 0.1 for top 10%)
            padding: Padding factor as fraction of grid spacing to add around each point

        Returns:
            dict: {'x': (x_min, x_max), 'y': (y_min, y_max)}
        """
        # Choose the coordinate system to operate in
        if (
            not use_thresholds
            and optimum.BBcut_sig_eff is not None
            and optimum.TTcut_sig_eff is not None
        ):
            X = optimum.BBcut_sig_eff
            Y = optimum.TTcut_sig_eff
        else:
            X = optimum.BBcut
            Y = optimum.TTcut

        fom_map = optimum.fom_map
        if fom_map is None or X is None or Y is None:
            # Fallback: keep previous limits
            x_min, x_max = (float(X.min()), float(X.max())) if X is not None else (0.0, 1.0)
            y_min, y_max = (float(Y.min()), float(Y.max())) if Y is not None else (0.0, 1.0)
            return {"x": (x_min, x_max), "y": (y_min, y_max)}

        # Flatten and select top fraction with lowest FOM
        flat_fom = fom_map.ravel()
        valid = np.isfinite(flat_fom)
        if not np.any(valid):
            return {"x": (float(X.min()), float(X.max())), "y": (float(Y.min()), float(Y.max()))}

        idx = np.where(valid)[0]
        k = max(1, int(len(idx) * max(0.0, min(top_fraction, 1.0))))
        best_idx = idx[np.argpartition(flat_fom[valid], k - 1)[:k]]

        # Get coordinates of best points
        x_flat = X.ravel()
        y_flat = Y.ravel()
        x_best = x_flat[best_idx]
        y_best = y_flat[best_idx]

        # Estimate grid spacing for padding
        # Use unique sorted values to compute typical spacing
        x_unique = np.unique(x_flat)
        y_unique = np.unique(y_flat)
        x_spacing = np.diff(x_unique).mean() if len(x_unique) > 1 else (X.max() - X.min()) / 10
        y_spacing = np.diff(y_unique).mean() if len(y_unique) > 1 else (Y.max() - Y.min()) / 10

        # Create padded regions around each best point
        x_pad = padding * x_spacing
        y_pad = padding * y_spacing

        # Compute union of all padded regions
        # For each best point, create a padded box and then take overall min/max
        x_mins = x_best - x_pad
        x_maxs = x_best + x_pad
        y_mins = y_best - y_pad
        y_maxs = y_best + y_pad

        # Union is the envelope of all these boxes
        x_min_refined = float(np.min(x_mins))
        x_max_refined = float(np.max(x_maxs))
        y_min_refined = float(np.min(y_mins))
        y_max_refined = float(np.max(y_maxs))

        # Clamp to current grid domain
        x_min_refined = max(float(X.min()), x_min_refined)
        x_max_refined = min(float(X.max()), x_max_refined)
        y_min_refined = max(float(Y.min()), y_min_refined)
        y_max_refined = min(float(Y.max()), y_max_refined)

        # Sanity check: ensure valid ranges
        if x_max_refined <= x_min_refined:
            x_min_refined, x_max_refined = float(X.min()), float(X.max())
        if y_max_refined <= y_min_refined:
            y_min_refined, y_max_refined = float(Y.min()), float(Y.max())

        print(
            f"  Refined limits: X=[{x_min_refined:.4f}, {x_max_refined:.4f}], Y=[{y_min_refined:.4f}, {y_max_refined:.4f}]"
        )

        return {"x": (x_min_refined, x_max_refined), "y": (y_min_refined, y_max_refined)}

    def adaptive_grid_search_opt(
        self,
        initial_gridsize,
        initial_gridlims,
        B_min_vals,
        foms,
        use_thresholds=False,
        stages: int = 2,
        refine_gridsize: int | None = None,
        top_fraction: float = 0.1,
        padding: float = 1.0,
        adaptive_bmin: int = None,
    ) -> CutOptimum:
        """Multi-stage adaptive refinement around best regions using existing grid_search_opt.

        Strategy:
        - Intermediate stages (1 to N-1): Run optimization with SINGLE B_min value for speed
        - Final stage (N): Run optimization with ALL B_min values for complete results

        Args:
            initial_gridsize: Initial grid size (NxN points)
            initial_gridlims: Initial grid limits, either tuple (min, max) or dict {'x': (min, max), 'y': (min, max)}
            B_min_vals: List of minimum background values (used in final stage only)
            foms: List of FOM functions to optimize
            use_thresholds: If True, use threshold coords; if False, use signal efficiency coords
            stages: Number of refinement stages
            refine_gridsize: Grid size for refinement stages (if None, uses initial_gridsize)
            top_fraction: Fraction of best points to consider (e.g., 0.1 = top 10%)
            padding: Padding in units of grid spacing around each best point (e.g., 1.0 = ±1 cell)
            adaptive_bmin: Single B_min value used for intermediate stages (default: DEFAULT_ADAPTIVE_BMIN)

        Returns:
            Final stage results dict in same format as grid_search_opt

        Note:
            The padding parameter is now interpreted as multiples of grid spacing, not fraction of range.
            Default of 1.0 means each best point is padded by ±1 grid cell.
        """
        if adaptive_bmin is None:
            adaptive_bmin = DEFAULT_ADAPTIVE_BMIN

        gridsize = initial_gridsize
        gridlims = initial_gridlims
        final_results = None

        for stage in range(max(1, stages)):
            # Use single B_min for intermediate stages, all B_min values for final stage
            is_final_stage = stage == stages - 1
            current_bmin_vals = B_min_vals if is_final_stage else [adaptive_bmin]

            stage_label = "FINAL" if is_final_stage else f"intermediate (B_min={adaptive_bmin})"
            print(f"\nAdaptive stage {stage+1}/{stages} ({stage_label}):")
            print(f"  gridsize={gridsize}, gridlims={gridlims}")
            print(f"  B_min values: {current_bmin_vals}")

            results = self.grid_search_opt(
                gridsize=gridsize,
                gridlims=gridlims,
                B_min_vals=current_bmin_vals,
                foms=foms,
                use_thresholds=use_thresholds,
            )
            final_results = results

            # Prepare next stage limits from the adaptive B_min result (not final stage)
            if stage < stages - 1:
                refined = False
                for fom in foms:
                    fom_res = results.get(fom.name, {})
                    if not fom_res:
                        continue
                    # Look specifically for the adaptive B_min result
                    opt_key = f"Bmin={adaptive_bmin}"
                    if opt_key in fom_res:
                        opt = fom_res[opt_key]
                        if isinstance(opt, CutOptimum) and opt.fom_map is not None:
                            gridlims = self._adaptive_refine_limits(
                                opt,
                                use_thresholds=use_thresholds,
                                top_fraction=top_fraction,
                                padding=padding,
                            )
                            refined = True
                            break

                if not refined:
                    print("Adaptive refinement: no valid regions found, stopping early.")
                    break

                gridsize = refine_gridsize or gridsize

        return final_results

    def perform_optimization(
        self,
        use_thresholds=False,
        plot=True,
        b_min_vals=None,
        adaptive: bool = False,
        adaptive_stages: int = 2,
        adaptive_top_frac: float = 0.1,
        adaptive_padding: float = 1.0,
        adaptive_refine_gridsize: int | None = None,
        adaptive_bmin: int = None,
    ):

        if b_min_vals is None:
            b_min_vals = [1, 10] if self.test_mode else DEFAULT_BMIN_VALUES

        if self.test_mode:
            gridlims = (0.3, 1) if use_thresholds else (0.2, 0.9)
            gridsize = 20
        else:
            gridlims = (0.7, 1) if use_thresholds else (0.25, 0.75)
            gridsize = 50 if not adaptive else 20

        foms = FOMS_TO_OPTIMIZE

        if adaptive:
            results = self.adaptive_grid_search_opt(
                initial_gridsize=gridsize,
                initial_gridlims=gridlims,
                B_min_vals=b_min_vals,
                foms=foms,
                use_thresholds=use_thresholds,
                stages=adaptive_stages,
                refine_gridsize=adaptive_refine_gridsize,
                top_fraction=adaptive_top_frac,
                padding=adaptive_padding,
                adaptive_bmin=adaptive_bmin,
            )
        else:
            results = self.grid_search_opt(
                gridsize=gridsize,
                gridlims=gridlims,
                B_min_vals=b_min_vals,
                foms=foms,
                use_thresholds=use_thresholds,
            )

        successful_b_min_vals = []
        saved_files = []
        for fom in foms:
            bmin_dfs = []
            for B_min in b_min_vals:
                optimum_result = results.get(fom.name, {}).get(f"Bmin={B_min}")
                if optimum_result:
                    successful_b_min_vals.append(B_min)
                    bmin_df = self.as_df(optimum_result, self.years, label=f"Bmin={B_min}")
                    bmin_dfs.append(bmin_df)

            if not bmin_dfs:
                print(f"No results for FOM {fom.name}, skipping.")
                continue

            # Create DataFrame for this FOM and transpose it
            fom_df = pd.concat(bmin_dfs).set_index("Label").T
            print(f"\nResults for FOM: {fom.name}")
            print(self.channel.label, "\n", fom_df.to_markdown())

            # Save separate CSV file for each FOM
            output_csv = (
                self.plot_dir
                / f"{'_'.join(self.years)}_opt_results_{fom.name}_{'thresh' if use_thresholds else 'sigeff'}.csv"
            )
            fom_df.to_csv(output_csv)
            saved_files.append(output_csv)
            print(f"Saved {fom.name} results to: {output_csv}")

        if not saved_files:
            print("No results to save.")
            return None

        print(f"\nSaved {len(saved_files)} CSV files:")
        for file in saved_files:
            print(f"  - {file}")

        if plot:
            while len(successful_b_min_vals) > 4:
                successful_b_min_vals = successful_b_min_vals[::2]

            if use_thresholds:
                plot_optimization_thresholds(
                    results=results,
                    years=self.years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(self.years)}_thresholds",
                    show=False,
                )
            else:
                # clip_value = 100 if "ggf" in self.sig_key.lower() else None
                clip_value = None  # should just do a 90% percentile or similar
                plot_optimization_sig_eff(
                    results=results,
                    years=self.years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(self.years)}_sigeff",
                    show=False,
                    use_log_scale=False,
                    clip_value=clip_value,
                )
                plot_optimization_sig_eff(
                    results=results,
                    years=self.years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(self.years)}_sigeff_log",
                    show=False,
                    use_log_scale=True,
                    clip_value=clip_value,
                )
        return results

    def study_foms(self, years):
        b_min_vals = np.arange(1, 17, 2)
        gridlims = (0.7, 1)
        gridsize = 10 if self.test_mode else 30

        foms = [FOMS["2sqrtB_S_var"], FOMS["punzi"]]

        results = self.grid_search_opt(
            gridsize=gridsize, gridlims=gridlims, B_min_vals=b_min_vals, foms=foms
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        markers = ["o", "s", "D", "^", "v", "<", ">", "p"]

        for i, fom in enumerate(foms):
            fom_name = fom.name
            if fom_name in results:
                ax.plot(
                    b_min_vals,
                    [results[fom_name][f"Bmin={B_min}"].limit for B_min in b_min_vals],
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    linewidth=2,
                    markersize=8,
                    label=fom.label,
                )

        ax.set_xlabel("$B_{min}$")
        ax.set_ylabel("FOM optimum  ")

        ax.text(
            0.05,
            0.72,
            self.channel.label,
            transform=ax.transAxes,
            fontsize=20,
            fontproperties="Tex Gyre Heros",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        hep.cms.label(
            ax=ax,
            label="Work in Progress",
            data=True,
            year="2022-23" if years == hh_vars.years else "+".join(years),
            com="13.6",
            fontsize=18,
            lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        )

        # Create plots directory if it doesn't exist
        plot_path = self.plot_dir / "fom_study"
        plot_path.mkdir(parents=True, exist_ok=True)

        # Save plots
        plt.savefig(
            plot_path / f"fom_comparison_{'_'.join(years)}.pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            plot_path / f"fom_comparison_{'_'.join(years)}.png",
            bbox_inches="tight",
        )

        plt.close()

        # Also create a summary table
        summary_data = []
        for B_min in b_min_vals:
            row = {"B_min": B_min}
            for fom in foms:
                fom_name = fom.name
                if fom_name in results:
                    optimum = results[fom_name][f"Bmin={B_min}"]
                    limit = fom.fom_func(
                        optimum.bkg_yield, optimum.signal_yield, optimum.transfer_factor
                    )
                    row[f"{fom_name}_limit"] = limit
                    row[f"{fom_name}_sig_yield"] = optimum.signal_yield
                    row[f"{fom_name}_bkg_yield"] = optimum.bkg_yield
                    row[f"{fom_name}_cuts"] = f"({optimum.cuts[0]:.3f}, {optimum.cuts[1]:.3f})"
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(plot_path / f"fom_summary_{'_'.join(years)}.csv", index=False)

        return

    @staticmethod
    def as_df(optimum, years, label="optimum"):
        """
        Converts an CutOptimum result to a pandas DataFrame row with derived quantities.

        Args:
            optimum: An CutOptimum dataclass instance.
            years: List of years used in the optimization.
            label: Optional label for the result row.
        Returns:
            pd.DataFrame with one row of results.
        """

        limits = {}
        limits["Label"] = label
        limits["Cut_Xbb"] = optimum.cuts[0]
        limits["Cut_Xtt"] = optimum.cuts[1]
        limits["Sig_Yield"] = optimum.signal_yield
        limits["Sideband Pass"] = optimum.bkg_yield
        limits["Higgs Mass Fail"] = optimum.hmass_fail
        limits["Sideband Fail"] = optimum.sideband_fail
        limits["BG_Yield_scaled"] = optimum.bkg_yield * optimum.transfer_factor
        limits["TF"] = optimum.transfer_factor
        limits["FOM"] = optimum.fom.label

        limits["Limit"] = optimum.limit
        limits["Limit_scaled_22_24"] = optimum.limit / np.sqrt(
            (124000 + hh_vars.LUMI["2022-2023"]) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        limits["Limit_scaled_Run3"] = optimum.limit / np.sqrt(
            (360000) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        return pd.DataFrame([limits])


def analyse_channel(
    events_dict,
    sr_config: SRConfig | None = None,
    actions=None,
    test_mode: bool | None = None,
    tt_pres=False,
    use_ParT=False,
    at_inference=False,
    adaptive=False,
    bdt_dir=None,
    plot_dir=None,
):
    """
    Analyse a given signal region configuration for a given channel.
    """

    print(
        f"Processing configuration: {sr_config.name} for channel: {sr_config.channel}. Test mode: {test_mode}."
    )

    # Create analyser with all signals (data and BDT predictions loaded once)
    analyser = Analyser(
        events_dict=events_dict,
        sr_config=sr_config,
        test_mode=test_mode,
        tt_pres=tt_pres,
        use_ParT=use_ParT,
        at_inference=at_inference,
        adaptive=adaptive,
        bdt_dir=bdt_dir,
        plot_dir=plot_dir,
    )

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_and_plot_rocs()
    if "plot_mass" in actions:
        analyser.plot_mass()
    if "sensitivity" in actions:
        # Run main signal optimization (with exclusions if configured)
        analyser.prepare_sensitivity()
        return analyser.perform_optimization(
            use_thresholds=False,
            adaptive=adaptive,
            **adaptive_kwargs,
        )

    if "fom_study" in actions:
        analyser.prepare_sensitivity()
        analyser.study_foms()

    # is this bad?
    # del analyser
    # gc.collect()


def get_plot_dir(
    test_mode: bool,
    tt_pres: bool,
    adaptive: bool,
    use_ParT: bool,
    do_vbf: bool,
    use_sm_signals: bool,
    overlapping_channels: bool,
    sr_config: SRConfig,
):

    if test_mode:
        test_dir = "test"
    elif tt_pres:
        test_dir = "tt_pres"
    else:
        test_dir = "full_presel"

    tag = "adaptive/" if adaptive else "grid/"
    tag += "ParT/" if use_ParT else "BDT/"
    tag += "do_vbf/" if do_vbf else "ggf_only/"
    tag += "sm_signals/" if use_sm_signals else "ggf_only/"
    tag += "overlapping_channels/" if overlapping_channels else "orthogonal_channels/"
    tag += f"{sr_config.name}/{sr_config.channel}"

    return Path(PLOT_DIR) / f"SensitivityStudy/{TODAY}/{test_dir}/{tag}"


def main(args):
    """
    Sensitivity study across channels and signal regions.

    Processes each channel sequentially, optimizing signal regions in order:
    GGF first, then VBF (if enabled). Later regions veto events selected by earlier ones.
    """
    # Signal regions to optimize: for now, GGF first, then VBF
    signal_regions = SIGNAL_ORDERING if args.do_vbf else ["ggfbbtt"]

    # BDT models to load (None if using ParT)
    models = None
    if not args.use_ParT:
        models = [args.ggf_modelname] + ([args.vbf_modelname] if args.do_vbf else [])

    # Track optimized regions for vetoes
    optimized_regions: dict[str, SRConfig] = {}  # across all channels (orthogonal mode)

    tt_disc_map = TT_DISC_NAMES_PART if args.use_ParT else TT_DISC_NAMES_BDT

    for channel_key in args.channels:
        print(f"\n{'='*60}\nProcessing channel: {channel_key}\n{'='*60}")

        events_dict = load_data_channel(
            years=args.years,
            signals=SM_SIGNALS,
            channel=CHANNELS[channel_key],
            test_mode=args.test_mode,
            tt_pres=args.tt_pres,
            models=models,
        )

        channel_regions: list[SRConfig] = []  # within current channel (overlapping mode)

        for region_name in signal_regions:
            print(f"\n  Optimizing: {region_name}")

            sr_config = SRConfig(
                name=region_name,
                signals=SM_SIGNALS if args.use_sm_signals else [region_name],
                channel=channel_key,
                bb_disc_name=BB_DISC_NAME,
                tt_disc_name=tt_disc_map[region_name],
            )

            # Apply vetoes from previously optimized regions
            regions_to_veto = (
                optimized_regions.values() if not args.overlapping_channels else channel_regions
            )
            for veto_region in regions_to_veto:
                sr_config.add_veto_from_optimum(veto_region, bmin=args.bmin_for_veto)

            plot_dir = get_plot_dir(
                args.test_mode,
                args.tt_pres,
                args.adaptive,
                args.use_ParT,
                args.do_vbf,
                args.use_sm_signals,
                args.overlapping_channels,
                sr_config,
            )
            plot_dir.mkdir(parents=True, exist_ok=True)

            sr_config.optima = analyse_channel(
                events_dict=events_dict,
                sr_config=sr_config,
                test_mode=args.test_mode,
                tt_pres=args.tt_pres,
                use_ParT=args.use_ParT,
                actions=args.actions,
                at_inference=args.at_inference,
                adaptive=args.adaptive,
                bdt_dir=args.bdt_dir,
                plot_dir=plot_dir,
            )

            # Register for future vetoes
            channel_regions.append(sr_config)
            if not args.overlapping_channels:
                optimized_regions[sr_config.name] = sr_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sensitivity Study Script")

    # Data loading arguments: years, channels, test_mode, tt_pres
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="List of years to include in the analysis",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=CHANNEL_ORDERING,
        help="List of channels to run (default: all)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Run in test mode (reduced data size)",
    )
    parser.add_argument(
        "--tt-pres",
        action="store_true",
        default=False,
        help="Apply tt preselection",
    )
    parser.add_argument(
        "--at-inference",
        action="store_true",
        default=False,
        help="Compute BDT predictions at inference time, instead of possibly loading cached predictions if array lengths match",
    )

    # Analysis arguments: actions, signals, models
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["compute_rocs", "plot_mass", "sensitivity", "fom_study", "minimization_study"],
        required=True,
        help="Actions to perform. Choose one or more.",
    )
    parser.add_argument(
        "--ggf-modelname",
        default="19oct25_ak4away_ggfbbtt",
        help="Name of the BDT model to use for sensitivity study in ggf channel",
    )
    parser.add_argument(
        "--use-sm-signals",
        action="store_true",
        default=False,
        help="Use total SM signals for sensitivity study, i.e. optimize for sum of SM signals",
    )
    parser.add_argument(
        "--do-vbf",
        action="store_true",
        default=False,
        help="Run VBF optimization first (with its own model) and veto its selection (Bmin=10) when optimizing the main signal",
    )
    parser.add_argument(
        "--vbf-modelname",
        type=str,
        default="19oct25_ak4away_vbfbbtt",
        help="BDT model name to use for VBF optimization (defaults to --modelname if empty)",
    )
    parser.add_argument(
        "--vbf-signal-key",
        type=str,
        default="vbfbbtt",
        help="Signal key for VBF optimization (e.g., vbfbbtt or vbfbbtt-k2v0)",
    )
    parser.add_argument(
        "--overlapping-channels",
        action="store_true",
        default=False,
        help="Make channels non orthogonal: each channel does not veto events selected by previous channels",
    )
    parser.add_argument(
        "--bmin-for-veto",
        type=int,
        default=10,
        help="Bmin value to use for cross-channel vetoes when not using --overlapping-channels",
    )

    # Options
    parser.add_argument(
        "--use-ParT",
        action="store_true",
        default=False,
        help="Use ParT for sensitivity study instead of BDTs",
    )
    parser.add_argument(
        "--use-thresholds",
        action="store_true",
        default=False,
        help="Use thresholds for optimization",
    )

    parser.add_argument(
        "--bdt-dir",
        help="directory where you have saved your bdt models for inference",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Use adaptive multi-stage grid refinement during sensitivity optimization",
    )

    args = parser.parse_args()

    main(args)
