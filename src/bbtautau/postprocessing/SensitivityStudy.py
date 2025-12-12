from __future__ import annotations

import argparse
import logging
from collections import defaultdict
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
from bbtautau.postprocessing.bbtautau_types import ABCD, FOM, SRConfig
from bbtautau.postprocessing.plotting import (
    plot_optimization_sig_eff,
    plot_optimization_thresholds,
)
from bbtautau.postprocessing.Regions import load_cuts_from_csv
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES, SM_SIGNALS
from bbtautau.postprocessing.utils import load_data_channel
from bbtautau.userConfig import (
    CHANNEL_ORDERING,
    CLASSIFIER_DIR,
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
DEFAULT_BMIN_VALUES = [1, 5, 10, 12]

# Define discriminant names
# BB_DISC_NAME = "bbFatJetParTXbbvsQCDTop" #Define in args for now
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

# Non-QCD backgrounds for data minus simulated non-QCD backgrounds ABCD estimation
NON_QCD_BGS = ["ttbarhad", "ttbarsl", "ttbarll"]


def fom_2sqrtB_S(b_qcd_sb, s, _tf, non_qcd_bg_in_pass_res=0):
    return np.where(
        s > 0, 2 * np.sqrt(b_qcd_sb * _tf + non_qcd_bg_in_pass_res) / s, -np.nan
    )  # Need nan or it will ruin the plot scales.


def fom_2sqrtB_S_var(b_qcd_sb, s, _tf, non_qcd_bg_in_pass_res=0):
    # qcd is data driven, i.e. data-ttbar
    qcd_in_pass_res = b_qcd_sb * _tf
    bg_in_pass_res = qcd_in_pass_res + non_qcd_bg_in_pass_res
    return np.where(
        (b_qcd_sb > 0) & (s > 0),
        2 * np.sqrt(bg_in_pass_res + (qcd_in_pass_res / np.sqrt(b_qcd_sb)) ** 2) / s,
        -np.nan,
    )


def fom_punzi(b_qcd_sb, s, _tf, a=3, non_qcd_bg_in_pass_res=0):
    """
    a is the number of sigmas of the test significance
    """
    return np.where(s > 0, (np.sqrt(b_qcd_sb * _tf + non_qcd_bg_in_pass_res) + a / 2) / s, -np.nan)


FOMS = {
    "2sqrtB_S": FOM(fom_2sqrtB_S, "$2\\sqrt{B}/S$", "2sqrtB_S"),
    "2sqrtB_S_var": FOM(fom_2sqrtB_S_var, "$2\\sqrt{B+B^2/\\tilde{B}}/S$", "2sqrtB_S_var"),
    "punzi": FOM(fom_punzi, "$(\\sqrt{B}+a/2)/S$", "punzi"),
}

FOMS_TO_OPTIMIZE = [FOMS["2sqrtB_S_var"]]

"""
# FoM function used in both standard and enhanced ABCD
# Region definitions:
#    High  |--------------------------------------|
#     |    |  A: pass & res  | B: pass & sideband |
#   score  |--------------------------------------|
#     |    |  C: fail & res  | D: fail & sideband |
#    Low   |---resonant-mass-|----mass-sideband---|
# Arguments:
# non_qcd_bg_in_pass_res = 0 means standard ABCD
#    b_qcd_sb = data in region B
#    tf = derived TF assuming all data in B, C, D are QCD/DY
# non_qcd_bg_in_pass_res != 0 is used for enhanced ABCD
#    b_qcd_sb = data-ttbar in region B
#    tf = TF derived from (data-ttbar) in B, C, D
"""


class Analyser:
    def __init__(
        self,
        events_dict,
        sr_config: SRConfig,
        test_mode: bool | None = None,
        use_ParT: bool | None = None,
        at_inference=False,
        tt_pres=False,
        bdt_dir=None,
        plot_dir: Path | None = None,
        llsl_weight=1,
        dataMinusSimABCD=False,
        showNonDataDrivenPortion=True,
        compute_ROC_metrics=False,
    ):
        """Initialize Analyser with support for multiple signals."""
        self.sr_config = sr_config
        self.events_dict = events_dict
        self.years = list(events_dict.keys())
        self.test_mode = test_mode
        self.tt_pres = tt_pres
        self.at_inference = at_inference
        self.use_ParT = use_ParT
        self.bdt_dir = bdt_dir

        self.llsl_weight = llsl_weight
        self.dataMinusSimABCD = dataMinusSimABCD
        self.showNonDataDrivenPortion = showNonDataDrivenPortion
        self.compute_ROC_metrics = compute_ROC_metrics
        print(
            "Using data minus simulated non-QCD backgrounds for ABCD estimation:",
            self.dataMinusSimABCD,
        )

        # Unpack some handy variables
        self.signals = sr_config.signals
        self.channel = CHANNELS[sr_config.channel]
        self.region_name = sr_config.name

        self.sig_keys_channel = [sig + sr_config.channel for sig in sr_config.signals]
        self.all_keys = (
            self.sig_keys_channel
            + self.channel.data_samples
            + (self.showNonDataDrivenPortion or self.dataMinusSimABCD) * NON_QCD_BGS
        )

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

        self.rocAnalyzer.compute_rocs(compute_metrics=self.compute_ROC_metrics)

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
        mbb1, mbb2 = SHAPE_VAR["blind_window"]

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
            "data": self.channel.data_samples,
        }
        if self.dataMinusSimABCD or self.showNonDataDrivenPortion:
            sig_vs_bkg_groups["nonQCDbkg"] = NON_QCD_BGS
            # Also add individual non-QCD backgrounds for breakdown
            for non_qcd_bkg in NON_QCD_BGS:
                sig_vs_bkg_groups[non_qcd_bkg] = [non_qcd_bkg]

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

        # pre-compute the sideband cuts for all groups
        self.sideband_cuts = {
            group_name: (self.sensitivity_vars["massbb"][group_name] < mbb1)
            | (self.sensitivity_vars["massbb"][group_name] > mbb2)
            for group_name in sig_vs_bkg_groups
        }

        del self.sensitivity_vars["massbb"]

    def _pass_cuts(self, group_name, txbbcut, txttcut):
        return (self.sensitivity_vars["txbbs"][group_name] > txbbcut) & (
            self.sensitivity_vars["txtts"][group_name] > txttcut
        )

    def compute_sig_bkg_abcd(self, txbbcut, txttcut):
        """Compute signal and background in ABCD regions, after significant cleanup."""

        # build the selections for the signals in the current channel
        cut_sigs_pass = self._pass_cuts("signal", txbbcut, txttcut)
        cut_data_pass = self._pass_cuts("data", txbbcut, txttcut)

        # Compute signal yield in the resonant region
        sig_pass = np.sum(
            self.sensitivity_vars["finalWeight"]["signal"][
                cut_sigs_pass & ~self.sideband_cuts["signal"]
            ]
        )

        bg_data = ABCD(
            pass_sideband=np.sum(
                self.sensitivity_vars["finalWeight"]["data"][
                    cut_data_pass & self.sideband_cuts["data"]
                ]
            ),
            fail_resonant=np.sum(
                self.sensitivity_vars["finalWeight"]["data"][
                    (~cut_data_pass) & (~self.sideband_cuts["data"])
                ]
            ),
            fail_sideband=np.sum(
                self.sensitivity_vars["finalWeight"]["data"][
                    (~cut_data_pass) & self.sideband_cuts["data"]
                ]
            ),
            isData=True,
        )

        bg_non_qcd = None  # Total non-QCD background
        breakdown_non_qcd_pass_res = (
            None  # If needed, breakdown of non-QCD background by category in pass, resonant region
        )

        # compute the non-QCD background if needed
        if self.showNonDataDrivenPortion or self.dataMinusSimABCD:
            cut_non_qcd_pass = self._pass_cuts("nonQCDbkg", txbbcut, txttcut)
            bg_non_qcd = ABCD(
                pass_sideband=np.sum(
                    self.sensitivity_vars["finalWeight"]["nonQCDbkg"][
                        cut_non_qcd_pass & self.sideband_cuts["nonQCDbkg"]
                    ]
                ),
                fail_resonant=np.sum(
                    self.sensitivity_vars["finalWeight"]["nonQCDbkg"][
                        (~cut_non_qcd_pass) & (~self.sideband_cuts["nonQCDbkg"])
                    ]
                ),
                fail_sideband=np.sum(
                    self.sensitivity_vars["finalWeight"]["nonQCDbkg"][
                        (~cut_non_qcd_pass) & self.sideband_cuts["nonQCDbkg"]
                    ]
                ),
                pass_resonant=np.sum(
                    self.sensitivity_vars["finalWeight"]["nonQCDbkg"][
                        cut_non_qcd_pass & ~self.sideband_cuts["nonQCDbkg"]
                    ]
                ),
                isData=False,
            )

            # Breakdown of non-QCD background by category in pass, resonant region
            breakdown_non_qcd_pass_res = defaultdict(float)
            for non_qcd_bkg in NON_QCD_BGS:
                cut_breakdown_non_qcd_pass = self._pass_cuts(non_qcd_bkg, txbbcut, txttcut)
                breakdown_non_qcd_pass_res[non_qcd_bkg] = np.sum(
                    self.sensitivity_vars["finalWeight"][non_qcd_bkg][
                        cut_breakdown_non_qcd_pass & ~self.sideband_cuts[non_qcd_bkg]
                    ]
                )

        # remove print statements or will flood output when doing scan
        # print("Am I showing the non QCD portion?", self.showNonDataDrivenPortion)

        return sig_pass, bg_data, bg_non_qcd, breakdown_non_qcd_pass_res

    def evaluate_at_cuts(self, txbbcut: float, txttcut: float, foms: list = None) -> dict:
        """
        Evaluate signal and background yields at a single cut point.

        Args:
            txbbcut: Cut value for bb tagger
            txttcut: Cut value for tt tagger
            foms: List of FOM functions to compute (default: FOMS_TO_OPTIMIZE)

        Returns:
            dict with evaluation results including signal, background, transfer factors, and FOM values
        """
        if foms is None:
            foms = FOMS_TO_OPTIMIZE

        # Get signal and background at the cut point
        sig_pass, bg_data, bg_non_qcd, breakdown = self.compute_sig_bkg_abcd(txbbcut, txttcut)

        # Build results dict
        results = {
            "cuts": (txbbcut, txttcut),
            "sig_pass": sig_pass,
            "data_pass_sideband": bg_data.pass_sideband,
            "data_fail_resonant": bg_data.fail_resonant,
            "data_fail_sideband": bg_data.fail_sideband,
            "TF_data": bg_data.tf,
            "bkg_ABCD": bg_data.pass_sideband * bg_data.tf,
        }

        # Add signal efficiency
        tot_sig_weight = np.sum(self.sensitivity_vars["finalWeight"]["signal"])
        results["sig_eff"] = sig_pass / tot_sig_weight if tot_sig_weight > 0 else 0

        # Add enhanced ABCD results if using data-minus-sim
        if self.dataMinusSimABCD and bg_non_qcd is not None:
            results["data_qcd_only_pass_sideband"] = bg_data.pass_sideband  # already subtracted
            results["TF_qcd_only"] = bg_data.tf
            results["non_qcd_pass_resonant"] = bg_non_qcd.pass_resonant
            results["bkg_EnhancedABCD"] = (
                bg_data.pass_sideband * bg_data.tf + bg_non_qcd.pass_resonant
            )

        # Add breakdown of non-QCD backgrounds by category
        if breakdown is not None:
            for non_qcd_bkg in NON_QCD_BGS:
                results[f"{non_qcd_bkg}_pass_resonant"] = breakdown.get(non_qcd_bkg, 0)

        # Compute FOM values
        for fom in foms:
            if self.dataMinusSimABCD and bg_non_qcd is not None:
                fom_val = fom.fom_func(
                    bg_data.pass_sideband,
                    sig_pass,
                    bg_data.tf,
                    non_qcd_bg_in_pass_res=bg_non_qcd.pass_resonant,
                )
            else:
                fom_val = fom.fom_func(
                    bg_data.pass_sideband, sig_pass, bg_data.tf, non_qcd_bg_in_pass_res=0
                )
            results[f"fom_{fom.name}"] = fom_val

        return results

    def run_evaluation(
        self,
        cuts_file: str | Path,
        bmin: int,
        outfile: str | Path | None = None,
    ) -> dict:
        """
        Full evaluation workflow: load cuts from CSV, evaluate, print results, and optionally save.

        This is the high-level method that orchestrates the evaluation process.

        Args:
            cuts_file: Path to CSV file with cuts (rows: Cut_Xbb, Cut_Xtt; columns: Bmin=N)
            bmin: B_min column to read from CSV
            outfile: Optional output CSV path (appends if exists)

        Returns:
            dict with all evaluation results including metadata
        """
        # Load cuts from CSV file
        txbbcut, txttcut = load_cuts_from_csv(cuts_file, bmin)

        # Prepare sensitivity variables if not already done
        if not hasattr(self, "sensitivity_vars") or self.sensitivity_vars is None:
            self.prepare_sensitivity()

        # Core evaluation
        results = self.evaluate_at_cuts(txbbcut, txttcut)

        # Add metadata
        results["region"] = self.region_name
        results["channel"] = self.channel.key
        results["years"] = "_".join(self.years)
        results["eval_bmin"] = bmin

        # Print summary
        self._print_evaluation_summary(results, txbbcut, txttcut)

        # Save if requested
        if outfile is not None:
            self._save_evaluation_results(results, outfile)

        return results

    def _print_evaluation_summary(self, results: dict, txbbcut: float, txttcut: float):
        """Print formatted evaluation results."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {self.region_name} / {self.channel.key}")
        print(f"{'='*60}")
        print(f"Cuts: txbb > {txbbcut:.4f}, txtt > {txttcut:.4f}")
        print(f"Signal yield: {results['sig_pass']:.2f}")
        print(f"Signal efficiency: {results['sig_eff']:.4f}")
        print(f"Background (ABCD): {results['bkg_ABCD']:.2f}")
        print(f"Transfer factor: {results['TF_data']:.4f}")
        for key, val in results.items():
            if key.startswith("fom_"):
                print(f"{key}: {val:.4f}")
        print(f"{'='*60}\n")

    def _save_evaluation_results(self, results: dict, outfile: str | Path):
        """Save evaluation results to CSV, appending if file exists."""
        outpath = Path(outfile)

        # Flatten tuple values (like cuts)
        flat_results = {}
        for k, v in results.items():
            if isinstance(v, tuple):
                flat_results[f"{k}_0"] = v[0]
                flat_results[f"{k}_1"] = v[1]
            else:
                flat_results[k] = v

        out = pd.DataFrame([flat_results])

        # Append if file exists
        if outpath.exists():
            existing_df = pd.read_csv(outpath)
            out = pd.concat([existing_df, out], ignore_index=True)

        out.to_csv(outpath, index=False)
        print(f"Results saved to: {outpath}")

    def grid_search_opt(
        self,
        gridsize,
        gridlims,
        B_min_vals,
        foms,
        use_thresholds=False,
    ) -> dict:
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
                print("ROC analyzer computed and plotted. Continuing with grid search...")

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
            # for optimization, pass a full grid
            bbcut = np.linspace(*gridlims_x, gridsize)
            ttcut = np.linspace(*gridlims_y, gridsize)

            BBcut, TTcut = np.meshgrid(bbcut, ttcut)

            # For consistency, set efficiency grids to None when not using signal efficiency
            BBcutSigEff = None
            TTcutSigEff = None

        # Flatten the grid for parallel evaluation
        bbcut_flat = BBcut.ravel()
        ttcut_flat = TTcut.ravel()

        # results is a list of (sig, bkg, tf) tuples
        print(f"Running grid search on {len(bbcut_flat)} points...")
        results = Parallel(n_jobs=-10, prefer="threads", verbose=1)(
            delayed(self.compute_sig_bkg_abcd)(_b, _t) for _b, _t in zip(bbcut_flat, ttcut_flat)
        )

        # Unpack results - each result is (sig_pass, bg_data, bg_non_qcd, breakdown_non_qcd_pass_res)
        grid_shape = BBcut.shape

        evals = defaultdict(list)

        for sig_pass, bg_data, bg_non_qcd, breakdown in results:
            evals["sig_pass"].append(sig_pass)
            evals["data_pass_resonant"].append(bg_data.pass_resonant)
            evals["data_pass_sideband"].append(bg_data.pass_sideband)

            evals["TF_data"].append(bg_data.tf)
            if bg_non_qcd is not None:
                evals["non_qcd_pass_resonant"].append(bg_non_qcd.pass_resonant)

            # Store breakdown of non-QCD backgrounds by category (only if breakdown was computed)
            if breakdown is not None:
                for non_qcd_bkg in NON_QCD_BGS:
                    evals[f"{non_qcd_bkg}_pass_resonant"].append(breakdown.get(non_qcd_bkg, 0))

            # subtract the non-QCD background from the data if desired
            if self.dataMinusSimABCD and bg_non_qcd is not None:
                bg_data_qcd_only = bg_data.subtract_MC(bg_non_qcd)
                evals["data_qcd_only_pass_sideband"].append(bg_data_qcd_only.pass_sideband)
                evals["TF_qcd_only"].append(bg_data_qcd_only.tf)

        # Reshape to 2D grid
        evals = {k: np.array(v).reshape(grid_shape) for k, v in evals.items()}

        # # Compute signal efficiency Need this?
        # evals["sig_eff"] = evals["sig_pass"] / np.sum(self.sensitivity_vars["finalWeight"]["signal"])

        # Now that have map with sigs, bkgs, go and compute FOMs
        evals_opt = {}

        for fom in foms:
            evals_opt[fom.name] = {}
            for B_min in B_min_vals:

                evals_opt[fom.name][f"Bmin={B_min}"] = {}

                sel_B_min = evals["data_pass_sideband"] >= B_min
                if self.dataMinusSimABCD:
                    limits = fom.fom_func(
                        evals["data_qcd_only_pass_sideband"],
                        evals["sig_pass"],
                        evals["TF_qcd_only"],
                        non_qcd_bg_in_pass_res=evals["non_qcd_pass_resonant"],
                    )
                else:
                    limits = fom.fom_func(
                        evals["data_pass_sideband"],
                        evals["sig_pass"],
                        evals["TF_data"],
                        non_qcd_bg_in_pass_res=0,
                    )
                if not np.any(sel_B_min):
                    print(f"Warning: No points satisfy B_min>{B_min} for FOM={fom.name}. Skipping.")
                    continue

                sel_indices = np.argwhere(sel_B_min)
                selected_limits = limits[sel_B_min]
                min_idx_in_selected = np.argmin(selected_limits)
                idx_opt = tuple(sel_indices[min_idx_in_selected])
                limit_opt = selected_limits[min_idx_in_selected]

                # Get optimal cuts (always in threshold space)
                bbcut_opt, ttcut_opt = BBcut[idx_opt], TTcut[idx_opt]

                # dictionary that stores all evaluated quantities at the optimum point
                evals_opt_fom_bmin = {k: v[idx_opt] for k, v in evals.items()}
                evals_opt_fom_bmin["limit"] = limit_opt
                evals_opt_fom_bmin["TXbb_opt"] = bbcut_opt
                evals_opt_fom_bmin["TXtt_opt"] = ttcut_opt
                evals_opt_fom_bmin["sel_B_min"] = sel_B_min

                # Add signal efficiency specific fields if using signal efficiency coordinates
                if not use_thresholds:
                    bbcut_sig_eff_opt = BBcutSigEff[idx_opt]
                    ttcut_sig_eff_opt = TTcutSigEff[idx_opt]

                    evals_opt_fom_bmin.update(
                        {
                            "sig_eff_cuts": (bbcut_sig_eff_opt, ttcut_sig_eff_opt),
                            "BBcut_sig_eff": BBcutSigEff,
                            "TTcut_sig_eff": TTcutSigEff,
                            "fom_map": limits,
                        }
                    )

                evals_opt[fom.name][f"Bmin={B_min}"] = evals_opt_fom_bmin

        return evals_opt

    def perform_optimization(self, use_thresholds=False, plot=True, b_min_vals=None):
        """
        Perform sensitivity optimization.

        Args:
            use_thresholds: Use threshold coordinates instead of signal efficiency
            plot: Generate plots
            b_min_vals: List of B_min values to optimize
        """
        if b_min_vals is None:
            b_min_vals = [1, 10] if self.test_mode else DEFAULT_BMIN_VALUES

        if self.test_mode:
            gridlims = (0.3, 1) if use_thresholds else (0.2, 0.9)
            gridsize = 20
        else:
            gridlims = (0.7, 1) if use_thresholds else (0.25, 0.75)
            gridsize = 50

        foms = FOMS_TO_OPTIMIZE

        results = self.grid_search_opt(
            gridsize=gridsize,
            gridlims=gridlims,
            B_min_vals=b_min_vals,
            foms=foms,
            use_thresholds=use_thresholds,
        )

        # Save results to CSV files
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
                    bb_disc_label=self.bb_disc_name,
                    tt_disc_label=self.tt_disc_name_channel,
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
                    bb_disc_label=self.bb_disc_name,
                    tt_disc_label=self.tt_disc_name_channel,
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
                    bb_disc_label=self.bb_disc_name,
                    tt_disc_label=self.tt_disc_name_channel,
                )
        return results

    def study_foms(self, years):

        # TODO: this needs to be tested

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
                    [results[fom_name][f"Bmin={B_min}"].get("limit", 0) for B_min in b_min_vals],
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
                    limit = optimum.get("limit", 0)
                    row[f"{fom_name}_limit"] = limit
                    row[f"{fom_name}_sig_yield"] = optimum.get("sig_pass", 0)
                    row[f"{fom_name}_bkg_yield"] = optimum.get("data_pass_sideband", 0)
                    row[f"{fom_name}_cuts"] = (
                        f"({optimum.get('TXbb_opt', 0):.3f}, {optimum.get('TXtt_opt', 0):.3f})"
                    )
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(plot_path / f"fom_summary_{'_'.join(years)}.csv", index=False)

        return

    @staticmethod
    def as_df(optimum: dict, years, label="optimum"):
        """
        Converts an optimum result dict to a pandas DataFrame row with derived quantities.

        Args:
            optimum: dict containing optimization results from grid_search_opt.
                Expected keys: TXbb_opt, TXtt_opt, limit, sig_pass, data_pass_sideband, etc.
            years: List of years used in the optimization.
            label: Optional label for the result row.
        Returns:
            pd.DataFrame with one row of results.
        """

        limits = {}
        limits["Label"] = label

        # Extract cuts from optimum (keys are TXbb_opt, TXtt_opt)
        limits["Cut_Xbb"] = optimum.get("TXbb_opt", 0)
        limits["Cut_Xtt"] = optimum.get("TXtt_opt", 0)

        # Add all evaluated quantities at optimum (they are stored directly, not nested)
        # Skip internal grid arrays and non-scalar values
        skip_keys = {"BBcut_sig_eff", "TTcut_sig_eff", "fom_map"}
        # {"BBcut_sig_eff", "TTcut_sig_eff", "fom_map", "TXbb_opt", "TXtt_opt", "sig_eff_cuts"}
        for k, v in optimum.items():
            if k not in skip_keys:
                limits[k] = v

        # Get limit for scaled projections
        limit = optimum.get("limit", 0)
        limits["Limit_scaled_22_24"] = limit / np.sqrt(
            (124000 + hh_vars.LUMI["2022-2023"]) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        limits["Limit_scaled_Run3"] = limit / np.sqrt(
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
    bdt_dir=None,
    plot_dir=None,
    llsl_weight=1,
    cuts_file=None,
    eval_bmin=None,
    outfile=None,
    dataMinusSimABCD=False,
    showNonDataDrivenPortion=True,
):
    """
    Analyse a given signal region configuration for a given channel.

    Args:
        cuts_file: Path to CSV file with cuts (for "evaluate" action)
        eval_bmin: B_min value to use from cuts_file (for "evaluate" action)
        outfile: Path to output CSV file (for "evaluate" action)
        dataMinusSimABCD: Use enhanced ABCD method (subtract simulated non-QCD)
        showNonDataDrivenPortion: Include non-QCD background in results
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
        llsl_weight=llsl_weight,
        bdt_dir=bdt_dir,
        plot_dir=plot_dir,
        dataMinusSimABCD=dataMinusSimABCD,
        showNonDataDrivenPortion=showNonDataDrivenPortion,
    )

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_and_plot_rocs()
    if "plot_mass" in actions:
        analyser.plot_mass()

    # Sensitivity actions are mutually exclusive
    if "sensitivity" in actions:
        analyser.prepare_sensitivity()
        return analyser.perform_optimization(
            use_thresholds=False,
        )
    elif "fom_study" in actions:
        analyser.prepare_sensitivity()
        analyser.study_foms(analyser.years)
        return None
    elif "evaluate" in actions:
        if cuts_file is None or eval_bmin is None:
            raise ValueError("'evaluate' action requires --cuts-file and --eval-bmin arguments")
        return analyser.run_evaluation(cuts_file, eval_bmin, outfile)

    return None


def get_plot_dir(
    test_mode: bool,
    tt_pres: bool,
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

    tag = "ParT/" if use_ParT else "BDT/"
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

        additional_samples = (
            {sample: SAMPLES[sample] for sample in NON_QCD_BGS}
            if (args.dataMinusSimABCD or args.showNonDataDrivenPortion)
            else None
        )

        events_dict = load_data_channel(
            years=args.years,
            signals=SM_SIGNALS,
            channel=CHANNELS[channel_key],
            test_mode=args.test_mode,
            tt_pres=args.tt_pres,
            models=models,
            additional_samples=additional_samples,
        )

        channel_regions: list[SRConfig] = []  # within current channel (overlapping mode)

        for region_name in signal_regions:
            print(f"\n  Optimizing: {region_name}")

            sr_config = SRConfig(
                name=region_name,
                signals=SM_SIGNALS if args.use_sm_signals else [region_name],
                channel=channel_key,
                bb_disc_name=args.bb_disc,
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
                bdt_dir=args.bdt_dir,
                plot_dir=plot_dir,
                cuts_file=args.cuts_file,
                eval_bmin=args.eval_bmin,
                outfile=args.outfile,
                dataMinusSimABCD=args.dataMinusSimABCD,
                showNonDataDrivenPortion=args.showNonDataDrivenPortion,
            )

            # Register for future vetoes
            channel_regions.append(sr_config)
            if not args.overlapping_channels:
                optimized_regions[sr_config.name] = sr_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sensitivity Study Script for HH→bbττ analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sensitivity optimization with BDT
  python SensitivityStudy.py --actions sensitivity --channels he hmu

  # Evaluate at specific cuts from a CSV file
  python SensitivityStudy.py --actions evaluate --cuts-file results.csv --eval-bmin 10 --outfile eval.csv

  # Run with ParT discriminants instead of BDT
  python SensitivityStudy.py --actions sensitivity --use-ParT
        """,
    )

    # =========================================================================
    # Data Selection
    # =========================================================================
    data_group = parser.add_argument_group(
        "Data Selection", "Arguments controlling which data to load and process"
    )
    data_group.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="Years to include (default: 2022-2023)",
    )
    data_group.add_argument(
        "--channels",
        nargs="+",
        default=CHANNEL_ORDERING,
        help="Channels to run (default: all)",
    )
    data_group.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Run with reduced data for quick testing",
    )
    data_group.add_argument(
        "--tt-pres",
        action="store_true",
        default=False,
        help="Apply ττ preselection cuts",
    )

    # =========================================================================
    # Action Selection
    # =========================================================================
    action_group = parser.add_argument_group(
        "Action Selection",
        "Choose which analysis to run (sensitivity actions are mutually exclusive)",
    )
    action_group.add_argument(
        "--actions",
        nargs="+",
        choices=["compute_rocs", "plot_mass", "sensitivity", "fom_study", "evaluate"],
        required=True,
        help="Actions: compute_rocs, plot_mass can combine with others; "
        "sensitivity/fom_study/evaluate are mutually exclusive",
    )

    # =========================================================================
    # Evaluate Action Arguments
    # =========================================================================
    eval_group = parser.add_argument_group(
        "Evaluate Action", "Arguments for --actions evaluate (evaluate yields at specific cuts)"
    )
    eval_group.add_argument(
        "--cuts-file",
        type=str,
        default=None,
        help="CSV file with cuts (rows: Cut_Xbb, Cut_Xtt; columns: Bmin=N)",
    )
    eval_group.add_argument(
        "--eval-bmin",
        type=int,
        default=None,
        help="B_min column to read from --cuts-file",
    )
    eval_group.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output CSV for results (appends if exists)",
    )

    # =========================================================================
    # Sensitivity Action Arguments
    # =========================================================================
    sens_group = parser.add_argument_group(
        "Sensitivity Action", "Arguments for --actions sensitivity (grid search optimization)"
    )
    sens_group.add_argument(
        "--use-thresholds",
        action="store_true",
        default=False,
        help="Optimize in threshold space (default: signal efficiency space)",
    )
    sens_group.add_argument(
        "--dataMinusSimABCD",
        action="store_true",
        help="Use enhanced ABCD: subtract simulated non-QCD from data",
    )
    sens_group.add_argument(  # TODO: make sense of this (right now can never be false)
        "--showNonDataDrivenPortion",
        action="store_true",
        default=True,
        help="Show non-QCD (ttbar) background portion in results (default: True)",
    )

    # =========================================================================
    # Signal Region Configuration
    # =========================================================================
    sr_group = parser.add_argument_group(
        "Signal Region Configuration", "Configure signal regions and cross-region vetoes"
    )
    sr_group.add_argument(
        "--use-sm-signals",
        action="store_true",
        default=False,
        help="Optimize for sum of SM signals (ggF+VBF) instead of single signal",
    )
    sr_group.add_argument(
        "--do-vbf",
        action="store_true",
        default=False,
        help="Also optimize VBF region (runs after ggF, applies veto)",
    )
    sr_group.add_argument(  # leave there for testing/legacy purposes
        "--overlapping-channels",
        action="store_true",
        default=False,
        help="Allow channels to overlap (no cross-channel vetoes)",
    )
    sr_group.add_argument(
        "--bmin-for-veto",
        type=int,
        default=10,
        help="B_min for cross-region/channel vetoes (default: 10)",
    )

    # =========================================================================
    # Discriminator Configuration
    # =========================================================================
    disc_group = parser.add_argument_group(
        "Discriminator Configuration", "Configure which discriminants to use"
    )
    disc_group.add_argument(  # Keep for legacy
        "--use-ParT",
        action="store_true",
        default=False,
        help="Use ParT scores instead of BDT",
    )
    disc_group.add_argument(
        "--ggf-modelname",
        default="19oct25_ak4away_ggfbbtt",
        help="BDT model name for ggF (default: 19oct25_ak4away_ggfbbtt)",
    )
    disc_group.add_argument(
        "--vbf-modelname",
        type=str,
        default="19oct25_ak4away_vbfbbtt",
        help="BDT model name for VBF (default: 19oct25_ak4away_vbfbbtt)",
    )
    disc_group.add_argument(
        "--bdt-dir",
        type=str,
        default=CLASSIFIER_DIR,
        help="Directory containing BDT model files",
    )
    disc_group.add_argument(
        "--at-inference",
        action="store_true",
        default=False,
        help="Force BDT inference (ignore cached predictions)",
    )
    disc_group.add_argument(
        "--llsl-weight",
        type=float,
        default=1.0,
        help="Weight for TTSL/TTLL BDT scores in ττ discriminant",
    )
    disc_group.add_argument(
        "--bb-disc",
        type=str,
        default="bbFatJetParTXbbvsQCDTop",
        choices=["bbFatJetParTXbbvsQCD", "bbFatJetParTXbbvsQCDTop", "bbFatJetPNetXbbvsQCDLegacy"],
        help="bb discriminator variable",
    )
    disc_group.add_argument(
        "--compute-ROC-metrics",
        action="store_true",
        default=False,
        help="Compute ROC metrics (default: False)",
    )

    # =========================================================================
    # Deprecated / Legacy
    # =========================================================================
    legacy_group = parser.add_argument_group(
        "Deprecated", "Legacy arguments (may be removed in future)"
    )
    legacy_group.add_argument(  # for now useless: define signals in SIGNAL_ORDERING
        "--vbf-signal-key",
        type=str,
        default="vbfbbtt",
        help="[Deprecated] Signal key for VBF",
    )

    args = parser.parse_args()

    main(args)

    # def update_cuts_from_csv_file(self, sensitivity_dir):
    #     # read the first available FOM CSV file
    #         csv_dir = Path(sensitivity_dir).joinpath(
    #             f"full_presel/{self.modelname if self.use_bdt else 'ParT'}/{self.channel.key}"
    #         )

    #         # Look for any FOM-specific CSV files
    #         csv_files = list(csv_dir.glob("*_opt_results_*.csv"))

    #         if len(csv_files) == 0:
    #             raise ValueError(f"No sensitivity CSV files found in {csv_dir}")

    #         # Take the first CSV file found and extract FOM name
    #         csv_file = sorted(csv_files)[0]  # Sort for reproducible behavior
    #         print(f"Reading CSV: {csv_file}")

    #         # Extract FOM name from filename like "2022_2022EE_opt_results_2sqrtB_S_var.csv"
    #         if "_opt_results_" in csv_file.name:
    #             fom_name = csv_file.name.split("_opt_results_")[1].replace(".csv", "")
    #         else:
    #             fom_name = "unknown"

    #         # Read as simple CSV (no multi-level headers)
    #         opt_results = pd.read_csv(csv_file, index_col=0)
    #         print(f"Using FOM: {fom_name}")
    #         print(f"Available B_min values: {opt_results.columns.tolist()}")

    #         # Check if the target Bmin column exists
    #         target_col = f"Bmin={self.bmin}"
    #         if target_col not in opt_results.columns:
    #             raise ValueError(
    #                 f"B_min={args.bmin} not found in CSV. Available: {opt_results.columns.tolist()}"
    #             )

    #         # update the CHANNEL cuts
    #         self.channel.txbb_cut = float(opt_results.loc["Cut_Xbb", target_col])
    #         if args.use_bdt:
    #             self.channel.txtt_BDT_cut = float(opt_results.loc["Cut_Xtt", target_col])
    #         else:
    #             self.channel.txtt_cut = float(opt_results.loc["Cut_Xtt", target_col])

    #         print(
    #             f"Updated TXbb and Txtt cuts to {self.channel.txbb_cut} and {self.channel.txtt_cut if not self.use_bdt else self.channel.txtt_BDT_cut} for {self.channel.key}"
    #         )
