from __future__ import annotations

import argparse
import gc
import logging
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import hh_vars
from boostedhh.utils import PAD_VAL
from joblib import Parallel, delayed

from bbtautau.postprocessing import utils
from bbtautau.postprocessing.bdt_utils import compute_or_load_bdt_preds
from bbtautau.postprocessing.plotting import (
    plot_optimization_sig_eff,
    plot_optimization_thresholds,
)
from bbtautau.postprocessing.postprocessing import (
    apply_triggers,
    base_filter,
    bbtautau_assignment,
    delete_columns,
    derive_lepton_variables,
    derive_variables,
    get_columns,
    leptons_assignment,
    load_samples,
    tt_filters,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.Samples import CHANNELS, SM_SIGNALS_CHANNELS
from bbtautau.userConfig import BDT_EVAL_DIR, DATA_PATHS, MODEL_DIR, SHAPE_VAR

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


class FOM:
    def __init__(self, fom_func, label, name):
        self.fom_func = fom_func
        self.label = label
        self.name = name


@dataclass
class Optimum:
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
class SignalConfig:
    """Configuration for a signal in multi-signal analysis.

    This supports defining orthogonal signal regions (e.g., VBF-enriched and GGF-enriched).
    When optimizing one signal, you can exclude events selected by other signals to avoid overlap.

    Attributes:
        key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt')
        modelname: BDT model name for this signal
        optimize: Whether to optimize cuts for this signal (default: True)
        bmin_for_exclusion: B_min value to use when extracting cuts for excluding this signal's
                           region from other signals (default: 10.0)

    Example:
        # Define VBF and GGF regions where GGF excludes VBF region
        signals = [
            SignalConfig(key="vbfbbtt", modelname="vbf_model", optimize=True),
            SignalConfig(key="ggfbbtt", modelname="ggf_model", optimize=True),
        ]
        # After VBF optimization, its cuts at bmin=10 will be used to exclude VBF region from GGF
    """

    key: str
    modelname: str
    optimize: bool = True
    bmin_for_exclusion: float = 10.0


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


class Analyser:
    def __init__(
        self,
        years,
        signals: (
            list[SignalConfig] | SignalConfig | str
        ),  # Can be list, single SignalConfig, or legacy str
        channel_key,
        test_mode,
        use_bdt,
        main_plot_dir,
        at_inference=False,
        tt_pres=False,
        use_sm_signals=True,
        # Legacy params for backward compatibility
        modelname: str | None = None,
        bdt_dir: str | None = None,  # Override MODEL_DIR from userConfig if provided
    ):
        """Initialize Analyser with support for multiple signals.

        Args:
            years: List of years to analyze
            signals: Either:
                - List of SignalConfig objects for multi-signal analysis
                - Single SignalConfig object
                - str (legacy): signal key like 'ggfbbtt' (uses modelname param)
            channel_key: Channel identifier
            test_mode: Whether in test mode
            use_bdt: Whether to use BDT (vs ParticleNet)
            main_plot_dir: Base directory for plots
            at_inference: Whether at inference time
            modelname: (Legacy) Model name when signals is a string
            tt_pres: Whether to use tt preselection
            use_sm_signals: Whether to use SM signal for sensitivity study (i.e. sum)
        """
        self.tt_pres = tt_pres
        self.channel = CHANNELS[channel_key]
        self.years = years
        self.test_mode = test_mode
        self.use_bdt = use_bdt
        self.at_inference = at_inference
        self.model_dir = Path(bdt_dir) if bdt_dir is not None else MODEL_DIR
        self.taukey = CHANNELS[channel_key].tagger_label

        # Convert legacy API to new multi-signal format
        if isinstance(signals, str):
            # Legacy: single signal key passed as string
            if modelname is None:
                raise ValueError("modelname must be provided when using legacy string signal key")
            signals = [SignalConfig(key=signals, modelname=modelname, optimize=True)]
        elif isinstance(signals, SignalConfig):
            # Single SignalConfig passed
            signals = [signals]

        # Store all signals
        self.signals: dict[str, SignalConfig] = {sig.key: sig for sig in signals}

        # The "target signal" is the one we're currently optimizing for
        # In single-signal mode, it's the only signal
        # In multi-signal mode, it's set dynamically during optimization
        self.target_signal_key: str | None = None
        if len(self.signals) == 1:
            self.target_signal_key = next(iter(self.signals.keys()))

        # Legacy compatibility attributes (use first signal or target signal)
        first_signal = next(iter(self.signals.values()))
        self.sig_key = first_signal.key
        self.sig_key_channel = first_signal.key + channel_key

        if use_sm_signals:
            self.sig_keys_sum = SM_SIGNALS_CHANNELS
        else:
            self.sig_keys_sum = [self.sig_key_channel]

        self.modelname = first_signal.modelname

        # Create discriminant names for all signals
        self.bb_disc_name = "bbFatJetParTXbbvsQCDTop"  # Same for all signals
        self.tt_disc_names: dict[str, str] = {}  # Maps signal_key -> discriminant_name

        for sig_key in self.signals:
            if use_bdt:
                # Remove trailing 'tt' from signal key to avoid redundancy in BDT names
                # e.g., ggfbbtt -> ggfbb, giving BDTggfbbtauhtauh instead of BDTggfbbtttauhtauh
                signal_base = sig_key.removesuffix("tt") if sig_key.endswith("tt") else sig_key
                self.tt_disc_names[sig_key] = f"BDT{signal_base}{self.taukey}vsAll"
            else:
                self.tt_disc_names[sig_key] = f"ttFatJetParTX{self.taukey}vsQCDTop"

        # Legacy compatibility: tt_disc_name points to first signal's discriminant
        self.tt_disc_name = self.tt_disc_names[first_signal.key]

        # Exclusion cuts: stores optimized cuts for each signal that should be excluded from others
        # Maps signal_key -> (txbb, txtt) cuts
        self.exclusion_cuts: dict[str, tuple[float, float]] = {}

        # Event data (shared across all signals)
        self.events_dict = {year: {} for year in years}

        # Setup plot directory
        if test_mode:
            test_dir = "test"
        elif tt_pres:
            test_dir = "tt_pres"
        else:
            test_dir = "full_presel"

        tt_tagger_name = first_signal.modelname if use_bdt else "ParT"
        self.plot_dir = (
            Path(main_plot_dir)
            / f"plots/SensitivityStudy/{TODAY}/{test_dir}/{first_signal.key}/{tt_tagger_name}/{channel_key}"
        )
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):

        for year in self.years:

            filters_dict = base_filter(self.test_mode)

            # Prefilters already applied in skimmer
            # filters_dict = bb_filters(filters_dict, num_fatjets=3, bb_cut=0.3)

            if self.tt_pres:
                filters_dict = tt_filters(
                    channel=self.channel, in_filters=filters_dict, num_fatjets=3, tt_cut=0.3
                )

            columns = get_columns(year, triggers_in_channel=self.channel)

            # Load data for ALL signals (creates keys like {signal_key}{channel_key})
            # This ensures all signal samples are available in events_dict
            signal_keys = list(self.signals.keys())
            self.events_dict[year] = load_samples(
                year=year,
                paths=DATA_PATHS[year],
                signals=signal_keys,
                channels=[self.channel],
                filters_dict=filters_dict,
                load_columns=columns,
                restrict_data_to_channel=True,
                loaded_samples=True,
                multithread=True,
            )

            apply_triggers(self.events_dict[year], year, self.channel)
            delete_columns(self.events_dict[year], year, channels=[self.channel])

            derive_variables(self.events_dict[year])
            bbtautau_assignment(self.events_dict[year], agnostic=True)
            leptons_assignment(self.events_dict[year], dR_cut=1.5)
            derive_lepton_variables(self.events_dict[year])

        # Load BDT predictions for ALL signals in a single pass (if using BDT)
        if self.use_bdt:
            for sig_key, sig_config in self.signals.items():
                print(
                    f"Loading BDT predictions for signal: {sig_key} (model: {sig_config.modelname})"
                )
                compute_or_load_bdt_preds(
                    events_dict=self.events_dict,
                    modelname=sig_config.modelname,
                    model_dir=self.model_dir,
                    signal_key=sig_key,
                    channel=self.channel,
                    bdt_preds_dir=BDT_EVAL_DIR,
                    tt_pres=self.tt_pres,
                    test_mode=self.test_mode,
                    at_inference=self.at_inference,
                    all_outs=True,
                )
        return

    @staticmethod
    def get_jet_vals(vals, mask, nan_to_pad=True):

        # TODO: Deprecate this (just need to use get_var properly)

        # check if vals is a numpy array
        if not isinstance(vals, np.ndarray):
            vals = vals.to_numpy()
        if len(vals.shape) == 1:
            warnings.warn(
                f"vals is a numpy array of shape {vals.shape}. Ignoring mask.", stacklevel=2
            )
            return vals if not nan_to_pad else np.nan_to_num(vals, nan=PAD_VAL)
        if nan_to_pad:
            return np.nan_to_num(vals[mask], nan=PAD_VAL)
        else:
            return vals[mask]

    def compute_and_plot_rocs(self, years, discs=None):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")

        self.events_dict_allyears = utils.concatenate_years(self.events_dict, years)

        background_names = self.channel.data_samples

        discs_tt = [
            f"ttFatJetParTX{self.taukey}vsQCD",
            f"ttFatJetParTX{self.taukey}vsQCDTop",
        ]

        discs_bb = ["bbFatJetParTXbbvsQCD", "bbFatJetParTXbbvsQCDTop", "bbFatJetPNetXbbvsQCDLegacy"]

        if self.use_bdt:
            # Use signal-specific BDT score names (includes signal key prefix, with trailing 'tt' removed)
            signal_base = (
                self.sig_key.removesuffix("tt") if self.sig_key.endswith("tt") else self.sig_key
            )
            discs_tt += [
                f"BDT{signal_base}{self.taukey}vsQCD",
                f"BDT{signal_base}{self.taukey}vsAll",
            ]

        discs_all = discs or (discs_bb + discs_tt)

        self.rocAnalyzer = ROCAnalyzer(
            years=years,
            signals={self.sig_key_channel: self.events_dict_allyears[self.sig_key_channel]},
            backgrounds={bkg: self.events_dict_allyears[bkg] for bkg in background_names},
        )

        self.rocAnalyzer.fill_discriminants(
            discs_all, signal_name=self.sig_key_channel, background_names=background_names
        )

        self.rocAnalyzer.compute_rocs()

        # Plot bb
        self.rocAnalyzer.plot_rocs(title="bbFatJet", disc_names=discs_bb, plot_dir=self.plot_dir)

        # Plot tt
        self.rocAnalyzer.plot_rocs(title="ttFatJet", disc_names=discs_tt, plot_dir=self.plot_dir)

        for disc in discs_all:
            self.rocAnalyzer.plot_disc_scores(disc, background_names, self.plot_dir)
            self.rocAnalyzer.plot_disc_scores(
                disc, [[bkg] for bkg in background_names], self.plot_dir
            )
            self.rocAnalyzer.compute_confusion_matrix(disc, plot_dir=self.plot_dir)

        print(f"ROCs computed and plotted for years {years} and signals {self.sig_key_channel}.")

    def plot_mass(self, years):
        for key in [self.sig_key_channel, "data"]:
            print(f"Plotting mass for {key}")
            if key == self.sig_key_channel:
                events = pd.concat([self.events_dict[year][key].events for year in years])
            else:
                events = pd.concat(
                    [
                        self.events_dict[year][dkey].events
                        for dkey in self.channel.data_samples
                        for year in years
                    ]
                )

            bins = np.linspace(0, 250, 50)

            fig, axs = plt.subplots(1, 2, figsize=(24, 10))

            for i, (jet, jlabel) in enumerate(
                zip(["bb", "tt"], ["bb FatJet", rf"{self.channel.label} FatJet"])
            ):
                ax = axs[i]
                if key == self.sig_key_channel:
                    mask = np.concatenate(
                        [
                            self.events_dict[year][self.sig_key_channel].get_mask(jet)
                            for year in years
                        ],
                        axis=0,
                    )
                else:
                    mask = np.concatenate(
                        [
                            self.events_dict[year][dkey].get_mask(jet)
                            for dkey in self.channel.data_samples
                            for year in years
                        ],
                        axis=0,
                    )

                print("Shape of events", events.shape)
                print("Shape of mask", mask.shape)

                for j, (mkey, mlabel) in enumerate(
                    zip(
                        [
                            "ak8FatJetMsd",
                            "ak8FatJetPNetmassLegacy",
                            "ak8FatJetParTmassResApplied",
                            "ak8FatJetParTmassVisApplied",
                        ],
                        ["SoftDrop", "PNetLegacy", "ParT Res", "ParT Vis"],
                    )
                ):
                    ax.hist(
                        self.get_jet_vals(events[mkey], mask),
                        bins=bins,
                        histtype="step",
                        weights=events["finalWeight"],
                        label=mlabel,
                        linewidth=2,
                        color=plt.cm.tab10.colors[j],
                    )

                ax.vlines(125, 0, ax.get_ylim()[1], linestyle="--", color="k", alpha=0.1)
                # ax.set_title(jlabel, fontsize=24)
                ax.set_xlabel("Mass [GeV]")
                ax.set_ylabel("Events")
                ax.legend()
                ax.set_ylim(0)
                hep.cms.label(
                    ax=ax,
                    label="Preliminary",
                    data=key == "data",
                    year="2022-23" if years == hh_vars.years else "+".join(years),
                    com="13.6",
                    fontsize=20,
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
                )

                ax.text(
                    0.03,
                    0.92,
                    jlabel,
                    transform=ax.transAxes,
                    fontsize=24,
                    # fontproperties="Tex Gyre Heros:bold",
                )

            # create mass directory if it doesn't exist
            (self.plot_dir / "mass").mkdir(parents=True, exist_ok=True)

            # save plots
            plt.savefig(
                self.plot_dir / f"mass/{key+'_'+jet+'_'.join(years)}.png",
                bbox_inches="tight",
            )
            plt.savefig(
                self.plot_dir / f"mass/{key+'_'+jet+'_'.join(years)}.pdf",
                bbox_inches="tight",
            )

    def prepare_sensitivity(self, years):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")

        mbbk = SHAPE_VAR["name"]
        mttk = self.channel.tt_mass_cut[0]

        """
        for tautau mass regression:
            -for hh use pnetlegacy
            -leptons : part Res
        """

        # TODO: apply pt cuts here!

        self.txbbs = {year: {} for year in years}
        self.txtts = {year: {} for year in years}  # Primary signal discriminants

        # Multi-signal discriminants: {signal_key: {year: {sample: scores}}}
        self.txtts_multi: dict[str, dict[str, dict[str, np.ndarray]]] = {
            sig_key: {year: {} for year in years} for sig_key in self.signals
        }

        self.masstt = {year: {} for year in years}
        self.massbb = {year: {} for year in years}
        self.ptbb = {year: {} for year in years}
        self.pttt = {year: {} for year in years}

        # Precompute to speedup - load discriminants for all signals
        for year in years:
            for key in [self.sig_key_channel] + self.channel.data_samples:
                # BB discriminant (same for all signals)
                self.txbbs[year][key] = self.events_dict[year][key].get_var(self.bb_disc_name)

                # Load TT discriminants for all signals
                for sig_key in self.signals:
                    disc_name = self.tt_disc_names[sig_key]
                    self.txtts_multi[sig_key][year][key] = self.events_dict[year][key].get_var(
                        disc_name
                    )

                # Primary signal discriminant (for legacy compatibility - use sig_key which is set to first signal)
                self.txtts[year][key] = self.txtts_multi[self.sig_key][year][key]

                # Mass and pT variables
                self.masstt[year][key] = self.events_dict[year][key].get_var(mttk)
                self.massbb[year][key] = self.events_dict[year][key].get_var(mbbk)
                self.ptbb[year][key] = self.events_dict[year][key].get_var("bbFatJetPt")
                self.pttt[year][key] = self.events_dict[year][key].get_var("ttFatJetPt")

    def compute_sig_bkg_abcd(
        self, years, txbbcut, txttcut, mbb1, mbb2, mtt1, mtt2, apply_exclusions=True
    ):
        """Compute signal and background in ABCD regions.

        Args:
            years: Years to process
            txbbcut: TxBB discriminant cut
            txttcut: TxTT discriminant cut
            mbb1, mbb2: bb mass window boundaries
            mtt1, mtt2: tt mass window boundaries
            apply_exclusions: Whether to apply exclusion cuts from other signals (default: True).
                             Set to False when optimizing a signal to define its exclusion region.
        """
        # pass/fail from taggers
        sig_pass = 0  # resonant region pass, signal
        bg_pass_sb = 0  # sideband region pass, data
        bg_fail_res = 0  # resonant region fail, data
        bg_fail_sb = 0  # sideband region fail, data
        for year in years:

            cut_sig_pass = (
                (self.txbbs[year][self.sig_key_channel] > txbbcut)
                & (self.txtts[year][self.sig_key_channel] > txttcut)
                & (self.massbb[year][self.sig_key_channel] > mbb1)
                & (self.massbb[year][self.sig_key_channel] < mbb2)
                & (self.ptbb[year][self.sig_key_channel] > 250)
                & (self.pttt[year][self.sig_key_channel] > 200)
            )
            if not self.use_bdt:
                cut_sig_pass &= (self.masstt[year][self.sig_key_channel] > mtt1) & (
                    self.masstt[year][self.sig_key_channel] < mtt2
                )

            # Apply exclusions: exclude regions defined by other signals
            # (e.g., when optimizing GGF, exclude the VBF-enriched region)
            if apply_exclusions:
                for excl_sig_key, excl_cuts in self.exclusion_cuts.items():
                    if excl_sig_key != self.sig_key:  # Don't exclude the signal we're optimizing
                        vbb, vtt = excl_cuts
                        exclusion_region_sig = (
                            (self.txbbs[year][self.sig_key_channel] > vbb)
                            & (self.txtts_multi[excl_sig_key][year][self.sig_key_channel] > vtt)
                            # & (self.ptbb[year][self.sig_key_channel] > 250) Should these be here?
                            # & (self.pttt[year][self.sig_key_channel] > 200)
                        )
                        cut_sig_pass = cut_sig_pass & (~exclusion_region_sig)

            sig_pass += np.sum(
                [
                    self.events_dict[year][sig_key].get_var("finalWeight")[cut_sig_pass]
                    for sig_key in self.sig_keys_sum
                ]
            )

            for key in self.channel.data_samples:
                cut_bg_pass_sb = (
                    (self.txbbs[year][key] > txbbcut)
                    & (self.txtts[year][key] > txttcut)
                    & (self.ptbb[year][key] > 250)
                    & (self.pttt[year][key] > 200)
                )
                if not self.use_bdt:
                    cut_bg_pass_sb &= (self.masstt[year][key] > mtt1) & (
                        self.masstt[year][key] < mtt2
                    )

                msb1 = (self.massbb[year][key] > SHAPE_VAR["range"][0]) & (
                    self.massbb[year][key] < mbb1
                )
                msb2 = (self.massbb[year][key] > mbb2) & (
                    self.massbb[year][key] < SHAPE_VAR["range"][1]
                )
                # Apply exclusions to background selections
                if apply_exclusions:
                    for excl_sig_key, excl_cuts in self.exclusion_cuts.items():
                        if excl_sig_key != self.sig_key:
                            vbb, vtt = excl_cuts
                            exclusion_region = (
                                (self.txbbs[year][key] > vbb)
                                & (self.txtts_multi[excl_sig_key][year][key] > vtt)
                                & (self.ptbb[year][key] > 250)
                                & (self.pttt[year][key] > 200)
                            )
                            cut_bg_pass_sb = cut_bg_pass_sb & (~exclusion_region)

                bg_pass_sb += np.sum(
                    self.events_dict[year][key].get_var("finalWeight")[cut_bg_pass_sb & msb1]
                )
                bg_pass_sb += np.sum(
                    self.events_dict[year][key].get_var("finalWeight")[cut_bg_pass_sb & msb2]
                )
                cut_bg_fail_sb = (
                    ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                    & (self.ptbb[year][key] > 250)
                    & (self.pttt[year][key] > 200)
                )
                if not self.use_bdt:
                    cut_bg_fail_sb &= (self.masstt[year][key] > mtt1) & (
                        self.masstt[year][key] < mtt2
                    )

                if apply_exclusions:
                    for excl_sig_key, excl_cuts in self.exclusion_cuts.items():
                        if excl_sig_key != self.sig_key:
                            vbb, vtt = excl_cuts
                            exclusion_region = (
                                (self.txbbs[year][key] > vbb)
                                & (self.txtts_multi[excl_sig_key][year][key] > vtt)
                                & (self.ptbb[year][key] > 250)
                                & (self.pttt[year][key] > 200)
                            )
                            cut_bg_fail_sb = cut_bg_fail_sb & (~exclusion_region)

                bg_fail_sb += np.sum(
                    self.events_dict[year][key].get_var("finalWeight")[cut_bg_fail_sb & msb1]
                )
                bg_fail_sb += np.sum(
                    self.events_dict[year][key].get_var("finalWeight")[cut_bg_fail_sb & msb2]
                )
                cut_bg_fail_res = (
                    ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                    & (self.massbb[year][key] > mbb1)
                    & (self.massbb[year][key] < mbb2)
                    & (self.ptbb[year][key] > 250)
                    & (self.pttt[year][key] > 200)
                )
                if not self.use_bdt:
                    cut_bg_fail_res &= (self.masstt[year][key] > mtt1) & (
                        self.masstt[year][key] < mtt2
                    )

                if apply_exclusions:
                    for excl_sig_key, excl_cuts in self.exclusion_cuts.items():
                        if excl_sig_key != self.sig_key:
                            vbb, vtt = excl_cuts
                            exclusion_region = (
                                (self.txbbs[year][key] > vbb)
                                & (self.txtts_multi[excl_sig_key][year][key] > vtt)
                                & (self.ptbb[year][key] > 250)
                                & (self.pttt[year][key] > 200)
                            )
                            cut_bg_fail_res = cut_bg_fail_res & (~exclusion_region)

                bg_fail_res += np.sum(
                    self.events_dict[year][key].get_var("finalWeight")[cut_bg_fail_res]
                )

        del cut_sig_pass, cut_bg_pass_sb, cut_bg_fail_sb, cut_bg_fail_res, msb1, msb2

        # signal, B, C, D, TF = C/D
        tf = bg_fail_res / bg_fail_sb if bg_fail_sb > 0 else 0
        return sig_pass, bg_pass_sb, bg_fail_res, bg_fail_sb, tf

    def grid_search_opt(
        self,
        years,
        gridsize,
        gridlims,
        B_min_vals,
        foms,
        use_thresholds=False,
        apply_exclusions=True,
    ) -> Optimum:
        """
        Grid search optimization for signal/background discrimination.

        Args:
            years: List of years to analyze
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
        mtt1, mtt2 = self.channel.tt_mass_cut[1]

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
                self.compute_and_plot_rocs(years)
                print("ROC analyzer computed and plotted.")

            # Check discriminants exist
            if self.bb_disc_name not in self.rocAnalyzer.discriminants:
                raise ValueError(
                    f"BB discriminant '{self.bb_disc_name}' not found in ROC analyzer. Available: {list(self.rocAnalyzer.discriminants.keys())}"
                )
            if self.tt_disc_name not in self.rocAnalyzer.discriminants:
                raise ValueError(
                    f"TT discriminant '{self.tt_disc_name}' not found in ROC analyzer. Available: {list(self.rocAnalyzer.discriminants.keys())}"
                )

            # Create signal efficiency grid with separate limits for each axis
            bbcutSigEff = np.linspace(*gridlims_x, gridsize)
            ttcutSigEff = np.linspace(*gridlims_y, gridsize)
            BBcutSigEff, TTcutSigEff = np.meshgrid(bbcutSigEff, ttcutSigEff)

            # Get discriminant objects for threshold lookup
            bb_discriminant = self.rocAnalyzer.discriminants[self.bb_disc_name]
            tt_discriminant = self.rocAnalyzer.discriminants[self.tt_disc_name]

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
                years=years,
                txbbcut=bbcut,
                txttcut=ttcut,
                mbb1=mbb1,
                mbb2=mbb2,
                mtt1=mtt1,
                mtt2=mtt2,
                apply_exclusions=apply_exclusions,
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

        tot_sig_weight = np.sum(
            np.concatenate(
                [
                    self.events_dict[year][self.sig_key_channel].get_var("finalWeight")
                    for year in years
                ]
            )
        )

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
                    "tt_disc_name": self.tt_disc_name,
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

                results[fom.name][f"Bmin={B_min}"] = Optimum(**optimum_args)

        return results

    def _adaptive_refine_limits(
        self, optimum: Optimum, use_thresholds: bool, top_fraction: float, padding: float
    ) -> dict:
        """Compute refined grid limits for the next stage from the union of padded regions around best points.

        This method:
        1. Identifies the top N% best grid points (lowest FOM values)
        2. Creates a padded region around each of these points
        3. Computes the union of all these regions
        4. Returns separate (min, max) limits for X and Y axes

        Args:
            optimum: Optimum object containing FOM map and grid coordinates
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
        years,
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
        apply_exclusions: bool = True,
    ) -> Optimum:
        """Multi-stage adaptive refinement around best regions using existing grid_search_opt.

        Strategy:
        - Intermediate stages (1 to N-1): Run optimization with SINGLE B_min value for speed
        - Final stage (N): Run optimization with ALL B_min values for complete results

        Args:
            years: List of years to analyze
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
                years=years,
                gridsize=gridsize,
                gridlims=gridlims,
                B_min_vals=current_bmin_vals,
                foms=foms,
                use_thresholds=use_thresholds,
                apply_exclusions=apply_exclusions,
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
                        if isinstance(opt, Optimum) and opt.fom_map is not None:
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
        years,
        use_thresholds=False,
        plot=True,
        b_min_vals=None,
        adaptive: bool = False,
        adaptive_stages: int = 2,
        adaptive_top_frac: float = 0.1,
        adaptive_padding: float = 1.0,
        adaptive_refine_gridsize: int | None = None,
        adaptive_bmin: int = None,
        apply_exclusions: bool = True,
    ):

        if b_min_vals is None:
            b_min_vals = [1, 3, 5] if self.test_mode else DEFAULT_BMIN_VALUES

        if self.test_mode:
            gridlims = (0.3, 1) if use_thresholds else (0.2, 0.9)
            gridsize = 20
        else:
            gridlims = (0.7, 1) if use_thresholds else (0.25, 0.75)
            gridsize = 50 if not adaptive else 20

        foms = [FOMS["2sqrtB_S_var"]]

        if adaptive:
            results = self.adaptive_grid_search_opt(
                years=years,
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
                apply_exclusions=apply_exclusions,
            )
        else:
            results = self.grid_search_opt(
                years,
                gridsize,
                gridlims=gridlims,
                B_min_vals=b_min_vals,
                foms=foms,
                use_thresholds=use_thresholds,
                apply_exclusions=apply_exclusions,
            )

        successful_b_min_vals = []
        saved_files = []
        for fom in foms:
            bmin_dfs = []
            for B_min in b_min_vals:
                optimum_result = results.get(fom.name, {}).get(f"Bmin={B_min}")
                if optimum_result:
                    successful_b_min_vals.append(B_min)
                    bmin_df = self.as_df(optimum_result, years, label=f"Bmin={B_min}")
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
                / f"{'_'.join(years)}_opt_results_{fom.name}_{'thresh' if use_thresholds else 'sigeff'}.csv"
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
                    years=years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(years)}_thresholds",
                    show=False,
                )
            else:
                clip_value = 100 if "ggf" in self.sig_key.lower() else None
                plot_optimization_sig_eff(
                    results=results,
                    years=years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(years)}_sigeff",
                    show=False,
                    use_log_scale=False,
                    clip_value=clip_value,
                )
                plot_optimization_sig_eff(
                    results=results,
                    years=years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(years)}_sigeff_log",
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
            years, gridsize, gridlims=gridlims, B_min_vals=b_min_vals, foms=foms
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
        Converts an Optimum result to a pandas DataFrame row with derived quantities.

        Args:
            optimum: An Optimum dataclass instance.
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
    years,
    sig_key,
    channel_key,
    test_mode,
    use_bdt,
    modelname,
    main_plot_dir,
    tt_pres=False,
    actions=None,
    at_inference=False,
    adaptive=False,
    do_vbf=False,
    vbf_modelname="",
    vbf_signal_key="vbfbbtt",
    vbf_bmin_for_veto=10,
    bdt_dir=None,
    use_sm_signals=True,
):

    print(f"Processing signal: {sig_key} channel: {channel_key}. Test mode: {test_mode}.")

    # Configure signals for multi-signal analyser
    signals = [SignalConfig(key=sig_key, modelname=modelname, optimize=True)]

    if do_vbf and use_bdt:
        # Add VBF signal - we'll optimize it first to define the VBF-enriched region
        signals.append(
            SignalConfig(
                key=vbf_signal_key,
                modelname=vbf_modelname or modelname,
                optimize=True,
                bmin_for_exclusion=vbf_bmin_for_veto,
            )
        )
        print(
            f"Multi-signal mode: Will optimize {vbf_signal_key} first, then {sig_key} excluding {vbf_signal_key} region"
        )

    # Create analyser with all signals (data and BDT predictions loaded once)
    analyser = Analyser(
        years,
        signals,
        channel_key,
        test_mode,
        use_bdt,
        main_plot_dir,
        at_inference,
        bdt_dir=bdt_dir,
        tt_pres=tt_pres,
        use_sm_signals=use_sm_signals,
    )

    analyser.load_data()  # Loads data once and BDT predictions for all signals

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_and_plot_rocs(years)
    if "plot_mass" in actions:
        analyser.plot_mass(years)
    if "sensitivity" in actions:
        # If VBF is configured, optimize VBF first to define its region
        if do_vbf and use_bdt:
            print("\n" + "=" * 80)
            print(f"STEP 1: Optimizing {vbf_signal_key} to define VBF-enriched region")
            print("=" * 80)

            # Create temporary VBF-only analyser for optimization
            vbf_only_analyser = Analyser(
                years,
                SignalConfig(
                    key=vbf_signal_key, modelname=vbf_modelname or modelname, optimize=True
                ),
                channel_key,
                test_mode,
                use_bdt,
                main_plot_dir,
                at_inference,
                bdt_dir=bdt_dir,
                use_sm_signals=use_sm_signals,
            )
            # Share the same event data (no reloading!)
            vbf_only_analyser.events_dict = analyser.events_dict
            vbf_only_analyser.prepare_sensitivity(years)

            vbf_results = vbf_only_analyser.perform_optimization(
                years,
                use_thresholds=False,
                adaptive=adaptive,
                adaptive_stages=ADAPTIVE_STAGES,
                adaptive_top_frac=ADAPTIVE_TOP_FRAC,
                adaptive_padding=ADAPTIVE_PADDING,
                adaptive_refine_gridsize=ADAPTIVE_REFINE_GRIDSIZE,
                adaptive_bmin=ADAPTIVE_BMIN,
                plot=True,
                apply_exclusions=False,  # Don't apply exclusions when defining the VBF region itself
            )

            # Extract VBF cuts at the specified B_min for exclusion
            try:
                vbf_sig_config = analyser.signals[vbf_signal_key]
                bmin_key = f"Bmin={vbf_sig_config.bmin_for_exclusion}"
                vbf_opt = vbf_results["2sqrtB_S_var"][bmin_key]
                vbf_exclusion_cuts = vbf_opt.cuts
                print(
                    f"\nVBF region cuts ({bmin_key}): TxBB > {vbf_exclusion_cuts[0]:.3f}, TxTT > {vbf_exclusion_cuts[1]:.3f}"
                )
                print(f"This region will be EXCLUDED from {sig_key} optimization")

                # Store exclusion cuts in main analyser
                analyser.exclusion_cuts[vbf_signal_key] = vbf_exclusion_cuts
            except Exception as e:
                print(f"Warning: Could not extract VBF exclusion cuts for {bmin_key}: {e}")

            del vbf_only_analyser
            gc.collect()

            print("\n" + "=" * 80)
            print(f"STEP 2: Optimizing {sig_key} excluding VBF region")
            print("=" * 80)

        # Run main signal optimization (with exclusions if configured)
        analyser.prepare_sensitivity(years)
        analyser.perform_optimization(
            years,
            use_thresholds=False,
            adaptive=adaptive,
            **adaptive_kwargs,
        )
    if "fom_study" in actions:
        analyser.prepare_sensitivity(years)
        analyser.study_foms(years)
    if "minimization_study" in actions:
        analyser.prepare_sensitivity(years)
        analyser.study_minimization_method(years)

    del analyser
    gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sensitivity Study Script")

    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="List of years to include in the analysis",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["ggfbbtt"],
        help="List of signals to run (default: ggf)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["hh", "hm", "he"],
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
        help="Use tt preselection",
    )
    parser.add_argument(
        "--use-bdt", action="store_true", default=False, help="Use BDT for sensitivity study"
    )
    parser.add_argument(
        "--modelname",
        default="19oct25_ak4away_ggfbbtt",
        help="Name of the BDT model to use for sensitivity study",
    )
    parser.add_argument(
        "--at-inference",
        action="store_true",
        default=False,
        help="Compute BDT predictions at inference time",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["compute_rocs", "plot_mass", "sensitivity", "fom_study", "minimization_study"],
        required=True,
        help="Actions to perform. Choose one or more.",
    )
    parser.add_argument(
        "--use-thresholds",
        action="store_true",
        default=False,
        help="Use thresholds for optimization",
    )
    parser.add_argument(
        "--use-sm-signals",
        action="store_true",
        default=True,
        help="Use total SM signals for sensitivity study",
    )
    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="/home/users/lumori/bbtautau/",
        type=str,
    )
    parser.add_argument(
        "--bdt-dir",
        help="directory where you save your bdt model for inference",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Use adaptive multi-stage grid refinement during sensitivity optimization",
    )
    # Adaptive parameters are defined as module-level defaults; no extra CLI flags
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
        "--vbf-bmin-for-veto",
        type=int,
        default=10,
        help="Bmin value from which to take the VBF veto cuts",
    )

    args = parser.parse_args()

    for sig_key in args.signals:
        for channel_key in args.channels:
            analyse_channel(
                years=args.years,
                sig_key=sig_key,
                channel_key=channel_key,
                test_mode=args.test_mode,
                use_bdt=args.use_bdt,
                modelname=args.modelname,
                main_plot_dir=args.plot_dir,
                actions=args.actions,
                at_inference=args.at_inference,
                adaptive=args.adaptive,
                do_vbf=args.do_vbf,
                vbf_modelname=args.vbf_modelname,
                vbf_signal_key=args.vbf_signal_key,
                vbf_bmin_for_veto=args.vbf_bmin_for_veto,
                bdt_dir=args.bdt_dir,
                tt_pres=args.tt_pres,
                use_sm_signals=args.use_sm_signals,
            )
