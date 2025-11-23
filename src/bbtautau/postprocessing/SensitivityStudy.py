from __future__ import annotations

import argparse
import gc
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import hh_vars
from boostedhh.utils import PAD_VAL
from joblib import Parallel, delayed

from bbtautau.postprocessing import utils
from bbtautau.postprocessing.plotting import (
    plot_optimization_sig_eff,
    plot_optimization_thresholds,
)
from bbtautau.postprocessing.postprocessing import (
    apply_triggers,
    base_filter,
    bb_filters,
    bbtautau_assignment,
    check_bdt_prediction_shapes,
    compute_bdt_preds,
    delete_columns,
    derive_lepton_variables,
    derive_variables,
    get_columns,
    leptons_assignment,
    load_bdt_preds,
    load_samples,
    tt_filters,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.userConfig import DATA_PATHS, MODEL_DIR, SHAPE_VAR, Enhanced_ABCD_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# Global variables
BDT_EVAL_DIR = Path("/home/users/haoyang/bbtautau_sen/BDT_predictions")
TODAY = date.today()  # to name output folder


class FOM:
    def __init__(self, fom_func, label, name):
        self.fom_func = fom_func
        self.label = label
        self.name = name


@dataclass
class Optimum:
    # conditions of the optimization
    dataMinusSimABCD: bool
    showNonDataDrivenPortion: bool
    
    # old stuff. Those entries are now in evaluations_at_optimum
    # limit: float
    # signal_yield: float
    # bkg_data_yield: float
    # hmass_data_fail: float
    # sideband_data_fail: float
    # data_transfer_factor: float

    # bkg_qcd_yield: float
    # hmass_qcd_fail: float
    # sideband_qcd_fail: float
    # qcd_transfer_factor: float
    # non_qcd_bg_in_A: float
    
    # evaluations at the optimum cuts
    evaluations_at_optimum: dict[str, float]
    
    # optimal cuts
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

    def __post_init__(self):
        """Automatically unpack evaluations_at_optimum dict into attributes."""
        if isinstance(self.evaluations_at_optimum, dict):
            for k, v in self.evaluations_at_optimum.items():
                setattr(self, k, v)
        


def fom_2sqrtB_S(b_qcd, s, _tf, non_qcd_bg_in_A=0, b_data=None):
    return np.where(s > 0, 2 * np.sqrt(b_qcd * _tf + non_qcd_bg_in_A) / s, -PAD_VAL)

# FoM function used in both standard and enhanced ABCD
# Region definitions:
#    High  |--------------------------------------|
#     |    |  A: pass & res  | B: pass & sideband |
#   score  |--------------------------------------|
#     |    |  C: fail & res  | D: fail & sideband |
#    Low   |---resonant-mass-|----mass-sideband---|
# Arguments:
# non_qcd_bg_in_A = 0 means standard ABCD
#    b_qcd = data in region B
#    tf = derived TF assuming all data in B, C, D are QCD/DY
# non_qcd_bg_in_A != 0 is used for enhanced ABCD
#    b_qcd = data-ttbar in region B
#    tf = TF derived from (data-ttbar) in B, C, D
def fom_2sqrtB_S_var(b_qcd, s, _tf, non_qcd_bg_in_A=0):
    # even tho the var name is qcd, it is actually data-ttbar
    # TODO: better var names
    qcd_in_A = b_qcd * _tf
    bg_in_A = qcd_in_A + non_qcd_bg_in_A 
    return np.where(
        (b_qcd > 0) & (s > 0), 2 * np.sqrt(bg_in_A + (qcd_in_A / np.sqrt(b_qcd)) ** 2) / s, -PAD_VAL
    )


def fom_punzi(b, s, _tf, a=3, non_qcd_bg_in_A=0):
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
        sig_key,
        channel_key,
        test_mode,
        use_bdt,
        modelname,
        main_plot_dir,
        at_inference=False,
        llsl_weight=1,
        bb_disc='bbFatJetParTXbbvsQCDTop',
        dataMinusSimABCD=False,
        showNonDataDrivenPortion=True,
        sensitivity_dir=None
    ):
        self.channel = CHANNELS[channel_key]
        self.years = years
        self.test_mode = test_mode
        test_dir = "test" if test_mode else "full_presel"
        tt_tagger_name = modelname if use_bdt else "ParT"

        self.sig_key = sig_key
        self.sig_key_channel = sig_key + channel_key
        self.taukey = CHANNELS[channel_key].tagger_label

        self.events_dict = {year: {} for year in years}

        self.use_bdt = use_bdt
        self.modelname = modelname
        self.at_inference = at_inference
        self.llsl_weight = llsl_weight
        self.bb_disc = bb_disc
        self.dataMinusSimABCD = dataMinusSimABCD
        self.showNonDataDrivenPortion = showNonDataDrivenPortion
        print("Using data minus simulated non-QCD backgrounds for ABCD estimation:", self.dataMinusSimABCD)

        self.model_dir = MODEL_DIR
        self.sensitivity_dir = sensitivity_dir
        # TODO: make it an init arg in the future
        self.bmin = 5 

        self.plot_dir = (
            Path(main_plot_dir)
            / f"plots/SensitivityStudy/{TODAY}/{test_dir}/{sig_key}/{tt_tagger_name}/{channel_key}"
        )
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.bb_disc_name = bb_disc
        self.tt_disc_name = (
            f"BDTScore{self.taukey}vsAll" if use_bdt else f"ttFatJetParTX{self.taukey}vsQCDTop"
        )

    def load_data(self, tt_pres=False):

        for year in self.years:

            filters_dict = base_filter(self.test_mode)

            # Prefilters already applied in skimmer
            # filters_dict = bb_filters(filters_dict, num_fatjets=3, bb_cut=0.3)

            if tt_pres:
                filters_dict = tt_filters(
                    channel=self.channel, in_filters=filters_dict, num_fatjets=3, tt_cut=0.3
                )

            columns = get_columns(year, triggers_in_channel=self.channel)

            self.events_dict[year] = load_samples(
                year=year,
                paths=DATA_PATHS[year],
                signals=[self.sig_key],
                channels=[self.channel],
                samples = Enhanced_ABCD_SAMPLES if self.showNonDataDrivenPortion else None,
                filters_dict=filters_dict,
                load_columns=columns,
                restrict_data_to_channel=not self.showNonDataDrivenPortion,
                loaded_samples=True,
                multithread=True,
            )

            apply_triggers(self.events_dict[year], year, self.channel)
            delete_columns(self.events_dict[year], year, channels=[self.channel])

            # Keep this for legacy issue, with old ntuples some branches are misnamed
            # derive_variables(self.events_dict[year], CHANNELS["hm"])

            derive_variables(self.events_dict[year], self.channel)
            bbtautau_assignment(self.events_dict[year], agnostic=True)
            leptons_assignment(self.events_dict[year], dR_cut=1.5)
            derive_lepton_variables(self.events_dict[year])

        if self.use_bdt:
            if self.at_inference:
                # evaluate bdt at inference time
                # Store the evals by default
                start_time = time.time()
                compute_bdt_preds(
                    events_dict=self.events_dict,
                    model_dir=self.model_dir,
                    modelname=self.modelname,
                    channel=self.channel,
                    test_mode=self.test_mode,
                    save_dir=BDT_EVAL_DIR,
                )
                print(
                    f"BDT predictions computed at inference time in {time.time() - start_time} seconds"
                )

            else:
                shapes_ok = check_bdt_prediction_shapes(
                    self.events_dict,
                    self.modelname,
                    self.channel,
                    BDT_EVAL_DIR,
                    self.test_mode,
                )
                if shapes_ok:
                    load_bdt_preds(
                        self.events_dict,
                        modelname=self.modelname,
                        channel=self.channel,
                        bdt_preds_dir=BDT_EVAL_DIR,
                        test_mode=self.test_mode,
                        all_outs=True,
                    )
                else:
                    print(
                        "BDT prediction don't exist or shapes do not match with data. You might have changed the preselection. I will recompute the predictions."
                    )
                    compute_bdt_preds(
                        events_dict=self.events_dict,
                        model_dir=self.model_dir,
                        modelname=self.modelname,
                        channel=self.channel,
                        test_mode=self.test_mode,
                        save_dir=BDT_EVAL_DIR,
                        llsl_weight=self.llsl_weight,
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
            discs_tt += [f"BDTScore{self.taukey}vsQCD", f"BDTScore{self.taukey}vsAll"]

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

        self.txbbs = {year: {} for year in years}
        self.txtts = {year: {} for year in years}
        self.masstt = {year: {} for year in years}
        self.massbb = {year: {} for year in years}
        self.ptbb = {year: {} for year in years}
        self.pttt = {year: {} for year in years}

        # precompute to speedup
        for year in years:
            # use all keys to enable enhanced ABCD
            for key in self.events_dict[year].keys():
                self.txbbs[year][key] = self.events_dict[year][key].get_var(self.bb_disc_name)
                self.txtts[year][key] = self.events_dict[year][key].get_var(self.tt_disc_name)
                self.masstt[year][key] = self.events_dict[year][key].get_var(mttk)
                self.massbb[year][key] = self.events_dict[year][key].get_var(mbbk)
                self.ptbb[year][key] = self.events_dict[year][key].get_var("bbFatJetPt")
                self.pttt[year][key] = self.events_dict[year][key].get_var("ttFatJetPt")

    def compute_sig_bkg_abcd(self, years, txbbcut, txttcut, mbb1, mbb2, mtt1, mtt2):
        # pass/fail from taggers
        sig_pass = 0  # resonant region pass, signal
        bg_pass_sb = 0  # sideband region pass
        bg_fail_res = 0  # resonant region fail
        bg_fail_sb = 0  # sideband region fail

        # variables to use if self.showNonDataDrivenPortion is True
        non_qcd_bgs = ["ttbarhad", "ttbarsl", "ttbarll"]
        qcd_pass_sb = 0  # sideband region pass, data - simulated_non_QCD_bg
        qcd_fail_res = 0  # resonant region fail, data - simulated_non_QCD_bg
        qcd_fail_sb = -1  # sideband region fail, data - simulated_non_QCD_bg
        non_qcd_bg_pass_res = 0 # resonant region pass, simulated backgrounds except QCD
        non_qcd_bg_pass_res_dict = defaultdict(float) # a dict storing every class of passed and res non_qcd_bg 

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

            sig_pass += np.sum(
                self.events_dict[year][self.sig_key_channel].get_var("finalWeight")[cut_sig_pass]
            )

            for key in self.channel.data_samples:
                # Region B
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
                sum_pass_msb1 =  np.sum(
                    self.events_dict[year][key].events["finalWeight"][cut_bg_pass_sb & msb1]
                )
                sum_pass_msb2 =  np.sum(
                    self.events_dict[year][key].events["finalWeight"][cut_bg_pass_sb & msb2]
                )
                bg_pass_sb += sum_pass_msb1
                bg_pass_sb += sum_pass_msb2
                if self.showNonDataDrivenPortion:
                    qcd_pass_sb += sum_pass_msb1
                    qcd_pass_sb += sum_pass_msb2

                # Region D
                cut_bg_fail_sb = (
                    ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                    & (self.ptbb[year][key] > 250)
                    & (self.pttt[year][key] > 200)
                )
                if not self.use_bdt:
                    cut_bg_fail_sb &= (self.masstt[year][key] > mtt1) & (
                        self.masstt[year][key] < mtt2
                    )
                sum_fail_msb1 = np.sum(
                    self.events_dict[year][key].events["finalWeight"][cut_bg_fail_sb & msb1]
                )
                sum_fail_msb2 = np.sum(
                    self.events_dict[year][key].events["finalWeight"][cut_bg_fail_sb & msb2]
                )
                bg_fail_sb += sum_fail_msb1
                bg_fail_sb += sum_fail_msb2
                if self.showNonDataDrivenPortion:
                    qcd_fail_sb += sum_fail_msb1
                    qcd_fail_sb += sum_fail_msb2

                # Region C
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
                sum_fail_res = np.sum(
                    self.events_dict[year][key].events["finalWeight"][cut_bg_fail_res]
                )
                bg_fail_res += sum_fail_res
                if self.showNonDataDrivenPortion:
                    qcd_fail_res += sum_fail_res

            if self.showNonDataDrivenPortion:
                for key in non_qcd_bgs:
                    # Region A
                    cut_sig_pass = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                        & (self.pttt[year][key] > 200)
                    )
                    if not self.use_bdt:
                        cut_sig_pass &= (self.masstt[year][key] > mtt1) & (
                            self.masstt[year][key] < mtt2
                        )
                    # estimate the non-qcd contribution to region A
                    non_qc_bg_pass_res_e = np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_sig_pass]
                    ) 
                    non_qcd_bg_pass_res_dict[key] += non_qc_bg_pass_res_e
                    non_qcd_bg_pass_res += non_qc_bg_pass_res_e
                
                    # Region B
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
                    sum_pass_msb1 =  np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_pass_sb & msb1]
                    )
                    sum_pass_msb2 =  np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_pass_sb & msb2]
                    )
                    ## subtract non-QCD simulated background for more precise ABCD estimation of QCD
                    qcd_pass_sb -= sum_pass_msb1
                    qcd_pass_sb -= sum_pass_msb2

                    # Region D
                    cut_bg_fail_sb = (
                        ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                        & (self.ptbb[year][key] > 250)
                        & (self.pttt[year][key] > 200)
                    )
                    if not self.use_bdt:
                        cut_bg_fail_sb &= (self.masstt[year][key] > mtt1) & (
                            self.masstt[year][key] < mtt2
                        )
                    sum_fail_msb1 = np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail_sb & msb1]
                    )
                    sum_fail_msb2 = np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail_sb & msb2]
                    )
                    ## subtract non-QCD simulated background for more precise ABCD estimation of QCD
                    qcd_fail_sb -= sum_fail_msb1
                    qcd_fail_sb -= sum_fail_msb2

                    # Region C
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
                    sum_fail_res = np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail_res]
                    )
                    ## subtract non-QCD simulated background for more precise ABCD estimation of QCD
                    qcd_fail_res -= sum_fail_res

        del cut_sig_pass, cut_bg_pass_sb, cut_bg_fail_sb, cut_bg_fail_res, msb1, msb2

        # bg equal qcd when not considering ttbar
        print("Am I showing the non QCD portion?", self.showNonDataDrivenPortion)
        print(non_qcd_bg_pass_res)

        # signal, B(data), C(...), D(...), TF = C(...)/D(...),
        #   B(data-non_qcd_sim), C(...), D(...), TF = C(...)/D(...), 
        #       sim_non_QCD_bg_in_A
        return {
            "sig_pass": sig_pass,
            "data_pass_sideband": bg_pass_sb,
            "data_fail_resonant": bg_fail_res,
            "data_fail_sideband": bg_fail_sb,
            "TF_data": bg_fail_res / bg_fail_sb,
            "dataMinusTT_pass_sideband": qcd_pass_sb,
            "dataMinusTT_fail_resonant": qcd_fail_res,
            "dataMinusTT_fail_sideband": qcd_fail_sb,
            "TF_dataMinusTT": qcd_fail_res / qcd_fail_sb,
            "TT_pass_resonant": non_qcd_bg_pass_res,
            **non_qcd_bg_pass_res_dict,
        }

    def grid_search_opt(
        self,
        years,
        gridsize,
        gridlims,
        B_min_vals,
        foms,
        # arguments for sensitivity evaluation
        txbbcut=None,
        txttcut=None,
        use_thresholds=False,
    ) -> Optimum:
        """
        Grid search optimization for signal/background discrimination.

        Args:
            years: List of years to analyze
            gridsize: Size of the grid (gridsize x gridsize points)
            gridlims: Tuple of (min, max) for grid limits
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
        
        for_evaluation = (txbbcut is not None) and (txttcut is not None)
        
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

            # Create signal efficiency grid
            bbcutSigEff = np.linspace(*gridlims, gridsize)
            ttcutSigEff = np.linspace(*gridlims, gridsize)
            BBcutSigEff, TTcutSigEff = np.meshgrid(bbcutSigEff, ttcutSigEff)

            # Get discriminant objects for threshold lookup
            bb_discriminant = self.rocAnalyzer.discriminants[self.bb_disc_name]
            tt_discriminant = self.rocAnalyzer.discriminants[self.tt_disc_name]

            # VECTORIZED threshold computation
            bbcut_thresholds = bb_discriminant.get_cut_from_sig_eff(bbcutSigEff)
            ttcut_thresholds = tt_discriminant.get_cut_from_sig_eff(ttcutSigEff)

            # Create threshold meshgrids for actual cuts
            BBcut, TTcut = np.meshgrid(bbcut_thresholds, ttcut_thresholds)
        
        elif not for_evaluation:
            # for optimization, pass a full grid
            bbcut = np.linspace(*gridlims, gridsize)
            ttcut = np.linspace(*gridlims, gridsize)
            BBcut, TTcut = np.meshgrid(bbcut, ttcut)

            # For consistency, set efficiency grids to None when not using signal efficiency
            BBcutSigEff = None
            TTcutSigEff = None
        else:
            # for evaluation, pass a meshgrid of size 1
            bbcut = np.linspace(txbbcut, txbbcut, 1)    
            ttcut = np.linspace(txttcut, txttcut, 1)    
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
            )

        # results is a list of (sig, bkg, tf) tuples
        print(f"Running grid search on {len(bbcut_flat)} points...")
        results = Parallel(n_jobs=-10, verbose=1)(
            delayed(sig_bg)(_b, _t) for _b, _t in zip(bbcut_flat, ttcut_flat)
        )

        # evals is a dict storying lists of sig, bkg, tf evaluated at each (bb, tt) cut
        evals = defaultdict(list)
        for r in results:
            for k, v in r.items():
                evals[k].append(v)
                
        # reshape each array back to 2D
        evals = {
            k: np.array(v).reshape(BBcut.shape)
            for k, v in evals.items()
        }

        tot_sig_weight = np.sum(
            np.concatenate(
                [
                    self.events_dict[year][self.sig_key_channel].get_var("finalWeight")
                    for year in years
                ]
            )
        )
        evals["sig_eff"] = evals["sig_pass"] / tot_sig_weight

        # compute the number of background events in A by ABCD method(s)
        evals["bgs_scaled_ABCD"] = evals["data_pass_sideband"] * evals["TF_data"]
        if self.dataMinusSimABCD:
            evals["bgs_scaled_EnhancedABCD"] = evals["dataMinusTT_pass_sideband"] * evals["TF_dataMinusTT"] + evals["TT_pass_resonant"]
            
        results = {}
        for fom in foms:
            results[fom.name] = {}
            for B_min in B_min_vals:

                results[fom.name][f"Bmin={B_min}"] = {}

                sel_B_min = evals["data_pass_sideband"] >= B_min
                if self.dataMinusSimABCD:
                    limits = fom.fom_func(evals["dataMinusTT_pass_sideband"],
                                          evals["sig_pass"],
                                          evals["TF_dataMinusTT"],
                                          non_qcd_bg_in_A = evals["TT_pass_resonant"],
                                        )
                    # limits = fom.fom_func(qcd_sb, sigs, qcd_tfs, non_qcd_bg_in_A=non_qcd_bg_pass_res, b_data=bgs_sb)
                else:
                    limits = fom.fom_func(evals["data_pass_sideband"],
                                          evals["sig_pass"],
                                          evals["TF_data"],
                                          non_qcd_bg_in_A = 0,
                                        )
                    # limits = fom.fom_func(bgs_sb, sigs, data_tfs, non_qcd_bg_in_A=0, b_data=bgs_sb)
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
                evals_opt = {k: v[idx_opt] for k, v in evals.items()}
                evals_opt["limit"] = limit_opt

                # Add signal efficiency specific fields if using signal efficiency coordinates
                if not use_thresholds:
                    bbcut_sig_eff_opt = BBcutSigEff[idx_opt]
                    ttcut_sig_eff_opt = TTcutSigEff[idx_opt]

                    evals_opt.update(
                        {
                            "sig_eff_cuts": (bbcut_sig_eff_opt, ttcut_sig_eff_opt),
                            "BBcut_sig_eff": BBcutSigEff,
                            "TTcut_sig_eff": TTcutSigEff,
                            "fom_map": limits,
                        }
                    )

                results[fom.name][f"Bmin={B_min}"] = Optimum(
                    dataMinusSimABCD=self.dataMinusSimABCD,
                    showNonDataDrivenPortion=self.showNonDataDrivenPortion,
                    evaluations_at_optimum=evals_opt,
                    cuts=(bbcut_opt, ttcut_opt),
                    BBcut=BBcut,
                    TTcut=TTcut,
                    sig_map=evals["sig_pass"],
                    bg_map=evals["bgs_scaled_ABCD"] if not self.dataMinusSimABCD else evals["bgs_scaled_EnhancedABCD"] ,
                    sel_B_min=sel_B_min,
                    fom=fom,
                )

        return results

    def perform_optimization(self, years, use_thresholds=False, plot=True, evaluation=False, b_min_vals=None):
        foms = [FOMS["2sqrtB_S_var"]]

        if evaluation:
            b_min_vals = [0]
            gridlims = None
            gridsize = 1 # single point evaluation 
            # extract cuts from self.channel
            txbbcut = self.channel.txbb_cut
            txttcut = self.channel.txtt_BDT_cut if self.use_bdt else self.channel.txtt_cut
            # Make one-point grid
            results = self.grid_search_opt(
                years,
                gridsize,
                gridlims=gridlims,
                B_min_vals=b_min_vals,
                foms=foms,
                use_thresholds=True,
                txbbcut=txbbcut,
                txttcut=txttcut,
            )
        else:
            # default fom optimization parameters
            if b_min_vals is None:
                b_min_vals = [1, 3, 5] if self.test_mode else np.arange(1, 17, 2)

            if self.test_mode:
                gridlims = (0.3, 1) if use_thresholds else (0.2, 0.9)
            else:
                gridlims = (0.7, 1) if use_thresholds else (0.25, 0.75)

            gridsize = 20 if self.test_mode else 50

            results = self.grid_search_opt(
                years,
                gridsize,
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
            return

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
                plot_optimization_sig_eff(
                    results=results,
                    years=years,
                    b_min_vals=successful_b_min_vals,
                    foms=foms,
                    channel=self.channel,
                    save_path=self.plot_dir / f"{'_'.join(years)}_sigeff",
                    show=False,
                    use_log_scale=False,
                    clip_value=100,
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
                )

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
        limits["use_dataMinusSimABCD"] = optimum.dataMinusSimABCD
        limits["show_NonDataDrivenPortion"] = optimum.showNonDataDrivenPortion   
        limits["Cut_Xbb"] = optimum.cuts[0]
        limits["Cut_Xtt"] = optimum.cuts[1]
        # limits["Sig_Yield"] = optimum.signal_yield
        # limits["Sideband Data Pass"] = optimum.bkg_data_yield
        # limits["Higgs Mass Data Fail"] = optimum.hmass_data_fail
        # limits["Sideband Data Fail"] = optimum.sideband_data_fail
        # limits["Standard ABCD BG_Yield_scaled"] = optimum.bkg_data_yield * optimum.data_transfer_factor
        # limits["Data TF"] = optimum.data_transfer_factor
        # limits["Sideband QCD Pass"] = optimum.bkg_qcd_yield
        # limits["Higgs Mass QCD Fail"] = optimum.hmass_qcd_fail
        # limits["Sideband QCD Fail"] = optimum.sideband_qcd_fail
        # limits["Enhanced ABCD BG_Yield_scaled"] = optimum.bkg_qcd_yield * optimum.qcd_transfer_factor + optimum.non_qcd_bg_in_A
        # limits["QCD TF"] = optimum.qcd_transfer_factor
        # limits["Non QCD bg in A"] = optimum.non_qcd_bg_in_A
        # limits["FOM"] = optimum.fom.label

        # limits["Limit"] = optimum.limit
        for k, v in optimum.evaluations_at_optimum.items():
            limits[k] = v
            
        limits["Limit_scaled_22_24"] = optimum.limit / np.sqrt(
            (124000 + hh_vars.LUMI["2022-2023"]) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        limits["Limit_scaled_Run3"] = optimum.limit / np.sqrt(
            (360000) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        return pd.DataFrame([limits])

    def update_cuts_from_csv_file(self, sensitivity_dir):
        # read the first available FOM CSV file
            csv_dir = Path(sensitivity_dir).joinpath(
                f"full_presel/{self.modelname if self.use_bdt else 'ParT'}/{self.channel.key}"
            )

            # Look for any FOM-specific CSV files
            csv_files = list(csv_dir.glob("*_opt_results_*.csv"))

            if len(csv_files) == 0:
                raise ValueError(f"No sensitivity CSV files found in {csv_dir}")

            # Take the first CSV file found and extract FOM name
            csv_file = sorted(csv_files)[0]  # Sort for reproducible behavior
            print(f"Reading CSV: {csv_file}")

            # Extract FOM name from filename like "2022_2022EE_opt_results_2sqrtB_S_var.csv"
            if "_opt_results_" in csv_file.name:
                fom_name = csv_file.name.split("_opt_results_")[1].replace(".csv", "")
            else:
                fom_name = "unknown"

            # Read as simple CSV (no multi-level headers)
            opt_results = pd.read_csv(csv_file, index_col=0)
            print(f"Using FOM: {fom_name}")
            print(f"Available B_min values: {opt_results.columns.tolist()}")

            # Check if the target Bmin column exists
            target_col = f"Bmin={self.bmin}"
            if target_col not in opt_results.columns:
                raise ValueError(
                    f"B_min={args.bmin} not found in CSV. Available: {opt_results.columns.tolist()}"
                )

            # update the CHANNEL cuts
            self.channel.txbb_cut = float(opt_results.loc["Cut_Xbb", target_col])
            if args.use_bdt:
                self.channel.txtt_BDT_cut = float(opt_results.loc["Cut_Xtt", target_col])
            else:
                self.channel.txtt_cut = float(opt_results.loc["Cut_Xtt", target_col])

            print(
                f"Updated TXbb and Txtt cuts to {self.channel.txbb_cut} and {self.channel.txtt_cut if not self.use_bdt else self.channel.txtt_BDT_cut} for {self.channel.key}"
            )
        
def analyse_channel(
    years,
    sig_key,
    channel_key,
    test_mode,
    use_bdt,
    modelname,
    main_plot_dir,
    actions=None,
    at_inference=False,
    llsl_weight=1,
    bb_disc='bbFatJetParTXbbvsQCD',
    dataMinusSimABCD=False,
    sensitivity_dir=None,
):

    print(f"Processing signal: {sig_key} channel: {channel_key}. Test mode: {test_mode}.")

    analyser = Analyser(
        years, sig_key, channel_key, test_mode, use_bdt, modelname, main_plot_dir, at_inference, llsl_weight=llsl_weight, bb_disc=bb_disc, dataMinusSimABCD=dataMinusSimABCD
    )

    analyser.load_data()

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_and_plot_rocs(years)
    if "plot_mass" in actions:
        analyser.plot_mass(years)
    if "sensitivity" in actions:
        analyser.prepare_sensitivity(years)
        analyser.perform_optimization(years, use_thresholds=False)  # Temporary but to save time
    if "fom_study" in actions:
        analyser.prepare_sensitivity(years)
        analyser.study_foms(years)
    if "minimization_study" in actions:
        analyser.prepare_sensitivity(years)
        analyser.study_minimization_method(years)
    if "sensitivity_evaluation" in actions:
        analyser.prepare_sensitivity(years)
        # no contor plot in the score space as it only evalautes one point
        if sensitivity_dir is not None:
            analyser.update_cuts_from_csv_file(sensitivity_dir)
        analyser.perform_optimization(years, plot=False, evaluation=True)

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
        "--use-bdt", action="store_true", default=False, help="Use BDT for sensitivity study"
    )
    parser.add_argument(
        "--modelname",
        default="29July25_loweta_lowreg",
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
        choices=["compute_rocs", "plot_mass", "sensitivity", "fom_study", "minimization_study", "sensitivity_evaluation"],
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
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="/home/users/lumori/bbtautau/",
        type=str,
    )
    parser.add_argument(
        "--bdt-dir",
        help="directory where you save your bdt model for inference",
        default="/home/users/lumori/bbtautau/src/bbtautau//postprocessing/classifier/trained_models/29July25_loweta_lowreg",
        type=str,
    )
    parser.add_argument(
        "--llsl-weight",
        help="coefficient to multiply BDT TTSL and TTLL score with, when calculating tt disc",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--bb-disc",
        help="bb discriminator to optimize",
        default="bbFatJetParTXbbvsQCD",
        choices=["bbFatJetParTXbbvsQCD", "bbFatJetParTXbbvsQCDTop", "bbFatJetPNetXbbvsQCDLegacy"],
        type=str
    )
    parser.add_argument(
        "--dataMinusSimABCD",
        help="Use data minus sim ABCD method",
        action="store_true"
    )
    parser.add_argument(
        "--sensitivity-dir",
        help="Senstivity study directory that contains CSV files having the bb and tt cuts to evaluate (only for sensitivity_evaluation",
        default=None,
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
                llsl_weight=args.llsl_weight,
                bb_disc=args.bb_disc,
                dataMinusSimABCD=args.dataMinusSimABCD,
                sensitivity_dir=args.sensitivity_dir,
            )
