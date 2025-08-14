from __future__ import annotations

import argparse
import gc
import logging
import time
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
from postprocessing import (
    apply_triggers,
    base_filter,
    bbtautau_assignment,
    check_bdt_prediction_shapes,
    compute_bdt_preds,
    delete_columns,
    derive_variables,
    get_columns,
    leptons_assignment,
    load_bdt_preds,
    load_samples,
    tt_filters,
)
from Samples import CHANNELS
from skopt import gp_minimize

from bbtautau.postprocessing import utils
from bbtautau.postprocessing.plotting import (
    plot_optimization_sig_eff,
    plot_optimization_thresholds,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.userConfig import DATA_PATHS, MODEL_DIR, SHAPE_VAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# Global variables
BDT_EVAL_DIR = Path("/ceph/cms/store/user/lumori/bbtautau/BDT_predictions/")
TODAY = date.today()  # to name output folder


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


def fom_2sqrtB_S(b, s, _tf):
    return np.where(s > 0, 2 * np.sqrt(b * _tf) / s, -PAD_VAL)


def fom_2sqrtB_S_var(b, s, _tf):
    return np.where(
        (b > 0) & (s > 0), 2 * np.sqrt(b * _tf + (b * _tf / np.sqrt(b)) ** 2) / s, -PAD_VAL
    )


def fom_punzi(b, s, _tf, a=3):
    """
    a is the number of sigmas of the test significance
    """
    return np.where(s > 0, (np.sqrt(b * _tf) + a / 2) / s, -PAD_VAL)


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
        self.model_dir = MODEL_DIR

        self.plot_dir = (
            Path(main_plot_dir)
            / f"plots/SensitivityStudy/{TODAY}/{test_dir}/{sig_key}/{tt_tagger_name}/{channel_key}"
        )
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # This is hardcoded here
        self.bb_disc_name = "bbFatJetParTXbbvsQCDTop"
        self.tt_disc_name = (
            f"BDTScore{self.taukey}vsAll" if use_bdt else f"ttFatJetParTX{self.taukey}vsQCDTop"
        )

    def load_data(self, tt_pres=False):

        for year in self.years:

            filters_dict = base_filter(self.test_mode)
            # filters_dict = bb_filters(filters_dict, num_fatjets=3, bb_cut=0.3)

            if tt_pres:
                filters_dict = tt_filters(self.channel, filters_dict, num_fatjets=3, tt_cut=0.3)

            columns = get_columns(year, triggers_in_channel=self.channel)

            self.events_dict[year] = load_samples(
                year=year,
                paths=DATA_PATHS[year],
                signals=[self.sig_key],
                channels=[self.channel],
                filters_dict=filters_dict,
                load_columns=columns,
                restrict_data_to_channel=True,
                loaded_samples=True,
                multithread=True,
            )

            apply_triggers(self.events_dict[year], year, self.channel)
            delete_columns(self.events_dict[year], year, channels=[self.channel])

            derive_variables(
                self.events_dict[year], CHANNELS["hm"]
            )  # legacy issue, muon branches are misnamed
            bbtautau_assignment(self.events_dict[year], agnostic=True)
            leptons_assignment(self.events_dict[year], dR_cut=1.5)

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
            for key in [self.sig_key_channel] + self.channel.data_samples:
                self.txbbs[year][key] = self.events_dict[year][key].get_var(self.bb_disc_name)
                self.txtts[year][key] = self.events_dict[year][key].get_var(self.tt_disc_name)
                self.masstt[year][key] = self.events_dict[year][key].get_var(mttk)
                self.massbb[year][key] = self.events_dict[year][key].get_var(mbbk)
                self.ptbb[year][key] = self.events_dict[year][key].get_var("bbFatJetPt")
                self.pttt[year][key] = self.events_dict[year][key].get_var("ttFatJetPt")

    def compute_sig_bkg_abcd(self, years, txbbcut, txttcut, mbb1, mbb2, mtt1, mtt2):
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

            sig_pass += np.sum(
                self.events_dict[year][self.sig_key_channel].get_var("finalWeight")[cut_sig_pass]
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

        else:
            # Raw threshold coordinate system
            bbcut = np.linspace(*gridlims, gridsize)
            ttcut = np.linspace(*gridlims, gridsize)
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
                            "bb_disc_name": self.bb_disc_name,
                            "tt_disc_name": self.tt_disc_name,
                        }
                    )

                results[fom.name][f"Bmin={B_min}"] = Optimum(**optimum_args)

        return results

    def bayesian_opt(self, years, B_min_vals, gridlims, foms) -> Optimum:

        mbb1, mbb2 = SHAPE_VAR["blind_window"]
        mtt1, mtt2 = self.channel.tt_mass_cut[1]

        sig_bkg_f = self.compute_sig_bkg_abcd

        def objective(cuts, B_min, fom):
            txbbcut, txttcut = cuts
            sig, bg, sig_fail, bg_fail, tf = sig_bkg_f(
                years, txbbcut, txttcut, mbb1, mbb2, mtt1, mtt2
            )

            if bg * tf > 1e-8 and bg > B_min:  # enforce soft constraint
                limit = fom.fom_func(bg, sig, tf)
            else:
                limit = np.abs(PAD_VAL)
            return float(limit)

        results = {}
        for fom in foms:
            results[fom.name] = {}
            for B_min in B_min_vals:
                results[fom.name][f"Bmin={B_min}"] = {}
                res = gp_minimize(
                    lambda cuts, B_min=B_min, fom=fom: objective(cuts, B_min, fom),
                    dimensions=[gridlims, gridlims],  # [txbbcut, txttcut] ranges
                    n_calls=40,
                    random_state=42,
                    verbose=True,
                )

                print("Bayesian optimization results:")
                print("Best limit:", res.fun)
                print("Optimal cuts:", res.x)

                sig_opt, bg_opt, sig_fail_opt, bg_fail_opt, tf_opt = sig_bkg_f(
                    years, res.x[0], res.x[1], mbb1, mbb2, mtt1, mtt2
                )

                results[fom.name][f"Bmin={B_min}"] = Optimum(
                    limit=res.fun,
                    signal_yield=sig_opt,
                    bkg_yield=bg_opt,
                    hmass_fail=sig_fail_opt,
                    sideband_fail=bg_fail_opt,
                    transfer_factor=tf_opt,
                    cuts=(res.x[0], res.x[1]),
                )

        return results

    def perform_optimization(self, years, use_thresholds=False, plot=True, b_min_vals=None):

        if b_min_vals is None:
            b_min_vals = [1, 3, 5] if self.test_mode else np.arange(1, 17, 2)

        if self.test_mode:
            gridlims = (0.3, 1) if use_thresholds else (0.2, 0.9)
        else:
            gridlims = (0.7, 1) if use_thresholds else (0.25, 0.75)

        gridsize = 20 if self.test_mode else 50

        foms = [FOMS["2sqrtB_S_var"]]

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

    def study_minimization_method(self, years):
        """
        Compare grid search vs Bayesian optimization methods for limit minimization.

        This method:
        1. Runs both grid search and Bayesian optimization for different B_min values
        2. Measures execution time for each method
        3. Compares the resulting limits and optimal cuts
        4. Creates plots and summary tables
        """
        import time

        # Configuration parameters
        b_min_vals = [1, 2]  # np.arange(1, 17, 2)  # [1, 3, 5, 7, 9, 11, 13, 15]
        gridlims = (0.7, 1.0)
        gridsize = 20 if self.test_mode else 50
        bayesian_calls = 100
        foms = [FOMS["2sqrtB_S_var"]]

        # Results storage
        results = {"grid_search": {}, "bayesian": {}}

        # Timing storage
        timing_results = {"grid_search": {}, "bayesian": {}}

        print("Starting minimization method comparison...")
        print(f"Testing B_min values: {b_min_vals}")
        print(f"Grid search: {gridsize}x{gridsize} grid")
        print(f"Bayesian optimization: {bayesian_calls} function calls")

        # Bayesian Optimization
        print("  Running Bayesian optimization...")
        start_time = time.perf_counter()
        bayesian_results = self.bayesian_opt(
            years=years, B_min_vals=b_min_vals, gridlims=gridlims, foms=foms
        )
        bayesian_time = time.perf_counter() - start_time
        timing_results["bayesian"] = bayesian_time
        results["bayesian"] = bayesian_results

        # Grid Search
        print("  Running grid search...")
        start_time = time.perf_counter()
        grid_results = self.grid_search_opt(
            years=years, gridsize=gridsize, gridlims=gridlims, B_min_vals=b_min_vals, foms=foms
        )
        grid_time = time.perf_counter() - start_time
        timing_results["grid_search"] = grid_time
        results["grid_search"] = grid_results

        print(f"  Grid search time: {grid_time:.2f}s")
        print(f"  Bayesian time: {bayesian_time:.2f}s")
        print(f"  Speedup: {grid_time/bayesian_time:.2f}x")

        # Create comparison plots
        self._plot_minimization_comparison(results, timing_results, b_min_vals, foms, years)

        # Create summary table
        self._create_minimization_summary(results, timing_results, b_min_vals, foms, years)

        print("\nMinimization method comparison completed!")
        return results, timing_results

    def _plot_minimization_comparison(self, results, timing_results, b_min_vals, foms, years):
        """
        Create comparison plots for grid search vs Bayesian optimization.
        """
        plt.rcdefaults()
        plt.style.use(hep.style.CMS)
        hep.style.use("CMS")

        # Create subplots: timing comparison and limit comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        colors = ["blue", "red", "green", "orange"]
        markers = ["o", "s", "D", "^"]

        # Plot 1: Timing comparison
        for i, method in enumerate(["grid_search", "bayesian"]):
            times = [timing_results[method][B_min] for B_min in b_min_vals]
            ax1.plot(
                b_min_vals,
                times,
                color=colors[i],
                marker=markers[i],
                linewidth=2,
                markersize=8,
                label=f"{method.replace('_', ' ').title()}",
            )

        ax1.set_xlabel("B_min (background constraint)")
        ax1.set_ylabel("Execution time (seconds)")
        ax1.set_title("Execution Time Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Limit comparison for each FOM function
        for i, fom in enumerate(foms):
            fom_name = fom.name

            # Grid search limits
            grid_limits = []
            bayesian_limits = []

            for B_min in b_min_vals:
                grid_limit = results["grid_search"][fom.name][f"Bmin={B_min}"].limit
                bayesian_limit = results["bayesian"][fom.name][f"Bmin={B_min}"].limit
                grid_limits.append(grid_limit)
                bayesian_limits.append(bayesian_limit)

            # Plot grid search results
            ax2.plot(
                b_min_vals,
                grid_limits,
                color=colors[0],
                marker=markers[0],
                linewidth=2,
                markersize=8,
                label=f"Grid Search ({fom_name})" if i == 0 else "",
            )

            # Plot Bayesian results
            ax2.plot(
                b_min_vals,
                bayesian_limits,
                color=colors[1],
                marker=markers[1],
                linewidth=2,
                markersize=8,
                label=f"Bayesian ({fom_name})" if i == 0 else "",
            )

        ax2.set_xlabel("B_min (background constraint)")
        ax2.set_ylabel("Limit")
        ax2.set_title("Limit Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add CMS label
        hep.cms.label(
            ax=ax1,
            label="Work in Progress",
            data=True,
            year="2022-23" if years == hh_vars.years else "+".join(years),
            com="13.6",
            fontsize=13,
            lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        )

        # Add channel label
        ax1.text(
            0.05,
            0.95,
            self.channel.label,
            transform=ax1.transAxes,
            fontsize=16,
            fontproperties="Tex Gyre Heros",
            verticalalignment="top",
        )

        # Save plots
        plot_path = self.plot_dir / "minimization_comparison"
        plot_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            plot_path / f"minimization_comparison_{'_'.join(years)}.pdf", bbox_inches="tight"
        )
        plt.savefig(
            plot_path / f"minimization_comparison_{'_'.join(years)}.png", bbox_inches="tight"
        )
        plt.close()

        print(f"Comparison plots saved to {plot_path}")

    def _create_minimization_summary(self, results, timing_results, b_min_vals, foms, years):
        """
        Create a comprehensive summary table of the comparison results.
        """
        summary_data = []

        for B_min in b_min_vals:
            for fom in foms:
                fom_name = fom.name

                row = {"B_min": B_min, "FOM_function": fom_name}

                # Grid search results
                grid_opt = results["grid_search"][fom_name][f"Bmin={B_min}"]
                row.update(
                    {
                        "grid_time": timing_results["grid_search"][B_min],
                        "grid_limit": grid_opt.limit,
                        "grid_sig_yield": grid_opt.signal_yield,
                        "grid_bkg_yield": grid_opt.bkg_yield,
                        "grid_cuts": f"({grid_opt.cuts[0]:.3f}, {grid_opt.cuts[1]:.3f})",
                        "grid_tf": grid_opt.transfer_factor,
                    }
                )

                # Bayesian results
                bayesian_opt = results["bayesian"][fom_name][f"Bmin={B_min}"]
                row.update(
                    {
                        "bayesian_time": timing_results["bayesian"][B_min],
                        "bayesian_limit": bayesian_opt.limit,
                        "bayesian_sig_yield": bayesian_opt.signal_yield,
                        "bayesian_bkg_yield": bayesian_opt.bkg_yield,
                        "bayesian_cuts": f"({bayesian_opt.cuts[0]:.3f}, {bayesian_opt.cuts[1]:.3f})",
                        "bayesian_tf": bayesian_opt.transfer_factor,
                    }
                )

                # Calculate differences
                if not np.isnan(row["grid_limit"]) and not np.isnan(row["bayesian_limit"]):
                    row["limit_diff"] = row["bayesian_limit"] - row["grid_limit"]
                    row["limit_ratio"] = row["bayesian_limit"] / row["grid_limit"]
                    row["time_ratio"] = row["grid_time"] / row["bayesian_time"]
                else:
                    row["limit_diff"] = np.nan
                    row["limit_ratio"] = np.nan
                    row["time_ratio"] = np.nan

                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        plot_path = self.plot_dir / "minimization_comparison"
        plot_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(plot_path / f"minimization_summary_{'_'.join(years)}.csv", index=False)

        # Print summary statistics
        print("\n" + "=" * 80)
        print("MINIMIZATION METHOD COMPARISON SUMMARY")
        print("=" * 80)

        for fom in foms:
            fom_name = fom.name
            print(f"\nResults for {fom_name}:")

            # Filter data for this FOM function
            fom_data = summary_df[summary_df["FOM_function"] == fom_name]

            if not fom_data.empty:
                # Time comparison
                avg_grid_time = fom_data["grid_time"].mean()
                avg_bayesian_time = fom_data["bayesian_time"].mean()
                avg_speedup = fom_data["time_ratio"].mean()

                print("  Average execution time:")
                print(f"    Grid search: {avg_grid_time:.2f}s")
                print(f"    Bayesian: {avg_bayesian_time:.2f}s")
                print(f"    Average speedup: {avg_speedup:.2f}x")

                # Limit comparison
                avg_limit_ratio = fom_data["limit_ratio"].mean()
                print(f"  Average limit ratio (Bayesian/Grid): {avg_limit_ratio:.3f}")

                # Best results
                best_grid_idx = fom_data["grid_limit"].idxmin()
                best_bayesian_idx = fom_data["bayesian_limit"].idxmin()

                print(
                    f"  Best grid search limit: {fom_data.loc[best_grid_idx, 'grid_limit']:.3f} (B_min={fom_data.loc[best_grid_idx, 'B_min']})"
                )
                print(
                    f"  Best Bayesian limit: {fom_data.loc[best_bayesian_idx, 'bayesian_limit']:.3f} (B_min={fom_data.loc[best_bayesian_idx, 'B_min']})"
                )

        print("\n" + "=" * 80)

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
    actions=None,
    at_inference=False,
):

    print(f"Processing signal: {sig_key} channel: {channel_key}. Test mode: {test_mode}.")

    analyser = Analyser(
        years, sig_key, channel_key, test_mode, use_bdt, modelname, main_plot_dir, at_inference
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
        analyser.perform_optimization(years, use_thresholds=True)
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
            )
