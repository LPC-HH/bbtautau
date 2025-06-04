from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import postprocessing
from boostedhh import hh_vars, plotting
from boostedhh.utils import PAD_VAL
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from Samples import CHANNELS
from sklearn.metrics import roc_curve

from bbtautau.HLTs import HLTs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# Global variables
MAIN_DIR = Path("/home/users/lumori/bbtautau/")
BDT_EVAL_DIR = Path("/ceph/cms/store/user/lumori/bbtautau/BDT_predictions/")
TODAY = "25Jun3"  # to name output folder
SIG_KEYS = {"hh": "bbtthh", "he": "bbtthe", "hm": "bbtthm"}  # TODO Generalize for other signals

data_dir_2022 = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal"
data_dir_otheryears = "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"

data_paths = {
    "2022": {
        "data": Path(data_dir_2022),
        "signal": Path(data_dir_2022),
    },
    "2022EE": {
        "data": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
    "2023": {
        "data": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
    "2023BPix": {
        "data": Path(data_dir_otheryears),
        "signal": Path(data_dir_otheryears),
    },
}


@dataclass
class Optimum:
    signal_yield: float
    bkg_yield: float
    transfer_factor: float
    cuts: tuple[float, float]


@dataclass
class SigBkgOptResult:
    max_signal: Optimum
    best_lims: Optimum
    # Optionally: could add grid arrays for plotting/diagnostics


def fom1(b, s):
    return 2 * np.sqrt(b) / s


def fom2(b, s, _tf):
    return 2 * np.sqrt(b * _tf + (b * _tf / np.sqrt(b)) ** 2) / s


class Analyser:
    def __init__(self, years, channel_key, test_mode, useBDT, modelname):
        self.channel = CHANNELS[channel_key]
        self.years = years
        self.test_mode = test_mode
        test_dir = "test" if test_mode else "full"
        self.plot_dir = Path(MAIN_DIR) / f"plots/SensitivityStudy/{TODAY}/{test_dir}/{channel_key}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # TODO we should get rid of this linw
        self.sig_key = SIG_KEYS[channel_key]

        self.taukey = CHANNELS[channel_key].tagger_label

        self.events_dict = {year: {} for year in years}

        self.useBDT = useBDT
        self.modelname = modelname

    def load_year(self, year):

        # This could be improved by adding channel-by-channel granularity
        # Now filter just requires that any trigger in that year fires
        filters_dict = postprocessing.trigger_filter(
            HLTs.hlts_list_by_dtype(year),
            year,
            fast_mode=self.test_mode,
            PNetXbb_cut=0.8 if not self.test_mode else None,
        )  # = {"data": [(...)], "signal": [(...)], ...}

        columns = postprocessing.get_columns(year, self.channel)

        self.events_dict[year] = postprocessing.load_samples(
            year,
            self.channel,
            data_paths[year],
            filters_dict=filters_dict,
            load_columns=columns,
            load_just_bbtt=True,
            loaded_samples=True,
        )
        self.events_dict[year] = postprocessing.apply_triggers(
            self.events_dict[year], year, self.channel
        )
        self.events_dict[year] = postprocessing.delete_columns(
            self.events_dict[year], year, self.channel
        )

        if self.useBDT:
            postprocessing.load_bdt_preds(
                self.events_dict[year], year, BDT_EVAL_DIR, modelname=self.modelname, all_outs=True
            )

        return

    def build_tagger_dict(self):
        # TODO integrate wiht postprocessing scripts
        self.taggers_dict = {year: {} for year in self.years}
        for year in self.years:
            for key, sample in self.events_dict[year].items():
                tvars = {}

                tvars["PQCD"] = sample.events["ak8FatJetParTQCD"].to_numpy()
                tvars["PTop"] = sample.events["ak8FatJetParTTop"].to_numpy()

                for disc in ["bb", self.taukey]:
                    tvars[f"X{disc}vsQCD"] = np.nan_to_num(
                        sample.events[f"ak8FatJetParTX{disc}"]
                        / (sample.events[f"ak8FatJetParTX{disc}"] + tvars["PQCD"]),
                        nan=PAD_VAL,
                    )
                    tvars[f"X{disc}vsQCDTop"] = np.nan_to_num(
                        sample.events[f"ak8FatJetParTX{disc}"]
                        / (sample.events[f"ak8FatJetParTX{disc}"] + tvars["PQCD"] + tvars["PTop"]),
                        nan=PAD_VAL,
                    )
                    # make sure not to choose padded jets below by accident
                    nojet3 = sample.events["ak8FatJetPt"][2] == PAD_VAL
                    tvars[f"X{disc}vsQCD"][:, 2][nojet3] = PAD_VAL
                    tvars[f"X{disc}vsQCDTop"][:, 2][nojet3] = PAD_VAL

                if self.useBDT:
                    tvars[f"BDTScore{self.taukey}vsQCD"] = np.nan_to_num(
                        sample.events[f"BDTScore{self.taukey}"]
                        / (sample.events[f"BDTScore{self.taukey}"] + sample.events["BDTScoreQCD"]),
                        nan=PAD_VAL,
                    )

                    tvars[f"BDTScore{self.taukey}vsQCDTop"] = np.nan_to_num(
                        sample.events[f"BDTScore{self.taukey}"]
                        / (
                            sample.events[f"BDTScore{self.taukey}"]
                            + sample.events["BDTScoreQCD"]
                            + sample.events["BDTScoreTThad"]
                            + sample.events["BDTScoreTTll"]
                            + sample.events["BDTScoreTTSL"]
                        ),
                        nan=PAD_VAL,
                    )

                tvars["PNetXbbvsQCD"] = np.nan_to_num(
                    sample.events["ak8FatJetPNetXbbLegacy"]
                    / (
                        sample.events["ak8FatJetPNetXbbLegacy"]
                        + sample.events["ak8FatJetPNetQCDLegacy"]
                    ),
                    nan=PAD_VAL,
                )

                # jet assignment
                fjbbpick = np.argmax(tvars["XbbvsQCD"], axis=1)
                fjttpick = np.argmax(tvars[f"X{self.taukey}vsQCD"], axis=1)
                overlap = fjbbpick == fjttpick
                fjbbpick[overlap] = np.argsort(tvars["XbbvsQCD"][overlap], axis=1)[:, 1]

                # convert ids to boolean masks
                fjbbpick_mask = np.zeros_like(tvars["XbbvsQCD"], dtype=bool)
                fjbbpick_mask[np.arange(len(fjbbpick)), fjbbpick] = True
                fjttpick_mask = np.zeros_like(tvars[f"X{self.taukey}vsQCD"], dtype=bool)
                fjttpick_mask[np.arange(len(fjttpick)), fjttpick] = True

                tvars["bb_mask"] = fjbbpick_mask
                tvars["tautau_mask"] = fjttpick_mask
                self.taggers_dict[year][key] = tvars

    @staticmethod
    def get_jet_vals(vals, mask):
        # check if vals is a numpy array
        if not isinstance(vals, np.ndarray):
            vals = vals.to_numpy()
        return vals[mask]

    def compute_rocs(self, years, jets=None, discs=None):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")
        if jets is None:
            jets = ["bb", "tautau"]
        if discs is None:
            discs = [
                "XbbvsQCD",
                "XbbvsQCDTop",
                f"X{self.taukey}vsQCD",
                f"X{self.taukey}vsQCDTop",
                "PNetXbbvsQCD",
            ]
        if not hasattr(self, "rocs"):
            self.rocs = {}
        self.rocs["_".join(years)] = {jet: {} for jet in jets}
        for jet in jets:
            for i, disc in enumerate(discs):
                bg_scores = np.concatenate(
                    [
                        self.get_jet_vals(
                            self.taggers_dict[year][key][disc],
                            self.taggers_dict[year][key][f"{jet}_mask"],
                        )
                        for key in self.channel.data_samples
                        for year in years
                    ]
                )
                bg_weights = np.concatenate(
                    [
                        self.events_dict[year][key].events["finalWeight"]
                        for key in self.channel.data_samples
                        for year in years
                    ]
                )

                sig_scores = np.concatenate(
                    [
                        self.get_jet_vals(
                            self.taggers_dict[year][self.sig_key][disc],
                            self.taggers_dict[year][self.sig_key][f"{jet}_mask"],
                        )
                        for year in years
                    ]
                )
                sig_weights = np.concatenate(
                    [self.events_dict[year][self.sig_key].events["finalWeight"] for year in years]
                )

                fpr, tpr, thresholds = roc_curve(
                    np.concatenate([np.zeros_like(bg_scores), np.ones_like(sig_scores)]),
                    np.concatenate([bg_scores, sig_scores]),
                    sample_weight=np.concatenate([bg_weights, sig_weights]),
                )

                self.rocs["_".join(years)][jet][disc] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds,
                    "label": disc,
                    "color": plt.cm.tab10.colors[i],
                }

        return fpr, tpr, thresholds

    def plot_rocs(self, years):
        if not hasattr(self, "rocs") or "_".join(years) not in self.rocs:
            print(f"No ROC curves computed yet in years {years}")
        for jet, title in zip(["bb", "tautau"], ["bb FatJet", rf"{self.channel.label} FatJet"]):

            # Choose which curves to plot
            if jet == "bb":
                list_disc = ["XbbvsQCD", "XbbvsQCDTop", "PNetXbbvsQCD"]
            else:
                list_disc = [f"X{self.taukey}vsQCD", f"X{self.taukey}vsQCDTop"]

            # create rocs directory if it doesn't exist
            (self.plot_dir / "rocs").mkdir(parents=True, exist_ok=True)

            plotting.multiROCCurve(
                {"": {k: self.rocs["_".join(years)][jet][k] for k in list_disc}},
                title=title,
                thresholds=[0.3, 0.7, 0.9, 0.95, 0.99, 0.998],
                show=True,
                plot_dir=self.plot_dir / "rocs",
                lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
                year="2022-23" if years == hh_vars.years else "+".join(years),
                name=jet + "_".join(years),
            )

    def plot_mass(self, years):
        for key, label in zip(["hhbbtt", "data"], ["HHbbtt", "Data"]):
            print(f"Plotting mass for {label}")
            if key == "hhbbtt":
                events = pd.concat([self.events_dict[year][self.sig_key].events for year in years])
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
                zip(["bb", "tautau"], ["bb FatJet", rf"{self.channel.label} FatJet"])
            ):
                ax = axs[i]
                if key == "hhbbtt":
                    mask = np.concatenate(
                        [self.taggers_dict[year][self.sig_key][f"{jet}_mask"] for year in years],
                        axis=0,
                    )
                else:
                    mask = np.concatenate(
                        [
                            self.taggers_dict[year][dkey][f"{jet}_mask"]
                            for dkey in self.channel.data_samples
                            for year in years
                        ],
                        axis=0,
                    )

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

        mbbk = "ParTmassResApplied"
        mttk = {"hh": "PNetmassLegacy", "hm": "ParTmassResApplied", "he": "ParTmassResApplied"}[
            self.channel.key
        ]

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

        # precompute to speedup
        for year in years:
            for key in [self.sig_key] + self.channel.data_samples:
                self.txbbs[year][key] = self.get_jet_vals(
                    self.taggers_dict[year][key]["XbbvsQCD"],
                    self.taggers_dict[year][key]["bb_mask"],
                )
                if self.useBDT:
                    self.txtts[year][key] = self.get_jet_vals(
                        self.taggers_dict[year][key][f"BDTScore{self.taukey}vsQCD"],
                        self.taggers_dict[year][key]["tautau_mask"],
                    )
                else:
                    self.txtts[year][key] = self.get_jet_vals(
                        self.taggers_dict[year][key][f"X{self.taukey}vsQCDTop"],
                        self.taggers_dict[year][key]["tautau_mask"],
                    )
                self.masstt[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].events[f"ak8FatJet{mttk}"],
                    self.taggers_dict[year][key]["tautau_mask"],
                )
                self.massbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].events[f"ak8FatJet{mbbk}"],
                    self.taggers_dict[year][key]["bb_mask"],
                )
                self.ptbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].events["ak8FatJetPt"],
                    self.taggers_dict[year][key]["bb_mask"],
                )

    def compute_sig_bg(self, years, txbbcut, txttcut, mbb1, mbb2, mbbw2, mtt1, mtt2):
        bg_yield = 0
        sig_yield = 0
        for year in years:
            for key in [self.sig_key] + self.channel.data_samples:
                if key == self.sig_key:
                    cut = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    sig_yield += np.sum(self.events_dict[year][key].events["finalWeight"][cut])
                else:
                    cut = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.ptbb[year][key] > 250)
                    )
                    msb1 = (self.massbb[year][key] > (mbb1 - mbbw2)) & (
                        self.massbb[year][key] < mbb1
                    )
                    msb2 = (self.massbb[year][key] > mbb2) & (
                        self.massbb[year][key] < (mbb2 + mbbw2)
                    )
                    bg_yield += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut & msb1]
                    )
                    bg_yield += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut & msb2]
                    )
        return sig_yield, bg_yield, 1

    def compute_sig_bkg_abcd(self, years, txbbcut, txttcut, mbb1, mbb2, mbbw2, mtt1, mtt2):
        # pass/fail from taggers
        sig_pass = 0  # resonant region pass
        sig_fail = 0  # resonant region fail
        bg_pass = 0  # sidebands pass
        bg_fail = 0  # sidebands fail
        for year in years:
            for key in [self.sig_key] + self.channel.data_samples:
                if key == self.sig_key:
                    cut = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    sig_pass += np.sum(self.events_dict[year][key].events["finalWeight"][cut])
                else:  # compute background
                    cut_bg_pass = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.ptbb[year][key] > 250)
                    )
                    msb1 = (self.massbb[year][key] > (mbb1 - mbbw2)) & (
                        self.massbb[year][key] < mbb1
                    )
                    msb2 = (self.massbb[year][key] > mbb2) & (
                        self.massbb[year][key] < (mbb2 + mbbw2)
                    )
                    bg_pass += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_pass & msb1]
                    )
                    bg_pass += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_pass & msb2]
                    )
                    cut_bg_fail = (
                        ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.ptbb[year][key] > 250)
                    )
                    bg_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail & msb1]
                    )
                    bg_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail & msb2]
                    )
                    cut_sig_fail = (
                        ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                        & (self.masstt[year][key] > mtt1)
                        & (self.masstt[year][key] < mtt2)
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    sig_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_sig_fail]
                    )

        return sig_pass, bg_pass, sig_fail / bg_fail

    def sig_bkg_opt(
        self,
        years,
        gridsize,
        gridlims,
        B_max,
        normalize_sig=True,
        plot=True,
        use_abcd=True,
        legacy=False,
    ) -> SigBkgOptResult:

        mbb1, mbb2 = 110.0, 140.0
        mbbw2 = (mbb2 - mbb1) / 2
        mtt1, mtt2 = {"hh": (50, 150), "hm": (70, 210), "he": (70, 210)}[self.channel.key]

        bbcut = np.linspace(*gridlims, gridsize)
        ttcut = np.linspace(*gridlims, gridsize)

        BBcut, TTcut = np.meshgrid(bbcut, ttcut)

        # Flatten the grid for parallel evaluation
        bbcut_flat = BBcut.ravel()
        ttcut_flat = TTcut.ravel()

        # define the FOM
        sig_bkg_f = self.compute_sig_bkg_abcd if use_abcd else self.compute_sig_bg

        # scalar function, must be vectorized
        def sig_bg(bbcut, ttcut):
            return sig_bkg_f(
                years=years,
                txbbcut=bbcut,
                txttcut=ttcut,
                mbb1=mbb1,
                mbb2=mbb2,
                mbbw2=mbbw2,
                mtt1=mtt1,
                mtt2=mtt2,
            )

        if legacy:
            sigs, bgs, tfs = np.vectorize(sig_bg)(BBcut, TTcut)
        else:
            # Run in parallel
            results = Parallel(n_jobs=-1, verbose=1)(
                delayed(sig_bg)(b, t) for b, t in zip(bbcut_flat, ttcut_flat)
            )
            # results is a list of (sig, bkg, tf) tuples

            sigs, bgs, tfs = zip(*results)
            sigs = np.array(sigs).reshape(BBcut.shape)
            bgs = np.array(bgs).reshape(BBcut.shape)
            tfs = np.array(tfs).reshape(BBcut.shape)

        bgs_scaled = bgs * tfs
        if normalize_sig:
            tot_sig_weight = np.sum(
                np.concatenate(
                    [self.events_dict[year][self.sig_key].events["finalWeight"] for year in years]
                )
            )
            sigs = sigs / tot_sig_weight

        sel = (bgs_scaled > 0) & (bgs_scaled <= B_max)

        B_initial = B_max
        if np.sum(sel) == 0:
            while np.sum(sel) == 0 and B_max < 100:
                B_max += 1
                sel = (bgs_scaled > 0) & (bgs_scaled <= B_max)
            print(
                f"Need a finer grid, no region with B={B_initial}. I'm extending the region to B in [1,{B_max}].",
                bgs_scaled,
            )
        sel_idcs = np.argwhere(sel)
        opt_i = np.argmax(sigs[sel])
        max_sig_idx = tuple(sel_idcs[opt_i])
        bbcut_opt, ttcut_opt = BBcut[max_sig_idx], TTcut[max_sig_idx]

        # Default on updated FOM. TODO: allow multiple foms
        significance = np.divide(
            sigs,
            np.sqrt(bgs_scaled + (bgs_scaled / np.sqrt(bgs)) ** 2),
            out=np.zeros_like(sigs),
            where=(bgs_scaled > 1e-8),
        )

        # significance = np.where(bgs > 0, sigs / np.sqrt(bgs), 0)
        max_significance_i = np.unravel_index(np.argmax(significance), significance.shape)
        bbcut_opt_significance, ttcut_opt_significance = (
            BBcut[max_significance_i],
            TTcut[max_significance_i],
        )

        if plot:
            # TODO scale and generalizetransfer to plotting module
            plt.rcdefaults()
            plt.style.use(hep.style.CMS)
            hep.style.use("CMS")
            fig, ax = plt.subplots(figsize=(10, 10))
            hep.cms.label(
                ax=ax,
                label="Work in Progress",
                data=True,
                year="2022-23" if years == hh_vars.years else "+".join(years),
                com="13.6",
                fontsize=13,
                lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
            )
            sigmap = ax.contourf(BBcut, TTcut, sigs, levels=10, cmap="viridis")
            ax.contour(BBcut, TTcut, sel, colors="r")
            proxy = Line2D([0], [0], color="r", label="B=1" if B_max == 1 else f"B in [1,{B_max}]")
            ax.scatter(bbcut_opt, ttcut_opt, color="r", label="Max. signal cut")
            ax.scatter(
                bbcut_opt_significance,
                ttcut_opt_significance,
                color="b",
                label="Opt. lim. $\sqrt{b+\sigma_b}/s$ cut",
            )
            ax.set_xlabel("Xbb vs QCD cut")
            ax.set_ylabel(
                f"XBDTScore{self.taukey} vs QCD cut"
                if self.useBDT
                else f"X{self.taukey} vs QCDTop cut"
            )
            cbar = plt.colorbar(sigmap, ax=ax)
            cbar.set_label("Signal efficiency" if normalize_sig else "Signal yield")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(proxy)
            ax.legend(handles=handles, loc="lower left")

            ax.text(
                0.05,
                0.72,
                self.channel.label,
                transform=ax.transAxes,
                fontsize=20,
                fontproperties="Tex Gyre Heros",
            )

            plot_path = self.plot_dir / (
                "sig_bkg_opt" + f"_BDT_{self.modelname}" if self.useBDT else "sig_bkg_opt"
            )
            plot_path.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                plot_path / f"{'_'.join(years)}_B={B_initial}{'_abcd' if use_abcd else ''}.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                plot_path / f"{'_'.join(years)}_B={B_initial}{'_abcd' if use_abcd else ''}.png",
                bbox_inches="tight",
            )
            plt.show()

        return SigBkgOptResult(
            max_signal=Optimum(
                signal_yield=sigs[max_sig_idx],
                bkg_yield=bgs[max_sig_idx],
                transfer_factor=tfs[max_sig_idx],
                cuts=(bbcut_opt, ttcut_opt),
            ),
            best_lims=Optimum(
                signal_yield=sigs[max_significance_i],
                bkg_yield=bgs[max_significance_i],
                transfer_factor=tfs[max_significance_i],
                cuts=(bbcut_opt_significance, ttcut_opt_significance),
            ),
        )

    @staticmethod
    def print_nicely(sig_yield, bg_yield, years):
        print(
            f"""

            Yield study year(s) {years}:

            """
        )

        print("Sig yield", sig_yield)
        print("BG yield", bg_yield)
        print("limit", 2 * np.sqrt(bg_yield) / sig_yield)

        if "2023" not in years or "2023BPix" not in years:
            print(
                "limit scaled to 22-23 all channels",
                2
                * np.sqrt(bg_yield)
                / sig_yield
                / np.sqrt(
                    hh_vars.LUMI["2022-2023"] / np.sum([hh_vars.LUMI[year] for year in years]) * 3
                ),
            )
        print(
            "limit scaled to 22-24 all channels",
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt(
                (124000 + hh_vars.LUMI["2022-2023"])
                / np.sum([hh_vars.LUMI[year] for year in years])
                * 3
            ),
        )
        print(
            "limit scaled to Run 3 all channels",
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt((360000) / np.sum([hh_vars.LUMI[year] for year in years]) * 3),
        )
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
        # Unpack for readability
        cut_bb, cut_tt = optimum.cuts
        sig_yield = optimum.signal_yield
        bg_yield = optimum.bkg_yield
        tf = optimum.transfer_factor

        limits = {}
        limits["Label"] = label
        limits["Cut_Xbb"] = cut_bb
        limits["Cut_Xtt"] = cut_tt
        limits["Sig_Yield"] = sig_yield
        limits["BG_Yield_scaled"] = bg_yield * tf
        limits["TF"] = tf

        # limits[r"Limit, 2$\sqrt{b}/s$"] = fom1(bg_yield, sig_yield)

        limits[r"Limit, 2$\sqrt{b + (b * \sigma_b)^2}/s$"] = fom2(bg_yield, sig_yield, tf)

        if "2023" not in years and "2023BPix" not in years:
            limits["Limit_scaled_22_23"] = fom2(bg_yield, sig_yield, tf) / np.sqrt(
                hh_vars.LUMI["2022-2023"] / np.sum([hh_vars.LUMI[year] for year in years])
            )

        limits["Limit_scaled_22_24"] = fom2(bg_yield, sig_yield, tf) / np.sqrt(
            (124000 + hh_vars.LUMI["2022-2023"]) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        limits["Limit_scaled_Run3"] = fom2(bg_yield, sig_yield, tf) / np.sqrt(
            (360000) / np.sum([hh_vars.LUMI[year] for year in years])
        )

        df_out = pd.DataFrame([limits])
        return df_out


def analyse_channel(
    years,
    channel,
    test_mode,
    use_bdt,
    modelname,
    use_abcd=True,
    actions=None,
    b_vals=None,
):
    print(f"Processing channel: {channel}. Test mode: {test_mode}.")
    analyser = Analyser(years, channel, test_mode, use_bdt, modelname)

    for year in years:
        analyser.load_year(year)
    analyser.build_tagger_dict()

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_rocs(years)
        analyser.plot_rocs(years)
    if "plot_mass" in actions:
        analyser.plot_mass(years)
    if "sensitivity" in actions:
        analyser.prepare_sensitivity(years)
        results = {}
        if b_vals is None:
            b_vals = [2] if test_mode else [1, 2, 8]
        for B_max in b_vals:
            result = analyser.sig_bkg_opt(
                years,
                gridlims=(0.8, 1),
                gridsize=5 if test_mode else 40,
                B_max=B_max,
                plot=True,
                use_abcd=use_abcd,
            )
            # result: SigBkgOptResult
            results[f"B={B_max}"] = analyser.as_df(result.max_signal, years, label=f"B={B_max}")
            print("done with B=", B_max)

        results["Best_lims"] = analyser.as_df(result.best_lims, years, label="Best limits")

        results_df = pd.concat(results, axis=0)
        results_df.index = results_df.index.droplevel(1)
        print(channel, "\n", results_df.T.to_markdown())
        output_csv = (
            analyser.plot_dir
            / f"{'_'.join(years)}-results{'_abcd' if use_abcd else ''}{'_BDT' if use_bdt else ''}.csv"
        )
        results_df.T.to_csv(output_csv)

    if "time-methods" in actions:
        analyser.prepare_sensitivity(years)
        print("Timing sig_bkg_opt_legacy...")
        start_legacy = time.perf_counter()
        analyser.sig_bkg_opt(
            years, gridsize=20, gridlims=(0.8, 1), B=1, plot=False, use_abcd=True, legacy=True
        )
        end_legacy = time.perf_counter()
        print(f"sig_bkg_opt_legacy: {end_legacy - start_legacy:.3f} seconds")

        print("Timing sig_bkg_opt (parallel)...")
        start_new = time.perf_counter()
        analyser.sig_bkg_opt(years, gridsize=20, gridlims=(0.8, 1), B=1, plot=False, use_abcd=True)
        end_new = time.perf_counter()
        print(f"sig_bkg_opt (parallel): {end_new - start_new:.3f} seconds")
    del analyser


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sensitivity Study Script")

    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="List of years to include in the analysis",
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
        default="28May25_baseline",
        help="Name of the BDT model to use for sensitivity study",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["compute_rocs", "plot_mass", "sensitivity", "time-methods"],
        required=True,
        help="Actions to perform. Choose one or more.",
    )

    # TODO: see what to do for these options
    # parser.add_argument(
    #     "--use-abcd", action="store_false", default=True,
    #     help="Use ABCD background estimation"
    # )

    # parser.add_argument(
    #     "--b-vals", nargs="+", type=int, default=[1, 2, 8],
    #     help="B values for significance scan (default: 1 2 8)"
    # )

    args = parser.parse_args()

    # check: test-mode and use-bdt are mutually exclusive
    if args.test_mode and args.use_bdt:
        raise ValueError("Test mode and use-bdt are mutually exclusive")

    for channel in args.channels:
        analyse_channel(
            years=args.years,
            channel=channel,
            test_mode=args.test_mode,
            use_bdt=args.use_bdt,
            modelname=args.modelname,
            use_abcd=True,  # temporary. will need to generalize framework
            actions=args.actions,
            b_vals=None,  # TODO: add b_vals
        )
