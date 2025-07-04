from __future__ import annotations

# start_time = time.time()
import argparse
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# mid_time = time.time()
# print(f"Time taken for block 1: {mid_time - start_time} seconds")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

# mid_time = time.time()
# print(f"Time taken for block 2: {mid_time - start_time} seconds")
from boostedhh import hh_vars
from boostedhh.utils import PAD_VAL
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from postprocessing import (
    bbtautau_assignment,
    compute_bdt_preds,
    delete_columns,
    derive_variables,
    get_columns,
    load_bdt_preds,
    load_samples,
    trigger_filter,
)
from Samples import CHANNELS

from bbtautau.HLTs import HLTs
from bbtautau.postprocessing import utils
from bbtautau.postprocessing.rocUtils import ROCAnalyzer

# mid_time = time.time()
# print(f"Time taken for block 3: {mid_time - start_time} seconds")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# end_time = time.time()
# print(f"Time taken for import setup: {end_time - start_time} seconds")

# Global variables
MAIN_DIR = Path("/home/users/lumori/bbtautau/")
BDT_EVAL_DIR = Path("/ceph/cms/store/user/lumori/bbtautau/BDT_predictions/")
TODAY = date.today()  # to name output folder
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
    hmass_fail: float
    sideband_fail: float
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
    def __init__(
        self, years, channel_key, test_mode, use_bdt, modelname, main_plot_dir, at_inference=False
    ):
        self.channel = CHANNELS[channel_key]
        self.years = years
        self.test_mode = test_mode
        test_dir = "test" if test_mode else "full"
        self.plot_dir = (
            Path(main_plot_dir) / f"plots/SensitivityStudy/{TODAY}/{test_dir}/{channel_key}"
        )
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # TODO we should get rid of this line
        self.sig_key = SIG_KEYS[channel_key]

        self.taukey = CHANNELS[channel_key].tagger_label

        self.events_dict = {year: {} for year in years}

        self.use_bdt = use_bdt
        self.modelname = modelname
        self.at_inference = at_inference

        # TODO This is very hardcoded. Need to fix
        self.model_dir = (
            Path(MAIN_DIR)
            / f"src/bbtautau/postprocessing/classifier/trained_models/{self.modelname}_{('-'.join(self.years) if self.years != hh_vars.years else 'all')}"
        )

    def load_data(self):

        # This could be improved by adding channel-by-channel granularity
        # Now filter just requires that any trigger in that year fires

        for year in self.years:
            filters_dict = trigger_filter(
                HLTs.hlts_list_by_dtype(year),
                year,
                fast_mode=self.test_mode,
                # PNetXbb_cut=0.8 if not self.test_mode else None,
            )  # = {"data": [(...)], "signal": [(...)], ...}

            columns = get_columns(year)

            self.events_dict[year] = load_samples(
                year=year,
                paths=data_paths[year],
                channels=[self.channel],
                filters_dict=filters_dict,
                load_columns=columns,
                load_just_ggf=True,
                restrict_data_to_channel=True,
                loaded_samples=True,
                multithread=True,
            )

            self.events_dict[year] = delete_columns(self.events_dict[year], year, [self.channel])

            derive_variables(
                self.events_dict[year], CHANNELS["hm"]
            )  # legacy issue, muon branches are misnamed
            bbtautau_assignment(self.events_dict[year], agnostic=True)

            if self.use_bdt:
                if self.at_inference:
                    # evaluate bdt at inference time
                    compute_bdt_preds(
                        data=self.events_dict,
                        model_dir=self.model_dir,
                        modelname=self.modelname,
                    )
                    print("BDT predictions computed at inference time")

                else:
                    load_bdt_preds(
                        self.events_dict[year],
                        year,
                        BDT_EVAL_DIR,
                        modelname=self.modelname,
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

        rocAnalyzer = ROCAnalyzer(
            years=years,
            signals={self.sig_key: self.events_dict_allyears[self.sig_key]},
            backgrounds={bkg: self.events_dict_allyears[bkg] for bkg in background_names},
        )

        discs_tt = [
            f"ttFatJetParTX{self.taukey}vsQCD",
            f"ttFatJetParTX{self.taukey}vsQCDTop",
        ]

        discs_bb = ["bbFatJetParTXbbvsQCD", "bbFatJetParTXbbvsQCDTop", "bbFatJetPNetXbbvsQCDLegacy"]

        if self.use_bdt:
            discs_tt += [f"BDTScore{self.taukey}vsQCD", f"BDTScore{self.taukey}vsAll"]

        discs_all = discs or discs_bb + discs_tt

        rocAnalyzer.fill_discriminants(
            discs_all, signal_name=self.sig_key, background_names=background_names
        )

        rocAnalyzer.compute_rocs()

        # Plot bb
        rocAnalyzer.plot_rocs(title="bbFatJet", disc_names=discs_bb, plot_dir=self.plot_dir)

        # Plot tt
        rocAnalyzer.plot_rocs(title="ttFatJet", disc_names=discs_tt, plot_dir=self.plot_dir)

        for disc in discs_all:
            rocAnalyzer.plot_disc_scores(disc, background_names, self.plot_dir)
            rocAnalyzer.plot_disc_scores(disc, [[bkg] for bkg in background_names], self.plot_dir)
            # rocAnalyzer.compute_confusion_matrix(disc, plot_dir=self.plot_dir)

        print(f"ROCs computed and plotted for years {years}.")

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
                zip(["bb", "tt"], ["bb FatJet", rf"{self.channel.label} FatJet"])
            ):
                ax = axs[i]
                if key == "hhbbtt":
                    mask = np.concatenate(
                        [self.events_dict[year][self.sig_key].get_mask(jet) for year in years],
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
                    self.events_dict[year][key].get_var("ak8FatJetParTXbbvsQCD"),
                    self.events_dict[year][key].get_mask("bb"),
                )
                if self.use_bdt:
                    # BDT is evaluated directly on the tagged jet
                    self.txtts[year][key] = self.events_dict[year][key].get_var(
                        f"BDTScore{self.taukey}vsAll"
                    )
                else:
                    self.txtts[year][key] = self.get_jet_vals(
                        self.events_dict[year][key].get_var(f"ak8FatJetParTX{self.taukey}vsQCDTop"),
                        self.events_dict[year][key].get_mask("tt"),
                    )
                self.masstt[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].get_var(f"ak8FatJet{mttk}"),
                    self.events_dict[year][key].get_mask("tt"),
                )
                self.massbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].get_var(f"ak8FatJet{mbbk}"),
                    self.events_dict[year][key].get_mask("bb"),
                )
                self.ptbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key].get_var("ak8FatJetPt"),
                    self.events_dict[year][key].get_mask("bb"),
                )

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
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    if not self.use_bdt:
                        cut &= (self.masstt[year][key] > mtt1) & (self.masstt[year][key] < mtt2)

                    sig_pass += np.sum(self.events_dict[year][key].events["finalWeight"][cut])
                else:  # compute background
                    cut_bg_pass = (
                        (self.txbbs[year][key] > txbbcut)
                        & (self.txtts[year][key] > txttcut)
                        & (self.ptbb[year][key] > 250)
                    )
                    if not self.use_bdt:
                        cut_bg_pass &= (self.masstt[year][key] > mtt1) & (
                            self.masstt[year][key] < mtt2
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
                        (self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut)
                    ) & (self.ptbb[year][key] > 250)
                    if not self.use_bdt:
                        cut_bg_fail &= (self.masstt[year][key] > mtt1) & (
                            self.masstt[year][key] < mtt2
                        )

                    bg_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail & msb1]
                    )
                    bg_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_bg_fail & msb2]
                    )
                    cut_sig_fail = (
                        ((self.txbbs[year][key] < txbbcut) | (self.txtts[year][key] < txttcut))
                        & (self.massbb[year][key] > mbb1)
                        & (self.massbb[year][key] < mbb2)
                        & (self.ptbb[year][key] > 250)
                    )
                    if not self.use_bdt:
                        cut_sig_fail &= (self.masstt[year][key] > mtt1) & (
                            self.masstt[year][key] < mtt2
                        )

                    sig_fail += np.sum(
                        self.events_dict[year][key].events["finalWeight"][cut_sig_fail]
                    )

        # signal, B, C, D, TF = C/D
        return sig_pass, bg_pass, sig_fail, bg_fail, sig_fail / bg_fail

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
            results = Parallel(n_jobs=-4, verbose=1)(
                delayed(sig_bg)(_b, _t) for _b, _t in zip(bbcut_flat, ttcut_flat)
            )
            # results is a list of (sig, bkg, tf) tuples

            sigs, bgs, sig_fails, bg_fails, tfs = zip(*results)
            sigs = np.array(sigs).reshape(BBcut.shape)
            bgs = np.array(bgs).reshape(BBcut.shape)
            sig_fails = np.array(sig_fails).reshape(BBcut.shape)
            bg_fails = np.array(bg_fails).reshape(BBcut.shape)
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
                f"BDTScore{self.taukey} vs All" if self.use_bdt else f"X{self.taukey} vs QCDTop cut"
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
                "sig_bkg_opt" + f"_BDT_{self.modelname}" if self.use_bdt else "sig_bkg_opt"
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
                hmass_fail=sig_fails[max_sig_idx],
                sideband_fail=bg_fails[max_sig_idx],
                transfer_factor=tfs[max_sig_idx],
                cuts=(bbcut_opt, ttcut_opt),
            ),
            best_lims=Optimum(
                signal_yield=sigs[max_significance_i],
                bkg_yield=bgs[max_significance_i],
                hmass_fail=sig_fails[max_significance_i],
                sideband_fail=bg_fails[max_significance_i],
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
        hmass_fail = optimum.hmass_fail
        sideband_fail = optimum.sideband_fail
        tf = optimum.transfer_factor

        limits = {}
        limits["Label"] = label
        limits["Cut_Xbb"] = cut_bb
        limits["Cut_Xtt"] = cut_tt
        limits["Sig_Yield"] = sig_yield
        limits["Sideband Pass"] = bg_yield
        limits["Higgs Mass Fail"] = hmass_fail
        limits["Sideband Fail"] = sideband_fail
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
    main_plot_dir,
    b_vals,
    use_abcd=True,
    actions=None,
    at_inference=False,
):

    print(f"Processing channel: {channel}. Test mode: {test_mode}.")
    analyser = Analyser(years, channel, test_mode, use_bdt, modelname, main_plot_dir, at_inference)

    analyser.load_data()

    if actions is None:
        actions = []

    if "compute_rocs" in actions:
        analyser.compute_and_plot_rocs(years)
    if "plot_mass" in actions:
        analyser.plot_mass(years)
    if "sensitivity" in actions:
        analyser.prepare_sensitivity(years)
        results = {}
        for B_max in b_vals:
            result = analyser.sig_bkg_opt(
                years,
                gridlims=(0.8, 1),
                gridsize=5 if test_mode else 50,
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
        "--use_bdt", action="store_true", default=False, help="Use BDT for sensitivity study"
    )
    parser.add_argument(
        "--modelname",
        default="28May25_baseline",
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
        choices=["compute_rocs", "plot_mass", "sensitivity", "time-methods"],
        required=True,
        help="Actions to perform. Choose one or more.",
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
        default="/home/users/lumori/bbtautau/src/bbtautau//postprocessing/classifier/trained_models/28May25_baseline_all",
        type=str,
    )

    parser.add_argument(
        "--b-vals",
        nargs="+",
        type=int,
        default=[5],
        help="Each value n runs an experiment by setting n as max bkg. events when maximizing the signal yield (and not always minimizing the limit) (default: [5]). Used to have an idea of the tagger efficiency when constraining the background yield.",
    )

    # TODO: see what to do for this options
    # parser.add_argument(
    #     "--use-abcd", action="store_false", default=True,
    #     help="Use ABCD background estimation"
    # )

    args = parser.parse_args()

    # check: test-mode and use-bdt are mutually exclusive
    if args.test_mode and args.use_bdt and not args.at_inference:
        raise ValueError("Test mode and use-bdt are mutually exclusive unless at-inference is True")

    for channel in args.channels:
        analyse_channel(
            years=args.years,
            channel=channel,
            test_mode=args.test_mode,
            use_bdt=args.use_bdt,
            modelname=args.modelname,
            main_plot_dir=args.plot_dir,
            use_abcd=True,  # temporary. will need to generalize framework
            actions=args.actions,
            b_vals=args.b_vals,
            at_inference=args.at_inference,
        )
