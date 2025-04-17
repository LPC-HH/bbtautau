from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import postprocessing
from boostedhh import hh_vars, plotting
from boostedhh.utils import PAD_VAL
from matplotlib.lines import Line2D
from Samples import CHANNELS, qcdouts, topouts
from sklearn.metrics import roc_curve

from bbtautau.HLTs import HLTs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")
logger.setLevel(logging.DEBUG)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

# Global variables
MAIN_DIR = Path("/home/users/lumori/bbtautau/")
SIG_KEYS = {"hh": "bbtthh", "he": "bbtthe", "hm": "bbtthm"}  # We should get rid of this

data_paths = {
    "2022": {
        "data": Path(
            "/ceph/cms/store/user/rkansal/bbtautau/skimmer/24Nov21ParTMass_v12_private_signal/"
        ),
        "signal": Path(
            "/ceph/cms/store/user/rkansal/bbtautau/skimmer/24Nov21ParTMass_v12_private_signal/"
        ),
    },
    "2022EE": {
        "data": Path(
            "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Jan22AddYears_v12_private_signal/"
        ),
        "signal": Path(
            "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Jan22AddYears_v12_private_signal/"
        ),
    },
    "2023": {
        "data": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Mar7Signal_v12_private_signal/"
        ),
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Mar7Signal_v12_private_signal/"
        ),
    },
    "2023BPix": {
        "data": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Mar7Signal_v12_private_signal/"
        ),
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Mar7Signal_v12_private_signal/"
        ),
    },
}


class Analyser:
    def __init__(self, years, channel_key, test_mode=False):
        self.channel = CHANNELS[channel_key]
        self.years = years
        self.test_mode = test_mode
        self.plot_dir = MAIN_DIR / f"plots/SensitivityStudy/25Mar7{channel_key}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # we should get rid of these two lines
        self.sig_key = SIG_KEYS[channel_key]
        self.taukey = {"hh": "Xtauhtauh", "he": "Xtauhtaue", "hm": "Xtauhtaum"}[channel_key]

        self.events_dict = {year: {} for year in years}

    def load_year(self, year):

        # This could be improved by adding channel-by-channel granularity
        # Now filter just requires that any trigger in that year fires
        filters_dict = postprocessing.trigger_filter(
            HLTs.hlts_list_by_dtype(year), fast_mode=self.test_mode
        )  # = {"data": [(...)], "signal": [(...)], ...}
        columns = postprocessing.get_columns(year, self.channel)

        self.events_dict[year] = postprocessing.load_samples(
            year,
            self.channel,
            data_paths[year],
            filters_dict=filters_dict,
            load_columns=columns,
            load_just_bbtt=True,
        )
        self.events_dict[year] = postprocessing.apply_triggers(
            self.events_dict[year], year, self.channel
        )
        self.events_dict[year] = postprocessing.delete_columns(
            self.events_dict[year], year, self.channel
        )
        return

    def build_tagger_dict(self):

        # print(self.events_dict)

        self.taggers_dict = {year: {} for year in self.years}
        for year in self.years:
            for key, events in self.events_dict[year].items():
                tvars = {}

                tvars["PQCD"] = sum([events[f"ak8FatJetParT{k}"] for k in qcdouts]).to_numpy()
                tvars["PTop"] = sum([events[f"ak8FatJetParT{k}"] for k in topouts]).to_numpy()

                for disc in ["Xbb", self.taukey]:
                    tvars[f"{disc}vsQCD"] = np.nan_to_num(
                        events[f"ak8FatJetParT{disc}"]
                        / (events[f"ak8FatJetParT{disc}"] + tvars["PQCD"]),
                        nan=PAD_VAL,
                    )
                    tvars[f"{disc}vsQCDTop"] = np.nan_to_num(
                        events[f"ak8FatJetParT{disc}"]
                        / (events[f"ak8FatJetParT{disc}"] + tvars["PQCD"] + tvars["PTop"]),
                        nan=PAD_VAL,
                    )

                    # make sure not to choose padded jets below by accident
                    nojet3 = events["ak8FatJetPt"][2] == PAD_VAL
                    tvars[f"{disc}vsQCD"][:, 2][nojet3] = PAD_VAL
                    tvars[f"{disc}vsQCDTop"][:, 2][nojet3] = PAD_VAL

                tvars["PNetXbbvsQCD"] = np.nan_to_num(
                    events["ak8FatJetPNetXbbLegacy"]
                    / (events["ak8FatJetPNetXbbLegacy"] + events["ak8FatJetPNetQCDLegacy"]),
                    nan=PAD_VAL,
                )

                # jet assignment
                fjbbpick = np.argmax(tvars["XbbvsQCD"], axis=1)
                fjttpick = np.argmax(tvars[f"{self.taukey}vsQCD"], axis=1)
                overlap = fjbbpick == fjttpick
                fjbbpick[overlap] = np.argsort(tvars["XbbvsQCD"][overlap], axis=1)[:, 1]

                # convert ids to boolean masks
                fjbbpick_mask = np.zeros_like(tvars["XbbvsQCD"], dtype=bool)
                fjbbpick_mask[np.arange(len(fjbbpick)), fjbbpick] = True
                fjttpick_mask = np.zeros_like(tvars[f"{self.taukey}vsQCD"], dtype=bool)
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
                f"{self.taukey}vsQCD",
                f"{self.taukey}vsQCDTop",
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
                        self.events_dict[year][key]["finalWeight"]
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
                    [self.events_dict[year][self.sig_key]["finalWeight"] for year in years]
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

    def plot_rocs(self, years, test_mode=False):
        if not hasattr(self, "rocs") or "_".join(years) not in self.rocs:
            print(f"No ROC curves computed yet in years {years}")
        for jet, title in zip(["bb", "tautau"], ["bb FatJet", rf"{self.channel.label}$ FatJet"]):
            plotting.multiROCCurveGrey(
                {"": self.rocs["_".join(years)][jet]},
                title=title + "+".join(years),
                show=True,
                plot_dir=self.plot_dir,
                name=f"roc_{jet+'_'.join(years)+test_mode*'_fast'}.pdf",
            )

    # here could block and save only data that needs after

    def prepare_sensitivity(self, years):
        if set(years) != set(self.years):
            raise ValueError(f"Years {years} not in {self.years}")

        mbbk = "ParTmassResApplied"
        mttk = "ParTmassResApplied"

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
                self.txtts[year][key] = self.get_jet_vals(
                    self.taggers_dict[year][key][f"{self.taukey}vsQCDTop"],
                    self.taggers_dict[year][key]["tautau_mask"],
                )
                self.masstt[year][key] = self.get_jet_vals(
                    self.events_dict[year][key][f"ak8FatJet{mttk}"],
                    self.taggers_dict[year][key]["tautau_mask"],
                )
                self.massbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key][f"ak8FatJet{mbbk}"],
                    self.taggers_dict[year][key]["bb_mask"],
                )
                self.ptbb[year][key] = self.get_jet_vals(
                    self.events_dict[year][key]["ak8FatJetPt"],
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
                    sig_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut])
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
                    bg_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut & msb1])
                    bg_yield += np.sum(self.events_dict[year][key]["finalWeight"][cut & msb2])
        return sig_yield, bg_yield

    def sig_bkg_opt(
        self, years, gridsize=10, gridlims=(0.7, 1), B=1, normalize_sig=True, plot=False
    ):
        """
        Will have to improve definition of global params
        """

        # bbeff, tteff = 0.44,0.36 #0.44, 0.36 values determined by highest sig for 1 bkg event
        mbb1, mbb2 = 110.0, 160.0
        mbbw2 = (mbb2 - mbb1) / 2
        mtt1, mtt2 = 50, 150

        bbcut = np.linspace(*gridlims, gridsize)
        ttcut = np.linspace(*gridlims, gridsize)

        BBcut, TTcut = np.meshgrid(bbcut, ttcut)

        # scalar function, must be vectorized
        def sig_bg(bbcut, ttcut):
            return self.compute_sig_bg(
                years=years,
                txbbcut=bbcut,
                txttcut=ttcut,
                mbb1=mbb1,
                mbb2=mbb2,
                mbbw2=mbbw2,
                mtt1=mtt1,
                mtt2=mtt2,
            )

        sigs, bgs = np.vectorize(sig_bg)(BBcut, TTcut)
        if normalize_sig:
            tot_sig_weight = np.sum(
                np.concatenate(
                    [self.events_dict[year][self.sig_key]["finalWeight"] for year in years]
                )
            )
            sigs = sigs / tot_sig_weight

        sel = (bgs >= 1) & (bgs <= B)
        if np.sum(sel) == 0:
            B_initial = B
            while np.sum(sel) == 0 and B < 100:
                B += 1
                sel = (bgs >= 1) & (bgs <= B)
            print(
                f"Need a finer grid, no region with B={B_initial}. I'm extending the region to B in [1,{B}].",
                bgs,
            )
        sel_idcs = np.argwhere(sel)
        opt_i = np.argmax(sigs[sel])
        max_sig_idx = tuple(sel_idcs[opt_i])
        bbcut_opt, ttcut_opt = BBcut[max_sig_idx], TTcut[max_sig_idx]

        significance = np.where(bgs > 0, sigs / np.sqrt(bgs), 0)
        max_significance_i = np.unravel_index(np.argmax(significance), significance.shape)
        bbcut_opt_significance, ttcut_opt_significance = (
            BBcut[max_significance_i],
            TTcut[max_significance_i],
        )

        """
        extract from roc data the efficiencies for the cuts:
        """
        # bbeff = rocs[year]["bb"]["XbbvsQCD"]["tpr"][

        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            hep.cms.label(
                ax=ax,
                label="Work in Progress",
                data=True,
                year="+".join(years),
                com="13.6",
                fontsize=13,
                lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
            )
            sigmap = ax.contourf(BBcut, TTcut, sigs, levels=10, cmap="viridis")
            ax.contour(BBcut, TTcut, sel, colors="r")
            proxy = Line2D([0], [0], color="r", label="B=1" if B == 1 else f"B in [1,{B}]")
            ax.scatter(bbcut_opt, ttcut_opt, color="r", label="Max. signal cut")
            # ax.scatter(bbcut_opt_significance, ttcut_opt_significance, color="b", label="Max. significance cut")
            ax.set_xlabel("Xbb vs QCD cut")
            ax.set_ylabel("Xtauhtauh vs QCD cut")
            cbar = plt.colorbar(sigmap, ax=ax)
            cbar.set_label("Signal efficiency" if normalize_sig else "Signal yield")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(proxy)
            ax.legend(handles=handles, loc="lower left")
            plt.savefig(
                self.plot_dir / f"sig_bkg_opt_{'_'.join(years)}_B={B}.pdf", bbox_inches="tight"
            )
            plt.savefig(
                self.plot_dir / f"sig_bkg_opt_{'_'.join(years)}_B={B}.png", bbox_inches="tight"
            )
            plt.show()

        return (
            [sigs[max_sig_idx], bgs[max_sig_idx]],
            [bbcut_opt, ttcut_opt],
            [sigs[max_significance_i], bgs[max_significance_i]],
            [bbcut_opt_significance, ttcut_opt_significance],
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

    def as_df(self, sig_yield, bg_yield, years):
        limits = {}
        limits["Sig_Yield"] = sig_yield
        limits["BG_Yield"] = bg_yield
        limits["Limit"] = 2 * np.sqrt(bg_yield) / sig_yield

        if "2023" not in years and "2023BPix" not in years:
            limits["Limit_scaled_22_23"] = (
                2
                * np.sqrt(bg_yield)
                / sig_yield
                / np.sqrt(
                    hh_vars.LUMI["2022-2023"] / np.sum([hh_vars.LUMI[year] for year in years])
                )
            )
        else:
            limits["Limit_scaled_22_23"] = np.nan

        limits["Limit_scaled_22_24"] = (
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt(
                (124000 + hh_vars.LUMI["2022-2023"])
                / np.sum([hh_vars.LUMI[year] for year in years])
            )
        )

        limits["Limit_scaled_Run3"] = (
            2
            * np.sqrt(bg_yield)
            / sig_yield
            / np.sqrt((360000) / np.sum([hh_vars.LUMI[year] for year in years]))
        )

        df_out = pd.DataFrame([limits])
        return df_out


if __name__ == "__main__":

    years = ["2022"]  # "2022EE", "2023", "2023BPix"]
    test_mode = True  # reduces size of data to run all quickly

    for c in [
        "hh",
        "hm",
        "he",
    ]:
        print(f"Channel: {c}")
        analyser = Analyser(years, c, test_mode=test_mode)
        for year in years:
            analyser.load_year(year)

        analyser.build_tagger_dict()
        analyser.compute_rocs(years)
        analyser.plot_rocs(years, test_mode=test_mode)
        print("ROCs computed for channel ", c)
        analyser.prepare_sensitivity(years)

        results = {}
        for B in [1, 2, 8]:
            yields_B, cuts_B, yields_max_significance, cuts_max_significance = analyser.sig_bkg_opt(
                years, gridsize=30, B=B, plot=True
            )
            sig_yield, bkg_yield = yields_B
            sig_yield_max_sig, bkg_yield_max_sig = (
                yields_max_significance  # not very clean rn, can be improved but should be the same
            )
            results[f"B={B}"] = analyser.as_df(sig_yield, bkg_yield, years)
            print("done with B=", B)
        results["Max_significance"] = analyser.as_df(sig_yield_max_sig, bkg_yield_max_sig, years)
        results_df = pd.concat(results, axis=0)
        results_df.index = results_df.index.droplevel(1)
        print(c, "\n", results_df.T.to_markdown())
        results_df.T.to_csv(
            analyser.plot_dir / f"{'_'.join(years)}-results{'_fast' * test_mode}.csv"
        )
        del analyser
