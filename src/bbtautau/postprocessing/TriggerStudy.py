from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import utils
from boostedhh.utils import HLT, PAD_VAL
from Samples import CHANNELS, SAMPLES, SIGNALS

from bbtautau.HLTs import HLTs

MAIN_DIR = Path("/home/users/lumori/bbtautau/")

exclude = {"highptMu": False}


data_paths = {
    "2022": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2022EE": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2023": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2023BPix": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
}

"""
Objectives:
- trigger study with all sig channels
- change trigger choice, add met
- correlation between triggers (subtract 1 at the time)
"""


class Analyser:
    """
    Process signal data and perform trigger study for (year, sig_key). sig_key in SIGNALS
    """

    def __init__(self, year, sig_key, test_mode=False):
        assert sig_key in SIGNALS, f"sig_key {sig_key} not in SIGNALS"
        self.sample = SAMPLES[sig_key]
        self.year = year
        self.test_mode = test_mode
        self.sig_path = data_paths[year]
        self.plot_dir = MAIN_DIR / f"plots/TriggerStudy/25Apr16/{year}/{sig_key}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.events_dict = {}

    def load_data(self):
        self.events_dict = utils.load_sample(self.sample, self.year, self.sig_path)
        print(f"Loaded {self.sample.label} for year {self.year} from {self.sig_path['signal']}")

    def _empty_mask(self) -> np.ndarray:
        """Return a False-filled Boolean vector, length = #events. Take as reference a generic trigger column"""
        any_col = self.events_dict["HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1"]
        return np.zeros(len(any_col), dtype=bool)

    def _trigger_mask(self, hlt: HLT | str) -> np.ndarray | None:
        """
        Output boolean vector for a single trigger **or** None if it is
        - not defined for this analysis year, or
        - not present in events_dict.

        Parameters
        ----------
        hlt : HLT or str (will be converted to HLT)
        """

        if isinstance(hlt, str):
            hlt = HLTs.get_hlt(hlt)

        if not isinstance(hlt, HLT):
            raise TypeError(f"Expected HLT or str, got {type(hlt)}")

        if self.year not in hlt.mc_years:
            # print(f"Trigger {hlt.name} not defined in MC for year {self.year}. Skipping.")
            return None

        name = hlt.get_name(True)
        col = self.events_dict.get(name)  # None if not found
        if col is None:
            print(f"Trigger {name} not found in events_dict")
            return None

        return col.to_numpy(dtype=bool).ravel()

    def _class_mask(self, cl: str) -> np.ndarray:
        """
        Return a Boolean numpy array (`shape = (#events,)`) that is True for
        every event that fired **any** trigger in *cl*.

        Parameters
        ----------
        classes : str
            Names of a single trigger class (e.g. 'pnet', 'met' etc.)
        """

        mask = self._empty_mask()
        for hlt in HLTs.HLTs[cl]:
            m = self._trigger_mask(hlt)
            if m is not None:
                mask |= m

        return mask

    def fired_events_by_trs(self, triggers: str | Iterable[str]) -> np.ndarray:
        """
        Return a Boolean numpy array (`shape = (#events,)`) that is True for
        every event that fired any trigger in *triggers*.

        Parameters
        ----------
        triggers : str or iterable[str]
            Names of a single trigger or a list / tuple / set of triggers.
        """
        if not isinstance(triggers, Iterable) or isinstance(triggers, str):
            triggers = (triggers,)

        mask = self._empty_mask()
        for tr in triggers:
            m = self._trigger_mask(tr)
            if m is not None:
                mask |= m
        return mask

    def fired_events_by_class(self, classes: str | Iterable[str]) -> np.ndarray:
        """
        Return a Boolean numpy array (`shape = (#events,)`) that is True for
        every event that fired any trigger in any class in classes.

        Parameters
        ----------
        triggers : str or iterable[str]
            Names of a single trigger or a list / tuple / set of triggers.

        """
        if not isinstance(classes, Iterable) or isinstance(classes, str):
            classes = (classes,)

        mask = self._empty_mask()
        for cl in classes:
            m = self._class_mask(cl)
            if m is not None:
                mask |= m

        return mask

    def set_plot_dict(self):

        plot_dict = {
            "hh": {
                "mask": self.events_dict["GenTauhh"][0],
                "label": r"$\tau_h\tau_h$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "pfjet", "quadjet", "singletau", "ditau", "met"]
                    ),
                    "PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    ),
                    "PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    ),
                    "PNet | SingleTau | Di-tau": self.fired_events_by_class(
                        ["pnet", "singletau", "ditau"]
                    ),
                    "PNet | PFJet | Quad-jet": self.fired_events_by_class(
                        ["pnet", "pfjet", "quadjet"]
                    ),
                    "Quad-jet": self.fired_events_by_class("quadjet"),
                    "PNet": self.fired_events_by_class("pnet"),
                    "PFJet": self.fired_events_by_class("pfjet"),
                    "MET": self.fired_events_by_class("met"),
                    # "PNet | PFJet | Quad-jet | SingleTau": self.fired_events_by_class(
                    #     ["pnet", "pfjet", "quadjet", "singletau"]
                    # ),
                    # "PNet | PFJet | Quad-jet | Di-tau": self.fired_events_by_class(
                    #     ["pnet", "pfjet", "quadjet", "ditau"]
                    # ),
                    # "PNet | PFJet | SingleTau | Di-tau": self.fired_events_by_class(
                    #     ["pnet", "pfjet", "singletau", "ditau"]
                    # ),
                },
            },
            "hh_minus": {
                "mask": self.events_dict["GenTauhh"][0],
                "label": r"$\tau_h\tau_h$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "pfjet", "quadjet", "met", "singletau", "ditau"]
                    ),
                    "-PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    )
                    | self.fired_events_by_class(["pfjet", "met", "quadjet", "singletau", "ditau"]),
                    "-PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    )
                    | self.fired_events_by_class(["pfjet", "quadjet", "met", "singletau", "ditau"]),
                    "-PFJet": self.fired_events_by_class(
                        ["pnet", "met", "quadjet", "singletau", "ditau"]
                    ),
                    "-Quad-jet": self.fired_events_by_class(
                        ["pnet", "met", "pfjet", "singletau", "ditau"]
                    ),
                    "-SingleTau": self.fired_events_by_class(
                        ["pnet", "met", "pfjet", "quadjet", "ditau"]
                    ),
                    "-Di-tau": self.fired_events_by_class(
                        ["pnet", "met", "pfjet", "quadjet", "singletau"]
                    ),
                    "-MET": self.fired_events_by_class(
                        ["pnet", "ditau", "pfjet", "quadjet", "singletau"]
                    ),
                },
            },
            "hm": {
                "mask": self.events_dict["GenTauhm"][0],
                "label": r"$\tau_h\mu$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "muon", "met", "muontau", "singletau", "ditau", "pfjet"]
                    ),
                    "PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    ),
                    "PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    ),
                    "PNetBB | TauTau": self.fired_events_by_class("pnet"),
                    "Muon": self.fired_events_by_class("muon"),
                    "Mu-tau": self.fired_events_by_class("muontau"),
                    "SingleTau": self.fired_events_by_class("singletau"),
                    "Di-tau": self.fired_events_by_class("ditau"),
                    "PFJet": self.fired_events_by_class("pfjet"),
                    "MET": self.fired_events_by_class("met"),
                    "HLT_IsoMu24": self.fired_events_by_trs("HLT_IsoMu24"),
                },
            },
            "hm_minus": {
                "mask": self.events_dict["GenTauhm"][0],
                "label": r"$\tau_h\mu$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "muon", "muontau", "singletau", "ditau", "met", "pfjet"]
                    ),
                    "-PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    )
                    | self.fired_events_by_class(
                        ["muon", "muontau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    )
                    | self.fired_events_by_class(
                        ["muon", "muontau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-Muon": self.fired_events_by_class(
                        ["pnet", "muontau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-Mu-tau": self.fired_events_by_class(
                        ["pnet", "muon", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-SingleTau": self.fired_events_by_class(
                        ["pnet", "muon", "muontau", "met", "ditau", "pfjet"]
                    ),
                    "-Di-tau": self.fired_events_by_class(
                        ["pnet", "muon", "muontau", "met", "singletau", "pfjet"]
                    ),
                    "-PFJet": self.fired_events_by_class(
                        ["pnet", "muon", "met", "muontau", "singletau", "ditau"]
                    ),
                    "-MET": self.fired_events_by_class(
                        ["pnet", "muon", "pfjet", "muontau", "singletau", "ditau"]
                    ),
                },
            },
            "he": {
                "mask": self.events_dict["GenTauhe"][0],
                "label": r"$\tau_he$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "met", "singletau", "ditau", "pfjet"]
                    ),
                    "PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    ),
                    "PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    ),
                    "PNetBB | TauTau": self.fired_events_by_class("pnet"),
                    "EGamma": self.fired_events_by_class("egamma"),
                    "e-tau": self.fired_events_by_class("etau"),
                    "SingleTau": self.fired_events_by_class("singletau"),
                    "Di-tau": self.fired_events_by_class("ditau"),
                    "MET": self.fired_events_by_class("met"),
                    "HLT_Ele30_WPTight_Gsf": self.fired_events_by_trs("HLT_Ele30_WPTight_Gsf"),
                },
            },
            "he_minus": {
                "mask": self.events_dict["GenTauhe"][0],
                "label": r"$\tau_he$",
                "triggers": {
                    "All": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-PNetBB": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        ]
                    )
                    | self.fired_events_by_class(
                        ["egamma", "etau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-PNetTauTau": self.fired_events_by_trs(
                        [
                            "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                            "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        ]
                    )
                    | self.fired_events_by_class(
                        ["egamma", "etau", "met", "singletau", "ditau", "pfjet"]
                    ),
                    "-EGamma": self.fired_events_by_class(
                        ["pnet", "etau", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-e-tau": self.fired_events_by_class(
                        ["pnet", "egamma", "singletau", "met", "ditau", "pfjet"]
                    ),
                    "-SingleTau": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "met", "ditau", "pfjet"]
                    ),
                    "-Di-tau": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "met", "singletau", "pfjet"]
                    ),
                    "-PFJet": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "met", "singletau", "ditau"]
                    ),
                    "-MET": self.fired_events_by_class(
                        ["pnet", "egamma", "etau", "singletau", "ditau", "pfjet"]
                    ),
                },
            },
        }

        if year in ["2023", "2023BPix"]:
            plot_dict["hh"]["triggers"].update(
                {
                    "All": self.fired_events_by_class(
                        ["pnet", "pfjet", "quadjet", "singletau", "ditau", "met", "parking"]
                    ),
                    "PNet | Parking ": self.fired_events_by_class(["pnet", "parking"]),
                    "Parking Quad-jet": self.fired_events_by_class("parking"),
                }
            )

            for key in plot_dict["hh_minus"]["triggers"]:
                plot_dict["hh_minus"]["triggers"][key] = plot_dict["hh_minus"]["triggers"][
                    key
                ] | self.fired_events_by_class("parking")

            plot_dict["hh_minus"]["triggers"].update(
                {
                    "-Parking Quad-jet": self.fired_events_by_class(
                        ["pnet", "pfjet", "quadjet", "met", "singletau", "ditau"]
                    ),
                }
            )
        self.plot_dict = plot_dict

    def set_quantities(self):
        self.weights = self.events_dict["weight"][0]

        higgs = utils.make_vector(self.events_dict, name="GenHiggs")

        try:
            self.mhh = (higgs[:, 0] + higgs[:, 1]).mass
            self.hbbpt = higgs[self.events_dict["GenHiggsChildren"] == 5].pt
            self.httpt = higgs[self.events_dict["GenHiggsChildren"] == 15].pt

        except Exception as e:
            print("Error in set_quantities", e)
            self.mhh = np.zeros_like(self.weights)
            self.hbbpt = np.zeros_like(self.weights)
            self.httpt = np.zeros_like(self.weights)

    def plot_channel(self, save=True):

        plt.rcParams.update({"font.size": 14})

        plot_vars = [
            (self.mhh, "mhh", r"$m_{HH}$ [GeV]", np.linspace(250, 1500, 30)),
            (self.hbbpt, "hbbpt", r"Hbb $p_{T}$ [GeV]", np.linspace(230, 500, 20)),
            (self.httpt, "httpt", r"H$\tau\tau$ $p_{T}$ [GeV]", np.linspace(230, 500, 20)),
        ]

        Nplots = len(self.plot_dict) * 3
        i = 0

        for cat, vals in self.plot_dict.items():
            for kinvar, kinname, kinlabel, bins in plot_vars:
                print(f"\rPlotting ({i+1}/{Nplots})", end="")
                i += 1

                (mask, label, triggers) = vals.values()
                mask = mask.to_numpy()

                fig, (ax, rax) = plt.subplots(
                    2,
                    1,
                    figsize=(9, 11),
                    gridspec_kw={"height_ratios": [4, 1], "hspace": 0.07},
                    sharex=True,
                )

                hists = {
                    "Preselection": np.histogram(
                        kinvar[mask], bins=bins, weights=self.weights[mask]
                    )
                }
                ratios = {}

                hep.histplot(
                    hists["Preselection"],
                    yerr=False,
                    label="Preselection",
                    ax=ax,
                )

                colours = plt.cm.tab20.colors[1:]

                for key, c in zip(triggers.keys(), colours):
                    hists[key] = np.histogram(
                        kinvar[mask & triggers[key]],
                        bins=bins,
                        weights=self.weights[mask & triggers[key]],
                    )
                    ratios[key] = hists[key][0] / hists["Preselection"][0]

                    hep.histplot(
                        hists[key],
                        yerr=False,
                        label=key,
                        ax=ax,
                        color=c,
                    )

                    hep.histplot(
                        (ratios[key], bins),
                        yerr=False,
                        label=key,
                        ax=rax,
                        histtype="errorbar",
                        color=c,
                        # markersize=20,
                        linestyle="--",
                        # markeredgewidth=3,
                        # fillstyle='none',
                    )

                ax.set_ylabel("Events [A.U.]")
                ax.legend()
                ax.set_title(self.sample.label + " " + label)
                ax.set_xlim(bins[0], bins[-1])
                ax.set_ylim(0)

                # rax.legend()
                rax.grid(axis="y")
                rax.set_xlabel(kinlabel)
                rax.set_ylabel("Triggered / Preselection")

                ylims = [0.5, 1] if (cat.endswith("minus") and kinname != "mhh") else [0, 1]
                rax.set_ylim(ylims)

                hep.cms.label(ax=ax, data=False, year=year, com="13.6")

                if save:
                    plt.savefig(self.plot_dir / f"{kinname}_{cat}.pdf", bbox_inches="tight")
                    plt.savefig(self.plot_dir / f"{kinname}_{cat}.png", bbox_inches="tight")

                plt.show()
                plt.close()

        print("\n")

    def define_taggers(self):
        tvars = {}
        qcdouts = ["QCD0HF", "QCD1HF", "QCD2HF"]  # HF = heavy flavor = {c,b}
        topouts = ["TopW", "TopbW"]  # "TopbWev", "TopbWmv", "TopbWtauhv", "TopbWq", "TopbWqq"]
        tvars["PQCD"] = sum([self.events_dict[f"ak8FatJetParT{k}"] for k in qcdouts]).to_numpy()
        tvars["PTop"] = sum([self.events_dict[f"ak8FatJetParT{k}"] for k in topouts]).to_numpy()
        tvars["XbbvsQCD"] = np.nan_to_num(
            self.events_dict["ak8FatJetParTXbb"]
            / (self.events_dict["ak8FatJetParTXbb"] + tvars["PQCD"]),
            nan=PAD_VAL,
        )
        tvars["XbbvsQCDTop"] = np.nan_to_num(
            self.events_dict["ak8FatJetParTXbb"]
            / (self.events_dict["ak8FatJetParTXbb"] + tvars["PQCD"] + tvars["PTop"]),
            nan=PAD_VAL,
        )
        self.tvars = tvars

    def N1_efficiency_table(self, save=True):
        boostedsels = {
            "1 boosted jet (> 250)": self.events_dict["ak8FatJetPt"][0] > 250,
            "2 boosted jets (> 250)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 250),
            "2 boosted jets (>250, >230)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 230),
            "2 boosted jets (>250, >200)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 200),
            "2 boosted jets (>250, >200), XbbvsQCD > 0.95": (
                self.events_dict["ak8FatJetPt"][0] > 250
            )
            & (self.events_dict["ak8FatJetPt"][1] > 200)
            & (self.tvars["XbbvsQCD"] > 0.95).any(axis=1),
            "2 boosted jets (>250, >200), XbbvsQCDTop > 0.95": (
                self.events_dict["ak8FatJetPt"][0] > 250
            )
            & (self.events_dict["ak8FatJetPt"][1] > 200)
            & (self.tvars["XbbvsQCDTop"] > 0.95).any(axis=1),
        }

        for ch in CHANNELS:
            print("\n", ch, "\n")
            trig_table = pd.DataFrame(index=list(boostedsels.keys()))
            mask = self.plot_dict[ch]["mask"]

            for tkey, tsel in self.plot_dict[f"{ch}_minus"]["triggers"].items():
                effs = []
                for sel in boostedsels.values():
                    eff = np.sum(mask & sel & tsel) / np.sum(mask & sel)
                    effs.append(f"{eff * 100:.1f}")

                ttkey = tkey.replace("- ", "-") if tkey.startswith("-") else "All"
                trig_table[ttkey] = effs

            sel_effs = []
            for sel in boostedsels.values():
                eff = np.sum(mask & sel) / np.sum(mask)
                sel_effs.append(f"{eff * 100:.1f}")
            trig_table["Preselection"] = sel_effs

            if save:
                trig_table.to_csv(self.plot_dir / f"trig_effs_{ch}.csv")
            print(trig_table.to_markdown(index=True))

    def N2_efficiency_table(self, save=True, normalize_rows=False):
        """
        Compute and print/save N-2 (double trigger) efficiencies for each boosted selection and channel.
        For each pair of triggers, computes the efficiency of passing both triggers given the boosted selection.
        """
        boostedsels = {
            "2 boosted jets (>250, >200)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 200),
            "2 boosted jets (>250, >200), XbbvsQCD > 0.95": (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & (self.tvars["XbbvsQCD"] > 0.95).any(axis=1)
            ),
            "2 boosted jets (>250, >200), XbbvsQCDTop > 0.95": (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & (self.tvars["XbbvsQCDTop"] > 0.95).any(axis=1)
            ),
        }

        for ch in CHANNELS:
            print(f"\nChannel: {ch}\n")
            mask = self.plot_dict[ch]["mask"]
            trig_cls = self.plot_dict[f"{ch}_minus"]["triggers"]
            trig_keys = list(trig_cls.keys())

            for bkey, sel in boostedsels.items():
                trig_table = pd.DataFrame(index=trig_keys, columns=trig_keys)
                for tkey1 in trig_keys:
                    tsel1 = trig_cls[tkey1]
                    for tkey2 in trig_keys:
                        tsel2 = trig_cls[tkey2]
                        if tkey1 == tkey2:
                            trig_table.loc[tkey1, tkey2] = "-"
                        else:
                            denom = (
                                np.sum(mask & sel)
                                if not normalize_rows
                                else np.sum(mask & sel & tsel1)
                            )
                            if denom > 0:
                                eff = np.sum(mask & sel & tsel1 & tsel2) / denom
                                trig_table.loc[tkey1, tkey2] = f"{eff * 100:.1f}"
                            else:
                                trig_table.loc[tkey1, tkey2] = "n/a"
                if save:
                    trig_table.to_csv(
                        self.plot_dir
                        / f"trig_N2_effs_{ch}_{bkey.replace(' ', '_')}_{normalize_rows*'norm_rows'}.csv"
                    )
                print(f"\nBoosted Selection: {bkey}")
                print(trig_table.to_markdown(index=True))

    def progressive_trigger_removal(
        self, trs: list[dict[str, str]], save: bool = True, name_tag: str = ""
    ) -> None:
        """
        Compute and print/save the efficiency of progressively removing triggers one by one in sequence.
        Args:
            trigger_classes (List[str]): List of trigger classes to analyze in sequence
            save (bool): Whether to save the results to CSV files
        """
        boostedsels = {
            "1 boosted jet (> 250)": self.events_dict["ak8FatJetPt"][0] > 250,
            "2 boosted jets (>250, >200)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 200),
            "2 boosted jets (>250, >200), XbbvsQCD > 0.95": (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & (self.tvars["XbbvsQCD"] > 0.95).any(axis=1)
            ),
            "2 boosted jets (>250, >200), XbbvsQCDTop > 0.95": (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & (self.tvars["XbbvsQCDTop"] > 0.95).any(axis=1)
            ),
        }
        # same as in plot_dict,
        trcls_by_ch = {
            "hh": ["pnet", "pfjet", "quadjet", "singletau", "ditau", "met"],
            "hm": ["pnet", "muon", "muontau", "singletau", "ditau", "met", "pfjet"],
            "he": ["pnet", "egamma", "etau", "singletau", "met", "ditau", "pfjet"],
        }

        for ch in CHANNELS:
            print(f"\nChannel: {ch}\n")
            mask = self.plot_dict[ch]["mask"]

            # Create DataFrame to store results for all boosted selections
            results = pd.DataFrame(index=["All"] + [f"-{t['show_name']}" for t in trs])

            # Calculate efficiency with all triggers for each boosted selection
            for bkey, sel in boostedsels.items():
                # Get the base efficiency with all triggers
                all_triggers_mask = self.plot_dict[ch]["triggers"]["All"]
                all_triggers_eff = np.sum(mask & sel & all_triggers_mask) / np.sum(mask & sel)

                # Add column for this boosted selection
                results.loc["All", bkey] = f"{all_triggers_eff * 100:.1f}%"

                trs_all = set(HLTs.hlts_by_type(self.year, trcls_by_ch[ch], as_str=True))

                # Progressively remove triggers one by one
                for cl_or_tr in trs:
                    if cl_or_tr["type"] == "cl":
                        # remove all hlts in that class
                        hlts_to_remove = HLTs.hlts_by_type(self.year, cl_or_tr["name"], as_str=True)
                        for hlt in hlts_to_remove:
                            trs_all.discard(hlt)  # Use discard instead of remove to avoid KeyError
                        current_mask = self.fired_events_by_trs(list(trs_all))
                        eff = np.sum(mask & sel & current_mask) / np.sum(mask & sel)
                    elif cl_or_tr["type"] == "tr":
                        # remove single hlt
                        trs_all.discard(cl_or_tr["name"])  # Use discard instead of remove
                        current_mask = self.fired_events_by_trs(list(trs_all))
                        eff = np.sum(mask & sel & current_mask) / np.sum(mask & sel)

                    results.loc[f"-{cl_or_tr['show_name']}", bkey] = f"{eff * 100:.1f}%"

            # Transpose the DataFrame
            results = results.T

            # Add title in cell (0,0)
            results.index.name = f"Trigger Efficiency ({self.sample.label}, {self.year}, {ch})"

            if save:
                results.to_csv(self.plot_dir / f"progressive_removal_{ch}_{name_tag}.csv")

            print(results.to_markdown())

    def trigger_correlation_table(self, channels=CHANNELS):
        """
        Compute and print/save the correlation table between individual triggers for each channel.
        """
        for channel in channels:
            triggers = [hlt.lower() for hlt in CHANNELS[channel].hlt_types]
            masks_by_class = pd.DataFrame({tr: self.fired_events_by_class(tr) for tr in triggers})
            phi_coeff = masks_by_class.corr(method="pearson")
            print(f"\nTrigger phi coefficient table for {channel}:")
            print(phi_coeff.to_markdown())
            phi_coeff.to_csv(self.plot_dir / f"trigger_phi_coefficient_{channel}.csv")


if __name__ == "__main__":
    # Load the data
    years = ["2022", "2022EE", "2023", "2023BPix"]
    for year in years:

        print(f"\n\n\nYEAR {year}\n\n\n")
        for sig_key in SIGNALS:
            print("sig_key : ", sig_key)
            analyser = Analyser(year, sig_key)
            analyser.load_data()
            analyser.define_taggers()
            analyser.set_plot_dict()
            analyser.set_quantities()
            analyser.define_taggers()

            analyser.plot_channel(save=True)
            analyser.N1_efficiency_table(save=True)

            if year == "2023BPix":
                remove_trs = [
                    {"type": "cl", "name": "parking", "show_name": "Parking"},
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        "show_name": "PNetTauTau ",
                    },
                    {"type": "cl", "name": "quadjet", "show_name": "QuadJet"},
                    {"type": "cl", "name": "pfjet", "show_name": "PFJet"},
                ]
                analyser.progressive_trigger_removal(remove_trs, name_tag="", save=True)
            if year == "2022":
                remove_trs = [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                        "show_name": "QuadJet70_50_40_35 ",
                    },
                    {"type": "cl", "name": "quadjet", "show_name": "QuadJet"},
                    {"type": "cl", "name": "pfjet", "show_name": "PFJet"},
                ]
                analyser.progressive_trigger_removal(remove_trs, name_tag="", save=True)

            # Some extra studies to keep in mind
            # analyser.N2_efficiency_table(save=True)
            # analyser.N2_efficiency_table(save=True,normalize_rows=True)
            # analyser.trigger_correlation_table()
