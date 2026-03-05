from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from boostedhh import utils
from boostedhh.utils import HLT, PAD_VAL

from bbtautau.HLTs import HLTs
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES, SIGNALS
from bbtautau.postprocessing.utils import base_filter, load_samples
from bbtautau.userConfig import DATA_PATHS, PLOT_DIR

"""
Objectives:
- trigger study with all sig channels
- change trigger choice, add met
- correlation between triggers (subtract 1 at the time)
"""


class Analyser:
    """
    Process signal data and perform trigger study for (year, sig_key). sig_key in SIGNALS.
    """

    def __init__(self, year, sig_key, test_mode=False, plot_dir=None):
        assert sig_key in SIGNALS, f"sig_key {sig_key} not in SIGNALS"
        self.sample = SAMPLES[sig_key]
        self.year = year
        self.test_mode = test_mode
        self.base_plot_dir = plot_dir or PLOT_DIR / f"TriggerStudy/{date.today()}/{year}/{sig_key}"
        self.sig_key = sig_key
        self.channel_samples = {}

    def load_data(self):
        """Load signal data for all channels using the standard loading infrastructure."""
        filters_dict = base_filter(self.test_mode)
        all_channels = list(CHANNELS.values())

        self.channel_samples = load_samples(
            year=self.year,
            paths=DATA_PATHS[self.year],
            signals=[self.sig_key],
            channels=all_channels,
            filters_dict=filters_dict,
            loaded_samples=True,
            load_bgs=False,
            load_data=False,
        )
        print(f"Loaded {self.sample.label} for year {self.year}")
        for key, sample in self.channel_samples.items():
            print(f"  {key}: {len(sample.events)} events")

    def set_channel(self, ch_key):
        """Set the current channel for analysis. Must be called before analysis methods."""
        self.ch_key = ch_key
        self.channel = CHANNELS[ch_key]
        sample_key = f"{self.sig_key}{ch_key}"
        self.current_sample = self.channel_samples[sample_key]
        self.events_dict = self.current_sample.events
        self.plot_dir = self.base_plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _empty_mask(self) -> np.ndarray:
        return np.zeros(len(self.events_dict), dtype=bool)

    def _trigger_mask(self, hlt: HLT | str) -> np.ndarray | None:
        """
        Output boolean vector for a single trigger **or** None if it is
        - not defined for this analysis year, or
        - not present in events_dict.
        """
        if isinstance(hlt, str):
            hlt = HLTs.get_hlt(hlt)

        if not isinstance(hlt, HLT):
            raise TypeError(f"Expected HLT or str, got {type(hlt)}")

        if self.year not in hlt.mc_years:
            return None

        name = hlt.get_name(True)
        col = self.events_dict.get(name)
        if col is None:
            print(f"Trigger {name} not found in events_dict")
            return None

        return col.to_numpy(dtype=bool).ravel()

    def _class_mask(self, cl: str) -> np.ndarray:
        """
        Return a Boolean numpy array that is True for every event that fired
        **any** trigger in class *cl*.
        """
        mask = self._empty_mask()
        for hlt in HLTs.HLTs[cl]:
            m = self._trigger_mask(hlt)
            if m is not None:
                mask |= m
        return mask

    def fired_events_by_trs(self, triggers: str | Iterable[str]) -> np.ndarray:
        """
        Return a Boolean numpy array that is True for every event that fired
        any trigger in *triggers*.
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
        Return a Boolean numpy array that is True for every event that fired
        any trigger in any class in *classes*.
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
        ch = self.ch_key
        label = self.channel.label
        n_events = len(self.events_dict)
        all_mask = np.ones(n_events, dtype=bool)

        if ch == "hh":
            plot_dict = {
                "hh": {
                    "mask": all_mask,
                    "label": label,
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
                    },
                },
                "hh_minus": {
                    "mask": all_mask,
                    "label": label,
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
                        | self.fired_events_by_class(
                            ["pfjet", "met", "quadjet", "singletau", "ditau"]
                        ),
                        "-PNetTauTau": self.fired_events_by_trs(
                            [
                                "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                                "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                            ]
                        )
                        | self.fired_events_by_class(
                            ["pfjet", "quadjet", "met", "singletau", "ditau"]
                        ),
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
            }

            if self.year in ["2023", "2023BPix"]:
                plot_dict["hh"]["triggers"].update(
                    {
                        "All": self.fired_events_by_class(
                            [
                                "pnet",
                                "pfjet",
                                "quadjet",
                                "singletau",
                                "ditau",
                                "met",
                                "parking",
                            ]
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

        elif ch == "hm":
            plot_dict = {
                "hm": {
                    "mask": all_mask,
                    "label": label,
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
                    "mask": all_mask,
                    "label": label,
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
            }

        elif ch == "he":
            plot_dict = {
                "he": {
                    "mask": all_mask,
                    "label": label,
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
                    "mask": all_mask,
                    "label": label,
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
        else:
            raise ValueError(f"Unknown channel key: {ch}")

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
        hep.style.use("CMS")

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

                mask = np.asarray(vals["mask"], dtype=bool)
                label = vals["label"]
                triggers = vals["triggers"]

                fig, (ax, rax) = plt.subplots(
                    2,
                    1,
                    figsize=(13, 16),
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
                        linestyle="--",
                    )

                ax.set_ylabel("Events [A.U.]")
                ax.legend()
                ax.set_title(self.sample.label + " " + label)
                ax.set_xlim(bins[0], bins[-1])
                ax.set_ylim(0)

                rax.grid(axis="y")
                rax.set_xlabel(kinlabel)
                rax.set_ylabel("Trig. / Presel.")

                ylims = [0.5, 1] if (cat.endswith("minus") and kinname != "mhh") else [0, 1]
                rax.set_ylim(ylims)

                hep.cms.label(ax=ax, data=False, year=self.year, com="13.6")

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

        ch = self.ch_key
        print("\n", ch, "\n")
        trig_table = pd.DataFrame(index=list(boostedsels.keys()))

        for tkey, tsel in self.plot_dict[f"{ch}_minus"]["triggers"].items():
            effs = []
            for sel in boostedsels.values():
                denom = np.sum(sel)
                eff = np.sum(sel & tsel) / denom if denom > 0 else 0
                effs.append(f"{eff * 100:.1f}")

            ttkey = tkey.replace("- ", "-") if tkey.startswith("-") else "All"
            trig_table[ttkey] = effs

        sel_effs = []
        n_total = len(self.events_dict)
        for sel in boostedsels.values():
            eff = np.sum(sel) / n_total
            sel_effs.append(f"{eff * 100:.1f}")
        trig_table["Preselection"] = sel_effs

        if save:
            trig_table.to_csv(self.plot_dir / f"trig_effs_{ch}.csv")
        print(trig_table.to_markdown(index=True))

    def N2_efficiency_table(self, save=True, normalize_rows=False):
        """
        Compute and print/save N-2 (double trigger) efficiencies for each boosted
        selection and the current channel.
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

        ch = self.ch_key
        print(f"\nChannel: {ch}\n")
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
                        denom = np.sum(sel) if not normalize_rows else np.sum(sel & tsel1)
                        if denom > 0:
                            eff = np.sum(sel & tsel1 & tsel2) / denom
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
        Compute and print/save the efficiency of progressively removing triggers
        one by one in sequence for the current channel.
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

        trcls_by_ch = {
            "hh": ["pnet", "pfjet", "quadjet", "singletau", "ditau", "met"],
            "hm": ["pnet", "muon", "muontau", "singletau", "ditau", "met", "pfjet"],
            "he": ["pnet", "egamma", "etau", "singletau", "met", "ditau", "pfjet"],
        }

        ch = self.ch_key
        print(f"\nChannel: {ch}\n")

        results = pd.DataFrame(index=["All"] + [f"-{t['show_name']}" for t in trs])

        for bkey, sel in boostedsels.items():
            all_triggers_mask = self.plot_dict[ch]["triggers"]["All"]
            denom = np.sum(sel)
            all_triggers_eff = np.sum(sel & all_triggers_mask) / denom if denom > 0 else 0

            results.loc["All", bkey] = f"{all_triggers_eff * 100:.1f}%"

            trs_all = set(HLTs.hlts_by_type(self.year, trcls_by_ch[ch], as_str=True))

            for cl_or_tr in trs:
                if cl_or_tr["type"] == "cl":
                    hlts_to_remove = HLTs.hlts_by_type(self.year, cl_or_tr["name"], as_str=True)
                    for hlt in hlts_to_remove:
                        trs_all.discard(hlt)
                    current_mask = self.fired_events_by_trs(list(trs_all))
                    eff = np.sum(sel & current_mask) / denom if denom > 0 else 0
                elif cl_or_tr["type"] == "tr":
                    trs_all.discard(cl_or_tr["name"])
                    current_mask = self.fired_events_by_trs(list(trs_all))
                    eff = np.sum(sel & current_mask) / denom if denom > 0 else 0

                results.loc[f"-{cl_or_tr['show_name']}", bkey] = f"{eff * 100:.1f}%"

        results = results.T
        results.index.name = f"Trigger Efficiency ({self.sample.label}, {self.year}, {ch})"

        if save:
            results.to_csv(self.plot_dir / f"progressive_removal_{ch}_{name_tag}.csv")

        print(results.to_markdown())

    def trigger_correlation_table(self):
        """Compute and print/save the trigger correlation table for the current channel."""
        ch = self.ch_key
        triggers = [hlt.lower() for hlt in self.channel.hlt_types]
        masks_by_class = pd.DataFrame({tr: self.fired_events_by_class(tr) for tr in triggers})
        phi_coeff = masks_by_class.corr(method="pearson")
        print(f"\nTrigger phi coefficient table for {ch}:")
        print(phi_coeff.to_markdown())
        phi_coeff.to_csv(self.plot_dir / f"trigger_phi_coefficient_{ch}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger efficiency study for bbtautau signals")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="Years to process (default: all Run3 years)",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=SIGNALS,
        choices=SIGNALS,
        help="Signal keys to process (default: all signals)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(CHANNELS.keys()),
        choices=list(CHANNELS.keys()),
        help="Channels to process (default: all channels)",
    )
    parser.add_argument("--test-mode", action="store_true", default=False)
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Base plot output directory (default: PLOT_DIR/TriggerStudy/...)",
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False, help="Disable saving plots and tables"
    )
    args = parser.parse_args()

    save = not args.no_save

    for year in args.years:
        print(f"\n\n\nYEAR {year}\n\n\n")
        for sig_key in args.signals:
            print("sig_key : ", sig_key)
            plot_dir = Path(args.plot_dir) if args.plot_dir else None
            analyser = Analyser(year, sig_key, test_mode=args.test_mode, plot_dir=plot_dir)
            analyser.load_data()

            for ch_key in args.channels:
                print(f"\n--- Channel: {ch_key} ---")
                analyser.set_channel(ch_key)
                analyser.define_taggers()
                analyser.set_plot_dict()
                analyser.set_quantities()

                analyser.plot_channel(save=save)
                analyser.N1_efficiency_table(save=save)

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
                    analyser.progressive_trigger_removal(remove_trs, name_tag="", save=save)
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
                    analyser.progressive_trigger_removal(remove_trs, name_tag="", save=save)
