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
from boostedhh.utils import HLT
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from bbtautau.HLTs import HLTs
from bbtautau.postprocessing.bdt_config import BDT_CONFIG
from bbtautau.postprocessing.bdt_utils import _get_bdt_key, compute_or_load_bdt_preds
from bbtautau.postprocessing.Regions import load_cuts_from_csv
from bbtautau.postprocessing.Samples import (
    CHANNELS,
    SAMPLES,
    SM_SIGNALS,
)
from bbtautau.postprocessing.utils import (
    base_filter,
    bbtautau_assignment,
    derive_lepton_variables,
    derive_variables,
    derive_vbf_variables,
    get_columns,
    leptons_assignment,
    load_samples,
)
from bbtautau.userConfig import BDT_EVAL_DIR, DATA_PATHS, MODEL_DIR, PLOT_DIR

# Default MC keys for ``--mode overlap`` when ``--overlap-samples`` is omitted.
DEFAULT_OVERLAP_SAMPLES = [*SM_SIGNALS]  # , *BGS]

STUDY_MODES = ("sig_eff", "overlap")

OVERLAP_PLOT_KINDS = ("multiplicity", "cofire-prob", "cofire-lift")

GLOPART_PRESEL_THRESHOLDS: dict[str, float] = {
    "ak8FatJetParTXbbvsQCDTop": 0.3,
    "ak8FatJetParTX{tagger_label}vsQCDTop": 0.5,
}

BDT_MODELNAMES = {"ggfbbtt": "May4_optimized_ggf", "vbfbbtt": "May4_optimized_vbf"}


def study_plot_dir(
    year: str,
    sample_key: str,
    *,
    plot_dir: Path | str | None = None,
    mode: str = "sig_eff",
) -> Path:
    """``.../TriggerStudy/<date>/<year>/<mode>/<sample_key>/`` (or ``<plot_dir>/<mode>/<sample_key>/``)."""
    if mode not in STUDY_MODES:
        raise ValueError(f"mode must be one of {STUDY_MODES}, got {mode!r}")
    root = (
        Path(plot_dir) if plot_dir is not None else PLOT_DIR / f"TriggerStudy/{date.today()}/{year}"
    )
    return root / mode / sample_key


"""
Objectives:
- trigger study with all sig channels
- change trigger choice, add met
- correlation between triggers (subtract 1 at the time)
"""


class Analyser:
    """
    Trigger study for one MC sample in ``SAMPLES``
    """

    def __init__(
        self,
        year,
        sample_key,
        test_mode=False,
        plot_dir=None,
        *,
        mode: str = "sig_eff",
        use_bdt: bool = False,
        bdt_cuts_csv: str | Path | None = None,
        bdt_modelname: str = None,
        model_dir: Path = MODEL_DIR,
        bdt_eval_dir: Path = BDT_EVAL_DIR,
    ):
        if sample_key not in SAMPLES:
            raise ValueError(f"Unknown sample_key {sample_key!r} (not in SAMPLES)")

        self.year = year
        self.sample_key = sample_key
        self.sample = SAMPLES[sample_key]
        self.test_mode = test_mode

        if self.sample.isData:
            raise ValueError(
                "TriggerStudy expects MC; pick a non-data key from SAMPLES (or extend loading)."
            )

        self.base_plot_dir = study_plot_dir(year, sample_key, plot_dir=plot_dir, mode=mode)
        self.channel_samples = {}
        self.use_bdt = use_bdt
        self.bdt_cuts_csv = Path(bdt_cuts_csv) if bdt_cuts_csv else None
        self.bdt_modelname = bdt_modelname
        self.model_dir = model_dir
        self.bdt_eval_dir = bdt_eval_dir

    def load_data(self, *, apply_event_filters: bool = False):
        """
        Load events. By default **no** ``base_filter`` row selection is applied (full MC before
        skimmer-level kinematics in parquet); ``base_filter`` is **not** HLT-based—it only adds
        fat-jet pT/mass-style cuts if enabled.

        Set ``apply_event_filters=True`` to apply :func:`~bbtautau.postprocessing.utils.base_filter`
        (e.g. to match an analysis ntuple that was already subset). Ensure loaded branches match
        those filters (the default ``get_columns`` preset must include columns referenced there).

        Branches are chosen with :func:`~bbtautau.postprocessing.utils.get_columns` (union of HLTs
        over hh/hm/he, ParT on, no legacy PNet / leptons / VBF / extra scatters).
        """
        filters_dict = base_filter(self.test_mode) if apply_event_filters else None
        all_channels = list(CHANNELS.values())
        if self.use_bdt:
            cols = get_columns(self.year, all_hlts=True)
        else:
            cols = get_columns(
                self.year,
                all_hlts=True,
                legacy_taggers=False,
                ParT_taggers=True,
                leptons=False,
                other=False,
                vbf=False,
            )

        print(self.sample)
        self.channel_samples = load_samples(
            year=self.year,
            paths=DATA_PATHS[self.year],
            signals=[self.sample_key] if self.sample.isSignal else [],
            channels=all_channels,
            samples={self.sample_key: self.sample},
            filters_dict=filters_dict,
            load_columns=cols,
            loaded_samples=True,
            restrict_data_to_channel=True,
        )

        if self.use_bdt:
            derive_variables(self.channel_samples)
            bbtautau_assignment(self.channel_samples)
            leptons_assignment(self.channel_samples, dR_cut=1.5)
            derive_lepton_variables(self.channel_samples)
            derive_vbf_variables(self.channel_samples)

            try:
                bdt_cfg = BDT_CONFIG[self.bdt_modelname]
                n_folds = int(bdt_cfg.get("n_folds", 1))
            except (KeyError, ValueError) as exc:
                print(f"Warning: BDT config lookup failed for {self.bdt_modelname}: {exc}")
                n_folds = 1

            for channel in all_channels:
                sample_name = (
                    f"{self.sample_key}{channel.key}" if self.sample.isSignal else self.sample_key
                )
                if sample_name not in self.channel_samples:
                    continue
                compute_or_load_bdt_preds(
                    events_dict={self.year: {sample_name: self.channel_samples[sample_name]}},
                    modelname=self.bdt_modelname,
                    model_dir=self.model_dir,
                    channel=channel,
                    bdt_preds_dir=self.bdt_eval_dir,
                    tt_pres=False,
                    test_mode=self.test_mode,
                    n_folds=n_folds,
                )

        print(f"Loaded {self.sample.label} for year {self.year}")
        for key, sample in self.channel_samples.items():
            print(f"  {key}: {len(sample.events)} events")

    def set_channel(self, ch_key):
        """Set the current channel for analysis. Must be called before analysis methods."""
        self.ch_key = ch_key
        self.channel = CHANNELS[ch_key]
        dict_key = f"{self.sample_key}{ch_key}" if self.sample.isSignal else self.sample_key
        self.current_sample = self.channel_samples[dict_key]
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
            mask |= self._class_mask(cl)

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

        try:
            higgs = utils.make_vector(self.events_dict, name="GenHiggs")
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
                presel_counts = hists["Preselection"][0].astype(float)

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
                    ratios[key] = np.divide(
                        hists[key][0].astype(float),
                        presel_counts,
                        out=np.full_like(presel_counts, np.nan),
                        where=presel_counts > 0,
                    )

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
            & (self.events_dict["ak8FatJetParTXbbvsQCD"] > 0.95).any(axis=1),
            "2 boosted jets (>250, >200), XbbvsQCDTop > 0.95": (
                self.events_dict["ak8FatJetPt"][0] > 250
            )
            & (self.events_dict["ak8FatJetPt"][1] > 200)
            & (self.events_dict["ak8FatJetParTXbbvsQCDTop"] > 0.95).any(axis=1),
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
                & (self.events_dict["ak8FatJetParTXbbvsQCD"] > 0.95).any(axis=1)
            ),
            "2 boosted jets (>250, >200), XbbvsQCDTop > 0.95": (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & (self.events_dict["ak8FatJetParTXbbvsQCDTop"] > 0.95).any(axis=1)
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
        self, trs: dict[str, list[dict[str, str]]], save: bool = True, name_tag: str = ""
    ) -> None:
        """
        Compute and print/save trigger fractions with a sequential exclusive logic:
        first column is full OR efficiency, subsequent columns are weighted fractions
        of remaining preselected events captured by each trigger step.
        """
        base_preselections = {
            "1 boosted jet (> 250)": self.events_dict["ak8FatJetPt"][0] > 250,
            "2 boosted jets (>250, >200)": (self.events_dict["ak8FatJetPt"][0] > 250)
            & (self.events_dict["ak8FatJetPt"][1] > 200),
        }

        ch = self.ch_key
        print(f"\nChannel: {ch}\n")

        preselection_masks = dict(base_preselections)
        if GLOPART_PRESEL_THRESHOLDS:
            glopart_terms = [
                np.asarray(
                    (
                        self.events_dict[col_name.format(tagger_label=self.channel.tagger_label)]
                        > threshold
                    ).any(axis=1),
                    dtype=bool,
                )
                for col_name, threshold in GLOPART_PRESEL_THRESHOLDS.items()
            ]
            glopart_mask = (
                (self.events_dict["ak8FatJetPt"][0] > 250)
                & (self.events_dict["ak8FatJetPt"][1] > 200)
                & np.logical_and.reduce(glopart_terms)
            )
            wp_desc = ", ".join(
                f"{col_name.format(tagger_label=self.channel.tagger_label)}>={threshold:.3f}"
                for col_name, threshold in sorted(GLOPART_PRESEL_THRESHOLDS.items())
            )
            preselection_masks[f"2 boosted jets, {wp_desc}"] = np.asarray(glopart_mask, dtype=bool)
        else:
            print("Skipping GloParT preselection row: configure GLOPART_PRESEL_THRESHOLDS")

        if self.use_bdt and self.bdt_cuts_csv:
            try:
                txbb_col = "ak8FatJetParTXbbvsQCDTop"
                bdt_col = _get_bdt_key(
                    self.sample_key, channel=self.channel, prefix_only=False, suffix="vsAll"
                )
                missing_cols = [
                    col for col in ("ak8FatJetPt", txbb_col, bdt_col) if col not in self.events_dict
                ]
                if missing_cols:
                    print(f"Skipping BDT+Xbb preselection rows; missing columns: {missing_cols}")
                else:
                    cuts_csv = _resolve_bdt_cuts_csv(
                        self.bdt_cuts_csv, self.sample_key, self.channel.key
                    )
                    for bmin in (10, 12):
                        txbb_cut, txtt_cut = load_cuts_from_csv(cuts_csv, bmin)
                        bdt_mask = (
                            (self.events_dict["ak8FatJetPt"][0] > 250)
                            & (self.events_dict["ak8FatJetPt"][1] > 200)
                            & (self.events_dict[txbb_col] > txbb_cut).any(axis=1)
                            & (self.events_dict[bdt_col] > txtt_cut)
                        )
                        preselection_masks[
                            f"2 boosted jets, Xbb>{txbb_cut:.3f} + BDT>{txtt_cut:.3f} (Bmin={bmin})"
                        ] = np.asarray(bdt_mask, dtype=bool)
            except Exception as exc:
                print(f"Skipping BDT+Xbb preselection rows; failed to load cuts: {exc}")

        result_cols = ["All"] + [f"-{t['show_name']}" for t in trs[ch]]
        results = pd.DataFrame(index=preselection_masks.keys(), columns=result_cols)
        weights = np.asarray(self.events_dict["weight"][0], dtype=np.float64).ravel()

        for bkey, sel in preselection_masks.items():
            _sel = np.asarray(sel, dtype=bool)
            all_triggers_mask = self.plot_dict[ch]["triggers"]["All"]
            total_weight = float(weights[_sel].sum())
            all_triggers_eff = (
                float(weights[_sel & all_triggers_mask].sum()) / total_weight
                if total_weight > 0
                else 0.0
            )

            results.loc[bkey, "All"] = f"{all_triggers_eff * 100:.1f}%"

            excluded_previous_cols = ~_sel
            for cl_or_tr in trs[ch]:
                if cl_or_tr["type"] == "cl":
                    hlts = HLTs.hlts_by_type(self.year, cl_or_tr["name"], as_str=True)
                else:
                    hlts = [cl_or_tr["name"]]
                current_col = self.fired_events_by_trs(hlts) & ~excluded_previous_cols
                excluded_previous_cols |= current_col

                eff = float(weights[current_col].sum()) / total_weight if total_weight > 0 else 0.0
                results.loc[bkey, f"-{cl_or_tr['show_name']}"] = f"{eff * 100:.1f}%"

        results.index.name = f"Trigger Efficiency ({self.sample.label}, {self.year}, {ch})"

        if save:
            results.to_csv(self.plot_dir / f"progressive_removal_{ch}_{name_tag}.csv")

        print(results.to_markdown())

    def trigger_correlation_table(self, save: bool = True):
        """Class-level trigger correlation (Pearson on boolean masks); figure only, no CSV."""
        ch = self.ch_key
        triggers = [hlt.lower() for hlt in self.channel.hlt_types]
        masks_by_class = pd.DataFrame({tr: self.fired_events_by_class(tr) for tr in triggers})
        phi_coeff = masks_by_class.corr(method="pearson")
        print(f"\nTrigger phi coefficient table for {ch}:")
        print(phi_coeff.to_markdown())

        plt.rcParams.update({"font.size": 12})
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(phi_coeff.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(phi_coeff.columns)))
        ax.set_yticks(range(len(phi_coeff.index)))
        ax.set_xticklabels(phi_coeff.columns, rotation=45, ha="right")
        ax.set_yticklabels(phi_coeff.index)
        plt.colorbar(im, ax=ax, label=r"Pearson $\phi$")
        ax.set_title(f"{self.sample.label} {self.year} {ch}")
        fig.tight_layout()
        if save:
            fig.savefig(self.plot_dir / f"trigger_phi_coefficient_{ch}.pdf", bbox_inches="tight")
            fig.savefig(self.plot_dir / f"trigger_phi_coefficient_{ch}.png", bbox_inches="tight")
        plt.close(fig)


def _slug_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "presel"


def _resolve_bdt_cuts_csv(
    base_path: str | Path,
    sample_key: str,
    channel_key: str,
) -> Path:
    base = Path(base_path)
    if base.is_file():
        return base

    csv_dir = base / sample_key / channel_key
    if not csv_dir.exists():
        raise FileNotFoundError(f"BDT cuts directory not found: {csv_dir}")

    csv_files = sorted(csv_dir.glob("*_opt_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No BDT cuts CSV files found in {csv_dir}")
    if len(csv_files) > 1:
        print(f"Multiple BDT cuts CSVs found in {csv_dir}; using {csv_files[0].name}")
    return csv_files[0]


def _event_hlt_mask(events: pd.DataFrame, year: str, hlt: HLT | str) -> np.ndarray | None:
    if isinstance(hlt, str):
        hlt = HLTs.get_hlt(hlt)
    if not isinstance(hlt, HLT):
        return None
    if year not in hlt.mc_years:
        return None
    name = hlt.get_name(True)
    col = events.get(name)
    if col is None:
        return None
    return col.to_numpy(dtype=bool).ravel()


def load_overlap_sample(
    year: str,
    sample_key: str,
    *,
    test_mode: bool = False,
    apply_event_filters: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load one MC sample for overlap plots (no GenTau channel split; full process in one table).
    """
    if sample_key not in SAMPLES or SAMPLES[sample_key].isData:
        raise ValueError(f"overlap sample must be a non-data SAMPLES key, got {sample_key!r}")

    all_channels = list(CHANNELS.values())
    cols = get_columns(
        year,
        all_hlts=True,
        legacy_taggers=False,
        ParT_taggers=True,
        leptons=False,
        other=False,
        vbf=False,
    )
    fd = base_filter(test_mode) if apply_event_filters else None
    loaded = load_samples(
        year=year,
        paths=DATA_PATHS[year],
        signals=[],
        channels=all_channels,
        samples={sample_key: SAMPLES[sample_key]},
        filters_dict=fd,
        load_columns=cols,
        loaded_samples=True,
        restrict_data_to_channel=True,
    )
    if sample_key not in loaded:
        raise KeyError(f"Sample {sample_key} was not loaded for year {year}")
    events = loaded[sample_key].events.reset_index(drop=True)
    weights = np.asarray(events["weight"][0], dtype=np.float64).ravel()
    print(f"Overlap load {sample_key} ({year}): {len(events)} events")
    return events, weights


def _overlap_hlt_menu(year: str) -> list[str]:
    """Union of MC HLT path names over hh, hm, he."""
    return sorted({hlt for ch in CHANNELS.values() for hlt in ch.triggers(year, mc_only=True)})


def _overlap_presel_masks(events: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(events)
    ak8 = events["ak8FatJetPt"]
    out = {
        "inclusive": np.ones(n, dtype=bool),
        "boosted_1ak8_pt250": ak8[0] > 250,
        "boosted_2ak8_pt250_200": (ak8[0] > 250) & (ak8[1] > 200),
    }
    if "ak8FatJetParTXbbvsQCD" in events.columns:
        out["boosted_2ak8_pt250_200_xbb0p3"] = (
            (ak8[0] > 250) & (ak8[1] > 200) & (events["ak8FatJetParTXbbvsQCD"] > 0.3).any(axis=1)
        )
    return out


def _overlap_hlt_fire_matrix(events: pd.DataFrame, year: str) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for name in _overlap_hlt_menu(year):
        m = _event_hlt_mask(events, year, name)
        if m is not None:
            cols[name] = m
    return pd.DataFrame(cols)


def _overlap_analysis_classes() -> list[str]:
    """HLT classes used in hh / hm / he menus (not every key in ``HLTs.HLTs``)."""
    return sorted({ht.lower() for ch in CHANNELS.values() for ht in ch.hlt_types})


def _overlap_class_fire_matrix(events: pd.DataFrame, year: str) -> pd.DataFrame:
    """One column per analysis trigger class (OR of paths valid for *year*)."""
    n = len(events)
    cols: dict[str, np.ndarray] = {}
    for cl in _overlap_analysis_classes():
        if cl not in HLTs.HLTs:
            continue
        m = np.zeros(n, dtype=bool)
        for hlt in HLTs.HLTs[cl]:
            hm = _event_hlt_mask(events, year, hlt)
            if hm is not None:
                m |= hm
        cols[cl] = m
    return pd.DataFrame(cols)


def _annotate_overlap_figure(
    ax,
    *,
    presel_name: str,
    plot_title: str,
    sample_key: str,
    year: str,
) -> None:
    """CMS-style label + legend for sample and preselection (no text box)."""
    ax.set_title(plot_title, fontsize=13, pad=8)
    presel_display = presel_name.replace("_", " ")
    sample_label = SAMPLES[sample_key].label
    handles = [
        Line2D([], [], linestyle="none", label=sample_label),
        Line2D([], [], linestyle="none", label=presel_display),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=11,
        handlelength=0,
        handletextpad=0,
    )
    hep.cms.label(ax=ax, data=False, year=year, com="13.6", fontsize=13)


def _plot_multiplicity(
    ax,
    fire_matrix: pd.DataFrame,
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
    presel_name: str,
    sample_key: str,
    year: str,
) -> None:
    """Bar chart of how many triggers fired per event."""
    n_fired = fire_matrix.to_numpy(dtype=np.int32).sum(axis=1)
    n_fired_sel = n_fired[mask]
    max_k = min(int(n_fired_sel.max()) if len(n_fired_sel) else 0, 40)

    edges = np.arange(0, max_k + 2) - 0.5
    hist, _ = np.histogram(n_fired_sel, bins=edges, weights=weights)

    centers = np.arange(0, max_k + 1)
    ax.bar(centers, hist, width=0.9)
    ax.set_xlim(-0.5, max_k + 0.5)
    ax.set_xticks(centers)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"$N_\mathrm{HLT}$ fired")
    ax.set_ylabel("Weighted events")
    _annotate_overlap_figure(
        ax,
        presel_name=presel_name,
        plot_title=f"Trigger multiplicity ({label})",
        sample_key=sample_key,
        year=year,
    )


def _cofire_fraction_matrix(
    fire_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted co-firing fraction matrix: entry (i,j) = P(trigger_i AND trigger_j fired)."""
    total_weight = float(weights.sum())
    n = fire_matrix.shape[1]
    co = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            co[i, j] = float(weights[fire_matrix[:, i] & fire_matrix[:, j]].sum())
    return co / total_weight


def _cofire_lift_matrix(
    fire_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Lift matrix P(i∩j)/(P(i)P(j)); diagonal set to NaN."""
    total_weight = float(weights.sum())
    n = fire_matrix.shape[1]
    p = np.array(
        [float(weights[fire_matrix[:, k]].sum()) / total_weight for k in range(n)],
        dtype=np.float64,
    )
    co_frac = _cofire_fraction_matrix(fire_matrix, weights)
    lift = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = p[i] * p[j]
            if denom > 0:
                lift[i, j] = co_frac[i, j] / denom
    return lift


def _plot_cofire_heatmap(
    ax,
    matrix: np.ndarray,
    labels: list[str],
    *,
    mode: str,
    matrix_label: str,
    presel_name: str,
    sample_key: str,
    year: str,
) -> None:
    """Heatmap of pairwise co-firing probability or lift (HLT paths or trigger classes)."""
    n = len(labels)
    tick_fs = 9 if n > 12 else 11
    off_diag = matrix[~np.eye(n, dtype=bool)]

    if mode == "prob":
        vmax = float(np.nanmax(matrix)) if matrix.size else 1.0
        im = ax.imshow(
            matrix,
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
        )
        cbar_label = "Weighted fraction of events"
        plot_title = f"Co-firing fraction ({matrix_label})"
    elif mode == "lift":
        finite = off_diag[np.isfinite(off_diag)]
        vmin = 0.0  # lift = P(i∩j)/[P(i)P(j)] is non-negative
        vmax = max(float(np.percentile(finite, 95)), 1.05) if finite.size else 1.5
        im = ax.imshow(
            matrix,
            norm=TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax),
            cmap="coolwarm",
            aspect="auto",
        )
        cbar_label = r"Lift $P(i \cap j) / [P(i)\,P(j)]$"
        plot_title = f"Co-firing lift ({matrix_label})"
    else:
        raise ValueError(f"mode must be 'prob' or 'lift', got {mode!r}")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.figure.colorbar(im, ax=ax, label=cbar_label, shrink=0.85, pad=0.02)
    _annotate_overlap_figure(
        ax,
        presel_name=presel_name,
        plot_title=plot_title,
        sample_key=sample_key,
        year=year,
    )


def _overlap_kind_dir(plot_dir: Path, kind: str) -> Path:
    if kind not in OVERLAP_PLOT_KINDS:
        raise ValueError(f"kind must be one of {OVERLAP_PLOT_KINDS}, got {kind!r}")
    out = plot_dir / kind
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_overlap_figure(fig: plt.Figure, out_dir: Path, stem: str, *, save: bool) -> None:
    if not save:
        return
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")


def plot_overlap_figures(
    year: str,
    sample_key: str,
    events: pd.DataFrame,
    weights: np.ndarray,
    plot_dir: Path,
    *,
    save: bool = True,
) -> None:
    """Multiplicity and co-firing figures for one sample (no CSV).

    Output layout: ``<plot_dir>/{multiplicity,cofire-prob,cofire-lift}/<hlt|class>_<presel>.{pdf,png}``.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    dir_multiplicity = _overlap_kind_dir(plot_dir, "multiplicity")
    dir_cofire_prob = _overlap_kind_dir(plot_dir, "cofire-prob")
    dir_cofire_lift = _overlap_kind_dir(plot_dir, "cofire-lift")

    weights = np.asarray(weights).ravel()
    hep.style.use("CMS")

    hlt_fire_matrix = _overlap_hlt_fire_matrix(events, year)
    cls_fire_matrix = _overlap_class_fire_matrix(events, year)

    for presel_name, mask in _overlap_presel_masks(events).items():
        _mask = np.asarray(mask, dtype=bool)
        if not np.any(_mask):
            continue
        slug = _slug_filename(presel_name)
        w_sel = weights[_mask]
        if w_sel.sum() <= 0:
            continue

        for matrix_label, fm in [("hlt", hlt_fire_matrix), ("class", cls_fire_matrix)]:
            if fm.empty:
                continue
            stem = f"{matrix_label}_{slug}"

            # --- multiplicity ---
            fig, ax = plt.subplots(figsize=(10, 6))
            _plot_multiplicity(
                ax,
                fm,
                mask,
                w_sel,
                label=matrix_label,
                presel_name=presel_name,
                sample_key=sample_key,
                year=year,
            )
            fig.tight_layout()
            _save_overlap_figure(fig, dir_multiplicity, stem, save=save)
            plt.close(fig)

            fire_arr = fm.to_numpy(dtype=bool)[mask]
            if fire_arr.shape[0] == 0:
                continue

            plot_labels = (
                [HLTs.short_label(c) for c in fm.columns]
                if matrix_label == "hlt"
                else list(fm.columns)
            )
            n_triggers = len(fm.columns)
            fig_side = max(8.0, 0.55 * n_triggers)

            for mode, out_dir in [("prob", dir_cofire_prob), ("lift", dir_cofire_lift)]:
                matrix = (
                    _cofire_fraction_matrix(fire_arr, w_sel)
                    if mode == "prob"
                    else _cofire_lift_matrix(fire_arr, w_sel)
                )
                fig, ax = plt.subplots(
                    figsize=(fig_side, fig_side),
                    constrained_layout=True,
                )
                _plot_cofire_heatmap(
                    ax,
                    matrix,
                    plot_labels,
                    mode=mode,
                    matrix_label=matrix_label,
                    presel_name=presel_name,
                    sample_key=sample_key,
                    year=year,
                )
                _save_overlap_figure(fig, out_dir, stem, save=save)
                plt.close(fig)


def run_overlap_study(
    year: str,
    sample_keys: list[str],
    *,
    test_mode: bool = False,
    plot_dir: Path | str | None = None,
    apply_event_filters: bool = False,
    save: bool = True,
) -> None:
    """One overlap figure set per sample under ``.../overlap/<sample_key>/{multiplicity,cofire-*}/``."""
    for sample_key in sample_keys:
        print(f"\n--- Overlap: {sample_key} ---")
        out_dir = study_plot_dir(year, sample_key, plot_dir=plot_dir, mode="overlap")
        events, weights = load_overlap_sample(
            year,
            sample_key,
            test_mode=test_mode,
            apply_event_filters=apply_event_filters,
        )
        plot_overlap_figures(year, sample_key, events, weights, out_dir, save=save)


def run_trigger_study_channels(
    analyser: Analyser,
    *,
    channels: list[str],
    save: bool = True,
) -> None:
    """Per-channel signal efficiency study: kinematic plots, trig_effs + progressive removal."""
    for ch_key in channels:
        print(f"\n--- Channel: {ch_key} ---")
        analyser.set_channel(ch_key)
        analyser.set_plot_dict()
        analyser.set_quantities()

        analyser.plot_channel(save=save)
        analyser.N1_efficiency_table(save=save)

        year = analyser.year
        if year == "2022":
            remove_trs = {
                "hh": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
                        "show_name": "DiTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
                        "show_name": "DiTauJet ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                        "show_name": "QuadJet",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
                "hm": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_IsoMu24",
                        "show_name": "MuIso24 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Mu50",
                        "show_name": "Mu50 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
                        "show_name": "MuTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
                "he": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele30_WPTight_Gsf",
                        "show_name": "Ele30Tight ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                        "show_name": "Ele50PFJet ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
                        "show_name": "ETau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
            }
            analyser.progressive_trigger_removal(remove_trs, name_tag="", save=save)

        if year in ("2023BPix", "2024"):
            remove_trs = {
                "hh": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2",
                        "show_name": "VBF_Quadjet2 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1",
                        "show_name": "VBF_Quadjet1 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60",
                        "show_name": "DiTauJet ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",
                        "show_name": "DiTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFHT280_QuadPFJet30_PNet2BTagMean0p55",
                        "show_name": "Parking",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
                "hm": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_IsoMu24",
                        "show_name": "MuIso24 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Mu50",
                        "show_name": "Mu50 ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1",
                        "show_name": "MuTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
                "he": [
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetBB0p06",
                        "show_name": "PNetBB ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele30_WPTight_Gsf",
                        "show_name": "Ele30Tight ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                        "show_name": "Ele50PFJet ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
                        "show_name": "ETau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_AK8PFJet230_SoftDropMass40_PNetTauTau0p03",
                        "show_name": "PNetTauTau ",
                    },
                    {
                        "type": "tr",
                        "name": "HLT_PFMET120_PFMHT120_IDTight",
                        "show_name": "MET ",
                    },
                ],
            }
            analyser.progressive_trigger_removal(remove_trs, name_tag="", save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trigger study: signal efficiency (per channel) or per-sample overlap figures"
    )
    parser.add_argument(
        "--mode",
        choices=list(STUDY_MODES),
        default="sig_eff",
        help=(
            "sig_eff: signal GenHiggs plots + trig_effs/progressive_removal CSVs per channel. "
            "overlap: per-sample multiplicity and co-firing figures under overlap/<sample_key>/."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
        help="Years to process (default: all Run3 years)",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=SM_SIGNALS,
        choices=SM_SIGNALS,
        help="Signal keys (--mode sig_eff only; default: SM signals)",
    )
    parser.add_argument(
        "--overlap-samples",
        nargs="+",
        default=DEFAULT_OVERLAP_SAMPLES,
        metavar="KEY",
        help=(
            "SAMPLES keys for --mode overlap (default: SM_SIGNALS + BGS). One plot set per key "
            "under overlap/<sample_key>/ (full process, no GenTau channel split)."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(CHANNELS.keys()),
        choices=list(CHANNELS.keys()),
        help="Channels (--mode sig_eff only)",
    )
    parser.add_argument("--test-mode", action="store_true", default=False)
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Base output directory; outputs go to <plot_dir>/<mode>/<sample_key>/",
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False, help="Disable saving plots and CSV"
    )
    parser.add_argument(
        "--apply-event-filters",
        action="store_true",
        default=False,
        help=(
            "Apply base_filter (fat-jet kinematics) when loading; default is unfiltered rows. "
            "Requires those branches in the parquet / get_columns preset (extend get_columns if needed)."
        ),
    )
    parser.add_argument(
        "--use-bdt",
        action="store_true",
        default=False,
        help=("Load BDT scores and full column set (default: light trigger-only columns)."),
    )
    parser.add_argument(
        "--bdt-cuts-csv",
        type=str,
        default=None,
        help=(
            "CSV file or base directory containing <signal>/<channel>/*_opt_results_*.csv "
            "for the BDT+Xbb preselection rows."
        ),
    )
    args = parser.parse_args()

    save = not args.no_save

    for year in args.years:
        print(f"\n\n\nYEAR {year}\n\n\n")
        plot_dir = Path(args.plot_dir) if args.plot_dir else None

        if args.mode == "sig_eff":
            for sample_key in args.signals:
                print("sample_key: ", sample_key)
                analyser = Analyser(
                    year,
                    sample_key,
                    test_mode=args.test_mode,
                    plot_dir=plot_dir,
                    mode="sig_eff",
                    use_bdt=args.use_bdt,
                    bdt_modelname=BDT_MODELNAMES[sample_key],
                    bdt_cuts_csv=args.bdt_cuts_csv,
                )
                analyser.load_data(apply_event_filters=args.apply_event_filters)
                run_trigger_study_channels(analyser, channels=args.channels, save=save)
        else:
            run_overlap_study(
                year,
                args.overlap_samples,
                test_mode=args.test_mode,
                plot_dir=plot_dir,
                apply_event_filters=args.apply_event_filters,
                save=save,
            )
