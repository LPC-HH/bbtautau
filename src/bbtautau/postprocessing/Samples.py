from __future__ import annotations

from boostedhh import hh_vars
from boostedhh.utils import Sample

from bbtautau.bbtautau_utils import Channel

CHANNELS = {
    "hh": Channel(
        key="hh",
        label=r"$\tau_h\tau_h$",
        hlt_types=["PNet", "PFJet", "QuadJet", "Parking", "DiTau", "DitauJet", "SingleTau", "MET"],
        data_samples=["jetmet", "tau"],
        isLepton=False,
        tagger_label="tauhtauh",
        txbb_cut=0.91,
        txtt_cut=0.999,
        tt_mass_cut=("PNetmassLegacy", [50, 150]),
    ),
    "hm": Channel(
        key="hm",
        label=r"$\tau_h \mu$",
        hlt_types=["PNet", "PFJet", "Muon", "MuonTau", "DiTau", "DitauJet", "SingleTau", "MET"],
        data_samples=["jetmet", "tau", "muon"],
        lepton_dataset="muon",
        isLepton=True,
        tagger_label="tauhtaum",
        txbb_cut=0.851025641025641,
        txtt_cut=0.9887948717948718,
        tt_mass_cut=("ParTmassResApplied", [70, 210]),
    ),
    "he": Channel(
        key="he",
        label=r"$\tau_h e$",
        hlt_types=["PNet", "PFJet", "EGamma", "ETau", "DiTau", "DitauJet", "SingleTau"],
        data_samples=["jetmet", "tau", "egamma"],
        lepton_dataset="egamma",
        isLepton=True,
        tagger_label="tauhtaue",
        txbb_cut=0.91,
        txtt_cut=0.999,
        tt_mass_cut=("ParTmassResApplied", [70, 210]),
    ),
}

# overall list of samples
SAMPLES = {
    "jetmet": Sample(
        selector="^(JetHT|JetMET)",
        label="JetMET",
        isData=True,
    ),
    "tau": Sample(
        selector="^Tau_Run",
        label="Tau",
        isData=True,
    ),
    "muon": Sample(
        selector="^Muon_Run",
        label="Muon",
        isData=True,
    ),
    "egamma": Sample(
        selector="^EGamma_Run",
        label="EGamma",
        isData=True,
    ),
    "qcd": Sample(
        selector="^QCD",
        label="QCD Multijet",
        isSignal=False,
    ),
    "ttbarhad": Sample(
        selector="^TTto4Q",
        label="TT Had",
        isSignal=False,
    ),
    "ttbarsl": Sample(
        selector="^TTtoLNu2Q",
        label="TT SL",
        isSignal=False,
    ),
    "ttbarll": Sample(
        selector="^TTto2L2Nu",
        label="TT LL",
        isSignal=False,
    ),
    "dyjets": Sample(
        selector="^DYto2L",
        label="DY+Jets",
        isSignal=False,
    ),
    "wjets": Sample(
        selector="^(Wto2Q-2Jets|WtoLnu-2Jets)",
        label="W+Jets",
        isSignal=False,
    ),
    "zjets": Sample(
        selector="^Zto2Q-2Jets",
        label="Z+Jets",
        isSignal=False,
    ),
    "hbb": Sample(
        selector="^(GluGluHto2B|VBFHto2B|WminusH_Hto2B|WplusH_Hto2B|ZH_Hto2B|ggZH_Hto2B)",
        label="Hbb",
        isSignal=False,
    ),
    "bbtt": Sample(
        selector=hh_vars.bbtt_sigs["bbtt"],
        label=r"ggF HHbb$\tau\tau$",
        isSignal=True,
    ),
    "vbfbbtt": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt"],
        label=r"VBF HHbb$\tau\tau$ (SM)",
        isSignal=True,
    ),
    "vbfbbtt-k2v0": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-k2v0"],
        label=r"VBF HHbb$\tau\tau$ ($\kappa_{2V}=0$)",
        isSignal=True,
    ),
}

SIGNALS = ["bbtt", "vbfbbtt", "vbfbbtt-k2v0"]
SIGNALS_CHANNELS = SIGNALS.copy()

sig_keys_ggf = ["bbtt"]
sig_keys_vbf = ["vbfbbtt-k2v0"]

# add individual bbtt channels
for signal in SIGNALS.copy():
    for channel, CHANNEL in CHANNELS.items():
        SAMPLES[f"{signal}{channel}"] = Sample(
            label=SAMPLES[signal].label.replace(r"$\tau\tau$", CHANNEL.label),
            isSignal=True,
        )
        SIGNALS_CHANNELS.append(f"{signal}{channel}")

DATASETS = ["jetmet", "tau", "egamma", "muon"]

BGS = [
    "qcd",
    "ttbarhad",
    "ttbarsl",
    "ttbarll",
    "dyjets",
    "wjets",
    "zjets",
    "hbb",
]

single_h_keys = ["hbb"]
ttbar_keys = ["ttbarhad", "ttbarsl", "ttbarll"]

qcdouts = ["QCD0HF", "QCD1HF", "QCD2HF"]
topouts = ["TopW", "TopbW", "TopbWev", "TopbWmv", "TopbWtauhv", "TopbWq", "TopbWqq"][:2]
sigouts = ["Xtauhtauh", "Xtauhtaue", "Xtauhtaum", "Xbb"]
