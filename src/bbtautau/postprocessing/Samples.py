from __future__ import annotations

from boostedhh import hh_vars
from boostedhh.utils import Channel, Sample

from bbtautau import bbtautau_vars

CHANNELS = {
    "hh": Channel(
        key="hh",
        label=r"$\tau_h\tau_h$",
        triggers=bbtautau_vars.HLT_hh,
        data_samples=["jetmet", "tau"],
        isLepton=False,
    ),
    "he": Channel(
        key="he",
        label=r"$\tau_h e$",
        triggers=bbtautau_vars.HLT_he,
        data_samples=["jetmet", "tau", "egamma"],
        lepton_dataset="egamma",
        lepton_triggers=bbtautau_vars.HLT_egammas,
        isLepton=True,
    ),
    "hm": Channel(
        key="hm",
        label=r"$\tau_h \mu$",
        triggers=bbtautau_vars.HLT_hmu,
        data_samples=["jetmet", "tau", "muon"],
        lepton_dataset="muon",
        lepton_triggers=bbtautau_vars.HLT_muons,
        isLepton=True,
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
    "vbfbbtt-k2v0": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-k2v0"],
        label=r"VBF HHbb$\tau\tau$ ($\kappa_{2V}=0$)",
        isSignal=True,
    ),
}

SIGNALS = ["bbtt", "vbfbbtt-k2v0"]
SIGNALS_CHANNELS = ["bbtt", "vbfbbtt-k2v0"]

# add individual bbtt channels
for signal in SIGNALS.copy():
    for channel in CHANNELS:
        SAMPLES[f"{signal}{channel}"] = Sample(
            label=SAMPLES[signal].label,
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
