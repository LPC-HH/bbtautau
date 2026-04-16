from __future__ import annotations

from boostedhh import hh_vars
from boostedhh.utils import Sample

from bbtautau.postprocessing.bbtautau_types import Channel

CHANNELS = {  # in alphabetical order
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
        txtt_BDT_cut=0.9918,
        tt_mass_cut=("ttFatJetParTmassResApplied", [70, 210]),
    ),
    "hh": Channel(
        key="hh",
        label=r"$\tau_h\tau_h$",
        hlt_types=[
            "PNet",
            "PFJet",
            "QuadJet",
            # "Parking", # remove for the moment, does not make a big difference.
            "DiTau",
            "DitauJet",
            "SingleTau",
            "MET",
        ],  # Probably remove parking
        data_samples=["jetmet", "tau"],
        isLepton=False,
        tagger_label="tauhtauh",
        txbb_cut=0.91,
        txtt_cut=0.999,
        txtt_BDT_cut=0.996,
        tt_mass_cut=("ttFatJetPNetmassLegacy", [50, 150]),
    ),
    "hm": Channel(
        key="hm",
        label=r"$\tau_h \mu$",
        hlt_types=["PNet", "PFJet", "Muon", "MuonTau", "DiTau", "DitauJet", "SingleTau", "MET"],
        data_samples=["jetmet", "tau", "muon"],
        lepton_dataset="muon",
        isLepton=True,
        tagger_label="tauhtaum",
        txbb_cut=0.85,
        txtt_cut=0.99,
        txtt_BDT_cut=0.8,
        tt_mass_cut=("ttFatJetParTmassResApplied", [70, 210]),
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
    "ggfbbtt": Sample(
        selector=hh_vars.bbtt_sigs["ggfbbtt"],
        label=r"ggF HHbb$\tau\tau$",
        isSignal=True,
    ),
    "ggfbbtt-kl0p00": Sample(
    selector=hh_vars.bbtt_sigs["ggfbbtt-kl0p00"],
    label=r"ggF HHbb$\tau\tau$ ($\kappa_\lambda=0$)",
    isSignal=True,
    ),
    "ggfbbtt-kl2p45": Sample(
        selector=hh_vars.bbtt_sigs["ggfbbtt-kl2p45"],
        label=r"ggF HHbb$\tau\tau$ ($\kappa_\lambda=2.45$)",
        isSignal=True,
    ),
    "ggfbbtt-kl5p00": Sample(
        selector=hh_vars.bbtt_sigs["ggfbbtt-kl5p00"],
        label=r"ggF HHbb$\tau\tau$ ($\kappa_\lambda=5.00$)",
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
    # BSM VBF samples (no own regions - contribute to ggf/vbf like vbfbbtt-k2v0)
    "vbfbbtt-kv1p74-k2v1p37-kl14p4": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-kv1p74-k2v1p37-kl14p4"],
        label=r"VBF HHbb$\tau\tau$ (BSM)",
        isSignal=True,
    ),
    "vbfbbtt-kvm0p758-k2v1p44-klm19p3": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-kvm0p758-k2v1p44-klm19p3"],
        label=r"VBF HHbb$\tau\tau$ (BSM)",
        isSignal=True,
    ),
    "vbfbbtt-kvm0p962-k2v0p959-klm1p43": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-kvm0p962-k2v0p959-klm1p43"],
        label=r"VBF HHbb$\tau\tau$ (BSM)",
        isSignal=True,
    ),
    "vbfbbtt-kvm1p6-k2v2p72-klm1p36": Sample(
        selector=hh_vars.bbtt_sigs["vbfbbtt-kvm1p6-k2v2p72-klm1p36"],
        label=r"VBF HHbb$\tau\tau$ (BSM)",
        isSignal=True,
    ),
}

# Do not include the unified SM signal below
SIGNALS = [
    "ggfbbtt",
    "ggfbbtt-kl0p00",
    "ggfbbtt-kl2p45",
    "ggfbbtt-kl5p00",
    "vbfbbtt",
    "vbfbbtt-k2v0",
    "vbfbbtt-kv1p74-k2v1p37-kl14p4",
    "vbfbbtt-kvm0p758-k2v1p44-klm19p3",
    "vbfbbtt-kvm0p962-k2v0p959-klm1p43",
    "vbfbbtt-kvm1p6-k2v2p72-klm1p36",
]
SIGNALS_CHANNELS = []

SM_SIGNALS = ["ggfbbtt", "vbfbbtt"]
SM_SIGNALS_CHANNELS = []

sig_keys_ggf = ["ggfbbtt", "ggfbbtt-kl0p00", "ggfbbtt-kl2p45", "ggfbbtt-kl5p00"]
sig_keys_vbf = [
    "vbfbbtt",
    "vbfbbtt-k2v0",
    "vbfbbtt-kv1p74-k2v1p37-kl14p4",
    "vbfbbtt-kvm0p758-k2v1p44-klm19p3",
    "vbfbbtt-kvm0p962-k2v0p959-klm1p43",
    "vbfbbtt-kvm1p6-k2v2p72-klm1p36",
]

# add individual bbtt channels
for signal in SIGNALS:
    for channel, CHANNEL in CHANNELS.items():
        SAMPLES[f"{signal}{channel}"] = Sample(
            label=SAMPLES[signal].label.replace(r"$\tau\tau$", CHANNEL.label),
            isSignal=True,
        )
        SIGNALS_CHANNELS.append(f"{signal}{channel}")
        if signal in SM_SIGNALS:
            SM_SIGNALS_CHANNELS.append(f"{signal}{channel}")

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

qcdouts = ["QCD", "QCD0HF", "QCD1HF", "QCD2HF"]
topouts = ["Top", "TopW", "TopbW", "TopbWev", "TopbWmv", "TopbWtauhv", "TopbWq", "TopbWqq"]
sigouts = ["Xtauhtauh", "Xtauhtaue", "Xtauhtaum", "Xbb"]
