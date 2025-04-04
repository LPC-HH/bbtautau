"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    MuonArray,
    TauArray,
)


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)
    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass
    fatjets["globalParT_QCD"] = (
        fatjets.globalParT_QCD0HF + fatjets.globalParT_QCD1HF + fatjets.globalParT_QCD2HF
    )
    fatjets["globalParT_XbbvsQCD"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT_{tautau}vsQCD"] = fatjets[f"globalParT_{tautau}"] / (
            fatjets[f"globalParT_{tautau}"] + fatjets["globalParT_QCD"]
        )

    fatjets["globalParT_massResCorr"] = fatjets.globalParT_massRes
    fatjets["globalParT_massVisCorr"] = fatjets.globalParT_massVis
    fatjets["globalParT_massResApplied"] = (
        fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
    )
    fatjets["globalParT_massVisApplied"] = (
        fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass
    )
    return fatjets


# ak8 jet definition
def good_ak8jets(
    fatjets: FatJetArray,
    object_pt: float,  # select objects based on this
    pt: float,  # make event selections based on this  # noqa: ARG001
    eta: float,
    msd: float,  # noqa: ARG001
    mreg: float,  # noqa: ARG001
    mreg_str="particleNet_mass_legacy",  # noqa: ARG001
):
    fatjet_sel = (
        fatjets.isTight
        & (fatjets.pt > object_pt)
        & (abs(fatjets.eta) < eta)
        # & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
    )
    return fatjets[fatjet_sel]


def good_electrons(events, electrons: ElectronArray):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    if events["HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1"]:
        ptcut = 25
    elif events["HLT_Ele30_WPTight_Gsf"]:
        ptcut = 31
    else:
        ptcut = 20

    ele_sel = (
        electrons.isTight
        & (electrons.pt > ptcut)
        & (abs(electrons.eta) < 2.5)
        & (abs(electrons.dz) < 0.2)
        & (abs(electrons.dxy) < 0.045)
    )

    return electrons[ele_sel]


def good_muons(events, muons: MuonArray):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    if events["HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1"]:
        ptcut = 22
    elif events["HLT_IsoMu24"]:
        ptcut = 26
    else:
        ptcut = 20

    muon_sel = (
        muons.isTight
        & (muons.pt > ptcut)
        & (abs(muons.eta) < 2.5)
        & (abs(muons.dz) < 0.2)
        & (abs(muons.dxy) < 0.045)
    )

    return muons[muon_sel]


def good_taus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 32) & (abs(taus.eta) < 2.5) & (abs(taus.dz) < 0.2)
    return taus[tau_sel]


def good_boostedtaus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 20) & (abs(taus.eta) < 2.5)
    return taus[tau_sel]
