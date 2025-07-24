"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from boostedhh.processors.objects import jetid_v12
from boostedhh.processors.utils import PDGID
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    JetArray,
    MuonArray,
    TauArray,
)

from bbtautau.HLTs import HLTs


def trig_match_sel(events, leptons, trig_leptons, year, trigger, filterbit, ptcut, trig_dR=0.2):
    """
    Returns selection for leptons which are trigger matched to the specified trigger.
    """
    trigger = HLTs.hlts_by_type(year, trigger, hlt_prefix=False)[0]  # picking first trigger in list
    trig_fired = events.HLT[trigger]
    # print(f"{trigger} rate: {ak.mean(trig_fired)}")

    filterbit = 2**filterbit

    pass_trig = (trig_leptons.filterBits & filterbit) == filterbit
    trig_l = trig_leptons[pass_trig]
    trig_l_matched = ak.any(leptons.metric_table(trig_l) < trig_dR, axis=2)
    trig_l_sel = trig_fired & trig_l_matched & (leptons.pt > ptcut)
    return trig_l_sel


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)

    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass

    # gloParT 2
    fatjets["globalParT2_QCD"] = (
        fatjets.globalParT2_QCD0HF + fatjets.globalParT2_QCD1HF + fatjets.globalParT2_QCD2HF
    )
    fatjets["globalParT2_Top"] = fatjets.globalParT2_TopW + fatjets.globalParT2_TopbW

    fatjets["particleNetLegacy_XbbvsQCD"] = fatjets.particleNetLegacy_Xbb / (
        fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
    )
    fatjets["globalParT2_XbbvsQCD"] = fatjets.globalParT2_Xbb / (
        fatjets.globalParT2_Xbb + fatjets["globalParT2_QCD"]
    )
    fatjets["globalParT2_XbbvsQCDTop"] = fatjets.globalParT2_Xbb / (
        fatjets.globalParT2_Xbb + fatjets["globalParT2_QCD"] + fatjets["globalParT2_Top"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT2_X{tautau}vsQCD"] = fatjets[f"globalParT2_X{tautau}"] / (
            fatjets[f"globalParT2_X{tautau}"] + fatjets["globalParT2_QCD"]
        )
        fatjets[f"globalParT2_X{tautau}vsQCDTop"] = fatjets[f"globalParT2_X{tautau}"] / (
            fatjets[f"globalParT2_X{tautau}"]
            + fatjets["globalParT2_QCD"]
            + fatjets["globalParT2_Top"]
        )

    fatjets["globalParT2_massResCorr"] = fatjets.globalParT2_massRes
    fatjets["globalParT2_massVisCorr"] = fatjets.globalParT2_massVis
    fatjets["globalParT2_massResApplied"] = (
        fatjets.globalParT2_massRes * (1 - fatjets.rawFactor) * fatjets.mass
    )
    fatjets["globalParT2_massVisApplied"] = (
        fatjets.globalParT2_massVis * (1 - fatjets.rawFactor) * fatjets.mass
    )

    # gloParT 3
    fatjets["globalParT3_Top"] = (
        fatjets.globalParT3_TopbWev
        + fatjets.globalParT3_TopbWmv
        + fatjets.globalParT3_TopbWq
        + fatjets.globalParT3_TopbWqq
        + fatjets.globalParT3_TopbWtauhv
    )

    fatjets["globalParT3_XbbvsQCD"] = fatjets.globalParT3_Xbb / (
        fatjets.globalParT3_Xbb + fatjets["globalParT3_QCD"]
    )
    fatjets["globalParT3_XbbvsQCDTop"] = fatjets.globalParT3_Xbb / (
        fatjets.globalParT3_Xbb + fatjets["globalParT3_QCD"] + fatjets["globalParT3_Top"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT3_X{tautau}vsQCD"] = fatjets[f"globalParT3_X{tautau}"] / (
            fatjets[f"globalParT3_X{tautau}"] + fatjets["globalParT3_QCD"]
        )
        fatjets[f"globalParT3_X{tautau}vsQCDTop"] = fatjets[f"globalParT3_X{tautau}"] / (
            fatjets[f"globalParT3_X{tautau}"]
            + fatjets["globalParT3_QCD"]
            + fatjets["globalParT3_Top"]
        )

    fatjets["globalParT3_massGenericCorr"] = fatjets.globalParT3_massGeneric
    fatjets["globalParT3_massX2pCorr"] = fatjets.globalParT3_massX2p
    fatjets["globalParT3_massGenericApplied"] = (
        fatjets.globalParT3_massGeneric * (1 - fatjets.rawFactor) * fatjets.mass
    )
    fatjets["globalParT3_massX2pApplied"] = (
        fatjets.globalParT3_massX2p * (1 - fatjets.rawFactor) * fatjets.mass
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
    nano_version: str,  # noqa: ARG001
    mreg_str: str = "particleNet_mass_legacy",  # noqa: ARG001
):
    # if nano_version.startswith("v12"):
    #     jetidtight, jetidtightlepveto = jetid_v12(fatjets)  # v12 jetid fix
    # else:
    #     raise NotImplementedError(f"Jet ID fix not implemented yet for {nano_version}")

    # Data does not have .neHEF etc. fields for fatjets, so above recipe doesn't work
    # Either way, doesn't matter since we only use tightID, and it is correct for eta < 2.7
    jetidtight = fatjets.isTight

    fatjet_sel = (
        jetidtight
        & (fatjets.pt > object_pt)
        & (abs(fatjets.eta) < eta)
        # & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
    )
    return fatjets[fatjet_sel]


def good_ak4jets(jets: JetArray, nano_version: str):
    if nano_version.startswith("v12"):
        jetidtight, jetidtightlepveto = jetid_v12(jets)  # v12 jetid fix
    else:
        raise NotImplementedError(f"Jet ID fix not implemented yet for {nano_version}")
    jet_sel = (jets.pt > 15) & (np.abs(jets.eta) < 4.7) & jetidtight & jetidtightlepveto

    return jets[jet_sel]


"""
Trigger quality bits in NanoAOD v12
0 => CaloIdL_TrackIdL_IsoVL,
1 => 1e (WPTight),
2 => 1e (WPLoose),
3 => OverlapFilter PFTau,
4 => 2e,
5 => 1e-1mu,
6 => 1e-1tau,
7 => 3e,
8 => 2e-1mu,
9 => 1e-2mu,
10 => 1e (32_L1DoubleEG_AND_L1SingleEGOr),
11 => 1e (CaloIdVT_GsfTrkIdT),
12 => 1e (PFJet),
13 => 1e (Photon175_OR_Photon200) for Electron;
"""


def good_electrons(events, leptons: ElectronArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    # baseline kinematic selection
    lsel = (
        leptons.mvaIso_WP90
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.5)
        & (abs(leptons.dz) < 0.2)
        & (abs(leptons.dxy) < 0.045)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"EGamma": (1, 31), "ETau": (6, 25)}
    trig_leptons = trigobj[trigobj.id == PDGID.e]

    TrigMatchDict = {
        f"ElectronTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => TrkIsoVVL,
1 => Iso,
2 => OverlapFilter PFTau,
3 => 1mu,
4 => 2mu,
5 => 1mu-1e,
6 => 1mu-1tau,
7 => 3mu,
8 => 2mu-1e,
9 => 1mu-2e,
10 => 1mu (Mu50),
11 => 1mu (Mu100),
12 => 1mu-1photon for Muon;
"""


def good_muons(events, leptons: MuonArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    lsel = (
        leptons.tightId
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.4)
        & (abs(leptons.dz) < 0.2)
        & (abs(leptons.dxy) < 0.045)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"Muon": (3, 26), "MuonTau": (6, 22)}
    trig_leptons = trigobj[trigobj.id == PDGID.mu]

    TrigMatchDict = {
        f"MuonTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => LooseChargedIso,
1 => MediumChargedIso,
2 => TightChargedIso,
3 => DeepTau,
4 => TightID OOSC photons,
5 => HPS,
6 => charged iso di-tau,
7 => deeptau di-tau,
8 => e-tau,
9 => mu-tau,
10 => single-tau/tau+MET,
11 => run 2 VBF+ditau,
12 => run 3 VBF+ditau,
13 => run 3 double PF jets + ditau,
14 => di-tau + PFJet,
15 => Displaced Tau,
16 => Monitoring,
17 => regional paths,
18 => L1 seeded paths,
19 => 1 prong tau paths for Tau;
"""


def good_taus(events, leptons: TauArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    lsel = (
        (leptons.idDeepTau2018v2p5VSjet >= 5)
        # & (leptons.idDeepTau2018v2p5VSe >= 3)
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.5)
        & (abs(leptons.dz) < 0.2)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"SingleTau": (10, 185), "DiTau": (7, 37), "ETau": (8, 32), "MuonTau": (9, 30)}
    trig_leptons = trigobj[trigobj.id == PDGID.tau]

    TrigMatchDict = {
        f"TauTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => HLT_AK8PFJetX_SoftDropMass40_PFAK8ParticleNetTauTau0p30,
1 => hltAK8SinglePFJets230SoftDropMass40PNetTauTauTag0p03 for BoostedTau;
"""


def good_boostedtaus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 20) & (abs(taus.eta) < 2.5)
    return taus[tau_sel]
