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
    MissingET,
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

    fatjets["globalParT_QCD"] = (
        fatjets.globalParT_QCD0HF + fatjets.globalParT_QCD1HF + fatjets.globalParT_QCD2HF
    )
    fatjets["globalParT_Top"] = fatjets.globalParT_TopW + fatjets.globalParT_TopbW

    fatjets["particleNetLegacy_XbbvsQCD"] = fatjets.particleNetLegacy_Xbb / (
        fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
    )
    fatjets["globalParT_XbbvsQCD"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"]
    )
    fatjets["globalParT_XbbvsQCDTop"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"] + fatjets["globalParT_Top"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT_X{tautau}vsQCD"] = fatjets[f"globalParT_X{tautau}"] / (
            fatjets[f"globalParT_X{tautau}"] + fatjets["globalParT_QCD"]
        )
        fatjets[f"globalParT_X{tautau}vsQCDTop"] = fatjets[f"globalParT_X{tautau}"] / (
            fatjets[f"globalParT_X{tautau}"] + fatjets["globalParT_QCD"] + fatjets["globalParT_Top"]
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
        leptons.mvaNoIso_WP90
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


def vbf_jets(
    # from HH4b https://github.com/LPC-HH/HH4b/blob/9d8038bcc31bf1872352e332eff84a4602934b3e/src/HH4b/processors/objects.py#L335
    jets: JetArray,
    fatjets: FatJetArray,
    events,
    pt: float,
    id: str,  # noqa: ARG001
    eta_max: float,
    dr_fatjets: float,
    dr_leptons: float,
    electron_pt: float,
    muon_pt: float,
):
    """Top 2 jets in pT passing the VBF selections"""
    electrons = events.Electron
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon
    muons = muons[muons.pt > muon_pt]

    ak4_sel = (
        (jets.pt >= pt)
        & (np.abs(jets.eta) <= eta_max)
        & (ak.all(jets.metric_table(fatjets) > dr_fatjets, axis=2))
        & ak.all(jets.metric_table(electrons) > dr_leptons, axis=2)
        & ak.all(jets.metric_table(muons) > dr_leptons, axis=2)
    )

    return jets[ak4_sel][:, :2]


def ak4_jets_awayfromak8(
    jets: JetArray,
    fatjets: FatJetArray,
    events,
    pt: float,
    id: str,  # noqa: ARG001
    eta_max: float,
    dr_fatjets: float,
    dr_leptons: float,
    electron_pt: float,
    muon_pt: float,
    sort_by: str = "btag",
):
    """AK4 jets nonoverlapping with AK8 fatjets"""
    electrons = events.Electron
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon
    muons = muons[muons.pt > muon_pt]

    ak4_sel = (
        (jets.pt >= pt)
        & (np.abs(jets.eta) <= eta_max)
        & (ak.all(jets.metric_table(fatjets) > dr_fatjets, axis=2))
        & ak.all(jets.metric_table(electrons) > dr_leptons, axis=2)
        & ak.all(jets.metric_table(muons) > dr_leptons, axis=2)
    )

    # return top 2 jets sorted by btagPNetB
    if sort_by == "btag":
        jets_pnetb = jets[ak.argsort(jets.btagPNetB, ascending=False)]
        return jets_pnetb[ak4_sel][:, :2]
    # return 2 jets closet to bbFatjet and ttFatjet, respectively
    elif sort_by == "nearest":
        jets_away = jets[ak4_sel]
        bbFatjet = ak.firsts(
            fatjets[
                ak.argsort(
                    fatjets["globalParT_XbbvsQCDTop"],
                    ascending=False,
                )
            ][:, 0:1]
        )

        ttFatjet = ak.firsts(
            fatjets[
                ak.argsort(
                    sum(
                        fatjets[f"globalParT_X{tautau}vsQCDTop"]
                        for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]
                    ),
                    ascending=False,
                )
            ][:, 0:1]
        )

        jet_near_bbFatjet = jets_away[ak.argsort(jets_away.delta_r(bbFatjet), ascending=True)][
            :, 0:1
        ]
        jet_near_ttFatjet = jets_away[ak.argsort(jets_away.delta_r(ttFatjet), ascending=True)][
            :, 0:1
        ]
        return [jet_near_bbFatjet, jet_near_ttFatjet]
    # return all nonoverlapping jets, no sorting
    else:
        return jets[ak4_sel]


# adopted from https://github.com/scikit-hep/coffea/blob/a315da1fa307f1ec0d21c29e908e5b733603d7c0/src/coffea/nanoevents/methods/vector.py#L106
def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(deta, dphi)


def CA_got(
    met_pt,
    met_phi,
    fatjets_mass,
    fatjets_masscorr,
    tau0_eta,
    tau1_eta,
    tau0_phi,
    tau1_phi,
    tau0_pt,
    tau1_pt,
):
    invalid = (
        (met_pt == -999)
        | (met_phi == -999)
        | (fatjets_mass == -999)
        | (fatjets_masscorr == -999)
        | (tau0_eta == -999)
        | (tau1_eta == -999)
        | (tau0_phi == -999)
        | (tau1_phi == -999)
        | (tau0_pt == -999)
        | (tau1_pt == -999)
    )
    # indeed, they are arrays, and if
    # tau1_phi is [[-999, -999], [1.82, 1.82]]
    # tau1_phi == -999 is [[True, True], [False, False]]

    dphi1 = met_phi - tau0_phi
    dphi0 = tau1_phi - met_phi
    dphi = tau0_phi - tau1_phi

    sin_dphi0 = np.sin(dphi0)
    sin_dphi1 = np.sin(dphi1)
    sin_dphi = np.sin(dphi)

    pmet_tau0 = np.abs(met_pt * sin_dphi0 / sin_dphi)
    pmet_tau1 = np.abs(met_pt * sin_dphi1 / sin_dphi)

    denom = np.sqrt(
        np.abs(tau0_pt / (tau0_pt + pmet_tau0)) * np.abs(tau1_pt / (tau1_pt + pmet_tau1))
    )
    denom = ak.where(denom == 0, 1, denom)

    mass = fatjets_mass * fatjets_masscorr / denom
    mass = ak.where(invalid, -999, mass)
    return mass


def calculate_invariant_mass(
    fatjets_mass,
    fatjets_pt,
    fatjets_eta,
    fatjets_phi,
    fatjets_masscorr,
    tau1_pt,
    tau1_eta,
    tau1_phi,
    met_pt,
    met_phi,
):
    invalid = (
        (fatjets_mass == -999)
        | (fatjets_pt == -999)
        | (fatjets_eta == -999)
        | (fatjets_phi == -999)
        | (fatjets_masscorr == -999)
        | (tau1_pt == -999)
        | (tau1_eta == -999)
        | (tau1_phi == -999)
        | (met_pt == -999)
        | (met_phi == -999)
    )

    fatjets_mass = fatjets_mass * fatjets_masscorr

    E_fatjet = np.sqrt(fatjets_pt**2 + fatjets_mass**2)
    E_tau1 = np.sqrt(tau1_pt**2 + 0)
    E_met = met_pt

    px_fatjet = fatjets_pt * np.cos(fatjets_phi)
    py_fatjet = fatjets_pt * np.sin(fatjets_phi)
    pz_fatjet = fatjets_pt * np.sinh(fatjets_eta)

    px_tau1 = tau1_pt * np.cos(tau1_phi)
    py_tau1 = tau1_pt * np.sin(tau1_phi)
    pz_tau1 = tau1_pt * np.sinh(tau1_eta)

    px_met = met_pt * np.cos(met_phi)
    py_met = met_pt * np.sin(met_phi)

    px_total = px_fatjet + px_tau1 + px_met
    py_total = py_fatjet + py_tau1 + py_met
    pz_total = pz_fatjet + pz_tau1

    E_total = E_fatjet + E_tau1 + E_met

    mass_squared = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    mass = np.sqrt(mass_squared)

    mass = ak.where(invalid, -999, mass)
    return mass


def calculate_invariant_mass_2d(
    fatjets_mass, fatjets_pt, fatjets_eta, fatjets_phi, fatjets_masscorr, met_pt, met_phi
):
    invalid = (
        (fatjets_mass == -999)
        | (fatjets_pt == -999)
        | (fatjets_eta == -999)
        | (fatjets_phi == -999)
        | (fatjets_masscorr == -999)
        | (met_pt == -999)
        | (met_phi == -999)
    )

    fatjets_mass = fatjets_mass * fatjets_masscorr

    E_fatjet = np.sqrt(fatjets_pt**2 + fatjets_mass**2)
    E_met = met_pt

    px_fatjet = fatjets_pt * np.cos(fatjets_phi)
    py_fatjet = fatjets_pt * np.sin(fatjets_phi)
    pz_fatjet = fatjets_pt * np.sinh(fatjets_eta)

    px_met = met_pt * np.cos(met_phi)
    py_met = met_pt * np.sin(met_phi)

    px_total = px_fatjet + px_met
    py_total = py_fatjet + py_met
    pz_total = pz_fatjet

    E_total = E_fatjet + E_met

    mass_squared = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    mass = np.sqrt(mass_squared)

    mass = ak.where(invalid, -999, mass)
    return mass


def dRdau(
    eta0,
    phi0,
    eta1,
    phi1,
):
    invalid = (eta0 == -999) | (phi0 == -999) | (eta1 == -999) | (phi1 == -999)

    deta = eta0 - eta1
    dphi = (phi0 - phi1 + np.pi) % (2 * np.pi) - np.pi

    dr = np.hypot(deta, dphi)
    dr = ak.where(invalid, -1, dr)
    return dr


def get_CA_MASS(
    fatjets: FatJetArray,
    taus: TauArray,
    met: MissingET,
    subjets: JetArray,
    muons: MuonArray,
    electrons: ElectronArray,
):

    init_fields = {
        "CA_mass_merged": (-999.0, float),
        "CA_msoftdrop_merged": (-999.0, float),
        "CA_globalParT_massVisApplied_merged": (-999.0, float),
        "CA_globalParT_massResApplied_merged": (-999.0, float),
        "CA_particleNet_mass_legacy_merged": (-999.0, float),
        "CA_Tauflag": (0, int),
        "CA_one_elec_in_fatjet": (0, int),
        "CA_one_muon_in_fatjet": (0, int),
        "CA_one_elec": (0, int),
        "CA_one_muon": (0, int),
        "CA_mass_boostedtaus": (-999.0, float),
        "CA_ntaus_perfatjets": (-1, int),
        "CA_mass_subjets": (-999.0, float),
        "CA_nsubjets_perfatjets": (-1, int),
        "CA_mass_fatjets": (-999.0, float),
        "CA_mass": (-999.0, float),
        "CA_msoftdrop": (-999.0, float),
        "CA_globalParT_massVisApplied": (-999.0, float),
        "CA_globalParT_massResApplied": (-999.0, float),
        "CA_particleNet_mass_legacy": (-999.0, float),
        "CA_isDauTau": (0, int),
        "CA_dau0_pt": (-999.0, float),
        "CA_dau1_pt": (-999.0, float),
        "CA_dau0_eta": (-999.0, float),
        "CA_dau1_eta": (-999.0, float),
        "CA_dau0_phi": (-999.0, float),
        "CA_dau1_phi": (-999.0, float),
        "CA_dau0_mass": (-999.0, float),
        "CA_dau1_mass": (-999.0, float),
        ##mt-channel
        "CA_mass_boostedtaus_mt": (-999.0, float),
        "CA_ntaus_perfatjets_mt": (-1, int),
        "CA_mass_subjets_mt": (-999.0, float),
        "CA_nsubjets_perfatjets_mt": (-1, int),
        "CA_mass_fatjet_mt": (-999.0, float),
        "CA_muon_subjet_dr02": (-1, int),
        "CA_mass_subjets_mt_1": (-999.0, float),
        "CA_mass_subjets_mt_0": (-999.0, float),
        "CA_mass_subjets_mt_01": (-999.0, float),
        "CA_mass_mt": (-999.0, float),
        "CA_msoftdrop_mt": (-999.0, float),
        "CA_globalParT_massVisApplied_mt": (-999.0, float),
        "CA_globalParT_massResApplied_mt": (-999.0, float),
        "CA_particleNet_mass_legacy_mt": (-999.0, float),
        "CA_isDauTau_mt": (0, int),
        "CA_dau0_pt_mt": (-999.0, float),
        "CA_dau1_pt_mt": (-999.0, float),
        "CA_dau0_eta_mt": (-999.0, float),
        "CA_dau1_eta_mt": (-999.0, float),
        "CA_dau0_phi_mt": (-999.0, float),
        "CA_dau1_phi_mt": (-999.0, float),
        "CA_dau0_mass_mt": (-999.0, float),
        "CA_dau1_mass_mt": (-999.0, float),
        ##et-channel
        "CA_mass_boostedtaus_et": (-999.0, float),
        "CA_ntaus_perfatjets_et": (-1, int),
        "CA_mass_subjets_et": (-999.0, float),
        "CA_nsubjets_perfatjets_et": (-1, int),
        "CA_mass_fatjet_et": (-999.0, float),
        "CA_elec_subjet_dr02": (-1, int),
        "CA_mass_subjets_et_1": (-999.0, float),
        "CA_mass_subjets_et_0": (-999.0, float),
        "CA_mass_subjets_et_01": (-999.0, float),
        "CA_mass_et": (-999.0, float),
        "CA_msoftdrop_et": (-999.0, float),
        "CA_globalParT_massVisApplied_et": (-999.0, float),
        "CA_globalParT_massResApplied_et": (-999.0, float),
        "CA_particleNet_mass_legacy_et": (-999.0, float),
        "CA_isDauTau_et": (0, int),
        "CA_dau0_pt_et": (-999.0, float),
        "CA_dau1_pt_et": (-999.0, float),
        "CA_dau0_eta_et": (-999.0, float),
        "CA_dau1_eta_et": (-999.0, float),
        "CA_dau0_phi_et": (-999.0, float),
        "CA_dau1_phi_et": (-999.0, float),
        "CA_dau0_mass_et": (-999.0, float),
        "CA_dau1_mass_et": (-999.0, float),
    }

    # basic number info
    # n_events = len(fatjets)
    n_fatjets = ak.num(fatjets, axis=1)
    n_taus = ak.num(taus, axis=1)
    # n_subjets = len(subjets)

    # more than one fatjet or tau
    has_fatjets = n_fatjets > 0
    has_taus = n_taus > 0
    can_match = has_fatjets & has_taus

    for name, (default, dtype) in init_fields.items():
        fatjets[name] = ak.full_like(fatjets.pt, default, dtype=dtype)

    # veto of taus/subjets/leptons
    no2tau = ak.full_like(fatjets.pt, False, dtype=bool)
    no2subjet = ak.full_like(fatjets.pt, False, dtype=bool)
    no1tau = ak.full_like(fatjets.pt, False, dtype=bool)
    no1subjet = ak.full_like(fatjets.pt, False, dtype=bool)
    no1muon = ak.full_like(fatjets.pt, False, dtype=bool)
    no1electron = ak.full_like(fatjets.pt, False, dtype=bool)

    # MET info
    met_pt = met.pt
    met_phi = met.phi

    if ak.any(can_match):

        # fatjet info
        fatjets_pt = fatjets.pt
        fatjets_eta = fatjets.eta
        fatjets_phi = fatjets.phi

        fatjets_mass = fatjets.mass
        fatjets_msoftdrop = fatjets.msoftdrop

        # fatjets_globalParT_massVis = fatjets.globalParT_massVis
        # fatjets_globalParT_massRes = fatjets.globalParT_massRes
        fatjets_globalParT_massResApplied = fatjets.globalParT_massResApplied
        fatjets_globalParT_massVisApplied = fatjets.globalParT_massVisApplied
        fatjets_particleNet_mass_legacy = fatjets.particleNetLegacy_mass

        fatjets_masscorr = fatjets.particleNet_massCorr
        fake_corr = ak.full_like(fatjets_masscorr, 1.0, dtype=float)

        # for muon

        ##fatjet_muon matching
        fatjet_muon_pairs = ak.cartesian([fatjets, muons], nested=True)
        fatjets_in_pairs = fatjet_muon_pairs["0"]
        muons_in_pairs = fatjet_muon_pairs["1"]

        ##dr_muons_fatjets
        dR_muons = delta_r(
            fatjets_in_pairs.eta, fatjets_in_pairs.phi, muons_in_pairs.eta, muons_in_pairs.phi
        )
        ##select muons for each fatjet
        close_matches_muons = dR_muons < 0.8
        matched_muons_per_fatjet = muons_in_pairs[close_matches_muons]
        ##flag for muons in fatjet (to make sure mt)
        n_matched_muons = ak.num(matched_muons_per_fatjet, axis=-1)
        no1muon = n_matched_muons < 1
        ##flag for muons (make sure m)
        n_muons = ak.num(muons_in_pairs, axis=-1)
        no1muon_ori = n_muons < 1
        ##1st and 2nd muons
        sorted_indices = ak.argsort(matched_muons_per_fatjet.pt, axis=-1, ascending=False)
        sorted_muons = matched_muons_per_fatjet[sorted_indices]
        top2_muons = ak.pad_none(sorted_muons, 2, axis=-1)[..., :2]

        # info of 1st muon
        muon0_eta = ak.fill_none(top2_muons.eta[..., 0], -999)
        muon0_phi = ak.fill_none(top2_muons.phi[..., 0], -999)
        muon0_mass = ak.fill_none(top2_muons.mass[..., 0], -999)
        muon0_pt = ak.fill_none(top2_muons.pt[..., 0], -999)

        # for electron

        ##fatjet_electron matching
        fatjet_electron_pairs = ak.cartesian([fatjets, electrons], nested=True)
        fatjets_in_pairs = fatjet_electron_pairs["0"]
        electrons_in_pairs = fatjet_electron_pairs["1"]
        ##dr_electrons_fatjets
        dR_electrons = delta_r(
            fatjets_in_pairs.eta,
            fatjets_in_pairs.phi,
            electrons_in_pairs.eta,
            electrons_in_pairs.phi,
        )
        ##select electrons for each fatjet
        close_matches_electrons = dR_electrons < 0.8
        matched_electrons_per_fatjet = electrons_in_pairs[close_matches_electrons]
        ##flag for electrons in fatjet (to make sure mt)
        n_matched_electrons = ak.num(matched_electrons_per_fatjet, axis=-1)
        no1electron = n_matched_electrons < 1
        ##flag for electrons in fatjet
        n_electrons = ak.num(electrons_in_pairs, axis=-1)
        no1electron_ori = n_electrons < 1
        ##1st and 2nd electrons
        sorted_indices = ak.argsort(matched_electrons_per_fatjet.pt, axis=-1, ascending=False)
        sorted_electrons = matched_electrons_per_fatjet[sorted_indices]
        top2_electrons = ak.pad_none(sorted_electrons, 2, axis=-1)[..., :2]

        # info of 1st electron
        electron0_eta = ak.fill_none(top2_electrons.eta[..., 0], -999)
        electron0_phi = ak.fill_none(top2_electrons.phi[..., 0], -999)
        electron0_mass = ak.fill_none(top2_electrons.mass[..., 0], -999)
        electron0_pt = ak.fill_none(top2_electrons.pt[..., 0], -999)

        # for subjet

        ##fatjet_subjet matching
        fatjet_subjet_pairs = ak.cartesian([fatjets, subjets], nested=True)
        fatjets_in_pairs = fatjet_subjet_pairs["0"]
        subjets_in_pairs = fatjet_subjet_pairs["1"]

        dR_subjets = delta_r(
            fatjets_in_pairs.eta, fatjets_in_pairs.phi, subjets_in_pairs.eta, subjets_in_pairs.phi
        )

        close_matches_subjets = dR_subjets < 0.8

        matched_subjets_per_fatjet = subjets_in_pairs[close_matches_subjets]

        n_matched_subjets = ak.num(matched_subjets_per_fatjet, axis=-1)
        no2subjet = n_matched_subjets < 2
        no1subjet = n_matched_subjets < 1

        sorted_indices = ak.argsort(matched_subjets_per_fatjet.pt, axis=-1, ascending=False)
        sorted_subjets = matched_subjets_per_fatjet[sorted_indices]
        top2_subjets = ak.pad_none(sorted_subjets, 2, axis=-1)[..., :2]

        subjet0_eta = ak.fill_none(top2_subjets.eta[..., 0], -999)
        subjet1_eta = ak.fill_none(top2_subjets.eta[..., 1], -999)
        subjet0_phi = ak.fill_none(top2_subjets.phi[..., 0], -999)
        subjet1_phi = ak.fill_none(top2_subjets.phi[..., 1], -999)
        subjet0_mass = ak.fill_none(top2_subjets.mass[..., 0], -999)
        subjet1_mass = ak.fill_none(top2_subjets.mass[..., 1], -999)
        subjet0_pt = ak.fill_none(top2_subjets.pt[..., 0], -999)
        subjet1_pt = ak.fill_none(top2_subjets.pt[..., 1], -999)

        ##flag whose dR_subjet_lepton<0.8
        dR_elec_vs_subjet0 = dRdau(subjet0_eta, electron0_eta, subjet0_phi, electron0_phi) < 0.2
        dR_muon_vs_subjet0 = dRdau(subjet0_eta, muon0_eta, subjet0_phi, muon0_phi) < 0.2
        dR_elec_vs_subjet1 = dRdau(subjet1_eta, electron0_eta, subjet1_phi, electron0_phi) < 0.2
        dR_muon_vs_subjet1 = dRdau(subjet1_eta, muon0_eta, subjet1_phi, muon0_phi) < 0.2

        # for tau

        ##fatjet_tau matching
        fatjet_boostedtau_pairs = ak.cartesian([fatjets, taus], nested=True)
        fatjets_in_pairs = fatjet_boostedtau_pairs["0"]
        boostedtaus_in_pairs = fatjet_boostedtau_pairs["1"]

        dR = delta_r(
            fatjets_in_pairs.eta,
            fatjets_in_pairs.phi,
            boostedtaus_in_pairs.eta,
            boostedtaus_in_pairs.phi,
        )

        close_matches = dR < 0.8

        matched_taus_per_fatjet = boostedtaus_in_pairs[close_matches]

        n_matched = ak.num(matched_taus_per_fatjet, axis=-1)
        no2tau = n_matched < 2
        no1tau = n_matched < 1

        sorted_indices = ak.argsort(matched_taus_per_fatjet.pt, axis=-1, ascending=False)
        sorted_taus = matched_taus_per_fatjet[sorted_indices]
        top2_taus = ak.pad_none(sorted_taus, 2, axis=-1)[..., :2]

        tau0_eta = ak.fill_none(top2_taus.eta[..., 0], -999)
        tau1_eta = ak.fill_none(top2_taus.eta[..., 1], -999)
        tau0_phi = ak.fill_none(top2_taus.phi[..., 0], -999)
        tau1_phi = ak.fill_none(top2_taus.phi[..., 1], -999)
        tau0_mass = ak.fill_none(top2_taus.mass[..., 0], -999)
        tau1_mass = ak.fill_none(top2_taus.mass[..., 1], -999)
        tau0_pt = ak.fill_none(top2_taus.pt[..., 0], -999)
        tau1_pt = ak.fill_none(top2_taus.pt[..., 1], -999)

        # mass

        ##2subjets
        mass_subjet = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            subjet0_eta,
            subjet1_eta,
            subjet0_phi,
            subjet1_phi,
            subjet0_pt,
            subjet1_pt,
        )
        msoftdrop_subjet = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            subjet0_eta,
            subjet1_eta,
            subjet0_phi,
            subjet1_phi,
            subjet0_pt,
            subjet1_pt,
        )
        globalParT_massVisApplied_subjet = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            subjet0_eta,
            subjet1_eta,
            subjet0_phi,
            subjet1_phi,
            subjet0_pt,
            subjet1_pt,
        )
        globalParT_massResApplied_subjet = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            subjet0_eta,
            subjet1_eta,
            subjet0_phi,
            subjet1_phi,
            subjet0_pt,
            subjet1_pt,
        )
        particleNet_mass_legacy_subjet = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            subjet0_eta,
            subjet1_eta,
            subjet0_phi,
            subjet1_phi,
            subjet0_pt,
            subjet1_pt,
        )
        ##subjet0+muon
        mass_subjet_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            subjet0_eta,
            muon0_eta,
            subjet0_phi,
            muon0_phi,
            subjet0_pt,
            muon0_pt,
        )
        msoftdrop_subjet_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            subjet0_eta,
            muon0_eta,
            subjet0_phi,
            muon0_phi,
            subjet0_pt,
            muon0_pt,
        )
        globalParT_massVisApplied_subjet_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            subjet0_eta,
            muon0_eta,
            subjet0_phi,
            muon0_phi,
            subjet0_pt,
            muon0_pt,
        )
        globalParT_massResApplied_subjet_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            subjet0_eta,
            muon0_eta,
            subjet0_phi,
            muon0_phi,
            subjet0_pt,
            muon0_pt,
        )
        particleNet_mass_legacy_subjet_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            subjet0_eta,
            muon0_eta,
            subjet0_phi,
            muon0_phi,
            subjet0_pt,
            muon0_pt,
        )
        ##subjet0+electron
        mass_subjet_et = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            subjet0_eta,
            electron0_eta,
            subjet0_phi,
            electron0_phi,
            subjet0_pt,
            electron0_pt,
        )
        msoftdrop_subjet_et = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            subjet0_eta,
            electron0_eta,
            subjet0_phi,
            electron0_phi,
            subjet0_pt,
            electron0_pt,
        )
        globalParT_massVisApplied_subjet_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            subjet0_eta,
            electron0_eta,
            subjet0_phi,
            electron0_phi,
            subjet0_pt,
            electron0_pt,
        )
        globalParT_massResApplied_subjet_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            subjet0_eta,
            electron0_eta,
            subjet0_phi,
            electron0_phi,
            subjet0_pt,
            electron0_pt,
        )
        particleNet_mass_legacy_subjet_et = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            subjet0_eta,
            electron0_eta,
            subjet0_phi,
            electron0_phi,
            subjet0_pt,
            electron0_pt,
        )
        ##subjet1+muon
        mass_subjet1_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            subjet1_eta,
            muon0_eta,
            subjet1_phi,
            muon0_phi,
            subjet1_pt,
            muon0_pt,
        )
        msoftdrop_subjet1_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            subjet1_eta,
            muon0_eta,
            subjet1_phi,
            muon0_phi,
            subjet1_pt,
            muon0_pt,
        )
        globalParT_massVisApplied_subjet1_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            subjet1_eta,
            muon0_eta,
            subjet1_phi,
            muon0_phi,
            subjet1_pt,
            muon0_pt,
        )
        globalParT_massResApplied_subjet1_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            subjet1_eta,
            muon0_eta,
            subjet1_phi,
            muon0_phi,
            subjet1_pt,
            muon0_pt,
        )
        particleNet_mass_legacy_subjet1_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            subjet1_eta,
            muon0_eta,
            subjet1_phi,
            muon0_phi,
            subjet1_pt,
            muon0_pt,
        )
        ##subjet1+electron
        mass_subjet1_et = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            subjet1_eta,
            electron0_eta,
            subjet1_phi,
            electron0_phi,
            subjet1_pt,
            electron0_pt,
        )
        msoftdrop_subjet1_et = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            subjet1_eta,
            electron0_eta,
            subjet1_phi,
            electron0_phi,
            subjet1_pt,
            electron0_pt,
        )
        globalParT_massVisApplied_subjet1_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            subjet1_eta,
            electron0_eta,
            subjet1_phi,
            electron0_phi,
            subjet1_pt,
            electron0_pt,
        )
        globalParT_massResApplied_subjet1_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            subjet1_eta,
            electron0_eta,
            subjet1_phi,
            electron0_phi,
            subjet1_pt,
            electron0_pt,
        )
        particleNet_mass_legacy_subjet1_et = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            subjet1_eta,
            electron0_eta,
            subjet1_phi,
            electron0_phi,
            subjet1_pt,
            electron0_pt,
        )

        ##fatjets+MET
        mass_fatjet_tt = calculate_invariant_mass_2d(
            fatjets_mass, fatjets_pt, fatjets_eta, fatjets_phi, fatjets_masscorr, met_pt, met_phi
        )
        msoftdrop_fatjet_tt = calculate_invariant_mass_2d(
            fatjets_msoftdrop,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fatjets_masscorr,
            met_pt,
            met_phi,
        )
        globalParT_massVisApplied_fatjet_tt = calculate_invariant_mass_2d(
            fatjets_globalParT_massVisApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            met_pt,
            met_phi,
        )
        globalParT_massResApplied_fatjet_tt = calculate_invariant_mass_2d(
            fatjets_globalParT_massResApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            met_pt,
            met_phi,
        )
        particleNet_mass_legacy_fatjet_tt = calculate_invariant_mass_2d(
            fatjets_particleNet_mass_legacy,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            met_pt,
            met_phi,
        )

        mass_fatjet_mt = calculate_invariant_mass(
            fatjets_mass,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fatjets_masscorr,
            muon0_pt,
            muon0_eta,
            muon0_phi,
            met_pt,
            met_phi,
        )
        msoftdrop_fatjet_mt = calculate_invariant_mass(
            fatjets_msoftdrop,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fatjets_masscorr,
            muon0_pt,
            muon0_eta,
            muon0_phi,
            met_pt,
            met_phi,
        )
        globalParT_massVisApplied_fatjet_mt = calculate_invariant_mass(
            fatjets_globalParT_massVisApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            muon0_pt,
            muon0_eta,
            muon0_phi,
            met_pt,
            met_phi,
        )
        globalParT_massResApplied_fatjet_mt = calculate_invariant_mass(
            fatjets_globalParT_massResApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            muon0_pt,
            muon0_eta,
            muon0_phi,
            met_pt,
            met_phi,
        )
        particleNet_mass_legacy_fatjet_mt = calculate_invariant_mass(
            fatjets_particleNet_mass_legacy,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            muon0_pt,
            muon0_eta,
            muon0_phi,
            met_pt,
            met_phi,
        )

        mass_fatjet_et = calculate_invariant_mass(
            fatjets_mass,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fatjets_masscorr,
            electron0_pt,
            electron0_eta,
            electron0_phi,
            met_pt,
            met_phi,
        )
        msoftdrop_fatjet_et = calculate_invariant_mass(
            fatjets_msoftdrop,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fatjets_masscorr,
            electron0_pt,
            electron0_eta,
            electron0_phi,
            met_pt,
            met_phi,
        )
        globalParT_massVisApplied_fatjet_et = calculate_invariant_mass(
            fatjets_globalParT_massVisApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            electron0_pt,
            electron0_eta,
            electron0_phi,
            met_pt,
            met_phi,
        )
        globalParT_massResApplied_fatjet_et = calculate_invariant_mass(
            fatjets_globalParT_massResApplied,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            electron0_pt,
            electron0_eta,
            electron0_phi,
            met_pt,
            met_phi,
        )
        particleNet_mass_legacy_fatjet_et = calculate_invariant_mass(
            fatjets_particleNet_mass_legacy,
            fatjets_pt,
            fatjets_eta,
            fatjets_phi,
            fake_corr,
            electron0_pt,
            electron0_eta,
            electron0_phi,
            met_pt,
            met_phi,
        )

        ##2taus
        mass_boostedtau = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            tau0_eta,
            tau1_eta,
            tau0_phi,
            tau1_phi,
            tau0_pt,
            tau1_pt,
        )
        msoftdrop_boostedtau = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            tau0_eta,
            tau1_eta,
            tau0_phi,
            tau1_phi,
            tau0_pt,
            tau1_pt,
        )
        globalParT_massVisApplied_boostedtau = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            tau0_eta,
            tau1_eta,
            tau0_phi,
            tau1_phi,
            tau0_pt,
            tau1_pt,
        )
        globalParT_massResApplied_boostedtau = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            tau0_eta,
            tau1_eta,
            tau0_phi,
            tau1_phi,
            tau0_pt,
            tau1_pt,
        )
        particleNet_mass_legacy_boostedtau = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            tau0_eta,
            tau1_eta,
            tau0_phi,
            tau1_phi,
            tau0_pt,
            tau1_pt,
        )
        ##tau+muon
        mass_boostedtau_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            tau0_eta,
            muon0_eta,
            tau0_phi,
            muon0_phi,
            tau0_pt,
            muon0_pt,
        )
        msoftdrop_boostedtau_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            tau0_eta,
            muon0_eta,
            tau0_phi,
            muon0_phi,
            tau0_pt,
            muon0_pt,
        )
        globalParT_massVisApplied_boostedtau_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            tau0_eta,
            muon0_eta,
            tau0_phi,
            muon0_phi,
            tau0_pt,
            muon0_pt,
        )
        globalParT_massResApplied_boostedtau_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            tau0_eta,
            muon0_eta,
            tau0_phi,
            muon0_phi,
            tau0_pt,
            muon0_pt,
        )
        particleNet_mass_legacy_boostedtau_mt = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            tau0_eta,
            muon0_eta,
            tau0_phi,
            muon0_phi,
            tau0_pt,
            muon0_pt,
        )
        ##tau+electron
        mass_boostedtau_et = CA_got(
            met_pt,
            met_phi,
            fatjets_mass,
            fatjets_masscorr,
            tau0_eta,
            electron0_eta,
            tau0_phi,
            electron0_phi,
            tau0_pt,
            electron0_pt,
        )
        msoftdrop_boostedtau_et = CA_got(
            met_pt,
            met_phi,
            fatjets_msoftdrop,
            fatjets_masscorr,
            tau0_eta,
            electron0_eta,
            tau0_phi,
            electron0_phi,
            tau0_pt,
            electron0_pt,
        )
        globalParT_massVisApplied_boostedtau_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            tau0_eta,
            electron0_eta,
            tau0_phi,
            electron0_phi,
            tau0_pt,
            electron0_pt,
        )
        globalParT_massResApplied_boostedtau_et = CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massResApplied,
            fake_corr,
            tau0_eta,
            electron0_eta,
            tau0_phi,
            electron0_phi,
            tau0_pt,
            electron0_pt,
        )
        particleNet_mass_legacy_boostedtau_et = CA_got(
            met_pt,
            met_phi,
            fatjets_particleNet_mass_legacy,
            fake_corr,
            tau0_eta,
            electron0_eta,
            tau0_phi,
            electron0_phi,
            tau0_pt,
            electron0_pt,
        )

        output_map = {
            # merged: et -> mt -> hh; eachchannel: tau -> subjet -> fatjet
            "CA_mass_merged": [
                (no2subjet & no2tau, mass_fatjet_tt),
                (~no2subjet, mass_subjet),
                (~no2tau, mass_boostedtau),
                (~no1muon, mass_fatjet_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), mass_subjet1_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), mass_subjet_mt),
                ((~no1tau) & (~no1muon), mass_boostedtau_mt),
                (~no1electron, mass_fatjet_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), mass_subjet1_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), mass_subjet_et),
                ((~no1tau) & (~no1electron), mass_boostedtau_et),
            ],
            "CA_msoftdrop_merged": [
                (no2subjet & no2tau, msoftdrop_fatjet_tt),
                (~no2subjet, msoftdrop_subjet),
                (~no2tau, msoftdrop_boostedtau),
                (~no1muon, msoftdrop_fatjet_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), msoftdrop_subjet1_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), msoftdrop_subjet_mt),
                ((~no1tau) & (~no1muon), msoftdrop_boostedtau_mt),
                (~no1electron, msoftdrop_fatjet_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), msoftdrop_subjet1_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), msoftdrop_subjet_et),
                ((~no1tau) & (~no1electron), msoftdrop_boostedtau_et),
            ],
            "CA_globalParT_massVisApplied_merged": [
                (no2subjet & no2tau, globalParT_massVisApplied_fatjet_tt),
                (~no2subjet, globalParT_massVisApplied_subjet),
                (~no2tau, globalParT_massVisApplied_boostedtau),
                (~no1muon, globalParT_massVisApplied_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    globalParT_massVisApplied_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    globalParT_massVisApplied_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), globalParT_massVisApplied_boostedtau_mt),
                (~no1electron, globalParT_massVisApplied_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    globalParT_massVisApplied_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    globalParT_massVisApplied_subjet_et,
                ),
                ((~no1tau) & (~no1electron), globalParT_massVisApplied_boostedtau_et),
            ],
            "CA_globalParT_massResApplied_merged": [
                (no2subjet & no2tau, globalParT_massResApplied_fatjet_tt),
                (~no2subjet, globalParT_massResApplied_subjet),
                (~no2tau, globalParT_massResApplied_boostedtau),
                (~no1muon, globalParT_massResApplied_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    globalParT_massResApplied_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    globalParT_massResApplied_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), globalParT_massResApplied_boostedtau_mt),
                (~no1electron, globalParT_massResApplied_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    globalParT_massResApplied_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    globalParT_massResApplied_subjet_et,
                ),
                ((~no1tau) & (~no1electron), globalParT_massResApplied_boostedtau_et),
            ],
            "CA_particleNet_mass_legacy_merged": [
                (no2subjet & no2tau, particleNet_mass_legacy_fatjet_tt),
                (~no2subjet, particleNet_mass_legacy_subjet),
                (~no2tau, particleNet_mass_legacy_boostedtau),
                (~no1muon, particleNet_mass_legacy_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    particleNet_mass_legacy_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    particleNet_mass_legacy_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), particleNet_mass_legacy_boostedtau_mt),
                (~no1electron, particleNet_mass_legacy_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    particleNet_mass_legacy_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    particleNet_mass_legacy_subjet_et,
                ),
                ((~no1tau) & (~no1electron), particleNet_mass_legacy_boostedtau_et),
            ],
            "CA_Tauflag": [
                (no2subjet & no2tau, 3),
                (~no2subjet, 2),
                (~no2tau, 1),
                (~no1muon, 8),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), 7),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), 6),
                ((~no1tau) & (~no1muon), 5),
                (~no1electron, 13),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), 12),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), 11),
                ((~no1tau) & (~no1electron), 10),
            ],
            # hh(tt)
            # matched 2 HPS boostedtaus: 1; matched 2 subjets: 2; none matching: 0
            "CA_isDauTau": [(~no2subjet, 2), (~no2tau, 1)],
            "CA_mass": [
                (no2subjet & no2tau, mass_fatjet_tt),
                (~no2subjet, mass_subjet),
                (~no2tau, mass_boostedtau),
            ],
            "CA_msoftdrop": [
                (no2subjet & no2tau, msoftdrop_fatjet_tt),
                (~no2subjet, msoftdrop_subjet),
                (~no2tau, msoftdrop_boostedtau),
            ],
            "CA_globalParT_massVisApplied": [
                (no2subjet & no2tau, globalParT_massVisApplied_fatjet_tt),
                (~no2subjet, globalParT_massVisApplied_subjet),
                (~no2tau, globalParT_massVisApplied_boostedtau),
            ],
            "CA_globalParT_massResApplied": [
                (no2subjet & no2tau, globalParT_massResApplied_fatjet_tt),
                (~no2subjet, globalParT_massResApplied_subjet),
                (~no2tau, globalParT_massResApplied_boostedtau),
            ],
            "CA_particleNet_mass_legacy": [
                (no2subjet & no2tau, particleNet_mass_legacy_fatjet_tt),
                (~no2subjet, particleNet_mass_legacy_subjet),
                (~no2tau, particleNet_mass_legacy_boostedtau),
            ],
            "CA_dau0_pt": [(~no2subjet, subjet0_pt), (~no2tau, tau0_pt)],
            "CA_dau1_pt": [(~no2subjet, subjet1_pt), (~no2tau, tau1_pt)],
            "CA_dau0_eta": [(~no2subjet, subjet0_eta), (~no2tau, tau0_eta)],
            "CA_dau1_eta": [(~no2subjet, subjet1_eta), (~no2tau, tau1_eta)],
            "CA_dau0_phi": [(~no2subjet, subjet0_phi), (~no2tau, tau0_phi)],
            "CA_dau1_phi": [(~no2subjet, subjet1_phi), (~no2tau, tau1_phi)],
            "CA_dau0_mass": [(~no2subjet, subjet0_mass), (~no2tau, tau0_mass)],
            "CA_dau1_mass": [(~no2subjet, subjet1_mass), (~no2tau, tau1_mass)],
            "CA_mass_subjets": [(~no2subjet, mass_subjet)],
            "CA_mass_boostedtaus": [(~no2tau, mass_boostedtau)],
            "CA_mass_fatjets": [(no2subjet & no2tau, mass_fatjet_tt)],
            "CA_ntaus_perfatjets": [(~no2tau, n_matched)],
            "CA_nsubjets_perfatjets": [(~no2subjet, n_matched_subjets)],
            # mt (mu+X)
            "CA_isDauTau_mt": [
                (~no1muon, 8),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), 7),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), 6),
                ((~no1tau) & (~no1muon), 5),
            ],
            "CA_mass_mt": [
                (~no1muon, mass_fatjet_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), mass_subjet1_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), mass_subjet_mt),
                ((~no1tau) & (~no1muon), mass_boostedtau_mt),
            ],
            "CA_msoftdrop_mt": [
                (~no1muon, msoftdrop_fatjet_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), msoftdrop_subjet1_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), msoftdrop_subjet_mt),
                ((~no1tau) & (~no1muon), msoftdrop_boostedtau_mt),
            ],
            "CA_globalParT_massVisApplied_mt": [
                (~no1muon, mass_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    globalParT_massVisApplied_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    globalParT_massVisApplied_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), globalParT_massVisApplied_boostedtau_mt),
            ],
            "CA_globalParT_massResApplied_mt": [
                (~no1muon, mass_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    globalParT_massResApplied_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    globalParT_massResApplied_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), globalParT_massResApplied_boostedtau_mt),
            ],
            "CA_particleNet_mass_legacy_mt": [
                (~no1muon, mass_fatjet_mt),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1),
                    particleNet_mass_legacy_subjet1_mt,
                ),
                (
                    (~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0),
                    particleNet_mass_legacy_subjet_mt,
                ),
                ((~no1tau) & (~no1muon), particleNet_mass_legacy_boostedtau_mt),
            ],
            "CA_one_muon_in_fatjet": [
                (~no1muon, 1),
            ],
            "CA_one_muon": [
                (~no1muon_ori, 1),
            ],
            "CA_dau0_pt_mt": [
                ((~no1subjet) & (~no1muon), subjet0_pt),
                ((~no1tau) & (~no1muon), tau0_pt),
            ],
            "CA_dau1_pt_mt": [
                ((~no1subjet) & (~no1muon), muon0_pt),
                ((~no1tau) & (~no1muon), muon0_pt),
            ],
            "CA_dau0_eta_mt": [
                ((~no1subjet) & (~no1muon), subjet0_eta),
                ((~no1tau) & (~no1muon), tau0_eta),
            ],
            "CA_dau1_eta_mt": [
                ((~no1subjet) & (~no1muon), muon0_eta),
                ((~no1tau) & (~no1muon), muon0_eta),
            ],
            "CA_dau0_phi_mt": [
                ((~no1subjet) & (~no1muon), subjet0_phi),
                ((~no1tau) & (~no1muon), tau0_phi),
            ],
            "CA_dau1_phi_mt": [
                ((~no1subjet) & (~no1muon), muon0_phi),
                ((~no1tau) & (~no1muon), muon0_phi),
            ],
            "CA_dau0_mass_mt": [
                ((~no1subjet) & (~no1muon), subjet0_mass),
                ((~no1tau) & (~no1muon), tau0_mass),
            ],
            "CA_dau1_mass_mt": [
                ((~no1subjet) & (~no1muon), muon0_mass),
                ((~no1tau) & (~no1muon), muon0_mass),
            ],
            "CA_ntaus_perfatjets_mt": [((~no1tau) & (~no1muon), n_matched)],
            "CA_nsubjets_perfatjets_mt": [((~no1subjet) & (~no1muon), n_matched_subjets)],
            "CA_mass_boostedtaus_mt": [((~no1tau) & (~no1muon), mass_boostedtau_mt)],
            "CA_mass_fatjet_mt": [(~no1muon, mass_fatjet_mt)],
            "CA_mass_subjets_mt_01": [
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), mass_subjet1_mt),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), mass_subjet_mt),
            ],
            "CA_muon_subjet_dr02": [
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet1), 2),
                ((~no1subjet) & (~no1muon) & (~dR_muon_vs_subjet0), 1),
            ],
            "CA_mass_subjets_mt_1": [((~no1subjet) & (~no1muon), mass_subjet1_mt)],
            "CA_mass_subjets_mt_0": [((~no1subjet) & (~no1muon), mass_subjet_mt)],
            # et (e+X) : subjet -> boosted
            "CA_isDauTau_et": [
                (~no1electron, 13),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), 12),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), 11),
                ((~no1tau) & (~no1electron), 10),
            ],
            "CA_mass_et": [
                (~no1electron, mass_fatjet_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), mass_subjet1_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), mass_subjet_et),
                ((~no1tau) & (~no1electron), mass_boostedtau_et),
            ],
            "CA_msoftdrop_et": [
                (~no1electron, msoftdrop_fatjet_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), msoftdrop_subjet1_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), msoftdrop_subjet_et),
                ((~no1tau) & (~no1electron), msoftdrop_boostedtau_et),
            ],
            "CA_globalParT_massVisApplied_et": [
                (~no1electron, mass_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    globalParT_massVisApplied_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    globalParT_massVisApplied_subjet_et,
                ),
                ((~no1tau) & (~no1electron), globalParT_massVisApplied_boostedtau_et),
            ],
            "CA_globalParT_massResApplied_et": [
                (~no1electron, mass_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    globalParT_massResApplied_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    globalParT_massResApplied_subjet_et,
                ),
                ((~no1tau) & (~no1electron), globalParT_massResApplied_boostedtau_et),
            ],
            "CA_particleNet_mass_legacy_et": [
                (~no1electron, mass_fatjet_et),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1),
                    particleNet_mass_legacy_subjet1_et,
                ),
                (
                    (~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0),
                    particleNet_mass_legacy_subjet_et,
                ),
                ((~no1tau) & (~no1electron), particleNet_mass_legacy_boostedtau_et),
            ],
            "CA_one_elec_in_fatjet": [
                (~no1electron, 1),
            ],
            "CA_one_elec": [
                (~no1electron_ori, 1),
            ],
            "CA_dau0_pt_et": [
                ((~no1subjet) & (~no1electron), subjet0_pt),
                ((~no1tau) & (~no1electron), tau0_pt),
            ],
            "CA_dau1_pt_et": [
                ((~no1subjet) & (~no1electron), electron0_pt),
                ((~no1tau) & (~no1electron), electron0_pt),
            ],
            "CA_dau0_eta_et": [
                ((~no1subjet) & (~no1electron), subjet0_eta),
                ((~no1tau) & (~no1electron), tau0_eta),
            ],
            "CA_dau1_eta_et": [
                ((~no1subjet) & (~no1electron), electron0_eta),
                ((~no1tau) & (~no1electron), electron0_eta),
            ],
            "CA_dau0_phi_et": [
                ((~no1subjet) & (~no1electron), subjet0_phi),
                ((~no1tau) & (~no1electron), tau0_phi),
            ],
            "CA_dau1_phi_et": [
                ((~no1subjet) & (~no1electron), electron0_phi),
                ((~no1tau) & (~no1electron), electron0_phi),
            ],
            "CA_dau0_mass_et": [
                ((~no1subjet) & (~no1electron), subjet0_mass),
                ((~no1tau) & (~no1electron), tau0_mass),
            ],
            "CA_dau1_mass_et": [
                ((~no1subjet) & (~no1electron), electron0_mass),
                ((~no1tau) & (~no1electron), electron0_mass),
            ],
            "CA_ntaus_perfatjets_et": [((~no1tau) & (~no1electron), n_matched)],
            "CA_nsubjets_perfatjets_et": [((~no1subjet) & (~no1electron), n_matched_subjets)],
            "CA_mass_subjets_et": [((~no1subjet) & (~no1electron), mass_subjet_et)],
            "CA_mass_boostedtaus_et": [((~no1tau) & (~no1electron), mass_boostedtau_et)],
            "CA_mass_fatjet_et": [(~no1electron, mass_fatjet_et)],
            "CA_mass_subjets_et_01": [
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), mass_subjet1_et),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), mass_subjet_et),
            ],
            "CA_elec_subjet_dr02": [
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet1), 2),
                ((~no1subjet) & (~no1electron) & (~dR_elec_vs_subjet0), 1),
            ],
            "CA_mass_subjets_et_1": [((~no1subjet) & (~no1electron), mass_subjet1_et)],
            "CA_mass_subjets_et_0": [((~no1subjet) & (~no1electron), mass_subjet_et)],
        }

        for field, val_pairs in output_map.items():
            for condition, value in val_pairs:
                fatjets[field] = ak.where(condition, value, fatjets[field])

    return fatjets
