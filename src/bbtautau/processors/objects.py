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
    MissingET,
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


#adopted from https://github.com/scikit-hep/coffea/blob/a315da1fa307f1ec0d21c29e908e5b733603d7c0/src/coffea/nanoevents/methods/vector.py#L106
def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(deta,dphi)


def CA_got(met_pt, met_phi, fatjets_mass, fatjets_masscorr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt):
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

    denom = np.sqrt(np.abs(tau0_pt/(tau0_pt + pmet_tau0)) * np.abs(tau1_pt/(tau1_pt + pmet_tau1)))
    denom = ak.where(denom == 0, 1, denom)

    mass = fatjets_mass * fatjets_masscorr / denom
    mass = ak.where(invalid, -999, mass)
    return mass


def get_CA_MASS(fatjets: FatJetArray, taus: TauArray, met: MissingET, subjets: JetArray):

    init_fields = {
        "CA_mass_boostedtaus": (-999.0, float),
        "CA_ntaus_perfatjets": (-1, int),
        "CA_mass_subjets": (-999.0, float),
        "CA_nsubjets_perfatjets": (-1, int),

        "CA_mass": (-999.0, float),
        "CA_msoftdrop": (-999.0, float),
        "CA_globalParT_massVisApplied": (-999.0, float),
        "CA_globalParT_massResApplied": (-999.0, float),
        "CA_particleNet_mass_legacy": (-999.0, float),


        "CA_isDauTau": (0, int),

        "CA_dau0_pt": (-999.0, float), "CA_dau1_pt": (-999.0, float),
        "CA_dau0_eta": (-999.0, float), "CA_dau1_eta": (-999.0, float),
        "CA_dau0_phi": (-999.0, float), "CA_dau1_phi": (-999.0, float),
        "CA_dau0_mass": (-999.0, float), "CA_dau1_mass": (-999.0, float),
    }

        
    n_events = len(fatjets)
    n_fatjets = ak.num(fatjets, axis=1)
    n_taus = ak.num(taus, axis=1)

    n_subjets = len(subjets)
    
    
    has_fatjets = n_fatjets > 0
    has_taus = n_taus > 0
    can_match = has_fatjets & has_taus
    

    for name, (default, dtype) in init_fields.items():
        fatjets[name] = ak.full_like(fatjets.pt, default, dtype=dtype)

    no2tau = ak.full_like(fatjets.pt, False, dtype=bool) 
    no2subjet = ak.full_like(fatjets.pt, False, dtype=bool) 

    met_pt = met.pt
    met_phi = met.phi
    
    
    if ak.any(can_match):

        fatjets_mass = fatjets.mass
        fatjets_msoftdrop = fatjets.msoftdrop

        fatjets_masscorr = fatjets.particleNet_massCorr

        fatjets_globalParT_massVis = fatjets.globalParT_massVis
        fatjets_globalParT_massRes = fatjets.globalParT_massRes
        fatjets_globalParT_massResApplied = fatjets.globalParT_massResApplied
        fatjets_globalParT_massVisApplied = fatjets.globalParT_massVisApplied
        fatjets_particleNet_mass_legacy = fatjets.particleNetLegacy_mass

        fake_corr = ak.full_like(fatjets_masscorr, 1.0, dtype=float)

        ###to change to subjet
        fatjet_subjet_pairs = ak.cartesian([fatjets, subjets], nested=True)
        fatjets_in_pairs = fatjet_subjet_pairs["0"]
        subjets_in_pairs = fatjet_subjet_pairs["1"]

        dR_subjets = delta_r(fatjets_in_pairs.eta, fatjets_in_pairs.phi, subjets_in_pairs.eta, subjets_in_pairs.phi)

        close_matches_subjets = dR_subjets < 0.8

        matched_subjets_per_fatjet = subjets_in_pairs[close_matches_subjets]

        n_matched_subjets = ak.num(matched_subjets_per_fatjet, axis=-1)
        no2subjet = n_matched_subjets < 2

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


        mass_subjet = CA_got(met_pt, met_phi, fatjets_mass, fatjets_masscorr, subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt)
        msoftdrop_subjet = CA_got(met_pt, met_phi, fatjets_msoftdrop, fatjets_masscorr, subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt)

        # ###
        globalParT_massVisApplied_subjet = CA_got(met_pt, met_phi, fatjets_globalParT_massVisApplied, fake_corr, subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt)
        globalParT_massResApplied_subjet = CA_got(met_pt, met_phi, fatjets_globalParT_massResApplied, fake_corr, subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt)
        particleNet_mass_legacy_subjet = CA_got(met_pt, met_phi, fatjets_particleNet_mass_legacy, fake_corr, subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt)


        fatjet_boostedtau_pairs = ak.cartesian([fatjets, taus], nested=True)
        fatjets_in_pairs = fatjet_boostedtau_pairs["0"]
        boostedtaus_in_pairs = fatjet_boostedtau_pairs["1"]

        dR = delta_r(fatjets_in_pairs.eta, fatjets_in_pairs.phi, boostedtaus_in_pairs.eta, boostedtaus_in_pairs.phi)

        close_matches = dR < 0.8

        matched_taus_per_fatjet = boostedtaus_in_pairs[close_matches]

        n_matched = ak.num(matched_taus_per_fatjet, axis=-1)
        no2tau = n_matched < 2

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


        mass_boostedtau = CA_got(met_pt, met_phi, fatjets_mass, fatjets_masscorr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)
        msoftdrop_boostedtau = CA_got(met_pt, met_phi, fatjets_msoftdrop, fatjets_masscorr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)

        globalParT_massVisApplied_boostedtau = CA_got(met_pt, met_phi, fatjets_globalParT_massVisApplied, fake_corr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)
        globalParT_massResApplied_boostedtau = CA_got(met_pt, met_phi, fatjets_globalParT_massResApplied, fake_corr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)
        particleNet_mass_legacy_boostedtau = CA_got(met_pt, met_phi, fatjets_particleNet_mass_legacy, fake_corr, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)

        output_map = {
            "CA_mass": [(~no2subjet, mass_subjet), (~no2tau, mass_boostedtau)],
            "CA_msoftdrop": [(~no2subjet, msoftdrop_subjet), (~no2tau, msoftdrop_boostedtau)],

            "CA_globalParT_massVisApplied": [(~no2subjet, globalParT_massVisApplied_subjet), (~no2tau, globalParT_massVisApplied_boostedtau)],
            "CA_globalParT_massResApplied": [(~no2subjet, globalParT_massResApplied_subjet), (~no2tau, globalParT_massResApplied_boostedtau)],
            "CA_particleNet_mass_legacy": [(~no2subjet, particleNet_mass_legacy_subjet), (~no2tau, particleNet_mass_legacy_boostedtau)],

            #matched 2 HPS boostedtaus: 1; matched 2 subjets: 2; none matching: 0
            "CA_isDauTau": [(~no2subjet, 2), (~no2tau, 1)],

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
            "CA_ntaus_perfatjets": [(~no2tau, n_matched)],
            "CA_nsubjets_perfatjets": [(~no2subjet, n_matched_subjets)],
        }

        for field, val_pairs in output_map.items():
            for condition, value in val_pairs:
                fatjets[field] = ak.where(condition, value, fatjets[field])

    return fatjets
