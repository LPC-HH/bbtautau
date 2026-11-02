"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

from functools import wraps

import awkward as ak
import numpy as np

# added v15
from boostedhh.processors.objects import jetid_v12, jetid_v15
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

# Trigger-catalog-type -> HLTs.MENU key, for the (type, filterbit, ptcut) combos that have an analysis-menu equivalent.
_MENU_KEY_BY_TRIGGER_TYPE = {
    "EGamma": "ele30",
    "Muon": "muiso24",
    "DiTau": "ditau",
}


def _is_systematic_collection(obj):
    """Return True for a {"": nominal, "Shift_up/down": varied} collection."""
    return isinstance(obj, dict) and "" in obj


def propagate_systematics(func):
    """
    Propagate object-level kinematic variations through a derived-variable function.

    Systematic-aware inputs use the form
        {"": nominal, "SomeShift_up": varied_up, "SomeShift_down": varied_down}.

    Each nuisance is evaluated independently: inputs carrying the same shift move
    together, while all other inputs fall back to nominal. No Cartesian products
    of independent nuisances are produced.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        syst_inputs = [obj for obj in (*args, *kwargs.values()) if _is_systematic_collection(obj)]

        # Preserve the old call/return behaviour when ordinary collections are used.
        if not syst_inputs:
            return func(*args, **kwargs)

        shifts = {shift for obj in syst_inputs for shift in obj if shift != ""}

        outputs = {}
        for shift in [""] + sorted(shifts):
            shifted_args = [
                obj.get(shift, obj[""]) if _is_systematic_collection(obj) else obj for obj in args
            ]
            shifted_kwargs = {
                key: (obj.get(shift, obj[""]) if _is_systematic_collection(obj) else obj)
                for key, obj in kwargs.items()
            }
            outputs[shift] = func(*shifted_args, **shifted_kwargs)

        return outputs

    return wrapper


def trig_match_sel(
    events, leptons, trig_leptons, year, trigger, filterbit, ptcut, trig_dR=0.2, menu_key=None
):
    """
    Returns selection for leptons which are trigger matched to the specified trigger.

    If `menu_key` is given (a key in `HLTs.MENU`), the physical HLT path is resolved via
    `HLTs.resolve_menu(year, menu_key, ...)` instead of the catalog-type lookup. Otherwise,
    falls back to the legacy `HLTs.hlts_by_type(year, trigger, ...)[0]` resolution (whatever
    happens to be first in that trigger-type's catalog list).
    """
    if menu_key is not None:
        trigger = HLTs.resolve_menu(year, menu_key, hlt_prefix=False)[0]
    else:
        trigger = HLTs.hlts_by_type(year, trigger, hlt_prefix=False)[
            0
        ]  # picking first trigger in list
    trig_fired = events.HLT[trigger]
    # print(f"{trigger} rate: {ak.mean(trig_fired)}")

    filterbit = 2**filterbit

    pass_trig = (trig_leptons.filterBits & filterbit) == filterbit
    trig_l = trig_leptons[pass_trig]
    trig_l_matched = ak.any(leptons.metric_table(trig_l) < trig_dR, axis=2)
    trig_l_sel = trig_fired & trig_l_matched & (leptons.pt > ptcut)
    return trig_l_sel


# added v15
def get_ak8jets(fatjets: FatJetArray, nano_version: str):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)

    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass

    fatjets["particleNetLegacy_XbbvsQCD"] = fatjets.particleNetLegacy_Xbb / (
        fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
    )

    if nano_version.startswith("v12"):
        fatjets["globalParT_QCD"] = (
            fatjets.globalParT_QCD0HF + fatjets.globalParT_QCD1HF + fatjets.globalParT_QCD2HF
        )
        fatjets["globalParT_Top"] = fatjets.globalParT_TopW + fatjets.globalParT_TopbW
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
                fatjets[f"globalParT_X{tautau}"]
                + fatjets["globalParT_QCD"]
                + fatjets["globalParT_Top"]
            )
    elif nano_version.startswith("v15"):
        fatjets["globalParT_QCD"] = fatjets.globalParT3_QCD
        fatjets["globalParT_Top"] = (
            fatjets.globalParT3_TopbWev
            + fatjets.globalParT3_TopbWmv
            + fatjets.globalParT3_TopbWq
            + fatjets.globalParT3_TopbWqq
            + fatjets.globalParT3_TopbWtauhv
        )
        fatjets["globalParT_TopbWev"] = fatjets.globalParT3_TopbWev
        fatjets["globalParT_TopbWmv"] = fatjets.globalParT3_TopbWmv
        fatjets["globalParT_TopbWq"] = fatjets.globalParT3_TopbWq
        fatjets["globalParT_TopbWqq"] = fatjets.globalParT3_TopbWqq
        fatjets["globalParT_TopbWtauhv"] = fatjets.globalParT3_TopbWtauhv
        fatjets["globalParT_XbbvsQCD"] = fatjets.globalParT3_Xbb / (
            fatjets.globalParT3_Xbb + fatjets["globalParT_QCD"]
        )
        fatjets["globalParT_XbbvsQCDTop"] = fatjets.globalParT3_Xbb / (
            fatjets.globalParT3_Xbb + fatjets["globalParT_QCD"] + fatjets["globalParT_Top"]
        )
        fatjets["globalParT_Xbb"] = fatjets.globalParT3_Xbb
        fatjets["globalParT_Xcc"] = fatjets.globalParT3_Xcc
        fatjets["globalParT_Xcs"] = fatjets.globalParT3_Xcs
        fatjets["globalParT_Xqq"] = fatjets.globalParT3_Xqq
        for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
            fatjets[f"globalParT_X{tautau}vsQCD"] = fatjets[f"globalParT3_X{tautau}"] / (
                fatjets[f"globalParT3_X{tautau}"] + fatjets["globalParT_QCD"]
            )
            fatjets[f"globalParT_X{tautau}vsQCDTop"] = fatjets[f"globalParT3_X{tautau}"] / (
                fatjets[f"globalParT3_X{tautau}"]
                + fatjets["globalParT_QCD"]
                + fatjets["globalParT_Top"]
            )
            fatjets[f"globalParT_X{tautau}"] = fatjets[f"globalParT3_X{tautau}"]

    if nano_version.startswith("v12"):
        fatjets["globalParT_massResCorr"] = fatjets.globalParT_massRes
        fatjets["globalParT_massVisCorr"] = fatjets.globalParT_massVis
        fatjets["globalParT_massResApplied"] = (
            fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
        )
        fatjets["globalParT_massVisApplied"] = (
            fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass
        )
    elif nano_version.startswith("v15"):
        fatjets["globalParT_massResCorr"] = fatjets.globalParT3_massCorrX2p
        fatjets["globalParT_massVisCorr"] = fatjets.globalParT3_massCorrGeneric
        # The mass uses variables from the official NanoAOD.
        # Only to ensure that the variable names are consistent with v12.
        fatjets["globalParT_massResApplied"] = (
            fatjets.globalParT3_massCorrX2p * (1 - fatjets.rawFactor) * fatjets.mass
        )
        fatjets["globalParT_massVisApplied"] = (
            fatjets.globalParT3_massCorrGeneric * (1 - fatjets.rawFactor) * fatjets.mass
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
    nano_version: str,
    mreg_str: str = "particleNet_mass_legacy",  # noqa: ARG001
):
    # if nano_version.startswith("v12"):
    #     jetidtight, jetidtightlepveto = jetid_v12(fatjets)  # v12 jetid fix
    # else:
    #     raise NotImplementedError(f"Jet ID fix not implemented yet for {nano_version}")

    # Data does not have .neHEF etc. fields for fatjets, so above recipe doesn't work
    # Either way, doesn't matter since we only use tightID, and it is correct for eta < 2.7

    if nano_version.startswith("v12"):
        jetidtight = fatjets.isTight
    # added v15
    elif nano_version.startswith("v15"):
        jetidtight, jetidtightlepveto = jetid_v15(fatjets)  # v12 jetid fix

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
    # added v15
    elif nano_version.startswith("v15"):
        jetidtight, jetidtightlepveto = jetid_v15(jets)  # v12 jetid fix
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

    # Trigger: (filterbit, ptcut for matched lepton); bit 1 = 1e (WPTight) in both v12 and v15
    triggers = {"EGamma": (1, 31)}
    trig_leptons = trigobj[trigobj.id == PDGID.e]

    TrigMatchDict = {
        f"ElectronTrigMatch{trigger}": trig_match_sel(
            events,
            leptons,
            trig_leptons,
            year,
            trigger,
            filterbit,
            ptcut,
            menu_key=_MENU_KEY_BY_TRIGGER_TYPE.get(trigger),
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

    # Trigger: (filterbit, ptcut for matched lepton); bit 3 = 1mu in both v12 and v15
    triggers = {"Muon": (3, 26)}
    trig_leptons = trigobj[trigobj.id == PDGID.mu]

    TrigMatchDict = {
        f"MuonTrigMatch{trigger}": trig_match_sel(
            events,
            leptons,
            trig_leptons,
            year,
            trigger,
            filterbit,
            ptcut,
            menu_key=_MENU_KEY_BY_TRIGGER_TYPE.get(trigger),
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

# Tau TrigObj filter bit of the analysis di-tau path (HLTs.MENU["ditau"]), from the
# TrigObj_filterBits branch doc: v12 "7 => deeptau di-tau"; v15 "11 => di-tau"
# (v15 7-10 = e-tau inside filter, mu-tau inside filter, Single Tau, VBF DiTau).
_TAU_DITAU_FILTERBIT = {"v12": 7, "v15": 11}


def good_taus(events, leptons: TauArray, year: str, nano_version: str):
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

    if nano_version.startswith("v12"):
        ditau_bit = _TAU_DITAU_FILTERBIT["v12"]
    elif nano_version.startswith("v15"):
        ditau_bit = _TAU_DITAU_FILTERBIT["v15"]
    else:
        raise NotImplementedError(f"Tau trigger filter bits not implemented for {nano_version}")

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"DiTau": (ditau_bit, 37)}
    trig_leptons = trigobj[trigobj.id == PDGID.tau]

    TrigMatchDict = {
        f"TauTrigMatch{trigger}": trig_match_sel(
            events,
            leptons,
            trig_leptons,
            year,
            trigger,
            filterbit,
            ptcut,
            menu_key=_MENU_KEY_BY_TRIGGER_TYPE.get(trigger),
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
    electrons: ElectronArray | None = None,
    muons: MuonArray | None = None,
):
    """Top 2 jets in pT passing the VBF selections.

    If corrected electron/muon collections are provided by the skimmer, use them
    for lepton overlap cleaning. Otherwise fall back to the raw NanoAOD
    collections to preserve the old call pattern.
    """
    electrons = events.Electron if electrons is None else electrons
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon if muons is None else muons
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
    electrons: ElectronArray | None = None,
    muons: MuonArray | None = None,
    sort_by: str = "btag",
):
    """AK4 jets nonoverlapping with AK8 fatjets.

    If corrected electron/muon collections are provided by the skimmer, use them
    for lepton overlap cleaning. Otherwise fall back to the raw NanoAOD
    collections to preserve the old call pattern.
    """
    electrons = events.Electron if electrons is None else electrons
    electrons = electrons[electrons.pt > electron_pt]

    muons = events.Muon if muons is None else muons
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


def project_met_to_fatjet_p4(
    fatjet_mass, fatjet_pt, fatjet_eta, fatjet_phi, fatjet_masscorr, met_pt, met_phi
):
    invalid = (
        (fatjet_pt == -999)
        | (fatjet_eta == -999)
        | (fatjet_phi == -999)
        | (fatjet_mass == -999)
        | (met_pt == -999)
        | (met_phi == -999)
    )

    px_fj = fatjet_pt * np.cos(fatjet_phi)
    py_fj = fatjet_pt * np.sin(fatjet_phi)
    pz_fj = fatjet_pt * np.sinh(fatjet_eta)

    px_met = met_pt * np.cos(met_phi)
    py_met = met_pt * np.sin(met_phi)

    px_met = px_met[..., None]
    py_met = py_met[..., None]

    dot = px_met * px_fj + py_met * py_fj
    pt2 = px_fj**2 + py_fj**2
    pt2 = np.where(pt2 == 0, 1e-6, pt2)

    proj_x = dot / pt2 * px_fj
    proj_y = dot / pt2 * py_fj

    pt_old = np.sqrt(px_fj**2 + py_fj**2)
    pt_new = np.sqrt((px_fj + proj_x) ** 2 + (py_fj + proj_y) ** 2)
    k = np.where(pt_old == 0, 1, pt_new / pt_old)

    px_new = px_fj * k
    py_new = py_fj * k
    pz_new = pz_fj * k

    phi_new = np.arctan2(py_new, px_new)

    pt_safe = np.where(pt_new == 0, 1e-6, pt_new)
    eta_new = np.arcsinh(pz_new / pt_safe)

    p2_vis = px_fj**2 + py_fj**2 + pz_fj**2
    fatjet_mass = fatjet_mass * fatjet_masscorr
    E_vis = np.sqrt(p2_vis + fatjet_mass**2)

    E_new = E_vis * k

    M2_new = E_new**2 - (px_new**2 + py_new**2 + pz_new**2)
    M2_new = np.where(M2_new < 0, 0, M2_new)  # 数值保护
    M_new = np.sqrt(M2_new)

    pt_new = ak.where(invalid, -999, pt_new)
    eta_new = ak.where(invalid, -999, eta_new)
    phi_new = ak.where(invalid, -999, phi_new)
    M_new = ak.where(invalid, -999, M_new)

    return pt_new, eta_new, phi_new, M_new


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


# jin for -,day 202601
def vector_subtraction(a_pt, a_eta, a_phi, a_mass, b_pt, b_eta, b_phi, b_mass):

    invalid = (
        (a_pt == -999)
        | (a_eta == -999)
        | (a_phi == -999)
        | (a_mass == -999)
        | (b_pt == -999)
        | (b_eta == -999)
        | (b_phi == -999)
        | (b_mass == -999)
    )

    a_px = a_pt * np.cos(a_phi)
    a_py = a_pt * np.sin(a_phi)
    a_pz = a_pt * np.sinh(a_eta)
    a_energy = np.sqrt(a_mass**2 + a_px**2 + a_py**2 + a_pz**2)

    b_px = b_pt * np.cos(b_phi)
    b_py = b_pt * np.sin(b_phi)
    b_pz = b_pt * np.sinh(b_eta)
    b_energy = np.sqrt(b_mass**2 + b_px**2 + b_py**2 + b_pz**2)

    px_diff = a_px - b_px
    py_diff = a_py - b_py
    pz_diff = a_pz - b_pz
    energy_diff = a_energy - b_energy

    result_pt = np.sqrt(px_diff**2 + py_diff**2)
    result_eta = 0.5 * np.log((energy_diff + pz_diff) / (energy_diff - pz_diff))
    result_phi = np.arctan2(py_diff, px_diff)
    result_mass = np.sqrt(energy_diff**2 - px_diff**2 - py_diff**2 - pz_diff**2)

    result_pt = ak.where(invalid, -999, result_pt)
    result_eta = ak.where(invalid, -999, result_eta)
    result_phi = ak.where(invalid, -999, result_phi)
    result_mass = ak.where(invalid, -999, result_mass)

    return result_pt, result_eta, result_phi, result_mass


def _match_to_fatjets(fatjets, objects, dr: float = 0.8):
    """Per fatjet: number of ``objects`` within ``dr``, total number of ``objects`` in the event,
    and the two leading matched objects (padded with None)."""
    pairs = ak.cartesian([fatjets, objects], nested=True)
    fatjets_in_pairs, objects_in_pairs = pairs["0"], pairs["1"]
    matched = objects_in_pairs[
        delta_r(
            fatjets_in_pairs.eta, fatjets_in_pairs.phi, objects_in_pairs.eta, objects_in_pairs.phi
        )
        < dr
    ]
    sorted_matched = matched[ak.argsort(matched.pt, axis=-1, ascending=False)]
    top2 = ak.pad_none(sorted_matched, 2, axis=-1)[..., :2]
    return ak.num(matched, axis=-1), ak.num(objects_in_pairs, axis=-1), top2


def _leading(top2, idx: int):
    """pt, eta, phi, mass of the ``idx``-th matched object, -999 if missing."""
    return tuple(ak.fill_none(top2[var][..., idx], -999) for var in ("pt", "eta", "phi", "mass"))


@propagate_systematics
def get_CA_MASS(
    fatjets: FatJetArray,
    taus: TauArray,
    met: MissingET,
    subjets: JetArray,
    muons: MuonArray,
    electrons: ElectronArray,
):
    """Collinear-approximation (CA) mass and tau / lepton-in-fatjet flags for each fatjet.

    Only the CA_* fields stored in the skim are computed (``ca_vars`` in bbtautauSkimmer):
    the current CA mass (``..._with_delta_axis_merged``) with its channel flag, the old CA mass
    (input of the current BDTs), and the numbers of taus / electrons / muons in the fatjet.
    """

    # Create a new Awkward Array wrapper without changing any values.
    # get_CA_MASS adds CA_* fields below, so each systematic evaluation
    # needs its own FatJet object to avoid sharing/overwriting CA fields.
    fatjets = ak.with_field(fatjets, fatjets.pt, "pt")

    init_fields = {
        "CA_tau_number_in_fatjet": (0, int),
        "CA_globalParT_massVisApplied_with_delta_axis_merged": (-999.0, float),
        "CA_globalParT_massVisApplied_oneHPSTauorLepton_flag": (0, int),
        "CA_one_elec_in_fatjet": (0, int),
        "CA_one_muon_in_fatjet": (0, int),
        "CA_globalParT_massVisApplied": (-999.0, float),
    }
    for name, (default, dtype) in init_fields.items():
        fatjets[name] = ak.full_like(fatjets.pt, default, dtype=dtype)

    # Skip only if there are no fatjets at all. Fatjets without matched taus still get the
    # lepton / no-lepton CA masses, independently of whether other events in the chunk have taus.
    if not ak.any(ak.num(fatjets, axis=1) > 0):
        return fatjets

    met_pt = met.pt
    met_phi = met.phi

    fatjets_pt = fatjets.pt
    fatjets_eta = fatjets.eta
    fatjets_phi = fatjets.phi
    fatjets_mass = fatjets.mass
    fatjets_globalParT_massVisApplied = fatjets.globalParT_massVisApplied
    fake_corr = ak.full_like(fatjets.particleNet_massCorr, 1.0, dtype=float)

    # objects matched to each fatjet (dR < 0.8)
    n_matched_muons, _, top2_muons = _match_to_fatjets(fatjets, muons)
    muon0_pt, muon0_eta, muon0_phi, muon0_mass = _leading(top2_muons, 0)
    no1muon = n_matched_muons < 1

    n_matched_electrons, _, top2_electrons = _match_to_fatjets(fatjets, electrons)
    electron0_pt, electron0_eta, electron0_phi, electron0_mass = _leading(top2_electrons, 0)
    no1electron = n_matched_electrons < 1

    n_matched_subjets, _, top2_subjets = _match_to_fatjets(fatjets, subjets)
    subjet0_pt, subjet0_eta, subjet0_phi, _ = _leading(top2_subjets, 0)
    subjet1_pt, subjet1_eta, subjet1_phi, _ = _leading(top2_subjets, 1)
    no2subjet = n_matched_subjets < 2

    n_matched, _, top2_taus = _match_to_fatjets(fatjets, taus)
    tau0_pt, tau0_eta, tau0_phi, tau0_mass = _leading(top2_taus, 0)
    tau1_pt, tau1_eta, tau1_phi, _ = _leading(top2_taus, 1)
    no2tau = n_matched < 2
    no1tau = n_matched < 1

    def ca_mass(d0_eta, d1_eta, d0_phi, d1_phi, d0_pt, d1_pt):
        """CA mass of the fatjet with ParT visible mass, for the two given daughters."""
        return CA_got(
            met_pt,
            met_phi,
            fatjets_globalParT_massVisApplied,
            fake_corr,
            d0_eta,
            d1_eta,
            d0_phi,
            d1_phi,
            d0_pt,
            d1_pt,
        )

    # two taus / tau + lepton
    massVis_boostedtau = ca_mass(tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)
    massVis_boostedtau_mt = ca_mass(tau0_eta, muon0_eta, tau0_phi, muon0_phi, tau0_pt, muon0_pt)
    massVis_boostedtau_et = ca_mass(
        tau0_eta, electron0_eta, tau0_phi, electron0_phi, tau0_pt, electron0_pt
    )

    # one tau or lepton + (FatJet - tau/lepton) as the other daughter
    other_pt, other_eta, other_phi, _ = vector_subtraction(
        fatjets_pt, fatjets_eta, fatjets_phi, fatjets_mass, tau0_pt, tau0_eta, tau0_phi, tau0_mass
    )
    massVis_boostedtau_tat = ca_mass(tau0_eta, other_eta, tau0_phi, other_phi, tau0_pt, other_pt)

    other_pt, other_eta, other_phi, _ = vector_subtraction(
        fatjets_pt,
        fatjets_eta,
        fatjets_phi,
        fatjets_mass,
        electron0_pt,
        electron0_eta,
        electron0_phi,
        electron0_mass,
    )
    massVis_boostedtau_et_at = ca_mass(
        other_eta, electron0_eta, other_phi, electron0_phi, other_pt, electron0_pt
    )

    other_pt, other_eta, other_phi, _ = vector_subtraction(
        fatjets_pt,
        fatjets_eta,
        fatjets_phi,
        fatjets_mass,
        muon0_pt,
        muon0_eta,
        muon0_phi,
        muon0_mass,
    )
    massVis_boostedtau_mt_at = ca_mass(
        other_eta, muon0_eta, other_phi, muon0_phi, other_pt, muon0_pt
    )

    # old CA mass: two subjets, or fatjet + MET
    massVis_subjet = ca_mass(
        subjet0_eta, subjet1_eta, subjet0_phi, subjet1_phi, subjet0_pt, subjet1_pt
    )
    massVis_fatjet_tt = calculate_invariant_mass_2d(
        fatjets_globalParT_massVisApplied,
        fatjets_pt,
        fatjets_eta,
        fatjets_phi,
        fake_corr,
        met_pt,
        met_phi,
    )

    # later conditions take precedence
    output_map = {
        "CA_tau_number_in_fatjet": [
            (~no2tau, n_matched),
            (no2tau, n_matched),
        ],
        # this one is the lastest type of CA mass
        "CA_globalParT_massVisApplied_with_delta_axis_merged": [
            (~no2tau, massVis_boostedtau),  # 2tau
            ((~no1tau) & no2tau & (~no1electron), massVis_boostedtau_et),  # 1t1e0m
            ((~no1tau) & no2tau & (~no1muon), massVis_boostedtau_mt),  # 1t1m(0-1e)
            ((~no1tau) & no2tau & no1electron & no1muon, massVis_boostedtau_tat),  # 1t0e0m
            (no1tau & (~no1electron), massVis_boostedtau_et_at),  # 0t1e0m
            (no1tau & (~no1muon), massVis_boostedtau_mt_at),  # 0t0e1m
            (no1tau & no1electron & no1muon, fatjets_globalParT_massVisApplied),  # 0t0e0m
        ],
        "CA_globalParT_massVisApplied_oneHPSTauorLepton_flag": [
            (~no2tau, 1),  # 2tau
            ((~no1tau) & no2tau & (~no1electron), 2),  # 1t1e0m
            ((~no1tau) & no2tau & (~no1muon), 3),  # 1t1m(0-1e)
            ((~no1tau) & no2tau & no1electron & no1muon, 4),  # 1t0e0m
            (no1tau & (~no1electron), 5),  # 0t1e0m
            (no1tau & (~no1muon), 6),  # 0t0e1m
            (no1tau & no1electron & no1muon, 7),  # 0t0e0m
        ],
        # old CA mass, input of the current BDTs
        "CA_globalParT_massVisApplied": [
            (no2subjet & no2tau, massVis_fatjet_tt),
            (~no2subjet, massVis_subjet),
            (~no2tau, massVis_boostedtau),
        ],
        "CA_one_muon_in_fatjet": [(~no1muon, 1)],
        "CA_one_elec_in_fatjet": [(~no1electron, 1)],
    }

    for field, val_pairs in output_map.items():
        for condition, value in val_pairs:
            fatjets[field] = ak.where(condition, value, fatjets[field])

    return fatjets
