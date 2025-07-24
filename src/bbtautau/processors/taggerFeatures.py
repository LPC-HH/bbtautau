"""
Methods for deriving input variables for the tagger and running inference.

Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan
"""

from __future__ import annotations

# import onnxruntime as ort
import awkward as ak
import numpy as np
import numpy.ma as ma
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray
from numpy.typing import ArrayLike


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


def get_pfcands_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int | ArrayLike,
    jet: FatJetArray = None,
    fatjet_label: str = "FatJet",
    pfcands_label: str = "FatJetPFCands",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the pf_candidate features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    if jet is None:
        jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]

    jet_ak_pfcands = preselected_events[pfcands_label][
        preselected_events[pfcands_label].jetIdx == jet_idx
    ]
    jet_pfcands = preselected_events.PFCands[jet_ak_pfcands.pFCandsIdx]

    # sort them by pt
    pfcand_sort = ak.argsort(jet_pfcands.pt, ascending=False)
    jet_pfcands = jet_pfcands[pfcand_sort]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.ones_like(jet_pfcands.eta)
    eta_sign = eta_sign * (ak.values_astype(jet.eta > 0, int) * 2 - 1)
    feature_dict["pfcand_etarel"] = eta_sign * (jet_pfcands.eta - jet.eta)
    feature_dict["pfcand_phirel"] = jet_pfcands.delta_phi(jet)
    feature_dict["pfcand_abseta"] = np.abs(jet_pfcands.eta)

    feature_dict["pfcand_pt_log_nopuppi"] = np.log(jet_pfcands.pt)
    feature_dict["pfcand_e_log_nopuppi"] = np.log(jet_pfcands.energy)

    pdgIds = jet_pfcands.pdgId
    feature_dict["pfcand_isEl"] = np.abs(pdgIds) == 11
    feature_dict["pfcand_isMu"] = np.abs(pdgIds) == 13
    feature_dict["pfcand_isChargedHad"] = np.abs(pdgIds) == 211
    feature_dict["pfcand_isGamma"] = np.abs(pdgIds) == 22
    feature_dict["pfcand_isNeutralHad"] = np.abs(pdgIds) == 130

    feature_dict["pfcand_charge"] = jet_pfcands.charge
    feature_dict["pfcand_VTX_ass"] = jet_pfcands.pvAssocQuality
    feature_dict["pfcand_lostInnerHits"] = jet_pfcands.lostInnerHits
    feature_dict["pfcand_quality"] = jet_pfcands.trkQuality
    feature_dict["pfcand_normchi2"] = np.floor(jet_pfcands.trkChi2)

    if "Cdz" in jet_ak_pfcands.fields:
        feature_dict["pfcand_dz"] = jet_ak_pfcands["Cdz"][pfcand_sort]
        feature_dict["pfcand_dxy"] = jet_ak_pfcands["Cdxy"][pfcand_sort]
        feature_dict["pfcand_dzsig"] = jet_ak_pfcands["Cdzsig"][pfcand_sort]
        feature_dict["pfcand_dxysig"] = jet_ak_pfcands["Cdxysig"][pfcand_sort]
    else:
        # this is for old PFNano (<= v2.3)
        feature_dict["pfcand_dz"] = jet_pfcands.dz
        feature_dict["pfcand_dxy"] = jet_pfcands.d0
        feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
        feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err

    feature_dict["pfcand_px"] = jet_pfcands.px
    feature_dict["pfcand_py"] = jet_pfcands.py
    feature_dict["pfcand_pz"] = jet_pfcands.pz
    feature_dict["pfcand_energy"] = jet_pfcands.E

    for var in tagger_vars["pf_features"]["var_names"]:
        if "btag" in var:
            feature_dict[var] = jet_ak_pfcands[var[len("pfcand_") :]][pfcand_sort]

    feature_dict["pfcand_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["pfcand_abseta"],
                    tagger_vars["pf_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)

    # if no padding is needed, mask will = 1.0
    if isinstance(feature_dict["pfcand_mask"], np.float32):
        feature_dict["pfcand_mask"] = np.ones(
            (len(feature_dict["pfcand_abseta"]), tagger_vars["pf_features"]["var_length"])
        ).astype(np.float32)

    repl_values_dict = {
        "pfcand_normchi2": [-1, 999],
        "pfcand_dz": [-1, 0],
        "pfcand_dzsig": [1, 0],
        "pfcand_dxy": [-1, 0],
        "pfcand_dxysig": [1, 0],
    }

    # convert to numpy arrays and normalize features
    for var in set(
        tagger_vars["pf_features"]["var_names"] + tagger_vars["pf_vectors"]["var_names"]
    ):
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["pf_features"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        a = np.nan_to_num(a)

        # replace values to match PKU's
        if var in repl_values_dict:
            vals = repl_values_dict[var]
            a[a == vals[0]] = vals[1]

        if normalize:
            if var in tagger_vars["pf_features"]["var_names"]:
                info = tagger_vars["pf_features"]["var_infos"][var]
            else:
                info = tagger_vars["pf_vectors"]["var_infos"][var]

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_svs_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int | ArrayLike,
    jet: FatJetArray = None,
    fatjet_label: str = "FatJet",
    svs_label: str = "JetSVs",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the sv features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    if jet is None:
        jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]

    jet_svs = preselected_events.SV[
        preselected_events[svs_label].sVIdx[
            (preselected_events[svs_label].sVIdx != -1)
            * (preselected_events[svs_label].jetIdx == jet_idx)
        ]
    ]

    # sort by dxy significance
    jet_svs = jet_svs[ak.argsort(jet_svs.dxySig, ascending=False)]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.ones_like(jet_svs.eta)
    eta_sign = eta_sign * (ak.values_astype(jet.eta > 0, int) * 2 - 1)

    feature_dict["sv_etarel"] = eta_sign * (jet_svs.eta - jet.eta)
    feature_dict["sv_phirel"] = jet_svs.delta_phi(jet)
    feature_dict["sv_abseta"] = np.abs(jet_svs.eta)
    feature_dict["sv_mass"] = jet_svs.mass
    feature_dict["sv_pt_log"] = np.log(jet_svs.pt)

    feature_dict["sv_ntracks"] = jet_svs.ntracks
    feature_dict["sv_normchi2"] = jet_svs.chi2
    feature_dict["sv_dxy"] = jet_svs.dxy
    feature_dict["sv_dxysig"] = jet_svs.dxySig
    feature_dict["sv_d3d"] = jet_svs.dlen
    feature_dict["sv_d3dsig"] = jet_svs.dlenSig
    svpAngle = jet_svs.pAngle
    feature_dict["sv_costhetasvpv"] = -np.cos(svpAngle)

    feature_dict["sv_px"] = jet_svs.px
    feature_dict["sv_py"] = jet_svs.py
    feature_dict["sv_pz"] = jet_svs.pz
    feature_dict["sv_energy"] = jet_svs.E

    feature_dict["sv_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["sv_etarel"],
                    tagger_vars["sv_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)

    # if no padding is needed, mask will = 1.0
    if isinstance(feature_dict["sv_mask"], np.float32):
        feature_dict["sv_mask"] = np.ones(
            (len(feature_dict["sv_abseta"]), tagger_vars["sv_features"]["var_length"])
        ).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in set(
        tagger_vars["sv_features"]["var_names"] + tagger_vars["sv_vectors"]["var_names"]
    ):
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["sv_features"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        a = np.nan_to_num(a)

        if normalize:
            if var in tagger_vars["sv_features"]["var_names"]:
                info = tagger_vars["sv_features"]["var_infos"][var]
            else:
                info = tagger_vars["sv_vectors"]["var_infos"][var]

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_met_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int,
    fatjet_label: str = "FatJetAK15",
    met_label: str = "MET",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the MET features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]
    met = preselected_events[met_label]

    # get features
    feature_dict["met_relpt"] = met.pt / jet.pt
    feature_dict["met_relphi"] = met.delta_phi(jet)

    for var in tagger_vars["met_features"]["var_names"]:
        a = (
            # ak.pad_none(
            #     feature_dict[var], tagger_vars["met_features"]["var_length"], axis=1, clip=True
            # )
            feature_dict[var]  # just 1d, no pad_none
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        if normalize:
            info = tagger_vars["met_features"]["var_infos"][var]
            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_lep_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int,
    fatjet_label: str = "FatJet",
    muon_label: str = "Muon",
    electron_label: str = "Electron",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the lepton features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]
    jet_muons = preselected_events.Muon[preselected_events[muon_label].jetIdx == jet_idx]
    jet_electrons = preselected_events.Electron[
        preselected_events[electron_label].jetIdx == jet_idx
    ]

    # get features of leading leptons
    leptons = ak.concatenate([jet_muons, jet_electrons], axis=1)
    index_lep = ak.argsort(leptons.pt, ascending=False)
    leptons = leptons[index_lep]
    lepton_cand = ak.firsts(leptons)
    lepton = build_p4(lepton_cand)

    # print(ak.firsts(leptons).__repr__)
    electron_iso = jet_electrons.pfRelIso03_all
    electron_pdgid = np.abs(jet_electrons.charge) * 11
    muon_iso = jet_muons.pfRelIso04_all
    muon_pdgid = np.abs(jet_muons.charge) * 13
    feature_dict["lep_iso"] = (
        ak.firsts(ak.concatenate([muon_iso, electron_iso], axis=1)[index_lep])
        .to_numpy()
        .filled(fill_value=0)
    )
    feature_dict["lep_pdgId"] = (
        ak.firsts(ak.concatenate([muon_pdgid, electron_pdgid], axis=1)[index_lep])
        .to_numpy()
        .filled(fill_value=0)
    )

    # this is for features that are shared
    feature_dict["lep_dR_fj"] = lepton.delta_r(jet).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt"] = (lepton.pt).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt_ratio"] = (lepton.pt / jet.pt).to_numpy().filled(fill_value=0)
    feature_dict["lep_miniiso"] = lepton_cand.miniPFRelIso_all.to_numpy().filled(fill_value=0)

    # get features
    if "el_features" in tagger_vars:
        feature_dict["elec_pt"] = jet_electrons.pt / jet.pt
        feature_dict["elec_eta"] = jet_electrons.eta - jet.eta
        feature_dict["elec_phi"] = jet_electrons.delta_phi(jet)
        feature_dict["elec_mass"] = jet_electrons.mass
        feature_dict["elec_charge"] = jet_electrons.charge
        feature_dict["elec_convVeto"] = jet_electrons.convVeto
        feature_dict["elec_deltaEtaSC"] = jet_electrons.deltaEtaSC
        feature_dict["elec_dr03EcalRecHitSumEt"] = jet_electrons.dr03EcalRecHitSumEt
        feature_dict["elec_dr03HcalDepth1TowerSumEt"] = jet_electrons.dr03HcalDepth1TowerSumEt
        feature_dict["elec_dr03TkSumPt"] = jet_electrons.dr03TkSumPt
        feature_dict["elec_dxy"] = jet_electrons.dxy
        feature_dict["elec_dxyErr"] = jet_electrons.dxyErr
        feature_dict["elec_dz"] = jet_electrons.dz
        feature_dict["elec_dzErr"] = jet_electrons.dzErr
        feature_dict["elec_eInvMinusPInv"] = jet_electrons.eInvMinusPInv
        feature_dict["elec_hoe"] = jet_electrons.hoe
        feature_dict["elec_ip3d"] = jet_electrons.ip3d
        feature_dict["elec_lostHits"] = jet_electrons.lostHits
        feature_dict["elec_r9"] = jet_electrons.r9
        feature_dict["elec_sieie"] = jet_electrons.sieie
        feature_dict["elec_sip3d"] = jet_electrons.sip3d
        # convert to numpy arrays and normalize features
        for var in tagger_vars["el_features"]["var_names"]:
            a = (
                ak.pad_none(
                    feature_dict[var], tagger_vars["el_features"]["var_length"], axis=1, clip=True
                )
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)

            if normalize:
                info = tagger_vars["el_features"]["var_infos"][var]
                a = (a - info["median"]) * info["norm_factor"]
                a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

            feature_dict[var] = a

    if "mu_features" in tagger_vars:
        feature_dict["muon_pt"] = jet_muons.pt / jet.pt
        feature_dict["muon_eta"] = jet_muons.eta - jet.eta
        feature_dict["muon_phi"] = jet_muons.delta_phi(jet)
        feature_dict["muon_mass"] = jet_muons.mass
        feature_dict["muon_charge"] = jet_muons.charge
        feature_dict["muon_dxy"] = jet_muons.dxy
        feature_dict["muon_dxyErr"] = jet_muons.dxyErr
        feature_dict["muon_dz"] = jet_muons.dz
        feature_dict["muon_dzErr"] = jet_muons.dzErr
        feature_dict["muon_ip3d"] = jet_muons.ip3d
        feature_dict["muon_nStations"] = jet_muons.nStations
        feature_dict["muon_nTrackerLayers"] = jet_muons.nTrackerLayers
        feature_dict["muon_pfRelIso03_all"] = jet_muons.pfRelIso03_all
        feature_dict["muon_pfRelIso03_chg"] = jet_muons.pfRelIso03_chg
        feature_dict["muon_segmentComp"] = jet_muons.segmentComp
        feature_dict["muon_sip3d"] = jet_muons.sip3d
        feature_dict["muon_tkRelIso"] = jet_muons.tkRelIso
        # convert to numpy arrays and normalize features
        for var in tagger_vars["mu_features"]["var_names"]:
            a = (
                ak.pad_none(
                    feature_dict[var], tagger_vars["mu_features"]["var_length"], axis=1, clip=True
                )
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)

            if normalize:
                info = tagger_vars["mu_features"]["var_infos"][var]
                a = (a - info["median"]) * info["norm_factor"]
                a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

            feature_dict[var] = a

    return feature_dict


def get_pfcands_evt_features(events, fatjet, jet_idx):
    """
    Extracts the pf_candidate features from the ``events`` and returns them as numpy arrays
    """
    feature_dict = {}
    evt_feature_dict = {}

    jet_pfcands = events.PFCands[
        events.FatJetPFCands.pFCandsIdx[
            (events.FatJetPFCands.pFCandsIdx != -1)
            * (events.FatJetPFCands.jetIdx == ak.flatten(jet_idx, axis=1))
        ]
    ]
    ptsorting = ak.argsort(jet_pfcands.pt, axis=-1, ascending=False)
    jet_pfcands = jet_pfcands[ptsorting]

    # print('jet_pfcands.pt',jet_pfcands.pt)
    # print('fatjet.pt',fatjet.pt)

    feature_dict["pf_pt"] = jet_pfcands.pt / fatjet.pt
    feature_dict["pf_eta"] = jet_pfcands.eta - fatjet.eta
    feature_dict["pf_phi"] = -fatjet.delta_phi(jet_pfcands)
    feature_dict["pf_charge"] = jet_pfcands.charge
    feature_dict["pf_pdgId"] = jet_pfcands.pdgId
    feature_dict["pf_dz"] = jet_pfcands.dz
    feature_dict["pf_dzErr"] = jet_pfcands.dzErr
    feature_dict["pf_d0"] = jet_pfcands.d0
    feature_dict["pf_d0Err"] = jet_pfcands.d0Err
    # feature_dict["pfcand_dz"] = jet_pfcands.dz
    # feature_dict["pfcand_dxy"] = jet_pfcands.d0
    # feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
    # feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err
    feature_dict["pf_dz"] = ak.fill_none(
        ak.mask(feature_dict["pf_dz"], feature_dict["pf_charge"] != 0.0), 0.0
    )
    feature_dict["pf_dzErr"] = ak.fill_none(
        ak.mask(feature_dict["pf_dzErr"], feature_dict["pf_charge"] != 0.0), 0.0
    )
    feature_dict["pf_d0"] = ak.fill_none(
        ak.mask(feature_dict["pf_d0"], feature_dict["pf_charge"] != 0.0), 0.0
    )
    feature_dict["pf_d0Err"] = ak.fill_none(
        ak.mask(feature_dict["pf_d0Err"], feature_dict["pf_charge"] != 0.0), 0.0
    )
    # feature_dict["pf_dxy"] = ak.fill_none(ak.mask(feature_dict["pf_dxy"],feature_dict["pf_charge"]!=0.),0.)
    # feature_dict["pf_dxyErr"] = ak.fill_none(ak.mask(feature_dict["pf_dxyErr"],feature_dict["pf_charge"]!=0.),0.)
    feature_dict["pf_puppiWeight"] = jet_pfcands.puppiWeight
    feature_dict["pf_puppiWeightNoLep"] = jet_pfcands.puppiWeightNoLep
    feature_dict["pf_trkChi2"] = jet_pfcands.trkChi2
    feature_dict["pf_vtxChi2"] = jet_pfcands.vtxChi2

    for iid, pid in enumerate([0.0, 211.0, 13.0, 22.0, 11.0, 130.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
        feature_dict["pf_id%i" % iid] = np.abs(feature_dict["pf_pdgId"]) == pid
    feature_dict["pf_idreg"] = feature_dict["pf_id0"] * 0.0
    for iid in range(1, 11):
        feature_dict["pf_idreg"] = feature_dict["pf_idreg"] + feature_dict["pf_id%i" % iid] * float(
            iid
        )

    feature_dict["met_covXX"] = events.MET.covXX
    feature_dict["met_covXY"] = events.MET.covXY
    feature_dict["met_covYY"] = events.MET.covYY
    # feature_dict['met_dphi'] = fatjet.delta_phi(events.MET)
    feature_dict["met_dphi"] = events.MET.delta_phi(fatjet)
    feature_dict["met_pt"] = events.MET.pt
    feature_dict["met_significance"] = events.MET.significance
    feature_dict["pupmet_pt"] = events.PuppiMET.pt
    # feature_dict['pupmet_dphi'] = fatjet.delta_phi(events.PuppiMET)
    feature_dict["pupmet_dphi"] = events.PuppiMET.delta_phi(fatjet)
    feature_dict["jet_pt"] = fatjet.pt
    feature_dict["jet_eta"] = fatjet.eta
    feature_dict["jet_phi"] = fatjet.phi
    feature_dict["jet_msd"] = fatjet.msoftdrop
    feature_dict["jet_muonenergy"] = ak.sum(
        ak.mask(feature_dict["pf_pt"], np.abs(feature_dict["pf_pdgId"]) == 13.0), 1
    )
    feature_dict["jet_elecenergy"] = ak.sum(
        ak.mask(feature_dict["pf_pt"], np.abs(feature_dict["pf_pdgId"]) == 11.0), 1
    )
    feature_dict["jet_photonenergy"] = ak.sum(
        ak.mask(feature_dict["pf_pt"], feature_dict["pf_pdgId"] == 22.0), 1
    )
    feature_dict["jet_chhadronenergy"] = ak.sum(
        ak.mask(feature_dict["pf_pt"], np.abs(feature_dict["pf_pdgId"]) == 211.0), 1
    )
    feature_dict["jet_nehadronenergy"] = ak.sum(
        ak.mask(feature_dict["pf_pt"], feature_dict["pf_pdgId"] == 130.0), 1
    )
    feature_dict["jet_muonnum"] = ak.sum((np.abs(feature_dict["pf_pdgId"]) == 13.0), 1)
    feature_dict["jet_elecnum"] = ak.sum((np.abs(feature_dict["pf_pdgId"]) == 11.0), 1)
    feature_dict["jet_photonnum"] = ak.sum((feature_dict["pf_pdgId"] == 22.0), 1)
    feature_dict["jet_chhadronnum"] = ak.sum((np.abs(feature_dict["pf_pdgId"]) == 211.0), 1)
    feature_dict["jet_nehadronnum"] = ak.sum((feature_dict["pf_pdgId"] == 130.0), 1)
    feature_dict["jet_unity"] = fatjet.pt / fatjet.pt
    if len(events.MET.covXX) > 1:
        # convert to numpy arrays and normalize features
        for var in feature_dict:
            a = (
                ak.pad_none(
                    (
                        feature_dict[var]
                        if var.startswith("pf")
                        else ak.unflatten(feature_dict[var], 1)
                    ),
                    _Nparts if var.startswith("pf") else 1,
                    axis=1,
                    clip=True,
                )
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)
            feature_dict[var] = a
    else:
        for var in feature_dict:
            if var.startswith("pf"):
                a = (
                    ak.pad_none(feature_dict[var], _Nparts, axis=1, clip=True)
                    .to_numpy()
                    .filled(fill_value=0)
                ).astype(np.float32)
            else:
                a = np.array(feature_dict[var]).astype(np.float32)
            feature_dict[var] = a

    return feature_dict


def get_svs_features(events, fatjet, jet_idx):
    """
    Extracts the sv features specified from the
    ``events`` and returns them as numpy arrays
    """
    feature_dict = {}

    sv_p4 = ak.zip(
        {"pt": events.SV.pt, "eta": events.SV.eta, "phi": events.SV.phi, "mass": events.SV.mass},
        behavior=vector.behavior,
        with_name="PtEtaPhiMLorentzVector",
    )

    sv_ak8_pair = ak.cartesian((sv_p4, fatjet))
    jet_svs = events.SV[sv_ak8_pair[:, :, "1"].delta_r(sv_ak8_pair[:, :, "0"]) < 0.8]

    # get features

    feature_dict["sv_pt"] = jet_svs.pt / fatjet.pt
    feature_dict["sv_eta"] = jet_svs.eta - fatjet.eta
    feature_dict["sv_phi"] = jet_svs.p4.delta_phi(fatjet)
    feature_dict["sv_mass"] = jet_svs.mass
    feature_dict["sv_dlen"] = jet_svs.dlen
    feature_dict["sv_dlenSig"] = jet_svs.dlenSig
    feature_dict["sv_dxy"] = jet_svs.dxy
    feature_dict["sv_dxySig"] = jet_svs.dxySig
    feature_dict["sv_chi2"] = jet_svs.chi2
    feature_dict["sv_pAngle"] = jet_svs.pAngle
    feature_dict["sv_x"] = jet_svs.x
    feature_dict["sv_y"] = jet_svs.y
    feature_dict["sv_z"] = jet_svs.z

    # del sv_ak8_pair
    del jet_svs

    # convert to numpy arrays and normalize features
    for var in feature_dict:
        a = (
            ak.pad_none(feature_dict[var], _Nsvs, axis=1, clip=True).to_numpy().filled(fill_value=0)
        ).astype(np.float32)

        # info = tagger_vars["sv_features"]["var_infos"][var]
        # a = (a - info["median"]) * info["norm_factor"]
        # a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_electron_features(events, fatjet, elec_features):
    """
    Extracts the electrons features specified from the ``events`` and returns them as numpy arrays
    """
    feature_dict = {}

    elec_p4 = build_p4(events.Electron)
    jet_elecs = events.Electron[elec_p4.delta_r(fatjet) < 0.8]

    jet_elecs["rel_pt"] = jet_elecs.pt / fatjet.pt
    jet_elecs["deta"] = jet_elecs.eta - fatjet.eta
    jet_elecs["dphi"] = jet_elecs.delta_phi(fatjet)

    # get features
    feature_dict["elec_pt"] = jet_elecs.pt / fatjet.pt
    feature_dict["elec_eta"] = jet_elecs.eta - fatjet.eta
    feature_dict["elec_phi"] = jet_elecs.delta_phi(fatjet)
    feature_dict["elec_mass"] = jet_elecs.mass
    feature_dict["elec_charge"] = jet_elecs.charge
    feature_dict["elec_convVeto"] = jet_elecs.convVeto
    feature_dict["elec_deltaEtaSC"] = jet_elecs.deltaEtaSC
    feature_dict["elec_dr03EcalRecHitSumEt"] = jet_elecs.dr03EcalRecHitSumEt
    feature_dict["elec_dr03HcalDepth1TowerSumEt"] = jet_elecs.dr03HcalDepth1TowerSumEt
    feature_dict["elec_dr03TkSumPt"] = jet_elecs.dr03TkSumPt
    feature_dict["elec_dxy"] = jet_elecs.dxy
    feature_dict["elec_dxyErr"] = jet_elecs.dxyErr
    feature_dict["elec_dz"] = jet_elecs.dz
    feature_dict["elec_dzErr"] = jet_elecs.dzErr
    feature_dict["elec_eInvMinusPInv"] = jet_elecs.eInvMinusPInv
    feature_dict["elec_hoe"] = jet_elecs.hoe
    feature_dict["elec_ip3d"] = jet_elecs.ip3d
    feature_dict["elec_lostHits"] = jet_elecs.lostHits
    feature_dict["elec_r9"] = jet_elecs.r9
    feature_dict["elec_sieie"] = jet_elecs.sieie
    feature_dict["elec_sip3d"] = jet_elecs.sip3d

    # convert to numpy arrays and normalize features
    for var in feature_dict:
        a = (
            ak.pad_none(feature_dict[var], _Nelecs, axis=1, clip=True)
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        # info = tagger_vars["sv_features"]["var_infos"][var]
        # a = (a - info["median"]) * info["norm_factor"]
        # a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_muons_features(events, fatjet, jet_idx):
    """
    Extracts the muon features specified from the
    ``events`` and returns them as numpy arrays
    """
    feature_dict = {}

    muon_p4 = ak.zip(
        {
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
        },
        behavior=vector.behavior,
        with_name="PtEtaPhiMLorentzVector",
    )

    muon_ak8_pair = ak.cartesian((muon_p4, fatjet))
    jet_muons = events.Muon[muon_ak8_pair[:, :, "0"].delta_r(muon_ak8_pair[:, :, "1"]) < 0.8]

    # get features

    feature_dict["muon_pt"] = jet_muons.pt / fatjet.pt
    feature_dict["muon_eta"] = jet_muons.eta - fatjet.eta
    feature_dict["muon_phi"] = jet_muons.delta_phi(fatjet)
    feature_dict["muon_mass"] = jet_muons.mass
    feature_dict["muon_charge"] = jet_muons.charge
    feature_dict["muon_dxy"] = jet_muons.dxy
    feature_dict["muon_dxyErr"] = jet_muons.dxyErr
    feature_dict["muon_dz"] = jet_muons.dz
    feature_dict["muon_dzErr"] = jet_muons.dzErr
    feature_dict["muon_ip3d"] = jet_muons.ip3d
    feature_dict["muon_nStations"] = jet_muons.nStations
    feature_dict["muon_nTrackerLayers"] = jet_muons.nTrackerLayers
    feature_dict["muon_pfRelIso03_all"] = jet_muons.pfRelIso03_all
    feature_dict["muon_pfRelIso03_chg"] = jet_muons.pfRelIso03_chg
    feature_dict["muon_segmentComp"] = jet_muons.segmentComp
    feature_dict["muon_sip3d"] = jet_muons.sip3d
    feature_dict["muon_tkRelIso"] = jet_muons.tkRelIso

    del muon_ak8_pair
    del jet_muons

    # convert to numpy arrays and normalize features
    for var in feature_dict:
        a = (
            ak.pad_none(feature_dict[var], _Nmuons, axis=1, clip=True)
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        # info = tagger_vars["muon_features"]["var_infos"][var]
        # a = (a - info["median"]) * info["norm_factor"]
        # a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_taus_features(events, fatjet, jet_idx):
    """
    Extracts the tau features specified from the
    ``events`` and returns them as numpy arrays
    """
    feature_dict = {}

    tau_p4 = ak.zip(
        {
            "pt": events.boostedTau.pt,
            "eta": events.boostedTau.eta,
            "phi": events.boostedTau.phi,
            "mass": events.boostedTau.mass,
        },
        behavior=vector.behavior,
        with_name="PtEtaPhiMLorentzVector",
    )

    tau_ak8_pair = ak.cartesian((tau_p4, fatjet))
    # jet_taus = events.Tau[tau_ak8_pair[:,:,'0'].delta_r(tau_ak8_pair[:,:,'1'])<0.8] #preUL
    jet_taus = events.boostedTau[tau_ak8_pair[:, :, "0"].delta_r(tau_ak8_pair[:, :, "1"]) < 0.8]

    # get features

    taupt_scale = 1.0

    feature_dict["tau_pt"] = taupt_scale * jet_taus.pt / fatjet.pt
    feature_dict["tau_eta"] = jet_taus.eta - fatjet.eta
    feature_dict["tau_phi"] = fatjet.delta_phi(jet_taus)
    feature_dict["tau_mass"] = jet_taus.mass
    feature_dict["tau_charge"] = jet_taus.charge
    feature_dict["tau_chargedIso"] = jet_taus.chargedIso
    # feature_dict["tau_dxy"] = jet_taus.dxy
    # feature_dict["tau_dz"] = jet_taus.dz
    feature_dict["tau_leadTkDeltaEta"] = jet_taus.leadTkDeltaEta
    feature_dict["tau_leadTkDeltaPhi"] = jet_taus.leadTkDeltaPhi
    feature_dict["tau_leadTkPtOverTauPt"] = jet_taus.leadTkPtOverTauPt
    feature_dict["tau_neutralIso"] = jet_taus.neutralIso
    feature_dict["tau_photonsOutsideSignalCone"] = jet_taus.photonsOutsideSignalCone
    # feature_dict["tau_rawAntiEle"] = jet_taus.rawAntiEle
    feature_dict["tau_rawAntiEle2018"] = jet_taus.rawAntiEle2018
    feature_dict["tau_rawAntiEle"] = jet_taus.rawIso
    feature_dict["tau_rawIso"] = jet_taus.rawIso
    feature_dict["tau_rawIsodR03"] = jet_taus.rawIsodR03
    # feature_dict["tau_rawMVAoldDM2017v2"] = jet_taus.rawMVAoldDM2017v2
    feature_dict["tau_rawMVAoldDM2017v2"] = jet_taus.rawIso
    # feature_dict["tau_rawMVAoldDMdR032017v2"] = jet_taus.rawMVAoldDMdR032017v2
    feature_dict["tau_rawMVAoldDMdR032017v2"] = jet_taus.rawIso

    del tau_ak8_pair
    del jet_taus

    # convert to numpy arrays and normalize features
    for var in feature_dict:
        a = (
            ak.pad_none(feature_dict[var], _Ntaus, axis=1, clip=True)
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        # info = tagger_vars["tau_features"]["var_infos"][var]
        # a = (a - info["median"]) * info["norm_factor"]
        # a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict
