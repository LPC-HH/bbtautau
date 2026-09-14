"""
Skimmer for bbtautau analysis.
Based on https://github.com/LPC-HH/HH4b/blob/main/src/HH4b/processors/bbbbSkimmer.py.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import logging
import pathlib
import time
from collections import OrderedDict

import awkward as ak
import numpy as np
from boostedhh import hh_vars
from boostedhh.processors import SkimmerABC, utils
from boostedhh.processors.corrections import (
    JECs,
    add_pileup_weight_update,
    add_ps_weight,
    get_jetveto_event,
    get_pdf_weights,
    get_scale_weights,
    # SF, Up/down
    get_tau_tes,
    get_tau_vsjet_sf,
    get_tau_trigger_sf,
    get_muon_scale_smearing,
    get_muon_id_sfs,
    get_muon_trigger_sfs,
    get_electron_scale_smearing,
    get_electron_reco_sfs,
    get_electron_id_sfs,
    get_electron_trigger_sfs,
    # get_btag_sfs,
)
from boostedhh.processors.utils import (
    P4,
    PAD_VAL,
    add_selection,
    pad_val,
)
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from bbtautau.HLTs import HLTs

from . import GenSelection, objects

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HHto4B": GenSelection.gen_selection_HH4b,
    "HHto2B2Tau": GenSelection.gen_selection_HHbbtautau,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


class bbtautauSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # name in nano files: name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "Jet": {
            **P4,
            "rawFactor": "rawFactor",
            "btagPNetB": "btagPNetB",
        },
        "MET": {
            "pt": "Pt",
            "phi": "Phi",
            "significance": "significance",
        },
        "Lepton": {
            **P4,
            "charge": "charge",
        },
        "Tau": {
            **P4,
            "charge": "charge",
            "idDeepTau2018v2p5VSjet": "DeepTauvsJet",
            "idDeepTau2018v2p5VSmu": "DeepTauvsMu",
            "idDeepTau2018v2p5VSe": "DeepTauvsE",
        },
        "BoostedTau": {
            **P4,
            "charge": "charge",
            "idMVAnewDM2017v2": "idMVAnewDM2017v2",
        },
        "SubJet": {
            **P4,
        },
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "t32": "Tau3OverTau2",
            "rawFactor": "rawFactor",
            "particleNet_massCorr": "particleNet_massCorr",
            # tagger variables added below
        },
        "GenHiggs": P4,
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
        "Pileup": {
            "nPU",
        },
        "TriggerObject": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "filterBits": "Bit",
        },
    }

    # only applied if fatjet_bb_preselection is True
    preselection = {  # noqa: RUF012
        # roughly, 85% signal efficiency, 2% QCD efficiency (pT: 250-400, mSD:0-250, mRegLegacy:40-250)
        "pnet-legacy": 0.8,
        "pnet-v12": 0.3,
        "glopart-v2": 0.3,
    }

    fatjet_selection = {  # noqa: RUF012
        "object_pt": 170,
        "pt": 230,
        "eta": 2.5,
        "msd": 50,
        "mreg": 0,
    }

    vbf_jet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 4.7,
        "id": "tight",
        "dr_fatjets": 1.2,
        "dr_leptons": 0.4,
    }

    vbf_veto_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
    }

    ak4_bjet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 2.5,
        "id": "tight",
        "dr_fatjets": 0.9,
        "dr_leptons": 0.4,
    }

    ak4_bjet_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
    }

    def __init__(
        self,
        xsecs: dict = None,
        save_systematics: bool = False,
        region: str = "signal",
        nano_version: str = "v12_private",
        fatjet_pt_cut: float = None,
        fatjet_bb_preselection: bool = False,
        prescale_factor: int = None,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        # HLT selection
        self.HLTs = {"signal": HLTs.hlt_list(hlt_prefix=False)}
        self.HLTs = self.HLTs[region]
        self._systematics = save_systematics
        self._nano_version = nano_version
        self._region = region
        self._accumulator = processor.dict_accumulator({})
        self._fatjet_bb_preselection = fatjet_bb_preselection
        self._prescale_factor = prescale_factor

        # JMSR
        self.jmsr_vars = ["msoftdrop", "particleNet_mass_legacy", "ParTmassVis", "ParTmassRes"]

        # particlenet NOT legacy variables
        pnet_vars = [
            "XbbVsQCD",
            "XteVsQCD",
            "XtmVsQCD",
            "XttVsQCD",
        ]
        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"particleNet_{var}": f"PNet{var}" for var in pnet_vars},
        }

        # 2022+2023, v12, more QCD/Top/ParTs var
        if nano_version.startswith("v12"):

            pnet_vars_legacy = [
                "Xbb",
                "QCD",
                "QCDb",
                "QCDbb",
                "QCDcc",
                "QCDc",
                "QCDothers",
                "XbbvsQCD",
                "mass",
            ]
            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                **{f"particleNetLegacy_{var}": f"PNet{var}Legacy" for var in pnet_vars_legacy},
            }

            # glopart variables
            glopart_vars = [
                "QCD1HF",
                "QCD2HF",
                "QCD0HF",
                "TopW",
                "TopbW",
                "TopbWev",
                "TopbWmv",
                "TopbWtauhv",
                "TopbWq",
                "TopbWqq",
                "Xbb",
                "Xcc",
                "Xcs",
                "Xgg",
                "Xqq",
                "Xtauhtaue",
                "Xtauhtauh",
                "Xtauhtaum",
                # Derived variables
                "massResCorr",
                "massVisCorr",
                "massResApplied",
                "massVisApplied",
                "QCD",
                "Top",
                "XbbvsQCD",
                "XbbvsQCDTop",
                "XtauhtauevsQCD",
                "XtauhtauevsQCDTop",
                "XtauhtaumvsQCD",
                "XtauhtaumvsQCDTop",
                "XtauhtauhvsQCD",
                "XtauhtauhvsQCDTop",
            ]

            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                **{f"globalParT_{var}": f"ParT{var}" for var in glopart_vars},
            }

        elif nano_version.startswith("v15"):

            pnet_vars_legacy = [
                "Xbb",
                "QCD",
                "XbbvsQCD",
                "mass",
            ]
            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                **{f"particleNetLegacy_{var}": f"PNet{var}Legacy" for var in pnet_vars_legacy},
            }

            # glopart variables
            glopart_vars = [
                "TopbWev",
                "TopbWmv",
                "TopbWq",
                "TopbWqq",
                "TopbWtauhv",
                "Xbb",
                "Xcc",
                "Xcs",
                "Xqq",
                "Xtauhtaue",
                "Xtauhtauh",
                "Xtauhtaum",
                # Derived variables
                "massResCorr",
                "massVisCorr",
                "massResApplied",
                "massVisApplied",
                "QCD",
                "Top",
                "XbbvsQCD",
                "XbbvsQCDTop",
                "XtauhtauevsQCD",
                "XtauhtauevsQCDTop",
                "XtauhtaumvsQCD",
                "XtauhtaumvsQCDTop",
                "XtauhtauhvsQCD",
                "XtauhtauhvsQCDTop",
            ]

            self.skim_vars["FatJet"] = {
                **self.skim_vars["FatJet"],
                **{f"globalParT_{var}": f"ParT{var}" for var in glopart_vars},
            }

        # CA variables
        ca_vars = [
            "tau_number",
            "tau_number_in_fatjet",
            "globalParT_massVisApplied_oneHPSTau",
            "globalParT_massVisApplied_oneHPSTau_thth",
            "globalParT_massVisApplied_oneHPSTauorMuon_thtm",
            "globalParT_massVisApplied_oneHPSTauorElectron_thte",
            "globalParT_massVisApplied_with_delta_axis_merged",
            "globalParT_massVisApplied_oneHPSTauorLepton_flag",
            "globalParT_massVisApplied_000_fatjetwithMET",
            "globalParT_massVisApplied_000_fatjet",
            "globalParT_massVisApplied_000_fatjet_MET_with_same_dirc",
            "mass_merged",
            "msoftdrop_merged",
            "globalParT_massVisApplied_merged",
            "globalParT_massResApplied_merged",
            "particleNet_mass_legacy_merged",
            "Tauflag",
            "one_elec_in_fatjet",
            "one_muon_in_fatjet",
            "one_elec",
            "one_muon",
            "mass_fatjet_et",
            "mass_fatjet_mt",
            "isDauTau",
            "mass",
            "msoftdrop",
            "globalParT_massVisApplied",
            "globalParT_massResApplied",
            "particleNet_mass_legacy",
            "dau0_pt",
            "dau1_pt",
            "dau0_eta",
            "dau1_eta",
            "dau0_phi",
            "dau1_phi",
            "dau0_mass",
            "dau1_mass",
            "ntaus_perfatjets",
            "mass_subjets",
            "mass_boostedtaus",
            "nsubjets_perfatjets",
            "mass_fatjets",
            "mass_mt",
            "msoftdrop_mt",
            "globalParT_massVisApplied_mt",
            "globalParT_massResApplied_mt",
            "particleNet_mass_legacy_mt",
            "isDauTau_mt",
            "dau0_pt_mt",
            "dau1_pt_mt",
            "dau0_eta_mt",
            "dau1_eta_mt",
            "dau0_phi_mt",
            "dau1_phi_mt",
            "dau0_mass_mt",
            "dau1_mass_mt",
            "ntaus_perfatjets_mt",
            "mass_subjets_mt",
            "mass_boostedtaus_mt",
            "nsubjets_perfatjets_mt",
            "mass_subjets_mt_01",
            "muon_subjet_dr02",
            "mass_subjets_mt_1",
            "mass_subjets_mt_0",
            "mass_et",
            "msoftdrop_et",
            "globalParT_massVisApplied_et",
            "globalParT_massResApplied_et",
            "particleNet_mass_legacy_et",
            "isDauTau_et",
            "dau0_pt_et",
            "dau1_pt_et",
            "dau0_eta_et",
            "dau1_eta_et",
            "dau0_phi_et",
            "dau1_phi_et",
            "dau0_mass_et",
            "dau1_mass_et",
            "ntaus_perfatjets_et",
            "mass_subjets_et",
            "mass_boostedtaus_et",
            "nsubjets_perfatjets_et",
            "mass_subjets_et_01",
            "elec_subjet_dr02",
            "mass_subjets_et_1",
            "mass_subjets_et_0",
        ]

        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"CA_{var}": f"CA{var}" for var in ca_vars},
        }

        # update fatjet pT cut
        if fatjet_pt_cut is not None:
            self.fatjet_selection["pt"] = fatjet_pt_cut

        logger.info(
            f"Running skimmer with:\nsystematics {self._systematics}\nregion {self._region}\nfatjet pt cut {self.fatjet_selection['pt']}"
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        start = time.time()
        logging.info(f"# events {len(events)}")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = not hasattr(events, "genWeight")

        # datasets for saving jec variations
        isJECs = (  # noqa: F841
            "HHto4B" in dataset
            or "TT" in dataset
            or "Wto2Q" in dataset
            or "Zto2Q" in dataset
            or "Hto2B" in dataset
            or "WW" in dataset
            or "ZZ" in dataset
            or "WZ" in dataset
        )

        # gen-weights
        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)

        # selection and cutflow
        selection = PackedSelection()
        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection_args = (selection, cutflow, isData, gen_weights)

        # JEC factory loader
        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################

        print("starting object selection", f"{time.time() - start:.2f}")

        # Leptons
        num_leptons = 2
        # electrons, etrigvars = objects.good_electrons(events, events.Electron, year)
        electrons_corr, electron_shifted_vars = get_electron_scale_smearing(events, events.Electron, year, isData)
        # Keep the original NanoAOD Electron index before good_electrons().
        # Selection and saved kinematics use corrected electrons, while EGM SFs are evaluated on the matched raw electrons.
        # rawIdx is only a temporary in-memory bridge and is not written to the skim.
        electrons_corr = ak.with_field(electrons_corr, ak.local_index(events.Electron.pt, axis=1), "rawIdx")
        electrons, etrigvars = objects.good_electrons(events, electrons_corr, year)
        electrons_raw_for_sf = events.Electron[electrons.rawIdx]

        # TODO: electron_shifted_vars contains ElectronScale/Smear up/down variations.
        # They are computed here but not propagated to the selected electrons, CA mass, MET, or skim output yet.

        electron_reco_vars = {}
        electron_id_vars = {}
        electron_trigger_vars = {}

        if not isData:
            electron_reco_sfs = get_electron_reco_sfs(electrons_raw_for_sf, year)
            electron_id_sfs = get_electron_id_sfs(electrons_raw_for_sf, year)
            electron_trigger_sfs = get_electron_trigger_sfs(electrons_raw_for_sf, year)

            electron_reco_vars = {
                "ElectronRecoSF": pad_val(electron_reco_sfs["nom"], num_leptons, 1.0, axis=1),
                "ElectronRecoEventSF": ak.to_numpy(ak.prod(electron_reco_sfs["nom"], axis=1)),
            }
            if self._systematics:
                electron_reco_vars.update(
                    {
                        "ElectronRecoSFUp": pad_val(electron_reco_sfs["up"], num_leptons, 1.0, axis=1),
                        "ElectronRecoSFDown": pad_val(electron_reco_sfs["down"], num_leptons, 1.0, axis=1),
                        "ElectronRecoEventSFUp": ak.to_numpy(ak.prod(electron_reco_sfs["up"], axis=1)),
                        "ElectronRecoEventSFDown": ak.to_numpy(ak.prod(electron_reco_sfs["down"], axis=1)),
                    }
                )

            electron_id_vars = {
                "ElectronWP90NoIsoSF": pad_val(electron_id_sfs["nom"], num_leptons, 1.0, axis=1),
                "ElectronWP90NoIsoEventSF": ak.to_numpy(ak.prod(electron_id_sfs["nom"], axis=1)),
            }
            if self._systematics:
                electron_id_vars.update(
                    {
                        "ElectronWP90NoIsoSFUp": pad_val(electron_id_sfs["up"], num_leptons, 1.0, axis=1),
                        "ElectronWP90NoIsoSFDown": pad_val(electron_id_sfs["down"], num_leptons, 1.0, axis=1),
                        "ElectronWP90NoIsoEventSFUp": ak.to_numpy(ak.prod(electron_id_sfs["up"], axis=1)),
                        "ElectronWP90NoIsoEventSFDown": ak.to_numpy(ak.prod(electron_id_sfs["down"], axis=1)),
                    }
                )

            electron_trigger_vars = {
                "ElectronTrigEle30SF": pad_val(electron_trigger_sfs["nom"], num_leptons, 1.0, axis=1),
            }
            if self._systematics:
                electron_trigger_vars.update(
                    {
                        "ElectronTrigEle30SFUp": pad_val(electron_trigger_sfs["up"], num_leptons, 1.0, axis=1),
                        "ElectronTrigEle30SFDown": pad_val(electron_trigger_sfs["down"], num_leptons, 1.0, axis=1),
                    }
                )

        # ============================================================
        # Electron correction checks
        # ============================================================
        debug_electron_corrections = False

        if debug_electron_corrections:
            print(f"\n========== Electron check: {year} ==========")
            print("isData:", isData)

            # Scale/smearing arrays correspond to the original Electron order
            raw_pt_flat = ak.flatten(events.Electron.pt)
            corrected_pt_flat = ak.flatten(electrons_corr.pt)
            electron_check_mask = (raw_pt_flat > 20.0) & (raw_pt_flat < 200.0)

            print("\n--- Scale / smearing inputs ---")
            print("raw pt:", ak.to_list(raw_pt_flat[electron_check_mask][:10]))
            print("eta:", ak.to_list(ak.flatten(events.Electron.eta)[electron_check_mask][:10]))
            print("ScEta:", ak.to_list(ak.flatten(events.Electron.eta + events.Electron.deltaEtaSC)[electron_check_mask][:10]))
            print("r9:", ak.to_list(ak.flatten(events.Electron.r9)[electron_check_mask][:10]))
            print("seedGain:", ak.to_list(ak.flatten(events.Electron.seedGain)[electron_check_mask][:10]))

            if isData:
                print("event run:", ak.to_list(events.run[:10]))

            print("\n--- Scale / smearing outputs ---")
            print("corrected pt:", ak.to_list(corrected_pt_flat[electron_check_mask][:10]))
            print("corrected/raw:", ak.to_list((corrected_pt_flat[electron_check_mask] / raw_pt_flat[electron_check_mask])[:10]))

            if not isData and electron_shifted_vars is not None:
                print("scale up pt:", ak.to_list(ak.flatten(electron_shifted_vars["pt"]["ElectronScale_up"])[electron_check_mask][:10]))
                print("scale down pt:", ak.to_list(ak.flatten(electron_shifted_vars["pt"]["ElectronScale_down"])[electron_check_mask][:10]))
                print("smear up pt:", ak.to_list(ak.flatten(electron_shifted_vars["pt"]["ElectronSmear_up"])[electron_check_mask][:10]))
                print("smear down pt:", ak.to_list(ak.flatten(electron_shifted_vars["pt"]["ElectronSmear_down"])[electron_check_mask][:10]))

            # SF arrays correspond to electrons after good_electrons()
            print("\n--- Selected electrons ---")
            print("selected pt:", ak.to_list(ak.flatten(electrons.pt)[:10]))
            print("selected eta:", ak.to_list(ak.flatten(electrons.eta)[:10]))
            print("selected phi:", ak.to_list(ak.flatten(electrons.phi)[:10]))
            print("number per event:", ak.to_list(ak.num(electrons, axis=1)[:10]))

            if not isData:
                print("\n--- Electron Reco SF ---")
                print("nominal:", ak.to_list(ak.flatten(electron_reco_sfs["nom"])[:10]))
                print("up:", ak.to_list(ak.flatten(electron_reco_sfs["up"])[:10]))
                print("down:", ak.to_list(ak.flatten(electron_reco_sfs["down"])[:10]))
                print("event nominal:", ak.to_list(ak.prod(electron_reco_sfs["nom"], axis=1)[:10]))
                print("event up:", ak.to_list(ak.prod(electron_reco_sfs["up"], axis=1)[:10]))
                print("event down:", ak.to_list(ak.prod(electron_reco_sfs["down"], axis=1)[:10]))

                print("\n--- Electron WP90NoIso ID SF ---")
                print("nominal:", ak.to_list(ak.flatten(electron_id_sfs["nom"])[:10]))
                print("up:", ak.to_list(ak.flatten(electron_id_sfs["up"])[:10]))
                print("down:", ak.to_list(ak.flatten(electron_id_sfs["down"])[:10]))
                print("event nominal:", ak.to_list(ak.prod(electron_id_sfs["nom"], axis=1)[:10]))
                print("event up:", ak.to_list(ak.prod(electron_id_sfs["up"], axis=1)[:10]))
                print("event down:", ak.to_list(ak.prod(electron_id_sfs["down"], axis=1)[:10]))

                print("\n--- Electron Ele30 Tight trigger SF ---")
                print("nominal:", ak.to_list(ak.flatten(electron_trigger_sfs["nom"])[:10]))
                print("up:", ak.to_list(ak.flatten(electron_trigger_sfs["up"])[:10]))
                print("down:", ak.to_list(ak.flatten(electron_trigger_sfs["down"])[:10]))
                print("event nominal:", ak.to_list(ak.prod(electron_trigger_sfs["nom"], axis=1)[:10]))
                print("event up:", ak.to_list(ak.prod(electron_trigger_sfs["up"], axis=1)[:10]))
                print("event down:", ak.to_list(ak.prod(electron_trigger_sfs["down"], axis=1)[:10]))

            print("============================================\n") 
        # ============================================================
        # Electron correction checks ending
        # ============================================================ 

        # muons, mtrigvars = objects.good_muons(events, events.Muon, year)
        muons_corr, muon_shifted_vars = get_muon_scale_smearing(events, events.Muon, year, isData)
        # Keep the original NanoAOD Muon index before good_muons().
        # Selection and saved kinematics use corrected muons, while MUO SFs are evaluated on the matched raw muons.
        # rawIdx is only a temporary in-memory bridge and is not written to the skim.
        muons_corr = ak.with_field(muons_corr, ak.local_index(events.Muon.pt, axis=1), "rawIdx")
        muons, mtrigvars = objects.good_muons(events, muons_corr, year)
        muons_raw_for_sf = events.Muon[muons.rawIdx]

        # TODO: muon_shifted_vars contains MuonScale/Reso up/down variations.
        # They are computed here but not propagated to the selected muons, CA mass, MET, or skim output yet.

        muon_id_vars = {}
        muon_trigger_vars = {}

        if not isData:
            muon_id_sfs = get_muon_id_sfs(muons_raw_for_sf, year)
            muon_trigger_sfs = get_muon_trigger_sfs(muons_raw_for_sf, year)

            muon_id_vars = {
                "MuonTightIDSF": pad_val(muon_id_sfs["nom"], num_leptons, 1.0, axis=1),
                "MuonTightIDEventSF": ak.to_numpy(ak.prod(muon_id_sfs["nom"], axis=1)),
            }
            if self._systematics:
                muon_id_vars.update(
                    {
                        "MuonTightIDSFUp": pad_val(muon_id_sfs["up"], num_leptons, 1.0, axis=1),
                        "MuonTightIDSFDown": pad_val(muon_id_sfs["down"], num_leptons, 1.0, axis=1),
                        "MuonTightIDEventSFUp": ak.to_numpy(ak.prod(muon_id_sfs["up"], axis=1)),
                        "MuonTightIDEventSFDown": ak.to_numpy(ak.prod(muon_id_sfs["down"], axis=1)),
                    }
                )

            for trigger, sf in muon_trigger_sfs.items():
                muon_trigger_vars[f"MuonTrig_{trigger}"] = pad_val(sf["nom"], num_leptons, 1.0, axis=1)
                if self._systematics:
                    muon_trigger_vars[f"MuonTrig_{trigger}Up"] = pad_val(sf["up"], num_leptons, 1.0, axis=1)
                    muon_trigger_vars[f"MuonTrig_{trigger}Down"] = pad_val(sf["down"], num_leptons, 1.0, axis=1)
        
        # ============================================================
        # Muon correction checks
        # ============================================================
        debug_muon_corrections = False

        if debug_muon_corrections:
            print(f"\n========== Muon check: {year} ==========")
            print("isData:", isData)

            # Scale/smearing arrays correspond to the original Muon order
            raw_pt_flat = ak.flatten(events.Muon.pt, axis=1)
            raw_eta_flat = ak.flatten(events.Muon.eta, axis=1)
            raw_phi_flat = ak.flatten(events.Muon.phi, axis=1)
            raw_charge_flat = ak.flatten(events.Muon.charge, axis=1)
            raw_nlayers_flat = ak.flatten(events.Muon.nTrackerLayers, axis=1)
            corrected_pt_flat = ak.flatten(muons_corr.pt, axis=1)

            muon_check_mask = (raw_pt_flat > 26.0) & (raw_pt_flat < 200.0)

            print("\n--- Scale / smearing inputs ---")
            print("raw pt:", ak.to_list(raw_pt_flat[muon_check_mask][:10]))
            print("eta (signed):", ak.to_list(raw_eta_flat[muon_check_mask][:10]))
            print("abs(eta):", ak.to_list(abs(raw_eta_flat)[muon_check_mask][:10]))
            print("phi:", ak.to_list(raw_phi_flat[muon_check_mask][:10]))
            print("charge:", ak.to_list(raw_charge_flat[muon_check_mask][:10]))
            print("nTrackerLayers:", ak.to_list(raw_nlayers_flat[muon_check_mask][:10]))
            print("raw number per event:", ak.to_list(ak.num(events.Muon, axis=1)[:10]))
            print("corrected number per event:", ak.to_list(ak.num(muons_corr, axis=1)[:10]))
            print("event number:", ak.to_list(events.event[:10]))
            print("luminosity block:", ak.to_list(events.luminosityBlock[:10]))

            print("\n--- Scale / smearing outputs ---")
            print("nominal corrected pt:", ak.to_list(corrected_pt_flat[muon_check_mask][:10]))
            print("corrected/raw:", ak.to_list((corrected_pt_flat[muon_check_mask] / raw_pt_flat[muon_check_mask])[:10]))

            if not isData and muon_shifted_vars is not None:
                muon_scale_up_flat = ak.flatten(muon_shifted_vars["pt"]["MuonScale_up"], axis=1)
                muon_scale_down_flat = ak.flatten(muon_shifted_vars["pt"]["MuonScale_down"], axis=1)
                muon_reso_up_flat = ak.flatten(muon_shifted_vars["pt"]["MuonReso_up"], axis=1)
                muon_reso_down_flat = ak.flatten(muon_shifted_vars["pt"]["MuonReso_down"], axis=1)

                print("scale up pt:", ak.to_list(muon_scale_up_flat[muon_check_mask][:10]))
                print("scale down pt:", ak.to_list(muon_scale_down_flat[muon_check_mask][:10]))
                print("resolution up pt:", ak.to_list(muon_reso_up_flat[muon_check_mask][:10]))
                print("resolution down pt:", ak.to_list(muon_reso_down_flat[muon_check_mask][:10]))
                print("scale up/nominal:", ak.to_list((muon_scale_up_flat[muon_check_mask] / corrected_pt_flat[muon_check_mask])[:10]))
                print("scale down/nominal:", ak.to_list((muon_scale_down_flat[muon_check_mask] / corrected_pt_flat[muon_check_mask])[:10]))
                print("resolution up/nominal:", ak.to_list((muon_reso_up_flat[muon_check_mask] / corrected_pt_flat[muon_check_mask])[:10]))
                print("resolution down/nominal:", ak.to_list((muon_reso_down_flat[muon_check_mask] / corrected_pt_flat[muon_check_mask])[:10]))

            # SF arrays correspond to muons after good_muons()
            selected_pt_flat = ak.flatten(muons.pt, axis=1)
            selected_eta_flat = ak.flatten(muons.eta, axis=1)
            selected_phi_flat = ak.flatten(muons.phi, axis=1)
            selected_charge_flat = ak.flatten(muons.charge, axis=1)

            print("\n--- Selected muons ---")
            print("selected pt:", ak.to_list(selected_pt_flat[:10]))
            print("selected eta (signed):", ak.to_list(selected_eta_flat[:10]))
            print("selected abs(eta):", ak.to_list(abs(selected_eta_flat)[:10]))
            print("selected phi:", ak.to_list(selected_phi_flat[:10]))
            print("selected charge:", ak.to_list(selected_charge_flat[:10]))
            print("selected number per event:", ak.to_list(ak.num(muons, axis=1)[:10]))

            if not isData and year in ["2022", "2022EE", "2023", "2023BPix", "2024"]:
                print("\n--- Muon TightID SF ---")
                print("nominal:", ak.to_list(ak.flatten(muon_id_sfs["nom"], axis=1)[:10]))
                print("up:", ak.to_list(ak.flatten(muon_id_sfs["up"], axis=1)[:10]))
                print("down:", ak.to_list(ak.flatten(muon_id_sfs["down"], axis=1)[:10]))
                print("SF number per event:", ak.to_list(ak.num(muon_id_sfs["nom"], axis=1)[:10]))
                print("event nominal:", ak.to_list(ak.prod(muon_id_sfs["nom"], axis=1)[:10]))
                print("event up:", ak.to_list(ak.prod(muon_id_sfs["up"], axis=1)[:10]))
                print("event down:", ak.to_list(ak.prod(muon_id_sfs["down"], axis=1)[:10]))

                print("\n--- Muon trigger SFs ---")

                if len(muon_trigger_sfs) == 0:
                    print("No trigger SF corrections found.")

                for trigger_name, trigger_sf in muon_trigger_sfs.items():
                    print(f"\nTrigger: {trigger_name}")
                    print("nominal:", ak.to_list(ak.flatten(trigger_sf["nom"], axis=1)[:10]))
                    print("up:", ak.to_list(ak.flatten(trigger_sf["up"], axis=1)[:10]))
                    print("down:", ak.to_list(ak.flatten(trigger_sf["down"], axis=1)[:10]))
                    print("SF number per event:", ak.to_list(ak.num(trigger_sf["nom"], axis=1)[:10]))
                    print("diagnostic event nominal:", ak.to_list(ak.prod(trigger_sf["nom"], axis=1)[:10]))
                    print("diagnostic event up:", ak.to_list(ak.prod(trigger_sf["up"], axis=1)[:10]))
                    print("diagnostic event down:", ak.to_list(ak.prod(trigger_sf["down"], axis=1)[:10]))

            print("========================================\n")

        # ============================================================
        # Muon correction checks ending
        # ============================================================

        taus_corrected, tes_shifted_vars = get_tau_tes(events.Tau, year, isData=isData)
        # Keep the original NanoAOD Tau index before good_taus().
        # Selection and CA kinematics use TES-corrected taus, while Tau ID/trigger SFs are evaluated on the matched raw taus.
        # rawIdx is only a temporary in-memory bridge and is not written to the skim.
        taus_corrected = ak.with_field(taus_corrected, ak.local_index(events.Tau.pt, axis=1), "rawIdx")
        taus, ttrigvars = objects.good_taus(events, taus_corrected, year)
        taus_raw_for_sf = events.Tau[taus.rawIdx]

        # TODO: tes_shifted_vars contains Tau TES up/down variations.
        # They are computed here but not propagated to the selected taus, CA mass, MET, or skim output yet.

        # print(f"\n========== Tau check: {year} ==========")

        # # Raw Tau
        # print("Raw Tau pt       :", ak.to_list(ak.flatten(events.Tau.pt)[:10]))
        # print("Raw Tau mass     :", ak.to_list(ak.flatten(events.Tau.mass)[:10]))
        # print("Raw Tau eta      :", ak.to_list(ak.flatten(events.Tau.eta)[:10]))
        # print("Raw Tau DM       :", ak.to_list(ak.flatten(events.Tau.decayMode)[:10]))
        # if not isData:
        #     print("Raw Tau genmatch :", ak.to_list(ak.flatten(events.Tau.genPartFlav)[:10]))

        # # TES nominal
        # print("TES nominal pt   :", ak.to_list(ak.flatten(taus_corrected.pt)[:10]))
        # print("TES nominal mass :", ak.to_list(ak.flatten(taus_corrected.mass)[:10]))

        # # TES up/down
        # if not isData and tes_shifted_vars is not None:
        #     print("TES up pt        :", ak.to_list(ak.flatten(tes_shifted_vars["pt"]["TES_up"])[:10]))
        #     print("TES down pt      :", ak.to_list(ak.flatten(tes_shifted_vars["pt"]["TES_down"])[:10]))
        #     print("TES up mass      :", ak.to_list(ak.flatten(tes_shifted_vars["mass"]["TES_up"])[:10]))
        #     print("TES down mass    :", ak.to_list(ak.flatten(tes_shifted_vars["mass"]["TES_down"])[:10]))

        # # Tau after good_taus selection
        # print("Selected Tau pt  :", ak.to_list(ak.flatten(taus.pt)[:10]))
        # print("Selected Tau mass:", ak.to_list(ak.flatten(taus.mass)[:10]))
        # print("Selected Tau eta :", ak.to_list(ak.flatten(taus.eta)[:10]))
        # print("Selected Tau DM  :", ak.to_list(ak.flatten(taus.decayMode)[:10]))
        # if not isData:
        #     print("Selected Tau gen :", ak.to_list(ak.flatten(taus.genPartFlav)[:10]))
        
        # Tau ID/Trigger SF
        tau_vsjet_sf = None
        tauSFVars = {}
        tauTriggerVars = {}

        if not isData:
            tau_vsjet_sf = get_tau_vsjet_sf(taus_raw_for_sf, year)

            tauSFVars = {
                "TauVSjetSF": pad_val(tau_vsjet_sf["nom"], num_leptons, 1.0, axis=1),
                "TauVSjetSFEvent": ak.to_numpy(ak.prod(tau_vsjet_sf["nom"], axis=1)),
            }
            if self._systematics:
                tauSFVars.update(
                    {
                        "TauVSjetSFUp": pad_val(tau_vsjet_sf["up"], num_leptons, 1.0, axis=1),
                        "TauVSjetSFDown": pad_val(tau_vsjet_sf["down"], num_leptons, 1.0, axis=1),
                        "TauVSjetSFEventUp": ak.to_numpy(ak.prod(tau_vsjet_sf["up"], axis=1)),
                        "TauVSjetSFEventDown": ak.to_numpy(ak.prod(tau_vsjet_sf["down"], axis=1)),
                    }
                )

            # print("\n--- Tau VSjet SF ---")
            # print("VSjet nominal    :", ak.to_list(ak.flatten(tau_vsjet_sf["nom"])[:10]))
            # print("VSjet up         :", ak.to_list(ak.flatten(tau_vsjet_sf["up"])[:10]))
            # print("VSjet down       :", ak.to_list(ak.flatten(tau_vsjet_sf["down"])[:10]))
            # print("VSjet Event nom  :", tauSFVars["TauVSjetSFEvent"][:10])
            # print("VSjet Event up   :", tauSFVars["TauVSjetSFEventUp"][:10])
            # print("VSjet Event down :", tauSFVars["TauVSjetSFEventDown"][:10])


            if year == "2024":
                tau_trig_types = ["ditau", "etau", "mutau", "ditaujet", "vbfditau", "vbfsingletau"]
            else:
                tau_trig_types = ["ditau", "etau", "mutau", "ditaujet", "vbftau", "vbfditau"]

            for trigtype in tau_trig_types:
                trig_sf = get_tau_trigger_sf(taus_raw_for_sf, year, trigtype)

                name = {
                    "ditau": "DiTau",
                    "etau": "ETau",
                    "mutau": "MuTau",
                    "ditaujet": "DiTauJet",
                    "vbftau": "VBFTau",
                    "vbfditau": "VBFDiTau",
                    "vbfsingletau": "VBFSingleTau",
                }[trigtype]

                # print(f"{name} nominal :", ak.to_list(ak.flatten(trig_sf["nom"])[:10]))
                # print(f"{name} up      :", ak.to_list(ak.flatten(trig_sf["up"])[:10]))
                # print(f"{name} down    :", ak.to_list(ak.flatten(trig_sf["down"])[:10]))
                # print("========================================\n")

                tauTriggerVars[f"TauTrig{name}SF"] = pad_val(trig_sf["nom"], num_leptons, 1.0, axis=1)
                if self._systematics:
                    tauTriggerVars[f"TauTrig{name}SFUp"] = pad_val(trig_sf["up"], num_leptons, 1.0, axis=1)
                    tauTriggerVars[f"TauTrig{name}SFDown"] = pad_val(trig_sf["down"], num_leptons, 1.0, axis=1)

        # taus, ttrigvars = objects.good_taus(events, events.Tau, year)

        boostedtaus = objects.good_boostedtaus(events, events.boostedTau)

        # SubJets
        num_subjets = 3
        subjets = events.SubJet

        # These are bools saying if the lepton is matched to a trigger object or not
        trigMatchVars = {**etrigvars, **mtrigvars, **ttrigvars}
        for key, val in trigMatchVars.items():
            trigMatchVars[key] = pad_val(val, num_leptons, False, axis=1).astype(int)

        print("Leptons", f"{time.time() - start:.2f}")

        if self._systematics and not isData:
            # TODO: lepton/TES kinematic variations tes_shifted_vars are intentionally not written here yet.
            # The nominal-corrected collections above are already used by object selection,
            # CA mass, and saved nominal kinematics. The corresponding up/down branches
            # should be added only after a consistent policy for shifted selection, MET,
            # and CA-mass propagation is fixed.
            pass

        # AK4 Jets
        num_ak4_jets = 4
        jets, jec_shifted_jetvars = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=utils.jecs if self._systematics and not isData else None,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        # TODO: jec_shifted_jetvars contains AK4 JES/JER up/down variations when save_systematics=True.
        # They are computed by the JEC/JER helper but not propagated to selections, derived variables, or skim output yet.

        # # ============================================================
        # # AK4 correction checks
        # # ============================================================
        # print("AK4 pt before JEC:", ak.to_list(events.Jet.pt[:2, :4]))
        # print("AK4 pt after  JEC:", ak.to_list(jets.pt[:2, :4]))
        # print("AK4 pt ratio:", ak.to_list((jets.pt / events.Jet.pt)[:2, :4]))
        # print("AK4 JEC shift keys:", jec_shifted_jetvars["pt"].keys() if jec_shifted_jetvars else None)
        # print("AK4 JES up pt:", ak.to_list(jec_shifted_jetvars["pt"]["JES_up"][:2, :4]))
        # print("AK4 JER up pt:", ak.to_list(jec_shifted_jetvars["pt"]["JER_up"][:2, :4]))


        # For NanoAOD v15, use PFMET directly.
        # The current CorrectedMETFactory expects MetUnclustEnUpDeltaX/Y branches, which are not available in PFMET.
        # MET/JEC systematic propagation to CA mass is left as a future TODO.
        if JEC_loader.met_factory is not None:
            if self._nano_version == "v15":
                # met = JEC_loader.met_factory.build(events.PFMET, jets, {}) if isData else events.PFMET
                met = events.PFMET
            else:
                met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            if self._nano_version == "v15":
                met = events.PFMET
            else:
                met = events.MET

        print("ak4 JECs", f"{time.time() - start:.2f}")

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        # btag_sfs = get_btag_sfs(jets[:, :num_ak4_jets], year) if not isData else None
        # if btag_sfs is not None:
        #     btagSFVars = {
        #         "ak4JetBTagSF": pad_val(btag_sfs["nom"], num_ak4_jets, 1.0, axis=1),
        #         "BTagEventSF": ak.to_numpy(ak.prod(btag_sfs["nom"], axis=1)),
        #     }
        # else:
        #     btagSFVars = {}

        # # if btag_sfs is not None and self._systematics:
        # if btag_sfs is not None:
        #     for source in ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2", "cferr1", "cferr2"]:
        #         btagSFVars[f"BTagEventSF_{source}Up"] = ak.to_numpy(ak.prod(btag_sfs[f"{source}_up"], axis=1))
        #         btagSFVars[f"BTagEventSF_{source}Down"] = ak.to_numpy(ak.prod(btag_sfs[f"{source}_down"], axis=1))

        #         # print(btag_sfs.keys())
        #         # print(btag_sfs["nom"])
        #         # print(btag_sfs["hf_up"])

        # AK8 Jets
        num_ak8_jets = 3
        # Added nano_version=self._nano_version for v12/v15
        fatjets = objects.get_ak8jets(events.FatJet, nano_version=self._nano_version)  # this adds all our extra variables e.g. TXbb
        fatjets_before_jec = fatjets
        fatjets, jec_shifted_fatjetvars = JEC_loader.get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            jecs=utils.jecs if self._systematics and not isData else None,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        # TODO: jec_shifted_fatjetvars contains AK8 JES/JER up/down variations when save_systematics=True.
        # They are computed by the JEC/JER helper but not propagated to selections, CA mass, or skim output yet.

        # # ============================================================
        # # AK8 correction checks
        # # ============================================================
        # print("AK8 pt before JEC:", ak.to_list(fatjets_before_jec.pt[:20, :3]))
        # print("AK8 pt after  JEC:", ak.to_list(fatjets.pt[:20, :3]))
        # print("AK8 pt ratio:", ak.to_list((fatjets.pt / fatjets_before_jec.pt)[:20, :3]))

        print("ak8 JECs", f"{time.time() - start:.2f}")

        fatjets = objects.good_ak8jets(
            fatjets, **self.fatjet_selection, nano_version=self._nano_version
        )

        # VBF objects
        vbf_jets = objects.vbf_jets(
            jets,
            fatjets[:, :2],
            events,
            **self.vbf_jet_selection,
            **self.vbf_veto_lepton_selection,
            electrons=electrons_corr,
            muons=muons_corr,
        )

        # # AK4 objects away from first two fatjets
        ak4_jets_awayfromak8 = objects.ak4_jets_awayfromak8(
            jets,
            fatjets[:, :2],
            events,
            **self.ak4_bjet_selection,
            **self.ak4_bjet_lepton_selection,
            electrons=electrons_corr,
            muons=muons_corr,
            sort_by="nearest",
        )

        # # JMSR
        # # TODO: add variations per variable
        # bb_jmsr_shifted_vars = get_jmsr(
        #     fatjets_xbb,
        #     2,
        #     jmsr_vars=self.jmsr_vars,
        #     jms_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
        #     jmr_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
        #     isData=isData,
        # )

        # fatjets = objects.get_CA_MASS(fatjets, boostedtaus, met, subjets, muons, electrons)
        fatjets = objects.get_CA_MASS(fatjets, taus, met, subjets, muons, electrons)
        print("CA mass", f"{time.time() - start:.2f}")

        #########################
        # Save / derive variables
        #########################

        # Gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, fatjets, selection_args)
                genVars = {**genVars, **vars_dict}

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )
        logging.info(f"Passing gen selection: {np.sum(gen_selected)} / {len(events)}")

        # Lepton variables
        electronVars = {
            f"Electron{key}": pad_val(electrons[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Lepton"].items()
        }
        muonVars = {
            f"Muon{key}": pad_val(muons[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Lepton"].items()
        }
        tauVars = {
            f"Tau{key}": pad_val(taus[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Tau"].items()
        }
        boostedtauVars = {
            f"BoostedTau{key}": pad_val(boostedtaus[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["BoostedTau"].items()
        }
        # leptonVars = {**electronVars, **electron_reco_vars, **electron_id_vars, **muonVars, **muon_trigger_vars, **muon_id_vars, **tauVars, **boostedtauVars}
        leptonVars = {
            **electronVars,
            **electron_reco_vars,
            **electron_id_vars,
            **electron_trigger_vars,
            **muonVars,
            **muon_trigger_vars,
            **muon_id_vars,
            **tauVars,
            **tauSFVars,
            **tauTriggerVars,
            **boostedtauVars,
        }

        # Subjets
        subjetVars = {
            f"SubJet{key}": pad_val(subjets[var], num_subjets, axis=1)
            for (var, key) in self.skim_vars["SubJet"].items()
        }

        # AK4 Jet variables
        jet_skimvars = self.skim_vars["Jet"]
        if not isData:
            jet_skimvars = {
                **jet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_ak4_jets, axis=1)
            for (var, key) in jet_skimvars.items()
        }

        if len(ak4_jets_awayfromak8) == 2:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(
                    ak.concatenate(
                        [ak4_jets_awayfromak8[0][var], ak4_jets_awayfromak8[1][var]], axis=1
                    ),
                    2,
                    axis=1,
                )
                for (var, key) in jet_skimvars.items()
            }
        else:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(ak4_jets_awayfromak8[var], 2, axis=1)
                for (var, key) in jet_skimvars.items()
            }

        # AK8 Jet variables
        fatjet_skimvars = self.skim_vars["FatJet"]
        if not isData:
            fatjet_skimvars = {
                **fatjet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_ak8_jets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }
        print("Jet vars", f"{time.time() - start:.2f}")

        # # JEC and JMSR
        # if self._region == "signal" and isJECs:
        #     # Jet JEC variables
        #     for var in ["pt"]:
        #         key = self.skim_vars["Jet"][var]
        #         for shift, vals in jec_shifted_jetvars[var].items():
        #             if shift != "":
        #                 ak4JetVars[f"ak4Jet{key}_{shift}"] = pad_val(vals, num_ak4_jets, axis=1)

        #     # FatJet JEC variables
        #     for var in ["pt"]:
        #         key = self.skim_vars["FatJet"][var]
        #         for shift, vals in jec_shifted_bbfatjetvars[var].items():
        #             if shift != "":
        #                 bbFatJetVars[f"bbFatJet{key}_{shift}"] = pad_val(vals, 2, axis=1)

        #     # FatJet JMSR
        #     for var in self.jmsr_vars:
        #         key = fatjet_skimvars[var]
        #         bbFatJetVars[f"bbFatJet{key}_raw"] = bbFatJetVars[f"bbFatJet{key}"]
        #         for shift, vals in bb_jmsr_shifted_vars[var].items():
        #             # overwrite saved mass vars with corrected ones
        #             label = "" if shift == "" else "_" + shift
        #             bbFatJetVars[f"bbFatJet{key}{label}"] = vals

        # MET
        metVars = {f"MET{key}": met[var].to_numpy() for (var, key) in self.skim_vars["MET"].items()}

        # Event variables
        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["ht"] = ht.to_numpy()
        eventVars["nElectrons"] = ak.num(electrons).to_numpy()
        eventVars["nMuons"] = ak.num(muons).to_numpy()
        eventVars["nTaus"] = ak.num(taus).to_numpy()
        eventVars["nBoostedTaus"] = ak.num(boostedtaus).to_numpy()
        eventVars["nJets"] = ak.num(jets).to_numpy()
        eventVars["nFatJets"] = ak.num(fatjets).to_numpy()
        eventVars["nSubJets"] = ak.num(subjets).to_numpy()

        # jin for CA
        # eventVars["CA_matched_tau_pt_sum"] = ca_tau_pt_sum.to_numpy()
        # eventVars["CA_tau_idx_0"] = ca_tau_indices[:, 0].to_numpy()
        # eventVars["CA_tau_idx_1"] = ca_tau_indices[:, 1].to_numpy()
        # eventVars["CA_best_fatjet_idx"] = ca_best_fatjet_idx.to_numpy()

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}
        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        HLTVars = {}
        zeros = np.zeros(len(events), dtype="int")
        for trigger in self.HLTs[year]:
            if trigger in events.HLT.fields:
                HLTVars[f"HLT_{trigger}"] = events.HLT[trigger].to_numpy().astype(int)
            else:
                logger.warning(f"Missing {trigger}!")
                HLTVars[f"HLT_{trigger}"] = zeros

        print("HLT vars", f"{time.time() - start:.2f}")

        # vbfJets
        vbfJetVars = {
            f"VBFJet{key}": pad_val(vbf_jets[var], 2, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # # JEC variations for VBF Jets
        # if self._region == "signal" and isJECs:
        #     for var in ["pt"]:
        #         key = self.skim_vars["Jet"][var]
        #         for label, shift in utils.jecs.items():
        #             if shift in ak.fields(vbf_jets):
        #                 for vari in ["up", "down"]:
        #                     vbfJetVars[f"VBFJet{key}_{label}_{vari}"] = pad_val(
        #                         vbf_jets[shift][vari][var], 2, axis=1
        #                     )

        skimmed_events = {
            **genVars,
            **eventVars,
            **pileupVars,
            **trigMatchVars,
            **HLTVars,
            **ak4JetAwayVars,
            **leptonVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **metVars,
            **subjetVars,
            # **bbFatJetVars,
            # **trigObjFatJetVars,
            **vbfJetVars,
            # **btagSFVars,
        }

        # if self._region == "signal":
        #     bdtVars = self.getBDT(bbFatJetVars, vbfJetVars, ak4JetAwayVars, met_pt, "")
        #     print(bdtVars)
        #     skimmed_events = {
        #         **skimmed_events,
        #         **bdtVars,
        #     }

        print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )

        # don't apply triggers for now, for trigger studies etc.
        apply_trigger = False
        if apply_trigger:
            add_selection("trigger", HLT_triggered, *selection_args)

        # metfilters
        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in utils.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)

        # jet veto maps
        cut_jetveto = get_jetveto_event(jets, year)
        add_selection("ak4_jetveto", cut_jetveto, *selection_args)

        # # >=2 AK8 jets passing selections
        # add_selection("ak8_numjets", (ak.num(fatjets) >= 2), *selection_args)

        # >=1 AK8 jets with pT cut (230 GeV by default)
        if self.fatjet_selection["pt"] >= 0:  # if < 0, don't apply any fatjet selection
            cut_pt = (
                np.sum(ak8FatJetVars["ak8FatJetPt"] >= self.fatjet_selection["pt"], axis=1) >= 1
            )
            add_selection("ak8_pt", cut_pt, *selection_args)

        # # >=1 AK8 jets with mSD >= 40 GeV
        # cut_mass = np.sum(ak8FatJetVars["ak8FatJetMsd"] >= 40, axis=1) >= 1
        # add_selection("ak8_mass", cut_mass, *selection_args)

        # Veto leptons
        # add_selection(
        #     "0lep",
        #     (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
        #     *selection_args,
        # )

        # if self._region == "signal":
        #     # >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
        #     cut_txbb = (
        #         np.sum(
        #             bbFatJetVars[f"bbFatJet{txbb_str}"] >= self.preselection[self.txbb],
        #             axis=1,
        #         )
        #         >= 1
        #     )
        #     add_selection("ak8bb_txbb0", cut_txbb, *selection_args)

        # VBF veto cut (not now)
        # add_selection("vbf_veto", ~(cut_vbf), *selection_args)

        if self._fatjet_bb_preselection:
            # at least 1 jet with ParTXbbvsQCDTop > 0.3
            cut_bb = (
                np.sum(
                    ak8FatJetVars["ak8FatJetParTXbbvsQCDTop"] >= self.preselection["glopart-v2"],
                    # ak8FatJetVars["ak8FatJetPNetXbbVsQCD"] >= self.preselection["pnet-v12"],
                    axis=1,
                )
                >= 1
            )
            add_selection("ak8_bb_preselection", cut_bb, *selection_args)

        if self._prescale_factor:
            cut_prescale = events.event % self._prescale_factor == 0
            add_selection("prescale", cut_prescale, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events,
                year,
                dataset,
                gen_weights,
                gen_selected,
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        ##############################
        # Reshape and apply selections
        ##############################

        sel_all = selection.all(*selection.names)
        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        dataframe = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(dataframe, fname)

        logger.info(f"Cutflow:\n{cutflow}")

        print("Return ", f"{time.time() - start:.2f}")
        print("Columns:", print(list(dataframe.columns)))
        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self,
        events,
        year,
        dataset,
        gen_weights,
        gen_selected,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        pileup_input = (
            events.Pileup.nPU.to_numpy()
            if "Pu60" in dataset or "Pu70" in dataset
            else events.Pileup.nTrueInt.to_numpy()
        )
        add_pileup_weight_update(weights, year, pileup_input, dataset)
        # add_pileup_weight(
        #     weights,
        #     year,
        #     events.Pileup.nPU.to_numpy(),
        #     dataset,
        # )
        add_ps_weight(weights, events.PSWeight)

        logger.debug("weights", extra=weights._weights.keys())

        ###################### Save all the weights and variations ######################

        # these weights should not change the overall normalization, so are saved separately
        norm_preserving_weights = hh_vars.norm_preserving_weights

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()

        # norm preserving weights, used to do normalization in post-processing
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np[gen_selected])

        if self._systematics:
            for systematic in list(weights.variations):
                weights_dict[f"weight_{systematic}"] = weights.weight(modifier=systematic)

                if utils.remove_variation_suffix(systematic) in norm_preserving_weights:
                    var_weight = weights.partial_weight(include=norm_preserving_weights)
                    # modify manually
                    if "Down" in systematic and systematic not in weights._modifiers:
                        var_weight = (
                            var_weight / weights._modifiers[systematic.replace("Down", "Up")]
                        )
                    else:
                        var_weight = var_weight * weights._modifiers[systematic]

                    # need to save total # events for each variation for normalization in post-processing
                    totals_dict[f"np_{systematic}"] = np.sum(var_weight[gen_selected])

        # TEMP: save each individual weight TODO: remove
        for key in weights._weights:
            weights_dict[f"single_weight_{key}"] = weights.partial_weight([key])

        ###################### alpha_S and PDF variations ######################

        if ("HHTobbbb" in dataset or "HHto4B" in dataset) or dataset.startswith("TTTo"):
            scale_weights = get_scale_weights(events)
            if scale_weights is not None:
                weights_dict["scale_weights"] = (
                    scale_weights * weights_dict["weight"][:, np.newaxis]
                )
                totals_dict["np_scale_weights"] = np.sum(
                    (scale_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
                )

        if "HHTobbbb" in dataset or "HHto4B" in dataset:
            pdf_weights = get_pdf_weights(events)
            weights_dict["pdf_weights"] = pdf_weights * weights_dict["weight"][:, np.newaxis]
            totals_dict["np_pdf_weights"] = np.sum(
                (pdf_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
            )

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict
