"""
Skimmer for GloParTv3 tagger finetuning.
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
    add_pileup_weight,
    add_ps_weight,
    get_jetveto_event,
    get_pdf_weights,
    get_scale_weights,
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


def tagger_gen_matching(events, genparts, fatjets, gen_vars, label="bbtautau", match_dR=0.8):
    """Simple tagger gen matching function for tagger training"""
    # For tagger training, we typically want all events to pass gen matching
    # unless there are specific generator-level requirements
    matched_mask = ak.ones_like(fatjets.pt[:, 0], dtype=bool)

    # Create basic gen variables for tagger training
    genVars = {}
    for var_name in gen_vars:
        if var_name.startswith("fj_"):
            # Fill with zeros for now - can be extended based on specific needs
            genVars[var_name] = ak.zeros_like(fatjets.pt[:, 0])

    return matched_mask, genVars


class taggerSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    skim_vars = {  # noqa: RUF012
        "GenPart": [
            "fj_genjetmsd",
            "fj_genjetmass",
            "fj_genRes_pt",
            "fj_genRes_eta",
            "fj_genRes_phi",
            "fj_genRes_mass",
            "fj_genX_pt",
            "fj_genX_eta",
            "fj_genX_phi",
            "fj_genX_mass",
            "fj_H_gg",
            "fj_H_qq",
            "fj_H_bb",
            "fj_H_cc",
            "fj_QCDb",
            "fj_QCDbb",
            "fj_QCDc",
            "fj_QCDcc",
            "fj_QCDothers",
            "fj_V_2q",
            "fj_V_elenu",
            "fj_V_munu",
            "fj_V_taunu",
            "fj_Top_bmerged",
            "fj_Top_2q",
            "fj_Top_elenu",
            "fj_Top_munu",
            "fj_Top_hadtauvqq",
            "fj_Top_leptauelvnu",
            "fj_Top_leptaumuvnu",
        ],
        # formatted to match weaver's preprocess.json
        "MET": {
            "met_features": {
                "var_names": [
                    "met_relpt",
                    "met_relphi",
                ],
            },
            "met_points": {"var_length": 1},
        },
        "Lep": {
            "fj_features": {
                "fj_lep_dR",
                "fj_lep_pt",
                "fj_lep_iso",
                "fj_lep_miniiso",
            },
        },
        "Jet": {
            **P4,
            "rawFactor": "rawFactor",
            "btagPNetB": "btagPNetB",
        },
        "SubJet": {
            **P4,
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
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "t32": "Tau3OverTau2",
            "rawFactor": "rawFactor",
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
    }

    tagger_features = {  # noqa: RUF012
        "electron": [
            "charge",
            "convVeto",
            "deltaEtaSC",
            "dr03EcalRecHitSumEt",
            "dr03HcalDepth1TowerSumEt",
            "dr03TkSumPt",
            "dxy",
            "dxyErr",
            "dz",
            "dzErr",
            "eInvMinusPInv",
            "eta",
            "hoe",
            "ip3d",
            "lostHits",
            "phi",
            "pt",
            "r9",
            "sieie",
            "sip3d",
        ],
        "muon": [
            "muon_charge",
            "muon_dxy",
            "muon_dxyErr",
            "muon_dz",
            "muon_dzErr",
            "muon_eta",
            "muon_ip3d",
            "muon_nStations",
            "muon_nTrackerLayers",
            "muon_pfRelIso03_all",
            "muon_pfRelIso03_chg",
            "muon_phi",
            "muon_pt",
            "muon_segmentComp",
            "muon_sip3d",
            "muon_tkRelIso",
        ],
        "tau": [
            "tau_charge",
            "tau_chargedIso",
            "tau_eta",
            "tau_leadTkDeltaEta",
            "tau_leadTkDeltaPhi",
            "tau_leadTkPtOverTauPt",
            "tau_mass",
            "tau_neutralIso",
            "tau_phi",
            "tau_photonsOutsideSignalCone",
            "tau_pt",
            "tau_rawAntiEle2018",
            "tau_rawIso",
            "tau_rawIsodR03",
        ],
        "evt": [
            "jet_muonenergy",
            "jet_elecenergy",
            "jet_photonenergy",
            "jet_chhadronenergy",
            "jet_nehadronenergy",
            "jet_muonnum",
            "jet_elecnum",
            "jet_photonnum",
            "jet_chhadronnum",
            "jet_nehadronnum",
        ],
        "evt_z": [
            "jet_muonenergy",
            "jet_elecenergy",
            "jet_photonenergy",
            "jet_chhadronenergy",
            "jet_nehadronenergy",
            "jet_muonnum",
            "jet_elecnum",
            "jet_photonnum",
            "jet_chhadronnum",
            "jet_nehadronnum",
        ],
        "evt_reg": [
            "met_covXX",
            "met_covXY",
            "met_covYY",
            "met_dphi",
            "met_pt",
            "met_significance",
            "pupmet_pt",
            "pupmet_dphi",
            "jet_msd",
            "jet_pt",
            "jet_eta",
            "jet_phi",
        ],
    }

    def __init__(self, nano_version: str = "v14_25v2"):
        super().__init__()

        self._nano_version = nano_version
        self._accumulator = processor.dict_accumulator({})

        # Fatjet selection criteria
        self.fatjet_selection = {"pt": 230}  # minimum pT cut for fatjets
        self.fatjet_label = "FatJet"
        self.label = "bbtautau"
        self.match_dR = 0.8  # deltaR matching threshold for gen particles

        # HLT selection
        self.HLTs = {
            year: HLTs.hlt_list(year, hlt_prefix=False)
            for year in ["2022", "2022EE", "2023", "2023BPix"]
        }

        # Systematics flag
        self._systematics = False

        # particlenet legacy variables
        pnet_vars = [
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
            **{f"particleNetLegacy_{var}": f"PNet{var}Legacy" for var in pnet_vars},
        }

        glopart2_vars = [
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

        glopart3_vars = [
            "QCD",
            "TopbWev",
            "TopbWmv",
            "TopbWq",
            "TopbWqq",
            "TopbWtauhv",
            "Xbb",
            "Xcc",
            "Xqq",
            "massX2p",
            "massGeneric",
            "Xtauhtaue",
            "Xtauhtauh",
            "Xtauhtaum",
            # Derived variables
            "massX2pCorr",
            "massGenericCorr",
            "massX2pApplied",
            "massGenericApplied",
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

        glopart3_vars += [f"hidNeuron{i}" for i in range(256)]

        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"globalParT2_{var}": f"ParT2{var}" for var in glopart2_vars},
            **{f"globalParT3_{var}": f"ParT3{var}" for var in glopart3_vars},
        }

        # self.tagger_resources_path = Path(__file__).parent.resolve() / "tagger_resources"

        # with (self.tagger_resources_path / "pyg_ef_ul_cw_8_2_preprocess.json").open() as f:
        #     self.tagger_vars = json.load(f)

        logger.info(f"Running tagger skimmer with:\nnano version {self._nano_version}")

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
        isMC = not isData

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
        electrons, etrigvars = objects.good_electrons(events, events.Electron, year)
        muons, mtrigvars = objects.good_muons(events, events.Muon, year)
        taus, ttrigvars = objects.good_taus(events, events.Tau, year)
        boostedtaus = objects.good_boostedtaus(events, events.boostedTau)

        print("Leptons", f"{time.time() - start:.2f}")

        # jets and MET
        jets, jec_shifted_jetvars = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=utils.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        if JEC_loader.met_factory is not None:
            # check if "MET" attribute exists
            if hasattr(events, "MET"):
                events_met = events.MET
            elif hasattr(events, "PuppiMET"):
                events_met = events.PuppiMET
                # No deltaX and deltaY in PuppiMET, so we have to calculate them
                # by definition: up - nominal
                deltaX_up = events_met.ptUnclusteredUp * np.cos(events_met.phiUnclusteredUp)
                deltaY_up = events_met.ptUnclusteredUp * np.sin(events_met.phiUnclusteredUp)
                deltaX_nom = events_met.pt * np.cos(events_met.phi)
                deltaY_nom = events_met.pt * np.sin(events_met.phi)
                events_met["MetUnclustEnUpDeltaX"] = deltaX_up - deltaX_nom
                events_met["MetUnclustEnUpDeltaY"] = deltaY_up - deltaY_nom
            else:
                raise AttributeError("Neither 'MET' nor 'PuppiMET' attribute found in events.")

            met = JEC_loader.met_factory.build(events_met, jets, {}) if isData else events_met
        else:
            if hasattr(events, "MET"):
                met = events.MET
            elif hasattr(events, "PuppiMET"):
                met = events.PuppiMET
            else:
                raise AttributeError("Neither 'MET' nor 'PuppiMET' attribute found in events.")

        print("ak4 JECs", f"{time.time() - start:.2f}")

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        # AK8 Jets
        num_jets = 2  # TODO: 2 only for bbtautau (?)
        fatjets = objects.get_ak8jets(events.FatJet)  # this adds all our extra variables e.g. TXbb
        fatjets, jec_shifted_fatjetvars = JEC_loader.get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            jecs=utils.jecs,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print("ak8 JECs", f"{time.time() - start:.2f}")

        fatjets = objects.good_ak8jets(
            fatjets, **self.fatjet_selection, nano_version=self._nano_version
        )
        num_ak8_jets = 2  # number of fatjets to consider
        fatjets = ak.pad_none(fatjets, num_ak8_jets, axis=1)
        print("ak8", f"{time.time() - start:.2f}")

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
        leptonVars = {**electronVars, **muonVars, **tauVars, **boostedtauVars}

        # AK4 Jet variables
        ak4JetVars = {
            f"Jet{key}": pad_val(jets[var], 4, axis=1)  # Save up to 4 AK4 jets
            for (var, key) in self.skim_vars["Jet"].items()
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

        # MET - fix the MET variables extraction
        metVars = {}
        if hasattr(met, "pt"):
            metVars["METpt"] = met.pt.to_numpy()
        if hasattr(met, "phi"):
            metVars["METphi"] = met.phi.to_numpy()

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
        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}
        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        trigMatchVars = {
            **etrigvars,
            **mtrigvars,
            **ttrigvars,
        }  # Combine all trigger matching variables

        HLTVars = {}
        zeros = np.zeros(len(events), dtype="int")
        for trigger in self.HLTs[year]:
            if trigger in events.HLT.fields:
                HLTVars[f"HLT_{trigger}"] = events.HLT[trigger].to_numpy().astype(int)
            else:
                logger.warning(f"Missing {trigger}!")
                HLTVars[f"HLT_{trigger}"] = zeros

        print("HLT vars", f"{time.time() - start:.2f}")

        # # vbfJets
        # vbfJetVars = {
        #     f"VBFJet{key}": pad_val(vbf_jets[var], 2, axis=1)
        #     for (var, key) in self.skim_vars["Jet"].items()
        # }

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
            # **ak4JetAwayVars,
            **leptonVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **metVars,
            # **bbFatJetVars,
            # **trigObjFatJetVars,
            # **vbfJetVars,
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

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)
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
