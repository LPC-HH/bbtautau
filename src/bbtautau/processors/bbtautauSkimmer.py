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

        # JMSR
        self.jmsr_vars = ["msoftdrop", "particleNet_mass_legacy", "ParTmassVis", "ParTmassRes"]

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

        #CA variables
        ca_vars = [
            "mass",
            "msoftdrop",
            "globalParT_massVisApplied",
            "globalParT_massResApplied",
            "particleNet_mass_legacy",
            "isDauTau",
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
        electrons, etrigvars = objects.good_electrons(events, events.Electron, year)
        muons, mtrigvars = objects.good_muons(events, events.Muon, year)
        taus, ttrigvars = objects.good_taus(events, events.Tau, year)
        boostedtaus = objects.good_boostedtaus(events, events.boostedTau)

        # SubJets
        num_subjets = 3
        subjets = events.SubJet

        # These are bools saying if the lepton is matched to a trigger object or not
        trigMatchVars = {**etrigvars, **mtrigvars, **ttrigvars}
        for key, val in trigMatchVars.items():
            trigMatchVars[key] = pad_val(val, num_leptons, False, axis=1).astype(int)

        print("Leptons", f"{time.time() - start:.2f}")

        # TODO: lepton systematics

        # AK4 Jets
        num_ak4_jets = 4
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
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        print("ak4 JECs", f"{time.time() - start:.2f}")

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        # AK8 Jets
        num_ak8_jets = 3
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

        # # TODO: VBF objects
        # vbf_jets = objects.vbf_jets(
        #     jets,
        #     fatjets_xbb[:, :2],
        #     events,
        #     **self.vbf_jet_selection,
        #     **self.vbf_veto_lepton_selection,
        # )

        # # AK4 objects away from first two fatjets
        # ak4_jets_awayfromak8 = objects.ak4_jets_awayfromak8(
        #     jets,
        #     fatjets_xbb[:, :2],
        #     events,
        #     **self.ak4_bjet_selection,
        #     **self.ak4_bjet_lepton_selection,
        #     sort_by="nearest",
        # )

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

        fatjets = objects.get_CA_MASS(fatjets, boostedtaus, met, subjets)
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
        leptonVars = {**electronVars, **muonVars, **tauVars, **boostedtauVars}

        #Subjets
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

        # if len(ak4_jets_awayfromak8) == 2:
        #     ak4JetAwayVars = {
        #         f"AK4JetAway{key}": pad_val(
        #             ak.concatenate(
        #                 [ak4_jets_awayfromak8[0][var], ak4_jets_awayfromak8[1][var]], axis=1
        #             ),
        #             2,
        #             axis=1,
        #         )
        #         for (var, key) in jet_skimvars.items()
        #     }
        # else:
        #     ak4JetAwayVars = {
        #         f"AK4JetAway{key}": pad_val(ak4_jets_awayfromak8[var], 2, axis=1)
        #         for (var, key) in jet_skimvars.items()
        #     }

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

        #jin for CA
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
            **subjetVars,
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
            # at least 1 jet with ParTXbbvsQCD > 0.8
            cut_bb = (
                np.sum(
                    ak8FatJetVars["ak8FatJetParTXbbvsQCD"] >= self.preselection["glopart-v2"],
                    axis=1,
                )
                >= 1
            )
            add_selection("ak8_bb_preselection", cut_bb, *selection_args)

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
