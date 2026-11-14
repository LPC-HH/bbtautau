"""
Light muon + AK8 skimmer (semileptonic ttbar region) for

- the JMS/JMR measurement of our jet masses (W -> qq probe, as HH4b's ttSkimmer), and
- the PNetBB trigger-efficiency measurement (IsoMu24 reference trigger, AK8 probe).

Same structure and output format (parquet + pickles with totals/cutflow) as bbtautauSkimmer.
The preselection is loose on purpose (union of both measurements); the tighter cuts of each
(Mu50 pT > 55, mini-isolation, MET, leptonic-W pT, b-tag, Delta-phi, electron veto, IsoMu24
match, ...) are applied offline from the saved columns.

Preselection: MET filters, jet veto map, HLT_IsoMu24 OR HLT_Mu50, >= 1 tight muon with
pT > 26, >= 1 AK8 jet (tight ID, pT > 200, |eta| < 2.5) with dR(jet, leading muon) > 0.8.

Saved AK8 jets are the ones away from the leading muon (dR > 0.8), pt-ordered, so jet 0 is the
natural probe for both measurements.

Based on HH4b's ttSkimmer.py (Billy Li, Raghav Kansal, Cristina Suarez).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import awkward as ak
import numpy as np
from boostedhh import hh_vars
from boostedhh.processors import SkimmerABC, utils
from boostedhh.processors.corrections import (
    JECs,
    add_pileup_weight_update,
    get_jetveto_event,
    get_jmsr,
)
from boostedhh.processors.utils import (
    GEN_FLAGS,
    P4,
    PAD_VAL,
    PDGID,
    add_selection,
    pad_val,
)
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from bbtautau.HLTs import HLTs, years_all

from . import objects

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Muon TrigObj filter bit of HLT_Mu50: "10 => 1mu (Mu50)" in the TrigObj_filterBits doc of both
# NanoAOD v12 and v15 (checked on local v12 2022 data and v15 2024 MC files).
MU50_FILTERBIT = 10

# reference triggers (event preselection: OR)
REFERENCE_HLTS = ["IsoMu24", "Mu50"]
# HLT paths of the HH4b trigger "soup" (saved only, for comparisons)
HH4B_AK8_HLTS = ["AK8PFJet425_SoftDropMass40", "AK8PFJet400_SoftDropMass40"]


def gen_selection_top(events, fatjets: ak.Array, num_fatjets: int) -> dict:
    """Gen top / W information and quark matching of the saved fat jets (ttbar samples).

    Ported from HH4b's GenSelection.gen_selection_Top: per fat jet, number of quarks from each
    top's W within dR < 0.8 (NumQMatchedTop1/2) and whether each top's b is within dR < 0.8
    (NumBMatchedTop1/2); plus gen top and W kinematics, used for the W-matched category.
    """
    tops = events.GenPart[
        (abs(events.GenPart.pdgId) == PDGID.t) & events.GenPart.hasFlags(GEN_FLAGS)
    ]
    genvars = {f"GenTop{key}": pad_val(tops[var], 2, axis=1) for (var, key) in P4.items()}

    daughters = ak.flatten(tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
    daughters_pdgId = abs(daughters.pdgId)

    wbosons = daughters[daughters_pdgId == PDGID.W]
    for i in range(2):
        w = ak.firsts(wbosons[:, i : i + 1])
        for var, key in P4.items():
            genvars[f"GenTopW{i}{key}"] = ak.to_numpy(ak.fill_none(w[var], PAD_VAL))

    w_daughters = ak.flatten(wbosons.distinctChildren, axis=2)
    w_daughters = w_daughters[w_daughters.hasFlags(GEN_FLAGS)]
    bquarks = daughters[daughters_pdgId == PDGID.b]

    # 0- or 1-element lists (not options), so a missing quark simply gives "not matched"
    qs = [w_daughters[:, i : i + 1] for i in range(4)]
    bs = [bquarks[:, i : i + 1] for i in range(2)]

    def matched(q):
        return ak.values_astype(ak.any(fatjets.metric_table(q) < 0.8, axis=2), np.int32)

    fjvars = {
        "TopMatch": ak.any(fatjets.metric_table(tops) < 0.8, axis=2),
        "NumQMatchedTop1": matched(qs[0]) + matched(qs[1]),
        "NumQMatchedTop2": matched(qs[2]) + matched(qs[3]),
        "NumBMatchedTop1": matched(bs[0]),
        "NumBMatchedTop2": matched(bs[1]),
    }
    for var, vals in fjvars.items():
        genvars[f"ak8FatJet{var}"] = pad_val(vals, num_fatjets, axis=1)

    return genvars


def muon_trig_match(events, muons, hlt: str, filterbit: int, dr: float = 0.2) -> ak.Array:
    """Per muon: ``hlt`` fired and a muon TrigObj with ``filterbit`` within ``dr``."""
    trigobj = events.TrigObj[events.TrigObj.id == PDGID.mu]
    trigobj = trigobj[(trigobj.filterBits & (1 << filterbit)) != 0]
    fired = events.HLT[hlt] if hlt in events.HLT.fields else ak.zeros_like(events.run, dtype=bool)
    near = ak.any(muons.metric_table(trigobj) < dr, axis=2)
    return fired & near


class ttSkimmer(SkimmerABC):
    """Skims NanoAOD files into a light muon + AK8 selection (see module docstring)."""

    # name in nano files: name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "Muon": {
            **P4,
            "charge": "charge",
            "pfIsoId": "pfIsoId",
            "miniPFRelIso_all": "miniPFRelIso_all",
            "pfRelIso04_all": "pfRelIso04_all",
        },
        "Jet": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "btagPNetB": "btagPNetB",
            "btagDeepFlavB": "btagDeepFlavB",
        },
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "mass_raw": "MassRaw",
            "globalParT_massResApplied": "ParTmassResApplied",
            "globalParT_massVisApplied": "ParTmassVisApplied",
            "globalParT_Xbb": "ParTXbb",
            "globalParT_Xcc": "ParTXcc",
            "globalParT_Xcs": "ParTXcs",
            "globalParT_Xqq": "ParTXqq",
            "globalParT_QCD": "ParTQCD",
            "globalParT_Top": "ParTTop",
            "globalParT_XbbvsQCD": "ParTXbbvsQCD",
            "globalParT_XbbvsQCDTop": "ParTXbbvsQCDTop",
            "dr_muon": "dRMuon",
        },
        "TriggerObject": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "filterBits": "Bit",
        },
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
    }

    muon_selection = {"pt": 26, "eta": 2.4}  # noqa: RUF012  (tight ID, dz/dxy from good_muons)

    fatjet_selection = {  # noqa: RUF012
        "object_pt": 200,
        "pt": 200,
        "eta": 2.5,
        "msd": 0,
        "mreg": 0,
        "dr_muon": 0.8,
    }

    ak4_selection = {"pt": 25, "eta": 2.5}  # noqa: RUF012

    num_muons = 2
    num_ak4_jets = 4
    num_ak8_jets = 3
    num_jmsr_jets = 2  # JMS/JMR variations only for the first two saved AK8 jets
    num_trigobjs = 4

    # masses with +-10% JMS/JMR variations (nominal values = 1: no correction applied)
    jmsr_vars = [  # noqa: RUF012
        "msoftdrop",
        "globalParT_massResApplied",
        "globalParT_massVisApplied",
    ]

    def __init__(self, xsecs: dict = None, nano_version: str = "v12_private"):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb
        self._nano_version = nano_version
        self._accumulator = processor.dict_accumulator({})

        # HLT bits to save (without the "HLT_" prefix): reference triggers, the analysis PNetBB
        # path(s) of the year, and the HH4b AK8 jet paths
        self.HLTs = {
            year: REFERENCE_HLTS
            + HLTs.resolve_menu(year, "pnetbb", hlt_prefix=False)
            + HH4B_AK8_HLTS
            for year in years_all
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor"""

        start = time.time()
        logging.info(f"# events {len(events)}")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = not hasattr(events, "genWeight")

        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)

        selection = PackedSelection()
        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection_args = (selection, cutflow, isData, gen_weights)

        JEC_loader = JECs(year)

        #########################
        # Objects
        #########################

        # muons: our analysis definition (tight ID, pT > 20, |eta| < 2.4, dz/dxy); the IsoMu24
        # match is the analysis one ("MuonTrigMatchMuon": HLT fired, filter bit 3, pT > 26)
        muons, mtrigvars = objects.good_muons(events, events.Muon, year)
        muons_isomu24_match = mtrigvars["MuonTrigMatchMuon"]
        muons_mu50_match = muon_trig_match(events, muons, "Mu50", MU50_FILTERBIT)

        ref_muons = muons[
            (muons.pt > self.muon_selection["pt"]) & (abs(muons.eta) < self.muon_selection["eta"])
        ]
        leading_muon = ak.firsts(ref_muons)

        electrons, _ = objects.good_electrons(events, events.Electron, year)

        # AK4 jets
        jets, _ = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=None,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )

        # MET (as in bbtautauSkimmer)
        if self._nano_version == "v15":
            met = events.PuppiMET
        elif JEC_loader.met_factory is not None:
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        central_jets = jets[
            (jets.pt > self.ak4_selection["pt"]) & (abs(jets.eta) < self.ak4_selection["eta"])
        ]

        # AK8 jets: tight ID, pT > 200, |eta| < 2.5, away from the leading muon
        fatjets = objects.get_ak8jets(events.FatJet, nano_version=self._nano_version)
        fatjets, _ = JEC_loader.get_jec_jets(
            events,
            fatjets,
            year,
            isData,
            jecs=None,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        fatjet_sel = {k: v for k, v in self.fatjet_selection.items() if k != "dr_muon"}
        fatjets = objects.good_ak8jets(fatjets, **fatjet_sel, nano_version=self._nano_version)
        fatjets["dr_muon"] = ak.fill_none(fatjets.delta_r(leading_muon), PAD_VAL)
        fatjets = fatjets[
            (fatjets.dr_muon > self.fatjet_selection["dr_muon"]) | (fatjets.dr_muon == PAD_VAL)
        ]

        print("Objects", f"{time.time() - start:.2f}")

        #########################
        # Variables
        #########################

        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if val in events.fields
        }
        eventVars["METPt"] = met.pt.to_numpy()
        eventVars["METPhi"] = met.phi.to_numpy()
        eventVars["nMuons"] = ak.to_numpy(ak.num(muons))
        eventVars["nElectrons"] = ak.to_numpy(ak.num(electrons))

        HLTVars = {}
        zeros = np.zeros(len(events), dtype="int")
        for trigger in self.HLTs[year]:
            if trigger in events.HLT.fields:
                HLTVars[f"HLT_{trigger}"] = events.HLT[trigger].to_numpy().astype(int)
            else:
                HLTVars[f"HLT_{trigger}"] = zeros

        muonVars = {
            f"Muon{key}": pad_val(muons[var], self.num_muons, axis=1)
            for (var, key) in self.skim_vars["Muon"].items()
        }
        muonVars["MuonTrigMatchIsoMu24"] = pad_val(
            muons_isomu24_match, self.num_muons, False, axis=1
        ).astype(int)
        muonVars["MuonTrigMatchMu50"] = pad_val(
            muons_mu50_match, self.num_muons, False, axis=1
        ).astype(int)

        ak4JetVars = {
            f"ak4Jet{key}": pad_val(central_jets[var], self.num_ak4_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        fatjet_skimvars = self.skim_vars["FatJet"]
        if not isData:
            fatjet_skimvars = {**fatjet_skimvars, "pt_gen": "MatchedGenJetPt"}
        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], self.num_ak8_jets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }

        # JMS/JMR +-10% variations (seeded smearing, as in HH4b's ttSkimmer); data: none
        jmsr_shifted_vars = get_jmsr(
            fatjets,
            self.num_jmsr_jets,
            jmsr_vars=self.jmsr_vars,
            jms_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
            jmr_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
            isData=isData,
        )
        for var in self.jmsr_vars:
            key = fatjet_skimvars[var]
            for shift, vals in jmsr_shifted_vars[var].items():
                if shift != "":  # nominal = saved mass (JMS = JMR = 1)
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = vals

        # fat-jet trigger objects (id 6, pT >= 100), sorted by dR to the leading saved AK8 jet
        fatjet_objs = events.TrigObj[(events.TrigObj.id == 6) & (events.TrigObj.pt >= 100)]
        dr_fj0 = ak.flatten(fatjet_objs.metric_table(fatjets[:, 0:1]), axis=-1)
        fatjet_objs = fatjet_objs[ak.argsort(dr_fj0)]
        trigObjVars = {
            f"TriggerObject{key}": pad_val(fatjet_objs[var], self.num_trigobjs, axis=1)
            for (var, key) in self.skim_vars["TriggerObject"].items()
        }

        genVars = {}
        if not isData and "TTto" in dataset:
            genVars = gen_selection_top(events, fatjets, self.num_ak8_jets)

        skimmed_events = {
            **eventVars,
            **HLTVars,
            **muonVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **trigObjVars,
            **genVars,
        }

        print("Vars", f"{time.time() - start:.2f}")

        #########################
        # Selection
        #########################

        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in utils.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)

        add_selection("ak4_jetveto", get_jetveto_event(jets, year), *selection_args)

        HLT_triggered = np.any(
            np.array([events.HLT[t] for t in REFERENCE_HLTS if t in events.HLT.fields]), axis=0
        )
        add_selection("trigger", HLT_triggered, *selection_args)

        add_selection("muon", ak.num(ref_muons) >= 1, *selection_args)

        add_selection("ak8_jet", ak.num(fatjets) >= 1, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        #########################
        # Weights
        #########################

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            # normalization uses all generated events (no gen-level selection here)
            gen_selected = np.ones(len(events), dtype=bool)
            weights_dict, totals_temp = self.add_weights(
                events, year, dataset, gen_weights, gen_selected
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
        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(self, events, year, dataset, gen_weights, gen_selected) -> tuple[dict, dict]:
        """Nominal weight (genweight x pileup) normalized to xsec x lumi, the unnormalized
        ``weight_noxsec``, and the ``np_nominal`` total, as in bbtautauSkimmer (no variations)."""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        pileup_input = (
            events.Pileup.nPU.to_numpy()
            if "Pu60" in dataset or "Pu70" in dataset
            else events.Pileup.nTrueInt.to_numpy()
        )
        add_pileup_weight_update(weights, year, pileup_input, dataset)

        weight_np = weights.partial_weight(include=hh_vars.norm_preserving_weights)
        totals_dict = {"np_nominal": np.sum(weight_np[gen_selected])}

        weight_norm = self.get_dataset_norm(year, dataset)
        weights_dict = {
            "weight": weights.weight() * weight_norm,
            # unnormalized, to check in postprocessing that the normalization was applied
            "weight_noxsec": weights.weight(),
        }
        return weights_dict, totals_dict
