"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from boostedhh.processors.utils import (
    GEN_FLAGS,
    P4,
    PDGID,
    add_selection,
    pad_val,
)
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray


def _iterate_children(children, parent_pdgId):
    """Iterates through the children of a particle in case of photon scattering to find the final daughter particles"""

    for _ in range(5):
        # check if any of the children are the same as the parent pdgId
        mask = ak.any(np.abs(children.pdgId) == parent_pdgId, axis=2)
        if not ak.any(mask):
            break

        children_children = ak.flatten(
            children[np.abs(children.pdgId) == parent_pdgId].children, axis=3
        )

        # get next layer of children
        children = ak.where(mask, children_children, children)

    return children


def _sum_taus(taut):
    return ak.sum(taut, axis=1)


def gen_selection_HHbbtautau(
    events: NanoEventsArray,
    fatjets: FatJetArray,
    selection_args: list,
):
    """Gets HH, bb, and tautau 4-vectors + tau decay information"""

    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]

    # finding the two gen higgs
    higgs = genparts[genparts.pdgId == PDGID.H]

    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in P4.items()}

    # saving whether H->bb or H->tautau
    higgs_children = higgs.children
    # pad_val is necessary to avoid a numpy MaskedArray even though all events have exactly 2 Higgs'
    GenHiggsVars["GenHiggsChildren"] = pad_val(higgs_children.pdgId[:, :, 0], 2, axis=1)

    # finding bb and VV children
    is_bb = np.abs(higgs_children.pdgId) == PDGID.b
    is_tt = np.abs(higgs_children.pdgId) == PDGID.tau

    # checking that there are 2 bs and 2 taus
    has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2
    has_tt = ak.sum(ak.flatten(is_tt, axis=2), axis=1) == 2
    if selection_args is not None:
        add_selection("has_bbtautau", has_bb * has_tt, *selection_args)

    bb = ak.flatten(higgs_children[is_bb], axis=2)
    GenbbVars = {f"Genbb{key}": pad_val(bb[var], 2, axis=1) for (var, key) in P4.items()}

    taus = higgs_children[is_tt]
    flat_taus = ak.flatten(taus, axis=2)
    GenTauVars = {f"GenTau{key}": pad_val(flat_taus[var], 2, axis=1) for (var, key) in P4.items()}

    tau_children = ak.flatten(taus.children, axis=2)
    tau_children = _iterate_children(tau_children, PDGID.tau)

    # check if tau children are leptons or hadrons
    # check neutral and charged pion IDs for hadronic taus
    tauh = _sum_taus(
        ak.any([ak.any(np.abs(tau_children.pdgId) == pid, axis=2) for pid in PDGID.pions], axis=0)
    )
    taumu = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.mu, axis=2))
    taue = _sum_taus(ak.any(np.abs(tau_children.pdgId) == PDGID.e, axis=2))

    GenTauVars["GenTauhh"] = (tauh == 2).to_numpy()
    GenTauVars["GenTauhm"] = ((tauh == 1) & (taumu == 1)).to_numpy()
    GenTauVars["GenTauhe"] = ((tauh == 1) & (taue == 1)).to_numpy()

    # fatjet gen matching
    Hbb = higgs[ak.sum(is_bb, axis=2) == 2]
    Hbb = ak.pad_none(Hbb, 1, axis=1, clip=True)[:, 0]

    Htt = higgs[ak.sum(is_tt, axis=2) == 2]
    Htt = ak.pad_none(Htt, 1, axis=1, clip=True)[:, 0]

    # TODO: check more than just the leading two fatjets!
    bbdr = fatjets.delta_r(Hbb)
    ttdr = fatjets.delta_r(Htt)

    match_dR = 0.8
    Hbb_match = bbdr <= match_dR
    Htt_match = ttdr <= match_dR

    # overlap removal - in the case where fatjet is matched to both, match it only to the closest Higgs
    Hbb_match = (Hbb_match * ~Htt_match) + (bbdr <= ttdr) * (Hbb_match * Htt_match)
    Htt_match = (Htt_match * ~Hbb_match) + (bbdr > ttdr) * (Hbb_match * Htt_match)

    GenMatchingVars = {
        "ak8FatJetHbb": pad_val(Hbb_match, 3, axis=1),
        "ak8FatJetHtt": pad_val(Htt_match, 3, axis=1),
    }

    return {**GenHiggsVars, **GenbbVars, **GenTauVars, **GenMatchingVars}


def gen_selection_HH4b(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,  # noqa: ARG001
):
    """
    Save GenVars for HH(4b) events
    """
    genparts = events.GenPart[events.GenPart.hasFlags(GEN_FLAGS)]

    # finding the two gen higgs
    higgs = genparts[genparts.pdgId == PDGID.H]
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in P4.items()}
    higgs_children = higgs.children
    is_bb = np.abs(higgs_children.pdgId) == PDGID.b
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": pad_val(bs[var], 4, axis=1) for (var, key) in P4.items()}

    return {**GenHiggsVars, **GenbVars}


def gen_selection_V(
    events: NanoEventsArray,
    fatjets: FatJetArray,  # noqa: ARG001
    selection_args: list,  # noqa: ARG001
):
    """Save the LHE-level V (W/Z) pT for the Wto2Q-/Zto2Q- V+jets samples.

    Ported from HH4b's ``gen_selection_V``
    (https://github.com/LPC-HH/HH4b/commit/7e8dca7, "Skimmer: save LHE-level V pT"); only the
    ``GenVLHEPt`` piece is needed here (not the fatjet/daughter-flavor matching HH4b also does in
    that function, which is for V-tagging and out of scope for this).

    Returns an empty dict (no ``GenVLHEPt`` column) if the input NanoAOD doesn't carry the
    ``LHEPart`` collection, same gating as HH4b's.
    """
    if "LHEPart" not in events.fields:
        return {}

    lhe = events.LHEPart
    lhe_vs = lhe[((abs(lhe.pdgId) == PDGID.W) | (abs(lhe.pdgId) == PDGID.Z)) & (lhe.status == 2)]
    return {"GenVLHEPt": pad_val(lhe_vs.pt, 1, axis=1)}
