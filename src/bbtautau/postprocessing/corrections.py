"""
TXbb tagger efficiency scale factors, and event-level trigger scale factors (``apply_trigger_sfs``).

Calibration: GloParT XbbVsQCDTop, muon-tagged g->bb method, per era, m_SD > 40 GeV, tau21 < 0.30
(``SFs/TXbb/AN_context_prompt.md``). JSONs at ``SFs/TXbb/sf_json/<year>/sf_globalParT_tau21_0p30.json``,
keyed ``WPgt<wp>_pt<lo>to<hi>`` with ``{"final": {"central", "high", "low"}}`` where high/low are the
*absolute* SF +/- total uncertainty (not deltas).

The WPs are cumulative (score > WP), so the SF of a single-threshold cut depends only on the cut value
and the jet pT, never on the event's own score. Analysis cuts don't sit on the calibrated grid, so the
SF at the actual cut is linearly interpolated between the two bracketing WPs (clamped to the first/last
WP outside the calibrated range).

Applied per event to any jet passing the cut (pass *and* fail regions -- the fail region is an OR of
failed bb/tt cuts, so it contains bb-passing jets), with one up/down variation per pT bin.

Author: Ludovico Mori
"""

from __future__ import annotations

import gzip
import json
import warnings
from functools import lru_cache

import correctionlib
import hist
import numpy as np
from boostedhh.processors.corrections import (
    _get_category_value,
    _replace_inf,
    get_electron_sf_era,
    get_pog_json_update,
)

from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.userConfig import MAIN_DIR

TXBB_SF_DIR = MAIN_DIR / "SFs" / "TXbb" / "sf_json"
TXBB_SF_FNAME = "sf_globalParT_tau21_0p30"


def _parse_wp(wp_str: str) -> float:
    return float(wp_str.removeprefix("WPgt").replace("p", "."))


def _parse_pt_bin(pt_str: str) -> tuple[float, float]:
    lo, hi = pt_str.removeprefix("pt").split("to")
    return float(lo), np.inf if hi == "Inf" else float(hi)


@lru_cache
def load_txbb_sfs(year: str) -> dict:
    """{"wps": (n_wp,), "pt_labels": [...], "pt_edges": [(lo, hi), ...], "central"/"high"/"low": (n_wp, n_pt)}"""
    with (TXBB_SF_DIR / year / f"{TXBB_SF_FNAME}.json").open() as f:
        raw = json.load(f)

    entries = {}
    for key, val in raw.items():
        wp_str, pt_str = key.split("_", 1)
        entries[_parse_wp(wp_str), pt_str] = val["final"]

    wps = sorted({wp for wp, _ in entries})
    pt_labels = sorted({pt for _, pt in entries}, key=lambda s: _parse_pt_bin(s)[0])

    out = {"wps": np.array(wps), "pt_labels": pt_labels}
    out["pt_edges"] = [_parse_pt_bin(p) for p in pt_labels]
    for comp in ["central", "high", "low"]:
        out[comp] = np.array([[entries[wp, pt][comp] for pt in pt_labels] for wp in wps])
    return out


def txbb_pt_labels(year: str) -> list[str]:
    return load_txbb_sfs(year)["pt_labels"]


def txbb_sf_at_cut(year: str, txbb_cut: float) -> dict[str, np.ndarray]:
    """SF central/high/low per pT bin for a cumulative cut at ``txbb_cut`` (np.interp clamps at the ends)."""
    sfs = load_txbb_sfs(year)
    return {
        comp: np.array(
            [np.interp(txbb_cut, sfs["wps"], sfs[comp][:, j]) for j in range(len(sfs["pt_labels"]))]
        )
        for comp in ["central", "high", "low"]
    }


def apply_txbb_sfs(
    sample: LoadedSample,
    year: str,
    txbb_cut: float,
    bb_disc_col: str,
    weight_key: str = "finalWeight",
) -> None:
    """
    In place: multiplies the nominal SF into ``sample.events[weight_key]`` for jets with
    ``bb_disc_col >= txbb_cut`` and adds ``{weight_key}_txbb_<ptlabel>_{up,down}`` columns, each
    varying only that pT bin.
    """
    if not len(sample.events):
        return

    sfs = load_txbb_sfs(year)
    at_cut = txbb_sf_at_cut(year, txbb_cut)

    passes = sample.get_var(bb_disc_col) >= txbb_cut
    pt = sample.get_var("bbFatJetPt")
    bin_masks = [passes & (pt >= lo) & (pt < hi) for lo, hi in sfs["pt_edges"]]

    sf_nom = np.ones(len(pt))
    for j, mask in enumerate(bin_masks):
        sf_nom[mask] = at_cut["central"][j]

    orig_weight = sample.events[weight_key].to_numpy().copy()
    sample.events[weight_key] = orig_weight * sf_nom

    for j, (label, mask) in enumerate(zip(sfs["pt_labels"], bin_masks)):
        for shift, comp in [("up", "high"), ("down", "low")]:
            sf_var = sf_nom.copy()
            sf_var[mask] = at_cut[comp][j]
            sample.events[f"{weight_key}_txbb_{label}_{shift}"] = orig_weight * sf_var

    # trigger weight variations made before the TXbb SF (apply_trigger_sfs) also get its nominal
    for col in {c[0] if isinstance(c, tuple) else c for c in sample.events.columns}:
        if col.startswith(f"{weight_key}_trigger_"):
            sample.events[col] = sample.get_var(col) * sf_nom


#######################################################################################
# Trigger SFs
#
# MC events pass the channel's trigger OR in emulation (utils.trigger_masks): pnetbb | <leg2>,
# with <leg2> = ditau (hh), muiso24 (hm), ele30 (he). Each MC event passing it is weighted by
#   SF = eps_OR_data / eps_OR_MC,   eps_OR = 1 - (1 - eps_pnetbb)(1 - eps_leg2)
# (legs assumed independent, no trigger-object matching).
#  - eps_pnetbb: HH4b's AK8 maps (``corrections.trigger_SF`` in HH4b), on the bb and tt jets:
#    eps = 1 - (1 - e(bb))(1 - e(tt)), e(jet) = e(pt, msd) * e(ParT Xbb-vs-QCD).
#  - eps_leg2: POG data/MC efficiencies on the offline lepton/taus of the skim.
# Uncertainties: HH4b's global stat factor for the pnetbb leg ("trigger"), POG eff variations
# for the second leg ("trigger_<leg2>").
#######################################################################################

TRIGGER_SF_DIR = MAIN_DIR / "SFs" / "trigger" / "hh4b"
TRIGGER_REGION = "QCD"  # HH4b nominal (PostProcess.py: trigger_region = "QCD")
TRIGGER_TXBB_LABEL = "txbbGloParT"
TRIGGER_TXBB_VAR = "ParTXbbvsQCD"  # Xbb / (Xbb + QCD), the input of HH4b's txbbGloParT maps

MUON_TRIGGER_CORR = "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"  # as bbtautauSkimmer.py
ELECTRON_TRIGGER_PATH = "HLT_SF_Ele30_TightID"  # as boostedhh get_electron_trigger_sfs
TAU_TRIGGER_WP = "Medium"  # as boostedhh get_tau_trigger_sf (dm = -1, inclusive)


def trigger_leg2(channel: Channel) -> str:
    """Second leg of the channel's trigger OR (``HLTs.MENU`` key)."""
    (leg2,) = (key for key in channel.hlt_menu if key != "pnetbb")
    return leg2


def _fatjet_trig_fname(year: str, label: str) -> str:
    return TRIGGER_SF_DIR / f"fatjet_triggereff_{year}_{label}_{TRIGGER_REGION}.json"


def _fatjet_trig_year(year: str) -> str:
    """Year of the maps to use: 2023BPix if a year has no maps (2024), as HH4b."""
    if all(_fatjet_trig_fname(year, lab).exists() for lab in ["ptmsd", TRIGGER_TXBB_LABEL]):
        return year
    warnings.warn(
        f"Trigger-efficiency maps not found for {year}; falling back to 2023BPix.", stacklevel=2
    )
    return "2023BPix"


@lru_cache
def _load_fatjet_trig_effs(eff_year: str) -> dict:
    """Data/MC corrections and bin edges of the ptmsd and txbb maps."""
    out = {}
    for label in ["ptmsd", TRIGGER_TXBB_LABEL]:
        fname = _fatjet_trig_fname(eff_year, label)
        cset = correctionlib.CorrectionSet.from_file(str(fname))
        out[f"data_{label}"] = cset[f"fatjet_triggereffdata_{eff_year}_{label}"]
        out[f"mc_{label}"] = cset[f"fatjet_triggereffmc_{eff_year}_{label}"]
        with fname.open() as f:
            node = json.load(f)["corrections"][0]["data"]["content"][0]["value"]
        # one input ("input", "edges") or several ("inputs", list of edges)
        if "inputs" in node:
            out.update(dict(zip(node["inputs"], node["edges"])))
        else:
            out[node["input"]] = node["edges"]
    return out


def _pnetbb_effs(sample: LoadedSample, year: str, weight: np.ndarray):
    """Per-event pnetbb-leg data and MC efficiencies and HH4b's (total, total_err).

    Follows HH4b ``trigger_SF``: zero efficiencies -> 1 for the nominal; inputs outside the maps
    get the maps' flow value (1.0, also for stat_up). total = sum_jets sum_events(w * sf_jet),
    total_err = per-jet stat error summed linearly within (pt, msd, xbb) bins (flow included),
    in quadrature across bins and jets.
    """
    effs = _load_fatjet_trig_effs(_fatjet_trig_year(year))
    txbb_edges = effs.get("txbb", effs.get("xbb"))
    axes = [
        hist.axis.Variable(effs["pt"], name="pt"),
        hist.axis.Variable(effs["msd"], name="msd"),
        hist.axis.Variable(txbb_edges, name="xbb"),
    ]

    eff_jets = {"data": [], "mc": []}
    totals, total_errs = [], []
    for jet in ["bb", "tt"]:
        pt = sample.get_var(f"{jet}FatJetPt")
        msd = sample.get_var(f"{jet}FatJetMsd")
        xbb = sample.get_var(f"{jet}FatJet{TRIGGER_TXBB_VAR}")

        eff = {}
        for dm in ["data", "mc"]:
            for var in ["nominal", "stat_up"]:
                e_ptmsd = effs[f"{dm}_ptmsd"].evaluate(pt, msd, var)
                e_txbb = effs[f"{dm}_{TRIGGER_TXBB_LABEL}"].evaluate(xbb, var)
                if var == "nominal":
                    e_ptmsd[e_ptmsd == 0] = 1.0
                    e_txbb[e_txbb == 0] = 1.0
                eff[dm, var] = e_ptmsd * e_txbb
            eff_jets[dm].append(eff[dm, "nominal"])

        sf_jet = eff["data", "nominal"] / eff["mc", "nominal"]
        sf_err_jet = sf_jet * np.sqrt(
            (eff["data", "stat_up"] / eff["data", "nominal"]) ** 2
            + (eff["mc", "stat_up"] / eff["mc", "nominal"]) ** 2
        )
        h_yield = hist.Hist(*axes)
        h_yield_err = hist.Hist(*axes)
        h_yield.fill(pt, msd, xbb, weight=weight * sf_jet)
        h_yield_err.fill(pt, msd, xbb, weight=weight * sf_err_jet)
        totals.append(np.sum(h_yield.values(flow=True)))
        total_errs.append(np.linalg.norm(np.nan_to_num(h_yield_err.values(flow=True))))

    eff_data = 1 - (1 - eff_jets["data"][0]) * (1 - eff_jets["data"][1])
    eff_mc = 1 - (1 - eff_jets["mc"][0]) * (1 - eff_jets["mc"][1])
    return eff_data, eff_mc, np.sum(totals), np.linalg.norm(total_errs)


@lru_cache
def _load_pog_corrections(obj: str, year: str, names: tuple[str, ...]):
    """POG corrections from cvmfs (``_replace_inf`` for the env's correctionlib version) and the
    raw json nodes, keyed by name."""
    with gzip.open(get_pog_json_update(obj, year), "rt") as f:
        payload = json.load(f)
    payload["corrections"] = [c for c in payload["corrections"] if c["name"] in names]
    nodes = {c["name"]: c["data"] for c in payload["corrections"]}
    cset = correctionlib.CorrectionSet.from_string(json.dumps(_replace_inf(payload)))
    return {name: cset[name] for name in names}, nodes


def _eval_where(valid: np.ndarray, fn, *inputs) -> np.ndarray:
    """``fn(*inputs)`` where ``valid``, 0 elsewhere (POG jsons have flow = error)."""
    out = np.zeros(len(valid))
    if np.any(valid):
        out[valid] = fn(*[x[valid] for x in inputs])
    return out


def _channel_lepton(sample: LoadedSample, lep: str) -> tuple[np.ndarray, np.ndarray]:
    """pt, eta of the lepton assigned to the tautau jet (``leptons_assignment``, the one that
    defines the a priori hm/he channel); the leading stored lepton if none is assigned."""
    mask = sample.m_mask if lep == "Muon" else sample.e_mask
    pt_all, eta_all = sample.get_var(f"{lep}Pt"), sample.get_var(f"{lep}Eta")
    idx = np.where(mask.any(axis=1), np.argmax(mask, axis=1), 0)
    rows = np.arange(len(idx))
    return pt_all[rows, idx], eta_all[rows, idx]


def _leg2_effs(sample: LoadedSample, year: str, leg2: str) -> dict[str, np.ndarray]:
    """Second-leg efficiencies {data, mc, data_up, data_down, mc_up, mc_down}; 0 outside the
    POG json range (that leg then does not contribute to the OR)."""
    effs = {}
    if leg2 == "muiso24":
        (corr,), (node,) = (
            d.values() for d in _load_pog_corrections("muon", year, (MUON_TRIGGER_CORR,))
        )
        pt, eta = _channel_lepton(sample, "Muon")
        eta_edges, pt_min = node["edges"], node["content"][0]["edges"][0]
        valid = (eta >= eta_edges[0]) & (eta < eta_edges[-1]) & (pt >= pt_min)
        # stat only (agreed choice; syst_DATAeff / syst_MCeff are ignored)
        for dm, tag in [("data", "DATAeff"), ("mc", "MCeff")]:
            nom, stat = (
                _eval_where(valid, lambda e, p, v=v: corr.evaluate(e, p, v), eta, pt)
                for v in [f"nominal_{tag}", f"stat_{tag}"]
            )
            effs[dm] = nom
            effs[f"{dm}_up"] = nom + stat
            effs[f"{dm}_down"] = nom - stat

    elif leg2 == "ele30":
        names = ("Electron-HLT-DataEff", "Electron-HLT-McEff")
        corrs, _ = _load_pog_corrections("electron_hlt", year, names)
        pt, eta = _channel_lepton(sample, "Electron")
        era = get_electron_sf_era(year)
        valid = pt >= 25  # json pt edges start at 25; eta edges are +-inf
        for dm, name in zip(["data", "mc"], names):
            for shift, valtype in [("", "nom"), ("_up", "up"), ("_down", "down")]:
                effs[f"{dm}{shift}"] = _eval_where(
                    valid,
                    lambda e, p, vt=valtype, c=corrs[name]: c.evaluate(
                        era, vt, ELECTRON_TRIGGER_PATH, e, p
                    ),
                    eta,
                    pt,
                )

    elif leg2 == "ditau":
        corrs, nodes = _load_pog_corrections("tau", year, ("tau_trigger",))
        corr = corrs["tau_trigger"]
        pts = [sample.get_var("TauPt")[:, i] for i in range(2)]
        for dm, corrtype in [("data", "eff_data"), ("mc", "eff_mc")]:
            node = nodes["tau_trigger"]
            for key in ["ditau", corrtype, TAU_TRIGGER_WP, -1, "nom"]:
                node = _get_category_value(node, key)  # handles transform nodes (2023 dm)
            pt_min = float(node["edges"][0])
            for shift, syst in [("", "nom"), ("_up", "up"), ("_down", "down")]:
                eff_taus = [
                    _eval_where(
                        pt >= pt_min,
                        lambda p, s=syst, ct=corrtype: corr.evaluate(
                            p, np.full(len(p), -1), "ditau", TAU_TRIGGER_WP, ct, s
                        ),
                        pt,
                    )
                    for pt in pts
                ]
                effs[f"{dm}{shift}"] = eff_taus[0] * eff_taus[1]
    else:
        raise ValueError(f"No second-leg efficiencies for trigger {leg2}")

    return {key: np.clip(val, 0.0, 1.0) for key, val in effs.items()}


def apply_trigger_sfs(
    sample: LoadedSample,
    year: str,
    channel: Channel,
    weight_key: str = "finalWeight",
) -> None:
    """
    In place, MC only: multiplies the event trigger SF into ``sample.events[weight_key]`` and adds
      ``{weight_key}_trigger_{up,down}``: pnetbb leg, HH4b's global relative stat uncertainty,
          SF * (1 +- total_err / total) (HH4b PostProcess.calculate_trigger_weights);
      ``{weight_key}_trigger_<leg2>_{up,down}``: second leg, the SF with the leg's data eff shifted
          and, separately, its MC eff shifted the opposite way, the two SF shifts in quadrature.
    """
    if sample.sample.isData:
        raise ValueError(f"Trigger SFs are for MC only, got {sample.sample}")
    if not len(sample.events):
        return

    leg2 = trigger_leg2(channel)
    orig_weight = sample.get_var(weight_key).copy()

    p_data, p_mc, total, total_err = _pnetbb_effs(sample, year, orig_weight)
    l_eff = _leg2_effs(sample, year, leg2)

    def eff_or(e1, e2):
        return 1 - (1 - e1) * (1 - e2)

    def get_sf(l_data, l_mc):
        eff_data, eff_mc = eff_or(p_data, l_data), eff_or(p_mc, l_mc)
        if np.any(eff_mc == 0):
            warnings.warn(f"{sample.sample.label}: MC trigger eff = 0, SF set to 1", stacklevel=2)
        return np.divide(eff_data, eff_mc, out=np.ones_like(eff_mc), where=eff_mc != 0)

    sf = get_sf(l_eff["data"], l_eff["mc"])
    sample.events[weight_key] = orig_weight * sf

    rel_unc = total_err / total
    sample.events[f"{weight_key}_trigger_up"] = orig_weight * sf * (1 + rel_unc)
    sample.events[f"{weight_key}_trigger_down"] = orig_weight * sf * (1 - rel_unc)

    d_up = np.hypot(
        get_sf(l_eff["data_up"], l_eff["mc"]) - sf, get_sf(l_eff["data"], l_eff["mc_down"]) - sf
    )
    d_down = np.hypot(
        get_sf(l_eff["data_down"], l_eff["mc"]) - sf, get_sf(l_eff["data"], l_eff["mc_up"]) - sf
    )
    sample.events[f"{weight_key}_trigger_{leg2}_up"] = orig_weight * (sf + d_up)
    sample.events[f"{weight_key}_trigger_{leg2}_down"] = orig_weight * (sf - d_down)

    print(
        f"Trigger SF {sample.sample.label} ({channel.key}): mean {np.mean(sf):.3f}, "
        f"pnetbb-leg rel. unc. {rel_unc:.3f}"
    )
