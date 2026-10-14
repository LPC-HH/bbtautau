"""
TXbb tagger efficiency scale factors.

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

import json
from functools import lru_cache

import numpy as np

from bbtautau.postprocessing.bbtautau_types import LoadedSample
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
