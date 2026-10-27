"""
Validation-region (VR) closure for the data-driven QCD+DY (alphabet) background estimate.

Tests whether the VR-fail mass shape, scaled by the pass/fail transfer factor, reproduces
the VR-pass data (after adding the non-QCD/DY MC). Good closure in this high-stats,
unblindable annulus supports the fail->pass extrapolation used in the signal region.

------------------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------------------
1. Produce the VR templates with the main postprocessor (--validation-region writes them under
   <template-dir>/validation/bmin_<b>/<signal>/<channel>/<year>_templates.pkl; runs made
   before the CR -> VR rename used <template-dir>/control/, which is read as a fallback):

    micromamba run -n hh python -m bbtautau.postprocessing.postprocessing \
        --validation-region --templates --year 2022 --channel hh --bmin 10 \
        --template-dir src/bbtautau/postprocessing/templates/<tag>   [other template args]

2. Run this closure on those templates:

    micromamba run -n hh python -m bbtautau.postprocessing.validation_region_closure \
        --template-dir src/bbtautau/postprocessing/templates/<tag> \
        --year 2022 --channel hh --bmin 10 --signal ggfbbtt

Args: --template-dir (parent dir passed in step 1), --year (one or more, summed),
--channel (he/hh/hm), --bmin, --signal (default ggfbbtt), --out (optional png path;
defaults under <template-dir>/validation/, or control/ for legacy runs). Prints a
chi2/ndf and the per-bin transfer factor, and saves a per-sample stacked closure plot (each MC background shown separately
plus the data-driven QCD+DY) with a data/prediction ratio panel.
------------------------------------------------------------------------------------------

Closure logic (per mass bin i):
    qcd_dy_fail_i = data_fail_i - sum(nonQCDDY_MC_fail_i)     # data-driven QCD+DY shape
    qcd_dy_pass_i = data_pass_i - sum(nonQCDDY_MC_pass_i)
    qcd_eff       = sum_i qcd_dy_pass_i / sum_i qcd_dy_fail_i # 0th-order TF (flat)
    pred_pass_i   = qcd_eff * qcd_dy_fail_i + sum(nonQCDDY_MC_pass_i)
Compare pred_pass to data_pass. The per-bin TF (qcd_dy_pass/qcd_dy_fail) vs mass reveals
any mass dependence -- exactly the residual the SR Bernstein polynomial absorbs.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import numpy as np
from boostedhh.hh_vars import LUMI
from hist import Hist

from bbtautau.postprocessing import plotting
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES

DATA_KEY = "data"

# Data-driven QCD+DY is stacked as a single synthetic sample under this key; reuse the
# colour/label that plotting.py already defines for it.
QCDDY_KEY = "qcddy"

# Non-QCD/DY MC backgrounds (subtracted from data to expose the data-driven QCD+DY).
# QCD and DY(->ll, 'dyjets') are intentionally NOT here -- they are the data-driven part.
NONQCDDY_BG_KEYS = [
    k for k, s in SAMPLES.items() if s.get_type() == "bg" and k not in ("qcd", "dyjets")
]


# Subdir postprocessing.py --validation-region writes to; "control" is the legacy name used
# by runs made before the CR -> VR rename (kept on disk, read as a fallback).
VR_SUBDIR = "validation"
LEGACY_VR_SUBDIR = "control"


def vr_subdir(template_dir: Path) -> str:
    """``validation`` if it exists under ``template_dir``, else the legacy ``control``."""
    if (Path(template_dir) / VR_SUBDIR).exists() or not (
        Path(template_dir) / LEGACY_VR_SUBDIR
    ).exists():
        return VR_SUBDIR
    return LEGACY_VR_SUBDIR


def vr_template_path(template_dir: Path, year: str, channel: str, signal: str, bmin: int) -> Path:
    """Path of the VR templates pickle (mirrors postprocessing.py's validation/ layout)."""
    return (
        Path(template_dir)
        / vr_subdir(template_dir)
        / f"bmin_{bmin}"
        / signal
        / channel
        / f"{year}_templates.pkl"
    )


def load_vr_templates(
    template_dir: Path, years: list[str], channel: str, signal: str, bmin: int
) -> dict:
    """Load VR templates for one or more years, summing the per-region hists across years."""
    combined: dict = {}
    for year in years:
        path = vr_template_path(template_dir, year, channel, signal, bmin)
        with path.open("rb") as f:
            tmpl = pickle.load(f)
        for region in ("pass", "fail"):
            combined[region] = (
                tmpl[region] if region not in combined else combined[region] + tmpl[region]
            )
    return combined


def _vals(h, sample: str) -> np.ndarray:
    return h[sample, :].values()


def qcd_dy_shape(h) -> np.ndarray:
    """Data-driven QCD+DY per-bin yield = data - non-QCD/DY MC."""
    mc = np.sum([_vals(h, k) for k in NONQCDDY_BG_KEYS], axis=0)
    return _vals(h, DATA_KEY) - mc


def nonqcddy_mc(h) -> np.ndarray:
    return np.sum([_vals(h, k) for k in NONQCDDY_BG_KEYS], axis=0)


def run_closure(templates: dict) -> dict:
    """Compute the closure arrays from a VR templates dict (keys 'pass'/'fail')."""
    hp, hf = templates["pass"], templates["fail"]
    # axes[0] is the Sample (StrCategory) axis; the mass axis is what remains after
    # indexing a sample, so read the binning from there.
    mass_axis = hp[DATA_KEY, :].axes[0]
    edges = mass_axis.edges
    centers = mass_axis.centers

    qcd_dy_fail = qcd_dy_shape(hf)
    qcd_dy_pass = qcd_dy_shape(hp)
    mc_pass = nonqcddy_mc(hp)
    data_pass = _vals(hp, DATA_KEY)

    # 0th-order (flat) transfer factor from the integrated pass/fail QCD+DY ratio.
    qcd_eff = qcd_dy_pass.sum() / qcd_dy_fail.sum()
    pred_qcd_dy_pass = qcd_eff * qcd_dy_fail
    pred_pass = pred_qcd_dy_pass + mc_pass

    # per-bin TF reveals mass dependence (the SR Bernstein residual)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf_perbin = np.where(qcd_dy_fail > 0, qcd_dy_pass / qcd_dy_fail, np.nan)

    return {
        "centers": centers,
        "edges": edges,
        "data_pass": data_pass,
        "mc_pass": mc_pass,
        "qcd_dy_fail": qcd_dy_fail,
        "qcd_dy_pass": qcd_dy_pass,
        "pred_qcd_dy_pass": pred_qcd_dy_pass,
        "pred_pass": pred_pass,
        "qcd_eff": qcd_eff,
        "tf_perbin": tf_perbin,
    }


def build_pred_hist(templates: dict, res: dict) -> tuple[Hist, list[str]]:
    """Pack the VR-pass prediction into a Hist for plotting.ratioHistPlot.

    Each non-QCD/DY MC background is kept as its own sample (so the stack shows them
    individually), the data-driven QCD+DY (TF x fail) goes into a single ``qcddy`` sample,
    and the observed VR-pass data goes into ``data``.
    Returns (hist, ordered background keys to stack).
    """
    return _stack_hist(templates["pass"], res["pred_qcd_dy_pass"])


def _stack_hist(h_region, qcd_dy_vals: np.ndarray) -> tuple[Hist, list[str]]:
    """New Hist with the region's non-QCD/DY MC samples, ``qcd_dy_vals`` as the ``qcddy``
    sample and the region's data. Returns (hist, ordered background keys to stack)."""
    present = list(h_region.axes["Sample"])
    mc_keys = [k for k in NONQCDDY_BG_KEYS if k in present]
    bg_keys = mc_keys + [QCDDY_KEY]

    mass_axis = h_region[DATA_KEY, :].axes[0]
    samples = bg_keys + [DATA_KEY]
    h = Hist(hist.axis.StrCategory(samples, name="Sample"), mass_axis, storage="double")

    for k in mc_keys:
        h.view()[samples.index(k), :] = _vals(h_region, k)
    h.view()[samples.index(QCDDY_KEY), :] = qcd_dy_vals
    h.view()[samples.index(DATA_KEY), :] = _vals(h_region, DATA_KEY)

    return h, bg_keys


def plot_closure(
    templates: dict,
    res: dict,
    channel: str,
    year_label: str,
    out: Path | None = None,
):
    """Per-sample VR-pass closure: each MC background stacked separately, the data-driven
    QCD+DY prediction (TF x fail) on top, compared to data with a data/pred ratio panel."""
    h, bg_keys = build_pred_hist(templates, res)

    # add_cms_label only knows single years or 'all'; use 'all' when several are summed.
    cms_year = year_label if year_label in LUMI else "all"

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)

    plotting.ratioHistPlot(
        h,
        cms_year,
        CHANNELS[channel],
        sig_keys=[],
        bg_keys=bg_keys,
        data_err=True,
        region_label=f"{CHANNELS[channel].label} VR Pass",
        name=str(out) if out is not None else "",
        show=False,
        ratio_ylims=[0, 2],
        cmslabel="Work in progress",
        # blind_region =SHAPE_VAR["blind_window"],
        leg_args={"fontsize": 16, "ncol": 2},
    )
    if out is not None:
        print(f"Saved closure plot to {out}")


def plot_fail(templates: dict, channel: str, year_label: str, out: Path):
    """VR-fail data vs the non-QCD/DY MC stack plus the data-driven QCD+DY (data - MC).
    Shows the stats and composition of the region the QCD+DY shape is taken from; the
    stack equals data by construction, so the ratio panel is 1 (except where QCD+DY < 0)."""
    hf = templates["fail"]
    h, bg_keys = _stack_hist(hf, qcd_dy_shape(hf))

    cms_year = year_label if year_label in LUMI else "all"

    out.parent.mkdir(parents=True, exist_ok=True)
    plotting.ratioHistPlot(
        h,
        cms_year,
        CHANNELS[channel],
        sig_keys=[],
        bg_keys=bg_keys,
        data_err=True,
        region_label=f"{CHANNELS[channel].label} VR Fail",
        name=str(out),
        show=False,
        ratio_ylims=[0, 3],
        cmslabel="Work in progress",
        leg_args={"fontsize": 16, "ncol": 2},
    )
    print(f"Saved VR-fail plot to {out}")


def _chi2(res: dict) -> tuple[float, int]:
    """Simple chi2 of data vs prediction using Poisson data errors (bins with data>0)."""
    data, pred = res["data_pass"], res["pred_pass"]
    mask = data > 0
    chi2 = np.sum((data[mask] - pred[mask]) ** 2 / data[mask])
    return float(chi2), int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", required=True, type=Path)
    parser.add_argument("--year", required=True, nargs="+", help="one or more years (summed)")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--signal", default="ggfbbtt")
    parser.add_argument("--bmin", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="output plot path (png)")
    parser.add_argument(
        "--plot-fail",
        action="store_true",
        help="also plot VR-fail data vs MC + data-driven QCD+DY, saved as <out stem>_fail<suffix>",
    )
    args = parser.parse_args()

    templates = load_vr_templates(
        args.template_dir, args.year, args.channel, args.signal, args.bmin
    )
    year_label = "+".join(args.year)

    res = run_closure(templates)
    chi2, ndf = _chi2(res)

    print(f"\nVR closure: channel={args.channel} year={year_label} bmin={args.bmin}")
    print(f"  non-QCD/DY MC subtracted: {NONQCDDY_BG_KEYS}")
    print(
        f"  VR-pass data={res['data_pass'].sum():.0f}  pred={res['pred_pass'].sum():.0f}  "
        f"qcd_eff={res['qcd_eff']:.3e}"
    )
    print(f"  chi2/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}")
    print("  per-bin TF (mass dependence -> SR residual):")
    for m, tf in zip(res["centers"], res["tf_perbin"]):
        print(f"    m={m:6.1f}  TF={tf:.3e}")

    out = args.out or (
        args.template_dir
        / vr_subdir(args.template_dir)
        / f"closure_{args.channel}_{year_label}_bmin{args.bmin}.png"
    )
    plot_closure(templates, res, channel=args.channel, year_label=year_label, out=out)

    if args.plot_fail:
        print(f"  VR-fail data={_vals(templates['fail'], DATA_KEY).sum():.0f}")
        fail_out = out.with_name(f"{out.stem}_fail{out.suffix}")
        plot_fail(templates, channel=args.channel, year_label=year_label, out=fail_out)


if __name__ == "__main__":
    main()
