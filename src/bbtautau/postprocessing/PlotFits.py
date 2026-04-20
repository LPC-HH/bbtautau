#!/usr/bin/env python3
"""
Plot post-fit shapes from FitShapes.root.

Converts fit output histograms into publication-ready ratio plots for each
region and shape variable. Optionally produces QCD transfer factor plots.

Usage:
    # Basic usage with default paths
    python -m bbtautau.postprocessing.PlotFits \\
        --folder 25Dec15-ggf-bmin5 \\
        --fit-shapes /path/to/cards/25Dec15-ggf-bmin5/FitShapes.root \\
        --templates-dir templates/25Dec15-ggf-bmin5/ \\
        --plot-dir /path/to/plots/PostFit/25Dec15-ggf-bmin5

    # With custom bmin and signal
    python -m bbtautau.postprocessing.PlotFits \\
        --folder 25Dec15-ggf-bmin5 \\
        --fit-shapes cards/25Dec15-ggf-bmin5/FitShapes.root \\
        --templates-dir templates/25Dec15-ggf-bmin5/ \\
        --plot-dir plots/PostFit/25Dec15-ggf-bmin5 \\
        --bmin 8 \\
        --signal ggfbbtt

    # Plot QCD transfer factors only
    python -m bbtautau.postprocessing.PlotFits \\
        --folder 25Dec15-ggf-bmin5 \\
        --fit-shapes cards/25Dec15-ggf-bmin5/FitShapes.root \\
        --plot-dir plots/PostFit/25Dec15-ggf-bmin5 \\
        --qcd-tf-only

    # For allsigsFitShapes.root (ggfbbtthepass/vbfbbtthepass naming)
    python -m bbtautau.postprocessing.PlotFits \\
        --fit-shapes cards/SM_nTF_per_channel/bmin_10/allsigsFitShapes.root \\
        --plot-dir plots/PostFit/SM_nTF_per_channel/bmin_10 \\
        --sig-region ggfbbtt

    # Plot both ggf and vbf from combined fit
    python -m bbtautau.postprocessing.PlotFits \\
        --fit-shapes cards/.../allsigsFitShapes.root \\
        --plot-dir plots/PostFit \\
        --sig-region all

Note: If the FitShapes file was not properly closed (e.g. combine job interrupted),
the subdirectory contents may be missing. Re-run PostFitShapesFromWorkspace to
produce a valid file.
"""

from __future__ import annotations

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

from boostedhh.hh_vars import data_key, years
from hist import Hist

from bbtautau.postprocessing.datacardHelpers import sum_templates
from bbtautau.postprocessing.postprocessing import shape_vars
from bbtautau.postprocessing import plotting
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.userConfig import SHAPE_VAR

try:
    import uproot
except ImportError:
    uproot = None
try:
    import ROOT
except ImportError:
    ROOT = None

# (name in templates, name in cards)
HIST_LABEL_MAP_INVERSE = OrderedDict(
    [
        ("qcddy", "CMS_bbtautau_boosted_qcd_datadriven"),
        ("ttbarsl", "ttbarsl"),
        ("ttbarll", "ttbarll"),
        ("ttbarhad", "ttbarhad"),
        ("wjets", "wjets"),
        ("zjets", "zjets"),
        ("hbb", "hbb"),
        (data_key, "data_obs"),
    ]
)
HIST_LABEL_MAP = {val: key for key, val in HIST_LABEL_MAP_INVERSE.items()}

PBG_KEYS = ["qcddy", "ttbarhad", "ttbarsl", "ttbarll", "wjets", "zjets", "hbb"]


def _samples_for_sig_region(sig_region: str | None) -> list[str]:
    """Build samples list including the appropriate signal key."""
    sig_keys = [sig_region] if sig_region else ["ggfbbtt"]
    return PBG_KEYS + sig_keys + [data_key]


SHAPES = {
    "prefit": "Pre-Fit",
    "postfit": "B-only Post-Fit",
}


def get_pre_templates(templates_dir: Path):
    """Load and sum pre-fit templates across years. Returns {} if no templates found."""
    templates_dict = {}
    for year in years:
        pkl_path = templates_dir / f"{year}_templates.pkl"
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as f:
            templates_dict[year] = pickle.load(f)
    if not templates_dict:
        return {}
    return sum_templates(templates_dict, years)


def build_selection_regions(sig_region: str | None = None):
    """Build region key -> label mapping.

    If sig_region is set (e.g. ggfbbtt, vbfbbtt), regions are {sig_region}hepass, etc.
    Otherwise legacy format: hepass, hefail, etc.
    """
    selection_regions = {}
    prefix = f"{sig_region}" if sig_region else ""
    for channel in CHANNELS.values():
        for passfail, plabel in [("pass", " Pass"), ("fail", " Fail")]:
            region = f"{prefix}{channel.key}{passfail}"
            selection_regions[region] = f"{channel.label}{plabel}"
    return selection_regions


def _th1_to_values(errors: bool = False):
    """Return a function that extracts values or errors from a TH1."""

    def extract(hist_obj):
        if hasattr(hist_obj, "values"):  # uproot/boost-histogram
            return hist_obj.errors() if errors else hist_obj.values()
        # ROOT TH1
        n = hist_obj.GetNbinsX()
        arr = np.zeros(n)
        for i in range(1, n + 1):
            arr[i - 1] = hist_obj.GetBinError(i) if errors else hist_obj.GetBinContent(i)
        return arr

    return extract


def _channel_key_from_region(region: str, sig_region: str | None) -> str:
    """Extract channel key (he, hh, hm) from region name."""
    if sig_region and region.startswith(sig_region):
        rest = region[len(sig_region) :]  # e.g. hepass, hefail
        return rest[:2]  # he, hh, hm
    return region[:2]  # legacy: hepass -> he


def _load_region_from_root(root_file, region: str, shape: str) -> tuple[dict, np.ndarray | None]:
    """Load one region's histograms from ROOT TFile. Returns (sample_arrays, bgerr)."""
    dir_path = f"{region}_{shape}"
    sample_arrays = {}
    for key, file_key in HIST_LABEL_MAP_INVERSE.items():
        obj = root_file.Get(f"{dir_path}/{file_key}")
        if obj and obj.InheritsFrom("TH1"):
            n = obj.GetNbinsX()
            vals = np.array([obj.GetBinContent(i) for i in range(1, n + 1)])
            sample_arrays[key] = vals
    obj = root_file.Get(f"{dir_path}/data_obs")
    if obj and obj.InheritsFrom("TH1"):
        n = obj.GetNbinsX()
        sample_arrays[data_key] = np.nan_to_num(
            [obj.GetBinContent(i) for i in range(1, n + 1)]
        )
    # TotalSig -> use region prefix (ggfbbtt or vbfbbtt) as key
    for sig in ["ggfbbtt", "vbfbbtt"]:
        if region.startswith(sig):
            obj = root_file.Get(f"{dir_path}/TotalSig")
            if obj and obj.InheritsFrom("TH1"):
                n = obj.GetNbinsX()
                sample_arrays[sig] = np.array(
                    [obj.GetBinContent(i) for i in range(1, n + 1)]
                )
            break
    bgerr = None
    obj = root_file.Get(f"{dir_path}/TotalBkg")
    if obj and obj.InheritsFrom("TH1"):
        n = obj.GetNbinsX()
        errs = np.array([obj.GetBinError(i) for i in range(1, n + 1)])
        vals = np.array([obj.GetBinContent(i) for i in range(1, n + 1)])
        bgerr = np.minimum(errs, vals)
    return sample_arrays, bgerr


def load_hists_from_fit(
    file_or_path,
    templates_dir: Path,
    signal: str,
    sig_region: str | None = None,
    available_shapes: list[str] | None = None,
):
    """Load prefit/postfit histograms from FitShapes.root.

    file_or_path: uproot file handle or path (uses ROOT fallback if uproot fails).
    sig_region: e.g. ggfbbtt or vbfbbtt for files with {sig_region}hepass naming.
    available_shapes: list of shapes present in file (prefit, postfit). Auto-detected if None.
    """
    selection_regions = build_selection_regions(sig_region)
    shapes_to_use = available_shapes or list(SHAPES.keys())

    # Try uproot first, fall back to ROOT
    use_root = False
    root_file = None
    uproot_file = None

    if uproot and hasattr(file_or_path, "keys"):
        uproot_file = file_or_path
        if len(list(uproot_file.keys())) == 0:
            use_root = True
    elif isinstance(file_or_path, (str, Path)) and ROOT:
        uproot_file = uproot.open(str(file_or_path)) if uproot else None
        if uproot_file and len(list(uproot_file.keys())) == 0:
            uproot_file.close()
            uproot_file = None
        if uproot_file is None:
            use_root = True
            root_file = ROOT.TFile.Open(str(file_or_path))

    if use_root and root_file:
        # Discover shapes from ROOT file
        keys = root_file.GetListOfKeys()
        shapes_in_file = set()
        for i in range(keys.GetSize()):
            k = keys.At(i).GetName()
            if "_prefit" in k:
                shapes_in_file.add("prefit")
            elif "_postfit" in k:
                shapes_in_file.add("postfit")
        if available_shapes is None:
            shapes_to_use = [s for s in ["prefit", "postfit"] if s in shapes_in_file]
        if not shapes_to_use:
            shapes_to_use = ["prefit"]  # fallback

    hists = {}
    bgerrs = {}

    for shape in shapes_to_use:
        if shape not in SHAPES:
            continue
        hists[shape] = {}
        bgerrs[shape] = {}

        samples_list = _samples_for_sig_region(sig_region)
        for region in selection_regions:
            h = Hist(
                hist.axis.StrCategory(samples_list, name="Sample"),
                *[sv.axis for sv in shape_vars],
                storage="double",
            )

            if use_root and root_file:
                sample_arrays, bgerr = _load_region_from_root(
                    root_file, region, shape
                )
                if not sample_arrays:
                    continue
                for key in samples_list:
                    if key in sample_arrays:
                        idx = np.where(np.array(list(h.axes[0])) == key)[0][0]
                        h.view(flow=False)[idx, :] = sample_arrays[key]
                if bgerr is not None:
                    bgerrs[shape][region] = bgerr
                hists[shape][region] = h
            else:
                try:
                    templates = uproot_file[f"{region}_{shape}"]
                except (KeyError, TypeError):
                    continue
                for key, file_key in HIST_LABEL_MAP_INVERSE.items():
                    if key != data_key and file_key in templates:
                        idx = np.where(np.array(list(h.axes[0])) == key)[0][0]
                        h.view(flow=False)[idx, :] = templates[file_key].values()
                # Signal from TotalSig in fit output, or pre-fit templates fallback
                for key in samples_list:
                    if key not in HIST_LABEL_MAP_INVERSE and key != data_key:
                        filled = False
                        if region.startswith(key):
                            try:
                                total_sig = templates["TotalSig"]
                                vals = total_sig.values()
                                idx = np.where(np.array(list(h.axes[0])) == key)[0][0]
                                h.view(flow=False)[idx, :] = vals
                                filled = True
                            except (KeyError, TypeError, AttributeError):
                                pass
                        if not filled:
                            ch = _channel_key_from_region(region, sig_region)
                            try:
                                sig_pre = get_pre_templates(
                                    Path(templates_dir) / ch / signal
                                )
                                if sig_pre:
                                    pf = "pass" if "pass" in region else "fail"
                                    idx = np.where(
                                        np.array(list(h.axes[0])) == key
                                    )[0][0]
                                    h.view(flow=False)[idx, :] = sig_pre[pf][
                                        key + ch, ...
                                    ].values()
                            except (FileNotFoundError, KeyError):
                                pass
                data_key_index = np.where(np.array(list(h.axes[0])) == data_key)[0][0]
                h.view(flow=False)[data_key_index, :] = np.nan_to_num(
                    templates[HIST_LABEL_MAP_INVERSE[data_key]].values()
                )
                bgerrs[shape][region] = np.minimum(
                    templates["TotalBkg"].errors(), templates["TotalBkg"].values()
                )
                hists[shape][region] = h

    if root_file:
        root_file.Close()
    if uproot_file and use_root and hasattr(uproot_file, "close"):
        uproot_file.close()

    return hists, bgerrs, selection_regions


def plot_fits(
    hists,
    bgerrs,
    selection_regions,
    plot_dir: Path,
    prelim: bool = True,
    sig_region: str | None = None,
):
    """Generate ratio plots for each shape, region, and variable."""
    plabel = "Preliminary" if prelim else ""
    pplotdir = "preliminary" if prelim else "final"
    out_dir = plot_dir / pplotdir
    out_dir.mkdir(exist_ok=True, parents=True)

    sig_keys = [sig_region] if sig_region else ["ggfbbtt"]
    sig_scale_dict = {k: 100 for k in sig_keys}

    for shape in SHAPES:
        if shape not in hists:
            continue
        for region, region_label in selection_regions.items():
            if region not in hists[shape]:
                continue
            pass_region = "pass" in region
            ch_key = _channel_key_from_region(region, sig_region)
            for shape_var in shape_vars:
                plot_params = {
                    "hists": hists[shape][region],
                    "sig_keys": sig_keys,
                    "bg_keys": PBG_KEYS,
                    "bg_err": bgerrs[shape].get(region),
                    "data_err": True,
                    "sig_scale_dict": sig_scale_dict if pass_region else None,
                    "show": False,
                    "year": "2022-2023",
                    "region_label": region_label,
                    "name": str(
                        out_dir / f"{pplotdir}_{shape}_{region}_{shape_var.var}.pdf"
                    ),
                    "ratio_ylims": [0, 2],
                    "cmslabel": plabel,
                    "leg_args": {"fontsize": 22, "ncol": 2},
                    "channel": CHANNELS[ch_key],
                    "blind_region": SHAPE_VAR["blind_window"] if pass_region else None,
                }
                plotting.ratioHistPlot(**plot_params)


def plot_qcd_transfer_factors(
    hists, selection_regions, plot_dir: Path, sig_region: str | None = None
):
    """Plot QCD transfer factors (pass/fail) for each pass region."""
    if "postfit" not in hists:
        return {}
    plt.style.use(hep.style.CMS)
    hep.style.use("CMS")
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))

    tfs = {}

    for region, region_label in selection_regions.items():
        if region.endswith("fail"):
            continue
        # fail_region: ggfbbtthepass -> ggfbbtthefail, hepass -> hefail
        if sig_region and region.startswith(sig_region):
            fail_region = region.replace("pass", "fail")
        else:
            fail_region = region[:2] + "fail"
        if fail_region not in hists["postfit"]:
            continue

        tf = (
            hists["postfit"][region]["qcddy", ...]
            / hists["postfit"][fail_region]["qcddy", ...]
        )
        tfs[region] = tf

        tf_vals = tf.values()
        max_val = np.nanmax(tf_vals) if np.any(np.isfinite(tf_vals)) else 1e-3
        ymax = 1.2 * max_val

        hep.histplot(tf)
        plt.title(f"{region_label} Region")
        plt.ylabel("QCD Transfer Factor")
        plt.xlim([50, 250])
        plt.ylim([0, ymax])
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        out_path = plot_dir / f"{region}_QCDTF.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    return tfs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot post-fit shapes from FitShapes.root",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="25Dec15-ggf-bmin5",
        help="Folder name for paths (e.g. 25Dec15-ggf-bmin5)",
    )
    parser.add_argument(
        "--fit-shapes",
        type=Path,
        required=True,
        help="Path to FitShapes.root from combine fit",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="Path to templates dir (for signal pre-fit fallback). Default: templates/{folder}/",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Default: plots/PostFit/{folder}/",
    )
    parser.add_argument(
        "--bmin",
        type=int,
        default=5,
        help="b-tag minimum (used in path construction if paths are relative)",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="ggfbbtt",
        help="Signal sample key for pre-fit template fallback",
    )
    parser.add_argument(
        "--sig-region",
        type=str,
        choices=["ggfbbtt", "vbfbbtt", "all"],
        default=None,
        help="Signal region prefix for FitShapes with ggfbbtthepass/vbfbbtthepass naming. "
        "Use 'all' to plot both ggf and vbf. Required for allsigsFitShapes.root.",
    )
    parser.add_argument(
        "--preliminary",
        action="store_true",
        default=True,
        help="Add 'Preliminary' CMS label (default: True)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Also produce final (non-preliminary) plots",
    )
    parser.add_argument(
        "--qcd-tf-only",
        action="store_true",
        help="Only plot QCD transfer factors, skip main fit plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    folder = args.folder
    fit_shapes_path = args.fit_shapes
    templates_dir = args.templates_dir or Path(f"templates/{folder}/bmin_{args.bmin}/")
    plot_dir = args.plot_dir or Path(f"plots/PostFit/{folder}/bmin_{args.bmin}/")

    plot_dir.mkdir(exist_ok=True, parents=True)

    if not fit_shapes_path.exists():
        raise FileNotFoundError(f"FitShapes.root not found: {fit_shapes_path}")

    sig_regions = (
        ["ggfbbtt", "vbfbbtt"]
        if args.sig_region == "all"
        else ([args.sig_region] if args.sig_region else [None])
    )

    for sig_region in sig_regions:
        sub_plot_dir = (plot_dir / sig_region) if sig_region else plot_dir
        sub_plot_dir.mkdir(exist_ok=True, parents=True)

        hists, bgerrs, selection_regions = load_hists_from_fit(
            fit_shapes_path,
            templates_dir,
            args.signal,
            sig_region=sig_region,
        )

        if not hists or all(not v for v in hists.values()):
            print(
                f"Warning: No histograms loaded for sig_region={sig_region}. "
                "File may be corrupted (not properly closed). Re-run PostFitShapesFromWorkspace."
            )
            continue

        if not args.qcd_tf_only:
            plot_fits(
                hists,
                bgerrs,
                selection_regions,
                sub_plot_dir,
                prelim=args.preliminary,
                sig_region=sig_region,
            )
            if args.final:
                plot_fits(
                    hists,
                    bgerrs,
                    selection_regions,
                    sub_plot_dir,
                    prelim=False,
                    sig_region=sig_region,
                )

        plot_qcd_transfer_factors(
            hists, selection_regions, sub_plot_dir, sig_region=sig_region
        )

    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
