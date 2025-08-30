"""
Plotting functions for bbtautau.

Enhanced plotting functions with data blinding capability for signal regions.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from boostedhh import hh_vars, plotting
from boostedhh.hh_vars import data_key
from hist import Hist
from Samples import SAMPLES

from bbtautau.bbtautau_utils import Channel

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))


bg_order = [
    "hbb",
    "dyjets",
    "ttbarll",
    "wjets",
    "zjets",
    "ttbarsl",
    "ttbarhad",
    "qcd",
    "qcddy",
]
sample_label_map = {s: SAMPLES[s].label for s in SAMPLES}
sample_label_map[data_key] = "Data"
sample_label_map["qcddy"] = "QCD + DYJets"

BG_COLOURS = {
    "qcd": "darkblue",
    "qcddy": "darkblue",
    "ttbarhad": "brown",
    "ttbarsl": "lightblue",
    "ttbarll": "lightgray",
    "dyjets": "orange",
    "wjets": "yellow",
    "zjets": "gray",
    "hbb": "beige",
}


def create_blinded_histogram(hists: Hist, blind_region: list, axis=0):
    """
    Create a copy of histogram with data points masked in the specified region.

    Args:
        hists: Input histogram
        blind_region: [min_value, max_value] range to blind
        axis: Which axis to blind on (default: 0, the first variable axis)

    Returns:
        Modified histogram with masked data
    """
    # Create a copy of the histogram to avoid modifying the original
    masked_hists = hists.copy()

    if axis > 0:
        raise Exception("not implemented > 1D blinding yet")

    bins = masked_hists.axes[axis + 1].edges
    lv = int(np.searchsorted(bins, blind_region[0], "right"))
    rv = int(np.searchsorted(bins, blind_region[1], "left") + 1)

    # Find data sample index
    sample_names = list(masked_hists.axes[0])
    if data_key in sample_names:
        data_key_index = sample_names.index(data_key)

        # Create a mask for the blinded region
        mask = np.ones(len(bins) - 1, dtype=bool)
        mask[lv:rv] = False

        print(type(masked_hists.view(flow=True)))

        masked_hists.view(flow=True)[data_key_index][lv:rv] = np.nan

    return masked_hists


def ratioHistPlot(
    hists: Hist,
    year: str,
    channel: Channel,
    sig_keys: list[str],
    bg_keys: list[str],
    plot_ratio: bool = True,
    plot_significance: bool = False,
    cutlabel: str = "",
    region_label: str = "",
    name: str = "",
    show: bool = False,
    blind_region: list = None,
    **kwargs,
):

    if plot_significance:
        fig, axraxsax = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.1},
            sharex=True,
        )
        (ax, rax, sax) = axraxsax
    elif plot_ratio:
        fig, axraxsax = plt.subplots(
            2,
            1,
            figsize=(12, 14),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
            sharex=True,
        )
        (ax, rax) = axraxsax
    else:
        fig, axraxsax = plt.subplots(1, 1, figsize=(12, 11))
        ax = axraxsax

    # Apply blinding if specified
    plot_hists = hists
    if blind_region is not None:
        plot_hists = create_blinded_histogram(hists, blind_region)

    plotting.ratioHistPlot(
        plot_hists,
        year,
        sig_keys,
        bg_keys,
        bg_order=bg_order,
        bg_colours=BG_COLOURS,
        sample_label_map=sample_label_map,
        plot_significance=plot_significance,
        axraxsax=axraxsax,
        **kwargs,
    )

    ax.text(
        0.03,
        0.92,
        region_label if region_label else channel.label,
        transform=ax.transAxes,
        fontsize=24,
        fontproperties="Tex Gyre Heros:bold",
    )

    if cutlabel:
        ax.text(
            0.02,
            0.8,
            cutlabel,
            transform=ax.transAxes,
            fontsize=14,
        )

    if len(name):
        plt.savefig(name, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def plot_optimization_thresholds(
    results, years, b_min_vals, foms, channel, save_path=None, show=False
):
    """
    Plot optimization results with tagger thresholds on axes and signal yield as color.

    This is the original plotting variant that shows:
    - X axis: Xbb vs QCD tagger thresholds
    - Y axis: Xtt vs QCDTop tagger thresholds
    - Color: Signal yield
    - Points: Optimal cuts for different B_min constraints

    Args:
        results: Dictionary of optimization results from grid_search_opt
        years: List of years used in optimization
        b_min_vals: List of B_min values used
        foms: List of FOM objects
        channel: Channel object with label
        save_path: Optional path to save plot
        show: Whether to show plot
    """
    plt.rcdefaults()
    plt.style.use(hep.style.CMS)
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))

    hep.cms.label(
        ax=ax,
        label="Work in Progress",
        data=True,
        year="2022-23" if years == hh_vars.years else "+".join(years),
        com="13.6",
        fontsize=13,
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
    )

    colors = ["orange", "r", "purple", "b"]
    markers = ["x", "o", "s", "D"]

    i = 0
    for B_min, c, m in zip(b_min_vals, colors, markers):
        optimum = results[foms[0].name][f"Bmin={B_min}"]  # only plot the first FOM
        if i == 0:
            # Assuming all optimums have same cut and signal maps
            sigmap = ax.pcolormesh(optimum.BBcut, optimum.TTcut, optimum.sig_map, cmap="viridis")
            i += 1

        if B_min == 1:
            ax.scatter(optimum.cuts[0], optimum.cuts[1], color=c, label="Global optimum", marker=m)
        else:
            ax.contour(
                optimum.BBcut,
                optimum.TTcut,
                ~optimum.sel_B_min,
                colors=c,
                linestyles="dashdot",
            )
            ax.scatter(
                optimum.cuts[0],
                optimum.cuts[1],
                color=c,
                label=f"Optimum $B\\geq {B_min}$",
                marker=m,
            )

    ax.set_xlabel(optimum.bb_disc_name + " score")
    ax.set_ylabel(optimum.tt_disc_name + " score")
    cbar = plt.colorbar(sigmap, ax=ax)
    cbar.set_label("Signal yield")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="lower left")

    text = channel.label + "\nFOM: " + foms[0].label

    ax.text(
        0.05,
        0.72,
        text,
        transform=ax.transAxes,
        fontsize=20,
        fontproperties="Tex Gyre Heros",
    )

    if save_path:
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_optimization_sig_eff(
    results,
    years,
    b_min_vals,
    foms,
    channel,
    save_path=None,
    show=False,
    use_log_scale=False,
    clip_value=100,
):
    """
    Plot optimization results with signal efficiency on axes and FOM values as color.

    This is the new plotting variant that shows:
    - X axis: Xbb signal efficiency (0-1)
    - Y axis: Xtt signal efficiency (0-1)
    - Color: FOM values (lower is better)
    - Points: Optimal signal efficiency cuts for different B_min constraints

    Args:
        results: Dictionary of optimization results from grid_search_opt_sig_eff
        years: List of years used in optimization
        b_min_vals: List of B_min values used
        foms: List of FOM objects
        channel: Channel object with label
        save_path: Optional path to save plot
        show: Whether to show plot
        use_log_scale: Whether to use logarithmic color scaling for FOM values
        clip_value: Maximum FOM value when clipping (default: 100)
    """
    plt.rcdefaults()
    plt.style.use(hep.style.CMS)
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))

    hep.cms.label(
        ax=ax,
        label="Work in Progress",
        data=True,
        year="2022-23" if years == hh_vars.years else "+".join(years),
        com="13.6",
        fontsize=13,
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
    )

    colors = ["b", "purple", "r", "orange"]
    markers = ["x", "o", "s", "D"]

    i = 0
    for B_min, c, m in zip(b_min_vals, colors, markers):
        optimum = results[foms[0].name][f"Bmin={B_min}"]  # only plot the first FOM
        if i == 0:
            # Prepare FOM data based on options
            fom_data = optimum.fom_map.copy()

            if clip_value:
                # Clip values above max_fom_value
                fom_data = np.clip(fom_data, None, clip_value)

            # Set up normalization for log scale if requested
            norm = None
            if use_log_scale:
                from matplotlib.colors import LogNorm

                # Use LogNorm for logarithmic color scaling
                norm = LogNorm(vmin=np.nanmin(fom_data), vmax=np.nanmax(fom_data))

            # Plot FOM values on signal efficiency grid
            fommap = ax.pcolormesh(
                optimum.BBcut_sig_eff,
                optimum.TTcut_sig_eff,
                fom_data,
                cmap="viridis_r",
                norm=norm,  # Reverse colormap since lower FOM is better
            )
            i += 1

        if B_min == 1:
            ax.scatter(
                optimum.sig_eff_cuts[0],
                optimum.sig_eff_cuts[1],
                color=c,
                label="Global optimum",
                marker=m,
            )
        else:
            # Note: sel_B_min is still in threshold space, need to map to sig eff space
            ax.contour(
                optimum.BBcut_sig_eff,
                optimum.TTcut_sig_eff,
                ~optimum.sel_B_min,
                colors=c,
                linestyles="dashdot",
            )
            ax.scatter(
                optimum.sig_eff_cuts[0],
                optimum.sig_eff_cuts[1],
                color=c,
                label=f"Optimum $B\\geq {B_min}$",
                marker=m,
            )

    ax.set_xlabel(optimum.bb_disc_name + " $\\epsilon_{sig}$")
    ax.set_ylabel(optimum.tt_disc_name + " $\\epsilon_{sig}$")

    cbar = plt.colorbar(fommap, ax=ax)
    cbar.set_label("FOM value")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="upper right")

    text = channel.label + "\n" + foms[0].label

    ax.text(
        0.05,
        0.82,
        text,
        transform=ax.transAxes,
        fontsize=20,
        fontproperties="Tex Gyre Heros",
    )

    if save_path:
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
