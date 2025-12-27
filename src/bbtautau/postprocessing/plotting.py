"""
Plotting functions for bbtautau.

Enhanced plotting functions with data blinding capability for signal regions.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from boostedhh import hh_vars, plotting
from boostedhh.hh_vars import data_key
from hist import Hist

from bbtautau.postprocessing.bbtautau_types import FOM, Channel
from bbtautau.postprocessing.Samples import SAMPLES

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


def _setup_optimization_plot(
    years: list[str],
) -> tuple[plt.Figure, plt.Axes]:
    """Helper function to set up common plot elements for optimization plots.

    Returns:
        fig, ax: Figure and axes objects with CMS styling applied
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

    return fig, ax


def _finalize_optimization_plot(
    ax: plt.Axes,
    channel: Channel,
    foms: list[FOM],
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Helper function to finalize optimization plots with legend, text, and save/show.

    Args:
        ax: Axes object to finalize
        channel: Channel object for label
        foms: List of FOM objects for label
        save_path: Optional path to save plot
        show: Whether to show plot interactively
    """
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


def plot_optimization_thresholds(
    results: dict[str, dict[str, dict]],
    years: list[str],
    b_min_vals: list[int],
    foms: list[FOM],
    channel: Channel,
    save_path: Path | None = None,
    show: bool = False,
    bb_disc_label: str = "Xbb",
    tt_disc_label: str = "Xtt",
) -> None:
    """
    Plot optimization results with tagger thresholds on axes and signal yield as color.

    This is the original plotting variant that shows:
    - X axis: Xbb vs QCD tagger thresholds
    - Y axis: Xtt vs QCDTop tagger thresholds
    - Color: Signal yield
    - Points: Optimal cuts for different B_min constraints

    Args:
        results: Dictionary of optimization results from grid_search_opt.
            Structure: {fom_name: {f"Bmin={bmin}": {optimum_dict}}}
        years: List of years used in optimization
        b_min_vals: List of B_min values used
        foms: List of FOM objects
        channel: Channel object with label
        save_path: Optional path to save plot
        show: Whether to show plot
        bb_disc_label: Label for bb discriminator axis
        tt_disc_label: Label for tt discriminator axis
    """
    fig, ax = _setup_optimization_plot(years)

    colors = ["b", "purple", "r", "orange"]
    markers = ["x", "o", "s", "D"]

    sigmap = None
    first_optimum = None

    for B_min, c, m in zip(b_min_vals, colors, markers):
        fom_results = results.get(foms[0].name, {})
        optimum = fom_results.get(f"Bmin={B_min}")
        if optimum is None:
            continue

        if first_optimum is None:
            first_optimum = optimum
            # Get signal map - use sig_pass if available
            sig_map = optimum.get("sig_pass")
            if sig_map is not None and hasattr(sig_map, "shape") and len(sig_map.shape) == 2:
                # Get threshold grids (BBcut, TTcut are stored during threshold-mode optimization)
                bb_grid = optimum.get("BBcut")
                tt_grid = optimum.get("TTcut")
                if bb_grid is not None and tt_grid is not None:
                    sigmap = ax.pcolormesh(bb_grid, tt_grid, sig_map, cmap="viridis")

        # Get optimal cuts
        bb_cut_opt = optimum.get("TXbb_opt")
        tt_cut_opt = optimum.get("TXtt_opt")
        sel_B_min = optimum.get("sel_B_min")

        if bb_cut_opt is None or tt_cut_opt is None:
            continue

        if B_min == 1:  # B_min=1 is global optimum
            ax.scatter(
                bb_cut_opt,
                tt_cut_opt,
                color=c,
                label="Global optimum",
                marker=m,
                s=80,
                zorder=10,
            )
        else:
            ax.contour(
                bb_grid,
                tt_grid,
                ~sel_B_min,
                colors=c,
                linestyles="dashdot",
            )
            ax.scatter(
                bb_cut_opt,
                tt_cut_opt,
                color=c,
                label=f"Optimum $B\\geq {B_min}$",
                marker=m,
                s=80,
                zorder=10,
            )

    ax.set_xlabel(f"{bb_disc_label} score")
    ax.set_ylabel(f"{tt_disc_label} score")

    if sigmap is not None:
        cbar = plt.colorbar(sigmap, ax=ax)
        cbar.set_label("Signal yield")

    _finalize_optimization_plot(ax, channel, foms, save_path, show)


def plot_optimization_sig_eff(
    results: dict[str, dict[str, dict]],
    years: list[str],
    b_min_vals: list[int],
    foms: list[FOM],
    channel: Channel,
    save_path: Path | None = None,
    show: bool = False,
    use_log_scale: bool = False,
    clip_value: float | None = 100,
    bb_disc_label: str = "Xbb",
    tt_disc_label: str = "Xtt",
) -> None:
    """
    Plot optimization results with signal efficiency on axes and FOM values as color.

    This is the new plotting variant that shows:
    - X axis: Xbb signal efficiency (0-1)
    - Y axis: Xtt signal efficiency (0-1)
    - Color: FOM values (lower is better)
    - Points: Optimal signal efficiency cuts for different B_min constraints

    Args:
        results: Dictionary of optimization results from grid_search_opt.
            Structure: {fom_name: {f"Bmin={bmin}": {optimum_dict}}}
            Expected keys in optimum_dict:
                - BBcut_sig_eff: 2D array of bb signal efficiency grid
                - TTcut_sig_eff: 2D array of tt signal efficiency grid
                - fom_map: 2D array of FOM values
                - sig_eff_cuts: tuple (bb_sig_eff, tt_sig_eff) at optimum
                - TXbb_opt, TXtt_opt: optimal threshold cuts
        years: List of years used in optimization
        b_min_vals: List of B_min values used
        foms: List of FOM objects with .name and .label attributes
        channel: Channel object with .label attribute
        save_path: Optional path to save plot (saves .pdf and .png)
        show: Whether to show plot interactively
        use_log_scale: Whether to use logarithmic color scaling for FOM values
        clip_value: Maximum FOM value when clipping (default: 100, None to disable)
        bb_disc_label: Label for bb discriminator axis
        tt_disc_label: Label for tt discriminator axis
    """
    fig, ax = _setup_optimization_plot(years)

    colors = ["b", "purple", "r", "orange"]
    markers = ["x", "o", "s", "D"]

    fommap = None
    first_optimum = None

    for B_min, c, m in zip(b_min_vals, colors, markers):
        fom_results = results.get(foms[0].name, {})
        optimum = fom_results.get(f"Bmin={B_min}")
        if optimum is None:
            continue

        if first_optimum is None:
            first_optimum = optimum

            # Get FOM map and signal efficiency grids from dict
            fom_map_raw = optimum.get("fom_map")
            bb_sig_eff_grid = optimum.get("BBcut_sig_eff")
            tt_sig_eff_grid = optimum.get("TTcut_sig_eff")

            if (
                fom_map_raw is not None
                and bb_sig_eff_grid is not None
                and tt_sig_eff_grid is not None
            ):
                # Prepare FOM data based on options
                fom_data = fom_map_raw.copy()

                if clip_value is not None:
                    # Clip values above max_fom_value
                    fom_data = np.clip(fom_data, None, clip_value)

                # Set up normalization for log scale if requested
                norm = None
                if use_log_scale:
                    from matplotlib.colors import LogNorm

                    vmin = np.nanmin(fom_data[fom_data > 0]) if np.any(fom_data > 0) else 1e-6
                    vmax = np.nanmax(fom_data)
                    if vmin < vmax:
                        norm = LogNorm(vmin=vmin, vmax=vmax)

                # Plot FOM values on signal efficiency grid
                fommap = ax.pcolormesh(
                    bb_sig_eff_grid,
                    tt_sig_eff_grid,
                    fom_data,
                    cmap="viridis_r",  # Reverse colormap since lower FOM is better
                    norm=norm,
                )

        # Get optimal signal efficiency cuts from dict
        sig_eff_cuts = optimum.get("sig_eff_cuts")
        sel_B_min = optimum.get("sel_B_min")

        if sig_eff_cuts is None:
            # Fall back to threshold cuts if sig_eff_cuts not available
            continue

        bb_sig_eff_opt, tt_sig_eff_opt = sig_eff_cuts

        if B_min == 1:  # B_min=1 is global optimum
            ax.scatter(
                bb_sig_eff_opt,
                tt_sig_eff_opt,
                color=c,
                label="Global optimum",
                marker=m,
                s=80,
                zorder=10,
            )
        else:
            ax.contour(
                bb_sig_eff_grid,
                tt_sig_eff_grid,
                ~sel_B_min,
                colors=c,
                linestyles="dashdot",
            )
            ax.scatter(
                bb_sig_eff_opt,
                tt_sig_eff_opt,
                color=c,
                label=f"Optimum $B\\geq {B_min}$",
                marker=m,
                s=80,
                zorder=10,
            )

    ax.set_xlabel(f"{bb_disc_label} $\\epsilon_{{sig}}$")
    ax.set_ylabel(f"{tt_disc_label} $\\epsilon_{{sig}}$")

    if fommap is not None:
        cbar = plt.colorbar(fommap, ax=ax)
        cbar.set_label("FOM value")

    _finalize_optimization_plot(ax, channel, foms, save_path, show)
