"""
Plotting functions for bbtautau.

Authors: Raghav Kansal, add your names
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
from boostedhh import plotting
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
]
sample_label_map = {s: SAMPLES[s].label for s in SAMPLES}
sample_label_map[data_key] = "Data"

BG_COLOURS = {
    "qcd": "darkblue",
    "ttbarhad": "brown",
    "ttbarsl": "lightblue",
    "ttbarll": "lightgray",
    "dyjets": "orange",
    "wjets": "yellow",
    "zjets": "gray",
    "hbb": "beige",
}


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

    plotting.ratioHistPlot(
        hists,
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
