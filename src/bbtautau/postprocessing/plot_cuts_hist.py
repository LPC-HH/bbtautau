from __future__ import annotations

import argparse
from pathlib import Path

import hist
import matplotlib as mpl
import numpy as np
import pandas as pd
from boostedhh import hh_vars
from hist import Hist

from bbtautau.postprocessing.bdt_utils import compute_or_load_bdt_preds
from bbtautau.postprocessing.postprocessing import (
    base_filter,
    bbtautau_assignment,
    delete_columns,
    derive_lepton_variables,
    derive_variables,
    get_columns,
    leptons_assignment,
    load_samples,
    tt_filters,
)
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.postprocessing.utils import concatenate_years
from bbtautau.userConfig import BDT_EVAL_DIR, DATA_PATHS, MODEL_DIR

"""
Quick script: read cut thresholds from an optimization CSV, load data, apply cuts,
and produce histograms for given variables.

Features:
- Only use tt cut (Xtt), ignoring bb cut
- Plot each sample separately with different Bmin cuts overlaid in a single plot
- Create separate plots for each sample
- Save both PDF and PNG formats
"""

mpl.use("Agg")


def _apply_basic_processing(events_dict: dict, year: str) -> None:
    # Minimal processing similar to bdt.Trainer
    delete_columns(events_dict, year, channels=list(CHANNELS.values()))
    derive_variables(events_dict)
    bbtautau_assignment(events_dict, agnostic=True)
    leptons_assignment(events_dict, dR_cut=1.5)
    derive_lepton_variables(events_dict)


def plot_overlaid_cuts(
    h: Hist,
    cut_labels: list[str],
    years: list[str],
    region_label: str = None,
    title: str = None,
    colors: list = None,
    ylim: tuple = None,
    log: bool = False,
    cmslabel: str = "Preliminary",
    name: str = "",
    show: bool = False,
    legend_labels: list[str] = None,
):
    """
    Plot histogram with multiple cuts overlaid as line plots.

    Args:
        h: Histogram with (cut_axis, variable_axis) structure
        cut_labels: List of cut labels to plot (from first axis, used for indexing)
        years: List of years for CMS label
        region_label: Text label for region/sample info
        title: Plot title
        colors: List of colors for each cut
        ylim: Y-axis limits
        log: Use log scale
        cmslabel: CMS label text
        name: Path to save plot
        show: Whether to show plot
        legend_labels: Optional list of labels for legend (if None, uses cut_labels)
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    # Get colors
    if colors is None:
        colors = plt.cm.tab10.colors

    # Use legend_labels if provided, otherwise use cut_labels
    if legend_labels is None:
        legend_labels = cut_labels

    # Create plots per variable and per sample (with all Bmin cuts overlaid)
    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"

    # Check histogram contents
    print(f"\n  Plotting histogram with {len(cut_labels)} cuts:")
    total_sum = 0
    for cut_label in cut_labels:
        vals = h[cut_label, :].values()
        cut_sum = np.sum(vals)
        total_sum += cut_sum
        print(
            f"    {cut_label}: sum={cut_sum:.2f}, entries={len(vals)}, range=[{np.min(vals):.2f}, {np.max(vals):.2f}]"
        )

    if total_sum == 0:
        print("  WARNING: Histogram is empty! No events to plot.")

    # Plot each cut as a line
    for i, cut_label in enumerate(cut_labels):
        hep.histplot(
            h[cut_label, :],
            ax=ax,
            histtype="step",
            label=legend_labels[i],
            color=colors[i % len(colors)],
            linewidth=2.5,
            flow="none",
        )

    # Set labels and styling
    ax.set_ylabel("Events")
    ax.set_xlabel(h.axes[1].label if h.axes[1].label else h.axes[1].name)
    ax.legend(fontsize=20, loc="best")

    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
    else:
        ax.set_ylim(bottom=0)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.margins(x=0)

    # Add CMS label
    hep.cms.label(
        cmslabel,
        data=True,
        com="13.6",
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        year=year_str,
        ax=ax,
        loc=0,
    )

    # Add region label
    if region_label:
        ax.text(
            0.03,
            0.82 if not log else 0.65,
            region_label,
            transform=ax.transAxes,
            fontsize=20,
            fontproperties="Tex Gyre Heros:bold",
        )

    # Add title
    if title:
        ax.set_title(title, y=1.02)

    # Save plot
    if name:
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def build_hist_for_sample_with_cuts(
    sample,
    channel_key: str,
    varname: str,
    cuts_dict: dict,
    bins: int,
    x_min: float,
    x_max: float,
    use_bdt: bool = False,
) -> Hist:
    """
    Build histogram for a single sample with different Bmin cuts.
    Returns a histogram with axes: (Bmin, variable).
    """
    channel = CHANNELS[channel_key]
    taukey = channel.tagger_label

    # Create histogram with Bmin axis
    bmin_labels = [f"Bmin={bmin}" for bmin in cuts_dict]
    h = Hist(
        hist.axis.StrCategory(bmin_labels, name="Bmin"),
        hist.axis.Regular(bins, x_min, x_max, name=varname),
        storage="weight",
    )

    # Get variable for cut
    if use_bdt:
        xtt = sample.get_var(f"BDTScore{taukey}vsAll")
    else:
        xtt = sample.get_var(f"ttFatJetParTX{taukey}vsQCDTop")

    vals = sample.get_var(varname)
    weight = sample.get_var("finalWeight")

    print(f"  Total events before cuts: {len(vals)}")
    print(f"  Variable '{varname}' range: [{np.min(vals):.3f}, {np.max(vals):.3f}]")
    print(f"  Histogram bin range: [{x_min}, {x_max}]")
    print(f"  Cut variable range: [{np.min(xtt):.3f}, {np.max(xtt):.3f}]")

    # Check if data is outside histogram range
    n_outside = np.sum((vals < x_min) | (vals > x_max))
    if n_outside > 0:
        print(f"  WARNING: {n_outside}/{len(vals)} events outside histogram range!")

    # Fill histogram for each Bmin cut
    for bmin_value, cut_xtt in cuts_dict.items():
        sel = xtt >= cut_xtt

        # Apply selection to values and weights
        vals_sel = vals[sel]
        weight_sel = weight[sel]

        print(
            f"    Bmin={bmin_value}: cut={cut_xtt:.3f}, events passing={len(vals_sel)}, sum(weights)={np.sum(weight_sel):.2f}"
        )

        if len(vals_sel):
            h.fill(Bmin=f"Bmin={bmin_value}", **{varname: vals_sel}, weight=weight_sel)

    # Check histogram total
    total_events = np.sum([h[f"Bmin={bmin}", :].values().sum() for bmin in cuts_dict])
    print(f"  Total events in histogram: {total_events:.2f}")

    return h


def main():
    parser = argparse.ArgumentParser(description="Plot histograms for CSV-derived cuts")
    parser.add_argument("--csv", required=True, type=str, help="Path to optimization CSV file")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["all"],
        help="Years to include (e.g. 2022 2022EE) or 'all'",
    )
    parser.add_argument(
        "--channel",
        required=True,
        choices=list(CHANNELS.keys()),
        help="Channel to process (e.g. tauhhtauhh)",
    )
    parser.add_argument(
        "--signal-key",
        default="ggfbbtt",
        help="Signal key (base, without channel suffix)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        required=True,
        help="Variable names to histogram (per-event); if 2D, max over jets is used",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--xmin", type=float, default=None, help="Minimum x-axis value (auto if not specified)"
    )
    parser.add_argument(
        "--xmax", type=float, default=None, help="Maximum x-axis value (auto if not specified)"
    )
    parser.add_argument(
        "--auto-range", action="store_true", help="Automatically determine xmin/xmax from data"
    )
    parser.add_argument("--output-dir", type=str, default="plots/CutHists")
    parser.add_argument("--use-bdt", action="store_true", default=True)
    parser.add_argument(
        "--max-bmin-plots",
        type=int,
        default=4,
        help="Maximum number of Bmin cuts to plot (default: 4)",
    )
    parser.add_argument(
        "--bmin-step",
        type=int,
        default=2,
        help="Step size for selecting Bmin columns (default: 2, i.e., every other)",
    )
    parser.add_argument("--modelname", type=str, default="29July25_loweta_lowreg")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing trained models (parent of <modelname>)",
    )
    parser.add_argument(
        "--bdt-eval-dir",
        type=str,
        default=str(BDT_EVAL_DIR),
        help="Directory to read/write cached BDT predictions",
    )
    parser.add_argument("--at-inference", action="store_true", default=False)
    parser.add_argument("--test-mode", action="store_true", default=False)

    args = parser.parse_args()

    years = hh_vars.years if args.years[0] == "all" else list(args.years)
    channel = CHANNELS[args.channel]

    # Create channel-specific output directory
    outdir = Path(args.output_dir) / args.channel
    outdir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    opt_results = pd.read_csv(args.csv, index_col=0)
    # Identify Bmin columns
    bmin_cols = [c for c in opt_results.columns if c.startswith("Bmin=")]
    if not bmin_cols:
        raise ValueError("No Bmin= columns found in CSV")

    # Extract cut values for each Bmin
    bmin_cuts = {}
    for col in bmin_cols:
        try:
            cut_xtt = float(opt_results.loc["Cut_Xtt", col])
            bmin_value = col.split("=")[1]  # Extract the number from "Bmin=X"
            bmin_cuts[bmin_value] = cut_xtt
        except Exception as e:
            print(f"Skipping column {col}: {e}")
            continue

    # Sort by Bmin value for consistent ordering
    bmin_cuts = dict(sorted(bmin_cuts.items(), key=lambda x: float(x[0])))

    # Select subset of Bmin values: every Nth one, up to max_bmin_plots
    bmin_items = list(bmin_cuts.items())
    selected_bmin_cuts = dict(bmin_items[:: args.bmin_step][: args.max_bmin_plots])

    print(
        f"Selected {len(selected_bmin_cuts)} Bmin values to plot: {list(selected_bmin_cuts.keys())}"
    )

    # Load events per year for this channel
    events_dict = {}
    for year in years:
        filters_dict = base_filter(test_mode=False)
        filters_dict = tt_filters(channel=None, in_filters=filters_dict, num_fatjets=3, tt_cut=0.3)
        columns = get_columns(year)
        events_dict[year] = load_samples(
            year=year,
            paths=DATA_PATHS[year],
            signals=[args.signal_key],
            channels=[channel],
            samples=None,
            filters_dict=filters_dict,
            load_columns=columns,
            restrict_data_to_channel=True,
            load_bgs=True,
            load_data=True,
            loaded_samples=True,
            multithread=True,
        )
        _apply_basic_processing(events_dict[year], year)

        # Optionally attach BDT scores, following SensitivityStudy logic
    if args.use_bdt:
        compute_or_load_bdt_preds(
            events_dict=events_dict,
            modelname=args.modelname,
            model_dir=Path(args.model_dir),
            signal_key=args.signal_key,
            channel=channel,
            bdt_preds_dir=Path(args.bdt_eval_dir),
            test_mode=args.test_mode,
            at_inference=args.at_inference,
            all_outs=True,
        )

    # Merge events across all years for plotting
    merged_events = concatenate_years(events_dict, years)

    for var in args.variables:
        # Determine x-axis range for this variable
        if args.auto_range or args.xmin is None or args.xmax is None:
            print(f"\nDetermining range for variable: {var}")
            var_mins, var_maxs = [], []
            for sample in merged_events.values():
                vals = sample.get_var(var)
                var_mins.append(np.min(vals))
                var_maxs.append(np.max(vals))

            xmin = np.min(var_mins) if args.xmin is None else args.xmin
            xmax = np.max(var_maxs) if args.xmax is None else args.xmax
            print(f"  Auto-determined range: [{xmin:.3f}, {xmax:.3f}]")
        else:
            xmin = args.xmin
            xmax = args.xmax

        # Create variable-specific subdirectory
        var_outdir = outdir / var
        var_outdir.mkdir(parents=True, exist_ok=True)

        for sample_key, sample in merged_events.items():
            print(f"\nProcessing sample: {sample_key}, variable: {var}")

            # Build histogram for this sample with all Bmin cuts
            h = build_hist_for_sample_with_cuts(
                sample,
                args.channel,
                var,
                selected_bmin_cuts,
                args.bins,
                xmin,
                xmax,
                args.use_bdt,
            )

            # Create filenames
            save_path_pdf = var_outdir / f"{sample_key}.pdf"
            save_path_png = var_outdir / f"{sample_key}.png"

            # Create region label with channel, sample
            region_label = f"{channel.label}\n{sample_key}"

            # Get list of axis labels (for indexing histogram) and legend labels (for display)
            axis_labels = [f"Bmin={bmin}" for bmin in selected_bmin_cuts]
            legend_labels = [
                f"BDT score > {selected_bmin_cuts[bmin]:.3f} (Bmin > {bmin})"
                for bmin in selected_bmin_cuts
            ]

            # Skip if no cuts to plot
            if not axis_labels:
                print(f"Warning: No Bmin cuts to plot for {sample_key}, skipping...")
                continue

            print(f"Plotting {sample_key} with {len(axis_labels)} Bmin cuts")
            print(f"  Axis labels: {axis_labels}")
            print(f"  Legend labels: {legend_labels}")
            print(f"  Histogram axes: {h.axes.name}")

            # Save PDF
            plot_overlaid_cuts(
                h,
                cut_labels=axis_labels,
                legend_labels=legend_labels,
                years=years,
                region_label=region_label,
                cmslabel="Work in progress",
                name=str(save_path_pdf),
                show=False,
            )
            print(f"Saved {save_path_pdf}")

            # Save PNG
            plot_overlaid_cuts(
                h,
                cut_labels=axis_labels,
                legend_labels=legend_labels,
                years=years,
                region_label=region_label,
                cmslabel="Work in progress",
                name=str(save_path_png),
                show=False,
            )
            print(f"Saved {save_path_png}")


if __name__ == "__main__":
    main()
