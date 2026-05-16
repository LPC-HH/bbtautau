from __future__ import annotations

import argparse
import json
from pathlib import Path

import hist
import matplotlib as mpl
import numpy as np
import pandas as pd
from boostedhh import hh_vars
from hist import Hist

# Use non-interactive backend for containerized/CLI environments
mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.postprocessing.utils import (
    concatenate_years,
    load_data_channel,
)
from bbtautau.userConfig import BDT_EVAL_DIR, MODEL_DIR

hep.style.use("CMS")

# Global constant for minimum log scale value
MIN_LOG = 1e-4

"""
Quick script: read cut thresholds from an optimization CSV, load data, apply cuts,
and produce histograms for given variables.

Features:
- Only use tt cut (Xtt), ignoring bb cut
- Plot each sample separately with different Bmin cuts overlaid in a single plot
- Create separate plots for each sample
- Save both PDF and PNG formats
"""


def compute_working_points_equal_events(
    discriminant_values: np.ndarray, weights: np.ndarray, n_wps: int, lower_thresh: float = 0.0
) -> tuple[np.ndarray, dict]:
    """
    Compute N working points (WPs) such that between each WP there is approximately
    the same number of weighted events.

    Args:
        discriminant_values: Array of discriminant values (e.g., tt glopart score)
        weights: Array of event weights
        n_wps: Number of working points to compute (creates n_wps+1 bins)

    Returns:
        tuple: (wp_thresholds, wp_info)
            - wp_thresholds: Array of threshold values (length n_wps)
            - wp_info: Dictionary with detailed information about each WP
    """
    # Sort by discriminant value (ascending)
    sort_idx = np.argsort(discriminant_values)
    sorted_disc = discriminant_values[sort_idx]
    sorted_weights = weights[sort_idx]

    # Below-threshold stats (before filtering)
    mask_above = sorted_disc >= lower_thresh
    events_below_thresh = int(np.sum(~mask_above))
    weight_below_thresh = float(np.sum(sorted_weights[~mask_above]))

    # Restrict to events at or above lower threshold for WP computation
    if lower_thresh > 0.0:
        sorted_disc = sorted_disc[mask_above]
        sorted_weights = sorted_weights[mask_above]

    cumsum_weights = np.cumsum(sorted_weights)
    total_weight_above = float(cumsum_weights[-1]) if len(cumsum_weights) > 0 else 0.0

    wp_thresholds = []
    wp_info = {
        "n_wps": n_wps,
        "lower_threshold": float(lower_thresh),
        "total_events": len(discriminant_values),
        "total_weight_above": total_weight_above,
        "weight_below_thresh": weight_below_thresh,
        "events_below_thresh": events_below_thresh,
        "wps": [],
    }

    if lower_thresh > 0.0:
        # First WP is fixed at lower_thresh; record below-threshold bin
        wp_thresholds.append(float(lower_thresh))
        wp_info["wps"].append(
            {
                "wp_index": 0,
                "threshold": float(lower_thresh),
                "events": events_below_thresh,
                "weight": weight_below_thresh,
                "cumulative_weight": weight_below_thresh,
            }
        )
        # Equal splits above threshold: (n_wps - 1) more thresholds -> n_wps bins above
        n_bins_above = n_wps
        target_weight_per_bin = total_weight_above / n_bins_above if n_bins_above > 0 else 0.0
        wp_info["target_weight_per_bin_above"] = float(target_weight_per_bin)
        n_to_compute = n_wps - 1
        prev_threshold = lower_thresh
    else:
        # No fixed first WP: equal splits over full range -> n_wps thresholds, n_wps+1 bins
        target_weight_per_bin = total_weight_above / (n_wps + 1) if (n_wps + 1) > 0 else 0.0
        wp_info["target_weight_per_bin"] = float(target_weight_per_bin)
        n_to_compute = n_wps
        prev_threshold = None

    for i in range(1, n_to_compute + 1):
        target_cumsum = i * target_weight_per_bin
        idx = np.searchsorted(cumsum_weights, target_cumsum, side="right")
        if idx >= len(sorted_disc):
            idx = len(sorted_disc) - 1
        elif idx == 0:
            idx = 1
        threshold = float(sorted_disc[idx])

        if prev_threshold is None:
            bin_mask = sorted_disc <= threshold
        else:
            bin_mask = (sorted_disc > prev_threshold) & (sorted_disc <= threshold)
        bin_weight = float(np.sum(sorted_weights[bin_mask]))
        bin_events = int(np.sum(bin_mask))

        wp_thresholds.append(threshold)
        wp_info["wps"].append(
            {
                "wp_index": i,
                "threshold": threshold,
                "events": bin_events,
                "weight": bin_weight,
                "cumulative_weight": float(cumsum_weights[idx]),
            }
        )
        prev_threshold = threshold

    # Last bin: above highest WP
    if len(wp_thresholds) > 0:
        last_mask = sorted_disc > wp_thresholds[-1]
        last_weight = float(np.sum(sorted_weights[last_mask]))
        last_events = int(np.sum(last_mask))
        last_wp_index = n_to_compute + 1
        wp_info["wps"].append(
            {
                "wp_index": last_wp_index,
                "threshold": float(np.max(sorted_disc)) if len(sorted_disc) > 0 else 0.0,
                "events": last_events,
                "weight": last_weight,
                "cumulative_weight": total_weight_above,
            }
        )

    wp_thresholds = np.array(wp_thresholds)

    print(f"  Computed {n_wps} working points:")
    if lower_thresh > 0.0:
        print(f"    First WP fixed at lower threshold: {lower_thresh:.2f}")
        print(
            f"    Weight above threshold: {total_weight_above:.2f}, target per bin above: {wp_info.get('target_weight_per_bin_above', 0):.2f}"
        )
    else:
        print(
            f"    Total weight: {total_weight_above:.2f}, target per bin: {wp_info.get('target_weight_per_bin', 0):.2f}"
        )
    print(
        f"    Total events: {len(discriminant_values)}, weight below threshold: {weight_below_thresh:.2f}"
    )
    for wp in wp_info["wps"]:
        print(
            f"    WP{wp['wp_index']}: threshold={wp['threshold']:.4f}, "
            f"events={wp['events']}, weight={wp['weight']:.2f}"
        )

    return wp_thresholds, wp_info


def plot_wp_visualization(
    discriminant_values: np.ndarray,
    weights: np.ndarray,
    wp_thresholds: np.ndarray,
    years: list[str],
    lower_thresh: float = None,
    discriminant_name: str = "Discriminant",
    channel_label: str = "",
    name: str = "",
    show: bool = False,
):
    """
    Create a visualization plot showing the discriminant distribution and working points.

    Args:
        discriminant_values: Array of discriminant values
        weights: Array of event weights
        wp_thresholds: Array of WP threshold values
        wp_info: Dictionary with WP information
        years: List of years for CMS label
        discriminant_name: Name/label for the discriminant
        name: Path to save plot
        show: Whether to show plot
    """
    import mplhep as hep

    plt.rcParams.update({"font.size": 24})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"

    # Top plot: Distribution with WP lines (no axis labels or legend; shared x with bottom)
    colors = plt.cm.tab10(np.linspace(0, 1, len(wp_thresholds)))
    counts, bins, _ = ax1.hist(
        discriminant_values,
        bins=100,
        weights=weights,
        histtype="step",
        linewidth=2,
        color="black",
    )

    for wp_thresh, color in zip(wp_thresholds, colors):
        ax1.axvline(
            wp_thresh,
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

    ax1.set_yscale("log")
    ax1.set_ylim(bottom=MIN_LOG)
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.3)

    # Add channel label as text in plot
    if channel_label:
        ax1.text(
            0.03,
            0.95,
            channel_label,
            transform=ax1.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
        )

    print(years)

    # Add CMS label
    hep.cms.label(
        "Work in progress",
        data=True,
        com="13.6",
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        year=year_str,
        ax=ax1,
        loc=0,
    )

    # Bottom plot: Cumulative distribution with WP markers
    sorted_disc = np.sort(discriminant_values)
    sorted_weights = weights[np.argsort(discriminant_values)]
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]
    cumsum_normalized = cumsum_weights / total_weight

    ax2.plot(sorted_disc, cumsum_normalized, linewidth=2, color="black", label="Cumulative")

    # Mark WP positions
    for i, wp_thresh in enumerate(wp_thresholds):
        # Find cumulative value at this threshold
        idx = np.searchsorted(sorted_disc, wp_thresh, side="right")
        if idx > 0:
            cumval = cumsum_normalized[idx - 1] if idx < len(cumsum_normalized) else 1.0
        else:
            cumval = 0.0

        ax2.plot(
            wp_thresh,
            cumval,
            marker="o",
            markersize=10,
            color=colors[i],
            label=f"WP{i+1}: {wp_thresh:.4f}",
        )
        ax2.axvline(wp_thresh, color=colors[i], linestyle="--", linewidth=1, alpha=0.5)

    if lower_thresh is not None:
        ax2.axvline(
            lower_thresh,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"Lower threshold: {lower_thresh}",
        )

    ax2.set_xlabel(discriminant_name)
    ax2.set_ylabel("Cumulative Fraction")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="best", fontsize=14, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if name:
        plt.savefig(name, bbox_inches="tight")
        print(f"Saved WP visualization to {name}")

    if show:
        plt.show()
    else:
        plt.close()


def save_working_points(
    wp_thresholds: np.ndarray,
    wp_info: dict,
    filepath: Path,
    variable_name: str = None,
    discriminant_name: str = None,
):
    """Save working points to a JSON file for reuse.

    Args:
        wp_thresholds: Array of WP threshold values
        wp_info: Dictionary with WP information
        filepath: Path to save the JSON file
        variable_name: Name of the variable/discriminant (e.g., "ttFatJetParTXtauhtauhvsQCDTop")
        discriminant_name: Alternative name for the discriminant (for backward compatibility)
    """
    # Use variable_name or discriminant_name (discriminant_name for backward compat)
    var_name = variable_name or discriminant_name

    # Load existing data if file exists (to support multiple variables in one file)
    if filepath.exists():
        with filepath.open("r") as f:
            save_data = json.load(f)
    else:
        save_data = {}

    # Store WPs for this variable
    if var_name:
        if "variables" not in save_data:
            save_data["variables"] = {}
        save_data["variables"][var_name] = {
            "wp_thresholds": wp_thresholds.tolist(),
            "wp_info": wp_info,
        }
        print(f"Saved working points for variable '{var_name}' to {filepath}")
    else:
        # Backward compatibility: save at top level if no variable name
        save_data["wp_thresholds"] = wp_thresholds.tolist()
        save_data["wp_info"] = wp_info
        print(f"Saved working points (legacy format) to {filepath}")

    with filepath.open("w") as f:
        json.dump(save_data, f, indent=2)


def load_working_points(filepath: Path, variable_name: str = None) -> tuple[np.ndarray, dict]:
    """Load working points from a JSON file.

    Args:
        filepath: Path to the JSON file
        variable_name: Name of the variable to load WPs for (if None, uses legacy format)

    Returns:
        tuple: (wp_thresholds, wp_info)
    """
    with filepath.open("r") as f:
        data = json.load(f)

    # Check if new format with variables dict
    if "variables" in data:
        if variable_name is None:
            # If no variable name specified, try to get the first one
            var_names = list(data["variables"].keys())
            if len(var_names) == 1:
                variable_name = var_names[0]
                print(f"No variable name specified, using '{variable_name}' from file")
            else:
                raise ValueError(
                    f"Multiple variables found in file: {var_names}. "
                    f"Please specify --wp-variable-name"
                )

        if variable_name not in data["variables"]:
            available = list(data["variables"].keys())
            raise ValueError(
                f"Variable '{variable_name}' not found in file. Available: {available}"
            )

        var_data = data["variables"][variable_name]
        wp_thresholds = np.array(var_data["wp_thresholds"])
        wp_info = var_data["wp_info"]
        print(f"Loaded working points for variable '{variable_name}' from {filepath}")
    else:
        # Legacy format (backward compatibility)
        wp_thresholds = np.array(data["wp_thresholds"])
        wp_info = data["wp_info"]
        print(f"Loaded working points (legacy format) from {filepath}")

    return wp_thresholds, wp_info


def list_working_points(filepath: Path) -> list[str]:
    """List all variable names stored in a working points file.

    Args:
        filepath: Path to the JSON file

    Returns:
        List of variable names
    """
    with filepath.open("r") as f:
        data = json.load(f)

    if "variables" in data:
        return list(data["variables"].keys())
    else:
        return ["legacy_format"]


def plot_overlaid_cuts(
    h: Hist,
    cut_labels: list[str],
    years: list[str],
    region_label: str = None,
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
        ax.set_ylim(bottom=MIN_LOG)
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

    # Title removed - channel label is shown in region_label text instead

    # Save plot
    if name:
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def build_hist_for_sample_with_cuts(
    sample,
    varname: str,
    cuts_dict: dict,
    bins: int,
    x_min: float,
    x_max: float,
    cut_var: str,
) -> Hist:
    """
    Build histogram for a single sample with different cuts overlaid.
    Returns a histogram with axes: (cut_level, variable).

    Args:
        sample: LoadedSample object
        varname: Variable name to histogram
        cuts_dict: Dictionary mapping cut labels to cut thresholds (applied on cut_var)
        bins: Number of bins
        x_min: Minimum x value
        x_max: Maximum x value
        cut_var: Variable name on which cuts are applied
    """
    cut_labels = list(cuts_dict.keys())
    h = Hist(
        hist.axis.StrCategory(cut_labels, name="cut_level"),
        hist.axis.Regular(bins, x_min, x_max, name=varname),
        storage="weight",
    )

    if cut_var == "aggregate_ParT":  # special case
        arrs = np.stack(
            [
                sample.get_var(f"ttFatJetParTX{channel.tagger_label}vsQCDTop")
                for channel in CHANNELS.values()
            ],
            axis=0,
        )
        xcut = np.max(arrs, axis=0)
    else:
        xcut = sample.get_var(cut_var)
    vals = sample.get_var(varname)
    weight = sample.get_var("finalWeight")

    print(f"  Total events before cuts: {len(vals)}")
    print(f"  Variable '{varname}' range: [{np.min(vals):.3f}, {np.max(vals):.3f}]")
    print(f"  Histogram bin range: [{x_min}, {x_max}]")
    print(f"  Cut variable '{cut_var}' range: [{np.min(xcut):.3f}, {np.max(xcut):.3f}]")

    n_outside = np.sum((vals < x_min) | (vals > x_max))
    if n_outside > 0:
        print(f"  WARNING: {n_outside}/{len(vals)} events outside histogram range!")

    for label, threshold in cuts_dict.items():
        sel = xcut >= threshold
        vals_sel = vals[sel]
        weight_sel = weight[sel]

        print(
            f"    {label}: cut={threshold:.3f}, events passing={len(vals_sel)}, sum(weights)={np.sum(weight_sel):.2f}"
        )

        if len(vals_sel):
            h.fill(cut_level=label, **{varname: vals_sel}, weight=weight_sel)

    total_events = np.sum([h[label, :].values().sum() for label in cuts_dict])
    print(f"  Total events in histogram: {total_events:.2f}")

    return h


def _match_gen_to_reco_jets(sample: LoadedSample):
    """
    Helper function to match gen Higgs jets to reco jets using deltaR.

    Returns a dictionary with matching information.
    """
    from bbtautau.postprocessing.utils import delta_eta, delta_phi

    # Get gen Higgs information
    gen_children = sample.events["GenHiggsChildren"].to_numpy()  # (N_events, 2)
    gen_eta = sample.events["GenHiggsEta"].to_numpy()  # (N_events, 2)
    gen_phi = sample.events["GenHiggsPhi"].to_numpy()  # (N_events, 2)

    # Get reco jet information
    reco_eta = sample.events["ak8FatJetEta"].to_numpy()  # (N_events, N_jets)
    reco_phi = sample.events["ak8FatJetPhi"].to_numpy()  # (N_events, N_jets)

    # Get weights
    weights = sample.get_var("finalWeight")  # (N_events,)
    n_events = len(weights)
    n_jets = reco_eta.shape[1]

    # Vectorized deltaR computation for all gen Higgs vs all reco jets
    gen_eta_expanded = gen_eta[:, :, np.newaxis]  # (N_events, 2, 1)
    gen_phi_expanded = gen_phi[:, :, np.newaxis]  # (N_events, 2, 1)
    reco_eta_expanded = reco_eta[:, np.newaxis, :]  # (N_events, 1, N_jets)
    reco_phi_expanded = reco_phi[:, np.newaxis, :]  # (N_events, 1, N_jets)

    # Compute deltaR for all combinations: (N_events, 2, N_jets)
    deta = delta_eta(gen_eta_expanded, reco_eta_expanded)  # (N_events, 2, N_jets)
    dphi = delta_phi(gen_phi_expanded, reco_phi_expanded)  # (N_events, 2, N_jets)
    dr = np.sqrt(deta**2 + dphi**2)  # (N_events, 2, N_jets)

    # Identify which gen Higgs is bb (5) and which is tt (15)
    is_hbb = gen_children == 5  # (N_events, 2) - True where Higgs is bb
    is_htt = gen_children == 15  # (N_events, 2) - True where Higgs is tt

    # Find which Higgs index (0 or 1) is bb/tt for each event
    hbb_higgs_idx = np.argmax(is_hbb, axis=1)  # (N_events,) - Higgs index that is bb
    htt_higgs_idx = np.argmax(is_htt, axis=1)  # (N_events,) - Higgs index that is tt

    # Get events that actually have bb/tt Higgs, just safety check
    has_hbb = is_hbb.any(axis=1)  # (N_events,) - True if event has bb Higgs
    has_htt = is_htt.any(axis=1)  # (N_events,) - True if event has tt Higgs

    if not has_hbb.any() or not has_htt.any():
        print("Warning: Some events have no bb or tt Higgs")

    row_indices = np.arange(n_events)

    # Step 1: Match Hbb to closest jet (priority)
    # Get dr values for Hbb: (N_events, N_jets)
    dr_bb = dr[row_indices, hbb_higgs_idx, :]  # (N_events, N_jets)
    bb_jet_indices = np.argmin(dr_bb, axis=1)  # (N_events,)
    dr_bb_values = dr_bb[row_indices, bb_jet_indices]  # (N_events,)

    # Step 2: Match Htt, but if it matches the same jet as Hbb, pick second-best
    # Get dr values for Htt: (N_events, N_jets)
    dr_tt = dr[
        row_indices, htt_higgs_idx, :
    ].copy()  # (N_events, N_jets) - copy to avoid modifying original

    # For events where Htt's closest jet is the same as Hbb's, mask it out
    tt_closest_initial = np.argmin(dr_tt, axis=1)  # (N_events,)
    same_jet_mask = (
        tt_closest_initial == bb_jet_indices
    )  # (N_events,) - True where they match same jet

    if same_jet_mask.any():
        # Set the matched jet's dr to inf for Htt in events where they conflict
        dr_tt[same_jet_mask, bb_jet_indices[same_jet_mask]] = np.inf

    # Find the final Htt match (either closest or second-best if there was a conflict)
    tt_jet_indices = np.argmin(dr_tt, axis=1)  # (N_events,)
    dr_tt_values = dr_tt[np.arange(n_events), tt_jet_indices]  # (N_events,)

    # Create matched_jet_indices array for compatibility (but note: Htt may be second-best)
    matched_jet_indices = np.zeros((n_events, 2), dtype=int)
    matched_jet_indices[row_indices, hbb_higgs_idx] = bb_jet_indices
    matched_jet_indices[row_indices, htt_higgs_idx] = tt_jet_indices

    return {
        "matched_jet_indices": matched_jet_indices,
        "hbb_higgs_idx": hbb_higgs_idx,
        "htt_higgs_idx": htt_higgs_idx,
        "bb_jet_indices": bb_jet_indices,
        "tt_jet_indices": tt_jet_indices,
        "dr_bb_values": dr_bb_values,
        "dr_tt_values": dr_tt_values,
        "weights": weights,
        "n_events": n_events,
        "n_jets": n_jets,
    }


def plot_jet_assignment_confusion(
    sample: LoadedSample,
    years: list[str],
    channel_label: str,
    sample_key: str,
    name: str = "",
    show: bool = False,
):
    """
    Create a confusion matrix plot comparing gen-matched jet assignment vs reconstructed assignment.

    Args:
        sample: LoadedSample object with events, bb_mask, and tt_mask
        years: List of years for CMS label
        channel_label: Channel label for plot
        sample_key: Sample key for plot
        name: Path to save plot
        show: Whether to show plot
    """
    import mplhep as hep

    # Get matching information
    match_info = _match_gen_to_reco_jets(sample)
    matched_jet_indices = match_info["matched_jet_indices"]
    hbb_higgs_idx = match_info["hbb_higgs_idx"]
    htt_higgs_idx = match_info["htt_higgs_idx"]

    print(np.sum(hbb_higgs_idx == htt_higgs_idx))

    weights = match_info["weights"]
    n_events = match_info["n_events"]

    # Get reconstructed assignment masks
    bb_mask = sample.bb_mask  # (N_events, N_jets) - Reconstructed bb jet
    tt_mask = sample.tt_mask  # (N_events, N_jets) - Reconstructed tt jet

    # Initialize confusion matrix: rows = predicted (reco), cols = true (gen)
    # Rows: bb, tt, Unmatched | Cols: Hbb, Htt, Unmatched
    cm = np.zeros((3, 3), dtype=float)

    # Get matched jet indices for bb Higgs and apply to masks

    row_indices = np.arange(n_events)

    ## Matched Hbb jet
    matched_jets_hbb = matched_jet_indices[
        row_indices, hbb_higgs_idx
    ]  # (N_events,), contains the index 0,1,2 of the matched jet for the bb Higgs
    bb_correct = bb_mask[row_indices, matched_jets_hbb]  # (N_events,)
    tt_wrong = tt_mask[row_indices, matched_jets_hbb]  # (N_events,)
    unidentified_Hbb = ~bb_correct & ~tt_wrong  # (N_events,)

    cm[0, 0] = np.sum(weights * bb_correct)  # bb-like (correct)
    cm[1, 0] = np.sum(weights * tt_wrong)  # tt-like (wrong)
    cm[2, 0] = np.sum(weights * unidentified_Hbb)  # Not identified

    ## Matched Htt jet
    matched_jets_htt = matched_jet_indices[
        row_indices, htt_higgs_idx
    ]  # (N_events,), contains the index 0,1,2 of the matched jet for the tt Higgs
    tt_correct = tt_mask[row_indices, matched_jets_htt]  # (N_events,)
    bb_wrong = bb_mask[row_indices, matched_jets_htt]  # (N_events,)
    unidentified_Htt = ~tt_correct & ~bb_wrong  # (N_events,)

    cm[0, 1] = np.sum(weights * bb_wrong)  # bb-like (wrong)
    cm[1, 1] = np.sum(weights * tt_correct)  # tt-like (correct)
    cm[2, 1] = np.sum(weights * unidentified_Htt)  # Not identified

    ## Un-gen-matched: jet left unassigned in gen-matching
    unmatched_gen = (3 - matched_jets_htt - matched_jets_hbb) % 3  # (N_events,)
    tt_wrong_unmatched = tt_mask[row_indices, unmatched_gen]
    bb_wrong_unmatched = bb_mask[row_indices, unmatched_gen]
    unmatched_correct = ~tt_wrong_unmatched & ~bb_wrong_unmatched  # (N_events,)

    cm[0, 2] = np.sum(weights * bb_wrong_unmatched)  # bb-like (wrong)
    cm[1, 2] = np.sum(weights * tt_wrong_unmatched)  # tt-like (correct)
    cm[2, 2] = np.sum(weights * unmatched_correct)  # Not identified

    classes_gen = ["Hbb", "Htt", "Unmatched"]
    classes_reco = ["bb-like", "tt-like", "Unidentified"]
    n_gen = len(classes_gen)
    n_reco = len(classes_reco)

    # Normalize by row (per identified class)
    # col_sums = cm.sum(axis=0, keepdims=True)
    row_sums = cm.sum(axis=1, keepdims=True)

    # cm_norm = np.divide(cm, col_sums, where=col_sums != 0)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    # Create plot
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(
        im,
        ax=ax,
        label="Fraction of true class (column-normalized)",
    )

    ax.set_xticks(np.arange(n_gen))
    ax.set_xticklabels(classes_gen)
    ax.set_yticks(np.arange(n_reco))
    ax.set_yticklabels(classes_reco)
    ax.set_xlabel("True class (Gen-matched)", fontsize=18)
    ax.set_ylabel("Predicted class (Reconstructed)", fontsize=18)

    # Add text annotations
    for i in range(n_reco):
        for j in range(n_gen):
            value = cm_norm[i, j]
            # raw_value = cm[i, j]
            display_val = f"{value:.3f}"  # \n({raw_value:.4f})"
            ax.text(
                j,
                i,
                display_val,
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=14,
                fontweight="bold",
            )

    # Add title
    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"
    title = f"Jet Assignment Confusion Matrix {channel_label}"
    ax.set_title(title, fontsize=16, pad=20)

    # Add CMS label
    hep.cms.label(
        "Work in progress",
        data=True,
        com="13.6",
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        year=year_str,
        ax=ax,
        fontsize=14,
        loc=0,
    )

    plt.tight_layout()

    if name:
        plt.savefig(name, bbox_inches="tight")
        print(f"Saved confusion matrix to {name}")

    if show:
        plt.show()
    else:
        plt.close()

    # Print summary statistics
    print(f"\nConfusion Matrix Summary for {sample_key}:")
    print(f"  Total weighted events: {cm.sum():.2f}")
    print(f"  Gen Hbb matched: {cm[:, 0].sum():.2f}")
    print(f"  Gen Htt matched: {cm[:, 1].sum():.2f}")
    print(f"  Unmatched: {cm[:, 2].sum():.2f}")
    print("\n  Correct assignments:")
    print(f"    Hbb -> bb: {cm[0, 0]:.2f} ({cm_norm[0, 0]*100:.1f}%)")
    print(f"    Htt -> tt: {cm[1, 1]:.2f} ({cm_norm[1, 1]*100:.1f}%)")
    print("  Misidentifications:")
    print(f"    Hbb -> tt: {cm[1, 0]:.2f} ({cm_norm[1, 0]*100:.1f}%)")
    print(f"    Htt -> bb: {cm[0, 1]:.2f} ({cm_norm[0, 1]*100:.1f}%)")


def plot_jet_scores_2d(
    sample: LoadedSample,
    channel: Channel,
    years: list[str],
    channel_label: str,
    name: str = "",
    show: bool = False,
    bins: int = 30,
):
    """
    Create a 2D density plot of gen-matched bb jet score vs gen-matched tautau jet score.
    Uses deltaR matching to find the closest reco jet to each gen Higgs.

    Args:
        sample: LoadedSample object with events
        channel: Channel object to get tagger_label
        years: List of years for CMS label
        channel_label: Channel label for plot
        sample_key: Sample key for plot
        name: Path to save plot
        show: Whether to show plot
        bins: Number of bins for 2D histogram
    """
    import mplhep as hep

    from bbtautau.postprocessing.utils import PAD_VAL

    # Get tagger label for this channel
    taukey = channel.tagger_label

    # Get matching information
    match_info = _match_gen_to_reco_jets(sample)
    bb_jet_indices = match_info["bb_jet_indices"]
    tt_jet_indices = match_info["tt_jet_indices"]
    weights = match_info["weights"]
    n_events = match_info["n_events"]
    n_jets = match_info["n_jets"]

    # Get scores for all jets
    bb_score_var = "ak8FatJetParTXbbvsQCD"
    tt_score_var = f"ak8FatJetParTX{taukey}vsQCD"
    bb_scores_all = sample.events[bb_score_var].to_numpy()  # (N_events, N_jets)
    tt_scores_all = sample.events[tt_score_var].to_numpy()  # (N_events, N_jets)

    # Extract scores for gen-matched jets
    # Use advanced indexing to get scores for matched jets
    bb_scores = np.full(n_events, np.nan)
    tt_scores = np.full(n_events, np.nan)

    # Get bb scores for events with matched bb jets
    valid_bb_events = (bb_jet_indices >= 0) & (bb_jet_indices < n_jets)
    if valid_bb_events.any():
        bb_scores[valid_bb_events] = bb_scores_all[valid_bb_events, bb_jet_indices[valid_bb_events]]

    # Get tt scores for events with matched tt jets
    valid_tt_events = (tt_jet_indices >= 0) & (tt_jet_indices < n_jets)
    if valid_tt_events.any():
        tt_scores[valid_tt_events] = tt_scores_all[valid_tt_events, tt_jet_indices[valid_tt_events]]

    # Filter out events with invalid scores (PAD_VAL, NaN, or missing gen match)
    valid_mask = (
        (bb_scores != PAD_VAL)
        & (tt_scores != PAD_VAL)
        & ~np.isnan(bb_scores)
        & ~np.isnan(tt_scores)
        & valid_bb_events
        & valid_tt_events
    )

    bb_scores_valid = bb_scores[valid_mask]
    tt_scores_valid = tt_scores[valid_mask]
    weights_valid = weights[valid_mask]

    print(f"  Valid events for 2D plot: {np.sum(valid_mask)}/{len(weights)}")
    if len(bb_scores_valid) > 0:
        print(f"  bb score range: [{np.min(bb_scores_valid):.3f}, {np.max(bb_scores_valid):.3f}]")
        print(f"  tt score range: [{np.min(tt_scores_valid):.3f}, {np.max(tt_scores_valid):.3f}]")
    else:
        print("  Warning: No valid events found for 2D plot!")
        if show:
            plt.show()
        return

    # Create 2D density plot
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Use hist2d for density plot with weights
    h, xedges, yedges, im = ax.hist2d(
        bb_scores_valid,
        tt_scores_valid,
        bins=bins,
        weights=weights_valid,
        # cmap="Blues",
        cmin=1e-6,  # Minimum value for color scale
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted Events", fontsize=18)

    # Set labels - indicate these are gen-matched jets
    ax.set_xlabel(f"Gen-matched BB Jet: {bb_score_var}", fontsize=18)
    ax.set_ylabel(f"Gen-matched Tautau Jet: {tt_score_var}", fontsize=18)

    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"

    # Add channel label as text in plot
    ax.text(
        0.03,
        0.95,
        channel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
    )

    # Add CMS label
    hep.cms.label(
        "Work in progress",
        data=True,
        com="13.6",
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        year=year_str,
        ax=ax,
        loc=0,
    )

    plt.tight_layout()

    if name:
        plt.savefig(name, bbox_inches="tight")
        print(f"Saved 2D density plot to {name}")

    if show:
        plt.show()
    else:
        plt.close()


def _get_gen_matched_var_values(
    sample: LoadedSample,
    match_info: dict,
    var_name: str,
    jet_type: str = "tt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract variable values for gen-matched jets.

    Args:
        sample: LoadedSample object
        match_info: Dictionary from _match_gen_to_reco_jets
        var_name: Variable name (e.g., "ak8FatJetParTXtauhtauhvsQCDTop")
        jet_type: "bb" or "tt" to select which gen-matched jet

    Returns:
        tuple: (values, weights, valid_mask)
            - values: Variable values for gen-matched jets (NaN where no match)
            - weights: Event weights
    """
    jet_indices = match_info[f"{jet_type}_jet_indices"]
    weights = match_info["weights"]
    n_jets = match_info["n_jets"]

    # Get variable values for all jets
    var_values_all = sample.events[var_name].to_numpy()  # (N_events, N_jets)

    # Extract values for gen-matched jets
    valid_mask = (jet_indices >= 0) & (jet_indices < n_jets)

    if not valid_mask.all():
        print(f"Warning: Some events have no {jet_type} jet")

    indices = np.arange(len(weights))
    values = var_values_all[indices, jet_indices]

    return values, weights


def plot_jet_comparison_histograms(
    sample: LoadedSample,
    match_info: dict,
    variables: list[str],
    years: list[str],
    channel_label: str,
    sample_key: str,
    output_dir: Path,
    bins: int = 50,
    show: bool = False,
):
    """
    Create comparison histograms for tt_mask-selected jets vs gen-matched jets.

    Args:
        sample: LoadedSample object
        match_info: Dictionary from _match_gen_to_reco_jets
        variables: List of variable names to plot (should be ak8FatJet variables)
        years: List of years for CMS label
        channel_label: Channel label for plot
        sample_key: Sample key for plot
        output_dir: Directory to save plots
        bins: Number of bins for histograms
        show: Whether to show plots
    """
    import mplhep as hep

    from bbtautau.postprocessing.utils import PAD_VAL

    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"
    weights = match_info["weights"]

    # Create output directory for comparison plots
    comp_dir = output_dir / "jet_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for var_name in variables:
        # Get gen-matched jet values
        gen_matched_values, _ = _get_gen_matched_var_values(
            sample, match_info, var_name, jet_type="tt"
        )

        # Identify misidentified jets: events where tt_mask selected jet != gen-matched jet
        tt_mask_jet_idx = np.argmax(sample.tt_mask, axis=1)  # Index of tt_mask selected jet
        # bb_mask_jet_idx = np.argmax(sample.bb_mask, axis=1)  # Index of bb_mask selected jet
        # gen_matched_bb_jet_idx = match_info["bb_jet_indices"]  # Index of gen-matched bb jet
        gen_matched_tt_jet_idx = match_info["tt_jet_indices"]  # Index of gen-matched tt jet

        print(
            f"Event fraction where indices differ: {np.sum(tt_mask_jet_idx != gen_matched_tt_jet_idx)/len(tt_mask_jet_idx)}"
        )

        row_indices = np.arange(len(weights))

        tt_var_name = var_name.replace("ak8FatJet", "ttFatJet")
        try:
            tt_like_values = sample.get_var(tt_var_name)  # Uses tt_mask automatically
            Htt_values = sample.get_var(var_name)[row_indices, gen_matched_tt_jet_idx]

            tt_like_valid = (tt_like_values != PAD_VAL) & ~np.isnan(tt_like_values)
            if np.sum(~tt_like_valid) > 0:
                print(f"Warning: invalid disc score {np.sum(~tt_like_valid)}")
        except (ValueError, KeyError):
            print(f"  Warning: Could not get {tt_var_name} or {var_name}, skipping...")
            continue

        # Misidentified rates
        misidentified = tt_mask_jet_idx != gen_matched_tt_jet_idx

        # Create comparison plot
        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot all three distributions
        ax.hist(
            tt_like_values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            color="blue",
            label="Tautau-like fatjets",
            alpha=0.8,
        )

        ax.hist(
            gen_matched_values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            color="red",
            label="Gen-matched tautau jets",
            linestyle="--",
            alpha=0.8,
        )

        # Plot misidentified jets
        ax.hist(
            tt_like_values[misidentified],
            bins=bins,
            weights=weights[misidentified],
            histtype="step",
            linewidth=2,
            color="orange",
            label="Misidentified tautau-like fatjets",
            linestyle=":",
            alpha=0.8,
        )

        ax.hist(
            Htt_values[misidentified],
            bins=bins,
            weights=weights[misidentified],
            histtype="step",
            linewidth=2,
            color="purple",
            label="Misidentified gen-matched Htt fatjets",
            linestyle="-",
            alpha=0.8,
        )

        ax.set_xlabel(var_name, fontsize=16)
        ax.set_ylabel("Weighted Events", fontsize=16)
        ax.set_yscale("log")
        ax.set_ylim(bottom=MIN_LOG)
        ax.legend(loc="best", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add channel label as text in plot
        ax.text(
            0.03,
            0.95,
            channel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            verticalalignment="top",
        )

        # Add CMS label
        hep.cms.label(
            "Work in progress",
            data=True,
            com="13.6",
            lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
            year=year_str,
            ax=ax,
            loc=0,
        )

        plt.tight_layout()

        # Save plot
        plot_name = comp_dir / f"{sample_key}_{var_name.replace('ak8FatJet', 'Jet')}_comparison.pdf"
        plt.savefig(plot_name, bbox_inches="tight")
        print(f"  Saved comparison plot: {plot_name}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_dr_distributions(
    match_info: dict,
    years: list[str],
    channel_label: str,
    sample_key: str,
    output_dir: Path,
    bins: int = 50,
    show: bool = False,
):
    """
    Plot deltaR distributions for gen matching of bb and tt jets.

    Args:
        sample: LoadedSample object
        match_info: Dictionary from _match_gen_to_reco_jets
        years: List of years for CMS label
        channel_label: Channel label for plot
        sample_key: Sample key for plot
        output_dir: Directory to save plots
        bins: Number of bins for histograms
        show: Whether to show plots
    """
    import mplhep as hep

    year_str = "-".join(years) if years != hh_vars.years else "2022-2023"

    # Create output directory for dR plots
    dr_dir = output_dir / "dr_distributions"
    dr_dir.mkdir(parents=True, exist_ok=True)

    # Get dR values for bb and tt (from precomputed values)
    dr_bb = match_info["dr_bb_values"]
    dr_tt = match_info["dr_tt_values"]
    weights = match_info["weights"]

    # Create combined plot
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot both distributions
    ax.hist(
        dr_bb,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=2,
        color="blue",
        label="H→bb gen matching",
        alpha=0.8,
    )

    ax.hist(
        dr_tt,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=2,
        color="red",
        label="H→ττ gen matching",
        linestyle="--",
        alpha=0.8,
    )

    ax.set_xlabel("ΔR (gen jet, nearest reco jet)", fontsize=16)
    ax.set_ylabel("Weighted Events", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(bottom=MIN_LOG)
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add channel label as text in plot
    ax.text(
        0.03,
        0.95,
        channel_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        verticalalignment="top",
    )

    # Add CMS label
    hep.cms.label(
        "Work in progress",
        data=True,
        com="13.6",
        lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
        year=year_str,
        ax=ax,
        loc=0,
    )

    plt.tight_layout()

    # Save plot
    plot_name = dr_dir / f"{sample_key}_dr_distributions.pdf"
    plt.savefig(plot_name, bbox_inches="tight")
    print(f"  Saved dR distribution plot: {plot_name}")

    if show:
        plt.show()
    else:
        plt.close()


def run_wps_mode(args, channel: Channel, channel_name: str, years: list[str]):
    """Run working points mode for a single channel."""
    print("\n" + "=" * 60)
    print("WORKING POINTS MODE")
    print("=" * 60)
    print(f"Processing channel: {channel_name}")

    # Create output directory structure: {output_dir}/{ttvsbb|agnostic}/wps/{channel}
    assignment_type = "ttvsbb" if args.ttvsbb else "agnostic"
    print(f"Using jet assignment: {assignment_type}")
    outdir = Path(args.output_dir) / assignment_type / "wps" / channel_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine WP file path (always in output-dir unless --wp-file specified for loading)
    default_wp_filepath = outdir / f"working_points_n{args.n_wps}.json"

    # Load signal-only events for this channel
    events_dict = load_data_channel(
        years=years,
        signals=[args.signal_key],
        channel=channel,
        test_mode=args.test_mode,
        tt_pres=False,
        models=None,  # No BDT needed for WP computation
        load_bgs=False,  # Signal only for WP computation
        load_data=False,  # Signal only for WP computation
        ttvsbb=args.ttvsbb,
    )

    # Merge events across all years
    merged_events = concatenate_years(events_dict, years)

    wp_filepath = default_wp_filepath
    # Compute WPs
    print(f"Computing {args.n_wps} working points")

    # Use first signal sample for WP computation
    signal_samples = [k for k in merged_events if merged_events[k].sample.isSignal]
    if not signal_samples:
        raise ValueError(f"No signal samples found for WP computation in channel {channel_name}")
    wp_sample_key = signal_samples[0]
    print(f"Using signal sample for WP computation: {wp_sample_key}")

    wp_sample = merged_events[wp_sample_key]
    taukey = channel.tagger_label

    # Get discriminant values (use GloParT, not BDT for WP computation)
    disc_name = f"ttFatJetParTX{taukey}vsQCDTop"
    disc_label = disc_name

    disc_values = wp_sample.get_var(disc_name)
    weights = wp_sample.get_var("finalWeight")

    # Compute working points
    wp_thresholds, wp_info = compute_working_points_equal_events(
        disc_values,
        weights,
        args.n_wps,
        lower_thresh=args.lower_thresh,
    )

    # Save working points with variable name
    save_working_points(
        wp_thresholds,
        wp_info,
        wp_filepath,
        variable_name=disc_name,
    )

    # Create visualization plot
    wp_viz_path = outdir / f"wp_visualization_n{args.n_wps}.pdf"
    plot_wp_visualization(
        disc_values,
        weights,
        wp_thresholds,
        years,
        lower_thresh=args.lower_thresh,
        discriminant_name=disc_label,
        channel_label=channel.label,
        name=str(wp_viz_path),
        show=False,
    )
    print(f"\nWorking points computed and saved. Visualization: {wp_viz_path}")
    print(f"  WP thresholds: {wp_thresholds}")
    print(f"  Saved to: {wp_filepath}")


def run_cuts_hist_mode(args, channel: Channel, channel_name: str, years: list[str]):
    """Run cuts histogram mode for a single channel."""
    print("\n" + "=" * 60)
    print("CUTS HISTOGRAM MODE")
    print("=" * 60)
    print(f"Processing channel: {channel_name}")

    # Create output directory structure: {output_dir}/{ttvsbb|agnostic}/cut_hists/{channel}
    assignment_type = "ttvsbb" if args.ttvsbb else "agnostic"
    print(f"Using jet assignment: {assignment_type}")
    outdir = Path(args.output_dir) / assignment_type / "cut_hists" / channel_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve the variable on which cuts are applied
    taukey = channel.tagger_label
    if args.cut_var is not None:
        cut_var = args.cut_var
    elif args.use_bdt:
        cut_var = f"BDTScore{taukey}vsAll"
    else:
        cut_var = f"ttFatJetParTX{taukey}vsQCDTop"
    print(f"Applying cuts on variable: {cut_var}")

    # Build cuts_dict: maps cut label -> threshold value
    if args.csv is not None:
        # Load optimized BDT thresholds indexed by Bmin from CSV
        opt_results = pd.read_csv(args.csv, index_col=0)
        bmin_cols = [c for c in opt_results.columns if c.startswith("Bmin=")]
        if not bmin_cols:
            raise ValueError("No Bmin= columns found in CSV")

        bmin_cuts = {}
        for col in bmin_cols:
            try:
                cut_xtt = float(opt_results.loc["Cut_Xtt", col])
                bmin_value = col.split("=")[1]
                bmin_cuts[bmin_value] = cut_xtt
            except Exception as e:
                print(f"Skipping column {col}: {e}")
                continue

        bmin_cuts = dict(sorted(bmin_cuts.items(), key=lambda x: float(x[0])))
        bmin_items = list(bmin_cuts.items())
        selected_bmin = dict(bmin_items[:: args.bmin_step][: args.max_bmin_plots])
        selected_cuts = {f"Bmin={bmin}": cut for bmin, cut in selected_bmin.items()}
        legend_labels = [
            f"{cut_var} > {cut:.3f} (Bmin > {bmin})" for bmin, cut in selected_bmin.items()
        ]
        print(f"Selected {len(selected_cuts)} Bmin values from CSV: {list(selected_bmin.keys())}")
    else:
        # Use thresholds provided directly on the command line
        thresholds = [float(t) for t in args.thresholds]
        selected_thresholds = sorted(thresholds)[: args.max_bmin_plots]
        selected_cuts = {f"{t:.3f}": t for t in selected_thresholds}
        legend_labels = [f"{cut_var} > {t:.3f}" for t in selected_thresholds]
        print(f"Using {len(selected_cuts)} thresholds: {selected_thresholds}")

    # Load events for this channel using load_data_channel (matching SensitivityStudy.py pattern)
    # One channel at a time, with backgrounds and data
    models = [args.modelname] if args.use_bdt else None
    events_dict = load_data_channel(
        years=years,
        signals=[args.signal_key],
        channel=channel,
        test_mode=args.test_mode,
        tt_pres=False,
        models=models,
        model_dir=Path(args.model_dir),
        bdt_eval_dir=Path(args.bdt_eval_dir),
        at_inference=args.at_inference,
        load_bgs=True,  # Load backgrounds for plotting
        load_data=True,  # Load data for plotting
    )

    # Merge events across all years for plotting
    merged_events = concatenate_years(events_dict, years)

    # Plot histograms for each variable
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

            h = build_hist_for_sample_with_cuts(
                sample,
                var,
                selected_cuts,
                args.bins,
                xmin,
                xmax,
                cut_var,
            )

            save_path_pdf = var_outdir / f"{sample_key}.pdf"
            save_path_png = var_outdir / f"{sample_key}.png"

            region_label = f"{channel.label}\n{sample_key}"
            axis_labels = list(selected_cuts.keys())

            if not axis_labels:
                print(f"Warning: No cuts to plot for {sample_key}, skipping...")
                continue

            print(f"Plotting {sample_key} with {len(axis_labels)} cuts")
            print(f"  Axis labels: {axis_labels}")
            print(f"  Legend labels: {legend_labels}")
            print(f"  Histogram axes: {h.axes.name}")

            for save_path in (save_path_pdf, save_path_png):
                plot_overlaid_cuts(
                    h,
                    cut_labels=axis_labels,
                    legend_labels=legend_labels,
                    years=years,
                    region_label=region_label,
                    cmslabel="Work in progress",
                    name=str(save_path),
                    show=False,
                )
                print(f"Saved {save_path}")


def run_jet_assignment_mode(args, channel: Channel, channel_name: str, years: list[str]):
    """Run jet assignment mode for a single channel."""
    print("\n" + "=" * 60)
    print("JET ASSIGNMENT CONFUSION MATRIX MODE")
    print("=" * 60)
    print(f"Processing channel: {channel_name}")

    # Create output directory structure: {output_dir}/{ttvsbb|agnostic}/jet_assignment/{channel}
    assignment_type = "ttvsbb" if args.ttvsbb else "agnostic"
    print(f"Using jet assignment: {assignment_type}")
    outdir = Path(args.output_dir) / assignment_type / "jet_assignment" / channel_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Load events for this channel (signal only for gen matching study)
    events_dict = load_data_channel(
        years=years,
        signals=[args.signal_key],
        channel=channel,
        test_mode=args.test_mode,
        tt_pres=False,
        models=None,  # No BDT needed for jet assignment study
        load_bgs=False,  # Signal only for gen matching
        load_data=False,  # Signal only for gen matching
        ttvsbb=args.ttvsbb,
    )

    # Merge events across all years
    merged_events = concatenate_years(events_dict, years)

    # Process each signal sample
    signal_samples = [k for k in merged_events if merged_events[k].sample.isSignal]
    if not signal_samples:
        raise ValueError(
            f"No signal samples found for jet assignment study in channel {channel_name}"
        )

    taukey = channel.tagger_label

    # Default variables to plot (discriminant + common jet variables)
    default_vars = [
        f"ak8FatJetParTX{taukey}vsQCDTop",  # Discriminant
        "ak8FatJetPt",
        "ak8FatJetEta",
    ]

    for sample_key in signal_samples:
        print(f"\nProcessing sample: {sample_key}")
        sample = merged_events[sample_key]

        # Check that masks are set
        if sample.bb_mask is None or sample.tt_mask is None:
            print(f"Warning: bb_mask or tt_mask not set for {sample_key}, skipping...")
            continue

        # Get gen matching information
        match_info = _match_gen_to_reco_jets(sample)

        # Create confusion matrix plot
        save_path_pdf = outdir / f"{sample_key}_confusion_matrix.pdf"

        plot_jet_assignment_confusion(
            sample,
            years,
            channel.label,
            sample_key,
            name=str(save_path_pdf),
            show=False,
        )

        print(f"Saved confusion matrix plots for {sample_key}")

        # Create 2D density plot of jet scores
        save_path_2d_pdf = outdir / f"{sample_key}_jet_scores_2d.pdf"

        plot_jet_scores_2d(
            sample,
            channel,
            years,
            channel.label,
            name=str(save_path_2d_pdf),
            show=False,
        )

        # Create comparison histograms (tt_mask vs gen-matched vs misidentified)
        print(f"\n  Creating jet comparison histograms for {sample_key}...")
        plot_jet_comparison_histograms(
            sample=sample,
            match_info=match_info,
            variables=default_vars,
            years=years,
            channel_label=channel.label,
            sample_key=sample_key,
            output_dir=outdir,
            bins=50,
            show=False,
        )

        # Create dR distribution plots
        print(f"\n  Creating dR distribution plots for {sample_key}...")
        dr_dir = Path(args.output_dir) / "dr_distributions"
        dr_dir.mkdir(parents=True, exist_ok=True)

        plot_dr_distributions(
            match_info=match_info,
            years=years,
            channel_label=channel.label,
            sample_key=sample_key,
            output_dir=dr_dir,
            bins=50,
            show=False,
        )

    print(f"\n  Comparison plots saved to: {outdir / 'jet_comparisons'}")
    print(f"  dR distribution plots saved to: {dr_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms for cuts or compute working points"
    )

    # Create mutually exclusive group for the three main tasks
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--cuts-hist",
        action="store_true",
        help="Task: Plot histograms for a given set of cuts (requires --csv)",
    )
    task_group.add_argument(
        "--wps",
        action="store_true",
        help="Task: Determine working points (computes or loads WPs)",
    )
    task_group.add_argument(
        "--jet-assignment",
        action="store_true",
        help="Task: Study jet assignment confusion matrix (gen-matched vs reconstructed)",
    )

    # Common arguments
    parser.add_argument(
        "--years",
        nargs="+",
        default=["all"],
        help="Years to include (e.g. 2022 2022EE) or 'all'",
    )
    parser.add_argument(
        "--channel",
        nargs="+",
        default=list(CHANNELS.keys()),
        choices=list(CHANNELS.keys()),
        help="Channel(s) to process, defaults to all",
    )
    parser.add_argument(
        "--signal-key",
        default="ggfbbtt",
        help="Signal key (base, without channel suffix)",
    )
    parser.add_argument("--output-dir", type=str, default="plots/CutHists")
    parser.add_argument(
        "--ttvsbb",
        action="store_true",
        default=False,
        help="Use tt vs bb discriminant for jet assignment",
    )

    # Arguments for --cuts-hist mode
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.8, 0.9, 0.95],
        help="Thresholds to plot (required for --cuts-hist)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to optimization CSV file (required for --cuts-hist)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variable names to histogram (required for --cuts-hist)",
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
    parser.add_argument("--use-bdt", action="store_true", default=True)
    parser.add_argument(
        "--cut-var",
        type=str,
        default=None,
        help=(
            "Variable to apply cuts on. If not set, defaults to the BDT score "
            "(when --use-bdt) or the tt discriminant. Use --csv to supply thresholds "
            "from an optimization file, or --thresholds for direct threshold values."
        ),
    )
    parser.add_argument(
        "--max-bmin-plots",
        type=int,
        default=4,
        help="Maximum number of Bmin cuts to plot (default: 4, only for --cuts-hist)",
    )
    parser.add_argument(
        "--bmin-step",
        type=int,
        default=2,
        help="Step size for selecting Bmin columns (default: 2, only for --cuts-hist)",
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

    # Arguments for --wps mode
    parser.add_argument(
        "--lower-thresh",
        type=float,
        default=0.0,
        help="Lower threshold for working points (default: 0.0, only for --wps)",
    )
    parser.add_argument(
        "--n-wps",
        type=int,
        default=5,
        help="Number of working points to compute (default: 5, only for --wps)",
    )

    args = parser.parse_args()

    # Validate task-specific arguments
    if args.cuts_hist and args.variables is None:
        parser.error("--variables is required for --cuts-hist mode")

    years = hh_vars.years if args.years[0] == "all" else list(args.years)

    # Process channels (can be single or multiple - argparse with nargs="+" always returns a list)
    channels_to_process = args.channel if isinstance(args.channel, list) else [args.channel]

    # Process each channel (data is reloaded for each channel)
    for channel_name in channels_to_process:
        channel = CHANNELS[channel_name]

        if args.wps:
            run_wps_mode(args, channel, channel_name, years)
        elif args.cuts_hist:
            run_cuts_hist_mode(args, channel, channel_name, years)
        elif args.jet_assignment:
            run_jet_assignment_mode(args, channel, channel_name, years)


if __name__ == "__main__":
    main()
