"""
Defines all the analysis regions.
****Important****: Region names used in the analysis cannot have underscores because of a rhalphalib convention.
Author(s): Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from boostedhh.utils import CUT_MAX_VAL

from bbtautau.postprocessing.bbtautau_types import Channel, Region
from bbtautau.postprocessing.bdt_utils import _get_bdt_key


def extract_optimal_cuts_from_csv(
    sensitivity_dir: Path,
    signal: str,
    channel_name: str,
    bmin: int,
    use_ParT: bool,
    do_vbf: bool,
):
    """
    Extract optimal cuts for a given bmin value from sensitivity study CSV files.

    Args:
        sensitivity_dir: Path to the sensitivity study's output directory
        signal: what signal region we are defining
        channel_name: Channel name (e.g., 'hh', 'hm', 'he')
        bmin: Minimum background yield value
        do_vbf: Whether using optimization accounting for vbf regions or not

    Returns:
        tuple: (txbb_cut, txtt_cut) - The optimal cuts for the given bmin
    """
    # read the first available FOM CSV file
    csv_dir = Path(sensitivity_dir).joinpath(
        f"full_presel/grid/{'ParT' if use_ParT else 'BDT'}/{'do_vbf' if do_vbf else 'ggf_only'}/sm_signals/orthogonal_channels/{signal}/{channel_name}"
    )

    # Look for any FOM-specific CSV files
    csv_files = list(csv_dir.glob("*_opt_results_*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No sensitivity CSV files found in {csv_dir}")

    # Take the first CSV file found and extract FOM name
    csv_file = sorted(csv_files)[0]  # Sort for reproducible behavior
    print(f"Reading CSV: {csv_file}")

    # Extract FOM name from filename like "2022_2022EE_opt_results_2sqrtB_S_var.csv"
    if "_opt_results_" in csv_file.name:
        fom_name = csv_file.name.split("_opt_results_")[1].replace(".csv", "")
    else:
        fom_name = "unknown"

    # Read as simple CSV (no multi-level headers)
    opt_results = pd.read_csv(csv_file, index_col=0)
    print(f"Using FOM: {fom_name}")
    print(f"Available B_min values: {opt_results.columns.tolist()}")

    # Check if the target Bmin column exists
    target_col = f"Bmin={bmin}"
    if target_col not in opt_results.columns:
        raise ValueError(
            f"B_min={bmin} not found in CSV. Available: {opt_results.columns.tolist()}"
        )

    # Extract the cuts
    txbb_cut = float(opt_results.loc["Cut_Xbb", target_col])
    txtt_cut = float(opt_results.loc["Cut_Xtt", target_col])

    print(f"Extracted cuts for Bmin={bmin}: TXbb={txbb_cut}, Txtt={txtt_cut}")

    return txbb_cut, txtt_cut


def get_selection_regions(
    signal: str,
    channel: Channel,
    vetoes: list[dict[str, Region]] = None,
    sensitivity_dir: Path = None,
    bmin: int = 10,
    use_ParT: bool = False,
    do_vbf: bool = False,
):
    """
    Get the selection regions for a given signal and channel.
    """
    pass_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
        "bbFatJetParTXbbvsQCDTop": [channel.txbb_cut, CUT_MAX_VAL],
    }

    fail_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
    }

    if sensitivity_dir is not None:
        txbb_cut, txtt_cut = extract_optimal_cuts_from_csv(
            Path(sensitivity_dir), signal, channel.key, bmin, use_ParT, do_vbf
        )
    else:
        txbb_cut = channel.txbb_cut
        txtt_cut = channel.txtt_cut if use_ParT else channel.txtt_BDT_cut

    if use_ParT:
        pass_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        pass_cuts[f"ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [txtt_cut, CUT_MAX_VAL]
        fail_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        fail_cuts[f"bbFatJetParTXbbvsQCD+ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [
            [-CUT_MAX_VAL, txbb_cut],
            [-CUT_MAX_VAL, txtt_cut],
        ]
    else:
        pass_cuts[_get_bdt_key(signal, channel, prefix_only=False)] = [txtt_cut, CUT_MAX_VAL]
        fail_cuts[f"bbFatJetParTXbbvsQCD+{_get_bdt_key(signal, channel, prefix_only=False)}"] = [
            [-CUT_MAX_VAL, txbb_cut],
            [-CUT_MAX_VAL, txtt_cut],
        ]

    if vetoes is not None:
        for veto in vetoes:
            pass_cuts.update(veto["fail"].cuts)
            # For the few events in the pass regions, we ignore their contribution to the fail regions that come after
            # Could add the logic in the future

    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts=pass_cuts,
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts=fail_cuts,
            signal=False,
            label="Fail",
        ),
    }

    return regions
