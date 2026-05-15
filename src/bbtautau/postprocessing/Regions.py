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


def load_cuts_from_csv(csv_file: str | Path, bmin: int) -> tuple[float, float]:
    """
    Load cuts from a sensitivity optimization CSV file.

    Args:
        csv_file: Path to CSV file (e.g., "2022_2022EE_opt_results_2sqrtB_S_var.csv")
        bmin: B_min value to look up (corresponds to column "Bmin={bmin}")

    Returns:
        tuple: (txbb_cut, txtt_cut) - The optimal cuts for the given bmin
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    cuts = pd.read_csv(csv_path, index_col=0)

    target_col = f"Bmin={bmin}"
    if target_col not in cuts.columns:
        raise ValueError(f"Column '{target_col}' not found. Available: {cuts.columns.tolist()}")

    txbb_cut = float(cuts.loc["Cut_Xbb", target_col])
    txtt_cut = float(cuts.loc["Cut_Xtt", target_col])

    print(
        f"Loaded cuts from {csv_path.name}: txbb={txbb_cut:.4f}, txtt={txtt_cut:.4f} (Bmin={bmin})"
    )

    return txbb_cut, txtt_cut


def _sensitivity_presel_dir(test_mode: bool, tt_pres: bool) -> str:
    """Match ``get_plot_dir`` presel segment in SensitivityStudy.py."""
    if test_mode:
        return "test"
    if tt_pres:
        return "tt_pres"
    return "full_presel"


def _sensitivity_disc_folder(
    use_ParT: bool,
    sensitivity_disc_tag: str | None,
    ggf_modelname: str | None,
) -> str:
    """Match ``disc_tag`` in ``SensitivityStudy.get_plot_dir`` (ParT, BDT export name, or literal ``BDT``)."""
    if use_ParT:
        return "ParT"
    if sensitivity_disc_tag:
        return sensitivity_disc_tag
    if ggf_modelname:
        return ggf_modelname
    return "BDT"


def extract_optimal_cuts_from_csv(
    sensitivity_dir: Path,
    signal: str,
    channel_name: str,
    combined_signals: str,
    bmin: int,
    use_ParT: bool,
    do_vbf: bool,
    *,
    test_mode: bool = False,
    tt_pres: bool = False,
    overlapping_channels: bool = False,
    sensitivity_disc_tag: str | None = None,
    ggf_modelname: str | None = None,
):
    """
    Extract optimal cuts for a given bmin value from sensitivity study CSV files.

    This function constructs the path to the CSV file based on the sensitivity study
    directory structure, then delegates to load_cuts_from_csv for the actual reading.

    Args:
        sensitivity_dir: Parent of the presel folder, i.e. ``.../SensitivityStudy/<date>/``
            (same level as ``tt_pres``, ``full_presel``, ``test``).
        signal: what signal region we are defining
        channel_name: Channel name (e.g., 'hh', 'hm', 'he')
        bmin: Minimum background yield value
        use_ParT: Whether using ParT tagger (True) or BDT (False)
        do_vbf: Whether using optimization accounting for vbf regions or not
        test_mode: Sensitivity study run with ``--test-mode`` (``test/`` presel folder).
        tt_pres: Sensitivity study run with ``--tt-pres`` (``tt_pres/`` presel folder).
        overlapping_channels: Use ``overlapping_channels/`` instead of ``orthogonal_channels/``.
        sensitivity_disc_tag: Optional folder name under presel (e.g. ``May4_optimized_ggf``).
            If omitted and not ParT, ``ggf_modelname`` is used, then ``\"BDT\"``.
        ggf_modelname: Default BDT export folder name when ``sensitivity_disc_tag`` is not set.

    Returns:
        tuple: (txbb_cut, txtt_cut) - The optimal cuts for the given bmin
    """
    presel = _sensitivity_presel_dir(test_mode, tt_pres)
    disc = _sensitivity_disc_folder(use_ParT, sensitivity_disc_tag, ggf_modelname)
    vbf_part = "do_vbf" if do_vbf else "ggf_only"
    ch_part = "overlapping_channels" if overlapping_channels else "orthogonal_channels"
    csv_dir = Path(sensitivity_dir).joinpath(
        presel, disc, vbf_part, combined_signals, ch_part, signal, channel_name
    )

    # Look for any FOM-specific CSV files
    csv_files = list(csv_dir.glob("*_opt_results_*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No sensitivity CSV files found in {csv_dir}")

    # Take the first CSV file found (sorted for reproducible behavior)
    csv_file = sorted(csv_files)[0]
    print(f"Reading CSV: {csv_file}")

    # Extract FOM name from filename for logging
    if "_opt_results_" in csv_file.name:
        fom_name = csv_file.name.split("_opt_results_")[1].replace(".csv", "")
        print(f"Using FOM: {fom_name}")

    return load_cuts_from_csv(csv_file, bmin)


def get_selection_regions(
    signal: str,
    channel: Channel,
    vetoes: list[dict[str, Region]] = None,
    sensitivity_dir: Path = None,
    bmin: int = 10,
    use_ParT: bool = False,
    do_vbf: bool = False,
    bb_disc: str = "ak8FatJetParTXbbvsQCDTop",
    # The following args are used to obtain the correct csv file for the cuts
    # ---------------
    # Combined signals refers to whether the optimization includes the total SM signal for all regions or just the ggf signal for ggf regions and vbf signal for vbf regions. At this stage it is only to get the right csv file directory. The separate signal is the default right now.
    combined_signals: str = "separate_signals",
    test_mode: bool = False,
    tt_pres: bool = False,
    overlapping_channels: bool = False,
    sensitivity_disc_tag: str | None = None,
    ggf_modelname: str | None = None,
):
    """
    Get the selection regions for a given signal and channel.
    """

    # parse bb discriminator
    if bb_disc.startswith("ak8"):
        bb_disc = "bb" + bb_disc[len("ak8") :]
    else:
        raise ValueError(f"bb_disc must start with 'ak8', but got: {bb_disc}")

    # Load cuts first, then build pass_cuts with the correct txbb_cut
    if sensitivity_dir is not None:
        txbb_cut, txtt_cut = extract_optimal_cuts_from_csv(
            Path(sensitivity_dir),
            signal,
            channel.key,
            combined_signals,
            bmin,
            use_ParT,
            do_vbf,
            test_mode=test_mode,
            tt_pres=tt_pres,
            overlapping_channels=overlapping_channels,
            sensitivity_disc_tag=sensitivity_disc_tag,
            ggf_modelname=ggf_modelname,
        )
    else:
        txbb_cut = channel.txbb_cut
        txtt_cut = channel.txtt_cut if use_ParT else channel.txtt_BDT_cut

    pass_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
        bb_disc: [txbb_cut, CUT_MAX_VAL],  # Use the loaded txbb_cut, not channel default
    }

    fail_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
    }

    if use_ParT:
        pass_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        pass_cuts[f"ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [txtt_cut, CUT_MAX_VAL]
        fail_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]
        fail_cuts[f"{bb_disc}+ttFatJetParTX{channel.tagger_label}vsQCDTop"] = [
            [-CUT_MAX_VAL, txbb_cut],
            [-CUT_MAX_VAL, txtt_cut],
        ]
    else:
        # if signal[:7] == "ggfbbtt":
        #     bdt_signal = "ggfbbtt"
        # else:
        bdt_signal = signal
        pass_cuts[_get_bdt_key(bdt_signal, channel, prefix_only=False)] = [txtt_cut, CUT_MAX_VAL]
        fail_cuts[f"{bb_disc}+{_get_bdt_key(bdt_signal, channel, prefix_only=False)}"] = [
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
