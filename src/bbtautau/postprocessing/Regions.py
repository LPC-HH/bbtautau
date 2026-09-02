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
    tt_glopart_cut: float | None = None,
) -> str:
    """Match ``disc_tag`` in ``SensitivityStudy.get_plot_dir`` (ParT, BDT export name, or literal
    ``BDT``), including the ``_ttglopart<cut>`` suffix appended there when a glopart cut is set.
    An explicit ``sensitivity_disc_tag`` is a full override (e.g. a manually-renamed folder) and
    is used verbatim -- the suffix is only auto-appended when falling back to ``ggf_modelname``/
    ``\"BDT\"``."""
    if use_ParT:
        return "ParT"
    if sensitivity_disc_tag:
        return sensitivity_disc_tag
    disc = ggf_modelname or "BDT"
    if tt_glopart_cut is not None:
        disc += f"_ttglopart{tt_glopart_cut:.2f}"
    return disc


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
    tt_glopart_cut: float | None = None,
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
        tt_glopart_cut: Extra fixed ttFatJetParTX<channel>vsQCDTop cut used when the
            sensitivity study was run with ``--tt-glopart-cut`` (appends
            ``_ttglopart<cut:.2f>`` to the disc folder, matching
            ``SensitivityStudy.get_plot_dir``). Not used to build the cut itself here --
            see ``get_selection_regions`` for that.

    Returns:
        tuple: (txbb_cut, txtt_cut) - The optimal cuts for the given bmin
    """
    presel = _sensitivity_presel_dir(test_mode, tt_pres)
    disc = _sensitivity_disc_folder(use_ParT, sensitivity_disc_tag, ggf_modelname, tt_glopart_cut)
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
    control_region: bool = False,
    tt_glopart_cut: float | None = None,
):
    """
    Get the selection regions for a given signal and channel.

    ``tt_glopart_cut``: extra fixed cut on ``ttFatJetParTX<channel>vsQCDTop``, matching
    ``SensitivityStudy.py --tt-glopart-cut``. No-op when ``use_ParT`` (the discriminant being
    optimized there already is this same column -- see ``main()``'s printed note in
    ``postprocessing.py``). Otherwise added as its own extra AND condition, separate from the
    BDT-based tt discriminant being optimized.

    If ``control_region`` is True, return the CR (orthogonal annulus) regions instead
    of the nominal SR pass/fail. The SR cuts resolved below are reused as the SR-pass
    veto, and loose cuts come from ``userConfig.CR_LOOSE_CUTS``.
    See notes/control_region_plan.md.
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
            tt_glopart_cut=None if use_ParT else tt_glopart_cut,
        )
    else:
        txbb_cut = channel.txbb_cut
        txtt_cut = channel.txtt_cut if use_ParT else channel.txtt_BDT_cut

    # Control region: orthogonal annulus built from the SR cuts (as veto) + loose cuts.
    if control_region:
        from bbtautau.userConfig import CR_LOOSE_CUTS

        loose = CR_LOOSE_CUTS[channel.key]
        return build_control_regions(
            channel,
            signal,
            use_ParT=use_ParT,
            bb_disc=bb_disc,
            txbb_sr=txbb_cut,
            txtt_sr=txtt_cut,
            txbb_loose=loose["txbb"],
            txtt_loose=loose["txtt"],
        )

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
        bdt_key = _get_bdt_key(bdt_signal, channel, prefix_only=False)
        pass_cuts[bdt_key] = [txtt_cut, CUT_MAX_VAL]
        fail_or_vars = f"{bb_disc}+{bdt_key}"
        fail_or_ranges = [[-CUT_MAX_VAL, txbb_cut], [-CUT_MAX_VAL, txtt_cut]]

        if tt_glopart_cut is not None:
            # Separate column from bdt_key (the raw ParT vsQCDTop tagger, not the BDT vsAll
            # score being optimized) -- an extra, independent AND condition, not a max() with
            # txtt_cut. Extend the fail region's OR-sideband to the matching 3-var complement
            # (make_selection's "var1+var2+..." OR-cut syntax isn't limited to 2 variables).
            glopart_col = f"ttFatJetParTX{channel.tagger_label}vsQCDTop"
            pass_cuts[glopart_col] = [tt_glopart_cut, CUT_MAX_VAL]
            fail_or_vars += f"+{glopart_col}"
            fail_or_ranges.append([-CUT_MAX_VAL, tt_glopart_cut])

        fail_cuts[fail_or_vars] = fail_or_ranges

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


def _tt_disc_key(channel: Channel, signal: str, use_ParT: bool) -> str:
    """tt discriminant column name for a channel (ParT or per-channel BDT). Mirrors the
    disc-key choice inside ``get_selection_regions`` so SR and CR stay consistent."""
    if use_ParT:
        return f"ttFatJetParTX{channel.tagger_label}vsQCDTop"
    return _get_bdt_key(signal, channel, prefix_only=False)


def build_control_regions(
    channel: Channel,
    signal: str,
    *,
    use_ParT: bool,
    bb_disc: str,
    txbb_sr: float,
    txtt_sr: float,
    txbb_loose: float,
    txtt_loose: float,
) -> dict[str, Region]:
    """Build the CR (orthogonal annulus) regions from explicit cut values.

    Geometry (see notes/control_region_plan.md):
      CR-pass = (bb > txbb_loose AND tt > txtt_loose) AND (bb < txbb_sr OR tt < txtt_sr)
                i.e. loose-pass with the SR-pass vetoed out -> orthogonal to the SR.
      CR-fail = bb < txbb_loose OR tt < txtt_loose  (anti-tag below the loose cuts)

    ``bb_disc`` must already be the event-branch name (``bbFatJet...``, not ``ak8...``).
    Both regions carry ``signal=False`` so the CR is NOT blinded (we want to see data
    across the full mass range). Taking explicit loose values keeps this reusable by the
    yield-scan tooling.
    """
    tt_disc = _tt_disc_key(channel, signal, use_ParT)

    base = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
    }
    # tt-mass window is only applied on the ParT path in the SR; mirror that here.
    if use_ParT:
        base[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]

    pass_cuts = {
        **base,
        bb_disc: [txbb_loose, CUT_MAX_VAL],
        tt_disc: [txtt_loose, CUT_MAX_VAL],
        # SR-pass veto (OR): bb < txbb_sr OR tt < txtt_sr
        f"{bb_disc}+{tt_disc}": [[-CUT_MAX_VAL, txbb_sr], [-CUT_MAX_VAL, txtt_sr]],
    }
    fail_cuts = {
        **base,
        # anti-tag below loose (OR): bb < txbb_loose OR tt < txtt_loose
        f"{bb_disc}+{tt_disc}": [[-CUT_MAX_VAL, txbb_loose], [-CUT_MAX_VAL, txtt_loose]],
    }

    return {
        "pass": Region(cuts=pass_cuts, signal=False, label="Pass CR"),
        "fail": Region(cuts=fail_cuts, signal=False, label="Fail CR"),
    }
