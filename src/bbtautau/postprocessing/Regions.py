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
from bbtautau.postprocessing.utils import PRODUCTION_CATEGORY_COL


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
        disc += f"_ttglopart{tt_glopart_cut:.4f}"
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
    a_priori_channels: bool = False,
    a_priori_production_mode: str | None = None,
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
        a_priori_channels: Matches ``SensitivityStudy.py --a-priori-channels`` -- only affects
            the path when ``do_vbf`` (ggf-only paths are unchanged regardless of this flag).
        a_priori_production_mode: Matches ``SensitivityStudy.py --a-priori-production-mode``
            (``"bdt"``/``"kinematic"``/``None``); only affects the path when ``do_vbf``.

    Returns:
        tuple: (txbb_cut, txtt_cut) - The optimal cuts for the given bmin
    """
    presel = _sensitivity_presel_dir(test_mode, tt_pres)
    disc = _sensitivity_disc_folder(use_ParT, sensitivity_disc_tag, ggf_modelname, tt_glopart_cut)
    if do_vbf:
        if a_priori_production_mode is not None:
            veto_mode_part = f"apriori_production_{a_priori_production_mode}"
        elif a_priori_channels:
            veto_mode_part = "apriori_channels_veto"
        else:
            veto_mode_part = "channel_ordering_veto"
        vbf_part = Path("do_vbf") / veto_mode_part
    else:
        vbf_part = Path("ggf_only")
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


def _get_sr_cuts(
    signal: str,
    channel: Channel,
    sensitivity_dir: Path = None,
    bmin: int = 10,
    use_ParT: bool = False,
    do_vbf: bool = False,
    bb_disc: str = "bbFatJetParTXbbvsQCDTop",
    combined_signals: str = "separate_signals",
    test_mode: bool = False,
    tt_pres: bool = False,
    overlapping_channels: bool = False,
    sensitivity_disc_tag: str | None = None,
    ggf_modelname: str | None = None,
    tt_glopart_cut: float | None = None,
    a_priori_channels: bool = False,
    a_priori_production_mode: str | None = None,
) -> tuple[dict, dict, dict, float]:
    """Resolve the SR cuts of one (signal, channel) and return
    ``(base_cuts, sr_pass_cuts, sr_veto_cuts, txbb_cut)``.

    ``base_cuts`` is the common preselection (pT cuts, plus the tt-mass window on the ParT path
    only, as in the SR). ``sr_pass_cuts`` is the SR pass without inter-channel/-signal vetoes;
    ``sr_veto_cuts`` its complement (base AND (bb < cut OR tt < cut [OR glopart < cut] [OR other
    production-mode category])), used to veto this SR in later regions. With
    ``a_priori_production_mode`` set, the SR pass also requires its own category (ggf/vbf, from
    ``signal``). ``bb_disc`` must already be the event-branch name.
    """
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
            a_priori_channels=a_priori_channels,
            a_priori_production_mode=a_priori_production_mode,
        )
    else:
        txbb_cut = channel.txbb_cut
        txtt_cut = channel.txtt_cut if use_ParT else channel.txtt_BDT_cut

    base_cuts = {
        "bbFatJetPt": [250, CUT_MAX_VAL],
        "ttFatJetPt": [200, CUT_MAX_VAL],
    }
    if use_ParT:
        base_cuts[channel.tt_mass_cut[0]] = channel.tt_mass_cut[1]

    pass_cuts = {
        **base_cuts,
        bb_disc: [txbb_cut, CUT_MAX_VAL],  # Use the loaded txbb_cut, not channel default
    }
    veto_cuts = dict(base_cuts)

    # veto = base AND (OR of the complements of each SR condition)
    if use_ParT:
        glopart_col = f"ttFatJetParTX{channel.tagger_label}vsQCDTop"
        pass_cuts[glopart_col] = [txtt_cut, CUT_MAX_VAL]
        veto_or_vars = f"{bb_disc}+{glopart_col}"
        veto_or_ranges = [[-CUT_MAX_VAL, txbb_cut], [-CUT_MAX_VAL, txtt_cut]]
    else:
        # if signal[:7] == "ggfbbtt":
        #     bdt_signal = "ggfbbtt"
        # else:
        bdt_signal = signal
        bdt_key = _get_bdt_key(bdt_signal, channel, prefix_only=False)
        pass_cuts[bdt_key] = [txtt_cut, CUT_MAX_VAL]
        veto_or_vars = f"{bb_disc}+{bdt_key}"
        veto_or_ranges = [[-CUT_MAX_VAL, txbb_cut], [-CUT_MAX_VAL, txtt_cut]]

        if tt_glopart_cut is not None:
            # Separate column from bdt_key (the raw ParT vsQCDTop tagger, not the BDT vsAll
            # score being optimized) -- an extra, independent AND condition, not a max() with
            # txtt_cut. Extend the veto's OR-sideband to the matching 3-var complement
            # (make_selection's "var1+var2+..." OR-cut syntax isn't limited to 2 variables).
            glopart_col = f"ttFatJetParTX{channel.tagger_label}vsQCDTop"
            pass_cuts[glopart_col] = [tt_glopart_cut, CUT_MAX_VAL]
            veto_or_vars += f"+{glopart_col}"
            veto_or_ranges.append([-CUT_MAX_VAL, tt_glopart_cut])

    if a_priori_production_mode is not None:
        # A priori production-mode split as a cut on the column set by
        # utils.assign_production_mode (1 = vbf, 0 = ggf): the SR requires its own category, and
        # its veto gains "OR in the other category".
        ggf_range, vbf_range = [-CUT_MAX_VAL, 0.5], [0.5, CUT_MAX_VAL]
        own, other = (vbf_range, ggf_range) if signal.startswith("vbf") else (ggf_range, vbf_range)
        pass_cuts[PRODUCTION_CATEGORY_COL] = own
        veto_or_vars += f"+{PRODUCTION_CATEGORY_COL}"
        veto_or_ranges.append(other)

    veto_cuts[veto_or_vars] = veto_or_ranges

    return base_cuts, pass_cuts, veto_cuts, txbb_cut


def _parse_bb_disc(bb_disc: str) -> str:
    """``ak8...`` CLI name -> ``bb...`` event-branch name."""
    if bb_disc.startswith("ak8"):
        return "bb" + bb_disc[len("ak8") :]
    raise ValueError(f"bb_disc must start with 'ak8', but got: {bb_disc}")


def get_sr_veto_cuts(signal: str, channel: Channel, **kwargs) -> dict:
    """Cuts selecting everything NOT in the (veto-free) SR pass of ``(signal, channel)``.
    Takes the same keyword arguments as ``get_selection_regions`` (minus the veto lists)."""
    kwargs["bb_disc"] = _parse_bb_disc(kwargs.get("bb_disc", "ak8FatJetParTXbbvsQCDTop"))
    return _get_sr_cuts(signal, channel, **kwargs)[2]


def get_selection_regions(
    signal: str,
    channel: Channel,
    vetoes: list[dict] = None,
    vr_vetoes: list[dict] = None,
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
    tt_glopart_cut: float | None = None,
    a_priori_channels: bool = False,
    a_priori_production_mode: str | None = None,
):
    """
    Get the selection regions for a given signal and channel:
    ``{"pass": SR pass, "fail": FR, "vrpass": VR pass}``
    (definitions in untracked_utils/notes/control_region_plan.md, "Region definitions").

    - ``pass``: bb > SR cut AND tt > SR cut [AND glopart tt > ``tt_glopart_cut``], AND NOT any
      region in ``vetoes`` (list of ``get_sr_veto_cuts`` dicts of SRs earlier in the ordering).
    - ``fail`` (FR): bb < L OR glopart tt < L, L = ``userConfig.VR_LOOSE_CUTS``.
    - ``vrpass`` (VR): bb >= L AND glopart tt >= L AND NOT any SR in ``vr_vetoes`` (the
      ``get_sr_veto_cuts`` dicts of the SRs of this channel to exclude; default: this SR only).

    "glopart tt" = ``ttFatJetParTX<channel>vsQCDTop``. FR and VR carry ``signal=False``.

    ``tt_glopart_cut``: extra fixed cut on ``ttFatJetParTX<channel>vsQCDTop``, matching
    ``SensitivityStudy.py --tt-glopart-cut``. No-op when ``use_ParT`` (the discriminant being
    optimized there already is this same column -- see ``main()``'s printed note in
    ``postprocessing.py``). Otherwise added as its own extra AND condition, separate from the
    BDT-based tt discriminant being optimized.
    """
    from bbtautau.userConfig import VR_LOOSE_CUTS

    bb_disc = _parse_bb_disc(bb_disc)

    base_cuts, pass_cuts, own_veto_cuts, txbb_cut = _get_sr_cuts(
        signal,
        channel,
        sensitivity_dir=sensitivity_dir,
        bmin=bmin,
        use_ParT=use_ParT,
        do_vbf=do_vbf,
        bb_disc=bb_disc,
        combined_signals=combined_signals,
        test_mode=test_mode,
        tt_pres=tt_pres,
        overlapping_channels=overlapping_channels,
        sensitivity_disc_tag=sensitivity_disc_tag,
        ggf_modelname=ggf_modelname,
        tt_glopart_cut=tt_glopart_cut,
        a_priori_channels=a_priori_channels,
        a_priori_production_mode=a_priori_production_mode,
    )

    if vetoes is not None:
        for veto in vetoes:
            pass_cuts.update(veto)
            # For the few events in the pass regions, we ignore their contribution to the fail regions that come after
            # Could add the logic in the future

    loose = VR_LOOSE_CUTS[channel.key]
    glopart_col = f"ttFatJetParTX{channel.tagger_label}vsQCDTop"
    sr_glopart_cut = pass_cuts[glopart_col][0] if glopart_col in pass_cuts else None
    if txbb_cut < loose["txbb"] or sr_glopart_cut is None or sr_glopart_cut < loose["txtt"]:
        print(
            f"WARNING: {signal} {channel.key} SR is not inside {{bb >= {loose['txbb']}, "
            f"glopart tt >= {loose['txtt']}}} (txbb cut {txbb_cut}, glopart cut "
            f"{sr_glopart_cut}): SR pass and FR overlap."
        )

    vr_regions = build_validation_regions(
        bb_disc=bb_disc,
        glopart_col=glopart_col,
        base_cuts=base_cuts,
        txbb_loose=loose["txbb"],
        txtt_loose=loose["txtt"],
        sr_vetoes=[own_veto_cuts] if vr_vetoes is None else vr_vetoes,
    )

    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts=pass_cuts,
            signal=True,
            label="Pass",
        ),
        **vr_regions,
    }

    return regions


def _and_cuts(cuts: dict, extra: dict):
    """AND ``extra`` into ``cuts`` in place. An OR-cut key already present with different
    ranges (e.g. two SRs on the same two ParT columns) is re-keyed with its variables in
    another order, which ``make_selection`` reads identically, instead of being overwritten."""
    from itertools import permutations

    for key, ranges in extra.items():
        if key not in cuts or cuts[key] == ranges:
            cuts[key] = ranges
            continue
        cvars = key.split("+")
        if len(cvars) < 2:
            raise ValueError(f"Conflicting cuts on {key}: {cuts[key]} vs {ranges}")
        for perm in permutations(range(len(cvars))):
            new_key = "+".join(cvars[i] for i in perm)
            new_ranges = [ranges[i] for i in perm]
            if new_key not in cuts or cuts[new_key] == new_ranges:
                cuts[new_key] = new_ranges
                break
        else:
            raise ValueError(f"Cannot AND another OR-cut on {key}")


def build_validation_regions(
    *,
    bb_disc: str,
    glopart_col: str,
    base_cuts: dict,
    txbb_loose: float,
    txtt_loose: float,
    sr_vetoes: list[dict],
) -> dict[str, Region]:
    """Build the FR and VR from explicit cut values.

    Geometry (see notes/control_region_plan.md, "Region definitions"):
      FR (``fail``)   = bb < txbb_loose OR glopart < txtt_loose
      VR (``vrpass``) = bb >= txbb_loose AND glopart >= txtt_loose AND NOT any SR in
                        ``sr_vetoes`` (each a ``get_sr_veto_cuts`` dict)

    ``bb_disc`` must already be the event-branch name (``bbFatJet...``, not ``ak8...``).
    Both regions carry ``signal=False`` so they are NOT blinded in data. Taking explicit loose
    values keeps this reusable by the yield-scan tooling.
    """
    fail_cuts = {
        **base_cuts,
        f"{bb_disc}+{glopart_col}": [[-CUT_MAX_VAL, txbb_loose], [-CUT_MAX_VAL, txtt_loose]],
    }
    vr_cuts = {
        **base_cuts,
        bb_disc: [txbb_loose, CUT_MAX_VAL],
        glopart_col: [txtt_loose, CUT_MAX_VAL],
    }
    for veto in sr_vetoes:
        _and_cuts(vr_cuts, veto)

    return {
        "fail": Region(cuts=fail_cuts, signal=False, label="Fail"),
        "vrpass": Region(cuts=vr_cuts, signal=False, label="Pass VR"),
    }
