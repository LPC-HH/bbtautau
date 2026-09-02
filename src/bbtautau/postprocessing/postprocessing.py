"""
Postprocessing functions for bbtautau.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import logging
import pickle
from pathlib import Path

import hist
import matplotlib as mpl
import numpy as np
import pandas as pd
from boostedhh import hh_vars, utils
from boostedhh.hh_vars import data_key
from boostedhh.utils import Sample, ShapeVar, add_bool_arg
from hist import Hist

import bbtautau.postprocessing.utils as putils
from bbtautau.postprocessing import Regions, Samples, plotting
from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.postprocessing.Samples import (
    CHANNELS,
    SAMPLES,
    SIGNALS,
    sig_keys_vbf,
)
from bbtautau.postprocessing.utils import load_data_channel
from bbtautau.userConfig import (
    CHANNEL_ORDERING,
    MODEL_DIR,
    SHAPE_VAR,
    SIGNAL_ORDERING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("boostedhh.utils")


control_plot_vars = (
    [
        ShapeVar(var=f"{jet}FatJetPt", label=rf"$p_T^{{{jlabel}}}$ [GeV]", bins=[20, 250, 1250])
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
    ]
    + [
        ShapeVar(
            var=f"{jet}FatJetCAglobalParT_massVisApplied",
            label=rf"$m^{{{jlabel}}}$ [GeV]",
            bins=[20, 50, 300],
        )
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
    ]
    + [
        ShapeVar(
            var=f"{jet}FatJetParTmassVisApplied",
            label=rf"$m^{{{jlabel}}}$ [GeV]",
            bins=[20, 50, 300],
        )
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
    ]
    + [
        ShapeVar(
            var=f"{jet}FatJetParTX{tautau}vsQCDTop",
            label=rf"ParT X{tautau}vsQCDTop {jlabel}",
            bins=[20, 0, 1],
        )
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
        for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]
    ]
    + [
        ShapeVar(
            var="METPt", label=r"$p^{miss}_T$ [GeV]", bins=[20, 0, 300]
        ),  # METPt is used for resel samples
        # ShapeVar(var="MET_phi", label=r"$\phi^{miss}$", bins=[20, -3.2, 3.2]),
    ]
    + [
        ShapeVar(var=f"ak8FatJetPt{i}", label=rf"$p_T^{{j{i + 1}}}$ [GeV]", bins=[20, 250, 1250])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetMsd{i}", label=rf"$m_{{SD}}^{{j{i + 1}}}$ [GeV]", bins=[20, 0, 300])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetEta{i}", label=rf"$\eta^{{j{i + 1}}}$", bins=[20, -2.5, 2.5])
        for i in range(3)
    ]
    + [
        ShapeVar(var=f"ak8FatJetPhi{i}", label=rf"$\phi^{{j{i + 1}}}$", bins=[20, -3.2, 3.2])
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetPNetmassLegacy{i}",
            label=rf"PNet Legacy $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetParTmassResApplied{i}",
            label=rf"ParT Resonance $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    + [
        ShapeVar(
            var=f"ak8FatJetParTmassVisApplied{i}",
            label=rf"ParT Visable $m_{{reg}}^{{j{i + 1}}}$",
            bins=[20, 50, 300],
        )
        for i in range(3)
    ]
    # ak8FatJetParXbbvsQCD
    + [
        ShapeVar(
            var=f"ak8FatJetParTXbbvsQCD{i}",
            label=rf"ParT XbbvsQCD j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetParTXbbvsQCDTop
    + [
        ShapeVar(
            var=f"ak8FatJetParTXbbvsQCDTop{i}",
            label=rf"ParT XbbvsQCDTop j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # ak8FatJetPNetXbbvsQCDLegacy
    + [
        ShapeVar(
            var=f"ak8FatJetPNetXbbvsQCDLegacy{i}",
            label=rf"PNet Legacy XbbvsQCD j{i+1}",
            bins=[20, 0, 1],
        )
        for i in range(3)
    ]
    # VBF dijet variables (padded/PAD_VAL for events with <2 VBF jets, falls outside
    # these ranges and lands in over/underflow, same convention as the ak8FatJet* vars above)
    + [
        ShapeVar(var="VBFMassjj", label=r"$m_{jj}^{VBF}$ [GeV]", bins=[25, 0, 3000]),
        # signed (jets are pT-ordered, not eta-ordered), so range is symmetric
        ShapeVar(var="VBFJetDeltaEta", label=r"$\Delta\eta_{jj}^{VBF}$", bins=[20, -10, 10]),
    ]
    #  nElectrons
    + [ShapeVar(var="nElectrons", label=r"Number of Electrons", bins=[3, 0, 3])]
    #  nMuons
    + [ShapeVar(var="nMuons", label=r"Number of Muons", bins=[3, 0, 3])]
    #  nTaus
    + [ShapeVar(var="nTaus", label=r"Number of Taus", bins=[3, 0, 3])]
    #  nBoostedTaus
    + [ShapeVar(var="nBoostedTaus", label=r"Number of Boosted Taus", bins=[3, 0, 3])]
)

# Control-plot variables whose values pile up near 1 (discriminator scores) are nicer to look
# at under x -> -log(1 - x + eps). Edit this set to control which vars get transformed.
TRANSFORM_EPS = 1e-4
TRANSFORM_VARS = {sv.var for sv in control_plot_vars if "vsQCD" in sv.var}


def _neglog_transform(x, eps: float = TRANSFORM_EPS):
    """x -> -log(1 - x + eps); eps only for numerical stability near x = 1."""
    return -np.log(np.maximum(1.0 - x + eps, 1e-12))


# fitting on bb regressed mass
shape_vars = [
    ShapeVar(
        SHAPE_VAR["name"],
        r"$m^{bb}_\mathrm{Reg}$ [GeV]",
        (SHAPE_VAR["nbins"], *SHAPE_VAR["range"]),
        reg=True,
        blind_window=SHAPE_VAR["blind_window"],
    )
]


def main(args: argparse.Namespace):
    """
    Main function that handles multiple bmin values.
    Data is loaded once, but templates are generated for each bmin value with updated cuts.
    """
    # For templates we currently enforce processing one year at a time
    years = args.years
    if len(years) != 1:
        raise ValueError(
            "Template maker currently supports exactly one year at a time. "
            "Please pass a single year via `--years YEAR`. "
            "Multi-year concatenation is supported for --control-plots only."
        )
    year_label = years[0]

    # Convert single bmin value to list for backward compatibility
    if isinstance(args.bmin, int):
        args.bmin = [args.bmin]

    print(f"Processing bmin values: {args.bmin}")

    # These are the regions to use: either only ggf or both ggf and vbf.
    # BSM ggf samples (ggfbbtt-kl0p00, etc.) have no separate regions - they contribute to
    # pass_ggfbbtt (using ggfbbtt cuts) and pass_vbfbbtt (using vbfbbtt cuts), like vbfbbtt-k2v0.
    signal_regions = copy.deepcopy(SIGNAL_ORDERING) if args.do_vbf else ["ggfbbtt"]

    if args.sigs is None:
        args.sigs = SIGNALS

    if args.bgs is None:
        args.bgs = {bkey: b for bkey, b in SAMPLES.items() if b.get_type() == "bg"}

    CHANNEL = CHANNELS[args.channel]

    if args.a_priori_channels and args.do_vbf:
        raise ValueError(
            "--a-priori-channels does not yet support --do-vbf: the cross-signal (ggf-vs-vbf) "
            "veto path hasn't been designed/tested against a priori channel splitting."
        )

    if args.use_ParT and args.tt_glopart_cut is not None:
        print(
            "--tt-glopart-cut is a no-op in --use_ParT mode (the optimized tt discriminant there "
            "already is ttFatJetParTX<channel>vsQCDTop); ignoring."
        )

    if args.sensitivity_dir is not None:
        # Fail fast on a bad --ggf-modelname/--sensitivity-disc-tag/--tt-glopart-cut combination
        # by resolving the same CSV lookup used later for real, before the slow (~1hr/year) data
        # load below -- reuses extract_optimal_cuts_from_csv itself so this can't drift from the
        # real resolution logic.
        for signal_region in signal_regions:
            for bmin in args.bmin:
                Regions.extract_optimal_cuts_from_csv(
                    args.sensitivity_dir,
                    signal_region,
                    CHANNEL.key,
                    args.combined_signals,
                    bmin,
                    args.use_ParT,
                    args.do_vbf,
                    test_mode=args.test_mode,
                    tt_pres=args.tt_pres,
                    overlapping_channels=args.overlapping_channels,
                    sensitivity_disc_tag=args.sensitivity_disc_tag,
                    ggf_modelname=args.ggf_modelname,
                    tt_glopart_cut=args.tt_glopart_cut,
                )

    models = None
    if not args.use_ParT:
        models = [args.ggf_modelname] + ([args.vbf_modelname] if args.do_vbf else [])

    events_dict, cutflow = load_data_channel(
        years=years,
        signals=args.sigs,
        channel=CHANNEL,
        test_mode=args.test_mode,
        tt_pres=args.tt_pres,
        models=models,
        cutflow=True,
        load_bgs=True,
        restrict_signal_to_channel_gen=args.gen_split,
    )

    # Keep dictionary structure consistent with legacy code, working out templates one year at a time
    events_dict = events_dict[year_label]

    if args.a_priori_channels:
        # Classify every sample (data/background/signal alike) by lepton content and keep only
        # this channel's events -- replaces the CHANNEL_ORDERING veto chain in get_templates.
        # copy_from_selection doesn't preserve e_mask/m_mask, but that's fine here: BDT
        # prediction (the only consumer of ttMuon*/ttElectron* features) already ran inside
        # load_data_channel above, and nothing downstream touches e_mask/m_mask directly.
        putils.assign_lepton_channel(events_dict)
        for key, sample in events_dict.items():
            keep = sample.get_var("lepton_channel") == CHANNEL.key
            events_dict[key] = sample.copy_from_selection(keep)
    args.sigs = {s + CHANNEL.key: SAMPLES[s + CHANNEL.key] for s in args.sigs}
    systematics: dict[str, dict] = {}
    systematics_path: Path | None = None
    if args.template_dir:
        systematics_path = args.template_dir / f"{year_label}_systematics.pkl"

        if systematics_path.exists() and not args.override_systs:
            try:
                with systematics_path.open("rb") as syst_file:
                    loaded_systematics = pickle.load(syst_file)
                if isinstance(loaded_systematics, dict):
                    systematics = copy.deepcopy(loaded_systematics)
                else:
                    logger.warning(
                        "Ignoring systematics file %s with unexpected type %s",
                        systematics_path,
                        type(loaded_systematics),
                    )
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
                logger.warning("Failed to load systematics from %s: %s", systematics_path, exc)

    systematics.setdefault(year_label, {})

    # Now process each bmin value
    for bmin in args.bmin:
        print(f"\n{'='*60}")
        print(f"Processing bmin = {bmin}")
        print(f"{'='*60}")

        print(f"\nGenerating templates for bmin={bmin}, signal regions={signal_regions}")

        for signal_key in signal_regions:

            # Create bmin-specific directories ("control" subdir keeps CR templates
            # from colliding with the nominal SR ones).
            cr_seg = "control" if args.control_region else ""
            template_dir_bmin = (
                args.template_dir
                / cr_seg
                / f"bmin_{bmin}"
                / signal_key
                / (CHANNEL.key if args.template_dir else "")
            )
            plot_dir_bmin = (
                args.plot_dir
                / cr_seg
                / f"bmin_{bmin}"
                / signal_key
                / (CHANNEL.key if args.plot_dir else "")
            )

            if template_dir_bmin:
                (template_dir_bmin / "cutflows" / year_label).mkdir(parents=True, exist_ok=True)
            if plot_dir_bmin:
                plot_dir_bmin.mkdir(parents=True, exist_ok=True)

            templates = get_templates(
                events_dict,  # Same data for all bmin values
                year_label,
                args.sigs,
                args.bgs,
                CHANNEL,  # Updated channel with new cuts
                signal_key,
                signal_regions,
                shape_vars,
                {},  # TODO: systematics
                sig_scale_dict={
                    f"ggfbbtt{CHANNEL.key}": 300,
                    f"ggfbbtt-kl0p00{CHANNEL.key}": 300,
                    f"ggfbbtt-kl2p45{CHANNEL.key}": 300,
                    f"ggfbbtt-kl5p00{CHANNEL.key}": 300,
                    f"vbfbbtt{CHANNEL.key}": 500,
                    f"vbfbbtt-k2v0{CHANNEL.key}": 400,
                    f"vbfbbtt-kv1p74-k2v1p37-kl14p4{CHANNEL.key}": 400,
                    f"vbfbbtt-kvm0p758-k2v1p44-klm19p3{CHANNEL.key}": 400,
                    f"vbfbbtt-kvm0p962-k2v0p959-klm1p43{CHANNEL.key}": 400,
                    f"vbfbbtt-kvm1p6-k2v2p72-klm1p36{CHANNEL.key}": 400,
                },
                # prev_cutflow=cutflow, Tried to add this but seems to not work. May want to fix later
                template_dir=template_dir_bmin,
                plot_dir=plot_dir_bmin,
                show=False,
                a_priori_channels=args.a_priori_channels,
                selection_region_kwargs={
                    "sensitivity_dir": args.sensitivity_dir,
                    "bmin": bmin,  # Use loop variable, not args.bmin
                    "combined_signals": args.combined_signals,
                    "use_ParT": args.use_ParT,
                    "do_vbf": args.do_vbf,
                    "bb_disc": args.bb_disc,
                    "test_mode": args.test_mode,
                    "tt_pres": args.tt_pres,
                    "overlapping_channels": args.overlapping_channels,
                    "sensitivity_disc_tag": args.sensitivity_disc_tag,
                    "ggf_modelname": args.ggf_modelname,
                    "control_region": args.control_region,
                    "tt_glopart_cut": args.tt_glopart_cut,
                },
            )

            if args.template_dir:
                print(f"Saving templates for bmin={bmin}")
                template_file = template_dir_bmin / f"{year_label}_templates.pkl"
                save_templates(
                    templates,
                    template_file,
                    args.blinded,
                    shape_vars,
                )

            # TODO:
            # if systematics_path is not None:
            #     try:
            #         systematics_path.parent.mkdir(parents=True, exist_ok=True)
            #         with systematics_path.open("wb") as syst_file:
            #             pickle.dump(systematics, syst_file)
            #         print("Saved systematics to", systematics_path)
            #     except OSError as exc:
            #         logger.warning("Failed to save systematics to %s: %s", systematics_path, exc)

            del templates
            gc.collect()

        print(f"Completed processing for bmin={bmin}")

    print(f"\nCompleted processing all bmin values: {args.bmin}")


def control_plots(
    events_dict: dict[str, LoadedSample],
    channel: Channel,
    sigs: dict[str, Sample],
    bgs: dict[str, Sample],
    control_plot_vars: list[ShapeVar],
    plot_dir: Path,
    year: str,
    weight_key: str = "finalWeight",
    hists: dict = None,
    cutstr: str = "",
    cutlabel: str = "",
    title: str = None,
    selection: dict[str, np.ndarray] = None,
    sig_scale_dict: dict[str, float] = None,
    combine_pdf: bool = True,
    plot_ratio: bool = True,
    plot_significance: bool = False,
    same_ylim: bool = False,
    show: bool = False,
    log: tuple[bool, str] = "both",
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.
        sig_splits: split up signals into different plots (in case there are too many for one)
        HEM2d: whether to plot 2D hists of FatJet phi vs eta for bb and VV jets as a check for HEM cleaning.
        plot_ratio: whether to plot the data/MC ratio.
        plot_significance: whether to plot the significance as well as the ratio plot.
        same_ylim: whether to use the same y-axis limits for all plots.
        log: True or False if plot on log scale or not - or "both" if both.
    """

    from PyPDF2 import PdfMerger

    if hists is None:
        hists = {}
    if sig_scale_dict is None:
        sig_scale_dict = {sig_key: 2e4 for sig_key in sigs}
        for key in sig_scale_dict:
            if "vbfbbtt" in key and "-" not in key:
                sig_scale_dict[key] = 1e6

    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            # For flagged vars, fill -log(1 - x + eps) over a matching (transformed) axis so that
            # ratioHistPlot — which just plots whatever axis the Hist carries — shows the stretched
            # view. The hist stays keyed by the original var name.
            if shape_var.var in TRANSFORM_VARS:
                edges = shape_var.axis.edges
                tlo, thi = _neglog_transform(np.array([edges[0], edges[-1]]))
                hist_shape_var = ShapeVar(
                    var=shape_var.var,
                    label=shape_var.label + r" [$-\log(1-x+\epsilon)$]",
                    bins=[shape_var.axis.size, float(tlo), float(thi)],
                    significance_dir=shape_var.significance_dir,
                )
                transform = _neglog_transform
            else:
                hist_shape_var = shape_var
                transform = None

            hists[shape_var.var] = putils.singleVarHist(
                events_dict,
                hist_shape_var,
                channel,
                weight_key=weight_key,
                selection=selection,
                transform=transform,
            )

    ylim = (np.max([h.values() for h in hists.values()]) * 1.05) if same_ylim else None

    with (plot_dir / "hists.pkl").open("wb") as f:
        pickle.dump(hists, f)

    do_log = [True, False] if log == "both" else [log]

    for log, logstr in [(False, ""), (True, "_log")]:
        if log not in do_log:
            continue

        merger_control_plots = PdfMerger()

        for shape_var in control_plot_vars:
            pylim = np.max(hists[shape_var.var].values()) * 1.4 if ylim is None else ylim

            name = f"{plot_dir}/{cutstr}{shape_var.var}{logstr}.pdf"
            plotting.ratioHistPlot(
                hists[shape_var.var],
                year,
                channel,
                list(sigs.keys()),
                list(bgs.keys()),
                name=name,
                title=title,
                sig_scale_dict=sig_scale_dict if not log else None,
                plot_significance=plot_significance,
                significance_dir=shape_var.significance_dir,
                cutlabel=cutlabel,
                show=show,
                log=log,
                plot_data=True,
                ylim=pylim if not log else 1e3,
                plot_ratio=plot_ratio,
                cmslabel="Work in progress",
                leg_args={"fontsize": 18},
            )
            merger_control_plots.append(name)

        if combine_pdf:
            merger_control_plots.write(f"{plot_dir}/{cutstr}{year}{logstr}_ControlPlots.pdf")

        merger_control_plots.close()

    return hists


def run_control_plots(args: argparse.Namespace) -> None:

    # Decide which years to load; allow combining years per channel
    if args.years == ["all"]:
        years = list(hh_vars.years)
        year_label = "all"
    elif not isinstance(args.years, list):
        raise ValueError("Cannot process multiple years at once other than 'all'")
    else:
        years = args.years
        year_label = args.years

    if args.sigs is None:

        # args.sigs = SM_SIGNALS
        args.sigs = sig_keys_vbf

    if args.bgs is None:
        args.bgs = {bkey: b for bkey, b in SAMPLES.items() if b.get_type() == "bg"}
    else:
        # CLI provides a list[str]; normalize to the dict form used elsewhere in this module
        args.bgs = {bkey: SAMPLES[bkey] for bkey in args.bgs}

    CHANNEL = CHANNELS[args.channel]

    models = None
    if not args.use_ParT:
        models = [args.ggf_modelname] + ([args.vbf_modelname] if args.do_vbf else [])

    events_dict = load_data_channel(
        years=years,
        signals=args.sigs,
        channel=CHANNEL,
        test_mode=args.test_mode,
        tt_pres=args.tt_pres,
        models=models,
        cutflow=False,
        load_bgs=True,
        restrict_signal_to_channel_gen=args.gen_split,
    )

    if len(years) > 1:
        events_dict = putils.concatenate_years(events_dict, years=years)
    else:
        events_dict = events_dict[years[0]]
    sigs = {s + CHANNEL.key: SAMPLES[s + CHANNEL.key] for s in args.sigs}
    bgs = args.bgs

    if not args.plot_dir:
        raise ValueError("--plot-dir is required for --control-plots")

    # Filter control plot vars if requested
    if args.control_plot_vars:
        requested = set(args.control_plot_vars)
        selected_vars = [sv for sv in control_plot_vars if sv.var in requested]
        missing = requested - {sv.var for sv in selected_vars}
        if missing:
            raise ValueError(
                "Unknown --control-plot-vars: "
                + ", ".join(sorted(missing))
                + ". Available: "
                + ", ".join(sorted({sv.var for sv in control_plot_vars}))
            )
    else:
        selected_vars = list(control_plot_vars)

    # Default: inclusive control plots. With --sr, apply the SR pass cut (resolved from --bmin).
    selection = None
    cutstr = ""
    cutlabel = "Inclusive"
    title = f"{CHANNEL.label} inclusive control plots"

    if args.sr:
        # Resolve the SR pass region for the ggf signal and turn its cuts into a per-sample mask.
        bmin = args.bmin[0] if isinstance(args.bmin, list) else args.bmin
        if isinstance(args.bmin, list) and len(args.bmin) > 1:
            print(f"--sr: multiple --bmin values given {args.bmin}; using the first ({bmin}).")
        if args.sensitivity_dir is None:
            print(
                "--sr: --sensitivity-dir not set; SR cuts fall back to Samples.py defaults and "
                "--bmin has no effect."
            )

        signal = "ggfbbtt"
        selection_regions = Regions.get_selection_regions(
            signal,
            CHANNEL,
            sensitivity_dir=args.sensitivity_dir,
            bmin=bmin,
            combined_signals=args.combined_signals,
            use_ParT=args.use_ParT,
            do_vbf=args.do_vbf,
            bb_disc=args.bb_disc,
            test_mode=args.test_mode,
            tt_pres=args.tt_pres,
            overlapping_channels=args.overlapping_channels,
            sensitivity_disc_tag=args.sensitivity_disc_tag,
            ggf_modelname=args.ggf_modelname,
            tt_glopart_cut=args.tt_glopart_cut,
        )
        pass_region = selection_regions["pass"]
        selection, _ = utils.make_selection(pass_region.cuts, events_dict)

        cutstr = f"sr_bmin{bmin}_"
        cutlabel = f"SR pass (Bmin={bmin})"
        title = f"{CHANNEL.label} SR pass control plots (Bmin={bmin})"

    plot_dir_cp = args.plot_dir
    plot_dir_cp.mkdir(parents=True, exist_ok=True)

    control_plots(
        events_dict=events_dict,
        channel=CHANNEL,
        sigs=sigs,
        bgs=bgs,
        control_plot_vars=selected_vars,
        plot_dir=plot_dir_cp,
        year=year_label,  # very inefficient rn
        selection=selection,
        cutstr=cutstr,
        cutlabel=cutlabel,
        title=title,
        combine_pdf=True,
        plot_ratio=True,
        show=False,
        log="both",
    )


def get_templates(
    events_dict: dict[str, LoadedSample],
    year: str,
    sig_keys: list[str],  # list of signal samples to load and plot
    bg_keys: list[str],
    channel: Channel,
    signal: str,  # identify which signal our region corresponds to, and what tagger we use to select
    signal_regions: list[
        str
    ],  # all the signal regions we are including (ggf or ggf+vbf); used to do the veto properly
    shape_vars: list[ShapeVar],
    systematics: dict,  # noqa: ARG001
    template_dir: Path = "",
    plot_dir: Path = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    plot_sig_keys: list[str] = None,
    sig_scale_dict: dict = None,
    weight_shifts: dict = None,
    jshift: str = "",
    plot_shifts: bool = False,
    pass_ylim: int = None,
    fail_ylim: int = None,
    blind: bool = True,
    blind_pass: bool = False,
    plot_data: bool = True,
    show: bool = False,
    selection_region_kwargs: dict = None,
    a_priori_channels: bool = False,
) -> dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) TODO: Applies the Txbb scale factor in the pass region,
    (3) TODO: Calculates trigger uncertainty,
    (4) TODO: Calculates weight variations if ``weight_shifts`` is not empty (and ``jshift`` is ""),
    (5) TODO: Takes JEC / JSMR shift into account if ``jshift`` is not empty,
    (6) Saves a plot of each (if ``plot_dir`` is not "").

    Args:
        selection_region (Dict[str, Dict]): Dictionary of ``Region``s including cuts and labels.
        bg_keys (list[str]): background keys to plot.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """
    import time

    start = time.time()

    if weight_shifts is None:
        weight_shifts = {}

    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates = {}

    # do TXbb SFs + uncs. for signals and Hbb samples only
    # txbb_samples = sig_keys + [key for key in bg_keys if key in hbb_bg_keys]

    # Inter-channel/-signal vetoes enforce SR orthogonality; the CR is a per-channel
    # validation region, so skip them when building control-region templates.
    control_region = bool((selection_region_kwargs or {}).get("control_region", False))

    vetoes = []
    found = False
    # veto all channels/signals earlier in the ordering than the current one. Skipped under
    # a_priori_channels: channel membership is already enforced upstream (see main()), and
    # (ggf-only, enforced there too) there's no signal-ordering axis left to veto.
    if not control_region and not a_priori_channels:
        for channel_iter in CHANNEL_ORDERING:
            for signal_iter in signal_regions:
                if channel_iter == channel.key and signal_iter == signal:
                    found = True
                    break
                vetoes.append(
                    Regions.get_selection_regions(
                        signal_iter, CHANNELS[channel_iter], **selection_region_kwargs
                    )
                )
            if found:
                break

    # Now a pass/fail region is defined for ggf and vbf. In each we will load all signal samples
    # Apply vetoes from regions that were optimized earlier in the ordering
    selection_regions = Regions.get_selection_regions(
        signal, channel, vetoes=vetoes, **selection_region_kwargs
    )

    for rname, region in selection_regions.items():
        pass_region = rname.startswith("pass")

        print(f"{rname} Region: {time.time() - start:.2f}")

        if not do_jshift:
            print(rname)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region.cuts,
            events_dict,
            prev_cutflow=prev_cutflow,
            jshift=jshift,
            weight_key=weight_key,
        )
        print(f"Selection: {time.time() - start:.2f}")

        if template_dir != "":
            cf.to_csv(f"{template_dir}/cutflows/{year}/{rname}_cutflow{jlabel}.csv")

        # trigger uncertainties
        # if not do_jshift:
        #     systematics[year][rname] = {}
        #     total, total_err = corrections.get_uncorr_trig_eff_unc(events_dict, bb_masks, year, sel)
        #     systematics[year][rname]["trig_total"] = total
        #     systematics[year][rname]["trig_total_err"] = total_err
        #     print(f"Trigger SF Unc.: {total_err / total:.3f}\n")

        # ParticleNetMD Txbb and ParT LP SFs
        sig_events = {}
        for sig_key in sig_keys:
            lsample = events_dict[sig_key]
            sig_events[sig_key] = lsample.copy_from_selection(sel[sig_key], do_deepcopy=True)

            # if region.signal:
            #     corrections.apply_txbb_sfs(
            #         sig_events[sig_key], sig_bb_mask, year, weight_key, do_shifts=not do_jshift
            #     )

            #     print(f"Txbb SFs: {time.time() - start:.2f}")

        # set up samples
        hist_samples = list(events_dict.keys())

        # Extra, diagnostic-only categories: split each signal sample's (already
        # reco-selected) events by true tau decay mode, so we can inspect whether
        # cross-channel-migrated/contaminating signal has a different shape. These are
        # *not* looked at by CreateDatacard.py (mc_samples/sig_keys there only ever
        # reference the base `skey` names below), so they can't affect the fit -- purely
        # additive bookkeeping. See "Signal channel splitting" in CLAUDE.md.
        for sig_key in sig_keys:
            hist_samples += [f"{sig_key}__true{origin}" for origin in putils.TRUTH_ORIGINS]

        # if not do_jshift:
        #     # add all weight-based variations to histogram axis
        #     for shift in ["down", "up"]:
        #         if region.signal:
        #             for sig_key in sig_keys:
        #                 hist_samples.append(f"{sig_key}_txbb_{shift}")

        #         for wshift, wsyst in weight_shifts.items():
        #             # if year in wsyst.years:
        #             # add to the axis even if not applied to this year to make it easier to sum later
        #             for wsample in wsyst.samples:
        #                 if wsample in events_dict:
        #                     hist_samples.append(f"{wsample}_{wshift}_{shift}")

        # histograms
        h = Hist(
            hist.axis.StrCategory(hist_samples + [data_key], name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="weight",
        )

        # fill histograms
        for skey, lsample in events_dict.items():
            if skey in sig_keys:
                sample = sig_events[skey]
            else:
                sample = lsample.copy_from_selection(sel[skey])

            if not len(sample.events):
                continue

            fill_data = utils.get_fill_data(
                sample, shape_vars, jshift=jshift if sample.sample.isData else None
            )
            weight = sample.get_var(weight_key)

            # breakpoint()
            h.fill(Sample=skey, **fill_data, weight=weight)

            if skey in sig_keys:
                # diagnostic truth-origin breakdown (nominal weight only, no shape
                # systematics) -- see hist_samples note above
                for origin, mask in putils.truth_origin_masks(sample).items():
                    sub_sample = sample.copy_from_selection(mask)
                    if not len(sub_sample.events):
                        continue
                    sub_fill_data = utils.get_fill_data(sub_sample, shape_vars)
                    h.fill(
                        Sample=f"{skey}__true{origin}",
                        **sub_fill_data,
                        weight=sub_sample.get_var(weight_key),
                    )

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if skey in wsyst.samples and year in wsyst.years:
                        if wshift not in ["scale", "pdf"]:
                            # fill histogram with weight variations
                            for shift_key, shift in [("Down", "down"), ("Up", "up")]:
                                h.fill(
                                    Sample=f"{skey}_{wshift}_{shift}",
                                    **fill_data,
                                    weight=sample.get_var(f"weight_{wshift}{shift_key}"),
                                )
                        else:
                            # get histograms for all QCD scale and PDF variations
                            whists = utils.get_qcdvar_hists(sample, shape_vars, fill_data, wshift)

                            if wshift == "scale":
                                # renormalization / factorization scale uncertainty is the max/min envelope of the variations
                                shape_up = np.max(whists.values(), axis=0)
                                shape_down = np.min(whists.values(), axis=0)
                            else:
                                # pdf uncertainty is the norm of each variation (corresponding to 103 eigenvectors) - nominal
                                nom_vals = h[sample, ...].values()
                                abs_unc = np.linalg.norm(
                                    (whists.values() - nom_vals), axis=0
                                )  # / np.sqrt(103)
                                # cap at 100% uncertainty
                                rel_unc = np.clip(abs_unc / nom_vals, 0, 1)
                                shape_up = nom_vals * (1 + rel_unc)
                                shape_down = nom_vals * (1 - rel_unc)

                            h.values()[
                                utils.get_key_index(h, f"{skey}_{wshift}_up"), ...
                            ] = shape_up
                            h.values()[
                                utils.get_key_index(h, f"{skey}_{wshift}_down"), ...
                            ] = shape_down

        print(f"Histograms: {time.time() - start:.2f}")

        # sum data histograms
        data_hist = sum(h[skey, ...] for skey in channel.data_samples)
        h.view(flow=True)[utils.get_key_index(h, data_key)].value = data_hist.values(flow=True)
        h.view(flow=True)[utils.get_key_index(h, data_key)].variance = data_hist.variances(
            flow=True
        )

        print(h)

        if region.signal and blind:
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        # if region.signal and not do_jshift:
        #     for sig_key in sig_keys:
        #         if not len(sig_events[sig_key].events):
        #             continue

        #         # ParticleNetMD Txbb SFs
        #         fill_data = utils.get_fill_data(sig_events[sig_key], shape_vars)
        #         for shift in ["down", "up"]:
        #             h.fill(
        #                 Sample=f"{sig_key}_txbb_{shift}",
        #                 **fill_data,
        #                 weight=sig_events[sig_key].get_var(f"{weight_key}_txbb_{shift}"),
        #             )

        templates[rname + jlabel] = h

        ################################
        # Plot templates incl variations
        ################################

        if plot_dir != "" and (not do_jshift or plot_shifts):
            print(f"Plotting templates: {time.time() - start:.2f}")
            if plot_sig_keys is None:
                plot_sig_keys = sig_keys

            if sig_scale_dict is None:
                sig_scale_dict = {skey: 10 for skey in plot_sig_keys}

            title = (
                f"{region.label} Region Pre-Fit Shapes"
                if not do_jshift
                else f"{region.label} Region {jshift} Shapes"
            )

            # don't plot qcd in the pass regions
            # if pass_region:
            #     p_bg_keys = [key for key in bg_keys if key != qcd_key]
            # else:
            p_bg_keys = bg_keys

            for i, shape_var in enumerate(shape_vars):
                plot_params = {
                    "hists": h.project(0, i + 1),
                    "sig_keys": plot_sig_keys,
                    "sig_scale_dict": (
                        {key: sig_scale_dict[key] for key in plot_sig_keys}
                        if region.signal
                        else None
                    ),
                    "channel": channel,
                    "show": show,
                    "year": year,
                    "ylim": pass_ylim if pass_region else fail_ylim,
                    "plot_data": (not (rname == "pass" and blind_pass)) and plot_data,
                    "leg_args": {"fontsize": 22, "ncol": 2},
                }

                plot_name = (
                    f"{plot_dir}/"
                    f"{'jshifts/' if do_jshift else ''}"
                    f"{rname}_region_{shape_var.var}"
                )

                plotting.ratioHistPlot(
                    **plot_params,
                    bg_keys=p_bg_keys,
                    title=title,
                    name=f"{plot_name}{jlabel}.pdf",
                    plot_ratio=plot_data,
                )

                if not do_jshift and plot_shifts:
                    plot_name = f"{plot_dir}/wshifts/" f"{rname}_region_{shape_var.var}"

                    for wshift, wsyst in weight_shifts.items():
                        plotting.ratioHistPlot(
                            **plot_params,
                            bg_keys=p_bg_keys,
                            syst=(wshift, wsyst.samples),
                            title=f"{region.label} Region {wsyst.label} Unc.",
                            name=f"{plot_name}_{wshift}.pdf",
                            plot_ratio=False,
                            reorder_legend=False,
                        )

                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            plotting.ratioHistPlot(
                                **plot_params,
                                bg_keys=p_bg_keys,  # don't plot QCD
                                syst=(wshift, wsyst.samples),
                                variation=shift,
                                title=f"{region.label} Region {wsyst.label} Unc. {skey} Shapes",
                                name=f"{plot_name}_{wshift}_{shift}.pdf",
                                plot_ratio=False,
                            )

                    if region.signal:
                        plotting.ratioHistPlot(
                            **plot_params,
                            bg_keys=p_bg_keys,
                            sig_err="txbb",
                            title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                            name=f"{plot_name}_txbb.pdf",
                        )

    return templates


def save_templates(
    templates: dict[str, Hist],
    template_file: Path,
    blind: bool,
    shape_vars: list[ShapeVar],
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    if blind:
        from copy import deepcopy

        blind_window = shape_vars[0].blind_window

        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}MCBlinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channel",
        required=True,
        choices=list(Samples.CHANNELS.keys()),
        help="channel",
        type=str,
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--bg-data-dirs",
        default=[],
        help="path to skimmed background parquets, if different from other data",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--signal-data-dirs",
        default=[],
        help="path to skimmed signal parquets, if different from other data",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--years",
        required=True,
        nargs="+",
        choices=hh_vars.years + ["all"],
        help="Year(s) to process. For templates, provide exactly one year. For --control-plots, multiple years will be concatenated.",
        type=str,
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Run in test mode (reduced data size)",
    )

    parser.add_argument(
        "--tt-pres",
        action="store_true",
        default=False,
        help="Apply tt preselection",
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    parser.add_argument(
        "--template-dir",
        help="If saving templates, path to file to save them in. If scanning, directory to save in.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--templates-name",
        help="If saving templates, optional name for folder (comes under cuts directory if scanning).",
        default="",
        type=str,
    )

    parser.add_argument(
        "--combined-signals",
        help="Name of the combined signals to use",
        default="separate_signals",
        choices=["sm_signals", "separate_signals"],
        type=str,
    )

    add_bool_arg(parser, "control-plots", "make control plots", default=False)
    add_bool_arg(
        parser,
        "sr",
        "for --control-plots, apply the signal-region pass cut (resolved from --bmin / "
        "--sensitivity-dir) before plotting",
        default=False,
    )

    add_bool_arg(parser, "blinded", "blind the data in the Higgs mass window", default=True)
    add_bool_arg(parser, "templates", "save m_bb templates", default=False)
    add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    add_bool_arg(parser, "do-jshifts", "Do JEC/JMC variations", default=True)
    add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)
    add_bool_arg(
        parser, "override-systs", "Override saved systematics file if it exists", default=False
    )

    parser.add_argument(
        "--sigs",
        help="specify signal samples. By default, will use the samples defined in `hh_vars`.",
        nargs="*",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--bgs",
        help="specify background samples",
        nargs="*",
        default=None,
        type=str,
    )

    add_bool_arg(parser, "read-sig-samples", "read signal samples from directory", default=False)
    add_bool_arg(parser, "data", "include data", default=True)
    add_bool_arg(parser, "filters", "apply filters", default=True)
    add_bool_arg(
        parser,
        "gen-split",
        "legacy behavior: restrict each channel's signal sample to events whose gen-truth tau "
        "decay mode matches the channel (GenTau<channel>). Default (False) instead loads the "
        "full, gen-truth-unsplit signal sample per channel, so channel membership is decided "
        "purely by reco-level SR cuts + the existing cross-channel veto chain, same as "
        "data/background -- signal migrating across channels at reco level is neither lost nor "
        "invisible. Pass --gen-split to recover the old behavior for comparison",
        default=False,
    )
    add_bool_arg(
        parser,
        "a-priori-channels",
        "dev_channel_separation: classify hh/hm/he a priori from lepton content instead of the "
        "CHANNEL_ORDERING veto chain -- hm if a tight muon is within userConfig.LEPTON_CONE_DR "
        "of the ttFatJet, else he if a tight electron is, else hh (exhaustive by construction, "
        "see utils.assign_lepton_channel). Not yet compatible with --do-vbf (cross-signal, i.e. "
        "ggf-vs-vbf, veto handling is unaffected by this flag and untested in combination with "
        "it).",
        default=False,
    )

    parser.add_argument(
        "--control-plot-vars",
        help="Specify control plot variables to plot. By default plots all.",
        default=[],
        nargs="*",
        type=str,
    )

    add_bool_arg(parser, "use_ParT", "Use ParT for sensitivity study", default=False)

    parser.add_argument(
        "--ggf-modelname",
        help="Name of the BDT model to use",
        default="May4_optimized_ggf",
        type=str,
    )
    parser.add_argument(
        "--do-vbf",
        action="store_true",
        default=False,
        help="Run VBF optimization first (with its own model) and veto its selection (Bmin=10) when optimizing the main signal",
    )
    parser.add_argument(
        "--control-region",
        action="store_true",
        default=False,
        help="Build CR (orthogonal annulus) templates for QCD+DY validation instead of the "
        "nominal SR pass/fail. Loose cuts from userConfig.CR_LOOSE_CUTS; see "
        "notes/control_region_plan.md. Templates go to a 'control' subdir; CR is unblinded.",
    )
    parser.add_argument(
        "--vbf-modelname",
        help="Name of the BDT model to use",
        default="May4_optimized_vbf",
        type=str,
    )

    parser.add_argument(
        "--model-dir",
        help="Path to the BDT model directory",
        default=MODEL_DIR,
        type=str,
    )

    parser.add_argument(
        "--sensitivity-dir",
        help=(
            "Directory ``.../plots/SensitivityStudy/<date>/`` (parent of ``tt_pres`` / ``full_presel`` / "
            "``test``). CSV is read from "
            "``<dir>/<presel>/<disc>/<do_vbf|ggf_only>/<combined_signals>/<orthogonal|overlapping>_channels/"
            "<signal>/<channel>/*_opt_results_*.csv``. Presel follows ``--tt-pres`` / ``--test-mode``. "
            "Disc folder is ``--sensitivity-disc-tag`` if set, else ParT / ``--ggf-modelname`` / ``BDT``. "
            "If not set, cuts from Samples.py are used."
        ),
        default=None,
        type=str,
    )

    parser.add_argument(
        "--sensitivity-disc-tag",
        default=None,
        type=str,
        help=(
            "Subfolder under the presel directory for optimized cuts (must match ``SensitivityStudy`` "
            "output, e.g. May4_optimized_ggf). Overrides ``--ggf-modelname`` for the path only when set."
        ),
    )

    parser.add_argument(
        "--tt-glopart-cut",
        type=float,
        default=None,
        help=(
            "Extra fixed cut on ttFatJetParTX<channel>vsQCDTop, applied on top of --bb-disc and "
            "the tt discriminant being optimized (matches SensitivityStudy.py's --tt-glopart-cut, "
            "which appends '_ttglopart<cut>' to the sensitivity-dir disc folder -- set this to the "
            "same value used there so the CSV WPs are found and reproduced correctly). No-op in "
            "--use_ParT mode (the tt discriminant being optimized there already is this same "
            "column, so a separate cut would be redundant); ignored with a printed note if set."
        ),
    )

    add_bool_arg(
        parser,
        "overlapping-channels",
        "Sensitivity study used overlapping (not orthogonal) channel regions; use overlapping_channels in CSV path",
        default=False,
    )

    parser.add_argument(
        "--bmin",
        help="Minimum bkg yield(s) for the TXbb/Txtt cuts. Can be a single value or a list. Need to be present in the csv file",
        default=[10],
        nargs="*",
        type=int,
    )

    parser.add_argument(
        "--bb-disc",
        help="bb discriminator to optimize",
        default="ak8FatJetParTXbbvsQCDTop",
        choices=[
            "ak8FatJetParTXbbvsQCD",
            "ak8FatJetParTXbbvsQCDTop",
            "ak8FatJetPNetXbbvsQCDLegacy",
        ],
        type=str,
    )

    args = parser.parse_args()
    # save_args = deepcopy(args)

    args.model_dir = Path(args.model_dir)

    if args.data_dir:
        args.data_dir = Path(args.data_dir)

    if args.bg_data_dirs:
        args.bg_data_dirs = [Path(bg_dir) for bg_dir in args.bg_data_dirs]
    elif args.data_dir:
        args.bg_data_dirs = [args.data_dir]

    if args.signal_data_dirs:
        args.signal_data_dirs = [Path(sig_dir) for sig_dir in args.signal_data_dirs]
    elif args.data_dir:
        args.signal_data_dirs = [args.data_dir]

    # save args in args.plot_dir and args.template_dir if they exist
    if args.plot_dir:
        year_label = "+".join(args.years)
        base_plot_dir = Path(args.plot_dir)

        # For control plots, add extra top-level structure:
        # <today>/<test|tt-presel|no-presel>/<channel>/<year_label>
        if args.control_plots:
            today = datetime.date.today().strftime("%Y%m%d")
            presel_label = (
                "test" if args.test_mode else ("tt-presel" if args.tt_pres else "no-presel")
            )
            args.plot_dir = base_plot_dir / today / presel_label / args.channel / year_label
        else:
            args.plot_dir = base_plot_dir / args.channel / year_label

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        # with (args.plot_dir / "args.json").open("w") as f:
        #     try:
        #         json.dump(save_args.__dict__, f, indent=4)
        #     except Exception as e:
        #         print(f"Error saving args: {e}")

    if args.template_dir:
        args.template_dir = Path(args.template_dir)
    #

    # (args.template_dir / "cutflows" / args.year).mkdir(parents=True, exist_ok=True)
    # with (args.template_dir / "args.json").open("w") as f:
    #     json.dump(save_args.__dict__, f, indent=4)

    return args


if __name__ == "__main__":
    mpl.use("Agg")
    args = parse_args()
    if args.control_plots:
        run_control_plots(args)
    else:
        main(args)
