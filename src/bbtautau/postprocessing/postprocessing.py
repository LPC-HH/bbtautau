"""
Postprocessing functions for bbtautau.

Authors: Raghav Kansal, Ludovico Mori
"""

from __future__ import annotations

import argparse
import copy
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
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES, SIGNALS
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
        ShapeVar(var=f"{jet}FatJetMass", label=rf"$m^{{{jlabel}}}$ [GeV]", bins=[20, 250, 1250])
        for jet, jlabel in [("bb", "bb"), ("tt", r"\tau\tau")]
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
    #  nElectrons
    + [ShapeVar(var="nElectrons", label=r"Number of Electrons", bins=[3, 0, 3])]
    #  nMuons
    + [ShapeVar(var="nMuons", label=r"Number of Muons", bins=[3, 0, 3])]
    #  nTaus
    + [ShapeVar(var="nTaus", label=r"Number of Taus", bins=[3, 0, 3])]
    #  nBoostedTaus
    + [ShapeVar(var="nBoostedTaus", label=r"Number of Boosted Taus", bins=[3, 0, 3])]
)

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
    # Convert single bmin value to list for backward compatibility
    if isinstance(args.bmin, int):
        args.bmin = [args.bmin]

    print(f"Processing bmin values: {args.bmin}")

    # These are the regions to use: either only ggf or both ggf and vbf
    signal_regions = copy.deepcopy(SIGNAL_ORDERING) if args.do_vbf else ["ggfbbtt"]

    if args.sigs is None:
        args.sigs = SIGNALS

    if args.bgs is None:
        args.bgs = {bkey: b for bkey, b in SAMPLES.items() if b.get_type() == "bg"}

    CHANNEL = CHANNELS[args.channel]

    models = None
    if not args.use_ParT:
        models = [args.ggf_modelname] + ([args.vbf_modelname] if args.do_vbf else [])

    events_dict, cutflow = load_data_channel(
        years=[args.year],  # Wrap single year in list
        signals=args.sigs,
        channel=CHANNEL,
        test_mode=args.test_mode,
        tt_pres=args.tt_pres,
        models=models,
        cutflow=True,
        load_bgs=True,
    )

    # Keep dictionary structure consistent with legacy code, working out templates one year at a time
    events_dict = events_dict[args.year]
    args.sigs = {s + CHANNEL.key: SAMPLES[s + CHANNEL.key] for s in args.sigs}
    systematics: dict[str, dict] = {}
    systematics_path: Path | None = None
    if args.template_dir:
        systematics_path = args.template_dir / f"{args.year}_systematics.pkl"

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

    systematics.setdefault(args.year, {})

    # Now process each bmin value
    for bmin in args.bmin:
        print(f"\n{'='*60}")
        print(f"Processing bmin = {bmin}")
        print(f"{'='*60}")

        print(f"\nGenerating templates for bmin={bmin}, signal regions={signal_regions}")

        for signal_key in signal_regions:

            # Create bmin-specific directories
            template_dir_bmin = (
                args.template_dir
                / f"bmin_{bmin}"
                / (CHANNEL.key if args.template_dir else "")
                / signal_key
            )
            plot_dir_bmin = (
                args.plot_dir / f"bmin_{bmin}" / (CHANNEL.key if args.plot_dir else "") / signal_key
            )

            if template_dir_bmin:
                (template_dir_bmin / "cutflows" / args.year).mkdir(parents=True, exist_ok=True)
            if plot_dir_bmin:
                plot_dir_bmin.mkdir(parents=True, exist_ok=True)

            templates = get_templates(
                events_dict,  # Same data for all bmin values
                args.year,
                args.sigs,
                args.bgs,
                CHANNEL,  # Updated channel with new cuts
                signal_key,
                signal_regions,
                shape_vars,
                {},  # TODO: systematics
                sig_scale_dict={
                    f"ggfbbtt{CHANNEL.key}": 300,
                    f"vbfbbtt{CHANNEL.key}": 40,
                    f"vbfbbtt-k2v0{CHANNEL.key}": 40,
                },
                # prev_cutflow=cutflow, Tried to add this but seems to not work. May want to fix later
                template_dir=template_dir_bmin,
                plot_dir=plot_dir_bmin,
                show=False,
                selection_region_kwargs={
                    "sensitivity_dir": args.sensitivity_dir,
                    "bmin": bmin,  # Use loop variable, not args.bmin
                    "use_ParT": args.use_ParT,
                    "do_vbf": args.do_vbf,
                    "bb_disc": args.bb_disc,
                },
            )

            if args.template_dir:
                print(f"Saving templates for bmin={bmin}")
                template_file = template_dir_bmin / f"{args.year}_templates.pkl"
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
    events_dict: dict[str, pd.DataFrame],
    channel: Channel,
    sigs: dict[str, Sample],
    bgs: dict[str, Sample],
    control_plot_vars: list[ShapeVar],
    plot_dir: Path,
    year: str,
    bbtt_masks: dict[str, pd.DataFrame] = None,
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
        sig_scale_dict = {sig_key: 2e5 for sig_key in sigs}

    print(control_plot_vars)
    print(selection)
    print(list(events_dict.keys()))

    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            hists[shape_var.var] = putils.singleVarHist(
                events_dict,
                shape_var,
                channel,
                bbtt_masks=bbtt_masks,
                weight_key=weight_key,
                selection=selection,
            )

    print(hists)

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
                ylim=pylim if not log else 1e15,
                plot_ratio=plot_ratio,
                cmslabel="Work in progress",
                leg_args={"fontsize": 18},
            )
            merger_control_plots.append(name)

        if combine_pdf:
            merger_control_plots.write(f"{plot_dir}/{cutstr}{year}{logstr}_ControlPlots.pdf")

        merger_control_plots.close()

    return hists


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

    vetoes = []
    found = False
    # veto all channels/signals earlier in the ordering than the current one
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
    selection_regions = Regions.get_selection_regions(signal, channel, **selection_region_kwargs)

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
        "--year",
        required=True,
        choices=hh_vars.years,
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

    add_bool_arg(parser, "control-plots", "make control plots", default=False)

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
        default="19oct25_ak4away_ggfbbtt",
        type=str,
    )
    parser.add_argument(
        "--do-vbf",
        action="store_true",
        default=False,
        help="Run VBF optimization first (with its own model) and veto its selection (Bmin=10) when optimizing the main signal",
    )
    parser.add_argument(
        "--vbf-modelname",
        help="Name of the BDT model to use",
        default="19oct25_ak4away_vbfbbtt",
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
        help="Path to the sensitivity study's output directory that has a csv file under {dir}/full_presel/grid/{do_vbf}/sm_signals/orthogonal_channels/{signal}/{channel}. The TXbb/Txtt cuts will be extracted/ If not provided, the script will use the ones in Samples.py",
        default=None,
        type=str,
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
        default="ak8FatJetParTXbbvsQCD",
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

    if args.control_plots:
        raise NotImplementedError("Control plots not implemented")

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

    # save args in args.plot_dir and args.template_dir if they exit
    if args.plot_dir:
        args.plot_dir = Path(args.plot_dir) / args.channel / args.year
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
    main(args)
