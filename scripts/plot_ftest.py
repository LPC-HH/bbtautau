#!/usr/bin/env python3
"""
F-test and GoF plotting for QCD background polynomial order.

Reads combine GoodnessOfFit output from run_ftest_bbtt.sh and produces:
- GoF test plots (saturated algorithm, chi2 fit)
- F-test plots (order N vs N+1)
- JSON summary of p-values and test statistics (with --save-json or --all-sr-ch)

Usage:
  # Combined (all channels): single set of plots
  python scripts/plot_ftest.py --cards-tag 0215_sm+bsm_ggf+vbf --sig-label ggf --bmin 10

  # Per (signal-region x channel): GoF + F-test plots for each of 6 combinations
  python scripts/plot_ftest.py --cards-tag 0215_sm+bsm_ggf+vbf --all-sr-ch --bmin 10

  # Single channel only
  python scripts/plot_ftest.py --cards-tag 0215_sm+bsm_ggf+vbf --sig-label ggf --channel hh --bmin 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot
from scipy import stats

CHANNEL_LABELS = {"hh": r"$\tau_h\tau_h$", "he": r"$\tau_h e$", "hm": r"$\tau_h \mu$"}


def _valid_toy_files(bindir: Path, pattern: str) -> list[str]:
    """Return ROOT files that contain a 'limit' tree (skip empty/corrupt)."""
    files = sorted(bindir.glob(pattern))
    valid = []
    for fp in files:
        try:
            with uproot.open(fp) as f:
                if "limit" in f:
                    valid.append(str(fp))
                else:
                    warnings.warn(f"Skipping file with no 'limit' tree: {fp}", UserWarning)
        except Exception as e:
            warnings.warn(f"Skipping invalid file {fp}: {e}", UserWarning)
    return valid


def p_value(data_ts: float, toy_ts: list[float]) -> float:
    return float(np.mean(np.array(toy_ts) >= data_ts))


def F_statistic(
    ts_low: np.ndarray,
    ts_high: np.ndarray,
    ord_low: int,
    ord_high: int,
    num_bins: int = 10 * 14,
    dim: int = 2,
) -> np.ndarray:
    numerator = (np.array(ts_low) - np.array(ts_high)) / (ord_high - ord_low)
    denominator = np.array(ts_high) / (num_bins - (ord_high + dim))
    return numerator / denominator


def plot_tests(
    data_ts: float,
    toy_ts: np.ndarray,
    name: str,
    plot_dir: Path,
    sig_label: str,
    title=None,
    bins=15,
    fit=None,
    fdof2=None,
    xlim=None,
    channel: str | None = None,
) -> None:
    category_label = {
        "ggf": "ggF",
        "vbf": "VBF",
        "allsigs": "ggF+VBF Combined",
    }
    legend_title = category_label.get(sig_label, sig_label)
    if channel is not None:
        legend_title = f"{legend_title} {CHANNEL_LABELS.get(channel, channel)}"
    plot_max = max(np.max(toy_ts), data_ts)
    plot_min = 0
    pval = p_value(data_ts, toy_ts)

    plt.figure(figsize=(12, 8))
    h = plt.hist(
        toy_ts,
        np.linspace(plot_min, plot_max if xlim is None else xlim, bins + 1),
        color="#8C8C8C",
        histtype="step",
        label=f"{len(toy_ts)} Toys",
    )
    plt.axvline(
        data_ts,
        color="#FF502E",
        linestyle=":",
        label=rf"Data ($p$-value = {pval:.2f})",
    )

    if fit is not None:
        x = np.linspace(plot_min + 0.01, plot_max, 100)

        if fit == "chi2":
            res = stats.fit(stats.chi2, toy_ts, [(0, 200)])
            pdf = stats.chi2.pdf(x, res.params.df)
            label = rf"$\chi^2_{{DoF = {res.params.df:.2f}}}$ Fit"
        elif fit == "f":
            assert fdof2 is not None
            pdf = stats.f.pdf(x, 1, fdof2)
            label = rf"$F$-dist$_{{DoF = (1, {fdof2})}}$"
        else:
            raise ValueError("Invalid fit")

        plt.plot(
            x,
            pdf * (np.max(h[0]) / np.max(pdf)),
            color="#1f78b4",
            linestyle="--",
            label=label,
        )

    hep.cms.label("Work in Progress", data=True, lumi=61, year=None)
    plt.legend(title=legend_title)
    plt.title(title)
    plt.ylabel("Number of Toys")
    plt.xlabel("Test Statistic")

    out_path = plot_dir / f"{name}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(plot_dir / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="F-test and GoF plotting")
    parser.add_argument(
        "--cards-tag",
        default="0214_sm+bsm_ggf+vbf",
        help="Cards tag (e.g. 0214_sm+bsm_ggf+vbf or 0215_sm+bsm_ggf+vbf)",
    )
    parser.add_argument(
        "--sig-label",
        default="ggf",
        choices=["ggf", "vbf", "allsigs"],
        help="Signal region label for plot",
    )
    parser.add_argument(
        "--bmin",
        type=int,
        default=10,
        help="Bmin value (analysis subdir)",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=None,
        help="Override cards base dir (default: src/bbtautau/combine/cards/f_tests)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Output plot directory",
    )
    parser.add_argument(
        "--test-orders",
        type=int,
        nargs="+",
        default=[0],
        help="Orders to test (default: 0)",
    )
    parser.add_argument(
        "--channel",
        choices=["hh", "he", "hm"],
        default=None,
        help="Plot single channel (requires per-channel f-test output)",
    )
    parser.add_argument(
        "--all-sr-ch",
        action="store_true",
        help="Plot GoF + F-test for each (sig-region x channel): ggf_hh, ggf_he, ggf_hm, vbf_hh, vbf_he, vbf_hm. Requires per-channel f-test output from run_ftest_bbtt.sh --all-sr-ch.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save p-values and test statistics to JSON file",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.cards_dir is not None:
        local_cards_dir = Path(args.cards_dir) / args.cards_tag
    else:
        local_cards_dir = (
            repo_root / "src/bbtautau/combine/cards/f_tests" / args.cards_tag
        )

    analysis_name = f"bmin_{args.bmin}"

    # Build list of (sig_label, channel) to process. channel=None means combined.
    if args.all_sr_ch:
        combos = [
            ("ggf", "hh"), ("ggf", "he"), ("ggf", "hm"),
            ("vbf", "hh"), ("vbf", "he"), ("vbf", "hm"),
        ]
        save_json = True
    elif args.channel is not None:
        combos = [(args.sig_label, args.channel)]
        save_json = args.save_json
    else:
        combos = [(args.sig_label, None)]  # combined
        save_json = args.save_json

    if not local_cards_dir.exists():
        raise FileNotFoundError(f"Cards directory not found: {local_cards_dir}")

    plt.style.use(hep.style.CMS)
    hep.style.use("CMS")
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    plt.rcParams.update({"font.size": 20})

    def bin_dir(nTF: int, sig_label: str, channel: str | None) -> Path:
        base = local_cards_dir / f"nTF_{nTF}"
        if channel is not None:
            return base / f"{analysis_name}_{sig_label}_{channel}"
        return base / analysis_name

    results_summary: dict = {}

    for sig_label, channel in combos:
        combo_str = f"{sig_label}_{channel}" if channel else sig_label
        if args.plot_dir is not None:
            plot_dir = Path(args.plot_dir) / combo_str
        else:
            plot_dir = repo_root / f"plots/FTests/{args.cards_tag}_{combo_str}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {combo_str} -> {plot_dir}")

        test_statistics = {}
        for o1 in args.test_orders:
            tdict = {"toys": {}, "data": {}, "ftoys": {}, "fdata": {}}
            tlabel = f"{o1}"

            for nTF in [o1, o1 + 1]:
                tflabel = f"{nTF}"
                bindir = bin_dir(nTF, sig_label, channel)

                toys_pattern = f"higgsCombineToys{tlabel}.GoodnessOfFit.mH125.*.root"
                data_file = bindir / "higgsCombineData.GoodnessOfFit.mH125.root"

                if not bindir.exists():
                    raise FileNotFoundError(f"Bin dir not found: {bindir}")

                valid_toys = _valid_toy_files(bindir, toys_pattern)
                if not valid_toys:
                    raise FileNotFoundError(
                        f"No valid toy files found for {bindir}/{toys_pattern}. "
                        "Run run_ftest_bbtt.sh with -t (goftoys) and -f (ffits) to generate toys."
                    )
                file = uproot.concatenate(valid_toys)
                tdict["toys"][tflabel] = np.array(file["limit"])

                file = uproot.concatenate(str(data_file))
                tdict["data"][tflabel] = float(file["limit"][0])

                if nTF != o1:
                    tdict["ftoys"][tflabel] = F_statistic(
                        tdict["toys"][tlabel], tdict["toys"][tflabel], o1, nTF
                    )
                    tdict["fdata"][tflabel] = float(
                        F_statistic(
                            np.array([tdict["data"][tlabel]]),
                            np.array([tdict["data"][tflabel]]),
                            o1,
                            nTF,
                        )[0]
                    )

            test_statistics[tlabel] = tdict

        for o1 in args.test_orders:
            tlabel = f"{o1}"
            ord1 = o1 + 1
            tflabel = f"{ord1}"

            data_ts = test_statistics[tlabel]["data"][tlabel]
            toy_ts = test_statistics[tlabel]["toys"][tlabel]
            gof_pval = p_value(data_ts, toy_ts)
            plot_tests(
                data_ts,
                toy_ts,
                f"gof{tlabel}",
                plot_dir,
                sig_label,
                fit="chi2",
                title=f"GoF test, order {o1}",
                bins=20,
                channel=channel,
            )

            data_ts = test_statistics[tlabel]["fdata"][tflabel]
            toy_ts = test_statistics[tlabel]["ftoys"][tflabel]
            f_pval = p_value(data_ts, toy_ts)
            plot_tests(
                data_ts,
                toy_ts,
                f"f{tlabel}_{tflabel}",
                plot_dir,
                sig_label,
                title=f"F-test, order {o1} vs. {ord1}",
                xlim=100,
                channel=channel,
            )

            if save_json:
                if combo_str not in results_summary:
                    results_summary[combo_str] = {
                        "sig_label": sig_label,
                        "channel": channel,
                        "cards_tag": args.cards_tag,
                        "bmin": args.bmin,
                    }
                results_summary[combo_str][f"order_{o1}"] = {
                    "gof_ts_data": float(test_statistics[tlabel]["data"][tlabel]),
                    "gof_pvalue": gof_pval,
                    "f_ts_data": float(data_ts),
                    "f_pvalue": f_pval,
                }

    if save_json and results_summary:
        if len(combos) > 1:
            json_dir = repo_root / f"plots/FTests/{args.cards_tag}_all_sr_ch"
        else:
            json_dir = plot_dir  # same dir as plots for single combo
        json_path = json_dir / "ftest_results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Saved results to {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()
