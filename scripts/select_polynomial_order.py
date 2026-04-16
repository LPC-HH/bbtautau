#!/usr/bin/env python3
"""
Select polynomial order per (signal region × decay channel) using F-test criteria.

Criteria for order n:
  - GoF_n p-value > 5%
  - F-test n vs n+1 p-value > 5% (order n+1 does not significantly improve fit)

Selects the smallest order satisfying both. For order 3 (max), only GoF is checked.

Requires plot_ftest.py to have been run with --all-sr-ch --save-json --test-orders 0 1 2 3
to produce ftest_results.json. Alternatively, pass --json to use an existing JSON file.

Usage:
  # Use ftest_results.json from plot_ftest (default path)
  python scripts/select_polynomial_order.py --cards-tag 0215_sm+bsm_ggf+vbf --bmin 10

  # Use custom JSON path
  python scripts/select_polynomial_order.py --json plots/FTests/0215_sm+bsm_ggf+vbf_all_sr_ch/ftest_results.json

  # Output CreateDatacard --nTF-per-channel format
  python scripts/select_polynomial_order.py --cards-tag 0215_sm+bsm_ggf+vbf --bmin 10 --output-ntf-json ntf_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

P_VALUE_THRESHOLD = 0.05
MAX_ORDER = 3

# Map sig_label to CreateDatacard sr key
SR_KEY_MAP = {"ggf": "ggfbbtt", "vbf": "vbfbbtt"}


def select_order(gof_pvalues: dict[int, float], f_pvalues: dict[tuple[int, int], float]) -> int:
    """
    Find smallest order n where GoF_n > 5% and F-test n vs n+1 > 5%.

    gof_pvalues: {0: p0, 1: p1, 2: p2, 3: p3}
    f_pvalues: {(0,1): p, (1,2): p, (2,3): p}  # (n, n+1) -> p-value
    """
    for n in range(MAX_ORDER + 1):
        gof_p = gof_pvalues.get(n)
        if gof_p is None or gof_p <= P_VALUE_THRESHOLD:
            continue

        if n < MAX_ORDER:
            f_p = f_pvalues.get((n, n + 1))
            if f_p is None or f_p <= P_VALUE_THRESHOLD:
                continue

        return n
    return MAX_ORDER


def load_results(json_path: Path) -> dict:
    """Load ftest_results.json."""
    with open(json_path) as f:
        return json.load(f)


def extract_pvalues(results: dict) -> dict[str, dict]:
    """
    Extract gof_pvalue and f_pvalue per combo and order from ftest_results.json.

    Returns: {combo_str: {gof: {0: p, 1: p, ...}, f: {(0,1): p, (1,2): p, ...}}}
    """
    out = {}
    for combo_str, combo_data in results.items():
        gof = {}
        f = {}
        for k, v in combo_data.items():
            if k.startswith("order_"):
                n = int(k.split("_")[1])
                gof[n] = v.get("gof_pvalue")
                if "f_pvalue" in v:
                    f[(n, n + 1)] = v["f_pvalue"]
        out[combo_str] = {"gof": gof, "f": f}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select polynomial order per (sr × channel) from F-test results"
    )
    parser.add_argument(
        "--cards-tag",
        default="0215_sm+bsm_ggf+vbf",
        help="Cards tag (used to locate ftest_results.json if --json not set)",
    )
    parser.add_argument(
        "--bmin",
        type=int,
        default=10,
        help="Bmin value",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to ftest_results.json. Default: plots/FTests/{cards_tag}_all_sr_ch/ftest_results.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output-ntf-json",
        type=Path,
        default=None,
        help="Write CreateDatacard --nTF-per-channel JSON to this path",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        default=0.05,
        help="Minimum p-value to pass (default: 0.05)",
    )
    args = parser.parse_args()

    global P_VALUE_THRESHOLD
    P_VALUE_THRESHOLD = args.p_value_threshold

    script_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root or script_dir.parent

    if args.json is not None:
        json_path = Path(args.json)
    else:
        json_path = repo_root / f"plots/FTests/{args.cards_tag}_all_sr_ch/ftest_results.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Results JSON not found: {json_path}\n"
            f"Run: python scripts/plot_ftest.py --cards-tag {args.cards_tag} "
            f"--all-sr-ch --bmin {args.bmin} --test-orders 0 1 2 3"
        )

    results = load_results(json_path)
    pvalues = extract_pvalues(results)

    # Select order per combo
    selections: dict[str, int] = {}
    details: dict[str, dict] = {}

    for combo_str, pv in pvalues.items():
        order = select_order(pv["gof"], pv["f"])
        selections[combo_str] = order
        details[combo_str] = {
            "selected_order": order,
            "gof_pvalues": pv["gof"],
            "f_pvalues": {f"{a}_{b}": v for (a, b), v in pv["f"].items()},
        }

    # Print table
    print("\nPolynomial order selection (GoF_n > 5% and F-test n vs n+1 > 5%)")
    print("=" * 80)
    print(f"{'Combo':<12} {'Order':<8} {'GoF p-values':<40} {'F-test p-values'}")
    print("-" * 80)
    for combo_str in sorted(selections.keys()):
        order = selections[combo_str]
        pv = pvalues[combo_str]
        gof_parts = []
        for n in range(MAX_ORDER + 1):
            v = pv["gof"].get(n)
            gof_parts.append(f"n{n}:{v:.2f}" if v is not None else f"n{n}:-")
        gof_str = " ".join(gof_parts)
        f_parts = []
        for (a, b) in [(0, 1), (1, 2), (2, 3)]:
            v = pv["f"].get((a, b))
            f_parts.append(f"{a}v{b}:{v:.2f}" if v is not None else f"{a}v{b}:-")
        f_str = " ".join(f_parts)
        print(f"{combo_str:<12} {order:<8} {gof_str:<40} {f_str}")
    print("=" * 80)

    # Build CreateDatacard --nTF-per-channel format
    ntf_config: dict[str, dict[str, int]] = {"ggfbbtt": {}, "vbfbbtt": {}}
    for combo_str, order in selections.items():
        sig_label, channel = combo_str.split("_")
        sr_key = SR_KEY_MAP.get(sig_label, sig_label + "bbtt")
        ntf_config[sr_key][channel] = order

    print("\nCreateDatacard --nTF-per-channel config:")
    print(json.dumps(ntf_config, indent=2))

    if args.output_ntf_json is not None:
        out_path = Path(args.output_ntf_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(ntf_config, f, indent=2)
        print(f"\nSaved to {out_path}")

    # Also save full results
    summary_path = json_path.parent / "polynomial_order_selection.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "selections": selections,
                "details": details,
                "ntf_config": ntf_config,
                "p_value_threshold": P_VALUE_THRESHOLD,
            },
            f,
            indent=2,
        )
    print(f"Full results saved to {summary_path}")


if __name__ == "__main__":
    main()
