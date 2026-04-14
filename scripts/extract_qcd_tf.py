#!/usr/bin/env python3
"""
Extract QCD transfer factor values from FitShapes.root.
Usage: python scripts/extract_qcd_tf.py <FitShapes.root> [--region ggfbbtthmpass]
"""
import argparse
import sys

import numpy as np

try:
    import uproot
except ImportError:
    print("ERROR: uproot required. Install with: pip install uproot")
    sys.exit(1)

QCD_KEY = "CMS_bbtautau_boosted_qcd_datadriven"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_shapes", help="Path to FitShapes.root")
    parser.add_argument("--region", default="ggfbbtthmpass", help="Pass region (e.g. ggfbbtthmpass)")
    args = parser.parse_args()

    f = uproot.open(args.fit_shapes)
    pass_region = args.region
    fail_region = pass_region.replace("pass", "fail")

    pass_dir = f"{pass_region}_postfit"
    fail_dir = f"{fail_region}_postfit"

    try:
        pass_qcd = f[f"{pass_dir}/{QCD_KEY}"]
        fail_qcd = f[f"{fail_dir}/{QCD_KEY}"]
    except KeyError as e:
        print(f"ERROR: {e}")
        print("Available keys:", [k for k in f.keys() if "postfit" in k][:20])
        sys.exit(1)

    pass_vals = np.array(pass_qcd.values())
    fail_vals = np.array(fail_qcd.values())
    axis = pass_qcd.axes[0]
    edges = np.array(axis.edges()) if callable(getattr(axis, "edges", None)) else np.array(axis.edges)
    centers = (edges[:-1] + edges[1:]) / 2

    tf = np.where(fail_vals > 0, pass_vals / fail_vals, np.nan)

    print(f"QCD Transfer Factor: {pass_region} / {fail_region}")
    print("=" * 50)
    print(f"{'Bin center':>12} | {'TF':>12} | Pass QCD  | Fail QCD")
    print("-" * 50)
    for i in range(len(centers)):
        print(f"{centers[i]:12.4f} | {tf[i]:12.6e} | {pass_vals[i]:9.4e} | {fail_vals[i]:9.4e}")
    print("-" * 50)
    valid = ~np.isnan(tf)
    if np.any(valid):
        print(f"Min: {np.nanmin(tf):.6e}  Max: {np.nanmax(tf):.6e}  Mean: {np.nanmean(tf):.6e}  Std: {np.nanstd(tf):.6e}")
        cv = np.nanstd(tf) / np.nanmean(tf) * 100 if np.nanmean(tf) != 0 else 0
        print(f"Coefficient of variation: {cv:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
