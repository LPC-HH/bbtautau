#!/usr/bin/env python3
"""
Extract 50% expected limits from combine AsymptoticLimits output files.
Usage: python extract_limits.py <card_directory> [--output OUTPUT.csv]
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple


def parse_limit(filepath: Path) -> Optional[float]:
    """Extract Expected 50.0% limit from AsymptoticLimits.txt file."""
    if not filepath.exists():
        return None
    pattern = re.compile(r"Expected\s+50\.0%:\s*r\s*<\s*([0-9.]+)")
    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def label_to_signal_channel(filename: str) -> Tuple[str, str]:
    """Convert filename like ggfhhAsymptoticLimits.txt to (signal_region, decay_channel)."""
    stem = filename.replace("AsymptoticLimits.txt", "")
    if stem == "allsigs":
        return "combined", "all"
    if stem.startswith("ggf"):
        return "ggF", stem[3:]  # hh, he, hm
    if stem.startswith("vbf"):
        return "VBF", stem[3:]
    return "unknown", stem


def main():
    parser = argparse.ArgumentParser(description="Extract 50% expected limits from combine outputs")
    parser.add_argument("card_dir", type=str, help="Card directory (e.g. cards/hello/bmin_10)")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file path")
    args = parser.parse_args()

    card_dir = Path(args.card_dir)
    outs_dir = card_dir / "outs"
    if not outs_dir.exists():
        print(f"ERROR: outs directory not found: {outs_dir}")
        return 1

    limit_files = sorted(outs_dir.glob("*AsymptoticLimits.txt"))
    if not limit_files:
        print(f"ERROR: No *AsymptoticLimits.txt files in {outs_dir}")
        return 1

    rows = []
    for f in limit_files:
        limit = parse_limit(f)
        sig_region, decay_channel = label_to_signal_channel(f.name)
        if limit is not None:
            rows.append((sig_region, decay_channel, limit))

    # Sort: combined first, then ggF, then VBF; within each by channel (hh, he, hm)
    channel_order = {"all": 0, "hh": 1, "he": 2, "hm": 3}
    sig_order = {"combined": 0, "ggF": 1, "VBF": 2}
    rows.sort(key=lambda r: (sig_order.get(r[0], 99), channel_order.get(r[1], 99)))

    # Print table
    print("\n50% Expected limits (r @ 95% CL)")
    print("-" * 50)
    print(f"{'Signal region':<14} {'Decay channel':<14} {'r (50%)':>10}")
    print("-" * 50)
    for sig, ch, lim in rows:
        print(f"{sig:<14} {ch:<14} {lim:>10.2f}")
    print("-" * 50)

    # Write CSV
    out_path = Path(args.output) if args.output else card_dir / "limits_50pct.csv"
    with open(out_path, "w") as f:
        f.write("signal_region,decay_channel,expected_50pct_limit\n")
        for sig, ch, lim in rows:
            f.write(f"{sig},{ch},{lim}\n")
    print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
