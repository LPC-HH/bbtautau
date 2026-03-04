"""
Generate Kubernetes jobs for hyperparameter exploration configurations.

Reads the summary JSON produced by generate_hp_configs.py and creates
Kubernetes job YAMLs for each model, optionally submitting them.

Since generate_hp_configs.py now writes configs directly into
postprocessing/bdt_configs/<group>/, no file copying is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "jobs"))


def load_summary(summary_file: Path) -> dict:
    """Load the hyperparameter exploration summary."""
    with summary_file.open() as f:
        return json.load(f)


def _resolve_summary(raw_path: str) -> Path:
    """Find the summary JSON: accept an explicit path or a group name."""
    p = Path(raw_path)
    if p.exists():
        return p
    # Treat as group name → look under bdt_configs/<group>/
    bdt_configs_dir = Path(__file__).parent.parent / "postprocessing" / "bdt_configs"
    candidate = bdt_configs_dir / raw_path / "hp_exploration_summary.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Summary not found: tried '{raw_path}' and '{candidate}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes jobs for hyperparameter exploration configs"
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to hp_exploration_summary.json, or the group name "
        "(e.g. 'key_pars_k2v0') to find it under bdt_configs/<group>/.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="hp_explore",
        help="Tag for organizing jobs (default: hp_explore)",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="25Sep23AddVars_v12_private_signal",
        help="Data path within PVC",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs to Kubernetes after generation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing job files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    summary_file = _resolve_summary(args.summary)
    summary = load_summary(summary_file)
    modelnames = summary["modelnames"]

    print(f"Group: {summary.get('group', '?')}")
    print(f"Found {len(modelnames)} models to process")
    print(f"Tag: {args.tag}\n")

    generated_jobs = []
    for modelname in modelnames:
        if args.dry_run:
            print(
                f"[dry-run] make_from_template.py --name {modelname} "
                f"--tag {args.tag} --datapath {args.datapath}"
            )
            continue

        from make_from_template import main as generate_job_main

        class _Args:
            def __init__(self, _mn=modelname):
                self.from_json = ""
                self.name = _mn
                self.job_name = ""
                self.compare_models = False
                self.models = None
                self.model_dirs = None
                self.years = ["all"]
                self.tag = args.tag
                self.datapath = args.datapath
                self.samples = None
                self.train_args = ""
                self.tt_preselection = False
                self.overwrite = args.overwrite
                self.submit = args.submit

        try:
            generate_job_main(_Args())
            generated_jobs.append(modelname)
            print(f"  + {modelname}")
        except Exception as e:
            print(f"  ! {modelname}: {e}")

    print(f"\n{'='*60}")
    print(f"  Models processed: {len(modelnames)}")
    print(f"  Jobs generated:   {len(generated_jobs)}")
    if args.submit and not args.dry_run:
        print("  Jobs submitted to Kubernetes")
    print(f"  Job files: kubernetes/bdt_trainings/{args.tag}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
