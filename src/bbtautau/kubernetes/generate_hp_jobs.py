"""
Generate Kubernetes jobs for hyperparameter exploration configurations.

This script reads generated config files and creates Kubernetes job YAMLs
for each configuration, optionally submitting them to the cluster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the job generation function
sys.path.insert(0, str(Path(__file__).parent / "jobs"))


def load_summary(summary_file: Path) -> dict:
    """Load the hyperparameter exploration summary."""
    with summary_file.open() as f:
        return json.load(f)


def extract_modelname_from_config(config_file: Path) -> str:
    """Extract modelname from a config file."""
    with config_file.open() as f:
        content = f.read()
        # Find CONFIG = {...} and extract modelname
        import re

        match = re.search(r'"modelname":\s*"([^"]+)"', content)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract modelname from {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes jobs for hyperparameter exploration configs"
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="bdt_configs/hp_exploration_summary.json",
        help="Path to hyperparameter exploration summary JSON file",
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

    # Load summary
    summary_file = Path(args.summary_file)
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    summary = load_summary(summary_file)
    config_files = [Path(f) for f in summary["generated_files"]]

    print(f"Found {len(config_files)} configurations to process")
    print(f"Tag: {args.tag}\n")

    # Copy config files to bdt_configs directory if needed
    bdt_configs_dir = Path(__file__).parent.parent.parent / "postprocessing" / "bdt_configs"
    bdt_configs_dir.mkdir(parents=True, exist_ok=True)

    generated_jobs = []
    for config_file in config_files:
        if not config_file.exists():
            print(f"Warning: Config file not found: {config_file}, skipping")
            continue

        # Extract modelname
        try:
            modelname = extract_modelname_from_config(config_file)
        except ValueError as e:
            print(f"Error: {e}, skipping")
            continue

        # Copy config to bdt_configs if not already there
        target_config = bdt_configs_dir / config_file.name
        if not target_config.exists() or args.overwrite:
            import shutil

            shutil.copy2(config_file, target_config)
            print(f"Copied config: {config_file.name} -> {target_config}")

        # Generate job using make_from_template
        print(f"\nGenerating job for model: {modelname}")

        # Prepare arguments for make_from_template
        job_args = [
            "--name",
            modelname,
            "--tag",
            args.tag,
            "--datapath",
            args.datapath,
        ]

        if args.overwrite:
            job_args.append("--overwrite")

        if args.submit:
            job_args.append("--submit")

        if args.dry_run:
            print(
                f"Would run: python jobs/make_from_template.py --name {modelname} --tag {args.tag} --datapath {args.datapath}"
            )
        else:
            # Import and call the main function
            import sys

            jobs_dir = Path(__file__).parent / "jobs"
            sys.path.insert(0, str(jobs_dir))
            from make_from_template import main as generate_job_main

            # Create a mock args object
            class MockArgs:
                def __init__(self, _modelname=modelname):
                    self.from_json = ""
                    self.name = _modelname
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
                generate_job_main(MockArgs())
                generated_jobs.append(modelname)
                print(f"  ✓ Generated job for {modelname}")
            except Exception as e:
                print(f"  ✗ Error generating job for {modelname}: {e}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Configurations processed: {len(config_files)}")
    print(f"  Jobs generated: {len(generated_jobs)}")
    if args.submit and not args.dry_run:
        print("  Jobs submitted to Kubernetes")
    print(f"\nJob files location: kubernetes/bdt_trainings/{args.tag}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
