"""
Generate Kubernetes jobs for hyperparameter exploration configurations.

Reads the summary JSON produced by generate_hp_configs.py and creates
Kubernetes job YAMLs for each model, optionally submitting them.

Each generated job embeds its corresponding config file and passes it
to `bdt.py --config-file`, so the config does not need to be committed.
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


def _build_config_map(summary: dict, summary_file: Path) -> dict[str, Path]:
    """Map model names to generated config files from the summary."""
    modelnames = summary["modelnames"]
    generated_files = summary.get("generated_files", [])
    if len(generated_files) != len(modelnames):
        raise ValueError(
            "Summary is missing generated_files entries for some models. "
            "Re-run generate_hp_configs.py before generating jobs."
        )

    config_map: dict[str, Path] = {}
    for modelname, raw_path in zip(modelnames, generated_files):
        path = Path(raw_path)
        if not path.is_absolute():
            path = (summary_file.parent / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config for model '{modelname}' not found: {path}")
        config_map[modelname] = path
    return config_map


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
        default="26Mar5All_v12_private_signal",
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
        "--tt-preselection",
        action="store_true",
        help="Apply tt preselection in training jobs",
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
    config_files = _build_config_map(summary, summary_file)

    print(f"Group: {summary.get('group', '?')}")
    print(f"Found {len(modelnames)} models to process")
    print(f"Tag: {args.tag}\n")

    generated_jobs = []

    from make_from_template import build_parser
    from make_from_template import main as generate_job_main

    mft_parser = build_parser()

    passthrough_flags = ["--tag", args.tag, "--datapath", args.datapath]
    if args.overwrite:
        passthrough_flags.append("--overwrite")
    if args.submit:
        passthrough_flags.append("--submit")
    if args.tt_preselection:
        passthrough_flags.append("--tt-preselection")

    for modelname in modelnames:
        job_argv = [
            "--modelname",
            modelname,
            "--config-file",
            str(config_files[modelname]),
            *passthrough_flags,
        ]

        if args.dry_run:
            print(f"[dry-run] make_from_template.py {' '.join(job_argv)}")
            continue

        try:
            job_args = mft_parser.parse_args(job_argv)
            generate_job_main(job_args)
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
