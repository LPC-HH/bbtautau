from __future__ import annotations

import argparse
import os
import sys
from argparse import BooleanOptionalAction
from pathlib import Path
from string import Template

kubernetes_dir = Path("/home/users/lumori/bbtautau/src/bbtautau/kubernetes")
templ_file = kubernetes_dir / "jobs" / "template.yaml"
templ_compare_file = kubernetes_dir / "jobs" / "template_compare.yaml"
templ_rescaling_file = kubernetes_dir / "jobs" / "template_rescaling.yaml"

PVC = Path("/bbtautauvol")
BDT_DIR = PVC / "bdt"


class objectview:
    """converts a dict into an object"""

    def __init__(self, d):
        self.__dict__ = d


def from_json(args):
    import json

    with Path.open(args.from_json) as f:
        json_args = json.load(f)

    args_dict = vars(args)

    for arg, val in json_args.items():
        args_dict[arg] = val

    args = objectview(args_dict)

    args.job_name = args.from_json.split(".json")[0]
    args.model_name = args.job_name

    return args


def main(args):

    if args.from_json != "":
        args = from_json(args)
    else:
        if args.job_name == "":
            if args.compare_models:
                if args.compare_tag:
                    models_key = args.compare_tag
                elif args.models:
                    models_key = "-".join(args.models)
                elif args.model_folders:
                    models_key = "-".join(Path(f).name for f in args.model_folders)
                else:
                    models_key = "models"
                args.job_name = "cmp_" + args.tag + "_" + models_key
            elif args.study_rescaling:
                args.job_name = "lm_" + args.tag + "_rescaling_" + args.name
            else:
                args.job_name = "lm_" + args.tag + "_" + args.name

    args.job_name = "_".join(args.job_name.split("-")).lower()  # hyphens to underscores, lowercase

    Path.mkdir(kubernetes_dir / f"bdt_trainings/{args.tag}", exist_ok=True)
    file_name = kubernetes_dir / f"bdt_trainings/{args.tag}/{args.job_name}.yml"

    if Path.exists(file_name):
        print(f"Job exists: {file_name}")
        if args.overwrite:
            print("Overwriting")
        else:
            try:
                resp = input(f"{file_name} exists. Overwrite? [y/N]: ").strip().lower()
            except EOFError:
                resp = ""
            if resp in ("y", "yes"):
                print("Overwriting")
            else:
                print("Exiting")
                sys.exit()

    # Choose appropriate template based on mode
    if args.compare_models:
        template_path = templ_compare_file
    elif args.study_rescaling:
        template_path = templ_rescaling_file
    else:
        template_path = templ_file

    with Path.open(template_path) as f:
        lines = Template(f.read())

    # Build extra args string (shared between modes)
    extra_args = args.train_args if args.train_args else ""
    if getattr(args, "samples", None):
        samples_str = " ".join(args.samples)
        extra_args += (" " if extra_args else "") + f"--samples {samples_str}"
    if getattr(args, "tt_preselection", False):
        extra_args += (" " if extra_args else "") + "--tt-preselection"

    if args.compare_models:
        has_explicit = args.models and args.model_dirs
        has_folders = args.model_folders

        if not has_explicit and not has_folders:
            raise ValueError(
                "--compare-models requires --models/--model-dirs or --model-folders (or both)"
            )
        if has_explicit and len(args.models) != len(args.model_dirs):
            raise ValueError("--models and --model-dirs must have the same number of arguments")

        # Build the compare_args string for the template
        compare_parts: list[str] = []
        if has_explicit:
            models_str = " ".join(args.models)
            model_dirs_str = " ".join([str(BDT_DIR / d) for d in args.model_dirs])
            compare_parts.append(f"--models {models_str} --model-dirs {model_dirs_str}")
        if has_folders:
            folders_str = " ".join([str(BDT_DIR / f) for f in args.model_folders])
            compare_parts.append(f"--model-folders {folders_str}")

        compare_tag = args.compare_tag
        if not compare_tag:
            if has_explicit:
                compare_tag = "-".join(args.models)
            else:
                compare_tag = "-".join(Path(f).name for f in args.model_folders)
        output_dir = str(BDT_DIR / args.tag / f"compare_{compare_tag}")

        years_str = " ".join(args.years)
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "output_dir": output_dir,
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
            "years": years_str,
            "compare_args": " ".join(compare_parts),
        }
    elif args.study_rescaling:
        # Rescaling study mode: same template vars as training
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "name": args.name,
            "output_dir": str(BDT_DIR / args.tag / f"rescaling_{args.name}"),
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
        }
    else:
        # Training mode arguments
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),  # change underscores to hyphens
            "name": args.name,
            "output_dir": str(BDT_DIR / args.tag / args.name),
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
        }

    with Path.open(file_name, "w") as f:
        f.write(lines.substitute(args_dict))

    if args.submit:
        os.system(f"kubectl create -f {file_name} -n cms-ml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default="", help="json file to load args from", type=str)
    parser.add_argument("--name", default="", help="", type=str)
    parser.add_argument("--job-name", default="", help="defaults to name", type=str)
    parser.add_argument(
        "--compare-models",
        default=False,
        help="use comparison mode to compare multiple trained models",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--study-rescaling",
        default=False,
        help="use rescaling study mode to sweep scale/balance rules",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="list of model names to compare when --compare-models is set",
        type=str,
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        help="list of model directories to compare when --compare-models is set",
        type=str,
    )
    parser.add_argument(
        "--model-folders",
        nargs="+",
        default=None,
        help="parent directories to scan for trained models (alternative to --models/--model-dirs)",
        type=str,
    )
    parser.add_argument(
        "--compare-tag",
        default=None,
        help="tag for the comparison output directory (defaults to model/folder names)",
        type=str,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["all"],
        help="years to use for comparison/training",
        type=str,
    )
    parser.add_argument("--tag", default="no_presel", help="tag for job / bdt", type=str)
    parser.add_argument(
        "--datapath", default="25Sep23AddVars_v12_private_signal", help="", type=str
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="samples to use in comparison/evaluation",
        type=str,
    )
    parser.add_argument(
        "--train-args",
        default="",
        help="Arguments for training. Use = syntax for args with dashes: --train-args='--flag value'",
        type=str,
    )
    parser.add_argument(
        "--tt-preselection",
        default=False,
        help="apply tt preselection",
        type=bool,
        action=BooleanOptionalAction,
    )

    parser.add_argument(
        "--overwrite",
        default=False,
        help="overwrite old job",
        type=bool,
        action=BooleanOptionalAction,
    )

    parser.add_argument(
        "--submit",
        default=False,
        help="submit",
        type=bool,
        action=BooleanOptionalAction,
    )

    args = parser.parse_args()

    main(args)
