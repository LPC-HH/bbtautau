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
templ_compare_light_file = kubernetes_dir / "jobs" / "template_compare_light.yaml"

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

    # Presel is independent of --tag, derived from --tt-preselection
    presel = "tt_presel" if getattr(args, "tt_preselection", False) else "no_presel"

    if args.from_json != "":
        args = from_json(args)
        job_type = getattr(args, "job_type", "training")
        default_tag = getattr(args, "model_name", getattr(args, "modelname", "default"))
        run_tag = getattr(args, "tag", None) or default_tag
    else:
        # Job type and default run_tag for path grouping
        if args.compare_models or args.compare_light:
            job_type = "comparisons"
            default_tag = (
                args.compare_tag or "-".join(Path(i).name for i in args.inputs)
                if args.inputs
                else "models"
            )
        elif args.study_rescaling:
            job_type = "rescaling"
            default_tag = args.modelname
        else:
            job_type = "training"
            default_tag = args.modelname

        run_tag = args.tag if args.tag else default_tag

        if args.job_name == "":
            if args.compare_models or args.compare_light:
                prefix = "cmpl_" if args.compare_light else "cmp_"
                models_key = default_tag.replace("-", "_")
                args.job_name = prefix + models_key
            elif args.study_rescaling:
                args.job_name = "rescaling_" + args.modelname.replace("-", "_")
            else:
                args.job_name = args.modelname.replace("-", "_")

    args.job_name = "_".join(args.job_name.split("-")).lower()
    run_tag = "_".join(str(run_tag).split("-")).lower()

    # Path: bdt_trainings/{job_type}/{presel}/{run_tag}/{job_name}.yml
    out_dir = kubernetes_dir / "bdt_trainings" / job_type / presel / run_tag
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    file_name = out_dir / f"{args.job_name}.yml"

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
    if args.compare_light:
        template_path = templ_compare_light_file
    elif args.compare_models:
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

    if args.compare_light:
        if not args.inputs:
            raise ValueError("--compare-light requires --inputs")

        inputs_str = " ".join(str(BDT_DIR / i) for i in args.inputs)
        compare_args_str = f"--inputs {inputs_str}"
        if args.disc_filter is not None:
            compare_args_str += f" --disc-filter {args.disc_filter}"

        compare_tag = args.compare_tag
        if not compare_tag:
            compare_tag = "-".join(Path(i).name for i in args.inputs)
        output_dir = str(BDT_DIR / job_type / presel / run_tag / f"compare_light_{compare_tag}")

        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "output_dir": output_dir,
            "args": extra_args,
            "compare_args": compare_args_str,
        }
    elif args.compare_models:
        if not args.inputs:
            raise ValueError("--compare-models requires --inputs")

        # Build the compare_args string for the template
        inputs_str = " ".join(str(BDT_DIR / i) for i in args.inputs)
        compare_args_str = f"--inputs {inputs_str}"

        compare_tag = args.compare_tag
        if not compare_tag:
            compare_tag = "-".join(Path(i).name for i in args.inputs)
        output_dir = str(BDT_DIR / job_type / presel / run_tag / f"compare_{compare_tag}")

        years_str = " ".join(args.years)
        memory = 25 if getattr(args, "tt_preselection", False) else 80
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "output_dir": output_dir,
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
            "years": years_str,
            "compare_args": compare_args_str,
            "memory": memory,
        }
    elif args.study_rescaling:
        memory = 25 if getattr(args, "tt_preselection", False) else 80
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "modelname": args.modelname,
            "output_dir": str(
                BDT_DIR / job_type / presel / run_tag / f"rescaling_{args.modelname}"
            ),
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
            "memory": memory,
        }
    else:
        # Training mode arguments (template uses $name for --model)
        memory = 25 if getattr(args, "tt_preselection", False) else 80
        args_dict = {
            "job_name": "-".join(args.job_name.split("_")),
            "name": args.modelname,
            "modelname": args.modelname,
            "output_dir": str(BDT_DIR / job_type / presel / run_tag / args.modelname),
            "args": extra_args,
            "datapath": str(PVC / args.datapath),
            "memory": memory,
        }

    with Path.open(file_name, "w") as f:
        f.write(lines.substitute(args_dict))

    if args.submit:
        os.system(f"kubectl create -f {file_name} -n cms-ml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default="", help="json file to load args from", type=str)
    parser.add_argument(
        "--modelname",
        default="",
        help="Model name to pass (used for training and base for rescaling study)",
        type=str,
    )
    parser.add_argument("--job-name", default="", help="defaults to name", type=str)
    parser.add_argument(
        "--compare-models",
        default=False,
        help="use comparison mode to compare multiple trained models",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--compare-light",
        default=False,
        help="lightweight comparison using only metrics_summary.csv files (no data loading)",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--disc-filter",
        default=None,
        help="substring filter for discriminant names in --compare-light (e.g. 'vsAll')",
        type=str,
    )
    parser.add_argument(
        "--study-rescaling",
        default=False,
        help="use balance study mode to sweep balance strategies",
        type=bool,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="model JSON files and/or directories to compare (paths relative to BDT_DIR or absolute)",
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
    parser.add_argument(
        "--tag",
        default="",
        help="optional custom tag for grouping (default: modelname for training/rescaling, compare_tag or input names for comparisons)",
        type=str,
    )
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
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
