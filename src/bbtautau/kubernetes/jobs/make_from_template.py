import argparse
from argparse import BooleanOptionalAction

from string import Template

import sys
import os
from os.path import exists


templ_file = "template.yml"


class objectview(object):
    """converts a dict into an object"""

    def __init__(self, d):
        self.__dict__ = d


def from_json(args):
    import json

    with open(args.from_json, "r") as f:
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
            args.job_name = args.name

    args.job_name = "_".join(args.job_name.split("-"))  # hyphens to underscores

    file_name = f"bdt_trainings/{args.job_name}.yml"

    if exists(file_name):
        print("Job Exists")
        if args.overwrite:
            print("Overwriting")
        else:
            print("Exiting")
            sys.exit()

    with open(templ_file, "r") as f:
        lines = Template(f.read())

    train_args = {
        "job_name": "-".join(args.job_name.split("_")),  # change underscores to hyphens
        "name": args.name,
        "args": args.train_args,
        "datapath": args.datapath,
    }

    with open(file_name, "w") as f:
        f.write(lines.substitute(train_args))

    if args.submit:
        os.system(f"kubectl create -f {file_name} -n cms-ml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default="", help="json file to load args from", type=str)
    parser.add_argument("--name", default="", help="", type=str)
    parser.add_argument("--job-name", default="", help="defaults to name", type=str)
    parser.add_argument("--datapath", default="", help="", type=str)
    parser.add_argument(
        "--train-args",
        default="",
        help="Arguments for training. If it's a single argument, may need to add a space afterwards",
        type=str,
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
