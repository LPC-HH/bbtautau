"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template

import yaml
from boostedhh import run_utils, submit_utils

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
    "ucsd": "root://redirector.t2.ucsd.edu:1095//",
}


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    proxy, t2_prefixes, outdir, local_dir = submit_utils.init_args(args)

    fileset = run_utils.get_fileset(
        f"data/index_{args.year}.json",
        args.year,
        args.samples,
        args.subsamples,
        get_num_files=True,
    )

    processor_args = f"--region {args.region}"
    submit_utils.submit(args, proxy, t2_prefixes, outdir, local_dir, fileset, processor_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_hh_args(parser)
    run_utils.parse_common_run_args(parser)
    submit_utils.parse_submit_args(parser)
    args = parser.parse_args()

    # YAML check
    if args.yaml is not None:
        with Path(args.yaml).open() as file:
            samples_to_submit = yaml.safe_load(file)

        tag = args.tag
        for key, tdict in samples_to_submit.items():
            # print(f"Submitting for year {key}")
            args.year = key
            for sample, sdict in tdict.items():
                args.samples = [sample]
                subsamples = sdict.get("subsamples", [])
                args.maxchunks = sdict.get("maxchunks", 0)
                args.chunksize = sdict.get("chunksize", 40000)
                args.tag = tag
                files_per_job = sdict["files_per_job"]
                if isinstance(files_per_job, dict):
                    for subsample in subsamples:
                        args.subsamples = [subsample]
                        args.files_per_job = files_per_job[subsample]
                        print(args)
                        main(args)
                else:
                    args.subsamples = subsamples
                    args.files_per_job = files_per_job
                    print(args)
                    main(args)
    else:
        main(args)