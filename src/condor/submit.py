"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template

import yaml
from boostedhh import hh_vars, run_utils, submit_utils

from bbtautau import bbtautau_utils

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
    if args.fatjet_pt_cut is not None:
        processor_args += f" --fatjet-pt-cut {args.fatjet_pt_cut}"
    if args.prescale_factor is not None:
        processor_args += f" --prescale-factor {args.prescale_factor}"
    if args.fatjet_bb_preselection_cut is not None:
        processor_args += f" --fatjet-bb-preselection-cut {args.fatjet_bb_preselection_cut}"
    if args.fatjet_tt_preselection_cut is not None:
        processor_args += f" --fatjet-tt-preselection-cut {args.fatjet_tt_preselection_cut}"
    processor_args += (
        " --save-kinematic-systematics"
        if args.save_kinematic_systematics
        else " --no-save-kinematic-systematics"
    )
    submit_utils.submit(args, proxy, t2_prefixes, outdir, local_dir, fileset, processor_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_hh_args(parser)
    run_utils.parse_common_run_args(parser)
    submit_utils.parse_submit_args(parser)
    bbtautau_utils.parse_common_run_args(parser)
    args = parser.parse_args()

    print(f"Submitting for years {args.year}")
    years = args.year

    # YAML check
    if args.yaml is not None:
        with Path(args.yaml).open() as file:
            # str keys: an unquoted year (2022) would otherwise load as an int
            samples_to_submit = {str(k): v for k, v in yaml.safe_load(file).items()}

        # per-year YAML: top-level keys are years (other top-level keys, e.g. YAML anchor
        # definitions, are ignored); old flat YAML: top-level keys are samples, used for all years
        per_year = any(k in hh_vars.years for k in samples_to_submit)

        tag = args.tag
        for year in years:
            print(f"Submitting for year {year}")
            if per_year:
                if year not in samples_to_submit:
                    raise KeyError(f"Year {year} not found in per-year YAML {args.yaml}")
                tdict = samples_to_submit[year]
            else:
                print(f"Flat YAML (no year keys): using the full YAML for {year}")
                tdict = samples_to_submit

            args.year = year
            for sample, sdict in tdict.items():
                args.samples = [sample]
                subsamples = sdict.get("subsamples", [])
                args.maxchunks = sdict.get("maxchunks", 0)
                args.chunksize = sdict.get("chunksize", 40000)
                args.batch_size = sdict.get("batch_size", 20)
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
