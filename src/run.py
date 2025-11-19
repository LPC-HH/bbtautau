"""
Runs coffea processors on the LPC via either condor or dask.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from boostedhh import run_utils
from boostedhh.hh_vars import DATA_SAMPLES
from boostedhh.xsecs import xsecs

from bbtautau import bbtautau_utils


def get_processor(
    processor: str,
    save_systematics: bool | None = None,
    region: str | None = None,
    nano_version: str | None = None,
    fatjet_pt_cut: float | None = None,
    fatjet_bb_preselection: bool | None = None,
    prescale_factor: int | None = None,
):
    # define processor
    if processor == "skimmer":
        from bbtautau.processors import bbtautauSkimmer

        return bbtautauSkimmer(
            xsecs=xsecs,
            save_systematics=save_systematics,
            region=region,
            nano_version=nano_version,
            fatjet_pt_cut=fatjet_pt_cut,
            fatjet_bb_preselection=fatjet_bb_preselection,
            prescale_factor=prescale_factor,
        )


def main(args):
    p = get_processor(
        args.processor,
        args.save_systematics,
        args.region,
        args.nano_version,
        args.fatjet_pt_cut,
        args.fatjet_bb_preselection,
        args.prescale_factor,
    )

    save_parquet = {"skimmer": True}[args.processor]
    save_root = {"skimmer": True}[args.processor]

    skipbadfiles = True

    if len(args.files):
        fileset = {f"{args.year}_{args.files_name}": args.files}
        skipbadfiles = False  # not added functionality for args.files yet
    else:
        if args.yaml:
            with Path(args.yaml).open() as file:
                samples_to_submit = yaml.safe_load(file)
            try:
                samples_to_submit = samples_to_submit[args.year]
            except Exception as e:
                raise KeyError(f"Year {args.year} not present in yaml dictionary") from e

            samples = samples_to_submit.keys()
            subsamples = []
            for sample in samples:
                subsamples.extend(samples_to_submit[sample].get("subsamples", []))
        else:
            samples = args.samples
            subsamples = args.subsamples

        fileset = run_utils.get_fileset(
            f"data/index_{args.year}.json",
            args.year,
            samples,
            subsamples,
            args.starti,
            args.endi,
        )

        # don't skip "bad" files for data - we want it throw an error in that case
        for key in fileset:
            if key in DATA_SAMPLES:
                skipbadfiles = False

    print(f"Running on fileset {fileset}")
    if args.executor == "dask":
        run_utils.run_dask(p, fileset, args)
    else:
        run_utils.run(
            p,
            fileset,
            chunksize=args.chunksize,
            maxchunks=args.maxchunks,
            skipbadfiles=skipbadfiles,
            save_parquet=save_parquet,
            save_root=save_root and args.save_root,
            filetag=f"{args.starti}-{args.endi}" if args.file_tag is None else args.file_tag,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_run_args(parser)
    run_utils.parse_common_hh_args(parser)
    bbtautau_utils.parse_common_run_args(parser)
    args = parser.parse_args()

    if isinstance(args.year, list):
        if len(args.year) == 1:
            args.year = args.year[0]
        else:
            raise ValueError("Running on multiple years is not supported yet")

    main(args)
