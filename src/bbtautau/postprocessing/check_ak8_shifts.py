"""Inspect saved AK8 columns and direction matches in existing skim files."""

from __future__ import annotations

import argparse
import ast
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bbtautau.postprocessing.systematics import (
    AK8_JET_SOURCES,
    DIRECTIONS,
    match_object_slots,
    shifted_ak8_pt_mask,
    values,
)
from bbtautau.postprocessing.utils import base_filters_default

PREFIX = "ak8FatJet"
SHIFTS = tuple(f"{source}_{direction}" for source in AK8_JET_SOURCES for direction in DIRECTIONS)
FIELDS = (
    "Pt",
    "Eta",
    "Phi",
    "Mass",
    "ParTXbbvsQCDTop",
    "ParTXtauhtauhvsQCDTop",
    "PNetmassLegacy",
    "ParTmassResApplied",
    "ParTmassVisApplied",
    "CAglobalParT_massVisApplied",
    "CAglobalParT_massVisApplied_with_delta_axis_merged",
)


def inspect_file(path):
    parquet = pq.ParquetFile(path)
    physical_names = {
        name: ast.literal_eval(name)[0] if name.startswith("(") else name
        for name in parquet.schema_arrow.names
    }
    slots = Counter(physical_names.values())
    print(f"\n{path} ({parquet.metadata.num_rows} events before postprocessing)")
    print("Columns per branch (0 means absent):")
    print("field\tnominal\t" + "\t".join(SHIFTS))
    for field in FIELDS:
        name = f"{PREFIX}{field}"
        counts = [slots[name], *(slots[f"{name}_{shift}"] for shift in SHIFTS)]
        print(field + "\t" + "\t".join(map(str, counts)))
    index_branches = sorted(
        name
        for name in slots
        if name.startswith((PREFIX, "FatJet"))
        and ("idx" in name.lower() or "index" in name.lower())
    )
    print("AK8 index-like branches:", index_branches)

    p4 = {f"{PREFIX}{field}" for field in ("Pt", "Eta", "Phi")}
    identifiers = ("run", "luminosityBlock", "event")
    wanted = p4 | {f"{name}_{shift}" for name in p4 for shift in SHIFTS} | set(identifiers)
    columns = [column for column, branch in physical_names.items() if branch in wanted]
    events = pd.read_parquet(path, columns=columns)
    for shift in SHIFTS:
        missing = sorted((p4 | {f"{name}_{shift}" for name in p4}) - slots.keys())
        if missing:
            print(f"{shift}: cannot match directions; missing {missing}")
            continue
        indices, valid, counts = match_object_slots(events, PREFIX, shift)
        unresolved = valid & (indices < 0)
        eligible = shifted_ak8_pt_mask(events, shift, base_filters_default)
        unresolved_events = unresolved.any(axis=1)
        print(
            f"{shift}: {np.count_nonzero(valid & (indices >= 0))}/{valid.sum()} slots matched, "
            f"{np.count_nonzero(unresolved & (counts == 0))} without nominal candidate, "
            f"{np.count_nonzero(unresolved & (counts > 0))} ambiguous; "
            f"{np.count_nonzero(unresolved_events & eligible)}/{unresolved_events.sum()} "
            "events with unresolved slots pass the shifted AK8 pT cuts"
        )
        if np.any(unresolved_events & eligible):
            pt = values(events, f"{PREFIX}Pt_{shift}")
            for row in np.flatnonzero(unresolved_events & eligible)[:5]:
                event_id = tuple(
                    int(values(events, name)[row, 0]) for name in identifiers if slots[name]
                )
                print(
                    f"  row={row} event_id={event_id} pT={pt[row].tolist()} "
                    f"unresolved_slots={np.flatnonzero(unresolved[row]).tolist()}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", help="One or more skim Parquet files")
    for path in parser.parse_args().parquet:
        inspect_file(path)


if __name__ == "__main__":
    main()
