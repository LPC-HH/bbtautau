"""Independent systematic views and template contracts, adapted from LPC-HH/HH4b.

HH4b's shift -> inference -> selection -> template pattern is retained. A view
places one shift's inputs under their nominal names, so bbtautau's existing
LoadedSample aliases, channel vetoes and k-fold feature definitions are reused.
No nuisance cross products and no in-place filtering of the source sample.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import re
from pathlib import Path

import numpy as np

AK8_JET_SOURCES = ("JES", "JER")
LEPTON_SHIFT_OBJECTS = {
    "ElectronScale": "Electron",
    "ElectronSmear": "Electron",
    "MuonScale": "Muon",
    "MuonReso": "Muon",
}
KINEMATIC_SOURCES = (
    *AK8_JET_SOURCES,
    "TES",
    *LEPTON_SHIFT_OBJECTS,
    "UnclusteredEnergy",
    "JMS",
    "JMR",
)
EVENT_SFS = (
    "ElectronRecoEventSF",
    "ElectronWP90NoIsoEventSF",
    "MuonTightIDEventSF",
    "TauVSjetSFEvent",
)
DEFAULT_KINEMATIC_SOURCES = KINEMATIC_SOURCES[:8]
DIRECTIONS = ("up", "down")
SCHEMA_VERSION = 1


def field_name(column):
    return str(column[0] if isinstance(column, tuple) else column)


def branches(events):
    return set(map(field_name, events.columns))


def values(events, name):
    array = events[name].to_numpy()
    return array[:, None] if array.ndim == 1 else array


def assign(events, name, array):
    """Assign a whole branch without pandas aligning string and integer slot labels."""
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if name in branches(events):
        cols = [c for c in events.columns if field_name(c) == name]
        if len(cols) != array.shape[1]:
            raise ValueError(f"{name}: {len(cols)} columns but {array.shape[1]} values")
        events[cols] = array
    elif getattr(events.columns, "nlevels", 1) > 1:
        for slot in range(array.shape[1]):
            events[(name, str(slot))] = array[:, slot]
    else:
        if array.shape[1] != 1:
            raise ValueError(f"Flat-column frame cannot hold vector branch {name}")
        events[name] = array[:, 0]


def copy_sample(sample):
    result = copy.copy(sample)
    result.events = sample.events.copy(deep=True)
    for attr in ("bb_mask", "tt_mask", "e_mask", "m_mask"):
        mask = getattr(sample, attr, None)
        setattr(result, attr, None if mask is None else mask.copy())
    return result


def validate_shifts(sample, sources):
    """Fail on a missing requested variation; never silently manufacture a template."""
    if sample.sample.isData or not len(sample.events):
        return
    names = branches(sample.events)
    for source in sources:
        for direction in DIRECTIONS:
            shift = f"{source}_{direction}"
            if not any(n.endswith("_" + shift) for n in names):
                raise ValueError(
                    f"{sample.sample.label}: no {shift} branches. Re-skim with "
                    "--save-kinematic-systematics or request only sources present in this sample."
                )
        # The primary object must exist, even if a downstream CA branch happens to exist.
        anchor = {
            "JES": "ak8FatJetPt",
            "JER": "ak8FatJetPt",
            "TES": "TauPt",
            **{source: f"{prefix}Pt" for source, prefix in LEPTON_SHIFT_OBJECTS.items()},
            "UnclusteredEnergy": "METPt",
        }.get(source)
        if anchor:
            for direction in DIRECTIONS:
                if f"{anchor}_{source}_{direction}" not in names:
                    raise ValueError(f"Missing primary variation {anchor}_{source}_{direction}")


def match_object_slots(events, prefix, shift):
    """Return the unique nominal slot, validity and candidate count for each shifted slot."""
    names = branches(events)
    nominal = tuple(f"{prefix}{field}" for field in ("Pt", "Eta", "Phi"))
    varied = tuple(f"{name}_{shift}" for name in nominal)
    if prefix != "ak8FatJet":
        # Older lepton inputs sometimes only provide shifted pT.
        varied = tuple(new if new in names else old for old, new in zip(nominal, varied))
    missing = sorted({*nominal, *varied} - names)
    if missing:
        raise ValueError(f"{shift}: missing columns for {prefix} direction matching: {missing}")

    old_pt, old_eta, old_phi = (values(events, name) for name in nominal)
    pt, eta, phi = (values(events, name) for name in varied)
    valid = pt > 0
    dphi = (phi[:, :, None] - old_phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    candidates = (
        valid[:, :, None]
        & (old_pt > 0)[:, None, :]
        & (np.abs(eta[:, :, None] - old_eta[:, None, :]) < 1e-5)
        & (np.abs(dphi) < 1e-5)
    )
    counts = candidates.sum(axis=2)
    one_to_one = (counts == 1) & (candidates & (candidates.sum(axis=1, keepdims=True) == 1)).any(
        axis=2
    )
    return np.where(one_to_one, candidates.argmax(axis=2), -1), valid, counts


def _missing_object_fields(events, names, prefix, shift):
    if f"{prefix}Pt_{shift}" not in names:
        return []
    p4 = {f"{prefix}{field}" for field in ("Pt", "Eta", "Phi", "Mass")}
    needed = [
        n
        for n in names
        if n.startswith(prefix)
        and n not in p4
        and "EventSF" not in n
        and not re.search(r"_(up|down)$", n)
        and not n.endswith(("Up", "Down"))
        and f"{n}_{shift}" not in names
    ]
    if prefix in ("Electron", "Muon"):
        needed = [n for n in needed if n == prefix + "charge"]
    if not needed:
        return []
    nslots = values(events, f"{prefix}Pt").shape[1]
    return sorted(n for n in needed if values(events, n).shape[1] == nslots)


def _align_auxiliary_columns(source, target, shift, sample_label):
    """Fill absent shifted fields from the same object in the nominal collection."""
    names = branches(source)
    for prefix in ("ak8FatJet", "Electron", "Muon"):
        needed = _missing_object_fields(source, names, prefix, shift)
        if not needed:
            continue
        indices, valid, counts = match_object_slots(source, prefix, shift)
        bad = valid & (indices < 0)
        if np.any(bad):
            raise ValueError(
                f"{shift}: {int(np.count_nonzero(bad))} {prefix} slots have no unique nominal "
                f"identity in {int(np.count_nonzero(bad.any(axis=1)))} events of {sample_label}. "
                f"No nominal candidate: {np.count_nonzero(bad & (counts == 0))}. "
                f"Ambiguous: {np.count_nonzero(bad & (counts > 0))}. "
                f"Missing aligned properties: {needed}."
            )
        for name in needed:
            aligned = np.take_along_axis(values(source, name), np.maximum(indices, 0), axis=1)
            aligned = np.where(valid, aligned, -99999)
            assign(target, name, aligned)


def shifted_ak8_pt_mask(events, shift, filters):
    """Keep events that can pass the deferred selection based on saved shifted pT."""
    if not filters:
        return np.ones(len(events), dtype=bool)
    groups = [filters] if isinstance(filters[0], tuple) else filters
    pt_filters = [
        [
            (str(column).replace("ak8FatJetPt", f"ak8FatJetPt_{shift}", 1), op, value)
            for column, op, value in group
            if str(column).startswith("('ak8FatJetPt',")
        ]
        for group in groups
    ]
    return evaluate_filters(events, pt_filters)


def _select_matched_events(samples, sources, shift_objects):
    """Build one common MC event subset for nominal and all requested shifts."""
    shifts = [
        (f"{source}_{direction}", shift_objects[source])
        for source in sources
        if source in shift_objects
        for direction in DIRECTIONS
    ]
    selected = dict(samples)
    summary = {}
    for key, sample in samples.items():
        if sample.sample.isData:
            continue
        events = sample.events
        names = branches(events)
        keep = np.ones(len(events), dtype=bool)
        unresolved = {}
        for shift, prefix in shifts:
            unresolved[shift] = 0
            if not len(events) or not _missing_object_fields(events, names, prefix, shift):
                continue
            indices, valid, _ = match_object_slots(events, prefix, shift)
            bad = (valid & (indices < 0)).any(axis=1)
            if prefix == "ak8FatJet":
                bad &= shifted_ak8_pt_mask(events, shift, events.attrs.get("systematics_filters"))
            unresolved[shift] = int(bad.sum())
            keep &= ~bad
        summary[key] = {
            "input_events": len(events),
            "excluded_events": int((~keep).sum()),
            "unresolved_events_by_shift": unresolved,
        }
        if not keep.all():
            selected[key] = sample.copy_from_selection(keep)
    return selected, summary


def select_ak8_matched_events(samples, sources):
    return _select_matched_events(samples, sources, dict.fromkeys(AK8_JET_SOURCES, "ak8FatJet"))


def select_lepton_matched_events(samples, sources):
    return _select_matched_events(samples, sources, LEPTON_SHIFT_OBJECTS)


def matched_template_tag(ak8_matched_only, lepton_matched_only):
    """Shared directory and filename tag for diagnostic templates."""
    if lepton_matched_only:
        return "ak8_lepton_matched_only" if ak8_matched_only else "lepton_matched_only"
    return "ak8_matched_only" if ak8_matched_only else ""


def shifted_sample(sample, shift):
    filters = sample.events.attrs.get("systematics_filters")
    if (
        shift
        and shift.rsplit("_", 1)[0] in AK8_JET_SOURCES
        and filters
        and not sample.sample.isData
    ):
        keep = shifted_ak8_pt_mask(sample.events, shift, filters)
        if not keep.all():
            sample = sample.copy_from_selection(keep)

    result = copy_sample(sample)
    if not shift or sample.sample.isData:
        result.events.attrs["kinematic_shift"] = ""
        return result
    suffix = "_" + shift
    replacements = {n[: -len(suffix)]: n for n in branches(sample.events) if n.endswith(suffix)}
    if not replacements:
        raise ValueError(f"{sample.sample.label}: missing {shift}")
    for nominal, varied in replacements.items():
        varied_values = values(sample.events, varied)
        if np.issubdtype(varied_values.dtype, np.number) and not np.isfinite(varied_values).all():
            raise ValueError(f"{sample.sample.label}: non-finite values in {varied}")
        assign(result.events, nominal, varied_values)
    _align_auxiliary_columns(sample.events, result.events, shift, sample.sample.label)
    result.events.attrs["kinematic_shift"] = shift
    return result


def apply_event_sfs(sample, sf_names):
    """Multiply explicitly selected SFs exactly once, keeping the pre-SF weight.

    Call again after overlaying shifted SF columns: reconstructing from the
    stored pre-SF weight handles a zero nominal SF without division by zero.
    This requires finalWeight to EXCLUDE these SFs before the first call.
    """
    if sample.sample.isData:
        return
    events = sample.events
    shift = events.attrs.get("kinematic_shift", "")
    names = branches(events)
    for name in sf_names:
        source = shift.rsplit("_", 1)[0]
        affected = (
            (name.startswith("Electron") and source.startswith("Electron"))
            or (name.startswith("Muon") and source.startswith("Muon"))
            or (name.startswith("Tau") and source == "TES")
        )
        if affected and f"{name}_{shift}" not in names:
            raise ValueError(f"{sample.sample.label}: missing shifted center {name}_{shift}")
    if "__weight_before_event_sfs" not in branches(events):
        assign(events, "__weight_before_event_sfs", values(events, "finalWeight"))
    base = values(events, "__weight_before_event_sfs")[:, 0]
    factors = {}
    for name in sf_names:
        if name not in branches(events):
            raise ValueError(f"{sample.sample.label}: missing requested event SF {name}")
        factors[name] = values(events, name)[:, 0]
        if not np.isfinite(factors[name]).all() or (factors[name] < 0).any():
            raise ValueError(f"Invalid SF in {name}")
    nominal = base.copy()
    for factor in factors.values():
        nominal *= factor
    assign(events, "finalWeight", nominal)
    if shift:
        return
    for name in sf_names:
        for direction in DIRECTIONS:
            branch = name + direction.title()
            if branch not in branches(events):
                raise ValueError(
                    f"{sample.sample.label}: missing {branch}; enable --save-systematics"
                )
            varied = values(events, branch)[:, 0]
            if not np.isfinite(varied).all() or (varied < 0).any():
                raise ValueError(f"Invalid SF in {branch}")
            weight = base * varied
            for other, factor in factors.items():
                if other != name:
                    weight *= factor
            assign(events, f"finalWeight_{name}_{direction}", weight)


def weight_variations(sample, sf_names, weight_sources):
    """Return complete normalized event weights, indexed by nuisance and direction."""
    if sample.sample.isData:
        return {}
    out = {}
    for source in sf_names:
        out[source] = {d: sample.get_var(f"finalWeight_{source}_{d}") for d in DIRECTIONS}
    sf_product = np.prod([sample.get_var(name) for name in sf_names], axis=0) if sf_names else 1.0
    for source in weight_sources:
        names = [f"weight_{source}{d.title()}" for d in DIRECTIONS]
        if any(n not in branches(sample.events) for n in names):
            raise ValueError(f"{sample.sample.label}: missing requested weights {names}")
        out[source] = {d: sample.get_var(n) * sf_product for d, n in zip(DIRECTIONS, names)}
    return out


def write_manifest(
    path,
    year,
    samples,
    sources,
    sf_names,
    weight_sources,
    *,
    ak8_match_summary=None,
    lepton_match_summary=None,
):
    affected = [k for k, s in samples.items() if not s.sample.isData]
    specs = {}
    for source in (*sources, *sf_names, *weight_sources):
        specs[source] = {
            "kind": "kinematic" if source in sources else "weight",
            "samples": affected,
            # Initial HH4b-like policy. Override after reviewing POG correlation guidance.
            "correlation": "all" if source == "JES" else "year",
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "year": year,
        "nuisances": specs,
        "event_sfs_multiplied": list(sf_names),
        "kinematic_sources": list(sources),
    }
    if ak8_match_summary is not None:
        payload["ak8_matched_only"] = ak8_match_summary
    if lepton_match_summary is not None:
        payload["lepton_matched_only"] = lepton_match_summary
    skipped = {
        key: sample.events.attrs["skipped_empty_mc_subsamples"]
        for key, sample in samples.items()
        if sample.events.attrs.get("skipped_empty_mc_subsamples")
    }
    if skipped:
        payload["skipped_empty_mc_subsamples"] = skipped
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported systematics manifest: {path}")
    return data


def template_variation(templates, region, sample, source, direction, kind):
    blind = "MCBlinded" if region.endswith("MCBlinded") else ""
    base = region.removesuffix(blind) if blind else region
    if kind == "kinematic":
        return templates[f"{base}_{source}_{direction}{blind}"][sample, ...].values()
    return templates[region][f"{sample}_{source}_{direction}", ...].values()


def load_sample_with_systematics(
    sample,
    year,
    paths,
    filters=None,
    weight_sources=(),
    *,
    defer_filters=False,
    skip_empty_mc_subsamples=False,
):
    """Strict per-subsample loading/normalization, using boostedhh's normalization.

    Read all columns because shift availability is encoded in each skim's schema.
    Do not push nominal MC kinematic cuts into parquet: migrated events must survive.
    """
    import hashlib

    import pandas as pd
    from boostedhh import utils

    roots = paths[sample.get_type()]
    roots = roots if isinstance(roots, list) else [roots]
    selected = []
    for root in roots:
        directory = Path(root) / year
        if not directory.is_dir():
            continue
        selected = sorted(
            p for p in directory.iterdir() if p.is_dir() and sample.get_selector(year).match(p.name)
        )
        if selected:
            break
    if not selected:
        raise FileNotFoundError(f"No input matched {sample.label} in {roots} for {year}")
    frames, provenance, skipped = [], [], []
    parquet_filters = None if defer_filters and not sample.isData else filters
    requested_weights = [
        f"weight_{source}{direction.title()}"
        for source in weight_sources
        if not sample.isData
        for direction in DIRECTIONS
    ]
    for directory in selected:
        parquet = directory / "parquet"
        if parquet.is_dir():
            files = sorted(p for p in parquet.iterdir() if p.name.endswith(".parquet"))
        else:
            try:
                parquet.lstat()
            except FileNotFoundError:
                files = []
            else:
                files = [parquet]
        if not files and skip_empty_mc_subsamples and not sample.isData:
            logging.getLogger(__name__).warning("Skipping empty MC subsample %s", directory)
            skipped.append(directory.name)
            continue
        if not files or not all(p.is_file() for p in files):
            raise FileNotFoundError(f"No parquet files in {parquet}")
        # Reading fragments individually prevents a mixed old/new schema being silently
        # truncated to the first fragment's schema by pyarrow.dataset.
        parts = [pd.read_parquet(p, filters=parquet_filters) for p in files]
        schemas = [set(p.columns) for p in parts]
        if any(s != schemas[0] for s in schemas[1:]):
            raise ValueError(f"Mixed parquet schemas in {parquet}; do not mix skim versions")
        frame = pd.concat(parts, ignore_index=True)
        totals_file = utils.get_pickles(directory / "pickles", year, directory.name)
        for branch in requested_weights:
            if branch not in frame:
                raise ValueError(f"{directory}: missing {branch}")
        if "totals" in totals_file:
            utils._normalize_weights(
                frame,
                year,
                totals_file["totals"],
                directory.name,
                sample.isData,
                variations=bool(weight_sources),
                weight_shifts=dict.fromkeys(weight_sources),
            )
        else:
            if weight_sources and not sample.isData:
                raise ValueError(f"{directory}: totals are required to normalize weight variations")
            total = (
                1
                if sample.isData
                else utils.get_nevents(directory / "pickles", year, directory.name)
            )
            assign(frame, "finalWeight", values(frame, "weight") / total)
        for branch in requested_weights:
            if not np.isfinite(values(frame, branch)).all():
                raise ValueError(f"{directory}: non-finite normalized {branch}")
        # dataset identifier, part of the BDT event key (as utils.load_sample_with_dataset_id)
        from bbtautau.postprocessing.bdt_utils import DATASET_ID_COLUMN, dataset_id

        frame[DATASET_ID_COLUMN] = dataset_id(year, directory.name)
        frames.append(frame)
        provenance += [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    if not frames:
        raise FileNotFoundError(
            f"No parquet files in any matched {sample.label} subsample for {year}"
        )
    result = pd.concat(frames, ignore_index=True)
    if defer_filters and not sample.isData:
        result.attrs["systematics_filters"] = filters
    result.attrs["systematics_weight_sources"] = tuple(weight_sources)
    result.attrs["systematics_dataset"] = hashlib.sha256(repr(provenance).encode()).hexdigest()[:16]
    if skipped:
        result.attrs["skipped_empty_mc_subsamples"] = skipped
    return result


def evaluate_filters(events, filters):
    """Evaluate pandas/pyarrow DNF filters on a prepared frame, without row loss upstream."""
    import ast
    import operator

    comparisons = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if not filters:
        return np.ones(len(events), dtype=bool)
    groups = [filters] if isinstance(filters[0], tuple) else filters
    result = np.zeros(len(events), dtype=bool)
    for group in groups:
        mask = np.ones(len(events), dtype=bool)
        for column, comparison, value in group:
            key = column
            if key not in events.columns:
                with contextlib.suppress(ValueError, SyntaxError):
                    key = ast.literal_eval(column)
            if key not in events.columns and isinstance(key, tuple) and len(key) == 2:
                branch, slot = key
                if isinstance(slot, str) and slot.isdecimal():
                    key = (branch, int(slot))
            if key not in events.columns:
                raise KeyError(f"Missing preselection column {column}")
            array = events[key].to_numpy().reshape(len(events))
            mask &= comparisons[comparison](array, value)
        result |= mask
    return result


def require_feature_variations(sample, shift, year, features):
    """Reject nominal fallbacks for known affected CA/MET inputs actually consumed."""
    if not shift or sample.sample.isData:
        return
    source = shift.rsplit("_", 1)[0]
    names = branches(sample.events)
    for feature in features:
        physical = feature.replace("bbFatJet", "ak8FatJet").replace("ttFatJet", "ak8FatJet")
        affected = physical.startswith("ak8FatJetCA") and source != "UnclusteredEnergy"
        affected |= physical in {"METPt", "METPhi"} and (
            source == "UnclusteredEnergy" or (year == "2024" and source not in {"JMS", "JMR"})
        )
        if affected and physical + "_" + shift not in names:
            raise ValueError(
                f"{sample.sample.label}: {feature} needs {physical}_{shift}; "
                "this model/template cannot use the nominal branch as a substitute"
            )
