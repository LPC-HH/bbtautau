"""
BDT utilities for prediction, loading, and score computation.

Authors: Ludovico Mori
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xgboost as xgb
from boostedhh.utils import PAD_VAL

from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.postprocessing.bdt_config import BDT_CONFIG
from bbtautau.postprocessing.Samples import CHANNELS

# Non-interactive backend for batch/containers
mpl.use("Agg")

# Channel order for BDT models (must match the order used in training)
# This defines the order of channels for each signal: he, hh, hm
CHANNEL_ORDER_BDT = ["he", "hh", "hm"]

WPS_FILENAME = "wps_ttpart.json"


def save_training_wps(output_dir: Path, wps: dict[str, np.ndarray]) -> None:
    """Save WPS bin edges used during training alongside the model artifacts.

    These are loaded at evaluation time to ensure feature binning is consistent.
    """
    wps_path = output_dir / WPS_FILENAME
    wps_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else list(v) for k, v in wps.items()
    }
    with wps_path.open("w") as f:
        json.dump(wps_serializable, f, indent=2)
    print(f"WPS bin edges saved to {wps_path}")


def load_training_wps(model_dir: Path, modelname: str) -> dict[str, np.ndarray] | None:
    """Load WPS bin edges saved during training.

    Looks for wps_ttpart.json in model_dir/modelname/.
    Returns None if the file does not exist.
    """
    wps_path = model_dir / modelname / WPS_FILENAME
    if not wps_path.exists():
        return None
    with wps_path.open("r") as f:
        wps_raw = json.load(f)
    return {k: np.array(v) for k, v in wps_raw.items()}


def bin_values_with_edges(values: np.ndarray | pd.Series, bin_edges: np.ndarray) -> np.ndarray:
    """Bin continuous values using custom bin edges (e.g., from working points).

    Values are assigned to the bin whose left edge they fall into.
    Values below the first edge are assigned to the first bin.
    Values above the last edge are assigned to the last bin.

    Args:
        values: Input array or Series
        bin_edges: Array of bin edge values (sorted, ascending)

    Returns:
        Binned values as numpy array (values are set to the left edge of their bin)
    """
    values = np.asarray(values)
    bin_edges = np.asarray(bin_edges)

    # Use digitize to find which bin each value belongs to
    # digitize returns indices 1..len(bin_edges), where:
    # - index 0 means value < bin_edges[0] (we'll handle separately)
    # - index i means bin_edges[i-1] <= value < bin_edges[i]
    # - index len(bin_edges) means value >= bin_edges[-1]

    bin_indices = np.digitize(values, bin_edges)

    # Create binned values: map indices to bin left edges
    # Index 0 -> use first edge
    # Index i (1..len-1) -> use bin_edges[i-1]
    # Index len -> use last edge
    binned = np.zeros_like(values)

    # Values below first edge: use first edge
    mask_below = bin_indices == 0
    if np.any(mask_below):
        binned[mask_below] = bin_edges[0]

    # Values in middle bins: use left edge of bin
    for i in range(1, len(bin_edges)):
        mask = bin_indices == i
        if np.any(mask):
            binned[mask] = bin_edges[i - 1]

    # Values above last edge: use last edge
    mask_above = bin_indices == len(bin_edges)
    if np.any(mask_above):
        binned[mask_above] = bin_edges[-1]

    return binned


def bin_features(
    events: pd.DataFrame,
    features: list[str],
    bin_edges_dict: dict[str, np.ndarray] = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """Bin specified DataFrame columns into discrete values using WP bin edges.

    Only features that have entries in bin_edges_dict will be binned.
    Features in the features list but not in bin_edges_dict will be skipped.

    Args:
        events: DataFrame containing the features
        features: List of feature names to potentially bin
        bin_edges_dict: Dictionary mapping feature names to custom bin edges (e.g., from WPs).
                       Only features in this dict will be binned.
        inplace: If True, modify df in place; if False, return a copy

    Returns:
        DataFrame with binned features (only those with entries in bin_edges_dict)
    """
    if not inplace:
        events = events.copy()

    if bin_edges_dict is None:
        bin_edges_dict = {}

    for feat in features:
        if feat not in events.columns:
            print(f"Warning: Feature '{feat}' not found in DataFrame, skipping binning")
            continue

        # Only bin if bin edges are provided
        if feat in bin_edges_dict:
            bin_edges = np.asarray(bin_edges_dict[feat])
            print(f"  Binning '{feat}' using {len(bin_edges)} WP bin edges")
            events[feat] = bin_values_with_edges(events[feat], bin_edges)
        else:
            print(f"  Skipping '{feat}' - no bin edges provided")

    return events


# Background order for BDT models
DEFAULT_BKG_ORDER = ["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]


def get_expected_sample_order(signals: list[str]) -> list[str]:
    """Get the expected sample order for BDT model training/inference.

    The order is: signal channels (he, hh, hm) for each signal, then backgrounds.
    This matches the class order expected by the BDT model.

    Args:
        signals: List of signal keys (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])

    Returns:
        List of sample names in the expected order for BDT model classes
    """
    sample_names = []
    # For each signal, add its channels in order: he, hh, hm
    for signal_key in signals:
        for ch in CHANNEL_ORDER_BDT:
            sample_names.append(f"{signal_key}{ch}")

    # Add backgrounds
    sample_names.extend(DEFAULT_BKG_ORDER)

    return sample_names


def get_bdt_preds_dir(
    base_dir: Path,
    modelname: str,
    signal_objectives: list[str],
    channel: Channel,
    year: str,
    n_folds: int = 1,
    tt_pres: bool = False,
    test_mode: bool = False,
) -> Path:
    """Get the directory path for BDT predictions cache files.

    Directory structure: base_dir / tag / signal / modelname[_kfold{n}] / channel.key / year

    Where:
    - tag is "test", "tt_pres", or "full_presel" based on test_mode and tt_pres
    - signal is the first signal objective if there's one, otherwise "unified_signals"
    - For n_folds>1, appends "_kfold{n}" to modelname

    Args:
        base_dir: Base directory for BDT predictions
        modelname: Name of the BDT model
        signal_objectives: List of signal objectives (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])
        channel: Channel object with .key attribute
        year: Year string (e.g., '2018')
        n_folds: Number of folds (1 for single model, >1 for k-fold)
        tt_pres: Whether tt preselection is applied
        test_mode: Whether in test mode

    Returns:
        Path to the directory for BDT predictions
    """
    # Determine tag based on test_mode and tt_pres
    if test_mode:
        tag = "test"
    elif tt_pres:
        tag = "tt_pres"
    else:
        tag = "full_presel"

    # Determine signal subdirectory
    if len(signal_objectives) == 1:
        signal_dir = signal_objectives[0]
    else:
        signal_dir = "unified_signals"

    # Add kfold suffix for multi-fold models
    model_dir_name = modelname if n_folds == 1 else f"{modelname}_kfold{n_folds}"

    return base_dir / tag / signal_dir / model_dir_name / channel.key / year


def load_models(model_dir: Path, modelname: str, n_folds: int = 1) -> list[xgb.Booster]:
    """Load trained model(s) from disk.

    This unified function handles both n_folds=1 (single model) and n_folds>1 (k models).

    Args:
        model_dir: Path to the directory containing the models
        modelname: Name of the BDT model
        n_folds: Number of folds (1 for single model, >1 for k-fold)

    Returns:
        List of XGBoost Booster objects (length 1 for single model, length k for k-fold)
    """
    boosters = []
    for fold_idx in range(n_folds):
        if n_folds == 1:
            model_path = model_dir / modelname / f"{modelname}.json"
        else:
            model_path = model_dir / modelname / f"{modelname}_fold{fold_idx}.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Expected structure: model_dir/modelname/modelname[_fold{{i}}].json"
            )
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        boosters.append(bst)
    return boosters


# Mapping from sample names to BDT column name prefixes
# This allows flexible naming of background columns
BKG_COLUMN_NAMES = {
    "dyjets": "BDTDY",
    "qcd": "BDTQCD",
    "ttbarhad": "BDTTThad",
    "ttbarll": "BDTTTll",
    "ttbarsl": "BDTTTSL",
    # Add more mappings here if new backgrounds are added
    "wjets": "BDTWJ",
    "singletop": "BDTST",
    "diboson": "BDTDB",
}


def _get_bdt_key(
    signal: str, channel: Channel = None, prefix_only: bool = True, suffix: str = "vsAll"
):
    """Get the BDT column key for a signal channel.

    Prefix all BDT columns with the signal key to distinguish models.
    Remove trailing 'tt' from signal_key to avoid redundancy (e.g., ggfbbtt -> ggfbb).
    This gives cleaner names like BDTggfbbtauhtauh instead of BDTggfbbtttauhtauh.
    """
    signal_base = signal.removesuffix("tt") if signal.endswith("tt") else signal
    if prefix_only:
        return f"BDT{signal_base}"
    else:
        if channel is None:
            raise ValueError("Channel is required for non-prefix BDT keys")
        return f"BDT{signal_base}{channel.tagger_label}{suffix}"


def _get_bkg_column_name(bkg_sample: str, jshift: str = "") -> str:
    """Get the BDT column name for a background sample.

    Args:
        bkg_sample: Background sample name (e.g., 'dyjets', 'qcd')
        jshift: JEC/JMSR shift label suffix

    Returns:
        Column name (e.g., 'BDTDY', 'BDTQCD_JEC_up')
    """
    if bkg_sample in BKG_COLUMN_NAMES:
        return f"{BKG_COLUMN_NAMES[bkg_sample]}{jshift}"
    else:
        # Fallback: use sample name with BDT prefix for unknown backgrounds
        return f"BDT{bkg_sample.upper()}{jshift}"


def _add_bdt_scores(
    events: pd.DataFrame,
    sample_bdt_preds: np.ndarray,
    sample_order: list[str],
    signal_objectives: list[str],
    jshift: str = "",
):
    """Add BDT scores to events DataFrame.

    This function dynamically adds BDT prediction columns based on the model's
    sample order, making it flexible for different model configurations.

    Args:
        events: DataFrame to add scores to
        sample_bdt_preds: BDT predictions array with shape (n_events, n_classes)
        sample_order: List of sample names in the order they appear in predictions.
            This should match the output of get_expected_sample_order() used during training.
            Example: ['ggfbbtthe', 'ggfbbtthh', 'ggfbbtthm', 'dyjets', 'qcd', 'ttbarhad', 'ttbarll', 'ttbarsl']
        signal_objectives: List of signal keys (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])
        jshift: JEC/JMSR shift label (e.g., 'JEC_up', 'JMSR_down')
    """
    if jshift != "":
        jshift = "_" + jshift

    # Validate inputs
    if not isinstance(signal_objectives, list):
        raise ValueError(f"signal_objectives must be a list, got {signal_objectives}")

    if sample_bdt_preds.shape[1] != len(sample_order):
        raise ValueError(
            f"Prediction shape {sample_bdt_preds.shape} doesn't match sample_order length {len(sample_order)}. "
            f"sample_order: {sample_order}"
        )

    # Build sample name to index mapping
    sample_to_idx = {name: idx for idx, name in enumerate(sample_order)}

    # Identify signal and background samples
    signal_samples = []
    bkg_samples = []
    for sample_name in sample_order:
        is_signal = any(sample_name.startswith(sig) for sig in signal_objectives)
        if is_signal:
            signal_samples.append(sample_name)
        else:
            bkg_samples.append(sample_name)

    # Add raw prediction columns for each class
    # 1. Signal channels
    for signal_obj in signal_objectives:
        for ch_key in CHANNEL_ORDER_BDT:
            sample_name = f"{signal_obj}{ch_key}"
            if sample_name in sample_to_idx:
                idx = sample_to_idx[sample_name]
                col_name = _get_bdt_key(
                    signal_obj, prefix_only=False, channel=CHANNELS[ch_key], suffix=jshift
                )
                events[col_name] = sample_bdt_preds[:, idx]

    # 2. Background samples
    bkg_column_map = {}  # Store column names for vsAll computation
    for bkg_sample in bkg_samples:
        idx = sample_to_idx[bkg_sample]
        col_name = _get_bkg_column_name(bkg_sample, jshift)
        events[col_name] = sample_bdt_preds[:, idx]
        bkg_column_map[bkg_sample] = col_name

    # Add discriminant columns (vsQCD, vsAll) for each signal channel
    qcd_col = bkg_column_map.get("qcd")

    for signal_obj in signal_objectives:
        for ch_key in CHANNEL_ORDER_BDT:
            sample_name = f"{signal_obj}{ch_key}"
            if sample_name not in sample_to_idx:
                continue

            sig_col = _get_bdt_key(
                signal_obj, prefix_only=False, channel=CHANNELS[ch_key], suffix=jshift
            )
            base_sig = events[sig_col]

            # vsQCD discriminant (if QCD is in the model)
            if qcd_col is not None:
                vsqcd_col = _get_bdt_key(
                    signal_obj, prefix_only=False, channel=CHANNELS[ch_key], suffix=f"vsQCD{jshift}"
                )
                events[vsqcd_col] = np.nan_to_num(
                    base_sig / (base_sig + events[qcd_col]),
                    nan=PAD_VAL,
                )

            # vsAll discriminant (signal vs sum of all backgrounds)
            bkg_sum = sum(events[col] for col in bkg_column_map.values())
            vsall_col = _get_bdt_key(
                signal_obj, prefix_only=False, channel=CHANNELS[ch_key], suffix=f"vsAll{jshift}"
            )
            events[vsall_col] = np.nan_to_num(
                base_sig / (base_sig + bkg_sum),
                nan=PAD_VAL,
            )


def compute_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    n_folds: int = 1,
    test_mode: bool = False,
    save_dir: Path = None,
    cache_info: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Compute BDT predictions for multiple years and samples.

    This unified function handles three prediction modes:
    1. Single model (n_folds=1): Standard prediction
    2. K-fold MC samples: Out-of-fold (OOF) predictions using per-sample fold assignments
    3. K-fold DATA samples: Average predictions from all k models

    For MC samples that were used in training, OOF predictions ensure each event is
    scored by a model that never saw it during training (unbiased evaluation).
    For DATA samples (and MC not in training), all k models are averaged.

    Args:
        events_dict: Dictionary mapping years to dictionaries of samples
        modelname: Name of the model to load
        model_dir: Path to the directory containing the model
        signal_objectives: List of signal objectives
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        n_folds: Number of folds (1 for single model, >1 for k-fold)
        test_mode: Whether in test mode
        save_dir: Directory to save predictions
    """
    import time

    start_time = time.time()

    # Get the expected sample order (must match training order)
    sample_order = get_expected_sample_order(signal_objectives)

    # Load model(s) - single model for n_folds=1, k models for n_folds>1
    if n_folds > 1:
        print(f"Loading {n_folds} k-fold models for '{modelname}'...")
    boosters = load_models(model_dir, modelname, n_folds)

    feature_names = [
        feat
        for cat in BDT_CONFIG[modelname]["train_vars"]
        for feat in BDT_CONFIG[modelname]["train_vars"][cat]
    ]

    # Resolve WPS for feature binning: load from training artifacts for consistency
    bin_feats = BDT_CONFIG[modelname].get("bin_features", None)
    bin_edges_loaded = {}
    bin_feat_indices = []
    if bin_feats:
        wps = load_training_wps(model_dir, modelname)
        if wps is None:
            from bbtautau.userConfig import WPS_TTPART

            print(
                f"Warning: No saved WPS found at {model_dir / modelname / WPS_FILENAME}. "
                f"Falling back to runtime WPS_TTPART from userConfig."
            )
            wps = dict(WPS_TTPART)

        bin_edges_loaded = {f: wps[f] for f in bin_feats if f in wps}
        if bin_edges_loaded:
            bin_feat_indices = [i for i, f in enumerate(feature_names) if f in bin_edges_loaded]
            print(
                f"Applying feature binning with WP bin edges to {len(bin_edges_loaded)} features: "
                f"{list(bin_edges_loaded.keys())}"
            )
            skipped = [f for f in bin_feats if f not in bin_edges_loaded]
            if skipped:
                print(f"  Skipping {len(skipped)} features (no bin edges found): {skipped}")
        else:
            print(f"No WP bin edges found for features: {bin_feats}")

    # Process each sample
    for year in events_dict:
        for sample, sample_data in events_dict[year].items():
            # Check if we can use cached predictions for some events
            use_incremental = False
            new_event_indices = None
            cached_preds = None
            cached_event_ids = None
            current_event_ids = None

            if cache_info is not None:
                info = cache_info.get(year, {}).get(sample, {})
                if info and info.get("cache_valid", False):
                    new_event_indices = info.get("new_event_indices", np.array([]))
                    cached_preds = info.get("cached_predictions")
                    cached_event_ids = info.get("cached_event_ids")
                    current_event_ids = info.get("current_event_ids")

                    if (
                        new_event_indices is not None
                        and current_event_ids is not None
                        and len(new_event_indices) < len(current_event_ids)
                    ):
                        # Some events are cached, use incremental computation
                        use_incremental = True
                        print(
                            f"  {sample} in {year}: {len(new_event_indices)} new events, "
                            f"{len(current_event_ids) - len(new_event_indices)} from cache"
                        )

            # Prepare features
            if use_incremental and new_event_indices is not None and len(new_event_indices) > 0:
                # Only prepare features for new events
                features = np.stack(
                    [sample_data.get_var(feat)[new_event_indices] for feat in feature_names],
                    axis=1,
                )
            else:
                # Prepare features for all events
                features = np.stack(
                    [sample_data.get_var(feat) for feat in feature_names],
                    axis=1,
                )

            # Apply feature binning if configured (only for features with WP bin edges)
            if bin_feats and bin_feat_indices:
                for idx in bin_feat_indices:
                    feat_name = feature_names[idx]
                    if feat_name in bin_edges_loaded:
                        features[:, idx] = bin_values_with_edges(
                            features[:, idx], bin_edges_loaded[feat_name]
                        )

            # Determine if this is a DATA sample
            is_data = sample_data.sample.isData

            # Get fold assignments for OOF predictions

            luminosity_block = sample_data.get_var("luminosityBlock")
            event = sample_data.get_var("event")

            fold_assignments = None
            if n_folds > 1 and not is_data:
                # Get fold assignments for all events first
                fold_assignments_all = get_sample_fold_assignments(
                    model_dir, modelname, luminosity_block=luminosity_block, event=event
                )

                if fold_assignments_all is not None:
                    if use_incremental and len(new_event_indices) > 0:
                        fold_assignments = fold_assignments_all[new_event_indices]
                    else:
                        fold_assignments = fold_assignments_all

            # Get predictions using unified predict function
            if use_incremental and len(new_event_indices) > 0:
                # Only compute predictions for new events
                y_pred_new = predict_bdt(
                    features=features,
                    boosters=boosters,
                    feature_names=feature_names,
                    fold_assignments=fold_assignments,
                    is_data=is_data,
                )

                # Merge cached and new predictions
                y_pred = _merge_predictions(
                    current_event_ids=current_event_ids,
                    cached_predictions=cached_preds,
                    cached_event_ids=cached_event_ids,
                    new_predictions=y_pred_new,
                    new_event_indices=new_event_indices,
                )
            else:
                # Compute predictions for all events
                y_pred = predict_bdt(
                    features=features,
                    boosters=boosters,
                    feature_names=feature_names,
                    fold_assignments=fold_assignments,
                    is_data=is_data,
                )

            # Add BDT scores to events
            _add_bdt_scores(
                events_dict[year][sample].events,
                y_pred,
                sample_order=sample_order,
                signal_objectives=signal_objectives,
            )

            # Save predictions if requested
            if save_dir is not None:
                file_dir = get_bdt_preds_dir(
                    base_dir=save_dir,
                    modelname=modelname,
                    signal_objectives=signal_objectives,
                    channel=channel,
                    year=year,
                    n_folds=n_folds,
                    tt_pres=tt_pres,
                    test_mode=test_mode,
                )

                # Extract event identifiers for caching
                event_ids = _extract_event_ids(sample_data)
                _save_cached_predictions(file_dir, sample, y_pred, event_ids)

    elapsed = time.time() - start_time
    if n_folds > 1:
        print(f"BDT predictions computed in {elapsed:.2f} seconds")


def _load_cached_predictions(
    cache_dir: Path,
    sample: str,
) -> tuple[np.ndarray | None, list[tuple[int, int]] | None]:
    """Load cached predictions with event identifiers.

    Args:
        cache_dir: Directory containing cache files
        sample: Sample name

    Returns:
        Tuple of (predictions, event_ids) or (None, None) if not found
    """
    preds_file = cache_dir / f"{sample}_preds.pkl"
    event_ids_file = cache_dir / f"{sample}_event_ids.pkl"

    if not all(f.exists() for f in [preds_file, event_ids_file]):
        return None, None

    try:
        with preds_file.open("rb") as f:
            predictions = pickle.load(f)
        with event_ids_file.open("rb") as f:
            event_ids = pickle.load(f)

        # Validate that predictions and event_ids have matching lengths
        if len(predictions) != len(event_ids):
            print(
                f"Warning: Cache corruption detected for {sample} - predictions and event_ids length mismatch"
            )
            return None, None

        return predictions, event_ids
    except Exception as e:
        print(f"Warning: Failed to load cache for {sample}: {e}")
        return None, None


def _save_cached_predictions(
    cache_dir: Path,
    sample: str,
    predictions: np.ndarray,
    event_ids: list[tuple[int, int]],
) -> None:
    """Save predictions with event identifiers.

    Args:
        cache_dir: Directory to save cache files
        sample: Sample name
        predictions: Predictions array
        event_ids: List of (luminosityBlock, event) tuples
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    with (cache_dir / f"{sample}_preds.pkl").open("wb") as f:
        pickle.dump(predictions, f)
    with (cache_dir / f"{sample}_event_ids.pkl").open("wb") as f:
        pickle.dump(event_ids, f)


def _extract_event_ids(sample_data: LoadedSample) -> list[tuple[int, int]]:
    """Extract (luminosityBlock, event) tuples from a sample.

    Args:
        sample_data: LoadedSample object

    Returns:
        List of (luminosityBlock, event) tuples
    """
    luminosity_block = sample_data.get_var("luminosityBlock")
    event = sample_data.get_var("event")
    return list(zip(luminosity_block, event))


def _merge_predictions(
    current_event_ids: list[tuple[int, int]],
    cached_predictions: np.ndarray,
    cached_event_ids: list[tuple[int, int]],
    new_predictions: np.ndarray,
    new_event_indices: np.ndarray,
) -> np.ndarray:
    """Merge cached and new predictions maintaining order of current events."""
    n_events = len(current_event_ids)
    n_classes = (
        cached_predictions.shape[1] if len(cached_predictions) > 0 else new_predictions.shape[1]
    )

    # Build lookup dictionaries once
    cached_event_to_idx = {event_id: idx for idx, event_id in enumerate(cached_event_ids)}
    # Map position in current_event_ids -> index in new_predictions array
    new_pos_to_pred_idx = {pos: pred_idx for pred_idx, pos in enumerate(new_event_indices)}

    # Single pass: build merged predictions directly
    merged_predictions = np.zeros((n_events, n_classes), dtype=new_predictions.dtype)

    for pos, event_id in enumerate(current_event_ids):
        if event_id in cached_event_to_idx:
            # Use cached prediction
            merged_predictions[pos] = cached_predictions[cached_event_to_idx[event_id]]
        else:
            # Use new prediction (pos must be in new_event_indices)
            merged_predictions[pos] = new_predictions[new_pos_to_pred_idx[pos]]

    return merged_predictions


def check_bdt_prediction_cache(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    n_folds: int = 1,
    test_mode: bool = False,
) -> dict[str, dict[str, dict]]:
    """Check BDT prediction cache and identify which events need computation.

    This function compares current events with cached predictions using (luminosityBlock, event)
    tuples, allowing for incremental updates when only some events are new.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_objectives: List of signal objectives
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing BDT predictions
        n_folds: Number of folds (1 for single model, >1 for k-fold)
        test_mode: Whether in test mode

    Returns:
        Dictionary mapping year -> sample -> cache_info with keys:
        - 'cache_valid': bool - whether cache exists and is valid
        - 'cached_predictions': np.ndarray | None - cached predictions if valid
        - 'cached_event_ids': list | None - cached event IDs if valid
        - 'current_event_ids': list | None - current event IDs
        - 'new_event_indices': np.ndarray | None - indices of events needing computation
    """
    cache_info = {}

    for year in events_dict:
        cache_info[year] = {}
        for sample, sample_data in events_dict[year].items():
            preds_dir = get_bdt_preds_dir(
                base_dir=bdt_preds_dir,
                modelname=modelname,
                signal_objectives=signal_objectives,
                channel=channel,
                year=year,
                n_folds=n_folds,
                tt_pres=tt_pres,
                test_mode=test_mode,
            )

            # Try to load cached predictions
            cached_preds, cached_event_ids = _load_cached_predictions(preds_dir, sample)

            # Extract current event IDs
            current_event_ids = _extract_event_ids(sample_data)

            if current_event_ids is None:
                # Cannot use event-based caching without event identifiers
                cache_info[year][sample] = {
                    "cache_valid": False,
                    "cached_predictions": None,
                    "cached_event_ids": None,
                    "current_event_ids": None,
                    "new_event_indices": None,
                }
                continue

            if cached_preds is None or cached_event_ids is None:
                # No cache exists or cache is corrupted
                cache_info[year][sample] = {
                    "cache_valid": False,
                    "cached_predictions": None,
                    "cached_event_ids": None,
                    "current_event_ids": current_event_ids,
                    "new_event_indices": np.arange(len(current_event_ids)),
                }
                continue

            # Find which events are new (not in cache)
            cached_event_set = set(cached_event_ids)
            new_event_indices = np.array(
                [
                    idx
                    for idx, event_id in enumerate(current_event_ids)
                    if event_id not in cached_event_set
                ]
            )

            cache_info[year][sample] = {
                "cache_valid": True,
                "cached_predictions": cached_preds,
                "cached_event_ids": cached_event_ids,
                "current_event_ids": current_event_ids,
                "new_event_indices": new_event_indices,
            }

    return cache_info


def load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    n_folds: int = 1,
    test_mode: bool = False,
    cache_info: dict[str, dict[str, dict]] | None = None,
):
    """Load BDT predictions from disk and add scores to events.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_objectives: List of signal objectives
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing BDT predictions
        n_folds: Number of folds (1 for single model, >1 for k-fold)
        test_mode: Whether in test mode
        cache_info: Optional cache info from check_bdt_prediction_cache (for smart loading)
    """
    # Get the expected sample order (must match training order)
    sample_order = get_expected_sample_order(signal_objectives)

    for year in events_dict:
        for sample in events_dict[year]:
            preds_dir = get_bdt_preds_dir(
                base_dir=bdt_preds_dir,
                modelname=modelname,
                signal_objectives=signal_objectives,
                channel=channel,
                year=year,
                n_folds=n_folds,
                tt_pres=tt_pres,
                test_mode=test_mode,
            )

            # Try to load from new format first
            cached_preds, cached_event_ids = _load_cached_predictions(preds_dir, sample)

            if cached_preds is not None and cache_info is not None:
                # Use smart cache loading - merge cached and new predictions
                info = cache_info.get(year, {}).get(sample, {})
                if info.get("cache_valid", False):
                    # All events are cached, use cached predictions directly
                    current_event_ids = info["current_event_ids"]
                    if len(current_event_ids) == len(cached_preds):
                        # Simple case: all events match, use cached predictions
                        bdt_preds = cached_preds
                    else:
                        # Need to reorder/merge
                        cached_event_to_idx = {eid: idx for idx, eid in enumerate(cached_event_ids)}
                        bdt_preds = np.array(
                            [
                                cached_preds[cached_event_to_idx[eid]]
                                for eid in current_event_ids
                                if eid in cached_event_to_idx
                            ]
                        )
                else:
                    bdt_preds = cached_preds
            else:
                # Fall back to old format or direct loading
                pred_file = preds_dir / f"{sample}_preds.pkl"
                if pred_file.exists():
                    bdt_preds = pickle.load(pred_file.open("rb"))
                else:
                    raise FileNotFoundError(f"Prediction file not found: {pred_file}")

            _add_bdt_scores(
                events_dict[year][sample].events,
                bdt_preds,
                sample_order=sample_order,
                signal_objectives=signal_objectives,
            )


def compute_or_load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    n_folds: int = 1,
    test_mode: bool = False,
    at_inference: bool = False,
):
    """Wrapper function to compute or load BDT predictions.

    This unified function handles three prediction modes:
    1. Single model (n_folds=1): Standard prediction
    2. K-fold MC samples: Out-of-fold (OOF) predictions
    3. K-fold DATA samples: Average predictions from all k models

    Logic:
    1. If at_inference is True, compute predictions directly
    2. Otherwise, check if saved predictions exist with correct shapes
    3. If they exist and match, load them
    4. If they don't exist or shapes mismatch, compute and save them

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        model_dir: Path to the directory containing the model
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing/to save BDT predictions
        n_folds: Number of folds (1 for single model, >1 for k-fold)
        test_mode: Whether in test mode
        at_inference: If True, always compute predictions (don't try to load)
    """
    if modelname not in BDT_CONFIG:
        raise ValueError(f"Could not find config for modelname {modelname}")

    if "signals" not in BDT_CONFIG[modelname]:
        raise ValueError(f"Model '{modelname}' must specify 'signals' in config")

    signal_objectives = BDT_CONFIG[modelname]["signals"]
    if not isinstance(signal_objectives, list) or len(signal_objectives) not in [1, 2]:
        raise ValueError(
            f"Model '{modelname}' has invalid 'signals' in config: {signal_objectives}. "
            f"Must be a list with 1 or 2 elements."
        )

    if n_folds > 1:
        print(f"Using {n_folds}-fold k-fold BDT inference.")

    if at_inference:
        # Compute predictions directly at inference time
        print(f"Computing BDT predictions for signals '{signal_objectives}'...")
        compute_bdt_preds(
            events_dict=events_dict,
            modelname=modelname,
            model_dir=model_dir,
            signal_objectives=signal_objectives,
            channel=channel,
            n_folds=n_folds,
            tt_pres=tt_pres,
            test_mode=test_mode,
            save_dir=bdt_preds_dir,
        )
    else:
        # Use smart cache checking with event-based identification
        print(f"Checking BDT prediction cache for signals '{signal_objectives}'...")
        cache_info = check_bdt_prediction_cache(
            events_dict=events_dict,
            modelname=modelname,
            signal_objectives=signal_objectives,
            channel=channel,
            n_folds=n_folds,
            tt_pres=tt_pres,
            bdt_preds_dir=bdt_preds_dir,
            test_mode=test_mode,
        )

        compute_bdt_preds(
            events_dict=events_dict,
            modelname=modelname,
            model_dir=model_dir,
            signal_objectives=signal_objectives,
            channel=channel,
            n_folds=n_folds,
            tt_pres=tt_pres,
            test_mode=test_mode,
            save_dir=bdt_preds_dir,
            cache_info=cache_info,
        )


def get_sample_fold_assignments(
    model_dir: Path,
    modelname: str,
    luminosity_block: np.ndarray | None = None,
    event: np.ndarray | None = None,
) -> np.ndarray | None:
    """Get fold assignments for a specific sample from the k-fold training metadata.

    For each event in the sample, returns which fold's validation set it was in during training.
    This is used for out-of-fold (OOF) predictions on MC samples.

    Uses (luminosityBlock, event) tuples to uniquely identify events, allowing
    proper handling of events that may not have been in training.

    Args:
        model_dir: Path to the directory containing the model
        modelname: Name of the BDT model
        sample_name: Name of the sample (e.g., 'ggfbbtthe', 'qcd')
        luminosity_block: Array of luminosityBlock values for each event (required)
        event: Array of event numbers for each event (required)

    Returns:
        Array of fold indices (one per event), or None if not a k-fold model.
        Events not in training will have fold index -1.
    """
    fold_indices_path = model_dir / modelname / "fold_indices.json"
    if not fold_indices_path.exists():
        return None

    with fold_indices_path.open("r") as f:
        fold_info = json.load(f)

    # Primary method: lookup by (luminosityBlock, event) tuple
    event_to_fold = fold_info.get("event_to_fold", {})

    if not event_to_fold:
        # Old model format without event_to_fold - cannot use event-based lookup
        print(
            f"Warning: Model '{modelname}' does not have event_to_fold mapping. "
            f"Cannot use event-based fold assignment. Using averaged predictions."
        )
        return None

    n_events = len(luminosity_block)
    fold_assignments = np.full(n_events, -1, dtype=int)  # -1 = not in training

    for idx in range(n_events):
        event_key = f"({luminosity_block[idx]},{event[idx]})"
        if event_key in event_to_fold:
            # Event was in training validation set - use OOF prediction
            fold_assignments[idx] = int(event_to_fold[event_key])
        # else: event not in training, keep -1 for averaged predictions

    return fold_assignments


def _predict_with_best_iteration(bst: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    """Predict with early-stopped best iteration when available."""
    # Preferred path for models trained with early stopping in xgb.train.
    best_iteration = getattr(bst, "best_iteration", None)
    if best_iteration is not None:
        return bst.predict(dmatrix, iteration_range=(0, best_iteration + 1))

    # No early-stopping metadata available: use full model.
    return bst.predict(dmatrix)


def predict_bdt(
    features: np.ndarray,
    boosters: list[xgb.Booster],
    feature_names: list[str],
    fold_assignments: np.ndarray | None = None,
    is_data: bool = False,
) -> np.ndarray:
    """Unified BDT prediction function handling single-fold, OOF, and averaged predictions.

    This function handles three prediction modes:
    1. Single model (len(boosters)==1): Standard single prediction
    2. K-fold with fold assignments (MC in training): Out-of-fold predictions
       - Each event is predicted by the model that was NOT trained on it
       - Events not in training (fold_assignment == -1) use averaged predictions
    3. K-fold without fold assignments (DATA or new MC): Average all k models

    Args:
        features: Feature array with shape (n_events, n_features)
        boosters: List of trained XGBoost Booster objects
        feature_names: List of feature names matching the training
        fold_assignments: Array of fold indices for each event (for OOF predictions).
            If provided and sample is not DATA, uses OOF mode.
            Values should be 0 to n_folds-1 indicating which fold's validation
            set each event was in, or -1 for events not in training (which will
            use averaged predictions).
        is_data: If True, always use averaged predictions (DATA is never in training)

    Returns:
        Predictions array with shape (n_events, n_classes)
    """
    n_events = features.shape[0]
    n_folds = len(boosters)

    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)

    if n_folds == 1:
        # Single model: standard prediction
        return _predict_with_best_iteration(boosters[0], dmatrix)

    # K-fold model
    if is_data or fold_assignments is None:
        # DATA or MC not in training: average all models
        all_preds = np.array([_predict_with_best_iteration(bst, dmatrix) for bst in boosters])
        return np.mean(all_preds, axis=0)
    else:
        # MC: use out-of-fold predictions for events in training, averaged for others
        # Validate fold assignments
        if len(fold_assignments) != n_events:
            raise ValueError(
                f"fold_assignments length ({len(fold_assignments)}) doesn't match "
                f"number of events ({n_events})"
            )

        # Get predictions from all models
        all_preds = [_predict_with_best_iteration(bst, dmatrix) for bst in boosters]
        n_classes = all_preds[0].shape[1]
        avg_preds = np.mean(all_preds, axis=0)

        # Assign each event's prediction from its OOF model (if in training)
        # or averaged prediction (if not in training)
        oof_preds = np.zeros((n_events, n_classes))

        # Events in training: use OOF predictions
        for fold_idx in range(n_folds):
            mask = fold_assignments == fold_idx
            if np.any(mask):
                oof_preds[mask] = all_preds[fold_idx][mask]

        # Events not in training (fold_assignment == -1): use averaged predictions
        not_in_training_mask = fold_assignments < 0
        if np.any(not_in_training_mask):
            oof_preds[not_in_training_mask] = avg_preds[not_in_training_mask]

        # Handle any events with invalid fold assignment (>= n_folds)
        invalid_mask = fold_assignments >= n_folds
        if np.any(invalid_mask):
            print(
                f"Warning: {np.sum(invalid_mask)} events have invalid fold assignment (>= {n_folds})"
            )
            # Fall back to averaging for invalid assignments
            oof_preds[invalid_mask] = avg_preds[invalid_mask]

        return oof_preds
