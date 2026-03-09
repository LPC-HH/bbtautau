"""
BDT training and inference script for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import ClassVar

import matplotlib as mpl
import numpy as np
import pandas as pd
import xgboost as xgb
from boostedhh import hh_vars, plotting
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabulate import tabulate

from bbtautau.postprocessing.bbtautau_types import LoadedSample
from bbtautau.postprocessing.bdt_config import BDT_CONFIG
from bbtautau.postprocessing.bdt_utils import (
    DEFAULT_BKG_ORDER,
    bin_features,
    get_expected_sample_order,
    predict_bdt,
    save_training_wps,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer, multiclass_confusion_matrix
from bbtautau.postprocessing.Samples import (
    CHANNELS,
    SAMPLES,
    sig_keys_ggf,
    sig_keys_vbf,
)
from bbtautau.postprocessing.utils import (
    _ensure_dir,
    base_filter,
    bbtautau_assignment,
    delete_columns,
    derive_lepton_variables,
    derive_variables,
    derive_vbf_variables,
    get_columns,
    label_transform,
    leptons_assignment,
    load_samples,
    tt_filters,
)
from bbtautau.userConfig import DATA_PATHS, MODEL_DIR, WPS_TTPART, path_dict

# Use non-interactive backend for containerized/CLI environments
mpl.use("Agg")
plt = mpl.pyplot

# Some global variables
DATA_DIR = Path(
    "/ceph/cms/store/user/lumori/bbtautau"
)  # default directory for saving BDT predictions


class Trainer:

    loaded_dmatrix = False

    # Background samples for training (defined once in bdt_utils.py)
    bkg_sample_names: ClassVar[list[str]] = DEFAULT_BKG_ORDER

    def __init__(
        self,
        years: list[str],
        modelname: str = None,
        data_path: str = None,
        output_dir: str = None,
        tt_preselection: bool = False,
    ) -> None:
        if years[0] == "all":
            print("Using all years")
            years = hh_vars.years
        else:
            years = list(years)
        self.years = years

        # ensure backwards compatibility. Choice of default data paths is done in userConfig.py
        self.data_paths = path_dict(data_path) if data_path is not None else DATA_PATHS

        self.modelname = modelname
        self.bdt_config = BDT_CONFIG
        self.tt_preselection = tt_preselection

        # Set output_dir early (needed for resolving n_folds from saved metadata)
        if output_dir is not None:
            output_dir_path = Path(output_dir)
            # If absolute, use as-is; otherwise resolve under classifier/ (MODEL_DIR parent)
            self.output_dir = (
                output_dir_path if output_dir_path.is_absolute() else MODEL_DIR.parent / output_dir
            )
        else:
            self.output_dir = MODEL_DIR / self.modelname
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # n_folds from config (default: 1)
        self.n_folds = self.bdt_config[self.modelname].get("n_folds", 1)
        self.use_kfold = self.n_folds > 1
        print(f"Using n_folds={self.n_folds}")

        # Read signals from config (must be specified in config)
        if "signals" not in self.bdt_config[self.modelname]:
            raise ValueError(
                f"Model '{self.modelname}' must specify 'signals' in config. "
                f"Expected: signals: ['signal1'] or signals: ['signal1', 'signal2']"
            )

        config_signals = self.bdt_config[self.modelname]["signals"]
        if not isinstance(config_signals, list) or len(config_signals) not in [1, 2]:
            raise ValueError(
                f"Model '{self.modelname}' has invalid 'signals' in config: {config_signals}. "
                f"Must be a list with 1 or 2 elements."
            )

        # Use signals from config
        self.signal_keys = list(config_signals)

        # Infer model_type from number of signals
        if len(self.signal_keys) == 1:
            self.model_type = "single_signal"
        elif len(self.signal_keys) == 2:
            self.model_type = "unified"
        else:
            raise ValueError(
                f"signals must contain 1 or 2 signals, got {len(self.signal_keys)}: {self.signal_keys}"
            )

        # Build samples dict: signals + backgrounds
        self.samples = {name: SAMPLES[name] for name in self.signal_keys + self.bkg_sample_names}

        self.train_vars = self.bdt_config[self.modelname]["train_vars"]
        self.hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.feats = [feat for cat in self.train_vars for feat in self.train_vars[cat]]

        # Validate num_class matches number of signals and backgrounds
        # Each signal has 3 channels (he, hh, hm), backgrounds are fixed
        num_signal_classes = len(self.signal_keys) * 3
        num_bkg_classes = len(self.bkg_sample_names)
        expected_num_class = num_signal_classes + num_bkg_classes
        if self.hyperpars.get("num_class") != expected_num_class:
            raise ValueError(
                f"Model '{self.modelname}' has num_class={self.hyperpars.get('num_class')} "
                f"but {len(self.signal_keys)} signal(s) * 3 channels + {num_bkg_classes} backgrounds "
                f"requires num_class={expected_num_class}. signals={self.signal_keys}, "
                f"bkg_samples={self.bkg_sample_names}"
            )

        self.cap_weights = True  # set to be always true, leave the logic here for record
        self.events_dict = {year: {} for year in self.years}

    def load_data(self, force_reload=False):
        # Check if data buffer file exists
        if (
            self.output_dir / "dtrain.buffer" in self.output_dir.glob("*.buffer")
            and not force_reload
        ):
            print("Loading data from buffer file")
            self.dtrain = xgb.DMatrix(self.output_dir / "dtrain.buffer")
            self.dval = xgb.DMatrix(self.output_dir / "dval.buffer")

            print(self.output_dir)
            self.dtrain_rescaled = xgb.DMatrix(self.output_dir / "dtrain_rescaled.buffer")
            self.dval_rescaled = xgb.DMatrix(self.output_dir / "dval_rescaled.buffer")

            self.loaded_dmatrix = True
        else:
            for year in self.years:

                filters_dict = base_filter(test_mode=False)
                # filters_dict = bb_filters(filters_dict, num_fatjets=3, bb_cut=0.3) # not needed, events are already filtered by skimmer
                if self.tt_preselection:
                    filters_dict = tt_filters(
                        channel=None, in_filters=filters_dict, num_fatjets=3, tt_cut=0.3
                    )

                columns = get_columns(year)

                self.events_dict[year] = load_samples(
                    year=year,
                    paths=self.data_paths[year],
                    signals=self.signal_keys,
                    channels=list(CHANNELS.values()),
                    samples=self.samples,
                    filters_dict=filters_dict,
                    load_columns=columns,
                    restrict_data_to_channel=False,
                    load_bgs=True,
                    loaded_samples=True,
                    multithread=True,
                )

                # Need to check this!
                # apply_triggers(self.events_dict[year], year, channel=None)

                self.events_dict[year] = delete_columns(
                    self.events_dict[year], year, channels=list(CHANNELS.values())
                )

                derive_variables(self.events_dict[year])
                bbtautau_assignment(self.events_dict[year])
                leptons_assignment(self.events_dict[year], dR_cut=1.5)
                derive_lepton_variables(self.events_dict[year])
                derive_vbf_variables(self.events_dict[year])

        # Build sample_names using the order defined in bdt_utils. Here order counts!
        self.sample_names = get_expected_sample_order(self.signal_keys)

        # Update samples dict to include all signal channels and remove base signal keys; these do not need to be ordered
        for signal_key in self.signal_keys:
            for ch in CHANNELS:
                channel_key = f"{signal_key}{ch}"
                self.samples[channel_key] = SAMPLES[channel_key]
            # Remove base signal key from samples dict
            if signal_key in self.samples:
                del self.samples[signal_key]

    @staticmethod
    def save_stats(stats, filename):
        """Save weight statistics to a CSV file"""
        with Path.open(filename, "w") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)

    @staticmethod
    def _get_sample_group(sample_name):
        """Determine which physics group a sample belongs to: 'ggf', 'vbf', or 'bkg'."""
        for key in sig_keys_ggf:
            if sample_name.startswith(key):
                return "ggf"
        for key in sig_keys_vbf:
            if sample_name.startswith(key):
                return "vbf"
        return "bkg"

    def _process_samples_for_training(self, balance="equal_groups_equal_channels"):
        """Process samples and compute weights for training.

        This is the common data processing logic used by prepare_training_set().
        Extracts features and computes rescaled weights for all samples.

        Args:
            balance: Strategy for balancing sample weights. Options:
                - 'bysample': each sample gets equal total weight
                - 'equal_groups': equal total weight per physics group (ggF, VBF, bkg),
                  distributed proportionally to n_events within each group
                - 'equal_groups_equal_channels': like equal_groups, but within signal
                  groups each channel gets equal total weight; within backgrounds
                  weight is distributed proportionally to n_events

        Returns:
            tuple: (X, y, weights, weights_rescaled) - processed data arrays
        """
        if balance not in [
            "bysample",
            "equal_groups",
            "equal_groups_equal_channels",
        ]:
            raise ValueError(f"Invalid balance rule: {balance}")

        # Initialize lists for features, labels, weights, and masks
        X_list = []
        weights_list = []
        weights_rescaled_list = []
        sample_names_labels = []
        event_ids_list = []  # List of (luminosityBlock, event) tuples
        masks_list = {"tt_mask": [], "bb_mask": [], "m_mask": [], "e_mask": []}
        weight_stats_by_stage_sample = {}

        # Process each sample
        for year in self.years:
            total_weight_rescaled = 0
            N_events = 0

            len_signal = sum(
                len(self.events_dict[year][sig_sample].events)
                for sig_sample in self.samples
                if self.samples[sig_sample].isSignal
            )

            len_signal_per_channel = len_signal / len(
                [sample for sample in self.samples if self.samples[sample].isSignal]
            )

            # Pre-compute per-group statistics for group-based balancing
            group_stats = {
                "ggf": {"n_events": 0, "n_channels": 0},
                "vbf": {"n_events": 0, "n_channels": 0},
                "bkg": {"n_events": 0, "n_channels": 0},
            }
            for sname in self.events_dict[year]:
                grp = self._get_sample_group(sname)
                group_stats[grp]["n_events"] += len(self.events_dict[year][sname].events)
                group_stats[grp]["n_channels"] += 1

            year_start_idx = len(weights_rescaled_list)

            for sample_name, sample in self.events_dict[year].items():
                X_sample = pd.DataFrame({feat: sample.get_var(feat) for feat in self.feats})
                weights = np.abs(sample.get_var("finalWeight"))
                weights_rescaled = weights

                key = ("Initial", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled)

                # Apply balance rescaling directly on abs(finalWeight)
                weights_rescaled = self._apply_balance_rescaling(
                    weights_rescaled=weights_rescaled,
                    balance=balance,
                    sample_name=sample_name,
                    len_signal=len_signal,
                    len_signal_per_channel=len_signal_per_channel,
                    group_stats=group_stats,
                )

                key = ("Balance rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled)

                if self.cap_weights:
                    median_w = np.median(weights_rescaled)
                    cap = 10.0 * max(median_w, 1e-12)
                    n_capped = np.sum(weights_rescaled > cap)
                    if n_capped > 0:
                        weight_before = np.sum(weights_rescaled)
                        weights_rescaled = np.minimum(weights_rescaled, cap)
                        print(
                            f"  {sample.sample.label}: capped {n_capped}/{len(weights_rescaled)} "
                            f"events at {cap:.4g} (total weight {weight_before:.1f} -> "
                            f"{np.sum(weights_rescaled):.1f})"
                        )

                    key = ("Weight capping", sample.sample.label)
                    if key not in weight_stats_by_stage_sample:
                        weight_stats_by_stage_sample[key] = []
                    weight_stats_by_stage_sample[key].append(weights_rescaled)

                X_list.append(X_sample)
                weights_list.append(weights)
                weights_rescaled_list.append(weights_rescaled)
                sample_names_labels.extend([sample_name] * len(sample.events))

                # Extract event identifiers (luminosityBlock, event) tuples
                luminosity_block = sample.get_var("luminosityBlock")
                event = sample.get_var("event")
                event_ids = list(zip(luminosity_block, event))
                event_ids_list.extend(event_ids)

                # Collect masks
                for mask_name in masks_list:
                    mask = getattr(sample, mask_name, None)
                    if mask is not None:
                        masks_list[mask_name].append(mask)

                total_weight_rescaled += np.sum(weights_rescaled)
                N_events += len(weights_rescaled)

            # Normalize so the average weight across all samples in this year is 1
            global_rescale_factor = total_weight_rescaled / N_events
            print(f"\nYear {year}: global rescale factor = {global_rescale_factor:.4g}")
            for i, sample in enumerate(self.events_dict[year].values()):
                idx = year_start_idx + i
                weights_rescaled_list[idx] = weights_rescaled_list[idx] / global_rescale_factor
                key = ("Global rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled_list[idx])

        # Save weight statistics
        weight_stats = []
        for (stage, sample_label), weights_for_sample in weight_stats_by_stage_sample.items():
            all_weights = np.concatenate(weights_for_sample)
            weight_stats.append(
                {
                    "stage": stage,
                    "sample": sample_label,
                    "n_events": len(all_weights),
                    "total_weight": np.sum(all_weights),
                    "average_weight": np.mean(all_weights),
                    "std_weight": np.std(all_weights),
                }
            )
        self.save_stats(weight_stats, self.output_dir / "weight_stats.csv")

        # Combine all samples
        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        weights = np.concatenate(weights_list)
        weights_rescaled = np.concatenate(weights_rescaled_list)
        y = np.array(label_transform(self.sample_names, sample_names_labels))

        # Concatenate masks (only if all samples had them)
        masks = {}
        for mask_name, mask_arrays in masks_list.items():
            if mask_arrays and len(mask_arrays) == len(X_list):
                masks[mask_name] = np.concatenate(mask_arrays, axis=0)
            else:
                masks[mask_name] = None

        # Apply feature binning if configured
        bin_feats = self.bdt_config[self.modelname].get("bin_features", None)
        self._wps_used = {}
        self._unbinned_glopart = None
        if bin_feats:
            # Resolve WPS: config 'wps_ttpart' overrides runtime userConfig.WPS_TTPART
            config_wps = self.bdt_config[self.modelname].get("wps_ttpart", None)
            if config_wps is not None:
                wps = {k: np.array(v) for k, v in config_wps.items()}
                print("\nUsing WPS from model config for feature binning")
            else:
                wps = dict(WPS_TTPART)
                print("\nUsing runtime WPS_TTPART from userConfig for feature binning")

            bin_edges = {f: wps[f] for f in bin_feats if f in wps}

            if bin_edges:
                print(f"  Binning features: {list(bin_edges.keys())}")
                cols_to_save = [f for f in bin_edges if f in X.columns]
                if cols_to_save:
                    self._unbinned_glopart = X[cols_to_save].copy()
                skipped = [f for f in bin_feats if f not in bin_edges]
                if skipped:
                    print(f"  Skipping {len(skipped)} features (no bin edges found): {skipped}")
            else:
                print(f"  No bin edges found for features: {bin_feats}")

            X = bin_features(X, bin_feats, bin_edges_dict=bin_edges, inplace=True)
            self._wps_used = bin_edges

        # Print class mapping
        print("\nClass mapping:")
        for i, class_name in enumerate(self.sample_names):
            print(f"Class {i}: {class_name}")

        return X, y, weights, weights_rescaled, masks, event_ids_list

    def _apply_balance_rescaling(
        self,
        weights_rescaled,
        balance,
        sample_name,
        len_signal,
        len_signal_per_channel,
        group_stats,
    ):
        """Apply balance rescaling to weights based on the chosen strategy.

        Args:
            weights_rescaled: Current rescaled weights (abs finalWeight)
            balance: Balance strategy name
            sample_name: Name of the sample
            len_signal: Total signal events across all signal channels
            len_signal_per_channel: Average signal events per channel
            group_stats: Dict with per-group n_events and n_channels

        Returns:
            Rescaled weights array
        """
        n_events = len(weights_rescaled)
        current_sum = np.sum(weights_rescaled)

        if balance == "bysample":
            target = len_signal_per_channel

        elif balance in ("equal_groups", "equal_groups_equal_channels"):
            group = self._get_sample_group(sample_name)
            group_budget = len_signal / 3.0

            if balance == "equal_groups":
                target = group_budget * (n_events / group_stats[group]["n_events"])
            else:  # equal_groups_equal_channels
                if group in ("ggf", "vbf"):
                    target = group_budget / group_stats[group]["n_channels"]
                else:
                    target = group_budget * (n_events / group_stats[group]["n_events"])

        scaling_factor = target / current_sum
        return weights_rescaled * scaling_factor

    def prepare_training_set(
        self,
        save_buffer=False,
        balance="equal_groups_equal_channels",
        prediction_only=False,
    ):
        """Prepare features and labels for training with k-fold cross-validation.

        This unified method handles both single-split (n_folds=1) and k-fold (n_folds>1) cases.
        The data is stored in self.fold_data.

        For n_folds=1: Uses train_test_split for a single train/validation split.
        For n_folds>1: Uses StratifiedKFold to create k stratified folds.

        Args:
            save_buffer: Whether to save DMatrix buffer files for quicker loading
            balance: Strategy for balancing sample weights
            prediction_only: If True, only create validation DMatrices (dval) and skip
                training DMatrices to save memory. Used by compare_models.
        """
        if self.loaded_dmatrix:
            return

        # Process samples (common logic)
        X, y, weights, weights_rescaled, masks, event_ids_list = self._process_samples_for_training(
            balance=balance
        )

        print(f"\nPreparing training sets with n_folds={self.n_folds}...")
        print(f"Total samples: {len(y)}")

        # Initialize unified fold data structure
        self.fold_data = {
            "X": X,
            "y": y,
            "weights": weights,
            "weights_rescaled": weights_rescaled,
            "masks": masks,  # Dict of mask arrays (tt_mask, bb_mask, m_mask, e_mask)
            "X_unbinned_glopart": self._unbinned_glopart,
            "fold_indices": [],  # List of (train_idx, val_idx) tuples
            "dtrain_rescaled": [],  # DMatrix for training (rescaled weights)
            "dval_rescaled": [],  # DMatrix for validation (rescaled weights)
            "dtrain": [],  # DMatrix with original weights (for ROC computation)
            "dval": [],  # DMatrix with original weights (for ROC computation)
        }

        # Create fold splits
        if self.n_folds == 1:
            # Single train/test split (original behavior)
            indices = np.arange(len(y))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.bdt_config[self.modelname]["test_size"],
                random_state=self.bdt_config[self.modelname]["random_seed"],
                stratify=y,
            )
            fold_splits = [(train_idx, val_idx)]
            print(f"Single split: {len(train_idx)} train, {len(val_idx)} val")
        else:
            # K-fold stratified splits
            skf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.bdt_config[self.modelname]["random_seed"],
            )
            fold_splits = list(skf.split(X, y))
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                print(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")

        # Create DMatrix objects for each fold
        for train_idx, val_idx in fold_splits:
            self.fold_data["fold_indices"].append((train_idx, val_idx))

            if not prediction_only:
                # DMatrix with rescaled weights (for training)
                dtrain_rescaled = xgb.DMatrix(
                    X.iloc[train_idx],
                    label=y[train_idx],
                    weight=weights_rescaled[train_idx],
                    nthread=8,
                )
                dval_rescaled = xgb.DMatrix(
                    X.iloc[val_idx], label=y[val_idx], weight=weights_rescaled[val_idx], nthread=8
                )
                self.fold_data["dtrain_rescaled"].append(dtrain_rescaled)
                self.fold_data["dval_rescaled"].append(dval_rescaled)

                # DMatrix with original weights (for training-side ROC)
                dtrain = xgb.DMatrix(
                    X.iloc[train_idx], label=y[train_idx], weight=weights[train_idx], nthread=8
                )
                self.fold_data["dtrain"].append(dtrain)

            # Validation DMatrix (always needed)
            dval = xgb.DMatrix(
                X.iloc[val_idx], label=y[val_idx], weight=weights[val_idx], nthread=8
            )
            self.fold_data["dval"].append(dval)

        # For backward compatibility, also set single-fold attributes
        if self.n_folds == 1:
            if not prediction_only:
                self.dtrain_rescaled = self.fold_data["dtrain_rescaled"][0]
                self.dval_rescaled = self.fold_data["dval_rescaled"][0]
                self.dtrain = self.fold_data["dtrain"][0]
            self.dval = self.fold_data["dval"][0]

        if prediction_only:
            # Free arrays that DMatrix has already internalized
            self.fold_data["X"] = None
            self.fold_data["weights_rescaled"] = None
            self.fold_data["masks"] = None
            self.fold_data["X_unbinned_glopart"] = None
            del X, weights_rescaled, masks
            return

        # Save fold indices (useful for k-fold inference on MC)
        if self.n_folds > 1:
            fold_indices_path = self.output_dir / "fold_indices.npz"

            # Create event-to-fold mapping using (luminosityBlock, event) tuples
            event_fold_assignment = np.full(len(y), -1, dtype=int)
            for fold_idx, (_, val_idx) in enumerate(self.fold_data["fold_indices"]):
                event_fold_assignment[val_idx] = fold_idx

            print("Cross check: ", np.sum(event_fold_assignment == -1)), "has to be zero "

            # Persist compact numeric arrays instead of a large JSON dict with string keys.
            lumi_values = np.asarray([lumi for lumi, _ in event_ids_list], dtype=np.int32)
            event_values = np.asarray([event for _, event in event_ids_list], dtype=np.int64)
            np.savez_compressed(
                fold_indices_path,
                n_folds=np.array([self.n_folds], dtype=np.int16),
                random_seed=np.array(
                    [self.bdt_config[self.modelname]["random_seed"]], dtype=np.int32
                ),
                luminosityBlock=lumi_values,
                event=event_values,
                fold=event_fold_assignment.astype(np.int16),
            )
            print(f"Fold indices saved to {fold_indices_path}")

        # Save WPS bin edges used for feature binning (if any)
        if self._wps_used:
            save_training_wps(self.output_dir, self._wps_used)

        # Save buffer for quicker loading (single-fold only for backward compat)
        if save_buffer and self.n_folds == 1:
            self.dtrain.save_binary(self.output_dir / "dtrain.buffer")
            self.dval.save_binary(self.output_dir / "dval.buffer")
            self.dtrain_rescaled.save_binary(self.output_dir / "dtrain_rescaled.buffer")
            self.dval_rescaled.save_binary(self.output_dir / "dval_rescaled.buffer")

    def get_oof_predictions(self):
        """Get out-of-fold predictions for training data evaluation.

        For each event, this returns the prediction from the model that was
        trained WITHOUT that event (i.e., the event was in the validation fold).

        This is the key for unbiased MC evaluation - each event is scored by
        a BDT that never saw it during training.

        For n_folds=1: Returns validation set predictions (single fold).
        For n_folds>1: Returns combined out-of-fold predictions from all folds.

        Note: This method is used during training when fold_data is in memory.
        For inference on new data, use predict_bdt() from bdt_utils instead.

        Returns:
            tuple: (oof_predictions, oof_weights, oof_labels) arrays
        """
        if not hasattr(self, "fold_data") or not hasattr(self, "boosters"):
            raise ValueError("Fold data and models not available. Train first.")

        # Build fold assignments array from fold_data
        n_events = len(self.fold_data["y"])
        fold_assignments = np.full(n_events, -1, dtype=int)
        for fold_idx, (_, val_idx) in enumerate(self.fold_data["fold_indices"]):
            fold_assignments[val_idx] = fold_idx

        # Use unified predict_bdt for predictions
        oof_predictions = predict_bdt(
            features=self.fold_data["X"].values,
            boosters=self.boosters,
            feature_names=list(self.fold_data["X"].columns),
            fold_assignments=fold_assignments,
            is_data=False,
        )

        # Gather weights from all folds
        oof_weights = self.fold_data["weights"].copy()

        return oof_predictions, oof_weights, self.fold_data["y"]

    def train_model(self, save=True, early_stopping_rounds=5):
        """Train model(s) using configured hyperparameters and evaluation sets.

        This unified method handles both n_folds=1 (single model) and n_folds>1 (k models).

        For n_folds=1: Trains one model, saves as {modelname}.json
        For n_folds>1: Trains k models, saves as {modelname}_fold{i}.json

        Args:
            save: Whether to save the trained models
            early_stopping_rounds: Early stopping rounds for XGBoost

        Returns:
            For n_folds=1: Single booster (also stored as self.bst)
            For n_folds>1: List of boosters
        """
        if not hasattr(self, "fold_data"):
            raise ValueError("Fold data not prepared. Call prepare_training_set first.")

        self.boosters = []
        self.evals_results = []

        for fold_idx in range(self.n_folds):
            if self.n_folds > 1:
                print(f"\n{'='*60}")
                print(f"Training fold {fold_idx + 1}/{self.n_folds}")
                print(f"{'='*60}")

            evals_result = {}
            evallist = [
                (self.fold_data["dtrain_rescaled"][fold_idx], "train"),
                (self.fold_data["dval_rescaled"][fold_idx], "eval"),
            ]

            bst = xgb.train(
                self.hyperpars,
                self.fold_data["dtrain_rescaled"][fold_idx],
                self.bdt_config[self.modelname]["num_rounds"],
                evals=evallist,
                evals_result=evals_result,
                early_stopping_rounds=early_stopping_rounds,
            )

            self.boosters.append(bst)
            self.evals_results.append(evals_result)

            if save:
                if self.n_folds == 1:
                    model_path = self.output_dir / f"{self.modelname}.ubj"
                else:
                    model_path = self.output_dir / f"{self.modelname}_fold{fold_idx}.ubj"
                bst.save_model(model_path)
                print(f"Model saved to {model_path}")

        # Save evaluation results as JSON
        evals_filename = "evals_result.json"
        evals_data = self.evals_results[0] if self.n_folds == 1 else self.evals_results
        with (self.output_dir / evals_filename).open("w") as f:
            json.dump(evals_data, f, indent=2)

        # For backward compatibility with n_folds=1
        if self.n_folds == 1:
            self.bst = self.boosters[0]
            return self.bst

        return self.boosters

    def load_model(self):
        """Load trained model(s) from disk.

        This unified method handles both n_folds=1 (single model) and n_folds>1 (k models).

        Returns:
            For n_folds=1: Single booster (also stored as self.bst)
            For n_folds>1: List of boosters
        """
        self.boosters = []

        for fold_idx in range(self.n_folds):
            if self.n_folds == 1:
                model_path = self.output_dir / f"{self.modelname}.json"
            else:
                model_path = self.output_dir / f"{self.modelname}_fold{fold_idx}.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            bst = xgb.Booster()
            bst.load_model(model_path)
            self.boosters.append(bst)
            print(f"Loaded model from {model_path}")

        # For backward compatibility with n_folds=1
        if self.n_folds == 1:
            self.bst = self.boosters[0]
            return self.bst

        return self.boosters

    def evaluate_training(self, savedir=None):
        """Plot training curves and feature importances from saved eval results.

        This unified method handles both n_folds=1 and n_folds>1 cases.
        For n_folds>1, plots all folds overlaid and averages feature importance.
        """
        savedir = self.output_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)

        # Load evaluation results - try both naming conventions
        if not hasattr(self, "evals_results"):
            evals_filename = "evals_result.json" if self.n_folds == 1 else "evals_results.json"
            evals_path = self.output_dir / evals_filename
            if evals_path.exists():
                with evals_path.open("r") as f:
                    loaded_data = json.load(f)
                # Normalize to list format
                self.evals_results = [loaded_data] if self.n_folds == 1 else loaded_data
            else:
                raise ValueError(f"Evaluation results not found at {evals_path}")

        # Plot training curves
        if self.n_folds == 1:
            # Single fold: simple plot
            plt.figure(figsize=(10, 8))
            evals_result = self.evals_results[0]
            plt.plot(evals_result["train"][self.hyperpars["eval_metric"]], label="Train")
            plt.plot(evals_result["eval"][self.hyperpars["eval_metric"]], label="Validation")
            plt.xlabel("Iteration")
            plt.ylabel(self.hyperpars["eval_metric"])
            plt.title("Training History")
            plt.legend()
            plt.tight_layout()
            plt.savefig(savedir / "training_history.pdf")
            plt.savefig(savedir / "training_history.png")
            plt.close()
        else:
            # Multi-fold: overlay all folds
            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_folds))

            for fold_idx, evals_result in enumerate(self.evals_results):
                plt.plot(
                    evals_result["train"][self.hyperpars["eval_metric"]],
                    label=f"Fold {fold_idx} Train",
                    color=colors[fold_idx],
                    linestyle="-",
                )
                plt.plot(
                    evals_result["eval"][self.hyperpars["eval_metric"]],
                    label=f"Fold {fold_idx} Val",
                    color=colors[fold_idx],
                    linestyle="--",
                )

            plt.xlabel("Iteration")
            plt.ylabel(self.hyperpars["eval_metric"])
            plt.title(f"{self.n_folds}-Fold Cross-Validation Training History")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(savedir / "training_history.pdf")
            plt.savefig(savedir / "training_history.png")
            plt.close()

        # Plot feature importance
        importance_types = ["weight", "gain", "total_gain"]

        try:
            if self.n_folds == 1:
                # Single fold: use xgb built-in plotting
                titles = [
                    "Feature Importance (Weight)",
                    "Feature Importance (Gain)",
                    "Feature Importance (Total Gain)",
                ]
                for imp_type, title in zip(importance_types, titles):
                    plt.figure(figsize=(14, 12))
                    ax = plt.gca()
                    xgb.plot_importance(
                        self.boosters[0], importance_type=imp_type, ax=ax, values_format="{v:.2f}"
                    )
                    ax.set_title(title)
                    plt.tight_layout()
                    plt.savefig(savedir / f"feature_importance_{imp_type}.pdf")
                    plt.savefig(savedir / f"feature_importance_{imp_type}.png")
                    plt.close()
            else:
                # Multi-fold: average importance across folds
                for imp_type in importance_types:
                    all_importances = {}
                    for bst in self.boosters:
                        imp = bst.get_score(importance_type=imp_type)
                        for feat, score in imp.items():
                            if feat not in all_importances:
                                all_importances[feat] = []
                            all_importances[feat].append(score)

                    # Compute mean importance
                    mean_importance = {
                        feat: np.mean(scores) for feat, scores in all_importances.items()
                    }

                    # Sort and plot
                    sorted_feats = sorted(mean_importance.items(), key=lambda x: x[1])
                    if sorted_feats:
                        feats, scores = zip(*sorted_feats)
                        plt.figure(figsize=(12, max(10, len(feats) * 0.3)))
                        plt.barh(range(len(feats)), scores)
                        plt.yticks(range(len(feats)), feats)
                        plt.xlabel(f"Mean {imp_type} across {self.n_folds} folds")
                        plt.title(f"Feature Importance ({imp_type}) - {self.n_folds}-Fold Average")
                        plt.tight_layout()
                        plt.savefig(savedir / f"feature_importance_{imp_type}.pdf")
                        plt.savefig(savedir / f"feature_importance_{imp_type}.png")
                        plt.close()

        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def complete_train(self, training_info=True, force_reload=False, **kwargs):
        """End-to-end training workflow including ROCs and optional plots."""

        # out-of-the-box for training
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(save_buffer=True, **kwargs)
        self.train_model(**kwargs)
        if training_info:
            self.evaluate_training()
        self.compute_rocs()

    def complete_load(self, force_reload=False, **kwargs):
        """Load data, build evaluation sets, load model, and compute ROCs."""
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(**kwargs)
        self.load_model(**kwargs)
        self.compute_rocs()

    def compute_rocs(self, discs=None, savedir=None):
        """Compute and plot ROCs and summary metrics for the validation set.

        This unified method handles both n_folds=1 and n_folds>1 cases.
        For n_folds=1: Uses validation set predictions.
        For n_folds>1: Uses out-of-fold predictions for unbiased evaluation.

        Metrics are computed at fixed background efficiency (bkg_eff=1e-3 by default)
        """
        time_start = time.time()

        savedir = self.output_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        (savedir / "rocs").mkdir(parents=True, exist_ok=True)
        (savedir / "outputs").mkdir(parents=True, exist_ok=True)

        # Get predictions and labels based on n_folds
        masks = self.fold_data.get("masks", {})
        X_unbinned_full = self.fold_data.get("X_unbinned_glopart", None)
        if self.n_folds == 1:
            # Single fold: predict on validation set
            y_pred = self.boosters[0].predict(self.dval)
            y_labels = self.dval.get_label()
            weights = self.dval.get_weight()
            val_idx = self.fold_data["fold_indices"][0][1]  # validation indices
            X_eval = self.fold_data["X"].iloc[val_idx]
            X_unbinned_eval = X_unbinned_full.iloc[val_idx] if X_unbinned_full is not None else None
            # Filter masks to validation indices
            masks_val = {k: v[val_idx] if v is not None else None for k, v in masks.items()}
            title_suffix = ""
        else:
            # Multi-fold: use out-of-fold predictions
            y_pred, weights, y_labels = self.get_oof_predictions()
            X_eval = self.fold_data["X"]
            X_unbinned_eval = X_unbinned_full
            masks_val = masks  # Use all masks for OOF
            title_suffix = f" ({self.n_folds}-fold OOF)"
            print(f"\nComputing ROCs with {self.n_folds}-fold out-of-fold predictions...")
            print(f"Total samples: {len(y_labels)}")

        time_end = time.time()
        print(f"Time taken to get predictions: {time_end - time_start:.2f} seconds")

        # Setup signal and background names
        signal_names = [sig_name for sig_name in self.samples if self.samples[sig_name].isSignal]
        background_names = [
            bkg_name for bkg_name in self.samples if not self.samples[bkg_name].isSignal
        ]

        print("signal_names", signal_names)
        print("background_names", background_names)

        # Keep only the non-BDT columns needed downstream by fill_discriminants.
        comparison_tagger_cols = []
        for sig_tagger in signal_names:
            taukey = CHANNELS[sig_tagger[-2:]].tagger_label
            disc_name = f"ttFatJetParTX{taukey}vsQCDTop"
            if disc_name in X_eval.columns:
                comparison_tagger_cols.append(disc_name)
            else:
                print(f"Warning: required discriminant column '{disc_name}' not found in X")
        comparison_tagger_cols = sorted(set(comparison_tagger_cols))

        # Map binned column names -> unbinned column names (only when binning was applied)
        unbinned_col_map = {}
        if X_unbinned_eval is not None:
            for col in comparison_tagger_cols:
                if col in X_unbinned_eval.columns:
                    unbinned_col_map[col] = f"{col}_unbinned"

        # Create event filters based on labels
        event_filters = {name: y_labels == i for i, name in enumerate(self.sample_names)}

        # Build preds_dict for ROCAnalyzer
        preds_dict = {}
        for class_name in self.sample_names:
            class_filter = event_filters[class_name]
            class_indices = np.flatnonzero(class_filter)

            # Start from model outputs + event weights (always needed).
            events = pd.DataFrame(y_pred[class_filter], columns=self.sample_names)
            events["finalWeight"] = weights[class_filter]

            # Add only required non-BDT discriminant columns (instead of all training features).
            if comparison_tagger_cols:
                feature_subset = X_eval.iloc[class_indices][comparison_tagger_cols].reset_index(
                    drop=True
                )
                events = pd.concat([feature_subset, events], axis=1)

            # Add unbinned GloParT columns for fair ROC comparison
            if unbinned_col_map and X_unbinned_eval is not None:
                for orig_col, unbinned_col in unbinned_col_map.items():
                    unbinned_vals = X_unbinned_eval.iloc[class_indices][orig_col]
                    events[unbinned_col] = unbinned_vals.reset_index(drop=True).to_numpy()

            # Filter masks for this class
            class_masks = {
                k: v[class_filter] if v is not None else None for k, v in masks_val.items()
            }

            preds_dict[class_name] = LoadedSample(
                sample=self.samples[class_name],
                events=events,
                tt_mask=class_masks.get("tt_mask"),
                bb_mask=class_masks.get("bb_mask"),
                m_mask=class_masks.get("m_mask"),
                e_mask=class_masks.get("e_mask"),
            )

        multiclass_confusion_matrix(preds_dict, plot_dir=savedir)

        rocAnalyzer = ROCAnalyzer(
            years=self.years,
            signals={sig: preds_dict[sig] for sig in signal_names},
            backgrounds={bkg: preds_dict[bkg] for bkg in background_names},
        )

        #########################################################
        #########################################################
        # This part configures what background outputs to put in the taggers

        # GloParT discriminants: fill unbinned version when available (fair
        # comparison against continuous BDT output), otherwise fall back to
        # the (possibly binned) original column.
        for sig_tagger in signal_names:
            taukey = CHANNELS[sig_tagger[-2:]].tagger_label
            disc_name = f"ttFatJetParTX{taukey}vsQCDTop"
            disc_names_to_fill = [disc_name]
            unbinned_name = unbinned_col_map.get(disc_name)
            if unbinned_name:
                disc_names_to_fill.append(unbinned_name)
            rocAnalyzer.fill_discriminants(
                discriminant_names=disc_names_to_fill,
                signal_name=sig_tagger,
                background_names=background_names,
            )

        # Then do BDT taggers
        bkg_tagger_groups = (
            # [["qcd"]] +
            [["qcd", "ttbarhad", "ttbarll", "ttbarsl"]]
            +
            # [["qcd", "dyjets"]] + # qcd and dy backgrounds
            [background_names]  # All backgrounds
        )

        for sig_tagger in signal_names:
            for bkg_taggers in bkg_tagger_groups:
                # Include full signal name to avoid ggf/vbf collision
                name = (
                    f"BDT {sig_tagger}vsAll"
                    if len(bkg_taggers) == 5
                    else f"BDT {sig_tagger}vsQCDTop"
                )
                rocAnalyzer.process_discriminant(
                    signal_name=sig_tagger,
                    background_names=background_names,
                    signal_tagger=sig_tagger,
                    background_taggers=bkg_taggers,
                    custom_name=name,
                )

        #########################################################
        #########################################################

        # Compute ROCs and comprehensive metrics
        discs_by_sig = {
            sig: [disc for disc in rocAnalyzer.discriminants.values() if disc.signal_name == sig]
            for sig in signal_names
        }

        rocAnalyzer.compute_rocs(compute_metrics=True)

        # Initialize results structure
        eval_results = {"metrics": {}}

        for sig, discs in discs_by_sig.items():
            disc_names = [disc.name for disc in discs]
            print("Plotting ROCs for", disc_names)
            rocAnalyzer.plot_rocs(
                title=f"BDT {sig}{title_suffix}",
                disc_names=disc_names,
                plot_dir=savedir,
                signal_name=sig,
            )

            for disc in discs:
                rocAnalyzer.plot_disc_scores(
                    disc.name, [[bkg] for bkg in background_names], savedir, signal_name=sig
                )
                try:
                    rocAnalyzer.compute_confusion_matrix(
                        disc.name, plot_dir=savedir, signal_name=sig
                    )
                except Exception as e:
                    print(f"Error computing confusion matrix for {disc.name}: {e}")

            # Find discriminant with all backgrounds for comprehensive evaluation
            disc_bkgall = [disc for disc in discs if set(background_names) == set(disc.bkg_names)]
            if len(disc_bkgall) == 0:
                print(f"No discriminant found for {sig} with background {background_names}")
                continue

            main_disc = disc_bkgall[0]

            # Store comprehensive metrics
            if hasattr(main_disc, "metrics"):
                eval_results["metrics"][sig] = main_disc.get_metrics(as_dict=True)
            else:
                print(f"Warning: Metrics not computed for {main_disc.name}")
                eval_results["metrics"][sig] = {}

            # Plot BDT output score distributions
            for i, sample in enumerate(self.sample_names):
                plotting.plot_hist(
                    [y_pred[y_labels == i, _s] for _s in range(len(self.sample_names))],
                    [
                        self.samples[self.sample_names[_s]].label
                        for _s in range(len(self.sample_names))
                    ],
                    nbins=100,
                    xlim=(0, 1),
                    weights=[weights[y_labels == i] for _s in range(len(self.sample_names))],
                    xlabel=f"BDT output score on {sample}",
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                    density=True,
                    year="-".join(self.years) if (self.years != hh_vars.years) else "2022-2023",
                    plot_dir=savedir / "outputs",
                    name=sample,
                )

        # Save comprehensive metrics summary
        rocAnalyzer.get_metrics_summary(
            signal_names=signal_names, save_path=savedir / "metrics_summary.csv"
        )

        # Print summary table
        self._print_metrics_summary(eval_results["metrics"])

        return eval_results

    def _print_metrics_summary(self, metrics_dict):
        """Print a formatted summary of metrics for all signals."""
        from tabulate import tabulate

        if not metrics_dict:
            print("No metrics to display")
            return

        # Prepare data for table (metrics at fixed background efficiency)
        headers = [
            "Signal",
            "ROC AUC",
            "PR AUC",
            "Signal Eff",
            "Precision",
            "F1",
            "Threshold",
        ]

        rows = []
        for signal, metrics in metrics_dict.items():
            if not metrics:  # Skip empty metrics
                continue

            row = [
                signal,
                f"{metrics.get('roc_auc', 0):.3f}",
                f"{metrics.get('pr_auc', 0):.3f}",
                f"{metrics.get('signal_eff', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('f1_score', 0):.3f}",
                f"{metrics.get('threshold', 0):.3f}",
            ]
            rows.append(row)

        print("\n" + "=" * 70)
        print("METRICS SUMMARY (at bkg_eff=1e-3)")
        print("=" * 70)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("\nSignal Eff = signal efficiency at 0.1% background efficiency")
        print("=" * 70)


def study_rescaling(
    years: list[str] = None,
    modelname: str = "24feb26_weak_base",
    output_dir: str | None = None,
    data_path: str | None = None,
    tt_preselection: bool = False,
    importance_only: bool = False,
) -> dict:
    """Study the impact of different balance strategies on BDT performance.

    Trains the same model architecture with each balance strategy,
    then produces comparison tables.

    Args:
        years: Year(s) of data to use (defaults to all years)
        modelname: Model config name (must exist in BDT_CONFIG)
        output_dir: Root directory for the study; each balance strategy
            gets its own subdirectory.
        data_path: Path to input data directories
        tt_preselection: Apply tt preselection
        importance_only: Only load existing models and compute feature importance

    Returns:
        Dictionary containing study results for each balance strategy
    """
    if years is None:
        years = ["all"]

    trainer = Trainer(
        years=years,
        modelname=modelname,
        output_dir=output_dir,
        data_path=data_path,
        tt_preselection=tt_preselection,
    )

    print(f"importance_only: {importance_only}")
    if not importance_only:
        trainer.load_data(force_reload=True)

    balance_rules = [
        "bysample",
        "equal_groups",
        "equal_groups_equal_channels",
    ]

    results = {}
    study_dir = trainer.output_dir

    for balance_rule in balance_rules:
        try:
            print(f"\nTraining with balance_rule={balance_rule}")

            current_test_dir = study_dir / balance_rule
            current_test_dir.mkdir(parents=True, exist_ok=True)

            trainer.output_dir = current_test_dir

            if importance_only:
                trainer.load_model()
            else:
                trainer.prepare_training_set(save_buffer=False, balance=balance_rule)
                trainer.train_model()
                results[balance_rule] = trainer.compute_rocs(savedir=current_test_dir)

            trainer.evaluate_training(savedir=current_test_dir)

        except Exception as e:
            print(f"Error training with balance_rule={balance_rule}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not importance_only:
        _rescaling_comparison(results, study_dir)

    return results


def _rescaling_comparison(results: dict, output_dir: Path) -> None:
    """Comparison of different balance strategies with comprehensive metrics.

    Args:
        results: Dictionary mapping balance_rule -> study results
        output_dir: Directory to save comparison plots and tables
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    balance_rules = list(results.keys())

    metrics = {
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "signal_eff": "Signal Eff",
        "precision": "Precision",
        "f1_score": "F1",
    }

    for sig in list(results[balance_rules[0]]["metrics"].keys()):
        for metric_key, metric_name in metrics.items():
            table_data = []
            for balance_rule in balance_rules:
                try:
                    metric_value = results[balance_rule]["metrics"][sig].get(metric_key, 0)
                    table_data.append([balance_rule, f"{metric_value:.3f}"])
                except (KeyError, TypeError):
                    table_data.append([balance_rule, "-"])

            print(f"\n{metric_name} for {sig} channel:")
            print(tabulate(table_data, headers=["Balance", metric_name], tablefmt="grid"))

            with (output_dir / f"{metric_key}_{sig}.txt").open("w") as f:
                f.write(f"{metric_name} for {sig} channel:\n")
                f.write(tabulate(table_data, headers=["Balance", metric_name], tablefmt="grid"))

        summary_data = []
        for balance_rule in balance_rules:
            try:
                metrics_dict = results[balance_rule]["metrics"][sig]
                summary_data.append(
                    [
                        balance_rule,
                        f"{metrics_dict.get('roc_auc', 0):.3f}",
                        f"{metrics_dict.get('pr_auc', 0):.3f}",
                        f"{metrics_dict.get('signal_eff', 0):.3f}",
                        f"{metrics_dict.get('precision', 0):.3f}",
                        f"{metrics_dict.get('f1_score', 0):.3f}",
                        f"{metrics_dict.get('threshold', 0):.3f}",
                    ]
                )
            except (KeyError, TypeError):
                summary_data.append([balance_rule] + ["-"] * 6)

        headers = ["Balance", "ROC AUC", "PR AUC", "Sig Eff", "Precision", "F1", "Threshold"]
        print(f"\nMetrics for {sig} channel (at bkg_eff=1e-3):")
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))

        with (output_dir / f"comprehensive_{sig}.txt").open("w") as f:
            f.write(f"Metrics for {sig} channel (at bkg_eff=1e-3):\n")
            f.write(tabulate(summary_data, headers=headers, tablefmt="grid"))

    _create_cross_channel_comparison(results, balance_rules, output_dir)


def _create_cross_channel_comparison(results, balance_rules, output_dir):
    """Create comparison tables across all channels for key metrics."""
    key_metrics = ["roc_auc", "signal_eff", "precision", "f1_score"]
    sigs = list(results[balance_rules[0]]["metrics"].keys())

    for metric in key_metrics:
        print(f"\nCross-channel comparison: {metric.upper()}")

        table_data = []
        for balance_rule in balance_rules:
            row = [balance_rule]
            for sig in sigs:
                try:
                    value = results[balance_rule]["metrics"][sig].get(metric, 0)
                    row.append(f"{value:.3f}")
                except (KeyError, TypeError):
                    row.append("-")
            table_data.append(row)

        headers = ["Balance"] + sigs
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        with (output_dir / f"cross_channel_{metric}.txt").open("w") as f:
            f.write(f"Cross-channel comparison: {metric.upper()}\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))


def eval_bdt_preds(
    years: list[str],
    samples: list[str],
    model: str,
    save: bool = True,
    output_dir: str | None = None,
    data_path: str | None = None,
    tt_preselection: bool = False,
):
    """Evaluate BDT predictions on data.

    Args:
        eval_samples: List of sample names to evaluate
        model: Name of model to use for predictions (signals are read from model config)

    One day to be made more flexible (here only integrated with the data you already train on)
    """

    years = hh_vars.years if years[0] == "all" else list(years)

    if samples[0] == "all":
        samples = list(SAMPLES.keys())

    if save:
        if output_dir is None:
            output_dir = DATA_DIR

        # check if output_dir is writable
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Directory {output_dir} is not writable")

    # Load model globally for all years, evaluate by year to reduce memory usage
    bst = Trainer(
        years=years,
        modelname=model,
        data_path=data_path,
        tt_preselection=tt_preselection,
    ).load_model()

    evals = {year: {sample_name: {} for sample_name in samples} for year in years}

    for year in years:

        # To reduce memory usage, load data once for each year
        trainer = Trainer(
            years=[year],
            modelname=model,
            data_path=data_path,
            tt_preselection=tt_preselection,
        )
        trainer.load_data(force_reload=True)

        feats = [feat for cat in trainer.train_vars for feat in trainer.train_vars[cat]]
        for sample_name in trainer.events_dict[year]:

            dsample = xgb.DMatrix(
                np.stack(
                    [trainer.events_dict[year][sample_name].get_var(feat) for feat in feats],
                    axis=1,
                ),
                feature_names=feats,
            )

            # Use global model to predict
            y_pred = bst.predict(dsample)
            evals[year][sample_name] = y_pred
            if save:
                pred_dir = Path(output_dir) / "BDT_predictions" / year / sample_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / f"{model}_preds.npy", y_pred)
                with Path.open(pred_dir / f"{model}_preds_shape.txt", "w") as f:
                    f.write(str(y_pred.shape) + "\n")

            print(f"Processed sample {sample_name} for year {year}")

        del trainer

    return evals


_NON_MODEL_JSON = {
    "evals_result",
    "wps_ttpart",
    "fold_indices",
}


def _discover_models_in_folder(folder: str | Path) -> list[tuple[str, str]]:
    """Discover trained BDT models in a directory tree.

    Scans *folder* and its immediate subdirectories for XGBoost model JSON
    files (``{modelname}.json`` or ``{modelname}_fold{i}.json``).

    Returns:
        List of ``(model_name, model_dir)`` tuples.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    discovered: list[tuple[str, str]] = []
    search_dirs = [folder] + sorted(d for d in folder.iterdir() if d.is_dir())

    for search_dir in search_dirs:
        seen_models: set[str] = set()
        for json_file in sorted(search_dir.glob("*.json")):
            stem = json_file.stem
            if stem in _NON_MODEL_JSON:
                continue
            model_name = re.sub(r"_fold\d+$", "", stem)
            if model_name not in seen_models:
                seen_models.add(model_name)
                discovered.append((model_name, str(search_dir)))

    return discovered


def _resolve_model_inputs(inputs: list[str | Path]) -> list[tuple[str, str, str]]:
    """Resolve a mixed list of paths into ``(model_name, model_dir, tag)`` triples.

    Each input can be:

    - A path to a ``.json`` model file (absolute or relative): the model name
      is derived from the file stem (``_fold{i}`` suffixes are stripped) and
      the parent directory is used as *model_dir*.
    - A directory path: scanned for model JSON files via
      :func:`_discover_models_in_folder`.

    Duplicate ``(model_name, model_dir)`` pairs are silently dropped.
    Each discovered model name is validated against :data:`BDT_CONFIG`; a
    ``ValueError`` is raised if no matching config file exists.

    Returns:
        Deduplicated list of ``(model_name, model_dir, tag)`` tuples.
    """
    result: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(model_name: str, model_dir: str, tag: str, source: str) -> None:
        key = (model_name, model_dir)
        if key in seen:
            return
        if model_name not in BDT_CONFIG:
            raise ValueError(
                f"Model '{model_name}' (found via {source}) has no BDT config. "
                f"Expected: bdt_configs/config_{model_name}.py"
            )
        seen.add(key)
        result.append((model_name, model_dir, tag))

    for item in inputs:
        p = Path(item)
        if p.suffix == ".json":
            if not p.is_file():
                raise ValueError(f"Model file not found: {p}")
            model_name = re.sub(r"_fold\d+$", "", p.stem)
            tag = p.parent.name
            if tag == model_name:
                tag = p.parent.parent.name
            _add(model_name, str(p.parent), tag, str(p))
        elif p.is_dir():
            tag = p.name
            discovered = _discover_models_in_folder(p)
            if not discovered:
                raise ValueError(f"No models found in directory: {p}")
            print(
                f"Discovered {len(discovered)} model(s) in {p}: "
                + ", ".join(n for n, _ in discovered)
            )
            for name, mdir in discovered:
                _add(name, mdir, tag, str(p))
        else:
            raise ValueError(f"Input '{item}' is neither a .json file nor a directory")

    return result


def _cms_distinct_colors(n: int) -> list[str]:
    """Return *n* visually distinct colours using the ``tab20`` colormap."""
    cmap = mpl.colormaps["tab20"]
    return [mpl.colors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _extract_hyperparameters(model_tags: list[str]) -> pd.DataFrame | None:
    """Try to load BDT_CONFIG for each model tag and return a DataFrame of HPs.

    Returns None if no config can be loaded for any model.
    """
    rows = []
    hp_keys_of_interest = [
        "max_depth",
        "eta",
        "subsample",
        "colsample_bytree",
        "num_parallel_tree",
        "alpha",
        "gamma",
        "lambda",
    ]
    for tag in model_tags:
        try:
            cfg = BDT_CONFIG[tag]
        except (KeyError, ValueError):
            continue
        hp = cfg.get("hyperpars", {})
        row = {"model": tag}
        for k in hp_keys_of_interest:
            if k in hp:
                row[k] = hp[k]
        row["num_rounds"] = cfg.get("num_rounds")
        row["n_folds"] = cfg.get("n_folds")
        if len(row) > 1:
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _plot_hp_correlations(
    ranking: pd.DataFrame,
    hp_df: pd.DataFrame,
    metric_col: str,
    base_out: Path,
) -> None:
    """Scatter-plot each varying hyperparameter against the aggregate metric.

    Also saves a Spearman correlation summary CSV.
    """
    from scipy import stats

    merged = ranking.merge(hp_df, on="model", how="inner")
    if merged.empty:
        return

    hp_cols = [c for c in hp_df.columns if c != "model"]
    # Keep only HPs that actually vary across models
    hp_cols = [c for c in hp_cols if len(merged[c].unique()) > 1]
    if not hp_cols:
        print("  All hyperparameters are constant across models — skipping HP plots.")
        return

    hp_dir = _ensure_dir(base_out / "hyperparameters")

    # --- individual scatter plots ----------------------------------------
    n_hp = len(hp_cols)
    ncols = min(n_hp, 3)
    nrows = (n_hp + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, hp in enumerate(hp_cols):
        ax = axes[idx // ncols][idx % ncols]
        x = merged[hp].astype(float)
        y = merged[metric_col].astype(float)
        ax.scatter(x, y, s=60, zorder=5, edgecolors="black", linewidths=0.5)

        rho, pval = stats.spearmanr(x, y)
        ax.set_xlabel(hp, fontsize=14)
        ax.set_ylabel(metric_col.replace("_", " "), fontsize=14)
        ax.set_title(f"$\\rho_{{S}}$={rho:+.2f}  (p={pval:.2g})", fontsize=12)
        ax.tick_params(labelsize=11)

    # hide unused subplots
    for idx in range(n_hp, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(hp_dir / "hp_vs_metric.pdf", bbox_inches="tight")
    fig.savefig(hp_dir / "hp_vs_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- correlation summary CSV -----------------------------------------
    corr_rows = []
    for hp in hp_cols:
        x = merged[hp].astype(float)
        y = merged[metric_col].astype(float)
        rho, pval = stats.spearmanr(x, y)
        corr_rows.append({"hyperparameter": hp, "spearman_rho": rho, "p_value": pval})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=abs, ascending=False)
    corr_df.to_csv(hp_dir / "hp_correlations.csv", index=False)
    print(f"  HP correlations saved to {hp_dir / 'hp_correlations.csv'}")


def compare_models_light(
    inputs: list[str | Path],
    output_dir: str | None = None,
    discriminant_filter: str | None = "vsAll",
    ranking_metric: str = "roc_auc",
) -> pd.DataFrame:
    """Light comparison of trained models using only metrics_summary.csv files.

    Reads pre-computed metrics from each model's output directory without
    reloading data or recomputing predictions.  Produces:

    * Per-signal summary tables (console)
    * CMS-styled bar charts of key metrics
    * An **aggregate ranking** across all signals
    * A **heatmap** of models x signals
    * Hyperparameter correlation plots (when configs are loadable)
    * A GP surrogate fit to predict the optimal HP set

    Each entry in *inputs* can be a directory that either directly contains a
    ``metrics_summary.csv`` file or has subdirectories that do.  The folder
    name is used as the model tag.

    Args:
        inputs: Directories containing trained models with metrics_summary.csv.
        output_dir: Where to save comparison outputs (default: ``comparison_light``).
        discriminant_filter: Only include discriminants whose name contains this
            substring.  Default ``"vsAll"`` keeps the BDT-vs-all-backgrounds
            discriminants.  Set to ``None`` to include everything.
        ranking_metric: Metric to use for aggregate ranking across signals.
            Default ``"roc_auc"``.

    Returns:
        Combined ``DataFrame`` with metrics from all input models.
    """
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    hep.style.use("CMS")

    base_out = Path(output_dir) if output_dir else Path("comparison_light")
    _ensure_dir(base_out)

    # -- collect metrics_summary.csv files --------------------------------
    csv_frames: list[pd.DataFrame] = []
    for item in inputs:
        p = Path(item)
        if not p.is_dir():
            raise ValueError(f"Input '{item}' is not a directory")

        csvs = list(p.rglob("metrics_summary.csv"))
        if not csvs:
            raise FileNotFoundError(f"No metrics_summary.csv found under {p}")

        for csv_path in sorted(csvs):
            tag = csv_path.parent.name
            summary = pd.read_csv(csv_path)
            summary.insert(0, "model", tag)
            csv_frames.append(summary)

    combined = pd.concat(csv_frames, ignore_index=True)

    if discriminant_filter:
        combined = combined[
            combined["discriminant"].str.contains(discriminant_filter, na=False)
        ].reset_index(drop=True)

    if combined.empty:
        print("No metrics matched the discriminant filter — nothing to compare.")
        return combined

    key_metrics = ["roc_auc", "pr_auc", "signal_eff", "precision", "f1_score", "threshold"]
    signals = sorted(combined["signal"].unique())
    models = combined["model"].unique()

    # -- per-signal summary tables ----------------------------------------
    for sig in signals:
        rows = []
        for model_tag in models:
            sub = combined[(combined["signal"] == sig) & (combined["model"] == model_tag)]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                rows.append(
                    [model_tag, r["discriminant"]] + [f"{r.get(m, 0):.4f}" for m in key_metrics]
                )
        headers = ["Model", "Discriminant"] + [m.replace("_", " ").title() for m in key_metrics]
        print(f"\n{'='*80}")
        print(f"  Signal: {sig}")
        print(f"{'='*80}")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    # =====================================================================
    # AGGREGATE RANKING across signals
    # =====================================================================
    # For each model, compute the mean of `ranking_metric` over all signals.
    # When a model has multiple discriminants per signal (rare with vsAll
    # filter), we take the best per signal first.
    best_per_signal = combined.groupby(["model", "signal"])[ranking_metric].max().reset_index()
    ranking = (
        best_per_signal.groupby("model")[ranking_metric]
        .agg(["mean", "std", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": f"mean_{ranking_metric}",
                "std": f"std_{ranking_metric}",
                "min": f"min_{ranking_metric}",
            }
        )
        .sort_values(f"mean_{ranking_metric}", ascending=False)
    )
    ranking["rank"] = range(1, len(ranking) + 1)

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RANKING  (metric: mean {ranking_metric} across {len(signals)} signals)")
    print(f"{'='*80}")
    rank_headers = ["Rank", "Model", f"Mean {ranking_metric}", "Std", "Worst-signal"]
    rank_rows = [
        [
            r["rank"],
            r["model"],
            f"{r[f'mean_{ranking_metric}']:.5f}",
            f"{r[f'std_{ranking_metric}']:.5f}" if pd.notna(r[f"std_{ranking_metric}"]) else "—",
            f"{r[f'min_{ranking_metric}']:.5f}",
        ]
        for _, r in ranking.iterrows()
    ]
    print(tabulate(rank_rows, headers=rank_headers, tablefmt="grid"))
    ranking.to_csv(base_out / "ranking.csv", index=False)

    # =====================================================================
    # CMS-styled bar charts (one subplot per signal)
    # =====================================================================
    n_models = len(models)
    colors = _cms_distinct_colors(n_models)
    model_color = {m: colors[i] for i, m in enumerate(models)}

    plot_metrics = ["roc_auc", "signal_eff", "f1_score"]
    plot_labels = {
        "roc_auc": "ROC AUC",
        "signal_eff": r"Signal efficiency ($\epsilon_{sig}$)",
        "f1_score": "F1 Score",
    }

    for metric in plot_metrics:
        n_sig = len(signals)
        fig, axes = plt.subplots(
            1,
            n_sig,
            figsize=(max(6, 4 * n_sig), max(5, 0.45 * n_models + 2)),
            squeeze=False,
            sharey=True,
        )

        for ax, sig in zip(axes[0], signals):
            sub = combined[combined["signal"] == sig].sort_values(metric, ascending=True)
            if sub.empty:
                continue
            labels = sub["model"].tolist()
            values = sub[metric].astype(float).to_numpy()
            bar_colors = [model_color[m] for m in labels]

            bars = ax.barh(
                range(len(values)),
                values,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.4,
                height=0.7,
            )
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(plot_labels[metric], fontsize=13)
            ax.set_title(sig, fontsize=13, fontweight="bold")

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=8,
                )

            ax.tick_params(axis="both", labelsize=10)

        hep.cms.label(ax=axes[0][0], label="Work in Progress", data=False, fontsize=13, loc=0)
        fig.tight_layout()
        fig.savefig(base_out / f"compare_{metric}.pdf", bbox_inches="tight")
        fig.savefig(base_out / f"compare_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # HEATMAP: models x signals for the ranking metric
    # =====================================================================
    pivot = best_per_signal.pivot_table(index="model", columns="signal", values=ranking_metric)
    # Sort models by aggregate ranking
    rank_order = ranking.sort_values("rank")["model"].tolist()
    pivot = pivot.reindex(rank_order)

    fig, ax = plt.subplots(
        figsize=(max(6, 1.5 * len(signals) + 2), max(4, 0.45 * len(rank_order) + 2))
    )
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="RdYlGn",
        vmin=pivot.to_numpy().min() - 0.01,
        vmax=pivot.to_numpy().max() + 0.001,
    )
    ax.set_xticks(range(len(signals)))
    ax.set_xticklabels(signals, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(rank_order)))
    ax.set_yticklabels(rank_order, fontsize=10)
    ax.set_ylabel("Model (ranked)", fontsize=13)

    for i in range(len(rank_order)):
        for j in range(len(signals)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if val < pivot.to_numpy().mean() else "black",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(ranking_metric.replace("_", " ").upper(), fontsize=12)
    hep.cms.label(ax=ax, label="Work in Progress", data=False, fontsize=13, loc=0)
    fig.tight_layout()
    fig.savefig(base_out / "heatmap_models_signals.pdf", bbox_inches="tight")
    fig.savefig(base_out / "heatmap_models_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # HYPERPARAMETER ANALYSIS
    # =====================================================================
    hp_df = _extract_hyperparameters(list(models))
    if hp_df is not None and not hp_df.empty:
        agg_col = f"mean_{ranking_metric}"
        print(f"\n  Loaded hyperparameters for {len(hp_df)} / {len(models)} models.")
        _plot_hp_correlations(ranking, hp_df, agg_col, base_out)
    else:
        print("\n  Could not load hyperparameters from BDT_CONFIG — skipping HP analysis.")

    # -- save combined csv ------------------------------------------------
    combined.to_csv(base_out / "combined_metrics.csv", index=False)
    print(f"\nCombined metrics saved to {base_out / 'combined_metrics.csv'}")
    print(f"Plots saved to {base_out}")

    return combined


def compare_models(
    inputs: list[str | Path],
    years: list[str] = ("all",),
    data_path: str | None = None,
    tt_preselection: bool = False,
    output_dir: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare multiple trained BDT models by computing ROC curves and metrics.

    For k-fold models, uses averaged predictions across all folds.

    Each entry in *inputs* can be:

    - An absolute or relative path to a ``.json`` model file.
    - A directory path, which is scanned for model JSON files.

    Mixed lists are supported. All discovered model names are validated against
    :data:`BDT_CONFIG` before any data is loaded.

    Args:
        inputs: Model JSON files and/or directories to scan.
        years: Years to include in the comparison.
        data_path: Optional path to data directory.
        tt_preselection: Whether to apply tt preselection.
        output_dir: Output directory for results (default: "comparison").

    Returns:
        Nested dict: metrics_by_model[model][signal] -> metrics_dict
    """
    resolved = _resolve_model_inputs(inputs)
    model_names = [m for m, _, _ in resolved]
    model_dirs = [d for _, d, _ in resolved]
    labels = [f"{tag}::{name}" for name, _, tag in resolved]

    if len(labels) < 2:
        raise ValueError("Need at least 2 models to compare")
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"Could not derive unique labels for models: {labels}. "
            "Ensure model directories have distinct names."
        )

    base_out = Path(output_dir) if output_dir else Path("comparison")
    _ensure_dir(base_out)

    # Load data once into a reference trainer; share events_dict with others
    ref_trainer = Trainer(
        years=list(years),
        modelname=model_names[0],
        output_dir=model_dirs[0],
        data_path=data_path,
        tt_preselection=tt_preselection,
    )
    ref_trainer.load_data(force_reload=True)
    shared_events_dict = ref_trainer.events_dict
    shared_samples = ref_trainer.samples
    ref_sample_names = ref_trainer.sample_names
    ref_years = ref_trainer.years

    # Process models one at a time to keep memory bounded.
    weights: np.ndarray | None = None
    preds_dict: dict[str, LoadedSample] = {}
    kfold_models: list[str] = []
    y_labels: np.ndarray | None = None
    for label, model_name, model_dir in zip(labels, model_names, model_dirs):
        print(f"\n{'='*60}\nProcessing model: {label}\n{'='*60}")
        tr = Trainer(
            years=list(years),
            modelname=model_name,
            output_dir=model_dir,
            data_path=data_path,
            tt_preselection=tt_preselection,
        )
        tr.events_dict = shared_events_dict
        tr.samples = shared_samples
        tr.sample_names = ref_sample_names
        tr.prepare_training_set(save_buffer=False, prediction_only=True)
        tr.load_model()

        if tr.sample_names != ref_sample_names:
            raise ValueError(
                f"Model '{label}' has sample order {tr.sample_names}, "
                f"expected {ref_sample_names}"
            )

        if tr.use_kfold:
            kfold_models.append(label)

        dval = tr.dval if hasattr(tr, "dval") else tr.fold_data["dval"][0]

        # Extract labels/weights from the first model (same data, same split seed)
        if y_labels is None:
            y_labels = dval.get_label()
            weights = dval.get_weight()
            for class_idx, class_name in enumerate(ref_sample_names):
                cls_mask = y_labels == class_idx
                preds_dict[class_name] = LoadedSample(
                    sample=shared_samples[class_name],
                    events={"finalWeight": weights[cls_mask]},
                )

        # Predict
        if len(tr.boosters) > 1:
            y_pred = np.mean([bst.predict(dval) for bst in tr.boosters], axis=0)
        else:
            y_pred = tr.boosters[0].predict(dval)

        if y_pred.ndim != 2 or y_pred.shape[1] != len(ref_sample_names):
            raise ValueError(
                f"Model '{label}' output shape {y_pred.shape} incompatible with "
                f"{len(ref_sample_names)} classes"
            )

        for class_idx, class_name in enumerate(ref_sample_names):
            mask = y_labels == class_idx
            for pred_idx, pred_name in enumerate(ref_sample_names):
                preds_dict[class_name].events[f"{label}::{pred_name}"] = y_pred[mask, pred_idx]

        # Free this trainer's DMatrices, boosters, and fold_data before the next model
        del tr
        gc.collect()

    # events_dict no longer needed — predictions are stored in preds_dict
    ref_trainer.events_dict = {y: {} for y in ref_years}
    del shared_events_dict
    gc.collect()

    # Convert to DataFrames
    for class_name in preds_dict:
        preds_dict[class_name].events = pd.DataFrame(preds_dict[class_name].events)

    # Separate signals and backgrounds
    signal_names = [n for n in ref_sample_names if shared_samples[n].isSignal]
    background_names = [n for n in ref_sample_names if not shared_samples[n].isSignal]

    # Set up ROC analyzer
    roc_analyzer = ROCAnalyzer(
        years=ref_years,
        signals={s: preds_dict[s] for s in signal_names},
        backgrounds={b: preds_dict[b] for b in background_names},
    )

    # Register discriminants for each (label, signal) pair
    for sig_name in signal_names:
        for label in labels:
            roc_analyzer.process_discriminant(
                signal_name=sig_name,
                background_names=background_names,
                signal_tagger=f"{label}::{sig_name}",
                background_taggers=[f"{label}::{bkg}" for bkg in background_names],
                custom_name=f"{label}_{sig_name}",
            )

    roc_analyzer.compute_rocs(compute_metrics=True)

    # Plot ROCs and extract metrics
    label_set = set(labels)
    metrics_by_model: dict[str, dict[str, dict[str, float]]] = {lab: {} for lab in labels}
    roc_dir = _ensure_dir(base_out / "rocs")

    for sig_name in signal_names:
        disc_names = [
            d.name for d in roc_analyzer.discriminants.values() if d.signal_name == sig_name
        ]
        roc_analyzer.plot_rocs(
            title=f"Model Comparison: {sig_name}",
            disc_names=disc_names,
            plot_dir=roc_dir,
            signal_name=sig_name,
        )

        for disc in roc_analyzer.discriminants.values():
            if disc.signal_name != sig_name:
                continue
            disc_label = disc.name.rsplit("_", 1)[0]
            if disc_label in label_set and hasattr(disc, "get_metrics"):
                metrics_by_model[disc_label][sig_name] = disc.get_metrics(as_dict=True)

    # Save metrics CSV
    metric_keys = ["roc_auc", "pr_auc", "signal_eff", "precision", "f1_score", "threshold"]
    csv_path = base_out / "comparison_metrics.csv"
    with csv_path.open("w") as f:
        f.write(",".join(["model", "signal"] + metric_keys) + "\n")
        for model, by_signal in metrics_by_model.items():
            for signal, metrics in by_signal.items():
                row = [model, signal] + [f"{metrics.get(k, 0):.6f}" for k in metric_keys]
                f.write(",".join(row) + "\n")

    # Save metadata
    with (base_out / "comparison_index.json").open("w") as f:
        json.dump(
            {
                "labels": labels,
                "model_names": model_names,
                "model_dirs": [str(d) for d in model_dirs],
                "years": list(years),
                "signals": signal_names,
                "backgrounds": background_names,
                "kfold_models": kfold_models,
            },
            f,
            indent=2,
        )

    print(f"Comparison results saved to {base_out}")
    return metrics_by_model


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a multiclass BDT model")

    parser.add_argument(
        "--years",
        nargs="+",
        default=["all"],
        help="Year(s) of data to use. Can be: 'all', or multiple years (e.g. --years 2022 2023 2024)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="3Mar26_weak_deep7_vbfk2v0",
        help="Name of the model configuration to use",
    )
    parser.add_argument(
        "--tt-preselection",
        action="store_true",
        default=False,
        help="Apply tt preselection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Subdirectory to save model and plots within `/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/` if training/evaluating. Full directory to store predictions if --eval-bdt-preds is specified (checks writing permissions).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the data directories",
    )
    parser.add_argument(
        "--force-reload", action="store_true", default=False, help="Force reload of data"
    )

    parser.add_argument(
        "--study-rescaling",
        action="store_true",
        default=False,
        help="Study the impact of different balance strategies on BDT performance",
    )
    parser.add_argument(
        "--eval-bdt-preds",
        action="store_true",
        default=False,
        help="Evaluate BDT predictions on data if specified",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        default=False,
        help="Compare multiple trained models with ROC overlays and CSV metrics",
    )
    parser.add_argument(
        "--compare-light",
        action="store_true",
        default=False,
        help=(
            "Light comparison using only metrics_summary.csv files already "
            "produced during training. No data loading or prediction needed."
        ),
    )
    parser.add_argument(
        "--disc-filter",
        type=str,
        default="vsAll",
        help=(
            "Substring filter for discriminant names in --compare-light. "
            "Default 'vsAll' keeps BDT-vs-all-backgrounds discriminants. "
            "Use '' to include everything."
        ),
    )
    parser.add_argument(
        "--ranking-metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "pr_auc", "signal_eff", "f1_score"],
        help=("Metric for aggregate model ranking in --compare-light. " "Default 'roc_auc'."),
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help=(
            "Model JSON files and/or directories to compare. "
            "Each entry can be a path to a .json model file or a directory "
            "that will be scanned for model files."
        ),
    )
    parser.add_argument(
        "--samples", nargs="+", default=None, help="Samples to evaluate BDT predictions on"
    )
    parser.add_argument(
        "--importance-only",
        action="store_true",
        default=False,
        help="Only compute importance of features",
    )
    parser.add_argument(
        "--bin-features",
        action="store_true",
        default=False,
        help="Apply feature binning to gloParT tautauvsQCDTop scores (discretize to 0.05 steps)",
    )
    parser.add_argument(
        "--cap-weights",
        action="store_true",
        default=False,
        help="Cap per-event training weights at 10x the sample median after balance rescaling",
    )

    # Add mutually exclusive group for train/load
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train a new model")
    group.add_argument("--load", action="store_true", default=True, help="Load model from file")

    args = parser.parse_args()

    if args.study_rescaling:
        study_rescaling(
            years=args.years,
            modelname=args.model,
            output_dir=args.output_dir,
            data_path=args.data_path,
            tt_preselection=args.tt_preselection,
            importance_only=args.importance_only,
        )
        exit()

    if args.eval_bdt_preds:
        if not args.samples:
            parser.error("--eval-bdt-preds requires --samples to be specified.")
        else:
            print(args.model)
            eval_bdt_preds(
                years=args.years,
                samples=args.samples,
                model=args.model,
                output_dir=args.output_dir,
                data_path=args.data_path,
                tt_preselection=args.tt_preselection,
            )
        exit()

    if args.compare_models:
        if not args.inputs:
            parser.error("--compare-models requires --inputs")

        # Resolve relative paths against MODEL_DIR.parent
        resolved_inputs = []
        for item in args.inputs:
            p = Path(item)
            if not p.is_absolute():
                p = MODEL_DIR.parent / p
            resolved_inputs.append(str(p))

        compare_models(
            inputs=resolved_inputs,
            years=args.years,
            data_path=args.data_path,
            tt_preselection=args.tt_preselection,
            output_dir=args.output_dir,
        )
        exit()

    if args.compare_light:
        if not args.inputs:
            parser.error("--compare-light requires --inputs")

        resolved_inputs = []
        for item in args.inputs:
            p = Path(item)
            if not p.is_absolute():
                p = MODEL_DIR.parent / p
            resolved_inputs.append(str(p))

        disc_filter = args.disc_filter if args.disc_filter else None
        compare_models_light(
            inputs=resolved_inputs,
            output_dir=args.output_dir,
            discriminant_filter=disc_filter,
            ranking_metric=args.ranking_metric,
        )
        exit()

    trainer = Trainer(
        years=args.years,
        modelname=args.model,
        output_dir=args.output_dir,
        data_path=args.data_path,
        tt_preselection=args.tt_preselection,
    )

    # Apply feature binning settings from CLI to config
    if args.bin_features:
        trainer.bdt_config[args.model]["bin_features"] = WPS_TTPART.keys()
        print(f"Feature binning enabled for features: {WPS_TTPART.keys()}")
        print("  Note: Features will only be binned if WP bin edges are provided in WPS_TTPART")

    if args.cap_weights:
        trainer.cap_weights = True
        print("Weight capping enabled: per-event weights will be capped at 10x sample median")

    if args.train:
        print("Running in training mode")
        trainer.complete_train(force_reload=args.force_reload)
    else:
        print("Running in load/evaluate mode")
        trainer.complete_load(force_reload=args.force_reload)
