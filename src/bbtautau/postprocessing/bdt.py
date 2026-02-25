"""
BDT training and inference script for the bbtautau package.

Authors: Ludovico Mori
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES
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

                # ARE WE APPLYING TRIGGERS?
                # apply_triggers(self.events_dict[year], year, channel)

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

    def _process_samples_for_training(self, scale_rule="signal", balance="bysample_clip_1to10"):
        """Process samples and compute weights for training.

        This is the common data processing logic used by prepare_training_set().
        Extracts features and computes rescaled weights for all samples.

        Args:
            scale_rule: Rule for global scaling weights
            balance: Rule for balancing samples

        Returns:
            tuple: (X, y, weights, weights_rescaled) - processed data arrays
        """
        if scale_rule not in ["signal", "signal_3e-1", "signal_3"]:
            raise ValueError(f"Invalid scale rule: {scale_rule}")

        if balance not in [
            "bysample",
            "bysample_clip_1to10",
            "bysample_clip_1to20",
            "grouped_physics",
            "sqrt_scaling",
            "ens_weighting",
        ]:
            raise ValueError(f"Invalid balance rule: {balance}")

        global_scale_factor = {
            "signal": 1.0,
            "signal_3e-1": 3e-1,
            "signal_3": 3,
        }

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
            # Compute signal statistics for this year
            total_signal_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][sig_sample].get_var("finalWeight"))
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
            ).sum()

            len_signal = sum(
                len(self.events_dict[year][sig_sample].events)
                for sig_sample in self.samples
                if self.samples[sig_sample].isSignal
            )

            len_signal_per_channel = len_signal / len(
                [sample for sample in self.samples if self.samples[sample].isSignal]
            )

            avg_signal_weight = total_signal_weight / len_signal

            for sample_name, sample in self.events_dict[year].items():
                X_sample = pd.DataFrame({feat: sample.get_var(feat) for feat in self.feats})
                weights = np.abs(sample.get_var("finalWeight").copy())
                weights_rescaled = weights.copy()

                # Aggregate for multi-year stats
                key = ("Initial", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled.copy())

                # Global rescaling by average signal weight
                weights_rescaled = (
                    weights_rescaled / avg_signal_weight * global_scale_factor[scale_rule]
                )

                key = ("Global rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled.copy())

                # Apply balance rescaling
                weights_rescaled = self._apply_balance_rescaling(
                    weights_rescaled=weights_rescaled,
                    balance=balance,
                    sample=sample,
                    sample_name=sample_name,
                    year=year,
                    len_signal_per_channel=len_signal_per_channel,
                    global_scale_factor=global_scale_factor[scale_rule],
                )

                key = ("Balance rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled.copy())

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
        sample,
        sample_name,
        year,
        len_signal_per_channel,
        global_scale_factor,
    ):
        """Apply balance rescaling to weights based on the chosen strategy.

        Args:
            weights_rescaled: Current rescaled weights
            balance: Balance strategy name
            sample: LoadedSample object
            sample_name: Name of the sample
            year: Year string
            len_signal_per_channel: Average signal events per channel
            global_scale_factor: Global scale factor value

        Returns:
            Rescaled weights array
        """
        n_events = len(sample.events)

        if balance == "bysample":
            weights_rescaled = weights_rescaled / np.sum(weights_rescaled) * len_signal_per_channel

        elif balance == "bysample_clip_1to10":
            target_total_weight = len_signal_per_channel * global_scale_factor
            avg_weight_if_scaled = target_total_weight / n_events
            min_avg_weight = global_scale_factor / 10.0
            if avg_weight_if_scaled < min_avg_weight:
                target_total_weight = min_avg_weight * n_events
            scaling_factor = target_total_weight / np.sum(weights_rescaled)
            weights_rescaled = weights_rescaled * scaling_factor

        elif balance == "bysample_clip_1to20":
            target_total_weight = len_signal_per_channel * global_scale_factor
            avg_weight_if_scaled = target_total_weight / n_events
            min_avg_weight = global_scale_factor / 20.0
            if avg_weight_if_scaled < min_avg_weight:
                target_total_weight = min_avg_weight * n_events
            scaling_factor = target_total_weight / np.sum(weights_rescaled)
            weights_rescaled = weights_rescaled * scaling_factor

        elif balance == "grouped_physics":
            if sample.sample.isSignal:
                target_total_weight = len_signal_per_channel * global_scale_factor
            elif "ttbar" in sample_name:
                ttbar_total_events = sum(
                    len(self.events_dict[year][s].events)
                    for s in self.samples
                    if "ttbar" in s and s in self.events_dict[year]
                )
                ttbar_fraction = n_events / ttbar_total_events
                target_total_weight = len_signal_per_channel * global_scale_factor * ttbar_fraction
            else:
                other_total_events = sum(
                    len(self.events_dict[year][s].events)
                    for s in self.samples
                    if not self.samples[s].isSignal
                    and "ttbar" not in s
                    and s in self.events_dict[year]
                )
                other_fraction = n_events / other_total_events if other_total_events > 0 else 1.0
                target_total_weight = len_signal_per_channel * global_scale_factor * other_fraction
            scaling_factor = target_total_weight / np.sum(weights_rescaled)
            weights_rescaled = weights_rescaled * scaling_factor

        elif balance == "sqrt_scaling":
            sqrt_factor = np.sqrt(n_events)
            target_total_weight = (
                sqrt_factor * np.sqrt(len_signal_per_channel) * global_scale_factor
            )
            scaling_factor = target_total_weight / np.sum(weights_rescaled)
            weights_rescaled = weights_rescaled * scaling_factor

        elif balance == "ens_weighting":  # Effective number of samples
            beta = 0.999
            cb_weight = (1 - beta) / (1 - beta**n_events)
            signal_cb_weight = (1 - beta) / (1 - beta**len_signal_per_channel)
            target_total_weight = (
                (cb_weight / signal_cb_weight) * len_signal_per_channel * global_scale_factor
            )
            scaling_factor = target_total_weight / np.sum(weights_rescaled)
            weights_rescaled = weights_rescaled * scaling_factor

        return weights_rescaled

    def prepare_training_set(
        self, save_buffer=False, scale_rule="signal", balance="bysample_clip_1to10"
    ):
        """Prepare features and labels for training with k-fold cross-validation.

        This unified method handles both single-split (n_folds=1) and k-fold (n_folds>1) cases.
        The data is stored in self.fold_data.

        For n_folds=1: Uses train_test_split for a single train/validation split.
        For n_folds>1: Uses StratifiedKFold to create k stratified folds.

        Args:
            save_buffer: Whether to save DMatrix buffer files for quicker loading
            scale_rule: Rule for global scaling weights ('signal', 'signal_3e-1', 'signal_3')
            balance: Rule for balancing samples
        """
        if self.loaded_dmatrix:
            return

        # Process samples (common logic)
        X, y, weights, weights_rescaled, masks, event_ids_list = self._process_samples_for_training(
            scale_rule=scale_rule, balance=balance
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

            # DMatrix with rescaled weights (for training)
            dtrain_rescaled = xgb.DMatrix(
                X.iloc[train_idx], label=y[train_idx], weight=weights_rescaled[train_idx], nthread=8
            )
            dval_rescaled = xgb.DMatrix(
                X.iloc[val_idx], label=y[val_idx], weight=weights_rescaled[val_idx], nthread=8
            )
            self.fold_data["dtrain_rescaled"].append(dtrain_rescaled)
            self.fold_data["dval_rescaled"].append(dval_rescaled)

            # DMatrix with original weights (for ROC computation)
            dtrain = xgb.DMatrix(
                X.iloc[train_idx], label=y[train_idx], weight=weights[train_idx], nthread=8
            )
            dval = xgb.DMatrix(
                X.iloc[val_idx], label=y[val_idx], weight=weights[val_idx], nthread=8
            )
            self.fold_data["dtrain"].append(dtrain)
            self.fold_data["dval"].append(dval)

        # For backward compatibility, also set single-fold attributes
        if self.n_folds == 1:
            self.dtrain_rescaled = self.fold_data["dtrain_rescaled"][0]
            self.dval_rescaled = self.fold_data["dval_rescaled"][0]
            self.dtrain = self.fold_data["dtrain"][0]
            self.dval = self.fold_data["dval"][0]

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
                    model_path = self.output_dir / f"{self.modelname}.json"
                else:
                    model_path = self.output_dir / f"{self.modelname}_fold{fold_idx}.json"
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

        Metrics are computed at fixed background efficiency (bkg_eff=1e-4 by default)
        """
        time_start = time.time()

        savedir = self.output_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        (savedir / "rocs").mkdir(parents=True, exist_ok=True)
        (savedir / "outputs").mkdir(parents=True, exist_ok=True)

        # Get predictions and labels based on n_folds
        masks = self.fold_data.get("masks", {})
        if self.n_folds == 1:
            # Single fold: predict on validation set
            y_pred = self.boosters[0].predict(self.dval)
            y_labels = self.dval.get_label()
            weights = self.dval.get_weight()
            val_idx = self.fold_data["fold_indices"][0][1]  # validation indices
            X_eval = self.fold_data["X"].iloc[val_idx]
            # Filter masks to validation indices
            masks_val = {k: v[val_idx] if v is not None else None for k, v in masks.items()}
            title_suffix = ""
        else:
            # Multi-fold: use out-of-fold predictions
            y_pred, weights, y_labels = self.get_oof_predictions()
            X_eval = self.fold_data["X"]
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

        # First do GloParT discriminants (use unbinned values for ROC analysis)
        for sig_tagger in signal_names:
            taukey = CHANNELS[sig_tagger[-2:]].tagger_label
            disc_name = f"ttFatJetParTX{taukey}vsQCDTop"  # _unbinned"
            rocAnalyzer.fill_discriminants(
                discriminant_names=[disc_name],
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
        print("METRICS SUMMARY (at bkg_eff=1e-4)")
        print("=" * 70)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("\nSignal Eff = signal efficiency at 0.01% background efficiency")
        print("=" * 70)


def study_rescaling(output_dir: str = "rescaling_study", importance_only=False) -> dict:
    """Study the impact of different rescaling rules on BDT performance.
    For now give little flexibility, but is not meant to be customized too much.

    Args:
        output_dir: Directory to save study results

    Returns:
        Dictionary containing study results for each rescaling rule
    """
    # Create output directory
    trainer = Trainer(years=["2022"], modelname="29July25_loweta_lowreg", output_dir=output_dir)

    print(f"importance_only: {importance_only}")
    if not importance_only:
        trainer.load_data(force_reload=True)

    # Define rescaling rules to study
    scale_rules = ["signal", "signal_3e-1", "signal_3"]
    balance_rules = [
        "bysample",
        "bysample_clip_1to10",
        "bysample_clip_1to20",
        "grouped_physics",
        "sqrt_scaling",
        "ens_weighting",
    ]

    results = {}

    # Store the original study directory
    study_dir = trainer.output_dir

    # Train models with different rescaling rules
    for scale_rule in scale_rules:
        if scale_rule not in results:
            results[scale_rule] = {}
        for balance_rule in balance_rules:
            try:
                print(f"\nTraining with scale_rule={scale_rule}, balance_rule={balance_rule}")

                # Create subdirectory for this configuration
                current_test_dir = study_dir / f"{scale_rule}_{balance_rule}"
                current_test_dir.mkdir(exist_ok=True)

                # Override output_dir to save in subdirectory
                trainer.output_dir = current_test_dir

                if importance_only:
                    trainer.load_model()
                else:
                    # Force reload data and train new model
                    trainer.prepare_training_set(
                        save_buffer=False, scale_rule=scale_rule, balance=balance_rule
                    )
                    trainer.train_model()
                    results[scale_rule][balance_rule] = trainer.compute_rocs(
                        savedir=current_test_dir
                    )

                trainer.evaluate_training(savedir=current_test_dir)

            except Exception as e:
                print(
                    f"Error training with scale_rule={scale_rule}, balance_rule={balance_rule}: {e}"
                )
                continue

    if not importance_only:
        _rescaling_comparison(results, study_dir)

    return results


def _rescaling_comparison(results: dict, output_dir: Path) -> None:
    """Enhanced comparison of different rescaling rules with comprehensive metrics.

    Args:
        results: Dictionary containing study results with comprehensive metrics
        output_dir: Directory to save comparison plots and tables
    """
    # Safety check in debugging
    if not isinstance(output_dir, Path):
        print(f"output_dir is not a Path, converting to Path: {output_dir}")
        output_dir = Path(output_dir)

    # Get unique scale and balance rules
    scale_rules = list(results.keys())
    balance_rules = list(results[scale_rules[0]].keys())

    # Define metrics to analyze (at fixed bkg_eff)
    metrics = {
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "signal_eff": "Signal Eff",
        "precision": "Precision",
        "f1_score": "F1",
    }

    for sig in ["hh", "he", "hm"]:
        # Create individual metric tables
        for metric_key, metric_name in metrics.items():
            table_data = []
            for scale_rule in scale_rules:
                row = [scale_rule]
                for balance_rule in balance_rules:
                    try:
                        metric_value = results[scale_rule][balance_rule]["metrics"][sig].get(
                            metric_key, 0
                        )
                        row.append(f"{metric_value:.3f}")
                    except (KeyError, TypeError):
                        row.append("-")
                table_data.append(row)

            # Print and save individual metric table
            print(f"\n{metric_name} for {sig} channel:")
            print(tabulate(table_data, headers=["Scale"] + balance_rules, tablefmt="grid"))

            with (output_dir / f"{metric_key}_{sig}.txt").open("w") as f:
                f.write(f"{metric_name} for {sig} channel:\n")
                f.write(tabulate(table_data, headers=["Scale"] + balance_rules, tablefmt="grid"))

        # Create comprehensive summary table for this signal
        summary_data = []
        for scale_rule in scale_rules:
            for balance_rule in balance_rules:
                try:
                    metrics_dict = results[scale_rule][balance_rule]["metrics"][sig]
                    summary_data.append(
                        [
                            scale_rule,
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
                    summary_data.append([scale_rule, balance_rule] + ["-"] * 6)

        # Print and save comprehensive summary
        headers = [
            "Scale",
            "Balance",
            "ROC AUC",
            "PR AUC",
            "Sig Eff",
            "Precision",
            "F1",
            "Threshold",
        ]
        print(f"\nMetrics for {sig} channel (at bkg_eff=1e-4):")
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))

        with (output_dir / f"comprehensive_{sig}.txt").open("w") as f:
            f.write(f"Metrics for {sig} channel (at bkg_eff=1e-4):\n")
            f.write(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Create cross-channel comparison for key metrics
    _create_cross_channel_comparison(results, scale_rules, balance_rules, output_dir)


def _create_cross_channel_comparison(results, scale_rules, balance_rules, output_dir):
    """Create comparison tables across all channels for key metrics."""
    key_metrics = ["roc_auc", "signal_eff", "precision", "f1_score"]
    channels = ["hh", "he", "hm"]

    for metric in key_metrics:
        print(f"\nCross-channel comparison: {metric.upper()}")

        # Create table with channels as columns
        table_data = []
        for scale_rule in scale_rules:
            for balance_rule in balance_rules:
                row = [f"{scale_rule}_{balance_rule}"]
                for sig in channels:
                    try:
                        value = results[scale_rule][balance_rule]["metrics"][sig].get(metric, 0)
                        row.append(f"{value:.3f}")
                    except (KeyError, TypeError):
                        row.append("-")
                table_data.append(row)

        headers = ["Method"] + channels
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Save to file
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


def compare_models(
    models: list[str],
    model_dirs: list[str],
    years: list[str],
    data_path: str | None = None,
    tt_preselection: bool = False,
    output_dir: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare multiple trained BDT models by computing ROC curves and metrics.

    For k-fold models, uses averaged predictions across all folds.

    Args:
        models: List of model names (configs are read from bdt_config)
        model_dirs: List of output directories corresponding to each model
        years: Years to include in the comparison
        data_path: Optional path to data directory
        tt_preselection: Whether to apply tt preselection
        output_dir: Output directory for results (default: "comparison")

    Returns:
        Nested dict: metrics_by_model[model][signal] -> metrics_dict
    """
    if len(models) != len(model_dirs):
        raise ValueError(f"models ({len(models)}) and model_dirs ({len(model_dirs)}) must match")
    if len(models) < 2:
        raise ValueError("Need at least 2 models to compare")

    base_out = Path(output_dir) if output_dir else Path("comparison")
    _ensure_dir(base_out)

    # Load data once, share across all trainers
    base_trainer = Trainer(
        years=list(years),
        modelname=models[0],
        output_dir=model_dirs[0],
        data_path=data_path,
        tt_preselection=tt_preselection,
    )
    base_trainer.load_data(force_reload=True)

    # Build all trainers, sharing loaded data
    trainers: dict[str, Trainer] = {}
    for model, model_dir in zip(models, model_dirs):
        if model == models[0]:
            tr = base_trainer
        else:
            tr = Trainer(
                years=list(years),
                modelname=model,
                output_dir=model_dir,
                data_path=data_path,
                tt_preselection=tt_preselection,
            )
            tr.events_dict = base_trainer.events_dict
            tr.samples = base_trainer.samples

        tr.prepare_training_set(save_buffer=False)
        tr.load_model()
        trainers[model] = tr

    # Validate all models have the same samples
    ref_sample_names = base_trainer.sample_names
    for model, tr in trainers.items():
        if tr.sample_names != ref_sample_names:
            raise ValueError(
                f"Model '{model}' has sample order {tr.sample_names}, "
                f"expected {ref_sample_names}"
            )

    # Get labels and weights from reference (same for all since data is shared)
    labels = base_trainer.dval.get_label()
    weights = base_trainer.dval.get_weight()

    # Initialize prediction storage with weights
    preds_dict: dict[str, LoadedSample] = {}
    for class_idx, class_name in enumerate(ref_sample_names):
        cls_mask = labels == class_idx
        preds_dict[class_name] = LoadedSample(
            sample=base_trainer.samples[class_name],
            events={"finalWeight": weights[cls_mask]},
        )

    # Collect predictions from each model
    # Note: We predict directly on tr.dval (already a DMatrix from prepare_training_set).
    for model, tr in trainers.items():
        if len(tr.boosters) > 1:
            y_pred = np.mean([bst.predict(tr.dval) for bst in tr.boosters], axis=0)
        else:
            y_pred = tr.boosters[0].predict(tr.dval)

        if y_pred.ndim != 2 or y_pred.shape[1] != len(ref_sample_names):
            raise ValueError(
                f"Model '{model}' output shape {y_pred.shape} incompatible with "
                f"{len(ref_sample_names)} classes"
            )

        # Store predictions for each class
        for class_idx, class_name in enumerate(ref_sample_names):
            mask = labels == class_idx
            preds_dict[class_name].events[f"{model}::{class_name}"] = y_pred[mask, class_idx]

    # Convert to DataFrames
    for class_name in preds_dict:
        preds_dict[class_name].events = pd.DataFrame(preds_dict[class_name].events)

    # Separate signals and backgrounds
    signal_names = [n for n in ref_sample_names if base_trainer.samples[n].isSignal]
    background_names = [n for n in ref_sample_names if not base_trainer.samples[n].isSignal]

    # Set up ROC analyzer
    roc_analyzer = ROCAnalyzer(
        years=base_trainer.years,
        signals={s: preds_dict[s] for s in signal_names},
        backgrounds={b: preds_dict[b] for b in background_names},
    )

    # Register discriminants for each (model, signal) pair
    for sig_name in signal_names:
        for model in models:
            roc_analyzer.process_discriminant(
                signal_name=sig_name,
                background_names=background_names,
                signal_tagger=f"{model}::{sig_name}",
                background_taggers=[f"{model}::{bkg}" for bkg in background_names],
                custom_name=f"{model}_{sig_name}",
            )

    roc_analyzer.compute_rocs(compute_metrics=True)

    # Plot ROCs and extract metrics
    metrics_by_model: dict[str, dict[str, dict[str, float]]] = {m: {} for m in models}
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
            # Extract model name from discriminant name (format: "{model}_{signal}")
            model_name = disc.name.rsplit("_", 1)[0]
            if model_name in models and hasattr(disc, "get_metrics"):
                metrics_by_model[model_name][sig_name] = disc.get_metrics(as_dict=True)

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
                "models": models,
                "model_dirs": model_dirs,
                "years": years,
                "signals": signal_names,
                "backgrounds": background_names,
                "kfold_models": [m for m, tr in trainers.items() if tr.use_kfold],
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
        default="28May25_baseline",
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
        help="Study the impact of different rescaling rules on BDT performance",
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
        "--models",
        nargs="+",
        default=None,
        help="List of model names to compare when --compare-models is set",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        help="List of model directories to compare when --compare-models is set",
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

    # Add mutually exclusive group for train/load
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train a new model")
    group.add_argument("--load", action="store_true", default=True, help="Load model from file")

    args = parser.parse_args()

    if args.study_rescaling:
        study_rescaling(importance_only=args.importance_only)
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
        if not args.models or len(args.models) < 2 or len(args.models) != len(args.model_dirs):
            parser.error("--compare-models requires at least two --models")
        # Validate that provided model directories and model files exist
        resolved_model_dirs = []
        for model, model_dir_str in zip(args.models, args.model_dirs):
            model_dir_path = Path(model_dir_str)
            resolved_dir = (
                model_dir_path if model_dir_path.is_absolute() else MODEL_DIR.parent / model_dir_str
            )
            if not resolved_dir.exists():
                parser.error(f"Model directory does not exist: {resolved_dir}")
            if not resolved_dir.is_dir():
                parser.error(f"Model directory is not a directory: {resolved_dir}")
            model_file = resolved_dir / f"{model}.json"
            if not model_file.is_file():
                parser.error(f"Model file not found for '{model}': {model_file}")
            resolved_model_dirs.append(str(resolved_dir))
        compare_models(
            models=args.models,
            model_dirs=resolved_model_dirs,
            years=args.years,
            data_path=args.data_path,
            tt_preselection=args.tt_preselection,
            output_dir=args.output_dir,
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

    if args.train:
        print("Running in training mode")
        trainer.complete_train(force_reload=args.force_reload)
    else:
        print("Running in load/evaluate mode")
        trainer.complete_load(force_reload=args.force_reload)
