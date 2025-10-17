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
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from bbtautau.postprocessing.bdt_config import bdt_config
from bbtautau.postprocessing.postprocessing import (
    base_filter,
    bbtautau_assignment,
    delete_columns,
    derive_lepton_variables,
    derive_variables,
    get_columns,
    leptons_assignment,
    load_samples,
    tt_filters,
)
from bbtautau.postprocessing.rocUtils import ROCAnalyzer, multiclass_confusion_matrix
from bbtautau.postprocessing.Samples import CHANNELS, SAMPLES
from bbtautau.postprocessing.utils import LoadedSample, label_transform
from bbtautau.userConfig import CLASSIFIER_DIR, DATA_PATHS, MODEL_DIR, path_dict

# Use non-interactive backend for containerized/CLI environments
mpl.use("Agg")
plt = mpl.pyplot

# TODO
# - k-fold cross validation

# Some global variables
DATA_DIR = Path(
    "/ceph/cms/store/user/lumori/bbtautau"
)  # default directory for saving BDT predictions


class Trainer:

    loaded_dmatrix = False

    # Default samples for training / evaluation
    bkg_sample_names: ClassVar[list[str]] = ["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]

    def __init__(
        self,
        years: list[str],
        signal_key: str,
        bkg_sample_names: list[str] = None,
        modelname: str = None,
        data_path: str = None,
        model_dir: str = None,
        tt_preselection: bool = False,
    ) -> None:
        if years[0] == "all":
            print("Using all years")
            years = hh_vars.years
        else:
            years = list(years)
        self.years = years

        self.signal_key = signal_key
        if bkg_sample_names is not None:
            self.bkg_sample_names = bkg_sample_names

        # need to load signal before splitting into channels
        self.samples = {name: SAMPLES[name] for name in [self.signal_key] + self.bkg_sample_names}

        # ensure backwards compatibility. Choice of default data paths is done in userConfig.py
        self.data_paths = path_dict(data_path) if data_path is not None else DATA_PATHS

        self.modelname = modelname
        self.bdt_config = bdt_config
        self.tt_preselection = tt_preselection
        self.train_vars = self.bdt_config[self.modelname]["train_vars"]
        self.hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.feats = [feat for cat in self.train_vars for feat in self.train_vars[cat]]

        self.events_dict = {year: {} for year in self.years}

        if model_dir is not None:
            model_dir_path = Path(model_dir)
            # If absolute, use as-is; otherwise resolve under CLASSIFIER_DIR
            self.model_dir = (
                model_dir_path if model_dir_path.is_absolute() else CLASSIFIER_DIR / model_dir
            )
        else:
            self.model_dir = MODEL_DIR / self.modelname
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, force_reload=False):
        # Check if data buffer file exists
        if self.model_dir / "dtrain.buffer" in self.model_dir.glob("*.buffer") and not force_reload:
            print("Loading data from buffer file")
            self.dtrain = xgb.DMatrix(self.model_dir / "dtrain.buffer")
            self.dval = xgb.DMatrix(self.model_dir / "dval.buffer")

            print(self.model_dir)
            self.dtrain_rescaled = xgb.DMatrix(self.model_dir / "dtrain_rescaled.buffer")
            self.dval_rescaled = xgb.DMatrix(self.model_dir / "dval_rescaled.buffer")

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
                    signals=[self.signal_key],
                    channels=list(CHANNELS.values()),
                    samples=self.samples,
                    filters_dict=filters_dict,
                    load_columns=columns,
                    restrict_data_to_channel=False,
                    load_bgs=True,
                    loaded_samples=True,
                    multithread=True,
                )
                self.events_dict[year] = delete_columns(
                    self.events_dict[year], year, channels=list(CHANNELS.values())
                )

                derive_variables(self.events_dict[year])
                bbtautau_assignment(self.events_dict[year], agnostic=True)
                leptons_assignment(self.events_dict[year], dR_cut=1.5)
                derive_lepton_variables(self.events_dict[year])

        for ch in CHANNELS:
            self.samples[f"{self.signal_key}{ch}"] = SAMPLES[f"{self.signal_key}{ch}"]
        del self.samples[self.signal_key]

        # define sample list as signals + bkg samples
        self.sample_names = [f"{self.signal_key}{ch}" for ch in CHANNELS] + self.bkg_sample_names

    @staticmethod
    def save_stats(stats, filename):
        """Save weight statistics to a CSV file"""
        with Path.open(filename, "w") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)

    def prepare_training_set(
        self, save_buffer=False, scale_rule="signal", balance="bysample_clip_1to10"
    ):
        """Prepare features and labels using LabelEncoder for multiclass classification.

        Args:
            train (bool, optional): Whether to prepare data for training. If true, will save a buffer file with training and eval files for quicker loading. Defaults to False.
            scale_rule (str, optional): Rule for global scaling weights. Can be 'signal' (average signal event weight = 1), 'signal_1e-1', or 'signal_1e-2'. Defaults to 'signal'.
            balance (str, optional): Rule for balancing samples. Can be
            - 'bysample' : each of 8 samples has same weight, 1/4 of signal weight
            - 'bysample_clip_1to10' : like bysample but clip min avg weight to 1/10 of signal avg
            - 'bysample_clip_1to20' : like bysample but clip min avg weight to 1/20 of signal avg
            - 'grouped_physics' : balance by physics groups (signal, ttbar, other_bkg) equally
            - 'sqrt_scaling' : total weight proportional to sqrt(number of events)
            - 'ens_weighting' : effective number of samples weighting (beta=0.999)
            Defaults to 'bysample_clip_1to10'.
            Total weight = 8*tot_sig
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

        # Initialize lists for features, labels, and weights
        X_list = []
        weights_list = []
        weights_rescaled_list = []
        sample_names_labels = []  # Store sample names for each event

        if self.loaded_dmatrix:
            # legacy way to handle this case, used to have to execute some code.
            return

        # Store weight statistics aggregated across all years
        weight_stats_by_stage_sample = {}

        # Process each sample
        for year in self.years:

            # Store weights for rescaling purposes
            total_signal_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][sig_sample].get_var("finalWeight"))
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
            ).sum()

            len_signal = sum(
                [
                    len(self.events_dict[year][sig_sample].events)
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
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
                weight_stats_by_stage_sample[key].append(weights_rescaled)

                global_scale_factor = {
                    "signal": 1.0,
                    "signal_3e-1": 3e-1,
                    "signal_3": 3,
                }

                # rescale by average signal weight, so average signal event has weight 1, .3 or 3
                weights_rescaled = (
                    weights_rescaled / avg_signal_weight * global_scale_factor[scale_rule]
                )

                # now total_signal_weight = len_signal * global_scale_factor[scale_rule]

                # Aggregate for multi-year stats
                key = ("Global rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled)

                # Rescale each different sample.
                if (
                    balance == "bysample"
                ):  # each of 8 samples has same weight, equal to signal weight per channel
                    weights_rescaled = (
                        weights_rescaled / np.sum(weights_rescaled) * len_signal_per_channel
                    )
                elif balance == "bysample_clip_1to10":
                    n_events = len(sample.events)
                    # Set the initial target total weight for this sample to be equal to that of the average signal samples
                    target_total_weight = len_signal_per_channel * global_scale_factor[scale_rule]

                    # Calculate what the average weight of this sample would become
                    avg_weight_if_scaled = target_total_weight / n_events

                    # Define the clipping threshold: 1/10th of the average signal weight
                    min_avg_weight = global_scale_factor[scale_rule] / 10.0

                    # If the potential average weight is too small, cap the target total weight
                    if avg_weight_if_scaled < min_avg_weight:
                        target_total_weight = min_avg_weight * n_events

                    # Calculate the final scaling factor and apply it
                    scaling_factor = target_total_weight / np.sum(weights_rescaled)
                    weights_rescaled = weights_rescaled * scaling_factor

                elif balance == "bysample_clip_1to20":
                    n_events = len(sample.events)
                    # Set the initial target total weight for this sample to be equal to other samples
                    target_total_weight = len_signal_per_channel * global_scale_factor[scale_rule]

                    # Calculate what the average weight of this sample would become
                    avg_weight_if_scaled = target_total_weight / n_events

                    # Define the clipping threshold: 1/20th of the average signal weight
                    min_avg_weight = global_scale_factor[scale_rule] / 20.0

                    # If the potential average weight is too small, cap the target total weight
                    if avg_weight_if_scaled < min_avg_weight:
                        target_total_weight = min_avg_weight * n_events

                    # Calculate the final scaling factor and apply it
                    scaling_factor = target_total_weight / np.sum(weights_rescaled)
                    weights_rescaled = weights_rescaled * scaling_factor

                elif balance == "grouped_physics":
                    # Group samples by physics process for balanced reweighting
                    # Calculate group-wise scaling factors
                    if sample.sample.isSignal:
                        # Each signal sample gets weight equal to len_signal_per_channel
                        target_total_weight = (
                            len_signal_per_channel * global_scale_factor[scale_rule]
                        )
                    elif "ttbar" in sample_name:
                        # TTbar group: calculate total events in ttbar group
                        ttbar_total_events = sum(
                            [
                                len(self.events_dict[year][s].events)
                                for s in self.samples
                                if "ttbar" in s and s in self.events_dict[year]
                            ]
                        )
                        ttbar_fraction = len(sample.events) / ttbar_total_events
                        # TTbar group gets same total weight as one signal sample, distributed proportionally
                        target_total_weight = (
                            len_signal_per_channel * global_scale_factor[scale_rule]
                        ) * ttbar_fraction
                    else:
                        # Other backgrounds group
                        other_total_events = sum(
                            [
                                len(self.events_dict[year][s].events)
                                for s in self.samples
                                if not self.samples[s].isSignal
                                and "ttbar" not in s
                                and s in self.events_dict[year]
                            ]
                        )
                        other_fraction = (
                            len(sample.events) / other_total_events
                            if other_total_events > 0
                            else 1.0
                        )
                        # Other bkg group gets same total weight as one signal sample, distributed proportionally
                        target_total_weight = (
                            len_signal_per_channel * global_scale_factor[scale_rule]
                        ) * other_fraction

                    scaling_factor = target_total_weight / np.sum(weights_rescaled)
                    weights_rescaled = weights_rescaled * scaling_factor

                elif balance == "sqrt_scaling":
                    # Scale total weight proportional to sqrt(number of events)
                    n_events = len(sample.events)
                    sqrt_factor = np.sqrt(n_events)
                    # Normalize: signal samples should get len_signal_per_channel weight
                    # Other samples get weight proportional to sqrt(events) relative to sqrt(len_signal_per_channel)
                    target_total_weight = (
                        sqrt_factor
                        * np.sqrt(len_signal_per_channel)
                        * global_scale_factor[scale_rule]
                    )
                    scaling_factor = target_total_weight / np.sum(weights_rescaled)
                    weights_rescaled = weights_rescaled * scaling_factor

                elif balance == "ens_weighting":
                    # Effective Number of Samples weighting with beta=0.999
                    n_events = len(sample.events)
                    beta = 0.999
                    # Weight inversely proportional to ENS (class-balanced loss approach)
                    cb_weight = (1 - beta) / (1 - beta**n_events)
                    # Normalize: signal samples should get len_signal_per_channel weight
                    # Other samples get weight proportional to their CB weight relative to signal CB weight
                    signal_cb_weight = (1 - beta) / (1 - beta**len_signal_per_channel)
                    target_total_weight = (
                        (cb_weight / signal_cb_weight)
                        * len_signal_per_channel
                        * global_scale_factor[scale_rule]
                    )
                    scaling_factor = target_total_weight / np.sum(weights_rescaled)
                    weights_rescaled = weights_rescaled * scaling_factor

                # Aggregate for multi-year stats
                key = ("Balance rescaling", sample.sample.label)
                if key not in weight_stats_by_stage_sample:
                    weight_stats_by_stage_sample[key] = []
                weight_stats_by_stage_sample[key].append(weights_rescaled)

                X_list.append(X_sample)
                weights_list.append(weights)
                weights_rescaled_list.append(weights_rescaled)

                sample_names_labels.extend([sample_name] * len(sample.events))

        # Create aggregated stats across all years
        weight_stats = []
        for (stage, sample), weights_for_sample in weight_stats_by_stage_sample.items():
            # Concatenate weights from all years for this stage and sample
            all_weights = np.concatenate(weights_for_sample)
            weight_stats.append(
                {
                    "stage": stage,
                    "sample": sample,
                    "n_events": len(all_weights),
                    "total_weight": np.sum(all_weights),
                    "average_weight": np.mean(all_weights),
                    "std_weight": np.std(all_weights),
                }
            )

        # Save only the aggregated stats
        self.save_stats(weight_stats, self.model_dir / "weight_stats.csv")

        # Combine all samples
        X = pd.concat(X_list, axis=0)
        weights = np.concatenate(weights_list)
        weights_rescaled = np.concatenate(weights_rescaled_list)

        y = label_transform(self.sample_names, sample_names_labels)

        # Print class mapping
        print("\nClass mapping:")
        for i, class_name in enumerate(self.sample_names):
            print(f"Class {i}: {class_name}")

        # Split into training and validation sets for training and training evaluation
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X,
            y,
            weights_rescaled,
            test_size=self.bdt_config[self.modelname]["test_size"],
            random_state=self.bdt_config[self.modelname]["random_seed"],
            stratify=y,
        )

        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}")

        # Create DMatrix objects
        self.dtrain_rescaled = xgb.DMatrix(X_train, label=y_train, weight=weights_train, nthread=8)
        self.dval_rescaled = xgb.DMatrix(X_val, label=y_val, weight=weights_val, nthread=8)

        # Split into training and validation sets for all other purposes, e.g. computing rocs
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X,
            y,
            weights,
            test_size=self.bdt_config[self.modelname]["test_size"],
            random_state=self.bdt_config[self.modelname]["random_seed"],
            stratify=y,
        )

        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train, nthread=8)
        self.dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val, nthread=8)

        # save buffer for quicker loading
        if save_buffer:
            self.dtrain.save_binary(self.model_dir / "dtrain.buffer")
            self.dval.save_binary(self.model_dir / "dval.buffer")
            self.dtrain_rescaled.save_binary(self.model_dir / "dtrain_rescaled.buffer")
            self.dval_rescaled.save_binary(self.model_dir / "dval_rescaled.buffer")

    def train_model(self, save=True, early_stopping_rounds=5):
        """Train model using configured hyperparameters and evaluation sets."""

        evals_result = {}

        evallist = [(self.dtrain_rescaled, "train"), (self.dval_rescaled, "eval")]
        self.bst = xgb.train(
            self.hyperpars,
            self.dtrain_rescaled,
            self.bdt_config[self.modelname]["num_rounds"],
            evals=evallist,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
        )
        if save:
            self.bst.save_model(self.model_dir / f"{self.modelname}.json")

        # Save evaluation results as JSON
        with (self.model_dir / "evals_result.json").open("w") as f:
            json.dump(evals_result, f, indent=2)

        return

    def load_model(self):
        self.bst = xgb.Booster()
        print(f"loading model {self.modelname}")
        try:
            self.bst.load_model(self.model_dir / f"{self.modelname}.json")
            print("loading successful")
        except Exception as e:
            print(e)
        return self.bst

    def evaluate_training(self, savedir=None):
        """Plot training curves and feature importances from saved eval results."""
        # Load evaluation results from JSON
        with (self.model_dir / "evals_result.json").open("r") as f:
            evals_result = json.load(f)

        savedir = self.model_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        plt.plot(evals_result["train"][self.hyperpars["eval_metric"]], label="Train")
        plt.plot(evals_result["eval"][self.hyperpars["eval_metric"]], label="Validation")
        plt.xlabel("Iteration")
        plt.ylabel(self.hyperpars["eval_metric"])
        plt.tight_layout()
        plt.legend()
        plt.savefig(savedir / "training_history.pdf")
        plt.savefig(savedir / "training_history.png")
        plt.close()

        # Create triple plot for feature importance
        importance_types = ["weight", "gain", "total_gain"]
        titles = [
            "Feature Importance (Weight)",
            "Feature Importance (Gain)",
            "Feature Importance (Total Gain)",
        ]

        try:
            for imp_type, title in zip(importance_types, titles):
                plt.figure(figsize=(10, 8))
                ax = plt.gca()
                xgb.plot_importance(
                    self.bst, importance_type=imp_type, ax=ax, values_format="{v:.2f}"
                )
                ax.set_title(title)

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
        """Compute and plot ROCs and summary metrics for the validation set."""

        time_start = time.time()

        y_pred = self.bst.predict(self.dval)

        time_end = time.time()
        print(f"Time taken to predict: {time_end - time_start} seconds")

        savedir = self.model_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        (savedir / "rocs").mkdir(parents=True, exist_ok=True)
        (savedir / "outputs").mkdir(parents=True, exist_ok=True)

        # "taggers" just indicates the branch names in the event df, and here they are the sample names
        signal_names = [sig_name for sig_name in self.samples if self.samples[sig_name].isSignal]
        background_names = [
            bkg_name for bkg_name in self.samples if not self.samples[bkg_name].isSignal
        ]

        print("signal_names", signal_names)
        print("background_names", background_names)

        event_filters = {
            name: self.dval.get_label() == i for i, name in enumerate(self.sample_names)
        }
        dval_df = pd.DataFrame(self.dval.get_data().toarray(), columns=self.feats)

        preds_dict = {}
        for class_name in self.sample_names:
            events = {}
            for i, pred_class_name in enumerate(self.sample_names):
                events[pred_class_name] = y_pred[event_filters[class_name], i]
            events["finalWeight"] = self.dval.get_weight()[event_filters[class_name]]
            events = pd.DataFrame(events)

            events = pd.concat(
                [
                    dval_df[event_filters[class_name]].reset_index(drop=True),
                    events.reset_index(drop=True),
                ],
                axis=1,
            )
            preds_dict[class_name] = LoadedSample(sample=self.samples[class_name], events=events)

        multiclass_confusion_matrix(preds_dict, plot_dir=savedir)

        rocAnalyzer = ROCAnalyzer(
            years=self.years,
            signals={sig: preds_dict[sig] for sig in signal_names},
            backgrounds={bkg: preds_dict[bkg] for bkg in background_names},
        )

        #########################################################
        #########################################################
        # This part configures what background outputs to put in the taggers

        # First do ParT taggers
        parT_bkg_taggers = [["ttFatJetParTQCD", "ttFatJetParTTop"]]  # ["ttFatJetParTQCD"],

        for sig_tagger in signal_names:
            taukey = CHANNELS[sig_tagger[-2:]].tagger_label
            parT_sig = f"ttFatJetParTX{taukey}"
            for bkg_taggers in parT_bkg_taggers:
                rocAnalyzer.process_discriminant(
                    signal_name=sig_tagger,
                    background_names=background_names,
                    signal_tagger=parT_sig,
                    background_taggers=bkg_taggers,
                    custom_name=f"ParT {sig_tagger[-2:]}vsQCDTop",
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
                name = (
                    f"BDT {sig_tagger[-2:]}vsAll"
                    if len(bkg_taggers) == 5
                    else f"BDT {sig_tagger[-2:]}vsQCDTop"
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

        rocAnalyzer.compute_rocs()

        # Initialize results structure
        eval_results = {"metrics": {}}

        for sig, discs in discs_by_sig.items():
            disc_names = [disc.name for disc in discs]
            print("Plotting ROCs for", disc_names)
            print(discs)
            rocAnalyzer.plot_rocs(title=f"BDT {sig}", disc_names=disc_names, plot_dir=savedir)

            for disc in discs:
                rocAnalyzer.plot_disc_scores(
                    disc.name, [[bkg] for bkg in background_names], savedir
                )
                try:
                    rocAnalyzer.compute_confusion_matrix(disc.name, plot_dir=savedir)
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
            weights_all = self.dval.get_weight()
            for i, sample in enumerate(self.sample_names):
                plotting.plot_hist(
                    [
                        y_pred[self.dval.get_label() == i, _s]
                        for _s in range(len(self.sample_names))
                    ],
                    [
                        self.samples[self.sample_names[_s]].label
                        for _s in range(len(self.sample_names))
                    ],
                    nbins=100,
                    xlim=(0, 1),
                    weights=[
                        weights_all[self.dval.get_label() == i]
                        for _s in range(len(self.sample_names))
                    ],
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

        # Prepare data for table
        headers = [
            "Signal",
            "ROC AUC",
            "PR AUC",
            "F1 (opt)",
            "Precision (opt)",
            "Recall (opt)",
            "F1 (0.5)",
            "Balanced Acc",
            "MCC",
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
                f"{metrics.get('f1_score', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                f"{metrics.get('f1_score_05', 0):.3f}",
                f"{metrics.get('balanced_accuracy', 0):.3f}",
                f"{metrics.get('matthews_corr', 0):.3f}",
                f"{metrics.get('optimal_threshold', 0.5):.3f}",
            ]
            rows.append(row)

        print("\n" + "=" * 80)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("\nLegend:")
        print("- (opt): Metrics at optimal threshold (maximizing F1)")
        print("- (0.5): Metrics at fixed 0.5 threshold")
        print("- MCC: Matthews Correlation Coefficient")
        print("=" * 80)


def study_rescaling(model_dir: str = "rescaling_study", importance_only=False) -> dict:
    """Study the impact of different rescaling rules on BDT performance.
    For now give little flexibility, but is not meant to be customized too much.

    Args:
        model_dir: Directory to save study results

    Returns:
        Dictionary containing study results for each rescaling rule
    """
    # Create output directory
    trainer = Trainer(years=["2022"], modelname="29July25_loweta_lowreg", model_dir=model_dir)

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
    study_dir = trainer.model_dir

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

                # Override model_dir to save in subdirectory
                trainer.model_dir = current_test_dir

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


def _rescaling_comparison(results: dict, model_dir: Path) -> None:
    """Enhanced comparison of different rescaling rules with comprehensive metrics.

    Args:
        results: Dictionary containing study results with comprehensive metrics
        model_dir: Directory to save comparison plots and tables
    """
    # Safety check in debugging
    if not isinstance(model_dir, Path):
        print(f"model_dir is not a Path, converting to Path: {model_dir}")
        model_dir = Path(model_dir)

    # Get unique scale and balance rules
    scale_rules = list(results.keys())
    balance_rules = list(results[scale_rules[0]].keys())

    # Define metrics to analyze
    metrics = {
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "f1_score": "F1 (optimal)",
        "precision": "Precision (optimal)",
        "recall": "Recall (optimal)",
        "f1_score_05": "F1 (0.5)",
        "balanced_accuracy": "Balanced Accuracy",
        "matthews_corr": "Matthews Corr",
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

            with (model_dir / f"{metric_key}_{sig}.txt").open("w") as f:
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
                            f"{metrics_dict.get('f1_score', 0):.3f}",
                            f"{metrics_dict.get('precision', 0):.3f}",
                            f"{metrics_dict.get('recall', 0):.3f}",
                            f"{metrics_dict.get('f1_score_05', 0):.3f}",
                            f"{metrics_dict.get('balanced_accuracy', 0):.3f}",
                            f"{metrics_dict.get('matthews_corr', 0):.3f}",
                            f"{metrics_dict.get('optimal_threshold', 0.5):.3f}",
                        ]
                    )
                except (KeyError, TypeError):
                    summary_data.append([scale_rule, balance_rule] + ["-"] * 9)

        # Print and save comprehensive summary
        headers = [
            "Scale",
            "Balance",
            "ROC AUC",
            "PR AUC",
            "F1 (opt)",
            "Prec (opt)",
            "Rec (opt)",
            "F1 (0.5)",
            "Bal Acc",
            "MCC",
            "Threshold",
        ]
        print(f"\nComprehensive metrics for {sig} channel:")
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))

        with (model_dir / f"comprehensive_{sig}.txt").open("w") as f:
            f.write(f"Comprehensive metrics for {sig} channel:\n")
            f.write(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Create cross-channel comparison for key metrics
    _create_cross_channel_comparison(results, scale_rules, balance_rules, model_dir)


def _create_cross_channel_comparison(results, scale_rules, balance_rules, model_dir):
    """Create comparison tables across all channels for key metrics."""
    key_metrics = ["roc_auc", "f1_score", "precision", "recall"]
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
        with (model_dir / f"cross_channel_{metric}.txt").open("w") as f:
            f.write(f"Cross-channel comparison: {metric.upper()}\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))


def eval_bdt_preds(
    years: list[str],
    samples: list[str],
    model: str,
    signal_key: str,
    save: bool = True,
    model_dir: str | None = None,
    data_path: str | None = None,
    tt_preselection: bool = False,
):
    """Evaluate BDT predictions on data.

    Args:
        eval_samples: List of sample names to evaluate
        model: Name of model to use for predictions

    One day to be made more flexible (here only integrated with the data you already train on)
    """

    years = hh_vars.years if years[0] == "all" else list(years)

    if samples[0] == "all":
        samples = list(SAMPLES.keys())

    if save:
        if model_dir is None:
            model_dir = DATA_DIR

        # check if model_dir is writable
        if not os.access(model_dir, os.W_OK):
            raise PermissionError(f"Directory {model_dir} is not writable")

    # Load model globally for all years, evaluate by year to reduce memory usage
    bst = Trainer(
        years=years,
        signal_key=signal_key,
        bkg_sample_names=samples,
        modelname=model,
        data_path=data_path,
        tt_preselection=tt_preselection,
    ).load_model()

    evals = {year: {sample_name: {} for sample_name in samples} for year in years}

    for year in years:

        # To reduce memory usage, load data once for each year
        trainer = Trainer(
            years=[year],
            signal_key=signal_key,
            bkg_sample_names=samples,
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
                pred_dir = Path(model_dir) / "BDT_predictions" / year / sample_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / f"{model}_preds.npy", y_pred)
                with Path.open(pred_dir / f"{model}_preds_shape.txt", "w") as f:
                    f.write(str(y_pred.shape) + "\n")

            print(f"Processed sample {sample_name} for year {year}")

        del trainer

    return evals


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
        "--signal-key",
        type=str,
        default="ggfbbtt",
        help="Key for the signal sample",
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
                signal_key=args.signal_key,
                model_dir=args.model_dir,
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
                model_dir_path if model_dir_path.is_absolute() else CLASSIFIER_DIR / model_dir_str
            )
            if not resolved_dir.exists():
                parser.error(f"Model directory does not exist: {resolved_dir}")
            if not resolved_dir.is_dir():
                parser.error(f"Model directory is not a directory: {resolved_dir}")
            model_file = resolved_dir / f"{model}.json"
            if not model_file.is_file():
                parser.error(f"Model file not found for '{model}': {model_file}")
            resolved_model_dirs.append(str(resolved_dir))
        # Lazy import to avoid circular import issues
        from bbtautau.postprocessing import bdt_utils as _bdt_utils

        _bdt_utils.compare_models(
            models=args.models,
            model_dirs=resolved_model_dirs,
            years=args.years,
            signal_key=args.signal_key,
            samples=args.samples,
            data_path=args.data_path,
            tt_preselection=args.tt_preselection,
            output_dir=args.output_dir,
        )
        exit()

    trainer = Trainer(
        years=args.years,
        bkg_sample_names=args.samples,
        modelname=args.model,
        model_dir=args.model_dir,
        data_path=args.data_path,
        tt_preselection=args.tt_preselection,
        signal_key=args.signal_key,
    )

    if args.train:
        trainer.complete_train(force_reload=args.force_reload)
    else:
        trainer.complete_load(force_reload=args.force_reload)
