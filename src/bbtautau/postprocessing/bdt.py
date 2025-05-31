from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from bdt_config import bdt_config
from boostedhh import hh_vars, plotting, utils
from Samples import CHANNELS, SAMPLES
from sklearn.metrics import (
    auc,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from bbtautau.postprocessing.postprocessing import (
    base_filters_default,
    bbtautau_assignment,
    derive_variables,
    get_columns,
)
from bbtautau.postprocessing.utils import LoadedSample

# TODO
# - k-fold cross validation
# - early stopping


class Trainer:

    loaded_dmatrix = False

    sample_names: ClassVar[list[str]] = [
        "qcd",
        "ttbarhad",
        "ttbarsl",
        "ttbarll",
        "dyjets",
        "bbtt",
    ]

    def __init__(self, years, sample_names=None, modelname=None, model_dir=None) -> None:
        if years[0] == "all":
            print("Using all years")
            years = ["2022", "2022EE", "2023", "2023BPix"]
        else:
            years = list(years)
        self.years = years

        if sample_names is not None:
            self.sample_names = sample_names

        self.samples = {name: SAMPLES[name] for name in self.sample_names}

        self.data_path = {
            "2022": {
                "signal": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
                ),
                "bg": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
                ),
                "data": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
                ),
            },
            "2022EE": {
                "signal": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "bg": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "data": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal/"
                ),
            },
            "2023": {
                "signal": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "bg": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "data": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal/"
                ),
            },
            "2023BPix": {
                "signal": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "bg": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
                ),
                "data": Path(
                    "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal/"
                ),
            },
        }
        self.bdt_config = bdt_config

        if modelname is not None:
            self.modelname = modelname
        else:
            self.modelname = "test"

        if model_dir is not None:
            self.model_dir = Path(
                f"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/{model_dir}"
            )
        else:
            self.model_dir = Path(
                f"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models/{self.modelname}_{('-'.join(self.years) if len(self.years) < 4 else 'all')}"
            )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, base_filters=True, force_reload=False):
        # Check if data buffer file exists
        if self.model_dir / "dtrain.buffer" in self.model_dir.glob("*.buffer") and not force_reload:
            print("Loading data from buffer file")
            self.dtrain = xgb.DMatrix(self.model_dir / "dtrain.buffer")
            self.dval = xgb.DMatrix(self.model_dir / "dval.buffer")
            for ch in CHANNELS:
                self.samples[f"bbtt{ch}"] = SAMPLES[f"bbtt{ch}"]
            del self.samples["bbtt"]
            self.loaded_dmatrix = True
        else:
            self.events_dict = {year: {} for year in self.years}
            for year in self.years:
                for key, sample in self.samples.items():
                    if sample.selector is not None:
                        sample.load_columns = get_columns(year)[sample.get_type()]
                        events = utils.load_sample(
                            sample,
                            year,
                            self.data_path[year],
                            base_filters_default if base_filters else None,
                        )
                        self.events_dict[year][key] = LoadedSample(sample=sample, events=events)
                        print(f"Successfully imported sample {sample.label} (key: {key}) to memory")
                derive_variables(
                    self.events_dict[year], CHANNELS["hm"]
                )  # legacy issue, muon branches are misnamed
                bbtautau_assignment(self.events_dict[year], agnostic=True)

            for ch, channel in CHANNELS.items():
                for year in self.years:
                    self.events_dict[year][f"bbtt{ch}"] = LoadedSample(
                        sample=SAMPLES[f"bbtt{ch}"],
                        events=self.events_dict[year]["bbtt"].events[
                            self.events_dict[year]["bbtt"].get_var(f"GenTau{ch}")
                        ],
                    )
                    bbtautau_assignment(
                        self.events_dict[year], channel
                    )  # overwrites jet assignment in signal channels
                self.samples[f"bbtt{ch}"] = SAMPLES[f"bbtt{ch}"]
            del self.samples["bbtt"]
            for year in self.years:
                del self.events_dict[year]["bbtt"]

    @staticmethod
    def shorten_df(df, N, seed=42):
        if len(df) < N:
            return df
        return df.sample(n=N, random_state=seed)

    def prepare_training_set(self, train=False, scale_rule="signal", balance="bysigbkg"):
        """Prepare features and labels using LabelEncoder for multiclass classification."""

        if scale_rule not in ["signal", "signal_1e-1", "signal_1e-2"]:
            raise ValueError(f"Invalid scale rule: {scale_rule}")

        if balance not in ["bysigbkg", "bysample_legacy", "bysample", "bysample_aggregate_ttbar"]:
            raise ValueError(f"Invalid balance rule: {balance}")

        # Get hyperparameters and training variables from config
        self.hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.train_vars = self.bdt_config[self.modelname]["train_vars"]

        # Initialize lists for features, labels, and weights
        X_list = []
        weights_list = []
        sample_names_labels = []  # Store sample names for each event

        if self.loaded_dmatrix:
            # Need to do this to keep a mapping of the sample labels
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(list(self.samples.keys()))
            self.classes = self.label_encoder.classes_
            return

        # Process each sample
        for year in self.years:
            total_signal_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][sig_sample].get_var("finalWeight"))
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
            ).sum()
            total_ttbar_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][ttbar_sample].get_var("finalWeight"))
                    for ttbar_sample in self.samples
                    if "ttbar" in ttbar_sample
                ]
            ).sum()
            total_bkg_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][bkg_sample].get_var("finalWeight"))
                    for bkg_sample in self.samples
                    if self.samples[bkg_sample].isBackground
                ]
            ).sum()

            len_signal = sum(
                [
                    len(self.events_dict[year][sig_sample].events)
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
            )
            len_bkg = sum(
                [
                    len(self.events_dict[year][bkg_sample].events)
                    for bkg_sample in self.samples
                    if self.samples[bkg_sample].isBackground
                ]
            )
            len_ttbar = sum(
                [
                    len(self.events_dict[year][ttbar_sample].events)
                    for ttbar_sample in self.samples
                    if "ttbar" in ttbar_sample
                ]
            )

            print(f"year: {year}")

            print("total_signal_weight", total_signal_weight)
            print("total_bkg_weight", total_bkg_weight)
            print("total_ttbar_weight", total_ttbar_weight)

            print("average signal weight", total_signal_weight / len_signal)
            print("average bkg weight", total_bkg_weight / len_bkg)
            print("average ttbar weight", total_ttbar_weight / len_ttbar)

            print("--------------------------------")

            for sample_name, sample in self.events_dict[year].items():

                X_sample = pd.DataFrame(
                    {
                        feat: sample.get_var(feat)
                        for feat in self.train_vars["misc"]["feats"]
                        + self.train_vars["fatjet"]["feats"]
                    }
                )

                weights = np.abs(sample.get_var("finalWeight").copy())

                # Rescale all weights based on rule
                if (
                    scale_rule == "signal"
                ):  # rescale by average signal weight, so median signal event has weight 1, .1 or .01
                    weights = weights / (total_signal_weight / len_signal)
                elif scale_rule == "signal_1e-1":
                    weights = weights / (total_signal_weight / len_signal) * 1e-1
                elif scale_rule == "signal_1e-2":
                    weights = weights / (total_signal_weight / len_signal) * 1e-2

                # Rescale each different sample. Scale such that total weight is 2*total_signal_weight to compare similar methods
                if balance == "bysample_legacy":  # tot_sig = tot_ttbar = qcd = dy
                    if sample.sample.isSignal:
                        weights = weights / 2
                    elif "ttbar" in sample_name:
                        weights = weights / total_ttbar_weight * total_signal_weight / 2.0
                    else:  # qcd, dy
                        weights = weights * total_signal_weight / np.sum(weights) / 2.0

                elif (
                    balance == "bysample"
                ):  # each of 8 samples has same weight, 1/4 of signal weight
                    weights = weights * total_signal_weight / np.sum(weights) / 4.0

                elif (
                    balance == "bysample_aggregate_ttbar"
                ):  # leave TTBar together, hh = he = hm = qcd = dy = ttbar, each has 1/3 of signal weight
                    if sample.sample.isSignal:
                        weights = weights * total_signal_weight / np.sum(weights) / 3.0
                    elif "ttbar" in sample_name:
                        weights = weights / total_ttbar_weight * total_signal_weight / 3.0
                    else:  # qcd, dy
                        weights = weights * total_signal_weight / np.sum(weights) / 3.0

                elif balance == "bysigbkg":  # tot_sig = tot_bkg
                    if sample.sample.isSignal:
                        pass
                    else:
                        weights = weights * (total_signal_weight / total_bkg_weight)

                X_list.append(X_sample)
                weights_list.append(weights)

                sample_names_labels.extend([sample_name] * len(sample.events))

        # Combine all samples
        X = pd.concat(X_list, axis=0)
        weights = np.concatenate(weights_list)

        # Use LabelEncoder to convert sample names to numeric labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(sample_names_labels)
        self.classes = self.label_encoder.classes_

        # Print class mapping
        print("\nClass mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"Class {i}: {class_name}")

        # Split into training and validation sets
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X,
            y,
            weights,
            test_size=self.bdt_config[self.modelname]["test_size"],
            random_state=self.bdt_config[self.modelname]["random_seed"],
            stratify=y,
        )

        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}")

        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train, nthread=-1)
        self.dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val, nthread=-1)

        # save buffer for quicker loading
        if train:
            self.dtrain.save_binary(self.model_dir / "dtrain.buffer")
            self.dval.save_binary(self.model_dir / "dval.buffer")

    def train_model(self):
        """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""

        # early_stopping_callback = xgb.callback.EarlyStopping(rounds=5, min_delta=0.0)
        # classifier_params = {
        #     **self.bdt_config[self.modelname]["hyperpars"],
        #     "callbacks": [early_stopping_callback],
        # }

        evals_result = {}

        evallist = [(self.dtrain, "train"), (self.dval, "eval")]
        self.bst = xgb.train(
            self.hyperpars,
            self.dtrain,
            self.bdt_config[self.modelname]["num_rounds"],
            evals=evallist,
            evals_result=evals_result,
        )
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
        plt.savefig(savedir / "training_history.png")
        plt.close()

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(self.bst, max_num_features=20)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(savedir / "feature_importance.png")
        plt.close()

    def complete_train(self, training_info=True, force_reload=False, **kwargs):
        """Train a multiclass BDT model.

        Args:
            year (str): Year of data to use
            modelname (str): Name of the model configuration to use
            save_dir (str, optional): Directory to save the model and plots. If None, uses default location.
        """

        # out-of-the-box for training
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(train=True, **kwargs)
        self.train_model(**kwargs)
        if training_info:
            self.evaluate_training()
        self.compute_rocs()

    def complete_load(self, force_reload=False, **kwargs):
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(**kwargs)
        self.load_model(**kwargs)
        self.compute_rocs()

    def compute_rocs(self, discs=None, savedir=None):

        y_pred = self.bst.predict(self.dval)

        savedir = self.model_dir if savedir is None else Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)

        # TODO: de-hardcode this
        sigs_strs = {"hh": 0, "he": 1, "hm": 2}
        bg_strs = {"QCD": [4], "all": [3, 4, 5, 6, 7]}

        # store some metrics for the various samples. for now only auc
        summary = {"auc": {sig: {} for sig in sigs_strs}}

        if not hasattr(self, "rocs"):
            self.rocs = {}
        self.rocs = {sig: {} for sig in sigs_strs}

        if discs is None:
            for sig in sigs_strs:
                for bg in bg_strs:
                    disc = f"bbtt{sig}vs{bg}"

                    bg_truth = np.isin(self.dval.get_label(), bg_strs[bg])
                    sig_truth = self.dval.get_label() == sigs_strs[sig]
                    weights = self.dval.get_weight()

                    truth = np.zeros_like(bg_truth)
                    truth[sig_truth] = 1

                    sig_score = y_pred[:, sigs_strs[sig]]
                    bg_scores = np.sum([y_pred[:, b] for b in bg_strs[bg]], axis=0)
                    disc_score = np.divide(
                        sig_score,
                        sig_score + bg_scores,
                        out=np.zeros_like(sig_score),
                        where=(sig_score + bg_scores != 0),
                    )

                    (savedir / "scores").mkdir(parents=True, exist_ok=True)
                    (savedir / "outputs").mkdir(parents=True, exist_ok=True)

                    plotting.plot_hist(
                        [disc_score[sig_truth]]
                        + [disc_score[self.dval.get_label() == b] for b in bg_strs[bg]],
                        [sig] + [self.samples[self.classes[b]].label for b in bg_strs[bg]],
                        nbins=100,
                        weights=[weights[sig_truth]]
                        + [weights[self.dval.get_label() == b] for b in bg_strs[bg]],
                        xlabel=f"BDT {disc} score",
                        lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                        density=True,
                        year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                        saveas=savedir / f"scores/bdt_scores_{sig}_{bg}.png",
                    )

                    # exclude events that are not signal or background
                    mask = bg_truth | sig_truth
                    disc_score = disc_score[mask]
                    truth = truth[mask]
                    weights_all = weights.copy()
                    weights = weights[mask]
                    fpr, tpr, thresholds = roc_curve(truth, disc_score, sample_weight=weights)
                    roc_auc = auc(fpr, tpr)

                    summary["auc"][sig][bg] = roc_auc

                    self.rocs[sig][bg] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "label": disc,
                        "auc": roc_auc,
                    }

                # Plot ROC curve
                (savedir / "rocs").mkdir(parents=True, exist_ok=True)
                plotting.multiROCCurve(
                    {"": {b: self.rocs[sig][b] for b in bg_strs}},
                    thresholds=[0.3, 0.5, 0.9, 0.99],
                    show=True,
                    plot_dir=savedir / "rocs",
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                    year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                    name=f"roc_{sig}",
                )

            # Plot BDT output score
            for i, sample in enumerate(self.classes):
                plotting.plot_hist(
                    [y_pred[self.dval.get_label() == i, _s] for _s in range(len(self.classes))],
                    [self.samples[self.classes[_s]].label for _s in range(len(self.classes))],
                    nbins=100,
                    xlim=(0, 1),
                    weights=[
                        weights_all[self.dval.get_label() == i] for _s in range(len(self.classes))
                    ],
                    xlabel=f"BDT output score on {sample}",
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                    density=True,
                    year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                    saveas=savedir / f"outputs/bdt_outputs_{sample}.png",
                )

        return summary


def study_rescaling(output_dir: str = "rescaling_study") -> dict:
    """Study the impact of different rescaling rules on BDT performance.

    Args:
        output_dir: Directory to save study results

    Returns:
        Dictionary containing study results for each rescaling rule
    """
    # Create output directory
    trainer = Trainer(years=["2022"], modelname="28May25_baseline", model_dir=output_dir)
    trainer.load_data(force_reload=True)

    # Define rescaling rules to study
    scale_rules = ["signal", "signal_1e-1", "signal_1e-2"]
    balance_rules = ["bysigbkg", "bysample_legacy", "bysample", "bysample_aggregate_ttbar"]

    results = {}

    # Train models with different rescaling rules
    for scale_rule in scale_rules:
        for balance_rule in balance_rules:
            print(f"\nTraining with scale_rule={scale_rule}, balance_rule={balance_rule}")

            # Create subdirectory for this configuration
            config_dir = trainer.model_dir / f"{scale_rule}_{balance_rule}"
            config_dir.mkdir(exist_ok=True)

            # Force reload data and train new model
            trainer.prepare_training_set(train=True, scale_rule=scale_rule, balance=balance_rule)
            trainer.train_model()
            summary = trainer.compute_rocs(savedir=config_dir)
            trainer.evaluate_training(savedir=config_dir)
            results[f"{scale_rule}_{balance_rule}"] = summary

            # Save model and plots
            trainer.bst.save_model(config_dir / "model.json")

    _rescaling_comparison(results, trainer.model_dir)
    return results


def _rescaling_comparison(results: dict, model_dir: Path) -> None:
    """comparison of different rescaling rules.

    Args:
        results: Dictionary containing study results
        study_dir: Directory to save comparison plots
    """
    # Get unique scale and balance rules
    scale_rules = list(results.keys())
    balance_rules = list(results[scale_rules[0]].keys())

    # Create a 2D table for each signal channel
    for sig in ["hh", "he", "hm"]:
        # Create 2D table data
        table_data = []
        for scale_rule in scale_rules:
            row = [scale_rule]  # First column is scale rule
            for balance_rule in balance_rules:
                # Get AUC value, or "-" if not available
                try:
                    auc_value = results[scale_rule][balance_rule]["auc"][sig]["all"]
                    row.append(f"{auc_value:.3f}")
                except KeyError:
                    row.append("-")
            table_data.append(row)

        # Print table with headers
        print(f"\nAUC scores for {sig} channel:")
        print(tabulate(table_data, headers=["Scale Rule"] + balance_rules, tablefmt="grid"))

        # Save table to file
        with (model_dir / f"auc_table_{sig}.txt").open("w") as f:
            f.write(f"AUC scores for {sig} channel:\n")
            f.write(tabulate(table_data, headers=["Scale Rule"] + balance_rules, tablefmt="grid"))


def eval_bdt_preds(eval_samples: list[str], model: str, save: bool = True):
    """Evaluate BDT predictions on data.

    Args:
        eval_samples: List of sample names to evaluate
        model: Name of model to use for predictions

    One day to be made more flexible (here only integrated with the data you already train on)
    """

    years = ["2022", "2022EE", "2023", "2023BPix"]

    trainer = Trainer(years=years, sample_names=eval_samples, modelname=model)
    trainer.load_model()
    trainer.load_data(force_reload=True)

    evals = {year: {sample_name: {} for sample_name in eval_samples} for year in years}

    for year in years:
        print(f"processing year: {year}")
        for sample_name, sample in trainer.samples.items():

            dsample = xgb.DMatrix(
                np.concatenate(
                    [
                        sample.get_var(feat)
                        for feat in trainer.train_vars["misc"]["feats"]
                        + trainer.train_vars["fatjet"]["feats"]
                    ],
                    axis=1,
                )
            )

            y_pred = trainer.bst.predict(dsample).to_numpy()
            evals[year][sample_name] = y_pred

            if save:
                # Create predictions directory if it doesn't exist
                pred_dir = (
                    trainer.data_path[year][sample.get_type()]
                    / year
                    / "BDT_predictions"
                    / sample_name
                )
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(pred_dir / f"{model}_preds.npy", y_pred)

    return evals


def parse_years(years_str):
    """Parse years input flexibly.
    Args:
        years_str: Can be:
            - "all" for all years
            - A list of years (e.g. ["2022", "2023"])
            - A single year (e.g. "2022")
    """
    if years_str == "all":
        return ["2022", "2022EE", "2023", "2023BPix"]
    elif isinstance(years_str, list):
        return years_str
    else:
        raise ValueError(f"Invalid years input: {years_str}")


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a multiclass BDT model")

    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022"],
        help="Year(s) of data to use. Can be: 'all', or multiple years (e.g. --years 2022 2023 2024)",
    )
    parser.add_argument(
        "--model", type=str, default="test", help="Name of the model configuration to use"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save model and plots"
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
        "--data-dir",
        type=str,
        help="Directory containing data to evaluate BDT predictions on",
        required="--eval-bdt-preds" in sys.argv,
    )

    # Add mutually exclusive group for train/load
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train a new model")
    group.add_argument("--load", action="store_true", default=True, help="Load model from file")

    args = parser.parse_args()

    if args.study_rescaling:
        study_rescaling()
        exit()

    if args.eval_bdt_preds:
        eval_bdt_preds(args.data_dir, args.model)
        exit()

    trainer = Trainer(args.years, args.model)

    if args.train:
        trainer.complete_train(force_reload=args.force_reload)
    else:
        trainer.complete_load(force_reload=args.force_reload)
