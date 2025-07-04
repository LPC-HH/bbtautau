"""
ROC utilities for postprocessing.

Author: Ludovico Mori

This module provides classes and functions to compute, store, and analyze ROC curves and related metrics for signal/background discrimination in HEP analyses. It includes utilities for constructing discriminants from raw or precomputed scores, parallelized ROC computation, and plotting tools for visualizing performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from boostedhh import hh_vars
from boostedhh.utils import PAD_VAL
from joblib import Parallel, delayed
from sklearn.metrics import auc, roc_curve
from utils import LoadedSample, rename_jetbranch_ak8

from bbtautau.postprocessing.plotting import plotting
from bbtautau.postprocessing.Samples import SAMPLES


@dataclass
class ROC:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    label: str
    auc: float


@dataclass
class Discriminant:
    """
    Container for discriminant scores and associated metadata.

    Attributes:
        name (str): Name of the discriminant.
        disc_scores (np.ndarray): Array of discriminant scores for all events (signal and background).
        weights (np.ndarray): Event weights, typically for MC normalization or event reweighting.
        disc_labels (np.ndarray): Binary labels (0 for background, 1 for signal) for each event.
        bkg_labels (np.ndarray): Names or identifiers for background samples/events.

    Methods:
        from_raw_scores: Construct a discriminant from raw signal/background tagger scores and weights.
        from_disc_scores: Construct a discriminant from precomputed discriminant scores and weights.
        get_discriminant_score: Return the array of discriminant scores.
        get_labels: Return the array of binary labels.
        get_weights: Return the array of event weights.
        get_bkg_labels: Return the array of background sample labels.
        get_name: Return the name of the discriminant.



    Note:
    Right now the name does not identify the signal and background samples. If we want to plot the ROC curves of same discriminant but evaluated on different signal and background samples, we need to add a structure to identify the signal and background samples. Previously we had a nested dictionary structure, like self.discriminants[signal_name][''.join(background_names)][disc_name], but was not super clean and so far not useful.
    """

    name: str  # name of the discriminant
    disc_scores: np.ndarray  # discriminant scores for each event.
    weights: (
        np.ndarray
    )  # weights for each event, useful for histogramming and computing efficiencies
    binary_labels: np.ndarray  # 0 for bkg or 1 for signal
    extended_labels: np.ndarray  # like binary labels but with names of samples
    signal_name: str  # name of the signal sample
    bkg_names: list[str]  # names of background samples

    @classmethod
    def from_raw_scores(
        cls,
        name,
        s_scores_sig,
        b_scores_sig,
        s_scores_bkg,
        b_scores_bkg,
        weights_sig,
        weights_bkg,
        extended_labels,
        signal_name,
        bkg_names,
    ):
        """
        Construct a discriminant from raw tagger scores and weights.

        Args:
            s_scores_sig (np.ndarray): Signal tagger scores for signal events.
            b_scores_sig (np.ndarray): Background tagger scores for signal events.
            s_scores_bkg (np.ndarray): Signal tagger scores for background events.
            b_scores_bkg (np.ndarray): Background tagger scores for signal events.
            weights_sig (np.ndarray): Weights for signal events.
            weights_bkg (np.ndarray): Weights for background events.
            bkg_labels (list[str]): Labels/names for background events.

        Returns:
            discriminant: Instance with computed discriminant scores and metadata.
        """
        # Compute signal and background discriminant scores as S/(S+B)
        disc_scores_sig = s_scores_sig / (s_scores_sig + b_scores_sig)
        disc_scores_bkg = s_scores_bkg / (s_scores_bkg + b_scores_bkg)
        disc_scores = np.concatenate([disc_scores_sig, disc_scores_bkg])
        binary_labels = np.concatenate([np.ones_like(s_scores_sig), np.zeros_like(s_scores_bkg)])
        weights = np.concatenate([weights_sig, weights_bkg])
        return cls(
            name, disc_scores, weights, binary_labels, extended_labels, signal_name, bkg_names
        )

    @classmethod
    def from_disc_scores(
        cls,
        name,
        disc_sig,
        disc_bkg,
        weights_sig,
        weights_bkg,
        extended_labels,
        signal_name,
        bkg_names,
    ):
        """
        Construct a discriminant from precomputed discriminant scores and weights.

        Args:
            disc_sig (np.ndarray): discriminant scores for signal events.
            disc_bkg (np.ndarray): discriminant scores for background events.
            weights_sig (np.ndarray): Weights for signal events.
            weights_bkg (np.ndarray): Weights for background events.
            bkg_labels (list[str]): Labels/names for background events.

        Returns:
            discriminant: Instance with provided scores and metadata.
        """
        disc_scores = np.concatenate([disc_sig, disc_bkg])
        binary_labels = np.concatenate([np.ones_like(disc_sig), np.zeros_like(disc_bkg)])
        weights = np.concatenate([weights_sig, weights_bkg])
        return cls(
            name, disc_scores, weights, binary_labels, extended_labels, signal_name, bkg_names
        )

    def compute_roc(self):
        """
        Compute the ROC curve and AUC for a given discriminant, and store it as an attribute.
        """
        fpr, tpr, thresholds = roc_curve(
            self.get_binary_labels(),
            self.get_discriminant_score(),
            sample_weight=self.get_weights(),
        )
        roc_auc = auc(fpr, tpr)
        self.roc = ROC(fpr, tpr, thresholds, rename_jetbranch_ak8(self.get_name()), roc_auc)

    def get_discriminant_score(self):
        return self.disc_scores

    def get_binary_labels(self):
        return self.binary_labels

    def get_weights(self):
        return self.weights

    def get_extended_labels(self):
        return self.extended_labels

    def get_signal_name(self):
        return self.signal_name

    def get_bkg_names(self):
        return self.bkg_names

    def get_name(self):
        return self.name

    def get_roc(self, unpacked=False):
        if not hasattr(self, "roc"):
            print(f"Warning: ROC curve not computed for discriminant {self.get_name()}")
            return None
        if unpacked:  # return a dict with the attributes as keys
            return {
                "fpr": self.roc.fpr,
                "tpr": self.roc.tpr,
                "thresholds": self.roc.thresholds,
                "label": self.roc.label,
                "auc": self.roc.auc,
            }
        else:
            return self.roc


class ROCAnalyzer:
    """
    Class for managing and analyzing multiple discriminants and their ROC curves.

    Attributes:
        years (list[str]): List of years for the analysis (e.g., ["2022", "2023"]).
        signal_taggers (list[str]): List of tagger names used for signal discrimination.
        background_taggers (list[str]): List of tagger names used for background discrimination.
        signals (dict[str, LoadedSample]): Dictionary of signal sample names to LoadedSample objects.
        backgrounds (dict[str, LoadedSample]): Dictionary of background sample names to LoadedSample objects.
        discriminants (dict): Nested dictionary to store discriminant objects for each signal/background combination.
        rocs (dict): Dictionary to store computed ROC curve results for each discriminant.

    Methods:
        process_discriminant: Compute and store a new discriminant from raw tagger scores.
        fill_discriminant: Fill an existing discriminant from precomputed scores.
        compute_roc: Static method to compute ROC curve and AUC for a given discriminant.
        compute_rocs: Compute ROC curves for all stored discriminants in parallel.
        plot_rocs: Plot ROC curves for a set of discriminants.
        compute_confusion_matrix: Compute and plot a confusion matrix for a given discriminant and threshold.
    """

    # Probably remove the attributes containing loadedsamples. and introduce alternative constructor for scores directly.

    def __init__(
        self,
        years: list[str],
        signals: dict[str, LoadedSample],
        backgrounds: dict[str, LoadedSample],
    ):
        """
        Initialize the ROCAnalyzer.

        Args:
            years (list[str]): List of years for the analysis.
            signal_taggers (list[str]): List of tagger names for signal.
            background_taggers (list[str]): List of tagger names for background.
            signals (dict[str, LoadedSample]): Dictionary of signal samples.
            backgrounds (dict[str, LoadedSample]): Dictionary of background samples.
        """
        self.years = years
        self.signals = signals
        self.backgrounds = backgrounds

        self.discriminants = {}

    def process_discriminant(
        self,
        signal_name: str,  # name of the signal sample
        background_names: list[str],  # names of the background samples
        signal_tagger: str,  # name of the signal tagger
        background_taggers: list[
            str
        ],  # names of the background taggers to include in the discriminant
        prefix: str = "",
    ):
        """
        Compute a discriminant from scratch using tagger scores and store it in the discriminants dict.

        Args:
            signal_name (str): Name of the signal sample.
            background_names (list[str]): Names of the background samples.
            signal_tagger (str): Name of the signal tagger.
            background_taggers (list[str]): Names of the background taggers to include in the discriminant.
            prefix (str, optional): Prefix to add to the discriminant name.
        """

        # Compute and store signal and background scores for the signal samples
        s_scores_sig = self.signals[signal_name].get_var(signal_tagger, pad_nan=True)
        b_scores_sig = np.sum(
            [
                self.signals[signal_name].get_var(tagger, pad_nan=True)
                for tagger in background_taggers
            ],
            axis=0,
        )

        # Compute and store signal and background scores for the background samples
        s_scores_bkg = np.concatenate(
            [self.backgrounds[bg].get_var(signal_tagger, pad_nan=True) for bg in background_names]
        )
        b_scores_bkg = np.concatenate(
            [
                np.sum(
                    [
                        self.backgrounds[bg].get_var(tagger, pad_nan=True)
                        for tagger in background_taggers
                    ],
                    axis=0,
                )
                for bg in background_names
            ]
        )

        bkg_labels = np.concatenate(
            [[bg] * len(self.backgrounds[bg].get_var(signal_tagger)) for bg in background_names]
        )
        extended_labels = np.concatenate([[signal_name] * len(s_scores_sig), bkg_labels])

        weights_sig = self.signals[signal_name].get_var("finalWeight", pad_nan=True)
        weights_bkg = np.concatenate(
            [self.backgrounds[bg].get_var("finalWeight", pad_nan=True) for bg in background_names]
        )

        bg_str = "".join(
            [
                SAMPLES[bg].label.replace(" ", "").replace("Multijet", "")
                for bg in background_taggers
            ]
        ).replace("TTHadTTLLTTSL", "Top")
        disc_name = f"{prefix}{signal_tagger}vs{bg_str}"

        # Store the new discriminant object
        self.discriminants[disc_name] = Discriminant.from_raw_scores(
            disc_name,
            s_scores_sig,
            b_scores_sig,
            s_scores_bkg,
            b_scores_bkg,
            weights_sig,
            weights_bkg,
            extended_labels,
            signal_name,
            background_names,
        )

    def fill_discriminants(
        self,
        discriminant_names: list[str],  # name of the discriminants to fill
        signal_name: str,  # name of the signal sample
        background_names: list[str],  # names of the background samples
    ):
        """
        Fill an existing discriminant in the discriminants dict using precomputed scores.

        Args:
            discriminant_names (list[str]): Names of the discriminants to fill.
            signal_name (str): Name of the signal sample.
            background_names (list[str]): Names of the background samples.
        """
        for disc_name in discriminant_names:
            # Compute and store signal and background scores for the signal samples
            try:
                disc_sig = self.signals[signal_name].get_var(disc_name, pad_nan=True)
                disc_bkg = np.concatenate(
                    [
                        self.backgrounds[bg].get_var(disc_name, pad_nan=True)
                        for bg in background_names
                    ]
                )
            except:
                # TODO could do fallback on process_discriminant but needs string interpretation, which could be ambiguous. save for later.
                print(
                    f"\n WARNING: discriminant {disc_name} not found for signal {signal_name} and backgrounds {background_names}\n"
                )
                continue

            bkg_labels = np.concatenate(
                [[bg] * len(self.backgrounds[bg].get_var(disc_name)) for bg in background_names]
            )
            extended_labels = np.concatenate([[signal_name] * len(disc_sig), bkg_labels])

            weights_sig = self.signals[signal_name].get_var("finalWeight", pad_nan=True)
            weights_bkg = np.concatenate(
                [
                    self.backgrounds[bg].get_var("finalWeight", pad_nan=True)
                    for bg in background_names
                ]
            )

            # print fraction of background that is padded
            print(
                f"Fraction of background that is padded: {np.sum(disc_bkg==PAD_VAL) / len(disc_bkg)}"
            )

            # Store the new discriminant object
            self.discriminants[disc_name] = Discriminant.from_disc_scores(
                disc_name,
                disc_sig,
                disc_bkg,
                weights_sig,
                weights_bkg,
                extended_labels,
                signal_name,
                background_names,
            )

    def compute_rocs(self, verbose=True):
        """
        Compute the ROC curves for all discriminants in parallel using joblib for speed.
        Stores the results in self.rocs, by erasing anything already there.
        """
        if verbose:
            print("Start computing ROCs...")
            t0 = time.time()

        Parallel(n_jobs=-1, prefer="threads")(
            delayed(disc.compute_roc)() for disc in self.discriminants.values()
        )

        if verbose:
            t1 = time.time()
            print(
                f"Computed ROCs for {len(self.discriminants)} discriminants in {t1 - t0:.2f} seconds"
            )

    def plot_disc_scores(
        self,
        disc_name: str,
        background_names_groups: list[list[str]],
        plot_dir: Path | str,
        nbins: int = 100,
    ):
        """
        Plot discriminant score distributions for signal and grouped background samples.

        This method creates a histogram showing the distribution of discriminant scores for signal events and background events grouped according to the provided background_names_groups.

        Args:
            disc (discriminant): discriminant object containing scores, labels, weights, and metadata.
            background_names_groups (list[list[str]]): List of background sample groups to plot together.
                Each inner list contains background sample names that will be combined into one histogram.
                Example: [["qcd", "ttbar"], ["wjets", "zjets"]] creates two background histograms.
            plot_dir (Path): Directory where the plot will be saved. Creates a "scores" subdirectory.
            nbins (int, optional): Number of bins for the histogram. Defaults to 100.

        Returns:
            None: The plot is saved to plot_dir/scores/ with an automatically generated filename.
        """

        disc = self.discriminants[disc_name]

        # if background_names_groups is a list of strings, convert it to a list of lists
        if isinstance(background_names_groups, list) and not isinstance(
            background_names_groups[0], list
        ):
            background_names_groups = [background_names_groups]

        # Check that background names are contained in disc.get_bkg_names()
        for b in {bg for bg_group in background_names_groups for bg in bg_group}:
            if b not in disc.get_bkg_names():
                print(
                    f"Warning: Background {b} not found in discriminant {disc} when plotting scores. Aborting."
                )
                return

        # Check that the folder exists
        (plot_dir / "scores").mkdir(parents=True, exist_ok=True)

        sig_disc = disc.get_discriminant_score()[disc.get_binary_labels() == 1]
        bkg_disc_groups = [
            np.concatenate(
                [
                    disc.get_discriminant_score()[disc.get_extended_labels() == bg]
                    for bg in background_group
                ]
            )
            for background_group in background_names_groups
        ]

        bkg_names_groups = [
            "".join(
                [SAMPLES[bg].label.replace(" ", "").replace("Multijet", "") for bg in bg_group]
            ).replace("TTHadTTSLTTLL", "Top")
            for bg_group in background_names_groups
        ]

        bkg_weights_groups = [
            np.concatenate(
                [disc.get_weights()[disc.get_extended_labels() == bg] for bg in background_group]
            )
            for background_group in background_names_groups
        ]

        # print("discs",([sig_disc] + bkg_disc_groups))
        # print("names",[disc.get_signal_name()] + bkg_names_groups)
        # print("weights",[disc.get_weights()[disc.get_binary_labels() == 1]]
        #                 + bkg_weights_groups)
        # print("\n\n")

        plotting.plot_hist(
            [sig_disc] + bkg_disc_groups,
            [disc.get_signal_name()] + bkg_names_groups,
            nbins=nbins,
            weights=[disc.get_weights()[disc.get_binary_labels() == 1]] + bkg_weights_groups,
            xlabel=f"{disc_name} score",
            xlim=(0, 1),
            lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
            density=True,
            year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
            plot_dir=plot_dir / "scores",
            name=f"{disc_name}_{'_'.join(''.join(background_group) for background_group in background_names_groups)}",
        )

    def plot_rocs(self, title, disc_names, plot_dir, thresholds=None):
        """
        Plot the ROC curves for a set of discriminants using the plotting utilities.

        Args:
            title (str): Title of the plot.
            discs (list[Discriminant]): Names of the discriminants to plot. Assumes that the signal name is the same for all discriminants.
            plot_dir (Path): Directory to save the plot.
        """
        # Check that the folder exists
        (plot_dir / "rocs").mkdir(parents=True, exist_ok=True)

        if thresholds is None:
            thresholds = [0.7, 0.9, 0.95, 0.99]

        # check that all discriminants have the same signal name
        signal_name = self.discriminants[disc_names[0]].get_signal_name()
        for disc_name in disc_names:
            if self.discriminants[disc_name].get_signal_name() != signal_name:
                print(
                    f"Warning: Discriminant {disc_name} has a different signal name than {signal_name}. Aborting."
                )
                return

        plotting.multiROCCurve(
            {
                "": {
                    disc_name: self.discriminants[disc_name].get_roc(unpacked=True)
                    for disc_name in disc_names
                }
            },
            title=title,
            thresholds=thresholds,
            show=True,
            plot_dir=plot_dir / "rocs",
            lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
            year="2022-23" if self.years == hh_vars.years else "+".join(self.years),
            name=title + "_".join(self.years),
        )

    def compute_confusion_matrix(self, discriminant_name, threshold=0.5, plot_dir=None):
        disc = self.discriminants[discriminant_name]
        disc_scores = disc.disc_scores
        extended_labels = disc.extended_labels
        binary_labels = disc.binary_labels
        bkg_names = disc.bkg_names
        signal_name = disc.signal_name
        weights = disc.weights

        # All possible columns: backgrounds + signal
        col_names = bkg_names + [signal_name]

        # Initialize the weighted confusion matrix
        cm = np.zeros((2, len(col_names)), dtype=float)

        # Signal events (row 1)
        signal_mask = binary_labels == 1
        signal_scores = disc_scores[signal_mask]
        signal_weights = weights[signal_mask]
        signal_ext_labels = extended_labels[signal_mask]
        # Predicted: signal if above threshold, else (optionally) their own label (rare for true signal)
        signal_pred = np.where(
            signal_scores >= threshold,
            len(col_names) - 1,  # Predicted as signal
            [
                bkg_names.index(lbl) if lbl in bkg_names else len(col_names) - 1
                for lbl in signal_ext_labels
            ],
        )
        for i, pred in enumerate(signal_pred):
            cm[1, pred] += signal_weights[i]

        # Background events (row 0)
        bkg_mask = binary_labels == 0
        bkg_scores = disc_scores[bkg_mask]
        bkg_weights = weights[bkg_mask]
        bkg_ext_labels = extended_labels[bkg_mask]
        bkg_pred = np.where(
            bkg_scores >= threshold,
            len(col_names) - 1,  # Predicted as signal
            [bkg_names.index(lbl) for lbl in bkg_ext_labels],
        )
        for i, pred in enumerate(bkg_pred):
            cm[0, pred] += bkg_weights[i]

        plt.imshow(cm, cmap="Blues", aspect="auto")
        plt.xticks(ticks=np.arange(len(col_names)), labels=col_names, rotation=45)
        plt.yticks([0, 1], ["Background", "Signal"])
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.title(f"Weighted Confusion Matrix\n{discriminant_name}")
        plt.colorbar(label="Sum of event weights")
        plt.tight_layout()
        if plot_dir is not None:
            (plot_dir / "confusion_matrix").mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / "confusion_matrix" / "2byN_weighted.png")
            plt.savefig(plot_dir / "confusion_matrix" / "2byN_weighted.pdf")
        plt.show()
