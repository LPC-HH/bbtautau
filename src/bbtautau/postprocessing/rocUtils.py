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
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from bbtautau.postprocessing.bbtautau_types import LoadedSample
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
class Metrics:
    """Container for comprehensive performance metrics."""

    # AUC-based metrics
    roc_auc: float
    pr_auc: float

    # Threshold-dependent metrics at optimal point
    optimal_threshold: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    balanced_accuracy: float
    matthews_corr: float

    # Threshold-dependent metrics at 0.5 threshold
    f1_score_05: float
    precision_05: float
    recall_05: float
    accuracy_05: float

    # Additional info
    n_signal: int
    n_background: int
    signal_weight: float
    background_weight: float


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
        roc = ROC(fpr, tpr, thresholds, LoadedSample.rename_jetbranch_ak8(self.get_name()), roc_auc)
        self.roc = roc
        return roc

    def compute_metrics(self):
        """
        Compute comprehensive performance metrics for the discriminant.
        """
        y_true = self.get_binary_labels()
        y_scores = self.get_discriminant_score()
        weights = self.get_weights()

        # AUC-based metrics
        roc_auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        pr_auc = average_precision_score(y_true, y_scores, sample_weight=weights)

        # Find optimal threshold using F1-score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            try:
                f1 = f1_score(y_true, y_pred, sample_weight=weights, zero_division=0)
                f1_scores.append(f1)
            except:
                f1_scores.append(0.0)

        optimal_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        y_pred_05 = (y_scores >= 0.5).astype(int)

        # Compute metrics at optimal threshold
        f1_optimal = f1_score(y_true, y_pred_optimal, sample_weight=weights, zero_division=0)
        precision_optimal = precision_score(
            y_true, y_pred_optimal, sample_weight=weights, zero_division=0
        )
        recall_optimal = recall_score(
            y_true, y_pred_optimal, sample_weight=weights, zero_division=0
        )
        accuracy_optimal = accuracy_score(y_true, y_pred_optimal, sample_weight=weights)
        balanced_acc_optimal = balanced_accuracy_score(
            y_true, y_pred_optimal, sample_weight=weights
        )
        matthews_optimal = matthews_corrcoef(y_true, y_pred_optimal, sample_weight=weights)

        # Compute metrics at 0.5 threshold
        f1_05 = f1_score(y_true, y_pred_05, sample_weight=weights, zero_division=0)
        precision_05 = precision_score(y_true, y_pred_05, sample_weight=weights, zero_division=0)
        recall_05 = recall_score(y_true, y_pred_05, sample_weight=weights, zero_division=0)
        accuracy_05 = accuracy_score(y_true, y_pred_05, sample_weight=weights)

        # Sample statistics
        signal_mask = y_true == 1
        n_signal = np.sum(signal_mask)
        n_background = np.sum(~signal_mask)
        signal_weight = np.sum(weights[signal_mask])
        background_weight = np.sum(weights[~signal_mask])

        metrics = Metrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            optimal_threshold=optimal_threshold,
            f1_score=f1_optimal,
            precision=precision_optimal,
            recall=recall_optimal,
            accuracy=accuracy_optimal,
            balanced_accuracy=balanced_acc_optimal,
            matthews_corr=matthews_optimal,
            f1_score_05=f1_05,
            precision_05=precision_05,
            recall_05=recall_05,
            accuracy_05=accuracy_05,
            n_signal=n_signal,
            n_background=n_background,
            signal_weight=signal_weight,
            background_weight=background_weight,
        )
        self.metrics = metrics
        return metrics

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

    def get_metrics(self, as_dict=False):
        """Get comprehensive metrics for the discriminant."""
        if not hasattr(self, "metrics"):
            print(f"Warning: Metrics not computed for discriminant {self.get_name()}")
            return None

        if as_dict:
            return {
                "roc_auc": self.metrics.roc_auc,
                "pr_auc": self.metrics.pr_auc,
                "optimal_threshold": self.metrics.optimal_threshold,
                "f1_score": self.metrics.f1_score,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "accuracy": self.metrics.accuracy,
                "balanced_accuracy": self.metrics.balanced_accuracy,
                "matthews_corr": self.metrics.matthews_corr,
                "f1_score_05": self.metrics.f1_score_05,
                "precision_05": self.metrics.precision_05,
                "recall_05": self.metrics.recall_05,
                "accuracy_05": self.metrics.accuracy_05,
                "n_signal": self.metrics.n_signal,
                "n_background": self.metrics.n_background,
                "signal_weight": self.metrics.signal_weight,
                "background_weight": self.metrics.background_weight,
            }
        else:
            return self.metrics

    def _spline_interpolate_threshold(self, tpr_values, threshold_values, target_tpr):
        """
        Interpolate threshold using spline with nearest neighbors around target TPR.

        Args:
            tpr_values (np.ndarray): Sorted unique TPR values
            threshold_values (np.ndarray): Corresponding threshold values
            target_tpr (float): Target TPR to find threshold for

        Returns:
            float: Interpolated threshold value
        """
        n_points = len(tpr_values)

        # Find the index where target_tpr would be inserted to maintain order
        insert_idx = np.searchsorted(tpr_values, target_tpr)

        # Determine neighborhood around target point
        # Use at least 4 points for cubic spline, more if available
        min_points = min(4, n_points)
        half_points = min_points // 2

        # Calculate start and end indices for neighborhood
        start_idx = max(0, insert_idx - half_points)
        end_idx = min(n_points, start_idx + min_points)

        # Adjust start if we hit the end boundary
        if end_idx - start_idx < min_points:
            start_idx = max(0, end_idx - min_points)

        # Extract neighborhood points
        tpr_neighborhood = tpr_values[start_idx:end_idx]
        threshold_neighborhood = threshold_values[start_idx:end_idx]

        # Handle edge cases
        if len(tpr_neighborhood) < 2:
            # Fall back to linear interpolation
            return np.interp(target_tpr, tpr_values, threshold_values)

        if len(tpr_neighborhood) == 2:
            # Linear interpolation for 2 points
            return np.interp(target_tpr, tpr_neighborhood, threshold_neighborhood)

        # Check if target is exactly at a data point
        exact_match = np.where(tpr_neighborhood == target_tpr)[0]
        if len(exact_match) > 0:
            return threshold_neighborhood[exact_match[0]]

        try:
            # Use cubic spline with smoothing
            # s=0 for interpolation (passes through all points)
            # k=min(3, len-1) for cubic or lower order if insufficient points
            k = min(3, len(tpr_neighborhood) - 1)
            spline = UnivariateSpline(tpr_neighborhood, threshold_neighborhood, k=k, s=0)
            interpolated_value = float(spline(target_tpr))

            # Sanity check: ensure result is within reasonable bounds
            min_thresh = np.min(threshold_neighborhood)
            max_thresh = np.max(threshold_neighborhood)

            # If spline extrapolates beyond reasonable bounds, fall back to linear
            if not (min_thresh <= interpolated_value <= max_thresh):
                return np.interp(target_tpr, tpr_neighborhood, threshold_neighborhood)

            return interpolated_value

        except Exception as e:
            # Fall back to linear interpolation if spline fails
            print(f"Warning: Spline interpolation failed ({e}), using linear interpolation")
            return np.interp(target_tpr, tpr_neighborhood, threshold_neighborhood)

    def _vectorized_spline_interpolate(self, tpr_values, threshold_values, target_tprs):
        """
        Vectorized spline interpolation for multiple target TPR values.

        Args:
            tpr_values (np.ndarray): Sorted unique TPR values from ROC curve
            threshold_values (np.ndarray): Corresponding threshold values
            target_tprs (np.ndarray): Array of target TPR values to interpolate

        Returns:
            np.ndarray: Interpolated threshold values
        """
        n_points = len(tpr_values)

        if n_points < 4:
            # Use linear interpolation for insufficient points
            return np.interp(target_tprs, tpr_values, threshold_values)

        try:
            # Create spline once and evaluate at all target points
            k = min(3, n_points - 1)  # Cubic or lower order
            # Check for problematic threshold values before spline creation
            finite_mask = np.isfinite(threshold_values)
            if not np.all(finite_mask):
                tpr_values = tpr_values[finite_mask]
                threshold_values = threshold_values[finite_mask]

                if len(tpr_values) < 2:
                    return np.full_like(target_tprs, np.nan)

            spline = UnivariateSpline(tpr_values, threshold_values, k=k, s=0)

            # Vectorized evaluation
            interpolated_values = spline(target_tprs)

            # Sanity check: ensure results are within reasonable bounds
            min_thresh = np.min(threshold_values)
            max_thresh = np.max(threshold_values)

            # Check which values are out of bounds
            out_of_bounds = (interpolated_values < min_thresh) | (interpolated_values > max_thresh)

            if np.any(out_of_bounds):
                # Fall back to linear interpolation for out-of-bounds values
                linear_values = np.interp(target_tprs, tpr_values, threshold_values)
                interpolated_values[out_of_bounds] = linear_values[out_of_bounds]

            return interpolated_values

        except Exception as e:
            # Fall back to linear interpolation if spline fails
            print(
                f"Warning: Vectorized spline interpolation failed ({e}), using linear interpolation"
            )
            return np.interp(target_tprs, tpr_values, threshold_values)

    def get_cut_from_sig_eff(self, target_sig_eff):
        """
        Get the discriminant threshold that yields a specific signal efficiency.

        Signal efficiency is defined as TPR (True Positive Rate) = TP / (TP + FN).
        This function interpolates the ROC curve to find the threshold corresponding
        to the desired signal efficiency.

        Args:
            target_sig_eff (float or array-like): Desired signal efficiency between 0 and 1.
                                                 Can be scalar or array for vectorized operation.

        Returns:
            float or ndarray: Discriminant threshold(s) that yield the target signal efficiency.
                             Returns np.nan if target efficiency is not achievable or ROC not computed.

        Examples:
            >>> threshold_90pct = discriminant.get_cut_from_sig_eff(0.9)  # 90% signal efficiency
            >>> thresholds = discriminant.get_cut_from_sig_eff([0.5, 0.7, 0.9])  # Vectorized
        """
        if not hasattr(self, "roc"):
            print(f"Warning: ROC curve not computed for discriminant {self.get_name()}")
            return np.nan if np.isscalar(target_sig_eff) else np.full_like(target_sig_eff, np.nan)

        # Handle both scalar and array inputs
        target_sig_eff = np.asarray(target_sig_eff)
        is_scalar = target_sig_eff.ndim == 0

        # Ensure we work with arrays for vectorized operations
        if is_scalar:
            target_sig_eff = target_sig_eff.reshape(1)

        # Validate input range
        invalid_mask = (target_sig_eff < 0) | (target_sig_eff > 1)
        if np.any(invalid_mask):
            print("Warning: Signal efficiency must be between 0 and 1, got invalid values")

        tpr = self.roc.tpr
        thresholds = self.roc.thresholds

        # Prepare lookup data (do this once for all target efficiencies)
        min_eff = np.min(tpr)
        max_eff = np.max(tpr)

        # Sort by TPR to ensure monotonic interpolation
        sort_indices = np.argsort(tpr)
        tpr_sorted = tpr[sort_indices]
        thresholds_sorted = thresholds[sort_indices]

        # Remove duplicate TPR values to avoid interpolation issues
        unique_tpr, unique_indices = np.unique(tpr_sorted, return_index=True)
        unique_thresholds = thresholds_sorted[unique_indices]

        if len(unique_tpr) < 2:
            print("Warning: Insufficient unique TPR values for interpolation")
            result = np.full_like(target_sig_eff, thresholds[0], dtype=float)
            return result[0] if is_scalar else result

        # Vectorized bounds checking and interpolation
        result = np.full_like(target_sig_eff, np.nan, dtype=float)

        # Handle out-of-bounds cases
        below_min = target_sig_eff < min_eff
        above_max = target_sig_eff > max_eff
        valid = ~below_min & ~above_max & ~invalid_mask

        if np.any(below_min):
            result[below_min] = thresholds[np.argmin(tpr)]
        if np.any(above_max):
            result[above_max] = thresholds[np.argmax(tpr)]

        # Vectorized interpolation for valid points
        if np.any(valid):
            result[valid] = self._vectorized_spline_interpolate(
                unique_tpr, unique_thresholds, target_sig_eff[valid]
            )

        return result[0] if is_scalar else result


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
        custom_name: str = None,
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

        if custom_name:
            disc_name = custom_name
        else:
            # this disc name works only when the name of the tagger score is also the name of the sample, for now default for the bdt setup
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

        def _rocs_metrics(disc):
            return disc.compute_roc(), disc.compute_metrics()

        # Compute both ROCs and comprehensive metrics
        results = Parallel(n_jobs=-1)(
            delayed(_rocs_metrics)(disc) for disc in self.discriminants.values()
        )

        for disc, (roc, metrics) in zip(self.discriminants.values(), results):
            disc.roc = roc
            disc.metrics = metrics

        if verbose:
            t1 = time.time()
            print(
                f"Computed ROCs for {len(self.discriminants)} discriminants in {t1 - t0:.2f} seconds"
            )

    def get_metrics_summary(self, signal_names=None, save_path=None):
        """
        Get a summary of all metrics for all discriminants.

        Args:
            signal_names: List of signal names to include. If None, include all.
            save_path: Path to save CSV summary. If None, don't save.

        Returns:
            dict: Nested dictionary with metrics organized by signal and discriminant
        """
        summary = {}

        for disc_name, disc in self.discriminants.items():
            if not hasattr(disc, "metrics"):
                continue

            signal_name = disc.get_signal_name()
            if signal_names is not None and signal_name not in signal_names:
                continue

            if signal_name not in summary:
                summary[signal_name] = {}

            summary[signal_name][disc_name] = disc.get_metrics(as_dict=True)

        # Save to CSV if requested
        if save_path is not None:
            self._save_metrics_csv(summary, save_path)

        return summary

    def _save_metrics_csv(self, summary, save_path):
        """Save metrics summary to CSV file."""
        import pandas as pd

        rows = []
        for signal_name, disc_data in summary.items():
            for disc_name, metrics in disc_data.items():
                row = {"signal": signal_name, "discriminant": disc_name, **metrics}
                rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(save_path, index=False)
        print(f"Metrics summary saved to {save_path}")

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
            thresholds = [0.7, 0.9, 0.95]

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

    def compute_confusion_matrix(
        self, discriminant_name, threshold=0.5, plot_dir=None, normalize=True
    ):
        disc = self.discriminants[discriminant_name]
        disc_scores = disc.disc_scores
        extended_labels = disc.extended_labels
        bkg_names = list(disc.bkg_names)
        signal_name = disc.signal_name
        weights = disc.weights

        # All possible columns: backgrounds + signal
        col_names = bkg_names + [signal_name]
        n_cols = len(col_names)

        # Map each label to its column index
        label_to_col = {name: i for i, name in enumerate(col_names)}

        # Rows: 0 = predicted background, 1 = predicted signal
        n_rows = 2
        y_pred = (disc_scores >= threshold).astype(int)

        # True class: index in col_names, determined from extended_labels
        y_true = np.array([label_to_col.get(lbl, n_cols - 1) for lbl in extended_labels])

        # Initialize weighted confusion matrix
        cm = np.zeros((n_rows, n_cols), dtype=float)

        # Fill matrix: for each event, add its weight to (pred, true_class)
        for pred, true_col, w in zip(y_pred, y_true, weights):
            cm[pred, true_col] += w

        # Normalize columns (per true class), if requested
        if normalize:
            col_sums = cm.sum(axis=0, keepdims=True)
            cm_norm = np.divide(cm, col_sums, where=col_sums != 0)
        else:
            cm_norm = cm

        # Plotting
        fig, ax = plt.subplots()
        im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else None)
        fig.colorbar(
            im,
            ax=ax,
            label=(
                "Fraction of true class (column-normalized)"
                if normalize
                else "Sum of event weights"
            ),
        )
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_names, rotation=45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Background", "Signal"])
        ax.set_xlabel("True class")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{'Normalized ' if normalize else ''}Confusion Matrix\n{discriminant_name}")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                value = cm_norm[i, j]
                display_val = f"{value:.2f}" if normalize else f"{value:.0f}"
                ax.text(
                    j,
                    i,
                    display_val,
                    ha="center",
                    va="center",
                    color="white" if value > (0.5 if normalize else 0.0) else "black",
                    fontsize=10,
                    fontweight="bold",
                )

        fig.tight_layout()
        if plot_dir is not None:
            (plot_dir / "confusion_matrix").mkdir(parents=True, exist_ok=True)
            suffix = "_normalized" if normalize else ""
            fig.savefig(
                plot_dir / "confusion_matrix" / f"2byN_weighted{disc.get_name()}_{suffix}.png",
                bbox_inches="tight",
            )
            fig.savefig(
                plot_dir / "confusion_matrix" / f"2byN_weighted{disc.get_name()}_{suffix}.pdf",
                bbox_inches="tight",
            )
        plt.close(fig)


###############################################################################
# Stand-alone N x N confusion-matrix utility for BDT outputs
###############################################################################
def multiclass_confusion_matrix(
    preds_dict: dict[str, LoadedSample],
    classes: list[str] | None = None,
    *,
    normalize: bool = True,
    plot_dir: Path | str | None = None,
) -> np.ndarray:
    """
    Build (and optionally plot) an NxN confusion matrix from the prediction
    files produced in Trainer.compute_rocs().

    Parameters
    ----------
    preds_dict : dict[str, LoadedSample]
        Mapping TRUE-class → LoadedSample whose ``events`` DataFrame contains
        one column per predicted class with the corresponding score/probability
        and a ``finalWeight`` column.
    classes : list[str] | None
        Desired class order.  Defaults to the keys of ``preds_dict``.
    normalize : bool
        If True (default) each column is scaled to sum to 1 so entries are
        class-wise efficiencies.  If False the raw weighted event counts are
        returned.
    plot_dir : Path | str | None
        Directory where ``confusion_matrix/<N>by<N>….{png,pdf}`` will be saved.
        Nothing is written if None.

    Returns
    -------
    np.ndarray
        (predicted class x true class) confusion matrix (normalized or raw).
    """
    if classes is None:
        classes = list(preds_dict.keys())

    n_cls = len(classes)
    label_to_col = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n_cls, n_cls), dtype=float)  # rows = predicted, cols = true

    # ------------------------------------------------------------------ fill
    for true_cls, ls in preds_dict.items():
        if true_cls not in label_to_col:
            continue
        true_col = label_to_col[true_cls]

        preds = ls.events
        if preds is None or preds.empty:
            continue

        missing = [c for c in classes if c not in preds]
        if missing:
            raise ValueError(
                f"Prediction dataframe for true class '{true_cls}' " f"is missing columns {missing}"
            )

        scores = preds[classes].to_numpy()
        y_pred = np.argmax(scores, axis=1)
        weights = preds.get("finalWeight", 1.0)
        if not isinstance(weights, np.ndarray):
            weights = weights.to_numpy()

        for pred_row in range(n_cls):
            mask = y_pred == pred_row
            if np.any(mask):
                cm[pred_row, true_col] += weights[mask].sum()

    # ---------------------------------------------------------------- normalise
    cm_norm = cm
    if normalize:
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_norm = np.divide(cm, col_sums, where=col_sums != 0)

    # ---------------------------------------------------------------- plotting
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        (plot_dir / "confusion_matrix").mkdir(parents=True, exist_ok=True)
        suffix = "_normalized" if normalize else ""
        fname = f"{n_cls}by{n_cls}_weighted{suffix}"

        fig, ax = plt.subplots()
        im = ax.imshow(
            cm_norm,
            cmap="Blues",
            vmin=0,
            vmax=1 if normalize else None,
        )
        fig.colorbar(
            im,
            ax=ax,
            label="Fraction of true class" if normalize else "Sum of event weights",
        )

        ax.set_xticks(np.arange(n_cls))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_cls))
        ax.set_yticklabels(classes)
        ax.set_xlabel("True class")
        ax.set_ylabel("Predicted class")
        ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")

        for i in range(n_cls):
            for j in range(n_cls):
                val = cm_norm[i, j]
                text = f"{val:.2f}" if normalize else f"{val:.0f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if val > (0.5 if normalize else 0.0) else "black",
                    fontsize=8,
                    fontweight="bold",
                )

        fig.tight_layout()
        fig.savefig(plot_dir / "confusion_matrix" / f"{fname}.png", bbox_inches="tight")
        fig.savefig(plot_dir / "confusion_matrix" / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    return cm_norm if normalize else cm
