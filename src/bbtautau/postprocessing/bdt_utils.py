from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xgboost as xgb
from boostedhh.utils import PAD_VAL

from bbtautau.postprocessing.bbtautau_types import Channel, LoadedSample
from bbtautau.postprocessing.bdt_config import BDT_CONFIG
from bbtautau.postprocessing.Samples import CHANNELS

"""
TODO: ensure that if load multiple models, the BDT scores are added correctly to the events dataframe and do not overwrite each other.

"""


# Non-interactive backend for batch/containers
mpl.use("Agg")

# Channel order for BDT models (must match the order used in training)
# This defines the order of channels for each signal: he, hh, hm
CHANNEL_ORDER_BDT = ["he", "hh", "hm"]

# Background order for BDT models (must match the order used in training)
# This defines the order of background samples
DEFAULT_BKG_ORDER = ["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]


def get_expected_sample_order(signals: list[str], bkg_sample_names: list[str]) -> list[str]:
    """Get the expected sample order for BDT model training/inference.

    The order is: signal channels (he, hh, hm) for each signal, then backgrounds.
    This matches the class order expected by the BDT model.

    Backgrounds are ordered according to DEFAULT_BKG_ORDER if they match,
    otherwise in the order provided.

    Args:
        signals: List of signal keys (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])
        bkg_sample_names: List of background sample names

    Returns:
        List of sample names in the expected order for BDT model classes
    """
    sample_names = []
    # For each signal, add its channels in order: he, hh, hm
    for signal_key in signals:
        for ch in CHANNEL_ORDER_BDT:
            sample_names.append(f"{signal_key}{ch}")

    # Add backgrounds: use DEFAULT_BKG_ORDER if backgrounds match it, otherwise use provided order
    if set(bkg_sample_names) == set(DEFAULT_BKG_ORDER):
        # Use the standard order
        sample_names.extend(DEFAULT_BKG_ORDER)
    else:
        # Use the order provided
        sample_names.extend(bkg_sample_names)

    return sample_names


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_bdt_preds_dir(
    base_dir: Path,
    modelname: str,
    signal_objectives: list[str],
    channel: Channel,
    year: str,
    tt_pres: bool = False,
    test_mode: bool = False,
) -> Path:
    """Get the directory path for BDT predictions cache files.

    Directory structure: base_dir / tag / signal / modelname / channel.key / year

    Where:
    - tag is "test", "tt_pres", or "full_presel" based on test_mode and tt_pres
    - signal is the first signal objective if there's one, otherwise "unified_signals"

    Args:
        base_dir: Base directory for BDT predictions
        modelname: Name of the BDT model
        signal_objectives: List of signal objectives (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])
        channel: Channel object with .key attribute
        year: Year string (e.g., '2018')
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

    return base_dir / tag / signal_dir / modelname / channel.key / year


# Centralize this for other applications:
# Prefix all BDT columns with the signal key to distinguish models
# Remove trailing 'tt' from signal_key to avoid redundancy (e.g., ggfbbtt -> ggfbb)
# This gives cleaner names like BDTggfbbtauhtauh instead of BDTggfbbtttauhtauh
def _get_bdt_key(
    signal: str, channel: Channel = None, prefix_only: bool = True, suffix: str = "vsAll"
):
    signal_base = signal.removesuffix("tt") if signal.endswith("tt") else signal
    if prefix_only:
        return f"BDT{signal_base}"
    else:
        if channel is None:
            raise ValueError("Channel is required for non-prefix BDT keys")
        return f"BDT{signal_base}{channel.tagger_label}{suffix}"


def _add_bdt_scores(
    events: pd.DataFrame,
    sample_bdt_preds: np.ndarray,
    signal_objectives: list[str],  # For unified model, list of signals
    jshift: str = "",
):
    """
    Add BDT scores to events DataFrame.

    Now uses signal_objectives (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt']) to namespace all BDT columns,
    so multiple models can coexist in the same DataFrame.

    Class ordering:
    - single_signal model (8 classes):
        Class 0: signal_he (e.g., ggfbbtthe or vbfbbtthe)
        Class 1: signal_hh (e.g., ggfbbtthh or vbfbbtthh)
        Class 2: signal_hm (e.g., ggfbbtthm or vbfbbtthm)
        Class 3: dyjets
        Class 4: qcd
        Class 5: ttbarhad
        Class 6: ttbarll
        Class 7: ttbarsl

    - unified model (11 classes):
        Class 0-2: first signal channels (he, hh, hm)
        Class 3-5: second signal channels (he, hh, hm)
        Class 6: dyjets
        Class 7: qcd
        Class 8: ttbarhad
        Class 9: ttbarll
        Class 10: ttbarsl

    Args:
        events: DataFrame to add scores to
        sample_bdt_preds: BDT predictions array with shape (n_events, n_classes)
            - For single signal: n_classes = 8 (3 signal channels + 5 backgrounds)
            - For unified model: n_classes = 11 (6 signal channels + 5 backgrounds)
        signal_objectives: List of 1 or 2 signal objectives (e.g., ['ggfbbtt'] or ['ggfbbtt', 'vbfbbtt'])
        jshift: JEC/JMSR shift label (e.g., 'JEC_up', 'JMSR_down')
    """

    if jshift != "":
        jshift = "_" + jshift

    # Validate and normalize signal_objectives
    if not isinstance(signal_objectives, list):
        raise ValueError(f"signal_objectives must be a list, got {signal_objectives}")

    single_signal = len(signal_objectives) == 1

    if single_signal:
        # Single signal model: 3 signal channels + 5 backgrounds

        signal_objective = signal_objectives[0]

        events[
            _get_bdt_key(signal_objective, prefix_only=False, channel=CHANNELS["he"], suffix=jshift)
        ] = sample_bdt_preds[
            :, 0
        ]  # signal channel he
        events[
            _get_bdt_key(signal_objective, prefix_only=False, channel=CHANNELS["hh"], suffix=jshift)
        ] = sample_bdt_preds[
            :, 1
        ]  # signal channel hh
        events[
            _get_bdt_key(signal_objective, prefix_only=False, channel=CHANNELS["hm"], suffix=jshift)
        ] = sample_bdt_preds[
            :, 2
        ]  # signal channel hm

        events[f"BDTDY{jshift}"] = sample_bdt_preds[:, 3]
        events[f"BDTQCD{jshift}"] = sample_bdt_preds[:, 4]
        events[f"BDTTThad{jshift}"] = sample_bdt_preds[:, 5]
        events[f"BDTTTll{jshift}"] = sample_bdt_preds[:, 6]
        events[f"BDTTTSL{jshift}"] = sample_bdt_preds[:, 7]

        for ch in CHANNELS.values():
            base_sig = events[
                _get_bdt_key(signal_objective, prefix_only=False, channel=ch, suffix=jshift)
            ]
            events[
                _get_bdt_key(
                    signal_objective, prefix_only=False, channel=ch, suffix=f"vsQCD{jshift}"
                )
            ] = np.nan_to_num(
                base_sig / (base_sig + events[f"BDTQCD{jshift}"]),
                nan=PAD_VAL,
            )

            events[
                _get_bdt_key(
                    signal_objective, prefix_only=False, channel=ch, suffix=f"vsAll{jshift}"
                )
            ] = np.nan_to_num(
                base_sig
                / (
                    base_sig
                    + events[f"BDTQCD{jshift}"]
                    + events[f"BDTTThad{jshift}"]
                    + events[f"BDTTTll{jshift}"]
                    + events[f"BDTTTSL{jshift}"]
                    + events[f"BDTDY{jshift}"]
                ),
                nan=PAD_VAL,
            )
    else:
        for i, signal_objective in enumerate(signal_objectives):
            events[
                _get_bdt_key(
                    signal_objective, prefix_only=False, channel=CHANNELS["he"], suffix=jshift
                )
            ] = sample_bdt_preds[:, i * 3 + 0]
            events[
                _get_bdt_key(
                    signal_objective, prefix_only=False, channel=CHANNELS["hh"], suffix=jshift
                )
            ] = sample_bdt_preds[:, i * 3 + 1]
            events[
                _get_bdt_key(
                    signal_objective, prefix_only=False, channel=CHANNELS["hm"], suffix=jshift
                )
            ] = sample_bdt_preds[:, i * 3 + 2]

        events[f"BDTDY{jshift}"] = sample_bdt_preds[:, 6]
        events[f"BDTQCD{jshift}"] = sample_bdt_preds[:, 7]
        events[f"BDTTThad{jshift}"] = sample_bdt_preds[:, 8]
        events[f"BDTTTll{jshift}"] = sample_bdt_preds[:, 9]
        events[f"BDTTTSL{jshift}"] = sample_bdt_preds[:, 10]

        # Add vsQCD and vsAll discriminants for both signals
        for signal_objective in signal_objectives:
            for ch in CHANNELS.values():
                base_sig = events[
                    _get_bdt_key(signal_objective, prefix_only=False, channel=ch, suffix=jshift)
                ]

                events[
                    _get_bdt_key(
                        signal_objective, prefix_only=False, channel=ch, suffix=f"vsQCD{jshift}"
                    )
                ] = np.nan_to_num(
                    base_sig / (base_sig + events[f"BDTQCD{jshift}"]),
                    nan=PAD_VAL,
                )

                events[
                    _get_bdt_key(
                        signal_objective, prefix_only=False, channel=ch, suffix=f"vsAll{jshift}"
                    )
                ] = np.nan_to_num(
                    base_sig
                    / (
                        base_sig
                        + events[f"BDTQCD{jshift}"]
                        + events[f"BDTTThad{jshift}"]
                        + events[f"BDTTTll{jshift}"]
                        + events[f"BDTTTSL{jshift}"]
                        + events[f"BDTDY{jshift}"]
                    ),
                    nan=PAD_VAL,
                )


def compute_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    test_mode: bool = False,
    save_dir: Path = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute BDT predictions for multiple years and samples.

    This function loads a trained XGBoost model and uses it to make predictions on multiple
    years and samples of data. The input data is expected to be organized by year and sample name.

    The predictions are saved in the directory structure:
        save_dir / tag / signal_objective / modelname / channel.key / year / {sample}_preds.pkl

    Args:
        events_dict: Dictionary mapping years to dictionaries of samples. Each sample should be a
            LoadedSample object containing the features needed for prediction.
        modelname: Name of the model to load
        model_dir: Path to the directory containing the model
        signal_objective: Signal objective (e.g., 'ggfbbtt', 'vbfbbtt') for organizing saved predictions
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        test_mode: Whether in test mode (uses different directory structure)
        save_dir: Directory to save predictions

    Returns:
        dict: Nested dictionary containing predictions for each year and sample:
            {year: {sample: predictions_array}}
    """

    # Load the model for this signal
    model_path = model_dir / modelname / f"{modelname}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Expected structure: model_dir/modelname/modelname.json"
        )

    bst = xgb.Booster()
    bst.load_model(str(model_path))

    feature_names = [
        feat
        for cat in BDT_CONFIG[modelname]["train_vars"]
        for feat in BDT_CONFIG[modelname]["train_vars"][cat]
    ]

    # Use ThreadPoolExecutor for parallel processing (threads share memory, no pickling needed)
    total_samples = sum(len(events_dict[year]) for year in events_dict)
    max_workers = min(4, total_samples)  # Use reasonable number of workers

    def predict_sample(year, sample, sample_data, feature_names, bst):
        """Worker function for parallel prediction"""
        dsample = xgb.DMatrix(
            np.stack(
                [sample_data.get_var(feat) for feat in feature_names],
                axis=1,
            ),
            feature_names=feature_names,
        )
        return year, sample, bst.predict(dsample)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit all prediction tasks
        for year in events_dict:
            for sample in events_dict[year]:
                future = executor.submit(
                    predict_sample, year, sample, events_dict[year][sample], feature_names, bst
                )
                futures.append(future)

        # Collect results and apply BDT scores
        for future in futures:
            year, sample, y_pred = future.result()
            _add_bdt_scores(
                events_dict[year][sample].events,
                y_pred,
                signal_objectives=signal_objectives,
            )

            if save_dir is not None:
                file_dir = get_bdt_preds_dir(
                    base_dir=save_dir,
                    modelname=modelname,
                    signal_objectives=signal_objectives,
                    channel=channel,
                    year=year,
                    tt_pres=tt_pres,
                    test_mode=test_mode,
                )
                file_dir.mkdir(exist_ok=True, parents=True)
                with (file_dir / f"{sample}_preds.pkl").open("wb") as f:
                    pickle.dump(y_pred, f)
                with (file_dir / f"{sample}_shapes.pkl").open("wb") as f:
                    pickle.dump(events_dict[year][sample].events.shape, f)


def check_bdt_prediction_shapes(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool,
) -> bool:
    """Check if BDT predictions exist and have correct shapes.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_objective: Signal objective (e.g., 'ggfbbtt', 'vbfbbtt')
        tt_pres: Whether tt preselection is applied
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing BDT predictions
        test_mode: Whether in test mode

    Returns:
        bool: True if all predictions exist and have correct shapes, False otherwise
    """
    for year in events_dict:
        for sample in events_dict[year]:
            preds_dir = get_bdt_preds_dir(
                base_dir=bdt_preds_dir,
                modelname=modelname,
                signal_objectives=signal_objectives,
                channel=channel,
                year=year,
                tt_pres=tt_pres,
                test_mode=test_mode,
            )
            shape_file = preds_dir / f"{sample}_shapes.pkl"
            if not shape_file.exists():
                print(f"Prediction file does not exist: {shape_file}")
                return False
            shape = pickle.load(shape_file.open("rb"))
            if shape[0] != events_dict[year][sample].events.shape[0]:
                print(
                    f"Shape mismatch for {sample} in {year}: {shape} != {events_dict[year][sample].events.shape}"
                )
                return False
    return True


def load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_objectives: list[str],
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool = False,
):
    """Load BDT predictions from disk and add scores to events.

    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.
    If ``jec_jmsr_shifts``, also loads BDT preds for every JEC / JMSR shift in MC.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_objectives: List of signal objectives (e.g., ['ggfbbtt', 'vbfbbtt'])
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing BDT predictions
        test_mode: Whether in test mode
    """

    for year in events_dict:
        for sample in events_dict[year]:
            preds_dir = get_bdt_preds_dir(
                base_dir=bdt_preds_dir,
                modelname=modelname,
                signal_objectives=signal_objectives,
                channel=channel,
                year=year,
                tt_pres=tt_pres,
                test_mode=test_mode,
            )
            pred_file = preds_dir / f"{sample}_preds.pkl"

            bdt_preds = pickle.load(pred_file.open("rb"))

            _add_bdt_scores(
                events_dict[year][sample].events,
                bdt_preds,
                signal_objectives=signal_objectives,
            )


def compute_or_load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool = False,
    at_inference: bool = False,
):
    """Wrapper function to compute or load BDT predictions.

    This function handles the logic of:
    1. If at_inference is True, compute predictions directly
    2. Otherwise, check if saved predictions exist with correct shapes
    3. If they exist and match, load them
    4. If they don't exist or shapes mismatch, compute and save them

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        model_dir: Path to the directory containing the model
        signal_objective: Signal objective (e.g., 'ggfbbtt', 'vbfbbtt')
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing/to save BDT predictions
        test_mode: Whether in test mode
        at_inference: If True, always compute predictions (don't try to load)

    Example:
        >>> compute_or_load_bdt_preds(
        ...     events_dict=my_events,
        ...     modelname="29July25_loweta_lowreg",
        ...     model_dir=Path("/path/to/models"),
        ...     channel=my_channel,
        ...     bdt_preds_dir=Path("/path/to/predictions"),
        ...     at_inference=False,  # Will try to load if available
        ... )
    """
    import time

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

    if at_inference:
        # Compute predictions directly at inference time
        print(f"Computing BDT predictions at inference for signal '{signal_objectives}'...")
        start_time = time.time()
        compute_bdt_preds(
            events_dict=events_dict,
            modelname=modelname,
            model_dir=model_dir,
            signal_objectives=signal_objectives,
            channel=channel,
            tt_pres=tt_pres,
            test_mode=test_mode,
            save_dir=bdt_preds_dir,
        )
        print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")
    else:
        # Try to load existing predictions
        print(f"Checking for existing BDT predictions for signals '{signal_objectives}'...")
        shapes_ok = check_bdt_prediction_shapes(
            events_dict=events_dict,
            modelname=modelname,
            signal_objectives=signal_objectives,
            channel=channel,
            tt_pres=tt_pres,
            bdt_preds_dir=bdt_preds_dir,
            test_mode=test_mode,
        )

        if shapes_ok:
            print(f"Loading existing BDT predictions for signals '{signal_objectives}'...")
            load_bdt_preds(
                events_dict=events_dict,
                modelname=modelname,
                signal_objectives=signal_objectives,
                channel=channel,
                tt_pres=tt_pres,
                bdt_preds_dir=bdt_preds_dir,
                test_mode=test_mode,
            )
            print("BDT predictions loaded successfully")
        else:
            print("BDT predictions don't exist or shapes don't match. " "Computing predictions...")
            start_time = time.time()
            compute_bdt_preds(
                events_dict=events_dict,
                modelname=modelname,
                model_dir=model_dir,
                signal_objectives=signal_objectives,
                channel=channel,
                tt_pres=tt_pres,
                test_mode=test_mode,
                save_dir=bdt_preds_dir,
            )
            print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")
