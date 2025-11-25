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
from bbtautau.postprocessing.bdt_config import bdt_config
from bbtautau.postprocessing.Samples import CHANNELS

# Non-interactive backend for batch/containers
mpl.use("Agg")
plt = mpl.pyplot


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    signal_objective: str,
    multiclass: bool,
    all_outs: bool = True,
    jshift: str = "",
):
    """
    Add BDT scores to events DataFrame.

    Now uses signal_objective (e.g., 'ggfbbtt', 'vbfbbtt') to namespace all BDT columns,
    so multiple models can coexist in the same DataFrame.

    Assumes class mapping to be:
    Class 0: bbtthe
    Class 1: bbtthh
    Class 2: bbtthm
    Class 3: dyjets
    Class 4: qcd
    Class 5: ttbarhad
    Class 6: ttbarll
    Class 7: ttbarsl

    Args:
        events: DataFrame to add scores to
        sample_bdt_preds: BDT predictions array
        signal_objective: Signal objective (e.g., 'ggfbbtt', 'vbfbbtt') to namespace columns
        multiclass: Whether this is multiclass classification
        all_outs: Whether to add all output scores
        jshift: JEC/JMSR shift label
    """

    if jshift != "":
        jshift = "_" + jshift

    prefix = _get_bdt_key(signal_objective, prefix_only=True)

    if not multiclass:
        events[prefix + jshift] = sample_bdt_preds
    else:
        he_score = sample_bdt_preds[:, 0]  # TODO could be de-hardcoded
        hh_score = sample_bdt_preds[:, 1]
        hm_score = sample_bdt_preds[:, 2]

        events[f"{prefix}tauhtauh{jshift}"] = hh_score
        events[f"{prefix}tauhtaum{jshift}"] = hm_score
        events[f"{prefix}tauhtaue{jshift}"] = he_score

        if all_outs:
            events[f"{prefix}DY{jshift}"] = sample_bdt_preds[:, 3]
            events[f"{prefix}QCD{jshift}"] = sample_bdt_preds[:, 4]
            events[f"{prefix}TThad{jshift}"] = sample_bdt_preds[:, 5]
            events[f"{prefix}TTll{jshift}"] = sample_bdt_preds[:, 6]
            events[f"{prefix}TTSL{jshift}"] = sample_bdt_preds[:, 7]

        for ch in CHANNELS.values():
            taukey = ch.tagger_label
            base_sig = events[f"{prefix}{taukey+jshift}"]
            events[f"{prefix}{taukey+jshift}vsQCD"] = np.nan_to_num(
                base_sig / (base_sig + events[f"{prefix}QCD{jshift}"]),
                nan=PAD_VAL,
            )

            events[f"{prefix}{taukey+jshift}vsAll"] = np.nan_to_num(
                base_sig
                / (
                    base_sig
                    + events[f"{prefix}QCD{jshift}"]
                    + events[f"{prefix}TThad{jshift}"]
                    + events[f"{prefix}TTll{jshift}"]
                    + events[f"{prefix}TTSL{jshift}"]
                    + events[f"{prefix}DY{jshift}"]
                ),
                nan=PAD_VAL,
            )


def compute_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    signal_objective: str,
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

    # add some flexibility to the modelname
    if modelname not in bdt_config and modelname.endswith("bbtt"):
        if modelname.endswith("ggfbbtt"):
            config_modelname = modelname.removesuffix("_ggfbbtt")
        elif modelname.endswith("vbfbbtt"):
            config_modelname = modelname.removesuffix("_vbfbbtt")
        else:
            raise ValueError(f"Modelname {modelname} does not end with ggfbbtt or vbfbbtt")

    else:
        config_modelname = modelname

    feature_names = [
        feat
        for cat in bdt_config[config_modelname]["train_vars"]
        for feat in bdt_config[config_modelname]["train_vars"][cat]
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
                signal_objective=signal_objective,
                multiclass=True,
                all_outs=True,
            )

            if save_dir:
                # Updated directory structure: modelname/signal/channel/year/

                if test_mode:
                    tag = "test"
                elif tt_pres:
                    tag = "tt_pres"
                else:
                    tag = "full_presel"

                file_dir = Path(save_dir) / tag / signal_objective / modelname / channel.key / year
                file_dir.mkdir(exist_ok=True, parents=True)
                with (file_dir / f"{sample}_preds.pkl").open("wb") as f:
                    pickle.dump(y_pred, f)
                with (file_dir / f"{sample}_shapes.pkl").open("wb") as f:
                    pickle.dump(events_dict[year][sample].events.shape, f)


def check_bdt_prediction_shapes(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_objective: str,
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
            # Updated directory structure: tag/signal/modelname/channel/year/
            if test_mode:
                tag = "test"
            elif tt_pres:
                tag = "tt_pres"
            else:
                tag = "full_presel"

            shape_file = (
                Path(bdt_preds_dir)
                / tag
                / signal_objective
                / modelname
                / channel.key
                / year
                / f"{sample}_shapes.pkl"
            )
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
    signal_objective: str,
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool = False,
    all_outs: bool = True,
):
    """Load BDT predictions from disk and add scores to events.

    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.
    If ``jec_jmsr_shifts``, also loads BDT preds for every JEC / JMSR shift in MC.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt')
        channel: Channel object with .key attribute
        bdt_preds_dir: Directory containing BDT predictions
        test_mode: Whether in test mode
        all_outs: Whether to load all outputs
    """

    for year in events_dict:
        for sample in events_dict[year]:
            # Updated directory structure: modelname/signal/channel/year/
            if test_mode:
                tag = "test"
            elif tt_pres:
                tag = "tt_pres"
            else:
                tag = "full_presel"

            pred_file = (
                Path(bdt_preds_dir)
                / tag
                / signal_objective
                / modelname
                / channel.key
                / year
                / f"{sample}_preds.pkl"
            )

            bdt_preds = pickle.load(pred_file.open("rb"))

            multiclass = len(bdt_preds.shape) > 1
            _add_bdt_scores(
                events_dict[year][sample].events,
                bdt_preds,
                signal_objective=signal_objective,
                multiclass=multiclass,
                all_outs=all_outs,
            )


def compute_or_load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    signal_objective: str,
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool = False,
    at_inference: bool = False,
    all_outs: bool = True,
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
        all_outs: Whether to load/compute all outputs

    Example:
        >>> compute_or_load_bdt_preds(
        ...     events_dict=my_events,
        ...     modelname="29July25_loweta_lowreg",
        ...     model_dir=Path("/path/to/models"),
        ...     signal_objective="ggfbbtt",
        ...     channel=my_channel,
        ...     bdt_preds_dir=Path("/path/to/predictions"),
        ...     at_inference=False,  # Will try to load if available
        ... )
    """
    import time

    if at_inference:
        # Compute predictions directly at inference time
        print(f"Computing BDT predictions at inference for signal '{signal_objective}'...")
        start_time = time.time()
        compute_bdt_preds(
            events_dict=events_dict,
            modelname=modelname,
            model_dir=model_dir,
            signal_objective=signal_objective,
            channel=channel,
            tt_pres=tt_pres,
            test_mode=test_mode,
            save_dir=bdt_preds_dir,
        )
        print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")
    else:
        # Try to load existing predictions
        print(f"Checking for existing BDT predictions for signal '{signal_objective}'...")
        shapes_ok = check_bdt_prediction_shapes(
            events_dict=events_dict,
            modelname=modelname,
            signal_objective=signal_objective,
            channel=channel,
            tt_pres=tt_pres,
            bdt_preds_dir=bdt_preds_dir,
            test_mode=test_mode,
        )

        if shapes_ok:
            print(f"Loading existing BDT predictions for signal '{signal_objective}'...")
            load_bdt_preds(
                events_dict=events_dict,
                modelname=modelname,
                signal_objective=signal_objective,
                channel=channel,
                tt_pres=tt_pres,
                bdt_preds_dir=bdt_preds_dir,
                test_mode=test_mode,
                all_outs=all_outs,
            )
            print("BDT predictions loaded successfully")
        else:
            print("BDT predictions don't exist or shapes don't match. " "Computing predictions...")
            start_time = time.time()
            compute_bdt_preds(
                events_dict=events_dict,
                modelname=modelname,
                model_dir=model_dir,
                signal_objective=signal_objective,
                channel=channel,
                tt_pres=tt_pres,
                test_mode=test_mode,
                save_dir=bdt_preds_dir,
            )
            print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")
