from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xgboost as xgb
from boostedhh.utils import PAD_VAL

from bbtautau.bbtautau_utils import Channel
from bbtautau.postprocessing.bdt_config import bdt_config
from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.Samples import CHANNELS
from bbtautau.postprocessing.utils import LoadedSample

from .bdt import Trainer

# Non-interactive backend for batch/containers
mpl.use("Agg")
plt = mpl.pyplot


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _add_bdt_scores(
    events: pd.DataFrame,
    sample_bdt_preds: np.ndarray,
    signal_key: str,
    multiclass: bool,
    all_outs: bool = True,
    jshift: str = "",
):
    """
    Add BDT scores to events DataFrame.

    Now uses signal_key (e.g., 'ggfbbtt', 'vbfbbtt') to namespace all BDT columns,
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
        signal_key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt') to namespace columns
        multiclass: Whether this is multiclass classification
        all_outs: Whether to add all output scores
        jshift: JEC/JMSR shift label
    """

    if jshift != "":
        jshift = "_" + jshift

    # Prefix all BDT columns with the signal key to distinguish models
    # Remove trailing 'tt' from signal_key to avoid redundancy (e.g., ggfbbtt -> ggfbb)
    # This gives cleaner names like BDTggfbbtauhtauh instead of BDTggfbbtttauhtauh
    signal_base = signal_key.removesuffix("tt") if signal_key.endswith("tt") else signal_key
    prefix = f"BDT{signal_base}"

    if not multiclass:
        events[f"{prefix}{jshift}"] = sample_bdt_preds
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
    signal_key: str,
    tt_pres: bool,
    channel: Channel,
    test_mode: bool = False,
    save_dir: Path = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute BDT predictions for multiple years and samples.

    This function loads a trained XGBoost model and uses it to make predictions on multiple
    years and samples of data. The input data is expected to be organized by year and sample name.

    The predictions are saved in the directory structure:
        save_dir / tag / signal_key / modelname / channel.key / year / {sample}_preds.pkl

    Args:
        events_dict: Dictionary mapping years to dictionaries of samples. Each sample should be a
            LoadedSample object containing the features needed for prediction.
        modelname: Name of the model to load
        model_dir: Path to the directory containing the model
        signal_key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt') for organizing saved predictions
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
                signal_key=signal_key,
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

                file_dir = Path(save_dir) / tag / signal_key / modelname / channel.key / year
                file_dir.mkdir(exist_ok=True, parents=True)
                with (file_dir / f"{sample}_preds.pkl").open("wb") as f:
                    pickle.dump(y_pred, f)
                with (file_dir / f"{sample}_shapes.pkl").open("wb") as f:
                    pickle.dump(events_dict[year][sample].events.shape, f)


def check_bdt_prediction_shapes(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    signal_key: str,
    tt_pres: bool,
    channel: Channel,
    bdt_preds_dir: Path,
    test_mode: bool,
) -> bool:
    """Check if BDT predictions exist and have correct shapes.

    Args:
        events_dict: Dictionary with structure {year: {sample: LoadedSample}}
        modelname: Name of the BDT model
        signal_key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt')
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
                / signal_key
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
    signal_key: str,
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
                / signal_key
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
                signal_key=signal_key,
                multiclass=multiclass,
                all_outs=all_outs,
            )


def compute_or_load_bdt_preds(
    events_dict: dict[str, dict[str, LoadedSample]],
    modelname: str,
    model_dir: Path,
    signal_key: str,
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
        signal_key: Signal key (e.g., 'ggfbbtt', 'vbfbbtt')
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
        ...     signal_key="ggfbbtt",
        ...     channel=my_channel,
        ...     bdt_preds_dir=Path("/path/to/predictions"),
        ...     at_inference=False,  # Will try to load if available
        ... )
    """
    import time

    if at_inference:
        # Compute predictions directly at inference time
        print(f"Computing BDT predictions at inference for signal '{signal_key}'...")
        start_time = time.time()
        compute_bdt_preds(
            events_dict=events_dict,
            modelname=modelname,
            model_dir=model_dir,
            signal_key=signal_key,
            channel=channel,
            tt_pres=tt_pres,
            test_mode=test_mode,
            save_dir=bdt_preds_dir,
        )
        print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")
    else:
        # Try to load existing predictions
        print(f"Checking for existing BDT predictions for signal '{signal_key}'...")
        shapes_ok = check_bdt_prediction_shapes(
            events_dict=events_dict,
            modelname=modelname,
            signal_key=signal_key,
            channel=channel,
            tt_pres=tt_pres,
            bdt_preds_dir=bdt_preds_dir,
            test_mode=test_mode,
        )

        if shapes_ok:
            print(f"Loading existing BDT predictions for signal '{signal_key}'...")
            load_bdt_preds(
                events_dict=events_dict,
                modelname=modelname,
                signal_key=signal_key,
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
                signal_key=signal_key,
                channel=channel,
                tt_pres=tt_pres,
                test_mode=test_mode,
                save_dir=bdt_preds_dir,
            )
            print(f"BDT predictions computed in {time.time() - start_time:.2f} seconds")


def compare_models(
    models: list[str],
    model_dirs: list[str],
    years: list[str],
    signal_key: str,
    samples: list[str] | None = None,
    data_path: str | None = None,
    tt_preselection: bool = False,
    output_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """Load multiple trained models, evaluate, and produce comparison outputs.

    - Saves per-signal overlay ROC plots across models (when comparable)
    - Saves a CSV with metrics for each (model, signal)

    Returns a nested dict: metrics_by_model[model][signal] -> metrics_dict
    """

    if samples is None:
        samples = ["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]

    # Use a top-level comparison directory
    base_out = Path(output_dir) if output_dir is not None else Path("comparison")
    _ensure_dir(base_out)

    # Build a base trainer for data loading and getting sample information
    base_trainer = Trainer(
        years=list(years),
        signal_key=signal_key,
        bkg_sample_names=list(samples),
        modelname=models[0],
        output_dir=model_dirs[0],
        data_path=data_path,
        tt_preselection=tt_preselection,
    )
    # Load data only once
    base_trainer.load_data(force_reload=True)

    # Load boosters and prepare training sets for all models
    trainers: dict[str, Trainer] = {}
    for model, model_dir in zip(models, model_dirs):
        tr = Trainer(
            years=list(years),
            signal_key=signal_key,
            bkg_sample_names=list(samples),
            modelname=model,
            output_dir=model_dir,
            data_path=data_path,
            tt_preselection=tt_preselection,
        )
        # Share the loaded raw events data from base_trainer instead of reloading
        tr.events_dict = base_trainer.events_dict
        tr.samples = base_trainer.samples
        tr.sample_names = base_trainer.sample_names
        # Prepare training set for this specific model (will use its own feature set)
        tr.prepare_training_set(save_buffer=False)
        tr.load_model()
        trainers[model] = tr

    # Construct combined preds_dict holding per-class events with per-model scores
    event_filters = {
        name: base_trainer.dval.get_label() == i for i, name in enumerate(base_trainer.sample_names)
    }

    preds_dict: dict[str, LoadedSample] = {}

    # Start with weights only; features are not required for ROCAnalyzer
    for class_name in base_trainer.sample_names:
        preds_dict[class_name] = LoadedSample(
            sample=base_trainer.samples[class_name],
            events=None,
        )

    # Fill events DataFrames incrementally with model-specific columns
    for model, tr in trainers.items():
        # Use the model-specific prepared dval for predictions
        y_pred = tr.bst.predict(tr.dval)
        # Sanity check: classifier outputs must match the class set used by base_trainer
        if y_pred.ndim != 2 or y_pred.shape[1] != len(base_trainer.sample_names):
            raise ValueError(
                f"Model '{model}' produces {y_pred.shape[1] if y_pred.ndim==2 else 'invalid'} classes,"
                f" but base comparison expects {len(base_trainer.sample_names)}."
                " Ensure models were trained with the same class set/order (signals per channel + backgrounds)."
            )
        for class_index, class_name in enumerate(base_trainer.sample_names):
            mask = event_filters[class_name]
            # Initialize events frame if needed
            if preds_dict[class_name].events is None:
                preds_dict[class_name].events = {
                    "finalWeight": base_trainer.dval.get_weight()[mask],
                }
            # Add per-model per-class score column
            y_pred_class = y_pred[mask, class_index]
            preds_dict[class_name].events[f"{model}::{class_name}"] = y_pred_class

    # Convert dicts to DataFrames
    for class_name in preds_dict:
        import pandas as pd

        preds_dict[class_name].events = pd.DataFrame(preds_dict[class_name].events)

    # Prepare ROC analyzer with all signals and backgrounds
    signal_names = [
        sig_name for sig_name in base_trainer.samples if base_trainer.samples[sig_name].isSignal
    ]
    background_names = [
        bkg_name for bkg_name in base_trainer.samples if not base_trainer.samples[bkg_name].isSignal
    ]

    roc_analyzer = ROCAnalyzer(
        years=base_trainer.years,
        signals={sig: preds_dict[sig] for sig in signal_names},
        backgrounds={bkg: preds_dict[bkg] for bkg in background_names},
    )

    # Register discriminants: for each signal and model, build vsAll background discriminant
    for sig_name in signal_names:
        for model in models:
            sig_tagger = f"{model}::{sig_name}"
            bkg_taggers = [f"{model}::{bkg}" for bkg in background_names]
            roc_analyzer.process_discriminant(
                signal_name=sig_name,
                background_names=background_names,
                signal_tagger=sig_tagger,
                background_taggers=bkg_taggers,
                custom_name=f"{model} {sig_name[-2:]}vsAll",
            )

    # Compute and plot overlay ROCs per signal
    roc_analyzer.compute_rocs()

    metrics_by_model: dict[str, dict[str, dict[str, float]]] = {m: {} for m in models}
    for sig_name in signal_names:
        # Collect discriminant names for this signal
        disc_names = [
            disc.name
            for disc in roc_analyzer.discriminants.values()
            if disc.signal_name == sig_name
        ]
        # Plot overlay
        out_dir = _ensure_dir(base_out / "rocs")
        roc_analyzer.plot_rocs(title=f"Compare {sig_name}", disc_names=disc_names, plot_dir=out_dir)

        # Extract per-disc metrics and group by model
        for disc in [d for d in roc_analyzer.discriminants.values() if d.signal_name == sig_name]:
            model = disc.name.split()[0] if " " in disc.name else disc.name
            if hasattr(disc, "get_metrics"):
                metrics_by_model[model][sig_name] = disc.get_metrics(as_dict=True)

    # Save combined metrics CSV
    csv_path = base_out / "comparison_metrics.csv"
    with csv_path.open("w") as f:
        all_metric_keys = [
            "roc_auc",
            "pr_auc",
            "f1_score",
            "precision",
            "recall",
            "f1_score_05",
            "balanced_accuracy",
            "matthews_corr",
            "optimal_threshold",
        ]
        header = ["model", "signal"] + all_metric_keys
        f.write(",".join(header) + "\n")
        for model, by_signal in metrics_by_model.items():
            for signal, m in by_signal.items():
                row = [model, signal] + [f"{m.get(k, 0):.6f}" for k in all_metric_keys]
                f.write(",".join(row) + "\n")

    # Save a JSON index for easy inspection
    with (base_out / "comparison_index.json").open("w") as jf:
        json.dump({"models": models, "years": years, "signals": list(signal_names)}, jf, indent=2)

    return metrics_by_model
