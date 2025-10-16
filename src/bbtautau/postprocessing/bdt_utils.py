from __future__ import annotations

import json
from pathlib import Path
from typing import dict, list

import matplotlib as mpl

from bbtautau.postprocessing.rocUtils import ROCAnalyzer
from bbtautau.postprocessing.utils import LoadedSample

from .bdt import Trainer

# Non-interactive backend for batch/containers
mpl.use("Agg")
plt = mpl.pyplot


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_models(
    models: list[str],
    save_dirs: list[str],
    years: list[str],
    signal_key: str,
    samples: list[str] | None = None,
    data_path: str | None = None,
    tt_preselection: bool = False,
    save_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """Load multiple trained models, evaluate, and produce comparison outputs.

    - Saves per-signal overlay ROC plots across models (when comparable)
    - Saves a CSV with metrics for each (model, signal)

    Returns a nested dict: metrics_by_model[model][signal] -> metrics_dict
    """

    if samples is None:
        samples = ["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]

    # Use a top-level comparison directory
    base_out = Path(save_dir) if save_dir is not None else Path("comparison")
    _ensure_dir(base_out)

    # Build a base trainer for data and validation split
    base_trainer = Trainer(
        years=list(years),
        signal_key=signal_key,
        bkg_sample_names=list(samples),
        modelname=models[0],
        save_dir=save_dirs[0],
        data_path=data_path,
        tt_preselection=tt_preselection,
    )
    base_trainer.load_data(force_reload=False)
    base_trainer.prepare_training_set(save_buffer=False)

    # Load boosters for all models
    boosters: dict[str, object] = {}
    for model, save_dir in zip(models, save_dirs):
        tr = Trainer(
            years=list(years),
            signal_key=signal_key,
            bkg_sample_names=list(samples),
            modelname=model,
            save_dir=save_dir,
            data_path=data_path,
            tt_preselection=tt_preselection,
        )
        boosters[model] = tr.load_model()

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
    for model, booster in boosters.items():
        y_pred = booster.predict(base_trainer.dval)
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
            preds_dict[class_name].events[f"{model}::{class_name}"] = y_pred[mask, class_index]

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
