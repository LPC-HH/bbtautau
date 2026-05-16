"""
BDT configuration for 28May25_baseline model.

Baseline model with ggF signal only.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "28May25_baseline",
    "signals": ["ggfbbtt"],
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 9,
        "subsample": 0.3,
        "alpha": 8.0,
        "gamma": 2.0,
        "lambda": 2.0,
        "colsample_bytree": 1.0,
        "num_parallel_tree": 100,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": 8,
    },
    "num_rounds": 30,
    "test_size": 0.8,
    "random_seed": 42,
    "train_vars": {
        "fatjet": [
            "ttFatJetPt",
            "ttFatJetPhi",
            "ttFatJetEta",
            "ttFatJetParTmassResApplied",
            "ttFatJetParTmassVisApplied",
            "ttFatJetParTQCD",
            "ttFatJetParTTop",
            "ttFatJetParTQCD0HF",
            "ttFatJetParTQCD1HF",
            "ttFatJetParTQCD2HF",
            "ttFatJetParTTopW",
            "ttFatJetParTTopbW",
            "ttFatJetParTXtauhtauh",
            "ttFatJetParTXtauhtaue",
            "ttFatJetParTXtauhtaum",
            "ttFatJetParTXbb",
        ],
        "misc": ["METPt", "METPhi"],
    },
}
