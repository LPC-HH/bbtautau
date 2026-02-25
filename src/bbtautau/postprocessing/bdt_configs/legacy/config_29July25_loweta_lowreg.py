"""
BDT configuration for 29July25_loweta_lowreg model.

Model with ggF signal, low eta and low regularization settings.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "29July25_loweta_lowreg",
    "signals": ["ggfbbtt"],
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 12,
        "eta": 0.07,
        "subsample": 0.2,
        "alpha": 1.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "colsample_bytree": 0.6,
        "num_parallel_tree": 20,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": 8,
    },
    "num_rounds": 300,
    "test_size": 0.2,
    "random_seed": 42,
    "var_classes": ["fatjet", "leptons", "misc"],
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
        ],
        "leptons": [
            "ttElectronPt",
            "ttElectronEta",
            "ttElectronPhi",
            "ttElectronMass",
            "ttMuonPt",
            "ttMuonEta",
            "ttMuonPhi",
            "ttMuonMass",
        ],
        "misc": ["METPt", "METPhi", "ht"],
    },
}
