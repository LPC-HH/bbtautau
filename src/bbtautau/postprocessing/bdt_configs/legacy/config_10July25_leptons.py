"""
BDT configuration for 10July25_leptons model.

Model with ggF signal including lepton variables.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "10July25_leptons",
    "signals": ["ggfbbtt"],
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 12,
        "subsample": 0.2,
        "alpha": 8.0,
        "gamma": 4.0,
        "lambda": 4.0,
        "colsample_bytree": 1.0,
        "num_parallel_tree": 70,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": 8,
    },
    "num_rounds": 50,
    "test_size": 0.2,
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
