"""
BDT configuration for 20aug25_loweta_lowreg_vbf model.

VBF-only model with low eta and low regularization settings.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "20aug25_loweta_lowreg_vbf",
    "signals": ["vbfbbtt"],
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
        "device": "cuda",
    },
    "num_rounds": 100,
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
            "ttElectronDeltaEta",
            "ttElectronDeltaPhi",
            "ttElectron_dRak8Jet",
            "ttElectronMass",
            "ttElectroncharge",
            "ttMuonPt",
            "ttMuonDeltaEta",
            "ttMuonDeltaPhi",
            "ttMuon_dRak8Jet",
            "ttMuonMass",
            "ttMuoncharge",
        ],
        "misc": ["METPt", "METPhi", "ht"],
    },
}
