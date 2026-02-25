"""
BDT configuration for 11Feb26Full model.

Unified model with both ggF and VBF signals, using 5-fold cross-validation.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "17Feb26Full",
    "signals": ["ggfbbtt", "vbfbbtt"],
    "n_folds": 5,  # 5-fold cross-validation
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 12,
        "eta": 0.07,
        "subsample": 0.2,
        "alpha": 1.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "colsample_bytree": 0.6,
        "num_parallel_tree": 100,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": 11,
        "device": "cuda",
    },
    "num_rounds": 200,
    "test_size": 0.2,
    "random_seed": 42,
    "var_classes": ["fatjet", "VBFJets", "leptons", "misc"],
    "train_vars": {
        "fatjet": [
            "ttFatJetPt",
            "ttFatJetPhi",
            "ttFatJetEta",
            "ttFatJetCAglobalParT_massVisApplied",
            "ttFatJetParTXtauhtauhvsQCDTop",
            "ttFatJetParTXtauhtauevsQCDTop",
            "ttFatJetParTXtauhtaumvsQCDTop",
            "ttFatJetTau3OverTau2",
        ],
        "VBFJets": ["VBFJetDeltaEta", "VBFMassjj"],
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
        "away": [
            "ttJetAwayPt",
            "ttJetAwayPhi",
            "ttJetAwayEta",
            "ttJetAwayMass",
        ],
        "misc": ["METPt", "METPhi", "ht"],
    },
}
