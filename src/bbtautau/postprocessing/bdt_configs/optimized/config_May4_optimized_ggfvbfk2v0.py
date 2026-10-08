"""
BDT configuration: unified May4_optimized model with both ggF and VBF (k2v0) signals (11 classes).

ggF model with AK4 away jet variables included.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "May4_optimized_ggfvbfk2v0",
    "signals": ["ggfbbtt", "vbfbbtt-k2v0"],
    "n_folds": 4,
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 8,
        "eta": 0.08,
        "subsample": 0.6,
        "alpha": 4.5,
        "gamma": 4.0,
        "lambda": 4.0,
        "colsample_bytree": 0.8,
        "num_parallel_tree": 20,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "num_class": 11,
        "device": "cuda",
    },
    "num_rounds": 1000,
    "test_size": 0.2,
    "random_seed": 42,
    "train_vars": {
        "fatjet": [
            "ttFatJetPt",
            "ttFatJetPhi",
            "ttFatJetEta",
            "ttFatJetParTmassResApplied",
            "ttFatJetCAglobalParT_massVisApplied",
            "ttFatJetParTmassVisApplied",
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
            # "ttElectroncharge",
            "ttMuonPt",
            "ttMuonDeltaEta",
            "ttMuonDeltaPhi",
            "ttMuon_dRak8Jet",
            "ttMuonMass",
            # "ttMuoncharge",
        ],
        "away": [
            "ttJetAwayPt",
            "ttJetAwayPhi",
            "ttJetAwayEta",
            "ttJetAwayMass",
        ],
        "misc": ["METPt", "METPhi", "ht", "METsignificance"],
    },
}
