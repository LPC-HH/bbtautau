"""
BDT configuration for 3Nov26_wp0p999_newCA_vbfk2v0 (vbfbbtt-k2v0).

Copy of May4_optimized_vbfk2v0 (same hyperparameters, 4-fold) with two changes:
- CA mass input: ttFatJetCAglobalParT_massVisApplied -> ..._with_delta_axis_merged (newest CA mass);
- GloParT tt inputs binned with a single WP edge at 0.999 (0/1 flag), via bin_features + wps_ttpart.
"""

from __future__ import annotations

CONFIG = {
    "modelname": "3Nov26_wp0p999_newCA_vbfk2v0",
    "signals": ["vbfbbtt-k2v0"],
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
        "num_class": 8,
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
            "ttFatJetCAglobalParT_massVisApplied_with_delta_axis_merged",
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
    # one GloParT WP: each tt score becomes a 0/1 flag at 0.999 (np.digitize with edges [0.999])
    "bin_features": [
        "ttFatJetParTXtauhtauhvsQCDTop",
        "ttFatJetParTXtauhtaumvsQCDTop",
        "ttFatJetParTXtauhtauevsQCDTop",
    ],
    "wps_ttpart": {
        "ttFatJetParTXtauhtauhvsQCDTop": [0.999],
        "ttFatJetParTXtauhtaumvsQCDTop": [0.999],
        "ttFatJetParTXtauhtauevsQCDTop": [0.999],
    },
}
