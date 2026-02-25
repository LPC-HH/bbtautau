# BDT Model Configurations

Each BDT model configuration is stored in a separate Python file in this directory.

## File Naming

Config files must be named `config_{modelname}.py` where `modelname` matches the `modelname` field in the CONFIG dictionary. Use underscores for special characters.

Examples:
- `config_29July25_loweta_lowreg.py` → modelname: `"29July25_loweta_lowreg"`
- `config_6Feb26test.py` → modelname: `"6Feb26test"`

## Config Structure

Each file must export a `CONFIG` dictionary with the following structure:

```python
CONFIG = {
    "modelname": "your_model_name",
    "signals": ["ggfbbtt"],  # or ["ggfbbtt", "vbfbbtt"] for unified models
    "n_folds": 1,  # or >1 for k-fold cross-validation
    "hyperpars": {
        "objective": "multi:softprob",
        "max_depth": 12,
        "eta": 0.07,
        "num_class": 8,  # 8 for single signal, 11 for unified (ggf+vbf)
        # ... other XGBoost parameters
    },
    "num_rounds": 100,
    "test_size": 0.2,
    "random_seed": 42,
    "train_vars": {
        "fatjet": [...],
        "leptons": [...],
        "misc": [...],
    },
}
```

## Usage

Configs are loaded on-demand when accessed via `BDT_CONFIG[modelname]`:

```python
from bbtautau.postprocessing.bdt_config import BDT_CONFIG

config = BDT_CONFIG["29July25_loweta_lowreg"]
```

## Configuration Requirements

- `signals`: List with 1 or 2 signal keys
  - 1 signal → 8 classes (3 signal channels + 5 backgrounds)
  - 2 signals → 11 classes (3 ggf + 3 vbf + 5 backgrounds)
- `hyperpars.num_class`: Must match `(n_signals * 3 + 5)`
- Background samples are fixed: `["dyjets", "qcd", "ttbarhad", "ttbarll", "ttbarsl"]`
