# Sensitivity Study Script

This script performs sensitivity optimization for the HH→bbττ analysis, including cut optimization, ROC curves, and evaluation at specific working points.

## Quick Start

```bash
# Basic sensitivity optimization
python SensitivityStudy.py --actions sensitivity --channels he hmu

# Evaluate at specific cuts
python SensitivityStudy.py --actions evaluate \
    --cuts-file path/to/opt_results.csv \
    --eval-bmin 10 \
    --outfile evaluation_results.csv
```

---

## Arguments Reference

### Data Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--years` | `2022 2022EE 2023 2023BPix` | Years to include in analysis |
| `--channels` | all channels | Channels to process (e.g., `he`, `hmu`, `emu`) |
| `--test-mode` | `False` | Run with reduced data for quick testing |
| `--tt-pres` | `False` | Apply ττ preselection cuts |

### Action Selection

| Argument | Required | Description |
|----------|----------|-------------|
| `--actions` | **Yes** | One or more actions to perform |

**Available actions:**
- `compute_rocs` - Compute and plot ROC curves (can combine with others)
- `plot_mass` - Plot mass distributions (can combine with others)
- `sensitivity` - Run grid search optimization (**mutually exclusive**)
- `fom_study` - Compare different figures of merit (**mutually exclusive**)
- `evaluate` - Evaluate at specific cuts from CSV (**mutually exclusive**)

### Evaluate Action

*Required when using `--actions evaluate`*

| Argument | Default | Description |
|----------|---------|-------------|
| `--cuts-file` | None | Path to CSV file with cuts |
| `--eval-bmin` | None | B_min column to read (e.g., `10` reads column `Bmin=10`) |
| `--outfile` | None | Output CSV path (appends if file exists) |

**CSV file format:**
```
,Bmin=1,Bmin=10
Cut_Xbb,0.92,0.95
Cut_Xtt,0.85,0.88
...
```

### Sensitivity Action

*Used with `--actions sensitivity`*

| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptive` | `False` | Use adaptive multi-stage grid refinement |
| `--use-thresholds` | `False` | Optimize in threshold space (default: signal efficiency) |
| `--dataMinusSimABCD` | `False` | Use enhanced ABCD: subtract simulated non-QCD backgrounds |
| `--showNonDataDrivenPortion` | `True` | Include non-QCD (ttbar) background in results |

### Signal Region Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-sm-signals` | `False` | Optimize for sum of SM signals (ggF+VBF) |
| `--do-vbf` | `False` | Also optimize VBF region (runs after ggF, applies veto) |
| `--overlapping-channels` | `False` | Allow channels to overlap (no vetoes) |
| `--bmin-for-veto` | `10` | B_min value for cross-region vetoes |

### Discriminator Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-ParT` | `False` | Use ParT scores instead of BDT |
| `--ggf-modelname` | `19oct25_ak4away_ggfbbtt` | BDT model for ggF channel |
| `--vbf-modelname` | `19oct25_ak4away_vbfbbtt` | BDT model for VBF channel |
| `--bdt-dir` | None | Directory with BDT model files |
| `--at-inference` | `False` | Force BDT inference (ignore cache) |
| `--llsl-weight` | `1.0` | Weight for TTSL/TTLL in ττ discriminant |
| `--bb-disc` | `bbFatJetParTXbbvsQCDTop` | bb discriminator variable |

---

## Usage Examples

### 1. Basic Sensitivity Optimization

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --channels he hmu \
    --years 2022 2022EE 2023 2023BPix
```

### 2. Adaptive Grid Search

Uses multi-stage refinement for better precision:

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --adaptive \
    --channels he
```

### 3. Using ParT Instead of BDT

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --use-ParT \
    --channels he hmu
```

### 4. Enhanced ABCD Method

Subtract simulated non-QCD backgrounds from data in ABCD estimation:

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --dataMinusSimABCD \
    --channels he
```

### 5. Evaluate at Specific Cuts

Evaluate signal/background yields at cuts from a previous optimization:

```bash
python SensitivityStudy.py \
    --actions evaluate \
    --cuts-file plots/SensitivityStudy/2025-01-01/full_presel/grid/BDT/ggf_only/ggfbbtt/he/2022_2022EE_2023_2023BPix_opt_results_2sqrtB_S_var_sigeff.csv \
    --eval-bmin 10 \
    --outfile my_evaluation.csv \
    --channels he hmu
```

### 6. Quick Test Run

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --test-mode \
    --channels he
```

### 7. VBF + ggF Orthogonal Optimization

```bash
python SensitivityStudy.py \
    --actions sensitivity \
    --do-vbf \
    --bmin-for-veto 10 \
    --channels he hmu
```

### 8. Combine Actions

ROC curves + mass plots + sensitivity (ROC/mass run first):

```bash
python SensitivityStudy.py \
    --actions compute_rocs plot_mass sensitivity \
    --channels he
```

---

## Output Files

### Sensitivity Action

Creates in `plots/SensitivityStudy/{date}/{config}/`:
- `{years}_opt_results_{fom}_sigeff.csv` - Optimization results
- `{years}_sigeff.pdf` - 2D optimization contour plots

### Evaluate Action

With `--outfile`:
- CSV with columns: `region`, `channel`, `years`, `cuts_0`, `cuts_1`, `sig_pass`, `sig_eff`, `bkg_ABCD`, `TF_data`, `fom_*`, ...

---

## ABCD Method

The script uses the ABCD method for background estimation:

```
           High score (pass)    Low score (fail)
          ┌─────────────────┬─────────────────┐
Resonant  │   A (signal)    │       C         │
mass      ├─────────────────┼─────────────────┤
Sideband  │       B         │       D         │
          └─────────────────┴─────────────────┘

Background in A = B × (C/D) = B × TF
```

With `--dataMinusSimABCD`: Subtracts simulated ttbar/other backgrounds from B, C, D before computing transfer factor.

---

## Figure of Merit

Default FOM is `2sqrtB_S_var` (approximate expected limit):

```
FOM = 2 × sqrt(B + σ_B²) / S
```

where:
- `S` = signal yield
- `B` = background estimate from ABCD
- `σ_B` = uncertainty on B
