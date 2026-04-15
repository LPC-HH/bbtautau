# HHbbtautau


[![Actions Status][actions-badge]][actions-link]
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/LPC-HH/bbtautau/main.svg)](https://results.pre-commit.ci/latest/github/LPC-HH/bbtautau/main)
<!-- [![Documentation Status][rtd-badge]][rtd-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LPC-HH/bbtautau/workflows/CI/badge.svg
[actions-link]:             https://github.com/LPC-HH/bbtautau/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/bbtautau
[conda-link]:               https://github.com/conda-forge/bbtautau-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/LPC-HH/bbtautau/discussions
[pypi-link]:                https://pypi.org/project/bbtautau/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/bbtautau
[pypi-version]:             https://img.shields.io/pypi/v/bbtautau
[rtd-badge]:                https://readthedocs.org/projects/bbtautau/badge/?version=latest
[rtd-link]:                 https://bbtautau.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to two beauty quarks (b) and two tau leptons.


- [HHbbtautau](#hhbbtautau)
  - [Setting up package](#setting-up-package)
    - [Creating a virtual environment](#creating-a-virtual-environment)
    - [Installing package](#installing-package)
    - [Troubleshooting](#troubleshooting)
  - [Running coffea processors](#running-coffea-processors)
    - [Setup](#setup)
    - [Running locally](#running-locally)
    - [Condor jobs](#condor-jobs)
    - [Checking jobs](#checking-jobs)
  - [Postprocessing](#postprocessing)
    - [Trigger study](#trigger-study)
    - [Sensitivity study](#sensitivity-study)
    - [Control plots](#control-plots)
    - [BDT](#bdt)
    - [Kubernetes: generate BDT jobs from templates](#kubernetes-generate-bdt-jobs-from-templates)
    - [Templates](#templates)
  - [Datacard and fits](#datacard-and-fits)
    - [CMSSW + Combine Quickstart](#cmssw--combine-quickstart)
    - [Create datacards](#create-datacards)
    - [Combine scripts](#combine-scripts)
    - [Postfit plots](#postfit-plots)
  - [Transferring files to FNAL with Rucio](#transferring-files-to-fnal-with-rucio)


## Setting up package

### Creating a virtual environment

First, create a virtual environment (`micromamba` is recommended):

```bash
# Clone the repository
git clone --recursive https://github.com/LPC-HH/bbtautau.git
cd bbtautau
# Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
# Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
micromamba env create -f environment.yaml
micromamba activate hh
```

### Installing package

**Remember to install this in your mamba environment**.

```bash
# Clone the repsitory as above if you haven't already
# Perform an editable installation
pip install -e .
# for committing to the repository
pip install pre-commit
pre-commit install
# Install as well the common HH utilities
cd boostedhh
pip install -e .
cd ..
```

### Troubleshooting

- If your default `python` in your environment is not Python 3, make sure to use
  `pip3` and `python3` commands instead.

- You may also need to upgrade `pip` to perform the editable installation:

```bash
python3 -m pip install -e .
```

## Running coffea processors

### Setup

For submitting to condor, all you need is python >= 3.7.

For running locally, follow the same virtual environment setup instructions
above and activate the environment.

```bash
micromamba activate hh
```

Clone the repository:

```
git clone https://github.com/LPC-HH/bbtautau/
pip install -e .
```

### Running locally

For testing, e.g.:

```bash
python src/run.py --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00 --starti 0 --endi 1 --year 2022 --processor skimmer
```

### Condor jobs

A single sample / subsample:

```bash
python src/condor/submit.py --analysis bbtautau --git-branch BRANCH-NAME --site ucsd --save-sites ucsd lpc --processor skimmer --samples HHbbtt --subsamples  GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00 --files-per-job 5 --tag 24Nov7Signal [--submit]
```

Or from a YAML:

```bash
python src/condor/submit.py --yaml src/condor/submit_configs/25Apr5All.yaml --analysis bbtautau --git-branch addmc --site lpc --save-sites ucsd lpc --processor skimmer --tag 25Apr5AddVars --year 2022 [--submit]
```

### Checking jobs

e.g.


```bash
python boostedhh/condor/check_jobs.py --analysis bbtautau --tag 25Apr24_v12_private_signal --processor skimmer --check-running --year 2022EE
```


## Postprocessing

### Trigger study

Trigger efficiency studies can be performed using the `src/bbtautau/postprocessing/TriggerStudy.py` script. The main execution logic is within the `if __name__ == "__main__"` block, where you can configure the years and signal samples to process.

The script will:
- Load the specified signal samples.
- Define trigger sets and tagger configurations.
- Calculate and plot trigger efficiencies for different channels (`hh`, `hm`, `he`).
- Generate N-1 efficiency tables to study the impact of individual triggers.

To run the study, configure the desired `years` and `SIGNALS` inside the script and then execute it:
```bash
python src/bbtautau/postprocessing/TriggerStudy.py
```
Output plots and tables will be saved in the `plots/TriggerStudy/` directory.


### Sensitivity study

```bash
python SensitivityStudy.py --actions compute_rocs plot_mass sensitivity --years 2022 2023 --channels hh hm
```

Arguments

`--years` (list, default: 2022 2022EE 2023 2023BPix): List of years to include in the analysis.
`--channels` (list, default: hh hm he): List of channels to run (default: all).
`--test-mode` (flag, default: False): Run in test mode (reduced data size).
`--use-bdt` (flag, default: False): Use BDT model for sensitivity study.
`--modelname` (str, default: 28May25_baseline): Name of the BDT model to use.
`--at-inference` (flag, default: False): Compute BDT predictions at inference time.
`--actions` (list, required): Actions to perform. Choose one or more: `compute_rocs`, `plot_mass`, `sensitivity`, `time-methods`.


Example Commands

*Run an optimization analysis for all years and all channels, with the GloParT tautau tagger:*

`python SensitivityStudy.py --actions sensitivity`

*Run a full analysis for all years and all channels, using the BDT for the tautau jet:*

`python SensitivityStudy.py --actions compute_rocs plot_mass sensitivity`

*Run only on selected years/channels in test mode:*

`--test-mode` will reduce the data loading time significantly. Practical for testing.

`python SensitivityStudy.py --actions sensitivity --years 2022 --channels hh --test-mode`

Notes:
- by default uses ABCD background estimation method, and FOM = $\sqrt{b+\sigma_b}/s$
- by default uses parallel thread data loading and optimization

### Control plots

@Billy - convert into script and add instructions here

### BDT

`src/bbtautau/postprocessing/bdt.py` trains, evaluates, compares, and studies multiclass BDT models. Model settings are normally looked up by name under `src/bbtautau/postprocessing/bdt_configs/`, but training can also be driven by an explicit config file via `--config-file`.

Basic invocation:

```bash
python src/bbtautau/postprocessing/bdt.py [mode] --model <modelname>
```

or, for explicit config injection:

```bash
python src/bbtautau/postprocessing/bdt.py [mode] --config-file path/to/config_my_model.py
```

Important flags:

- `--years`: years to use, e.g. `--years 2022 2023` or `--years all`
- `--model`: model configuration name to load from `bdt_configs/`
- `--config-file`: explicit Python config file defining `CONFIG`; if passed together with `--model`, the two model names must match
- `--output-dir`: output directory for trained models, plots, comparisons, or prediction files
- `--data-path`: input data directory key/path
- `--force-reload`: force reloading the input data
- `--train` / `--load`: train a new model or load an existing one
- `--study-rescaling`: run the balance/rescaling study workflow
- `--eval-bdt-preds`: write BDT predictions for selected samples
- `--compare-models`: compare trained models using ROC overlays and CSV outputs
- `--compare-light`: compare trained models using only existing `metrics_summary.csv` files
- `--inputs`: paths to model JSON files and/or directories for comparison modes
- `--samples`: samples to evaluate with `--eval-bdt-preds`

One of `--model` or `--config-file` must be provided.

**Example: train a new model by name**

```bash
python src/bbtautau/postprocessing/bdt.py \
  --train \
  --years all \
  --model 26Mar26_optimized
```

**Example: train from an explicit config file**

```bash
python src/bbtautau/postprocessing/bdt.py \
  --train \
  --years all \
  --config-file src/bbtautau/postprocessing/bdt_configs/standard/config_26Mar26_optimized.py
```

**Evaluate predictions**

```bash
python src/bbtautau/postprocessing/bdt.py \
  --eval-bdt-preds \
  --years 2022 \
  --samples dyjets qcd ttbarhad ttbarll ttbarsl \
  --model 26Mar26_optimized \
  --output-dir /writable/output
```

This writes `BDT_predictions/<year>/<sample>/<model>_preds.npy` under `--output-dir` (or the default `DATA_DIR`).

**Compare multiple trained models**

```bash
python src/bbtautau/postprocessing/bdt.py \
  --compare-models \
  --years 2022 \
  --inputs \
    /bbtautauvol/bdt/training/no_presel/model_a \
    /bbtautauvol/bdt/training/no_presel/model_b \
  --output-dir comparison_out
```

This produces:
- Overlay ROC plots per signal in `comparison_out/rocs/`
- A consolidated CSV `comparison_out/comparison_metrics.csv`
- An index JSON `comparison_out/comparison_index.json`

Notes:
- Headless/containers: plotting uses a non-interactive backend (`Agg`), so no display server is needed.
- The model config determines the signal setup (`signals` field); that is no longer configured via a separate CLI flag.
- If Python cannot resolve internal modules, run from the repo root inside the project environment.


### Kubernetes: generate BDT jobs from templates

Use `src/bbtautau/kubernetes/jobs/make_from_template.py` to generate Kubernetes job YAMLs for training, comparison, lightweight comparison, or rescaling studies. It fills one of the templates in `src/bbtautau/kubernetes/jobs/` and writes YAMLs under `src/bbtautau/kubernetes/bdt_trainings/<job_type>/<presel>/<tag>/<job_name>.yml`.

Key flags:
- `--modelname`: training or rescaling model name
- `--config-file`: optional local config file to embed into a training job and pass to `bdt.py --config-file`
- `--compare-models`: switch to comparison mode (uses `template_compare.yaml`)
- `--compare-light`: switch to lightweight metrics-only comparison mode
- `--study-rescaling`: switch to rescaling-study mode
- `--inputs`: model JSON files and/or directories to compare
- `--years`: years to use for training/comparison (space-separated)
- `--samples`: samples to use in comparison/evaluation
- `--datapath`: data subdirectory on the PVC (joined to `/bbtautauvol`)
- `--train-args`: extra CLI args forwarded to `bdt.py` (quote this string)
- `--tt-preselection`: append flag into `train-args`
- `--job-name`: override auto-generated name (auto-generated names are lowercased)
- `--tag`: grouping tag used in the output path
- `--overwrite`: allow overwriting an existing YAML
- `--submit`: immediately `kubectl create -f <yaml>` in namespace `cms-ml`
- `--from-json`: load all args from a JSON file (keys match the CLI flags)

Training mode example:
```bash
python src/bbtautau/kubernetes/jobs/make_from_template.py \
  --modelname 26Mar26_optimized \
  --config-file src/bbtautau/postprocessing/bdt_configs/standard/config_26Mar26_optimized.py \
  --tag kfold5 \
  --datapath 26Mar5All_v12_private_signal \
  --train-args "--years 2022 2023" \
  --submit
```
This writes a training job YAML under `src/bbtautau/kubernetes/bdt_trainings/training/no_presel/kfold5/` and submits it. When `--config-file` is provided, the config is embedded in the generated job and passed to `bdt.py` inside the pod, so the config does not need to exist in the cloned repo checkout.

Comparison mode example:
```bash
python src/bbtautau/kubernetes/jobs/make_from_template.py \
  --compare-models \
  --inputs \
    training/no_presel/model_a \
    training/no_presel/model_b \
  --compare-tag july_vs_aug \
  --job-name cmp_july_aug_nopresel \
  --submit
```
The script auto-generates `job_name` when not provided:
- Training: based on `modelname`
- Comparison/light comparison: based on the compared input names or `compare_tag`
- Rescaling: `rescaling_<modelname>`

Generated YAMLs are grouped by job type and preselection state:

- `training/no_presel/<tag>/...`
- `training/tt_presel/<tag>/...`
- `comparisons/no_presel/<tag>/...`
- `rescaling/no_presel/<tag>/...`

You can also place all arguments in a JSON file and run:
```bash
python src/bbtautau/kubernetes/jobs/make_from_template.py --from-json my_job.json --submit
```
Where `my_job.json` can contain fields matching the CLI flags, such as `modelname`, `config_file`, `inputs`, `compare_models`, `compare_light`, `study_rescaling`, `years`, `tag`, `samples`, `datapath`, and `train_args`.


### Templates

These are made using the `postprocessing/postprocessing.py` script with the `--templates` option.
See `postprocessing/bash_scripts/MakeTemplates.sh` for an example.


## Datacard and fits

Foreword: when dealing with multiple signals and signal regions:
- to specify one or more signal processes to be included in the cards (e.g. ggf + SM vbf or just BSM vbf), specify the argument `--sigs [ggfbbtt, vbfbbtt, vbfbbttk2v0]
- to specify the strategy according to what we do in the `SensitivityStudy.py` step, i.e. using one signal region per channel (ggf) or using two regions per channel (ggf and vbf), we use the `--do-vbf` argument in `run_blinded_bbtt.sh` when running combine.
These past two items are independent: with either strategy, one can choose the signal samples to consider freely. (One should clearly not mix SM with BSM samples in the cards.)


### CMSSW + Combine Quickstart

**Warning: this should be done outside of your conda/mamba environment!**

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv
scram-venv
cmsenv
git clone -b v10.1.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git clone -b v3.0.0-pre1 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
# Important: this scram has to be run from src dir
scramv1 b clean; scramv1 b
pip3 install --upgrade rhalphalib
```

Then, install this repo as well:

```bash
```bash
cd /path/to/your/local/bbtautau/repo
pip3 install -e .
```

### Create datacards

After activating the above CMSSW environment (go inside the CMSSW folder and do `cmsenv`), you can use the CreateDatacard.py script as so (from your src/bbtautau folder):

```bash
python3 postprocessing/CreateDatacard.py --sigs ggfbbtt --templates-dir postprocessing/templates/25Apr25LudoCuts --model-name 25Apr25PassFix
```

By default, this will create datacards for all three channels summed across years in the `cards/model-name` directory.

As always, do the following to see a full list of options.

```bash
python3 postprocessing/CreateDatacard.py --help
```

### Combine scripts

All combine commands while blinded can be run via the `src/bbtautau/combine/run_blinded_bbtt.sh` script.

e.g. (always from inside the cards folders), this will combine the cards, create a workspace, do a background-only fit, and calculate expected limits:

```bash
run_blinded_bbtt.sh --workspace --bfit --limits
```

Another script, 'src/bbtautau/combine/run_blinded_bbtt_frzAllConstrainedNuisances.sh' can be used to fit with all constrained nuisances frozen.

See more comments inside the file.


I also add this to my .bashrc for convenience:

```
export PATH="$PATH:/home/user/rkansal/bbtautau/src/bbtautau/combine"
```

### Postfit plots

Run the following to run FitDiagnostics and save FitShapes:

```bash
run_blinded_bbtt.sh --workspace --dfit
```

Then see `postprocessing/PlotFits.ipynb` for plotting. **TODO:** convert into script!


## Transferring files to FNAL with Rucio

Set up Rucio following the [Twiki](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookFileTransfer#Option_1_Using_the_Command_L_AN1). Then:

```bash
rucio add-rule cms:/Tau/Run2022F-22Sep2023-v1/MINIAOD 1 T1_US_FNAL_Disk --activity "User AutoApprove" --lifetime 15552000 --ask-approval
```
