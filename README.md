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
above and install `coffea`

```bash
micromamba activate hh
pip install coffea
```

Clone the repository:

```
git clone https://github.com/LPC-HH/bbtautau/
pip install -e .
```

### Running locally

For testing, e.g.:

```bash
python src/run.py --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8 --starti 0 --endi 1 --year 2022 --processor skimmer
```

### Condor jobs

A single sample / subsample:

```bash
python src/condor/submit.py --analysis bbtautau --git-branch BRANCH-NAME --site ucsd --save-sites ucsd lpc --processor skimmer --samples HHbbtt --subsamples GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00_LHEweights_TuneCP5_13p6TeV_powheg-pythia8 --files-per-job 5 --tag 24Nov7Signal [--submit]
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

@Ludo - add instructions here.



### Sensitivity study

```bash
python SensitivityStudy.py --actions compute_rocs plot_mass sensitivity --years 2022 2023 --channels hh hm
```

Arguments

`--years` (list, default: 2022 2022EE 2023 2023BPix): List of years to include in the analysis.
`--channels` (list, default: hh hm he): List of channels to run (default: all).
`--test-mode` (flag, default: False): Run in test mode (reduced data size).
`--use_bdt` (flag, default: False): Use BDT model for sensitivity study.
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

This script provides a command-line interface to train, load, and evaluate a multiclass Boosted Decision Tree (BDT) model on data from one or more years. It includes options for studying rescaling effects, evaluating BDT predictions, and managing data reloading.

Data paths defined in `Trainer.__init__` in `Trainer.data_path` by year and sample type.

```
python bdt.py [options]
```

Options:

`--years`
Specify which years of data to store in Trainer object. This establishes which years of data are loaded for training/evaluation.
Examples: `--years 2022 2022EE 2023BPix` or `--years all`
`--model`
Model configuration name (e.g. "test"). Names are keys in /home/users/lumori/bbtautau/src/bbtautau/postprocessing/bdt_config.py configuration dictionaries
`--save-dir`
Name to save the trained model and generated plots in `"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/{model_dir}"`. Defaults to `"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models/{self.modelname}_{('-'.join(self.years) if not self.years == hh_vars.years else 'all')}"`
`--force-reload`
Force reloading of data, even if cache/files exist.
`--samples`
List of sample names to use for training or evaluation. Defaults to [ggf signals, QCD, ttbar, DY]
`--train`
Train a new model (mutually exclusive with --load).
`--load`
Load a previously trained model (default if neither is specified).
`--study-rescaling`
Script to study the impact of different weight and rescaling rules on BDT performance.
`--eval-bdt-preds`
Evaluate BDT predictions on the given data samples and years. Outputs are stored in the data directory as .npy files, and can later be handled through `postprocessing.load_bdt_preds`.

**Example: train a new model ``mymodel''**
```
python bdt.py --train --years all --model mymodel
```
Models are stored in global `CLASSIFIER_PATH` defined on top of file.


** Run predictions **
Note: define global variable `DATA_PATH` in `bdt.py` for default prediction output directory, or provide a full path in input as .


### Templates

These are made using the `postprocessing/postprocessing.py` script with the `--templates` option.
See `postprocessing/bash_scripts/MakeTemplates.sh` for an example.


## Datacard and fits


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

After activating the above CMSSW environment (go inside the CMSSW folder and do `cmsenv`), you can use the CreateDatacard.py script as so:

```bash
python3 postprocessing/CreateDatacard.py --sigs bbtt --templates-dir templates/25Apr25LudoCuts --model-name 25Apr25PassFix
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
