# Kubernetes Training Infrastructure

This directory contains meta-scripts and Kubernetes configurations for launching BDT (Boosted Decision Tree) training jobs in a Kubernetes cluster.

## Overview

The system provides a workflow to:
1. Generate Kubernetes Job manifests from templates
2. Submit training or model comparison jobs to the cluster
3. Access results via a web interface
4. Clean up completed jobs

## Directory Structure

```
kubernetes/
├── jobs/                    # Job generation scripts and templates
│   ├── template.yaml        # Template for BDT training jobs
│   ├── template_compare.yaml # Template for model comparison jobs
│   └── make_from_template.py # Script to generate job YAMLs from templates
├── setup/                   # Infrastructure setup files
│   ├── pvc/                 # Persistent volume configuration
│   │   ├── vol.yaml         # PersistentVolumeClaim (2000Gi shared storage)
│   │   └── volpod.yaml      # Pod for manual PVC access
│   └── http/                # Web server for browsing results
│       ├── server.yaml      # Nginx deployment with fancyindex
│       ├── server-config.yaml # Nginx configuration
│       ├── ingress.yaml    # Ingress for external access
│       └── expose.yaml     # Service to expose nginx
├── bdt_trainings/           # Generated job YAML files (organized by tag)
└── cleanup.ipynb            # Jupyter notebook for job cleanup
```

## Components

### Job Generation (`jobs/`)

#### `make_from_template.py`
Main script for generating Kubernetes Job manifests. Supports two modes:

**Training Mode** (default):
- Generates jobs that train BDT models
- Uses `template.yaml` as the base template
- Job name defaults to the model name
- Automatically includes `--bin-features` flag

**Comparison Mode** (`--compare-models`):
- Generates jobs that compare multiple trained models
- Uses `template_compare.yaml` as the base template
- Job name defaults to a tag derived from the compared inputs

**Light Comparison Mode** (`--compare-light`):
- Uses `template_compare_light.yaml`
- Compares existing `metrics_summary.csv` outputs without reloading training data

**Rescaling Study Mode** (`--study-rescaling`):
- Uses `template_rescaling.yaml`
- Runs the balance/rescaling study workflow for a single model

**Key Arguments:**
- `--modelname`: Model name for training or rescaling jobs
- `--config-file`: Optional local config file to embed in a training job and pass to `bdt.py --config-file`
- `--tag`: Tag for organizing jobs/outputs
- `--datapath`: Path to data within PVC (default: `26Mar5All_v12_private_signal`)
- `--years`: Years to process (default: `all`)
- `--compare-models`: Enable comparison mode
- `--compare-light`: Enable metrics-only comparison mode
- `--study-rescaling`: Enable rescaling study mode
- `--inputs`: Model JSON files and/or directories to compare
- `--compare-tag`: Optional label for comparison output grouping
- `--train-args`: Additional arguments for training script
- `--tt-preselection`: Route output under `tt_presel/` and append the flag to the runtime command
- `--from-json`: Load arguments from JSON file
- `--submit`: Automatically submit job after generation
- `--overwrite`: Overwrite existing job files without prompting

**Example Usage:**
```bash
# Generate a training job
python jobs/make_from_template.py \
    --modelname my_model \
    --config-file ../postprocessing/bdt_configs/standard/config_my_model.py \
    --tag my_scan \
    --submit

# Generate a comparison job
python jobs/make_from_template.py \
    --compare-models \
    --inputs training/no_presel/model1 training/no_presel/model2 \
    --compare-tag model1_vs_model2 \
    --submit
```

#### Templates

**`template.yaml`**: Training job template
- Clones repository from GitHub (`dev-bdt` branch)
- Installs dependencies
- Runs `bdt.py --train --bin-features` with specified parameters
- If `--config-file` is used when generating the YAML, the config content is embedded into the job and passed back to `bdt.py --config-file` inside the pod
- Requires GPU (1x nvidia.com/gpu)
- Resources: 32Gi memory for `tt_presel`, 80Gi otherwise
- Outputs to `/bbtautauvol/bdt/training/{presel}/{tag}/{modelname}/`

**`template_compare.yaml`**: Comparison job template
- Similar setup but runs `bdt.py --compare-models`
- No GPU required
- Resources: 25Gi memory for `tt_presel`, 80Gi otherwise
- Outputs to `/bbtautauvol/bdt/comparisons/{presel}/{tag}/compare_{compare_tag}/`

Both templates:
- Mount persistent volume at `/bbtautauvol`
- Exclude node `service-02.nrp.mghpcc.org` from scheduling
- Use Docker image: `gitlab-registry.nrp-nautilus.io/ludomori99/bbtautau-docker:latest`

### Infrastructure Setup (`setup/`)

#### Persistent Volume (`pvc/`)

**`vol.yaml`**: Creates a 2000Gi PersistentVolumeClaim named `bbtautauvol`
- Storage class: `rook-cephfs`
- Access mode: `ReadWriteMany` (multiple pods can mount simultaneously)

**`volpod.yaml`**: Creates a pod for manual access to the PVC
- Useful for debugging or manual file operations
- Uses minimal resources

**Setup:**
```bash
kubectl create -f setup/pvc/vol.yaml -n cms-ml
kubectl create -f setup/pvc/volpod.yaml -n cms-ml
```

#### Web Server (`http/`)

Provides a web interface to browse training results stored in the PVC.

**Components:**
- **`server.yaml`**: Nginx deployment with fancyindex module for directory browsing
- **`server-config.yaml`**: ConfigMap with nginx configuration
- **`expose.yaml`**: Service exposing nginx on port 80
- **`ingress.yaml`**: Ingress for external access at `bbtautau.nrp-nautilus.io`

**Setup:**
```bash
kubectl create -f setup/http/server-config.yaml -n cms-ml
kubectl create -f setup/http/server.yaml -n cms-ml
kubectl create -f setup/http/expose.yaml -n cms-ml
kubectl create -f setup/http/ingress.yaml -n cms-ml
```

### Hyperparameter Exploration

Configs are written directly into `postprocessing/bdt_configs/<group>/`, and `generate_hp_jobs.py` now forwards each generated config file into the corresponding Kubernetes job. That means hyperparameter scans no longer require the generated configs to be committed before the pod can train with them.

#### `generate_hp_configs.py`
Generates BDT configuration files by systematically exploring the hyperparameter space using Latin Hypercube Sampling (LHS) for efficient coverage.

**Key Arguments:**
- `--base-config`: Base configuration file (absolute path, relative path, or bare filename — searched recursively in `bdt_configs/`)
- `--group` *(required)*: Subfolder name inside `postprocessing/bdt_configs/` (e.g. `key_pars_k2v0`). Created if it doesn't exist.
- `--n-configs`: Number of configurations to generate (default: 20)
- `--prefix`: Prefix for generated model names (default: `hp`)
- `--strategy`: Search strategy - `lhs` (default), `grid`, or `random`
- `--explore-all`: Explore all hyperparameters (not just key ones)
- `--seed`: Random seed for reproducibility

**Example Usage:**
```bash
# Generate 20 configs in a new group, using a base config found by filename
python generate_hp_configs.py \
    --base-config config_24Feb26_weak_deeper.py \
    --group my_scan \
    --prefix my_scan \
    --n-configs 20

# Explore all hyperparameters
python generate_hp_configs.py \
    --base-config config_24Feb26_weak_deeper.py \
    --group my_full_scan \
    --prefix full \
    --n-configs 30 \
    --explore-all
```

**Hyperparameters Explored:**
- **Key parameters** (default): `max_depth`, `eta`, `subsample`, `colsample_bytree`, `num_parallel_tree`
- **All parameters** (with `--explore-all`): Above plus `alpha`, `gamma`, `lambda`

#### `generate_hp_jobs.py`
Generates Kubernetes jobs for all models created by `generate_hp_configs.py`.

**Key Arguments:**
- `--summary` *(required)*: Path to `hp_exploration_summary.json`, **or** just the group name (e.g. `my_scan`) to look it up under `bdt_configs/<group>/`
- `--tag`: Tag for organizing jobs (default: `hp_explore`)
- `--datapath`: Data path within PVC
- `--submit`: Submit jobs to Kubernetes after generation
- `--overwrite`: Overwrite existing job files
- `--dry-run`: Print commands without executing

**Example Usage:**
```bash
# Generate jobs by group name
python generate_hp_jobs.py \
    --summary my_scan \
    --tag my_scan \
    --submit

# Dry-run to preview
python generate_hp_jobs.py \
    --summary my_scan \
    --tag my_scan \
    --dry-run
```

Each generated training job passes both:
- `--modelname <generated_model>`
- `--config-file <local path to generated config>`

so the pod can reconstruct the config at startup and call `bdt.py --config-file ...` without relying on the cloned repository to already contain that config file.

**End-to-end workflow:**
```bash
# 1. Generate configs (written directly to postprocessing/bdt_configs/my_scan/)
python generate_hp_configs.py \
    --base-config config_24Feb26_weak_deeper.py \
    --group my_scan \
    --prefix my_scan \
    --n-configs 20

# 2. Generate and submit Kubernetes jobs (no copying needed)
python generate_hp_jobs.py \
    --summary my_scan \
    --tag my_scan \
    --submit

# 3. Configs are immediately usable in analysis code:
#    from bbtautau.postprocessing.bdt_config import BDT_CONFIG
#    config = BDT_CONFIG["my_scan_001"]
```

### Job Cleanup (`cleanup.ipynb`)

Jupyter notebook for cleaning up completed Kubernetes jobs:
- Lists all jobs in `cms-ml` namespace
- Identifies completed (succeeded/failed) `bbtautau-*` jobs
- Provides option to delete them

Run cells sequentially to review and clean up jobs.

## Workflow

1. **Initial Setup** (one-time):
   ```bash
   # Create persistent volume
   kubectl create -f setup/pvc/vol.yaml -n cms-ml

   # Setup web server (optional)
   kubectl create -f setup/http/server-config.yaml -n cms-ml
   kubectl create -f setup/http/server.yaml -n cms-ml
   kubectl create -f setup/http/expose.yaml -n cms-ml
   kubectl create -f setup/http/ingress.yaml -n cms-ml
   ```

2. **Generate and Submit Training Job**:
   ```bash
   python jobs/make_from_template.py \
       --modelname my_model \
       --config-file ../postprocessing/bdt_configs/standard/config_my_model.py \
       --tag no_presel \
       --submit
   ```

   Note: for training jobs, either the cloned repo must contain the config under `postprocessing/bdt_configs/`, or you should pass `--config-file` so the generated YAML carries the config explicitly.

3. **Monitor Jobs**:
   ```bash
   kubectl get jobs -n cms-ml | grep bbtautau
   kubectl logs -n cms-ml job/bbtautau-bdt-{job-name}
   ```

4. **Access Results**:
   - Via web interface: `https://bbtautau.nrp-nautilus.io`
   - Or via PVC pod: `kubectl exec -it bbtautauvolpod -n cms-ml -- bash`

5. **Cleanup** (optional):
   - Use `cleanup.ipynb` notebook to remove completed jobs

## Generated Job Files

Job YAML files are stored under `bdt_trainings/{job_type}/{presel}/{tag}/`.

Examples:
- Training: `bdt_trainings/training/no_presel/<tag>/<modelname>.yml`
- Comparison: `bdt_trainings/comparisons/no_presel/<tag>/cmp_<inputs>.yml`
- Rescaling: `bdt_trainings/rescaling/no_presel/<tag>/rescaling_<modelname>.yml`

These files can be:
- Manually edited if needed
- Submitted directly: `kubectl create -f bdt_trainings/{tag}/{job_name}.yml -n cms-ml`
- Reused for similar jobs

## Notes

- All jobs run in the `cms-ml` namespace
- Jobs use the `dev-bdt` branch of the repository
- Outputs are stored in `/bbtautauvol/bdt/` within the PVC
- GPU resources are required for training jobs but not for comparison jobs
- The system automatically excludes certain nodes from scheduling
- `--bin-features` flag is automatically included in training jobs (applies feature binning to gloParT scores)
- Signal configuration is now determined by the model's config file (`signals` field), not via command-line argument
