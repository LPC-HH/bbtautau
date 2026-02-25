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
- Job name format: `lm_{tag}_{name}`
- Automatically includes `--bin-features` flag

**Comparison Mode** (`--compare-models`):
- Generates jobs that compare multiple trained models
- Uses `template_compare.yaml` as the base template
- Job name format: `cmp_{tag}_{models}`

**Key Arguments:**
- `--name`: Model name (training mode) - must match a config file in `bdt_configs/`
- `--tag`: Tag for organizing jobs/outputs (default: `no_presel`)
- `--datapath`: Path to data within PVC (default: `25Sep23AddVars_v12_private_signal`)
- `--years`: Years to process (default: `all`)
- `--compare-models`: Enable comparison mode
- `--models`: List of model names to compare (comparison mode)
- `--model-dirs`: List of model directories to compare (comparison mode)
- `--train-args`: Additional arguments for training script
- `--from-json`: Load arguments from JSON file
- `--submit`: Automatically submit job after generation
- `--overwrite`: Overwrite existing job files without prompting

**Example Usage:**
```bash
# Generate a training job
python jobs/make_from_template.py \
    --name my_model \
    --tag no_presel \
    --submit

# Generate a comparison job
python jobs/make_from_template.py \
    --compare-models \
    --models model1 model2 \
    --model-dirs model1_dir model2_dir \
    --tag no_presel \
    --submit
```

#### Templates

**`template.yaml`**: Training job template
- Clones repository from GitHub (`kube` branch)
- Installs dependencies
- Runs `bdt.py --train --bin-features` with specified parameters
- Requires GPU (1x nvidia.com/gpu)
- Resources: 32Gi memory, 8 CPU cores
- Outputs to `/bbtautauvol/bdt/{tag}/{name}/`

**`template_compare.yaml`**: Comparison job template
- Similar setup but runs `bdt.py --compare-models`
- No GPU required
- Resources: 32Gi memory, 6 CPU cores
- Outputs to `/bbtautauvol/bdt/{tag}/compare_{models}/`

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

### Hyperparameter Exploration (to be checked)

#### `generate_hp_configs.py`
Intelligent hyperparameter exploration script that generates BDT configuration files using Latin Hypercube Sampling (LHS) for efficient parameter space coverage.

**Features:**
- Systematic exploration of hyperparameter space
- LHS ensures good coverage with fewer samples than random search
- Generates Python config files compatible with existing BDT training system
- Focuses on most impactful parameters by default (max_depth, eta, subsample, colsample_bytree, num_parallel_tree)

**Key Arguments:**
- `--base-config`: Path to base configuration file (e.g., `config_11Feb26Full.py`)
- `--n-configs`: Number of configurations to generate (default: 20)
- `--output-dir`: Directory to save generated config files (default: `bdt_configs`)
- `--prefix`: Prefix for generated model names (default: `hp`)
- `--strategy`: Search strategy - `lhs` (default), `grid`, or `random`
- `--explore-all`: Explore all hyperparameters (not just key ones)
- `--seed`: Random seed for reproducibility

**Example Usage:**
```bash
# Generate 20 configurations using LHS
python generate_hp_configs.py \
    --base-config config_11Feb26Full.py \
    --n-configs 20 \
    --prefix hp_explore \
    --output-dir bdt_configs

# Generate configs exploring all hyperparameters
python generate_hp_configs.py \
    --base-config config_11Feb26Full.py \
    --n-configs 30 \
    --explore-all \
    --prefix hp_full
```

**Hyperparameters Explored:**
- **Key parameters** (default): `max_depth`, `eta`, `subsample`, `colsample_bytree`, `num_parallel_tree`
- **All parameters** (with `--explore-all`): Above plus `alpha`, `gamma`, `lambda`

#### `generate_hp_jobs.py`
Generates Kubernetes jobs for all configurations created by `generate_hp_configs.py`.

**Key Arguments:**
- `--summary-file`: Path to hyperparameter exploration summary JSON (default: `bdt_configs/hp_exploration_summary.json`)
- `--tag`: Tag for organizing jobs (default: `hp_explore`)
- `--datapath`: Data path within PVC
- `--submit`: Submit jobs to Kubernetes after generation
- `--overwrite`: Overwrite existing job files

**Example Usage:**
```bash
# Generate and submit jobs for all HP configs
python generate_hp_jobs.py \
    --summary-file bdt_configs/hp_exploration_summary.json \
    --tag hp_explore \
    --submit
```

**Workflow:**
1. Generate config files: `python generate_hp_configs.py --base-config config_11Feb26Full.py --n-configs 20`
2. Copy configs to `bdt_configs/` directory (or use `generate_hp_jobs.py` which does this automatically)
3. Generate Kubernetes jobs: `python generate_hp_jobs.py --summary-file bdt_configs/hp_exploration_summary.json --submit`

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
       --name my_model \
       --tag no_presel \
       --submit
   ```

   Note: The model name must correspond to a config file in `bdt_configs/` (e.g., `config_my_model.py`).

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

Job YAML files are stored in `bdt_trainings/{tag}/` directory with naming convention:
- Training: `lm_{tag}_{name}.yml`
- Comparison: `cmp_{tag}_{models}.yml`

These files can be:
- Manually edited if needed
- Submitted directly: `kubectl create -f bdt_trainings/{tag}/{job_name}.yml -n cms-ml`
- Reused for similar jobs

## Notes

- All jobs run in the `cms-ml` namespace
- Jobs use the `kube` branch of the repository
- Outputs are stored in `/bbtautauvol/bdt/` within the PVC
- GPU resources are required for training jobs but not for comparison jobs
- The system automatically excludes certain nodes from scheduling
- `--bin-features` flag is automatically included in training jobs (applies feature binning to gloParT scores)
- Signal configuration is now determined by the model's config file (`signals` field), not via command-line argument
