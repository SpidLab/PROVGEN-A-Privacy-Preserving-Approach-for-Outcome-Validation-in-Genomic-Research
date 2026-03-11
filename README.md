# PROVGEN: A Privacy-Preserving Approach for Outcome Validation in Genomic Research

Run all commands from inside `artifact/`.

For PoPETs artifact review metadata and badge-oriented instructions, see:

- `ARTIFACT-APPENDIX.md`

This folder provides three role-specific entrypoints:

- `generation.py`: generate synthetic/shared datasets
- `experiments.py`: run evaluations only (no data generation)
- `plotting.py`: generate figure PDFs from `results/*.csv`

`run_experiments.py` is the shared backend.

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Docker is optional and **not required** for running this artifact.

## 2) Required Input Layout

The minimum input data should exist at:

- `data/cleansed/hair/`
- `data/cleansed/eye/`
- `data/cleansed/lactose/`

Each dataset folder should contain:

- `data.csv`, `reference.csv`
- `data_100_0.csv ... data_100_9.csv`
- `reference_100_0.csv ... reference_100_9.csv`

This artifact package does not include DPSyn/PrivBayes/LDP source code. Baseline comparison results are stored in `precomputed_results/` and are merged automatically with freshly computed PROVGEN results.

## 3) Validate Wiring First

Validate all required inputs and config paths before running heavy jobs:

```bash
python run_experiments.py --mode validate
```

Useful scoped validation examples:

```bash
python run_experiments.py --mode validate --datasets lactose
python run_experiments.py --mode validate --datasets lactose --only-100-snp --copies 1
```

## 4) Experiments-Only (No Generation, recommended for reviewers)

Use this when generated PROVGEN data already exists in `data/{proposed,proposed_dp_maf}` (and `data/proposed` for the 100-SNP branch). Baseline methods are loaded from `precomputed_results/`.

### Main paper experiments (all datasets, standard scale)

```bash
python experiments.py
```

### Large-scale MIA experiment (`10^-2 ... 10^2`)

```bash
python experiments.py --include-large-mia
```

### Run one experiment at a time

GWAS (standard):

```bash
python experiments.py --experiment gwas_standard
```

GWAS (MAF):

```bash
python experiments.py --experiment gwas_maf
```

MIA (standard):

```bash
python experiments.py --experiment mia_standard
```

MIA (large scale):

```bash
python experiments.py --experiment mia_large --include-large-mia
```

Utility (standard):

```bash
python experiments.py --experiment utility_standard
```

Utility (100 SNP):

```bash
python experiments.py --experiment utility_100
```

### Dataset-scoped examples

```bash
python experiments.py --datasets lactose
python experiments.py --datasets lactose --include-large-mia
```

### 100-SNP branch only

```bash
python experiments.py --only-100-snp
```

or explicitly:

```bash
python experiments.py --only-100-snp --experiment utility_100
```


## 5) Generation (Optional)

If synthetic/shared datasets are not available yet:

```bash
python generation.py
python generation.py --include-large-mia
python generation.py --only-100-snp
```

Run only one generator target (separated):

```bash
python generation.py --generation-target proposed
python generation.py --generation-target proposed_dp_maf
python generation.py --generation-target proposed_100
```

## 6) Plot Figures

After experiments complete:

```bash
python plotting.py
```

Plot only one experiment group (prints absolute figure paths):

```bash
python plotting.py --plot-target gwas_standard
python plotting.py --plot-target gwas_maf
python plotting.py --plot-target mia_standard
python plotting.py --plot-target mia_large
```

Utility experiments do not generate a dedicated figure. Reviewers should inspect the terminal summary printed by `python experiments.py --experiment utility_standard` / `python experiments.py --experiment utility_100` and the CSVs in `results/`.

## 7) Command Cheat Sheet

### A) Plotting only

```bash
source .venv/bin/activate
python plotting.py
```

```bash
source .venv/bin/activate
python plotting.py --plot-target gwas_standard
```

### B) Experiments -> Plotting (no generation)

```bash
source .venv/bin/activate
python experiments.py --include-large-mia
python plotting.py
```

### C) Generation -> Experiments -> Plotting

```bash
source .venv/bin/activate
python generation.py --include-large-mia
python experiments.py --include-large-mia
python plotting.py
```

Outputs:

- CSV results: `results/`
- Figures: `figures/`

Expected main CSVs:

- `results/gwas_df_full.csv`
- `results/gwas_df_full_maf.csv`
- `results/mia_experiments_results_full.csv`
- `results/mia_experiments_results_large_scale.csv` (if `--include-large-mia` used)
- `results/utility_df_full.csv`
- `results/utility_100_df_full.csv`
