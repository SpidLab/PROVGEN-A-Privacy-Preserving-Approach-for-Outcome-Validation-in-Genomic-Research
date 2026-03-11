# Artifact Appendix

Paper title: **PROVGEN: A Privacy-Preserving Approach for Outcome Validation in Genomic Research**

Requested Badge(s):
- [x] **Available**
- [x] **Functional**
- [x] **Reproduced** (via visual/qualitative comparison of generated outputs)

## Description

This artifact supports the PoPETs paper above. It contains:

- input datasets under `data/cleansed/` (hair, eye, lactose),
- generation pipeline (`generation.py`) for synthetic/shared datasets,
- experiment pipeline (`experiments.py`) for evaluation CSVs,
- plotting pipeline (`plotting.py`) for final figures.

The artifact is designed so reviewers can run experiments without regenerating datasets if generated data is already provided.

### Security/Privacy Issues and Ethical Concerns

- No exploit or malware code is included.
- The artifact executes local Python scripts and external baseline code (`DPSyn`, `PrivBayes`) in this repository.
- Reviewers should run in an isolated environment (virtualenv/container/VM) as standard best practice.
- Datasets in `data/cleansed` are preprocessed and expected to be non-identifying for artifact review usage.

## Basic Requirements

### Hardware Requirements

- Functional checks and smoke tests: can run on a standard laptop.
- Full generation and full experiment runs: can require high CPU, substantial RAM, and significant disk.
- Experiments-only mode (when generated data is already available) is much lighter and recommended for artifact review.

### Software Requirements

- OS tested: Linux (Ubuntu-like environment).
- Python: `3.10.x`.
- Python dependencies: see `requirements.txt` (pinned versions can be added if required by reviewers).
- Docker: **not required**. The artifact is runnable with Python virtualenv.
- Docker is optional if reviewers prefer container isolation.
- Optional baseline runtimes:
  - `DPSyn/` folder for DPSyn generation.
  - `PrivBayes/` folder for PrivBayes generation.

### Estimated Time and Storage Consumption

- Setup (`pip install -r requirements.txt`): 10-30 minutes (network dependent).
- Environment validation: minutes.
- Experiments-only (pre-generated data): minutes to hours depending on selected scope.
- Full generation + full experiments: potentially many hours (hardware dependent).
- Storage:
  - input + generated arrays + CSVs can grow to many GB.
  - figures and summary CSVs are small (<1 GB combined).

## Environment

### Accessibility

Use a public repository link (GitHub/GitLab/Zenodo).  
At camera-ready/final artifact stage, provide a stable commit/tag DOI link.

### Set up the environment

Run from the `artifact/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Docker setup is required for standard artifact evaluation.

### Testing the Environment

Quick wiring check:

```bash
python run_experiments.py --mode validate
```

Expected output includes:

- `[ok] input validation passed`

Quick functional smoke check (100-SNP branch, one replicate):

```bash
python experiments.py --datasets lactose --only-100-snp --experiment utility_100 --copies 1
```

## Artifact Evaluation

### Main Results and Claims

The artifact supports these core claims:

1. **GWAS outcome validation quality** under perturbation and error settings.
2. **MIA behavior** across methods under standard epsilon scale.
3. **MIA large-scale trend** under epsilon scale `10^-2 ... 10^2`.
4. **Utility comparison** across methods, including 100-SNP setting with external baselines.
5. **Figure regeneration** from experiment outputs.

### Experiments

All commands below are experiments-only (assume generated datasets already exist).

#### Experiment 1: GWAS Outcome Validation (supports claim 1)

- Time: medium
- Storage: `results/gwas_df_full.csv`, `results/gwas_df_full_maf.csv`

Commands:

```bash
python experiments.py --experiment gwas_standard
python experiments.py --experiment gwas_maf
```

#### Experiment 2: MIA Standard Scale (supports claim 2)

- Time: medium to high
- Storage: `results/mia_experiments_results_full.csv`

Command:

```bash
python experiments.py --experiment mia_standard
```

#### Experiment 3: MIA Large Scale (supports claim 3)

- Time: medium to high
- Storage: `results/mia_experiments_results_large_scale.csv`

Command:

```bash
python experiments.py --experiment mia_large --include-large-mia
```

#### Experiment 4: Utility Comparisons (supports claim 4)

- Time: medium
- Storage: `results/utility_df_full.csv`, `results/utility_100_df_full.csv`

Commands:

```bash
python experiments.py --experiment utility_standard
python experiments.py --experiment utility_100
```

#### Experiment 5: Figure Reproduction (supports claim 5)

- Time: low to medium
- Storage: PDF files in `figures/`

Command:

```bash
python plotting.py
```

Expected figures include:

- `figures/gwas_results_chi2_flipping.pdf`
- `figures/gwas_results_chi2_noise.pdf`
- `figures/gwas_results_odds_flipping.pdf`
- `figures/gwas_results_odds_noise.pdf`
- `figures/gwas_results_maf.pdf`
- `figures/gwas_results_maf_large.pdf`
- `figures/mia_hair_color.pdf`
- `figures/mia_eye_color.pdf`
- `figures/mia_lactose_intolerance.pdf`
- `figures/mia_hair_color_large_scale.pdf`
- `figures/mia_eye_color_large_scale.pdf`
- `figures/mia_lactose_intolerance_large_scale.pdf`
- `figures/time.pdf`

Optional one-shot command for all experiments:

```bash
python experiments.py --include-large-mia
```

## Limitations

- Full generation (`generation.py`) can be resource-intensive and may be impractical on low-resource machines.
- For artifact review, we recommend experiments-only mode with pre-generated data.
- Runtime variability may occur due to hardware and library differences.
- Reproduced assessment is based on qualitative/visual agreement of generated CSV trends and figures with paper claims.

## Notes on Reusability

- The workflow is modular:
  - `generation.py` for data production,
  - `experiments.py` for evaluation,
  - `plotting.py` for visualization.
- `experiments.py` supports per-experiment execution via `--experiment`:
  - `gwas_standard`, `gwas_maf`, `mia_standard`, `mia_large`, `utility_standard`, `utility_100`.
- `generation.py` supports per-generator execution via `--generation-target`:
  - `proposed`, `ldp`, `proposed_dp_maf`, `proposed_100`, `privbayes`, `dpsyn`.
- `--datasets`, `--copies`, and `--only-100-snp` allow scoped runs for rapid debugging and extension.
- Researchers can add new datasets/methods by extending loaders and evaluation loops in `run_experiments.py`.
