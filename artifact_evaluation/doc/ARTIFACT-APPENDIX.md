# Artifact Appendix

Paper title: **PROVGEN: A Privacy-Preserving Approach for Outcome Validation in Genomic Research**

## Description

This artifact contains a self-contained evaluation workflow under `artifact_evaluation/` for reproducing the paper experiments.

Main directories and entrypoints:

- `data/cleansed/`: cleaned input datasets (`hair`, `eye`, `lactose`)
- `generated/`: generated datasets used by evaluation
- `results/`: experiment CSV outputs
- `plots/`: generated figure PDFs
- `comparison_methods/`: bundled PrivBayes and DPSyn runtimes and wrappers
- `run_generation.py`: full data-generation entrypoint
- `run_evaluation.py`: full experiment entrypoint
- `run_plotting.py`: full plotting entrypoint

## Dataset Provenance and Redistribution

- The bundled evaluation datasets under `artifact_evaluation/data/cleansed/` are cleaned and preprocessed derivatives of publicly shared data from the openSNP project: <https://opensnp.org/>.
- These datasets correspond to phenotype-driven subsets derived from openSNP participant uploads used in our evaluation workflow.
- The software license in this repository applies to the code. The packaged dataset files remain derived from the original public openSNP release and are included only to support artifact evaluation and reproduction.

## Security and Ethical Concerns

- No exploit, malware, or offensive-security code is included.
- The artifact executes local Python scripts and bundled comparison-method code only.
- Reviewers should run the artifact in an isolated environment as standard best practice.
- The packaged datasets are preprocessed derivatives of public openSNP data and are included only for evaluation and reproduction.

## Requirements

### Hardware

- Validation, plotting, and small checks: 4 GB RAM and a few GB of free disk are sufficient.
- Standard generation and evaluation: 16-32 GB RAM recommended.
- Full regeneration including large-scale MIA and 100-SNP baselines: 64-128 GB RAM recommended.
- Full PROVGEN regeneration for the `eye` dataset is substantially more memory intensive than `hair` and `lactose`, and in our local tests one run can require about 200 GB RAM.

### Software

- OS tested: Ubuntu-like Linux environments
- Python: `3.10.x`
- Docker: not required
- Python dependencies: pinned in `requirements.txt`

## Set Up

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd artifact_evaluation
```

On minimal Ubuntu-like installations, `python -m venv .venv` may fail if the system Python was installed without `ensurepip` / `venv` support. In that case, install the OS package that provides Python 3.10 venv support, or create the environment with `virtualenv -p /usr/bin/python3.10 .venv`.

## End-to-End Command Bundle

To run the full artifact workflow from `artifact_evaluation/`:

```bash
python run_generation.py --include-large-mia
python run_evaluation.py --include-large-mia --experiment all
python run_plotting.py --plot-target all
```

After these commands finish:

- generated datasets will be under `artifact_evaluation/generated/`
- experiment CSV outputs will be under `artifact_evaluation/results/`
- figure PDFs will be under `artifact_evaluation/plots/`

## Full Generation

Run from `artifact_evaluation/`:

```bash
python run_generation.py --include-large-mia
```

This command generates all paper-required datasets under `artifact_evaluation/generated/`, including:

- standard PROVGEN outputs
- standard LDP outputs
- protected-MAF PROVGEN outputs
- 100-SNP PROVGEN outputs
- 100-SNP PrivBayes outputs
- 100-SNP DPSyn outputs

## Per-Stage Generation

To run each generation stage separately:

```bash
python run_generation.py --generation-target proposed --include-large-mia
python run_generation.py --generation-target ldp --include-large-mia
python run_generation.py --generation-target proposed_dp_maf
python run_generation.py --only-100-snp --generation-target proposed_100
python run_generation.py --only-100-snp --generation-target privbayes
python run_generation.py --only-100-snp --generation-target dpsyn
```

## Full Experiments

After generation completes, run:

```bash
python run_evaluation.py --include-large-mia --experiment all
```

This command runs the paper experiment suite and writes CSV outputs under `artifact_evaluation/results/`, including:

- `gwas_df_full.csv`
- `gwas_df_full_maf.csv`
- `mia_experiments_results_full.csv`
- `mia_experiments_results_large_scale.csv`
- `utility_df_full.csv`
- `utility_100_df_full.csv`
- `utility_summary.csv`
- `utility_summary.md`
- `utility_100_summary.csv`
- `utility_100_summary.md`

## Full Plotting

After evaluation completes, run:

```bash
python run_plotting.py --plot-target all
```

This command generates the figure PDFs under `artifact_evaluation/plots/`.

Utility does not have a plotting stage in this artifact. Its reviewer-facing outputs are the summary files already written under `artifact_evaluation/results/`.

The main paper figures will be written there, including:

- `gwas_results_chi2_flipping.pdf`
- `gwas_results_chi2_noise.pdf`
- `gwas_results_odds_flipping.pdf`
- `gwas_results_odds_noise.pdf`
- `gwas_results_maf.pdf`
- `mia_hair_color.pdf`
- `mia_eye_color.pdf`
- `mia_lactose_intolerance.pdf`
- `mia_hair_color_large_scale.pdf`
- `mia_eye_color_large_scale.pdf`
- `mia_lactose_intolerance_large_scale.pdf`

## Per-Experiment Run -> Plot

The following commands assume generation has already completed.

### GWAS Standard

Run:

```bash
python run_evaluation.py --experiment gwas_standard
python run_plotting.py --plot-target gwas_standard
```

Results:

- `artifact_evaluation/results/gwas_df_full.csv`

Plots:

- `artifact_evaluation/plots/gwas_results_chi2_flipping.pdf`
- `artifact_evaluation/plots/gwas_results_chi2_noise.pdf`
- `artifact_evaluation/plots/gwas_results_odds_flipping.pdf`
- `artifact_evaluation/plots/gwas_results_odds_noise.pdf`

### GWAS MAF

Run:

```bash
python run_evaluation.py --experiment gwas_maf
python run_plotting.py --plot-target gwas_maf
```

Results:

- `artifact_evaluation/results/gwas_df_full_maf.csv`

Plots:

- `artifact_evaluation/plots/gwas_results_maf.pdf`

### MIA Standard

Run:

```bash
python run_evaluation.py --experiment mia_standard
python run_plotting.py --plot-target mia_standard
```

Results:

- `artifact_evaluation/results/mia_experiments_results_full.csv`

Plots:

- `artifact_evaluation/plots/mia_hair_color.pdf`
- `artifact_evaluation/plots/mia_eye_color.pdf`
- `artifact_evaluation/plots/mia_lactose_intolerance.pdf`

### MIA Large-Scale

Run:

```bash
python run_evaluation.py --experiment mia_large --include-large-mia
python run_plotting.py --plot-target mia_large
```

Results:

- `artifact_evaluation/results/mia_experiments_results_large_scale.csv`

Plots:

- `artifact_evaluation/plots/mia_hair_color_large_scale.pdf`
- `artifact_evaluation/plots/mia_eye_color_large_scale.pdf`
- `artifact_evaluation/plots/mia_lactose_intolerance_large_scale.pdf`

### Utility Standard

Run:

```bash
python run_evaluation.py --experiment utility_standard
```

Results:

- `artifact_evaluation/results/utility_df_full.csv`
- `artifact_evaluation/results/utility_summary.csv`
- `artifact_evaluation/results/utility_summary.md`

Plots:

- none; use the summary files above

### Utility 100-SNP

Run:

```bash
python run_evaluation.py --experiment utility_100
```

Results:

- `artifact_evaluation/results/utility_100_df_full.csv`
- `artifact_evaluation/results/utility_100_summary.csv`
- `artifact_evaluation/results/utility_100_summary.md`

Plots:

- none; use the summary files above
