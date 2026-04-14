# Artifact Appendix

Paper title: **PROVGEN: A Privacy-Preserving Approach for Outcome Validation in Genomic Research**

## Description

This artifact is split into a small reusable implementation at the repository root and a self-contained artifact workflow under `artifact_evaluation/`.

Repository root:

- `generation/`: reusable PROVGEN implementation and CLI for arbitrary input/output paths.
- `requirements.txt`: pinned Python dependencies.
- `README.md`, `LICENSE`.

Artifact workflow under `artifact_evaluation/`:

- `data/cleansed/`: cleaned input datasets (`hair`, `eye`, `lactose`).
- `generated/`: generated datasets consumed by evaluation (`proposed`, `proposed_dp_maf`, `ldp`, `privbayes`, `dpsyn`).
- `results/`: active experiment CSV output directory. It starts empty except for `.gitkeep`.
- `plots/`: active generated-plot output directory. It starts empty except for `.gitkeep`.
- `doc/`: artifact documentation.
- `comparison_methods/`: local LDP, PrivBayes, and DPSyn generation wrappers plus bundled PrivBayes/DPSyn runtimes.
- `run_generation.py`: paper-layout generation dispatcher.
- `run_evaluation.py`: paper evaluation dispatcher.
- `run_plotting.py`: paper plotting dispatcher.
- `run_experiments.py`: lower-level backend used by the workflow entrypoints.
- `gwas_standard/`, `gwas_maf/`, `mia_standard/`, `mia_large/`, `utility_standard/`, `utility_100/`, `time_complexity/`: one folder per paper experiment.

The execution order for a full artifact run is:

1. environment setup
2. validation
3. generation
4. evaluation
5. plotting

Experiment result CSVs are **not** imported from precomputed result tables. The `run_evaluation.py` commands compute result CSVs from generated datasets currently present under `artifact_evaluation/generated/`, and `run_generation.py` populates that directory from the cleansed inputs.

Generated datasets are intentionally not committed, except for `.gitkeep` placeholders. Full PROVGEN generation for the `eye` dataset is memory-heavy, so reviewers without a high-RAM machine should use the smaller-dataset smoke workflow rather than full from-scratch `eye` generation.

### Dataset Provenance and Redistribution

- The bundled evaluation datasets under `artifact_evaluation/data/cleansed/` are cleaned/preprocessed derivatives of publicly shared data from the openSNP project: <https://opensnp.org/>.
- These datasets correspond to phenotype-driven subsets derived from openSNP participant uploads used in our evaluation workflow.
- openSNP makes participant-contributed genotype/phenotype data publicly available for reuse; we therefore include cleaned derivatives needed for artifact evaluation rather than requiring reviewers to reconstruct them from the raw public dump.
- The software license in this repository applies to the code. The packaged dataset files remain derived from the original openSNP public data release and are included here only to support artifact evaluation and reproduction.

### Security/Privacy Issues and Ethical Concerns

- No exploit or malware code is included.
- The artifact executes local Python scripts and bundled comparison-method code only.
- Reviewers should run it in an isolated environment as standard best practice.
- Datasets in `artifact_evaluation/data/cleansed/` are preprocessed derivatives of public openSNP data and are intended only for artifact evaluation and reproduction.

## Basic Requirements

### Hardware Requirements

Recommended minimums:

- Validation, plotting, and smoke tests: 4 GB RAM, <5 GB free disk.
- Standard PROVGEN generation/evaluation: 16-32 GB RAM recommended.
- Full regeneration including large-scale MIA and 100-SNP baselines: 64-128 GB RAM recommended, multi-core CPU strongly recommended.
- Full PROVGEN regeneration for the `eye` dataset is substantially more memory intensive than `hair` and `lactose`. In our local tests, regenerating full `eye` PROVGEN standard/large-scale outputs can require about 200 GB RAM. Reviewers without this memory should skip full from-scratch `eye` PROVGEN generation and run the smaller-dataset smoke workflow instead.

Runtime depends strongly on CPU count, selected datasets, number of copies, and whether data generation is rerun. The evaluation scripts now expose `--workers` for the expensive loops, so runtimes should be estimated from the chosen worker count rather than from a fixed single-machine number.

Practical expectations:

- Smoke checks and plotting dry-runs finish in minutes.
- `utility_standard` is usually short compared with GWAS and MIA.
- `gwas_standard`, `gwas_maf`, `mia_standard`, and `mia_large` are long-running full-artifact experiments and should be run on a multi-core server.
- Full `eye` PROVGEN regeneration is the main high-RAM step and may require about 200 GB RAM.

### Parallelism

The evaluation backend uses Python multiprocessing for the expensive experiment loops:

- `gwas_standard`
- `gwas_maf`
- `mia_standard`
- `mia_large`
- `utility_standard`
- `utility_100`

Control the process count with `--workers`, for example:

```bash
python run_evaluation.py --datasets hair,lactose --copies 1 --experiment gwas_standard --workers 4
python run_evaluation.py --datasets hair,lactose --copies 1 --experiment mia_large --include-large-mia --workers 4
```

Some ML classifiers used in the MIA experiments are configured with single-threaded internal jobs where possible, so the outer experiment-level multiprocessing controls the main parallelism and avoids CPU oversubscription.

Generation is partly parallelizable through `run_experiments.py --mode generate --workers N`, but reviewers should be conservative for memory-heavy runs. In particular, use `--workers 1` for full PROVGEN `eye` generation.

### Software Requirements

- OS tested: Ubuntu-like Linux environments.
- Python: `3.10.x`.
- Docker: **not required** and not provided by this artifact.
- Python dependencies are pinned in `requirements.txt`.
- PrivBayes workflow used here is derived from DataSynthesizer: <https://github.com/DataResponsibly/DataSynthesizer/blob/master/DataSynthesizer/lib/PrivBayes.py>
- DPSyn runtime and static loader config are included under `artifact_evaluation/comparison_methods/DPSyn/` and correspond to: <https://github.com/agl-c/deid2_dpsyn>. Per-run DPSyn schema/datatype/epsilon config files are generated automatically from the selected cleansed CSV during `run_generation.py`.

## Environment

### Accessibility

Public repository:

- <https://github.com/SpidLab/PROVGEN-A-Privacy-Preserving-Approach-for-Outcome-Validation-in-Genomic-Research>

Artifact appendix source:

- <https://github.com/SpidLab/PROVGEN-A-Privacy-Preserving-Approach-for-Outcome-Validation-in-Genomic-Research/blob/main/artifact_evaluation/doc/ARTIFACT-APPENDIX.md>

### Set Up

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd artifact_evaluation
```

### Quick Functional Smoke Check

Run from `artifact_evaluation/`. These commands are intentionally small and fast; they validate paths, imports, dispatchers, and plotting without running the multi-hour evaluations.

```bash
python run_experiments.py --mode validate

for target in proposed ldp proposed_dp_maf; do
  python run_generation.py --datasets lactose --copies 1 --generation-target "$target" --dry-run
done

for target in proposed_100 privbayes dpsyn; do
  python run_generation.py --datasets lactose --copies 1 --only-100-snp --generation-target "$target" --dry-run
done

python run_generation.py --datasets lactose --copies 1 --include-large-mia --generation-target proposed --dry-run
python run_evaluation.py --datasets lactose --copies 1 --include-large-mia --experiment all --workers 2 --dry-run
python run_evaluation.py --datasets lactose --copies 1 --only-100-snp --experiment utility_100 --workers 2 --dry-run
python run_plotting.py --plot-target time --plots-dir /tmp/provgen_plot_smoke --dry-run
```

Expected outputs include:

- `[ok] input validation passed`
- dry-run generation paths under `generated/proposed/`, `generated/ldp/`, `generated/proposed_dp_maf/`, `generated/privbayes/`, and `generated/dpsyn/`
- dry-run evaluation paths for `results/gwas_df_full.csv`, `results/gwas_df_full_maf.csv`, `results/mia_experiments_results_full.csv`, `results/mia_experiments_results_large_scale.csv`, `results/utility_df_full.csv`, `results/utility_100_df_full.csv`, and `results/time.csv`
- dry-run time-plot path; after actual evaluation writes `results/`, `python run_plotting.py --plot-target all` writes all paper PDFs

The plotting smoke command above avoids archived result CSVs. Full reproduction should run the evaluation commands below and then plot from the newly created active `results/` directory.

### Reusable PROVGEN Generation

The reusable PROVGEN implementation can be used independently of the paper workflow. Run this from the repository root:

```bash
python -m generation \
  --input artifact_evaluation/data/cleansed/lactose/data_100_0.csv \
  --reference artifact_evaluation/data/cleansed/lactose/reference_100_0.csv \
  --epsilon 100 \
  --output /tmp/provgen_lactose.npy
```

The reusable CLI accepts `.csv` or `.npy` inputs and infers the output format from the output suffix. For the command above, the output array should have shape `(60, 100)` with genotype values in `{0, 1, 2}`.

## Artifact Evaluation

Run artifact commands from `artifact_evaluation/`.

### Full Generation

```bash
python run_generation.py --include-large-mia
python run_generation.py --only-100-snp --generation-target privbayes
python run_generation.py --only-100-snp --generation-target dpsyn
```

These commands generate the datasets consumed by the evaluation scripts:

- `generated/proposed/`: PROVGEN datasets for standard, large-scale MIA, and 100-SNP settings.
- `generated/proposed_dp_maf/`: PROVGEN-MAF datasets for GWAS MAF experiments.
- `generated/ldp/`: LDP baseline datasets.
- `generated/privbayes/`: PrivBayes 100-SNP baseline datasets.
- `generated/dpsyn/`: DPSyn 100-SNP baseline datasets.

The artifact evaluation commands below use generated datasets from `generated/`; they do not load precomputed experiment-result CSVs. Generated datasets are produced by `run_generation.py` and are not committed to the repository.

Reviewers on machines with less than about 200 GB RAM should not attempt full from-scratch PROVGEN generation for the `eye` dataset. To exercise the full pipeline on lower-memory machines, run generation/evaluation on a smaller slice such as `--datasets hair,lactose --copies 1`.

### Experiment 1: GWAS Outcome Validation

```bash
python run_evaluation.py --experiment gwas_standard
python run_evaluation.py --experiment gwas_maf
python run_plotting.py --plot-target gwas_standard
python run_plotting.py --plot-target gwas_maf
```

Expected outputs:

- `results/gwas_df_full.csv`
- `results/gwas_df_full_maf.csv`
- `plots/gwas_results_chi2_flipping.pdf`
- `plots/gwas_results_chi2_noise.pdf`
- `plots/gwas_results_odds_flipping.pdf`
- `plots/gwas_results_odds_noise.pdf`
- `plots/gwas_results_maf.pdf`
- `plots/gwas_results_maf_large.pdf`

### Experiment 2: MIA Standard Scale

```bash
python run_evaluation.py --experiment mia_standard
python run_plotting.py --plot-target mia_standard
```

Expected outputs:

- `results/mia_experiments_results_full.csv`
- `plots/mia_hair_color.pdf`
- `plots/mia_eye_color.pdf`
- `plots/mia_lactose_intolerance.pdf`

### Experiment 3: MIA Large Scale

```bash
python run_generation.py --include-large-mia
python run_evaluation.py --experiment mia_large --include-large-mia
python run_plotting.py --plot-target mia_large
```

Expected outputs:

- `results/mia_experiments_results_large_scale.csv`
- `plots/mia_hair_color_large_scale.pdf`
- `plots/mia_eye_color_large_scale.pdf`
- `plots/mia_lactose_intolerance_large_scale.pdf`

### Experiment 4: Utility Comparisons

```bash
python run_evaluation.py --experiment utility_standard
python run_generation.py --only-100-snp --generation-target privbayes
python run_generation.py --only-100-snp --generation-target dpsyn
python run_evaluation.py --experiment utility_100
```

Expected outputs:

- terminal summary tables printed by `run_evaluation.py`
- `results/utility_df_full.csv`
- `results/utility_100_df_full.csv`

There is no dedicated utility PDF in this artifact. The utility claim is checked from the printed summary tables and CSVs.

### Experiment 5: Figure Reproduction

```bash
python run_plotting.py
```

Expected additional output:

- `plots/time.pdf`
- absolute paths of all written figure files printed to the terminal

The time-complexity plot uses the included timing table materialized as `results/time.csv`; it is not a live wall-clock benchmark rerun.

## Limitations

- Full generation can be resource-intensive and is not suitable for low-resource laptops.
- Experiment result CSVs are recomputed from generated datasets rather than loaded from precomputed result tables.
- Generated datasets are intentionally excluded from the repository and should be regenerated with `run_generation.py`.
- The artifact is primarily tested on Ubuntu-like Linux; macOS and Windows have not been exhaustively validated.

## Reusability Notes

- Use `../generation/` or `python -m generation` for standalone PROVGEN generation on arbitrary input/output paths.
- Use `run_generation.py`, `run_evaluation.py`, and `run_plotting.py` for paper reproduction.
- Add or replace a paper comparison mechanism by extending `shared_path`, generation helpers, and evaluation loops in `run_experiments.py`.
- Use `--datasets`, `--copies`, and `--only-100-snp` to run smaller debug slices.

## Uninstall

Removing the repository directory and the virtual environment is sufficient to uninstall the artifact:

```bash
rm -rf .venv
```

To preserve source files but discard active outputs, remove the generated output folders inside `artifact_evaluation/`:

```bash
rm -rf generated/{proposed,ldp,proposed_dp_maf,privbayes,dpsyn}
rm -rf results/* plots/*
touch results/.gitkeep plots/.gitkeep
```
