# Artifact Experiment Workflow

This directory contains all PETS/PoPETs artifact-evaluation material. The reusable PROVGEN implementation is intentionally outside this folder in `../generation/`.

## Layout

- `data/cleansed/`: cleaned input datasets (`hair`, `eye`, `lactose`).
- `generated/`: generated datasets consumed by evaluation (`proposed`, `proposed_dp_maf`, `ldp`, `privbayes`, `dpsyn`).
- `results/`: active experiment CSV output directory. This starts empty except for `.gitkeep`.
- `plots/`: active generated-plot output directory. This starts empty except for `.gitkeep`.
- `doc/`: artifact appendix and review-facing documentation.
- `comparison_methods/`: LDP, PrivBayes, and DPSyn generation wrappers plus bundled PrivBayes/DPSyn runtimes. DPSyn per-run schema/datatype/epsilon configs are generated on demand.
- `run_generation.py`: regenerate paper-layout datasets under `generated/`.
- `run_evaluation.py`: compute paper evaluation CSVs under `results/`.
- `run_plotting.py`: regenerate paper figure PDFs under `plots/`.
- `run_experiments.py`: lower-level backend used by the three workflow entrypoints.
- `common.py`: shared evaluation and plotting helpers.
- `gwas_standard/`, `gwas_maf/`, `mia_standard/`, `mia_large/`, `utility_standard/`, `utility_100/`, `time_complexity/`: one folder per paper experiment.

## Quick Smoke Check

Run from this `artifact_evaluation/` directory:

```bash
python run_experiments.py --mode validate
python run_generation.py --datasets lactose --copies 1 --generation-target proposed --dry-run
python run_evaluation.py --datasets lactose --copies 1 --include-large-mia --experiment all --workers 2 --dry-run
python run_evaluation.py --datasets lactose --copies 1 --only-100-snp --experiment utility_100 --workers 2 --dry-run
python run_plotting.py --plot-target time --plots-dir /tmp/provgen_plot_smoke --dry-run
```

## Typical Full Flow

```bash
python run_generation.py --include-large-mia
python run_generation.py --only-100-snp --generation-target privbayes
python run_generation.py --only-100-snp --generation-target dpsyn
python run_evaluation.py --include-large-mia
python run_plotting.py
```

Generation is intentionally single-process at the artifact dispatcher level. The `eye` PROVGEN generation path is memory-heavy and can require about 200 GB RAM for one run, so PROVGEN generation should not be manually parallelized unless the machine has enough memory for every concurrent run. PrivBayes and DPSyn generation jobs are also invoked one at a time by this artifact because their bundled runtimes may manage their own internal processing. Machines below the full-eye memory range should use smaller-dataset smoke runs such as `--datasets hair,lactose --copies 1` instead of full from-scratch `eye` generation.
