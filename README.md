# PROVGEN

This repository is split into two top-level parts:

- `generation/`: reusable PROVGEN implementation for generating a privacy-preserving genomic dataset from explicit input, reference, epsilon, and output paths.
- `artifact_evaluation/`: PETS/PoPETs artifact-evaluation workflow, including cleansed input data, comparison methods, generation/evaluation/plotting scripts, and documentation.

Root-level files are intentionally minimal:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `generation/`
- `artifact_evaluation/`

## Setup

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reusable PROVGEN Generation

Use this path when you want to run PROVGEN on your own matrix rather than reproduce the paper experiments:

```bash
python -m generation \
  --input artifact_evaluation/data/cleansed/lactose/data_100_0.csv \
  --reference artifact_evaluation/data/cleansed/lactose/reference_100_0.csv \
  --epsilon 100 \
  --output /tmp/provgen_lactose.npy
```

Inputs may be `.csv` or `.npy`; the output format is inferred from the output suffix.

PROVGEN generation can have a high per-run memory cost on large SNP matrices. For the paper's full `eye` dataset, one generation run can require about 200 GB RAM. The artifact generation dispatcher therefore runs all generation jobs serially. Users should not manually run multiple PROVGEN generation jobs in parallel unless the machine has enough memory for each concurrent run, and the PrivBayes/DPSyn comparison-method wrappers are also invoked one job at a time because those runtimes may manage their own internal processing.

## Artifact Evaluation

All artifact-evaluation material is under `artifact_evaluation/`.

```bash
cd artifact_evaluation
python run_experiments.py --mode validate
python run_generation.py --datasets lactose --copies 1 --generation-target proposed --dry-run
python run_evaluation.py --datasets lactose --copies 1 --include-large-mia --experiment all --workers 2 --dry-run
python run_evaluation.py --datasets lactose --copies 1 --only-100-snp --experiment utility_100 --workers 2 --dry-run
python run_plotting.py --plot-target time --plots-dir /tmp/provgen_plot_smoke --dry-run
```

See `artifact_evaluation/README.md` and `artifact_evaluation/doc/ARTIFACT-APPENDIX.md` for the full artifact instructions.
