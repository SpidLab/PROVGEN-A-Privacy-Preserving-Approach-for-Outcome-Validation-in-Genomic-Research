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
- `gwas_standard/`, `gwas_maf/`, `mia_standard/`, `mia_large/`, `utility_standard/`, `utility_100/`: one folder per paper experiment.

## Clean Start

To rerun from a clean workspace while preserving the directory structure:

```bash
find generated results plots -type f ! -name '.gitkeep' -delete
```

## Quick Validation

Run from this `artifact_evaluation/` directory:

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
python run_plotting.py --plot-target gwas_standard --plots-dir /tmp/provgen_plot_check --dry-run
```

The `--experiment all` dispatcher now covers `gwas_standard`, `gwas_maf`, `mia_standard`, `mia_large`, `utility_standard`, and `utility_100`.

## Reduced End-to-End Verification

This reduced workflow exercises every experiment family without the full-memory `eye` generation path:

```bash
python run_generation.py --datasets hair,lactose --copies 1 --include-large-mia
python run_evaluation.py --datasets hair,lactose --copies 1 --workers 2 --include-large-mia --experiment all
python run_plotting.py --plot-target all
```

Notes:

- On a reduced `hair,lactose` slice, the current MIA plotting helper still writes `mia_eye_color*.pdf` placeholders because the paper plot layout always includes the three paper datasets.

## Experiment Inputs

- `gwas_standard`: requires standard `generated/proposed/` and `generated/ldp/` outputs.
- `gwas_maf`: requires `generated/proposed_dp_maf/`; plotting also requires `results/gwas_df_full.csv` from `gwas_standard`.
- `mia_standard`: requires standard `generated/proposed/` and `generated/ldp/` outputs.
- `mia_large`: requires `--include-large-mia` generation for `generated/proposed/` and `generated/ldp/`.
- `utility_standard`: requires standard `generated/proposed/` and `generated/ldp/` outputs.
- `utility_100`: requires 100-SNP `generated/proposed/` outputs plus `generated/privbayes/` and `generated/dpsyn/`.

Utility reviewer-facing outputs are tabular summaries written during evaluation:

- `results/utility_summary.csv`
- `results/utility_summary.md`
- `results/utility_100_summary.csv`
- `results/utility_100_summary.md`
## Simple All-In-One Commands

```bash
python run_generation.py --include-large-mia
python run_evaluation.py --include-large-mia --experiment all
python run_plotting.py --plot-target all
```

The generation command above runs all generation stages:

- standard PROVGEN
- standard LDP
- protected-MAF PROVGEN
- 100-SNP PROVGEN
- 100-SNP PrivBayes
- 100-SNP DPSyn

The evaluation command above runs:

- `gwas_standard`
- `gwas_maf`
- `mia_standard`
- `mia_large`
- `utility_standard`
- `utility_100`

## Step-by-Step Commands

### Generation

Generate all stages at once:

```bash
python run_generation.py --include-large-mia
```

Generate each stage separately:

```bash
python run_generation.py --generation-target proposed --include-large-mia
python run_generation.py --generation-target ldp --include-large-mia
python run_generation.py --generation-target proposed_dp_maf
python run_generation.py --only-100-snp --generation-target proposed_100
python run_generation.py --only-100-snp --generation-target privbayes
python run_generation.py --only-100-snp --generation-target dpsyn
```

### Evaluation

Evaluate all experiments at once:

```bash
python run_evaluation.py --include-large-mia --experiment all
```

Evaluate each experiment separately:

```bash
python run_evaluation.py --experiment gwas_standard
python run_evaluation.py --experiment gwas_maf
python run_evaluation.py --experiment mia_standard
python run_evaluation.py --experiment mia_large --include-large-mia
python run_evaluation.py --experiment utility_standard
python run_evaluation.py --experiment utility_100
```

### Plotting

Render all plot groups at once:

```bash
python run_plotting.py --plot-target all
```

Render each plot group separately:

```bash
python run_plotting.py --plot-target gwas_standard
python run_plotting.py --plot-target gwas_maf
python run_plotting.py --plot-target mia_standard
python run_plotting.py --plot-target mia_large
```

Utility has no plotting step. Use the summary files already written under `results/` instead.

Generation is intentionally single-process at the artifact dispatcher level. The `eye` PROVGEN generation path is memory-heavy and can require about 200 GB RAM for one run, so PROVGEN generation should not be manually parallelized unless the machine has enough memory for every concurrent run. PrivBayes and DPSyn generation jobs are also invoked one at a time by this artifact because their bundled runtimes may manage their own internal processing. Machines below the full-eye memory range should use a reduced-slice run such as `--datasets hair,lactose --copies 1` instead of full from-scratch `eye` generation.

If you only need Experiment 1, `--include-large-mia` is not required; that flag is only for the large-scale MIA branch.
