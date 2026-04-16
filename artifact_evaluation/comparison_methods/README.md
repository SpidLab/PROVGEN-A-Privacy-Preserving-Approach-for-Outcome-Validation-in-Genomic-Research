# Comparison Methods

This directory contains the local baseline runtimes used by the artifact to generate comparison-method outputs.

- `PrivBayes/`: local wrapper/runtime for the PrivBayes baseline used in the 100-SNP utility comparison.
- `DPSyn/`: local DPSyn runtime plus the static `config/data.yaml` and import-path files needed by the runtime. Per-run DPSyn schema, datatype, and epsilon config files are generated automatically from the selected cleansed CSV by `run_generation.py`.

Upstream references:

- PrivBayes implementation lineage in DataSynthesizer:
  <https://github.com/DataResponsibly/DataSynthesizer/blob/master/DataSynthesizer/lib/PrivBayes.py>
- DPSyn repository:
  <https://github.com/agl-c/deid2_dpsyn>

These folders are invoked by `run_experiments.py` through:

- `python run_generation.py --only-100-snp --generation-target privbayes`
- `python run_generation.py --only-100-snp --generation-target dpsyn`

The artifact wrapper invokes PrivBayes and DPSyn generation jobs serially. These comparison-method runtimes may use their own internal processing, so the artifact does not add another multiprocessing layer around them.

Generated datasets are written under:

- `generated/privbayes/`
- `generated/dpsyn/`

Evaluation result CSVs are computed from these generated datasets. They are not filled from precomputed experiment-result tables.
