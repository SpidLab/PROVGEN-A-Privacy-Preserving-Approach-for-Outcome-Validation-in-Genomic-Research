"""Shared helpers for comparison-method generation entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_evaluation.run_experiments import (  # noqa: E402
    DATASET,
    Context,
    maybe_generate_100_snp_methods,
    maybe_generate_ldp,
    maybe_generate_proposed,
    maybe_generate_proposed_dp_maf,
    run_dpsyn_generation,
    run_privbayes_generation,
    validate_inputs,
)

DEFAULT_DATASETS = "hair,lactose,eye"
DEFAULT_COPIES = 10


def build_context(dry_run: bool = False) -> Context:
    return Context(root=ROOT, workers=1, dry_run=dry_run)


def parse_datasets(raw: str) -> list[DATASET]:
    return [DATASET[item.strip()] for item in raw.split(",") if item.strip()]


def validate_generation(ctx: Context, datasets: list[DATASET], include_large_mia: bool, only_100_snp: bool, copies: int) -> int:
    return validate_inputs(ctx, datasets, include_large_mia=include_large_mia, only_100_snp=only_100_snp, copies=copies)


__all__ = [
    "Context",
    "DATASET",
    "DEFAULT_COPIES",
    "DEFAULT_DATASETS",
    "ROOT",
    "build_context",
    "maybe_generate_100_snp_methods",
    "maybe_generate_ldp",
    "maybe_generate_proposed",
    "maybe_generate_proposed_dp_maf",
    "parse_datasets",
    "run_dpsyn_generation",
    "run_privbayes_generation",
    "validate_generation",
]
