"""Artifact generation entrypoint for the paper's proposed method.

This module is intentionally under ``artifact_evaluation/`` because it reproduces the
paper's fixed dataset layout, epsilon schedules, and output naming convention.
For reusable PROVGEN generation on arbitrary input/output paths, use
``python -m generation`` instead.

Paper mapping:
- standard scale: eps in {1, 2, 3, 4, 5}
- large scale: eps in {10^-2, 10^-1, 1, 10, 10^2}
- MAF-restored branch: protected/public MAF experiment
- 100-SNP branch: utility comparison setting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.comparison_methods.generators.common import (  # noqa: E402
    DEFAULT_COPIES,
    DEFAULT_DATASETS,
    build_context,
    maybe_generate_100_snp_methods,
    maybe_generate_proposed,
    maybe_generate_proposed_dp_maf,
    parse_datasets,
    validate_generation,
)


def run(datasets: str = DEFAULT_DATASETS, copies: int = DEFAULT_COPIES, target: str = "all", dry_run: bool = False) -> int:
    ctx = build_context(dry_run=dry_run)
    selected = parse_datasets(datasets)
    include_large_mia = target in {"all", "large"}
    only_100_snp = target == "100-snp"
    code = validate_generation(ctx, selected, include_large_mia=include_large_mia, only_100_snp=only_100_snp, copies=copies)
    if code != 0:
        return code

    if target in {"all", "standard", "large"}:
        maybe_generate_proposed(ctx, selected, include_large_mia=include_large_mia, validate_only=False, copies=copies)
    if target in {"all", "maf"}:
        maybe_generate_proposed_dp_maf(ctx, selected, validate_only=False, copies=copies)
    if target in {"all", "100-snp"}:
        maybe_generate_100_snp_methods(ctx, selected, validate_only=False, copies=copies)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-layout datasets for PROVGEN.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--target",
        choices=["all", "standard", "large", "maf", "100-snp"],
        default="all",
        help="standard=eps 1..5, large=eps 10^-2..10^2, maf=MAF-restored branch, 100-snp=utility comparison branch",
    )
    args = parser.parse_args()
    return run(datasets=args.datasets, copies=args.copies, target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
