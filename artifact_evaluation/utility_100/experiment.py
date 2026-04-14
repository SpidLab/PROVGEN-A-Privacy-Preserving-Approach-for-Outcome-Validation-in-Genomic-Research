"""Run the 100-SNP utility evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.common import (  # noqa: E402
    DEFAULT_COPIES,
    DEFAULT_DATASETS,
    build_context,
    evaluate_utility_100,
    parse_datasets,
    report_evaluation_dry_run,
    validate_evaluation,
)


def run(
    datasets: str = DEFAULT_DATASETS,
    copies: int = DEFAULT_COPIES,
    workers: int | None = None,
    dry_run: bool = False,
    no_overwrite_results: bool = False,
) -> int:
    ctx = build_context(dry_run=dry_run, no_overwrite_results=no_overwrite_results, workers=workers)
    selected = parse_datasets(datasets)
    code = validate_evaluation(ctx, selected, include_large_mia=False, only_100_snp=True, copies=copies)
    if code != 0:
        return code
    if dry_run:
        report_evaluation_dry_run(ctx, "utility_100_df_full.csv", selected, copies)
        return 0
    evaluate_utility_100(ctx, selected, copies=copies)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 100-SNP utility evaluation.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite-results", action="store_true")
    args = parser.parse_args()
    return run(
        datasets=args.datasets,
        copies=args.copies,
        workers=args.workers,
        dry_run=args.dry_run,
        no_overwrite_results=args.no_overwrite_results,
    )


if __name__ == "__main__":
    raise SystemExit(main())
