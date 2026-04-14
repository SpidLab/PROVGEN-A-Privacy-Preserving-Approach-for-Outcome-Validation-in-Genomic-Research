"""Run the large-scale MIA evaluation."""

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
    LARGE_SCALE_EFFECTIVE_EPS,
    build_context,
    evaluate_mia,
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
    code = validate_evaluation(ctx, selected, include_large_mia=True, only_100_snp=False, copies=copies)
    if code != 0:
        return code
    if dry_run:
        report_evaluation_dry_run(ctx, "mia_experiments_results_large_scale.csv", selected, copies)
        return 0
    evaluate_mia(ctx, selected, LARGE_SCALE_EFFECTIVE_EPS, "mia_experiments_results_large_scale.csv", copies=copies)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the large-scale MIA evaluation.")
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
