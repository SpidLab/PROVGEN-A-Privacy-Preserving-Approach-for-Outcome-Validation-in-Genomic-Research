"""Generation entrypoint for the 100-SNP DPSyn baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.comparison_methods.generators.common import (  # noqa: E402
    DEFAULT_COPIES,
    DEFAULT_DATASETS,
    build_context,
    parse_datasets,
    run_dpsyn_generation,
    validate_generation,
)


def run(datasets: str = DEFAULT_DATASETS, copies: int = DEFAULT_COPIES, dry_run: bool = False) -> int:
    ctx = build_context(dry_run=dry_run)
    selected = parse_datasets(datasets)
    code = validate_generation(ctx, selected, include_large_mia=False, only_100_snp=True, copies=copies)
    if code != 0:
        return code
    run_dpsyn_generation(ctx, selected, validate_only=False, copies=copies)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 100-SNP datasets for DPSyn.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(datasets=args.datasets, copies=args.copies, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
