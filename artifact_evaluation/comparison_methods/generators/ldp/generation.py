"""Generation entrypoint for the internal LDP baseline."""

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
    maybe_generate_ldp,
    parse_datasets,
    validate_generation,
)


def run(datasets: str = DEFAULT_DATASETS, copies: int = DEFAULT_COPIES, target: str = "all", dry_run: bool = False) -> int:
    ctx = build_context(dry_run=dry_run)
    selected = parse_datasets(datasets)
    include_large_mia = target in {"all", "large"}
    code = validate_generation(ctx, selected, include_large_mia=include_large_mia, only_100_snp=False, copies=copies)
    if code != 0:
        return code
    maybe_generate_ldp(ctx, selected, include_large_mia=include_large_mia, validate_only=False, copies=copies)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate datasets for the LDP baseline.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--copies", type=int, default=DEFAULT_COPIES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", choices=["all", "standard", "large"], default="all")
    args = parser.parse_args()
    return run(datasets=args.datasets, copies=args.copies, target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
