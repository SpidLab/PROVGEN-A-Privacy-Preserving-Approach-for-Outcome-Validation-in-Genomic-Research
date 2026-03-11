#!/usr/bin/env python3
"""Evaluation-only entrypoint for artifact experiments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluations only (no generation).")
    parser.add_argument("--datasets", type=str, default="hair,lactose,eye")
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--include-large-mia", action="store_true")
    parser.add_argument("--only-100-snp", action="store_true")
    parser.add_argument("--no-overwrite-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=["all", "gwas_standard", "gwas_maf", "mia_standard", "mia_large", "utility_standard", "utility_100"],
        default="all",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = [
        "python",
        str(root / "run_experiments.py"),
        "--mode",
        "evaluate",
        "--datasets",
        args.datasets,
        "--copies",
        str(args.copies),
        "--experiment",
        args.experiment,
    ]
    if args.include_large_mia:
        cmd.append("--include-large-mia")
    if args.only_100_snp:
        cmd.append("--only-100-snp")
    if args.no_overwrite_results:
        cmd.append("--no-overwrite-results")
    if args.dry_run:
        cmd.append("--dry-run")

    return subprocess.run(cmd, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
