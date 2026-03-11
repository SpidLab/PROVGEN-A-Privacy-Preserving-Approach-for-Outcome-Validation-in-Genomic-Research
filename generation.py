#!/usr/bin/env python3
"""Generation-only entrypoint for artifact data synthesis."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generation only (no evaluations).")
    parser.add_argument("--datasets", type=str, default="hair,lactose,eye")
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--include-large-mia", action="store_true")
    parser.add_argument("--only-100-snp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generation-target",
        choices=["all", "proposed", "proposed_dp_maf", "proposed_100"],
        default="all",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = [
        "python",
        str(root / "run_experiments.py"),
        "--mode",
        "generate",
        "--datasets",
        args.datasets,
        "--copies",
        str(args.copies),
        "--generation-target",
        args.generation_target,
    ]
    if args.include_large_mia:
        cmd.append("--include-large-mia")
    if args.only_100_snp:
        cmd.append("--only-100-snp")
    if args.dry_run:
        cmd.append("--dry-run")

    return subprocess.run(cmd, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
