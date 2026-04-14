#!/usr/bin/env python3
"""Paper-artifact evaluation dispatcher."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_evaluation.gwas_maf import experiment as gwas_maf_experiment
from artifact_evaluation.gwas_standard import experiment as gwas_standard_experiment
from artifact_evaluation.mia_large import experiment as mia_large_experiment
from artifact_evaluation.mia_standard import experiment as mia_standard_experiment
from artifact_evaluation.time_complexity import experiment as time_experiment
from artifact_evaluation.utility_100 import experiment as utility_100_experiment
from artifact_evaluation.utility_standard import experiment as utility_standard_experiment


def run_step(step: str, fn, **kwargs) -> int:
    print(f"[step] {step}")
    return fn(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-artifact evaluations only (no generation).")
    parser.add_argument("--datasets", type=str, default="hair,lactose,eye")
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--include-large-mia", action="store_true")
    parser.add_argument("--only-100-snp", action="store_true")
    parser.add_argument("--no-overwrite-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=["all", "gwas_standard", "gwas_maf", "mia_standard", "mia_large", "utility_standard", "utility_100", "time"],
        default="all",
    )
    args = parser.parse_args()

    if args.only_100_snp and args.experiment not in {"all", "utility_100"}:
        print("[error] --only-100-snp is only compatible with --experiment all|utility_100")
        return 2

    common = {
        "datasets": args.datasets,
        "copies": args.copies,
        "workers": args.workers,
        "dry_run": args.dry_run,
        "no_overwrite_results": args.no_overwrite_results,
    }

    if args.only_100_snp:
        return run_step("Utility 100-SNP evaluation", utility_100_experiment.run, **common)

    if args.experiment == "all":
        steps: list[tuple[str, object, dict]] = [
            ("GWAS standard evaluation", gwas_standard_experiment.run, common),
            ("GWAS MAF evaluation", gwas_maf_experiment.run, common),
            ("MIA standard evaluation", mia_standard_experiment.run, common),
            ("Utility standard evaluation", utility_standard_experiment.run, common),
            ("Time complexity results", time_experiment.run, {"dry_run": args.dry_run}),
        ]
        if args.include_large_mia:
            steps.insert(3, ("MIA large-scale evaluation", mia_large_experiment.run, common))

        for label, fn, kwargs in steps:
            code = run_step(label, fn, **kwargs)
            if code != 0:
                return code
        return 0

    mapping = {
        "gwas_standard": ("GWAS standard evaluation", gwas_standard_experiment.run, common),
        "gwas_maf": ("GWAS MAF evaluation", gwas_maf_experiment.run, common),
        "mia_standard": ("MIA standard evaluation", mia_standard_experiment.run, common),
        "mia_large": ("MIA large-scale evaluation", mia_large_experiment.run, common),
        "utility_standard": ("Utility standard evaluation", utility_standard_experiment.run, common),
        "utility_100": ("Utility 100-SNP evaluation", utility_100_experiment.run, common),
        "time": ("Time complexity results", time_experiment.run, {"dry_run": args.dry_run}),
    }
    label, fn, kwargs = mapping[args.experiment]
    return run_step(label, fn, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
