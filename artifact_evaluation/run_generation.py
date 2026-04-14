#!/usr/bin/env python3
"""Paper-artifact generation dispatcher.

This command runs generation for the fixed experiment layout used by the
artifact: PROVGEN, LDP, PrivBayes, and DPSyn outputs under ``generated/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_evaluation.comparison_methods.generators.dpsyn import generation as dpsyn_generation
from artifact_evaluation.comparison_methods.generators.ldp import generation as ldp_generation
from artifact_evaluation.comparison_methods.generators.privbayes import generation as privbayes_generation
from artifact_evaluation import provgen_generation


def run_step(step: str, fn, **kwargs) -> int:
    print(f"[step] {step}")
    return fn(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-artifact generation only (no evaluations).")
    parser.add_argument("--datasets", type=str, default="hair,lactose,eye")
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--include-large-mia", action="store_true")
    parser.add_argument("--only-100-snp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generation-target",
        choices=["all", "proposed", "ldp", "proposed_dp_maf", "proposed_100", "privbayes", "dpsyn"],
        default="all",
    )
    args = parser.parse_args()

    if args.only_100_snp and args.generation_target not in {"all", "proposed_100", "privbayes", "dpsyn"}:
        print("[error] --only-100-snp is only compatible with --generation-target all|proposed_100|privbayes|dpsyn")
        return 2

    common = {"datasets": args.datasets, "copies": args.copies, "dry_run": args.dry_run}
    scale_target = "large" if args.include_large_mia else "standard"

    if args.generation_target == "all":
        steps: list[tuple[str, object, dict]] = []
        if args.only_100_snp:
            steps.extend(
                [
                    ("PROVGEN 100-SNP generation", provgen_generation.run, {**common, "target": "100-snp"}),
                    ("PrivBayes 100-SNP generation", privbayes_generation.run, common),
                    ("DPSyn 100-SNP generation", dpsyn_generation.run, common),
                ]
            )
        else:
            steps.extend(
                [
                    ("PROVGEN standard generation", provgen_generation.run, {**common, "target": scale_target}),
                    ("LDP standard generation", ldp_generation.run, {**common, "target": scale_target}),
                    ("PROVGEN MAF generation", provgen_generation.run, {**common, "target": "maf"}),
                    ("PROVGEN 100-SNP generation", provgen_generation.run, {**common, "target": "100-snp"}),
                    ("PrivBayes 100-SNP generation", privbayes_generation.run, common),
                    ("DPSyn 100-SNP generation", dpsyn_generation.run, common),
                ]
            )

        for label, fn, kwargs in steps:
            code = run_step(label, fn, **kwargs)
            if code != 0:
                return code
        return 0

    if args.generation_target == "proposed":
        return run_step("PROVGEN generation", provgen_generation.run, **common, target=scale_target)
    if args.generation_target == "ldp":
        return run_step("LDP generation", ldp_generation.run, **common, target=scale_target)
    if args.generation_target == "proposed_dp_maf":
        return run_step("PROVGEN MAF generation", provgen_generation.run, **common, target="maf")
    if args.generation_target == "proposed_100":
        return run_step("PROVGEN 100-SNP generation", provgen_generation.run, **common, target="100-snp")
    if args.generation_target == "privbayes":
        return run_step("PrivBayes 100-SNP generation", privbayes_generation.run, **common)
    if args.generation_target == "dpsyn":
        return run_step("DPSyn 100-SNP generation", dpsyn_generation.run, **common)

    print(f"[error] unsupported generation target: {args.generation_target}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
