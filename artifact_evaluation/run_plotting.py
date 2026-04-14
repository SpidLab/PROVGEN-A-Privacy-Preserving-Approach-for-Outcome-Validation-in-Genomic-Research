#!/usr/bin/env python3
"""Paper-artifact plotting dispatcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_evaluation.gwas_maf import plotting as gwas_maf_plotting
from artifact_evaluation.gwas_standard import plotting as gwas_standard_plotting
from artifact_evaluation.mia_large import plotting as mia_large_plotting
from artifact_evaluation.mia_standard import plotting as mia_standard_plotting
from artifact_evaluation.time_complexity import plotting as time_plotting
from artifact_evaluation.utility_100 import plotting as utility_100_plotting
from artifact_evaluation.utility_standard import plotting as utility_standard_plotting
from artifact_evaluation.common import DEFAULT_PLOTS_DIR, DEFAULT_RESULTS_DIR, resolve_plots_dir


def run_step(step: str, fn, **kwargs) -> list[Path]:
    print(f"[step] {step}")
    return fn(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figure PDFs from artifact_evaluation result CSVs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="path to folder containing result CSVs (default: results)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help="path to output figure folder (default: plots)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="deprecated alias for --plots-dir",
    )
    parser.add_argument(
        "--plot-target",
        choices=["all", "gwas_standard", "gwas_maf", "mia_standard", "mia_large", "utility_standard", "utility_100", "time"],
        default="all",
        help="which experiment-style plot group to generate",
    )
    parser.add_argument("--dry-run", action="store_true", help="render plots without writing figure files")
    args = parser.parse_args()

    plots_dir = args.figures_dir or args.plots_dir
    common = {"results_dir": args.results_dir, "plots_dir": plots_dir, "dry_run": args.dry_run}
    generated: list[Path] = []

    if args.plot_target == "all":
        steps: list[tuple[str, object, dict]] = [
            ("GWAS standard plots", gwas_standard_plotting.run, common),
            ("GWAS MAF plots", gwas_maf_plotting.run, common),
            ("MIA standard plots", mia_standard_plotting.run, common),
            ("MIA large-scale plots", mia_large_plotting.run, common),
            ("Utility standard plot hints", utility_standard_plotting.run, common),
            ("Utility 100-SNP plot hints", utility_100_plotting.run, common),
            ("Time complexity plot", time_plotting.run, common),
        ]
        for label, fn, kwargs in steps:
            generated.extend(run_step(label, fn, **kwargs))
    else:
        mapping = {
            "gwas_standard": ("GWAS standard plots", gwas_standard_plotting.run),
            "gwas_maf": ("GWAS MAF plots", gwas_maf_plotting.run),
            "mia_standard": ("MIA standard plots", mia_standard_plotting.run),
            "mia_large": ("MIA large-scale plots", mia_large_plotting.run),
            "utility_standard": ("Utility standard plot hints", utility_standard_plotting.run),
            "utility_100": ("Utility 100-SNP plot hints", utility_100_plotting.run),
            "time": ("Time complexity plot", time_plotting.run),
        }
        label, fn = mapping[args.plot_target]
        generated.extend(run_step(label, fn, **common))

    plots = resolve_plots_dir(plots_dir, dry_run=args.dry_run)
    if args.dry_run:
        print("[done] dry-run completed; no figures written")
    else:
        print(f"[done] figures written to {plots}")
    if generated:
        print("[plots]")
        for path in generated:
            print(f" - {path.resolve()}")


if __name__ == "__main__":
    main()
