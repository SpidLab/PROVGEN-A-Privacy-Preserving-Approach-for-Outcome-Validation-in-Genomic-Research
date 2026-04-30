"""Utility standard plotting entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    resolve_results_dir,
)


def run(
    results_dir: str = DEFAULT_RESULTS_DIR,
    plots_dir: str = "plots",
    dry_run: bool = False,
) -> list[Path]:
    results = resolve_results_dir(results_dir)
    summary_csv = results / "utility_summary.csv"
    summary_md = results / "utility_summary.md"
    prefix = "[dry-run]" if dry_run else "[info]"
    print(f"{prefix} utility standard has no plotting step; use {summary_csv}")
    print(f"{prefix} utility standard has no plotting step; use {summary_md}")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Report utility standard summary files.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(results_dir=args.results_dir, plots_dir=args.plots_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
