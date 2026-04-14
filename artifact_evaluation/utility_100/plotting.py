"""Utility 100-SNP plotting entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.common import (  # noqa: E402
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    print_utility_plot_hint,
    resolve_results_dir,
)


def run(
    results_dir: str = DEFAULT_RESULTS_DIR,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    dry_run: bool = False,
) -> list[Path]:
    del plots_dir, dry_run
    results = resolve_results_dir(results_dir)
    print_utility_plot_hint("utility_100", results, "utility_100_df_full.csv")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the utility 100-SNP CSV location.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(results_dir=args.results_dir, plots_dir=args.plots_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
