"""Plot the standard-scale MIA figures."""

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
    load_mia_results,
    plot_mia,
    resolve_plots_dir,
    resolve_results_dir,
)


def run(
    results_dir: str = DEFAULT_RESULTS_DIR,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    dry_run: bool = False,
) -> list[Path]:
    results = resolve_results_dir(results_dir)
    plots = resolve_plots_dir(plots_dir, dry_run=dry_run)
    mia_df, _ = load_mia_results(results)
    return plot_mia(mia_df, plots, large_scale=False, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the standard-scale MIA figures.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(results_dir=args.results_dir, plots_dir=args.plots_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
