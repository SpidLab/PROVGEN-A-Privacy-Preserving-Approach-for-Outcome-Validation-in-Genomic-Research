"""Materialize the time complexity CSV used for plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_evaluation.common import DEFAULT_RESULTS_DIR, ensure_time_results, resolve_results_dir  # noqa: E402


def run(results_dir: str = DEFAULT_RESULTS_DIR, dry_run: bool = False) -> int:
    results = resolve_results_dir(results_dir)
    out = ensure_time_results(results, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] prepared time-complexity results at {out}")
    else:
        print(f"[done] {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the time-complexity CSV.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(results_dir=args.results_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
