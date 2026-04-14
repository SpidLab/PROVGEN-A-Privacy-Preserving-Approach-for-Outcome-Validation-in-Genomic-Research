"""Reusable PROVGEN dataset generation.

This module is the user-facing generation utility for applying PROVGEN to an
arbitrary genotype matrix. It is deliberately independent of the paper's fixed
experiment layout. For reproducing the artifact's predefined datasets and
epsilon schedules, use ``python -m artifact_evaluation.run_generation``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generation.core import generate_proposed_dataset


def load_matrix(path: str | Path) -> np.ndarray:
    """Load a genotype matrix from ``.npy`` or CSV."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    df = pd.read_csv(path)
    drop_cols = [
        col
        for col in df.columns
        if str(col).startswith("Unnamed") or str(col) in {"PATIENT_ID", "SNP_ID", "epsilon"}
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.to_numpy()


def save_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    """Save a generated genotype matrix as ``.npy`` or CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, matrix)
    else:
        pd.DataFrame(matrix).to_csv(path, index=False)
    return path


def generate_from_paths(
    input_dataset: str | Path,
    output_path: str | Path,
    epsilon: float,
    reference_dataset: str | Path | None = None,
) -> Path:
    """Generate one PROVGEN dataset from explicit input and output paths."""
    input_matrix = load_matrix(input_dataset)
    reference_matrix = load_matrix(reference_dataset or input_dataset)
    generated = generate_proposed_dataset(input_matrix, reference_matrix, epsilon)
    return save_matrix(generated, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one PROVGEN dataset from explicit paths.")
    parser.add_argument("--input", required=True, help="input genotype matrix path (.csv or .npy)")
    parser.add_argument("--output", required=True, help="output path (.csv or .npy)")
    parser.add_argument("--epsilon", type=float, required=True, help="privacy parameter passed to PROVGEN")
    parser.add_argument(
        "--reference",
        default=None,
        help="optional reference genotype matrix for the XOR mechanism; defaults to --input",
    )
    args = parser.parse_args()
    out = generate_from_paths(args.input, args.output, args.epsilon, reference_dataset=args.reference)
    print(f"[done] wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
