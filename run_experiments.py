#!/usr/bin/env python3
"""Standalone artifact experiment runner.

This script is designed to run from inside `artifact/` and use only files under it.

It supports:
- generation: proposed XOR, proposed_dp_maf, proposed_100
- evaluation: proposed/proposed_dp_maf metrics merged with precomputed baseline result CSVs
- validate-only mode to verify data/config wiring without heavy generation
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import subprocess
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cityblock
from scipy.stats import chi2_contingency, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class DATASET(Enum):
    hair = "hair"
    lactose = "lactose"
    eye = "eye"


EPSILON_BASES = {
    DATASET.lactose: 9091,
    DATASET.hair: 9389,
    DATASET.eye: 28396,
}

STANDARD_EFFECTIVE_EPS = [1, 2, 3, 4, 5]
LARGE_SCALE_EFFECTIVE_EPS = [1e-2, 1e-1, 1, 10, 100]
MAF_EPS = [0.1, 0.5, 1.0]

MIA_METHODS = ["hamming_distance", "xgboost", "decision_tree", "random_forest", "svm", "nn"]


@dataclass
class Context:
    root: Path
    workers: int
    no_overwrite_results: bool = False
    dry_run: bool = False

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cleansed_dir(self) -> Path:
        return self.data_dir / "cleansed"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def precomputed_results_dir(self) -> Path:
        return self.root / "precomputed_results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def next_results_path(ctx: Context, filename: str) -> Path:
    out = ctx.results_dir / filename
    if ctx.no_overwrite_results and out.exists():
        i = 1
        while True:
            candidate = ctx.results_dir / f"{out.stem}_new{i}{out.suffix}"
            if not candidate.exists():
                out = candidate
                break
            i += 1
    return out


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in df.columns if c and not str(c).startswith("Unnamed")]
    return df.loc[:, keep].copy()


def write_results_csv(ctx: Context, filename: str, rows: list[dict]) -> Path:
    out = next_results_path(ctx, filename)
    if ctx.dry_run:
        print(f"[dry-run] would write {out}")
        return out
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_results_dataframe(ctx: Context, filename: str, df: pd.DataFrame) -> Path:
    out = next_results_path(ctx, filename)
    if ctx.dry_run:
        print(f"[dry-run] would write {out}")
        return out
    clean_dataframe(df).to_csv(out, index=False)
    return out


def load_precomputed_results(ctx: Context, filename: str, datasets: list[DATASET]) -> pd.DataFrame:
    path = ctx.precomputed_results_dir / filename
    if not path.exists():
        return pd.DataFrame()
    df = clean_dataframe(pd.read_csv(path))
    if df.empty or "Dataset" not in df.columns:
        return df
    df["Dataset"] = df["Dataset"].astype(str).str.replace("DATASET.", "", regex=False)
    allowed = [d.value for d in datasets]
    return df[df["Dataset"].isin(allowed)].copy()


def merge_result_rows(ctx: Context, rows: list[dict], baseline_filename: str | None, datasets: list[DATASET]) -> pd.DataFrame:
    fresh = clean_dataframe(pd.DataFrame(rows))
    baseline = clean_dataframe(load_precomputed_results(ctx, baseline_filename, datasets)) if baseline_filename else pd.DataFrame()
    if baseline.empty:
        return fresh
    if fresh.empty:
        return baseline
    return pd.concat([baseline, fresh], ignore_index=True, sort=False)


def print_dry_run_preview(label: str, df: pd.DataFrame, max_rows: int = 5) -> None:
    if df.empty:
        print(f"[dry-run] {label}: no rows produced")
        return
    print(f"[dry-run] {label}: {len(df)} rows")
    print(df.head(max_rows).to_string(index=False))


def load_target_dataframe(ctx: Context, dataset: DATASET, snp_count: int = 0, idx: int = 0) -> pd.DataFrame:
    if snp_count:
        path = ctx.cleansed_dir / dataset.value / f"data_{snp_count}_{idx}.csv"
    else:
        path = ctx.cleansed_dir / dataset.value / "data.csv"
    return pd.read_csv(path, index_col=0)


def load_reference_dataframe(ctx: Context, dataset: DATASET, snp_count: int = 0, idx: int = 0) -> pd.DataFrame:
    if snp_count:
        path = ctx.cleansed_dir / dataset.value / f"reference_{snp_count}_{idx}.csv"
    else:
        path = ctx.cleansed_dir / dataset.value / "reference.csv"
    return pd.read_csv(path, index_col=0)


def load_target_data(ctx: Context, dataset: DATASET, snp_count: int = 0, idx: int = 0) -> np.ndarray:
    return load_target_dataframe(ctx, dataset, snp_count, idx).values


def load_reference_data(ctx: Context, dataset: DATASET, snp_count: int = 0, idx: int = 0) -> np.ndarray:
    return load_reference_dataframe(ctx, dataset, snp_count, idx).values


def shared_path(ctx: Context, dataset: DATASET, epsilon: float, method: str, snp_count: int = 0, idx: int = 0) -> Path:
    if method in {"proposed", "proposed_dp_maf", "ldp", "ldp_pp"}:
        base = ctx.data_dir / method / dataset.value
        if snp_count:
            return base / f"{epsilon:.4f}_{snp_count}_{idx}.npy"
        return base / f"{epsilon:.4f}_{idx}.npy"

    if method == "privbayes":
        base = ctx.data_dir / "privbayes"
        return base / f"{dataset.value}_{epsilon:.1f}_{snp_count}_{idx}.csv"

    if method == "dpsyn":
        base = ctx.data_dir / "dpsyn"
        return base / f"{dataset.value}_{epsilon:.4f}_{snp_count}_{idx}.csv"

    raise ValueError(f"Unsupported method: {method}")


def save_shared_data(ctx: Context, data: np.ndarray, dataset: DATASET, epsilon: float, method: str, snp_count: int = 0, idx: int = 0) -> None:
    out = shared_path(ctx, dataset, epsilon, method, snp_count, idx)
    if ctx.dry_run:
        print(f"[dry-run] generated {method} dataset in memory; would save {out}")
        return
    ensure_dir(out.parent)
    np.save(out, data)


def load_shared_data(ctx: Context, dataset: DATASET, epsilon: float, method: str, snp_count: int = 0, idx: int = 0) -> np.ndarray:
    path = shared_path(ctx, dataset, epsilon, method, snp_count, idx)
    if method in {"proposed", "proposed_dp_maf", "ldp", "ldp_pp"}:
        return np.load(path)
    if method == "privbayes":
        return pd.read_csv(path, index_col=0).values
    if method == "dpsyn":
        return pd.read_csv(path).values[:, :-1]
    raise ValueError(f"Unsupported method: {method}")


# --------------------
# Generation utilities
# --------------------
def encode(matrix: np.ndarray) -> np.ndarray:
    encoded = np.zeros((matrix.shape[0], matrix.shape[1] * 2), dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                encoded[i, 2 * j : 2 * j + 2] = [0, 0]
            elif matrix[i, j] == 1:
                encoded[i, 2 * j : 2 * j + 2] = [0, 1]
            else:
                encoded[i, 2 * j : 2 * j + 2] = [1, 1]
    return encoded


def decode(matrix: np.ndarray) -> np.ndarray:
    decoded = np.zeros((matrix.shape[0], matrix.shape[1] // 2), dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] // 2):
            decoded[i, j] = matrix[i, 2 * j] + matrix[i, 2 * j + 1]
    return decoded


def get_mafs(encoded_matrix: np.ndarray) -> np.ndarray:
    n, p = encoded_matrix.shape
    return np.sum(encoded_matrix.reshape(n, p // 2, 2), axis=(0, 2)) / np.full(p // 2, n * 2)


def transport(noisy_encoded_matrix: np.ndarray, target_mafs: np.ndarray) -> np.ndarray:
    noisy_mafs = get_mafs(noisy_encoded_matrix)
    points_to_flip = ((noisy_mafs - target_mafs) * noisy_encoded_matrix.shape[0] * 2).astype(int)

    for j in range(len(noisy_mafs)):
        if points_to_flip[j] == 0:
            continue
        snp_values = noisy_encoded_matrix[:, 2 * j : 2 * j + 2]
        value_flipped = 1 if points_to_flip[j] > 0 else 0
        value_indices = np.argwhere(snp_values == value_flipped)
        if value_indices.shape[0] == 0:
            continue
        choose_n = min(abs(points_to_flip[j]), value_indices.shape[0])
        chosen = np.random.choice(value_indices.shape[0], choose_n, replace=False)
        indices_to_flip = value_indices[chosen]
        for i, k in indices_to_flip:
            snp_values[i, k] = 1 - value_flipped
        noisy_encoded_matrix[:, 2 * j : 2 * j + 2] = snp_values
    return noisy_encoded_matrix


def xor_mechanism(matrix: np.ndarray, epsilon: float, reference_matrix: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    matrix = np.array(matrix)
    nrows, ncols = matrix.shape
    sens = ncols

    A = np.array(reference_matrix)
    A_nrows, A_ncols = A.shape
    M_11 = A.T @ A
    ones = np.ones((A_nrows, A_ncols))
    M_01 = ones.T @ A - M_11
    M_10 = A.T @ ones - M_11
    M_00 = A_nrows * np.ones((A_ncols, A_ncols)) - M_01 - M_11 - M_10
    M_1 = A.sum(axis=0)
    M_0 = A_nrows - M_1

    theta_tilde = np.log((M_11 * M_00 + eps) / (M_10 * M_01 + eps))
    diag_val = np.log((M_1 + eps) / (M_0 + eps))
    theta_tilde = theta_tilde - np.diag(np.diag(theta_tilde)) + np.diag(diag_val)

    f_theta = np.linalg.norm(theta_tilde, "fro")
    theta = (epsilon / (sens * f_theta)) * theta_tilde

    off_diagonal = theta - np.diag(np.diag(theta))
    min_value = 2 * np.sum((off_diagonal < 0) * off_diagonal, axis=1) + np.diag(theta)
    max_value = 2 * np.sum((off_diagonal > 0) * off_diagonal, axis=1) + np.diag(theta)
    coeff = np.exp(min_value) - 1
    bound = (coeff < 0) * min_value + (coeff > 0) * max_value
    b_probs = (1 + coeff / (1 + np.exp(bound))) / 2

    B = np.zeros((nrows, ncols), dtype=int)
    for j in range(nrows):
        B[j] = np.random.binomial(n=1, p=b_probs, size=(ncols))
    return np.logical_xor(matrix, B).astype(int)


def generate_ldp_dataset(matrix: np.ndarray, epsilon_per_snp: float) -> np.ndarray:
    nrows, ncols = matrix.shape
    p = np.exp(epsilon_per_snp / ncols) / (np.exp(epsilon_per_snp / ncols) + 2)
    perturbed = np.copy(matrix)
    flip_mask = np.random.binomial(1, p, size=matrix.shape)
    random_values = np.random.choice([0, 1, 2], size=matrix.shape)
    perturbed[flip_mask == 1] = random_values[flip_mask == 1]
    return perturbed


def generate_proposed_dataset(data: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    encoded = encode(data)
    target_mafs = get_mafs(encoded)
    xor_binary = xor_mechanism(encoded, epsilon, encode(reference))
    return decode(transport(xor_binary, target_mafs))


def _effective_eps(include_large_mia: bool) -> list[float]:
    out = list(STANDARD_EFFECTIVE_EPS)
    if include_large_mia:
        out.extend(LARGE_SCALE_EFFECTIVE_EPS)
    # keep order stable and remove duplicates
    return list(dict.fromkeys(out))


def maybe_generate_proposed(ctx: Context, datasets: list[DATASET], include_large_mia: bool, validate_only: bool, copies: int) -> None:
    eps_list = _effective_eps(include_large_mia)
    total_tasks = len(datasets) * len(eps_list) * copies
    pbar = tqdm(total=total_tasks, desc="Generate proposed", dynamic_ncols=True)
    for dataset in datasets:
        base = EPSILON_BASES[dataset]
        target = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)
        for eff_eps in eps_list:
            total_eps = eff_eps * base
            for idx in range(copies):
                proposed_out = shared_path(ctx, dataset, total_eps, "proposed", idx=idx)
                if validate_only:
                    pbar.update(1)
                    continue
                if ctx.dry_run or not proposed_out.exists():
                    proposed = generate_proposed_dataset(target, reference, total_eps)
                    save_shared_data(ctx, proposed, dataset, total_eps, "proposed", idx=idx)
                pbar.update(1)
    pbar.close()


def maybe_generate_ldp(ctx: Context, datasets: list[DATASET], include_large_mia: bool, validate_only: bool, copies: int) -> None:
    eps_list = _effective_eps(include_large_mia)
    total_tasks = len(datasets) * len(eps_list) * copies
    pbar = tqdm(total=total_tasks, desc="Generate ldp", dynamic_ncols=True)
    for dataset in datasets:
        base = EPSILON_BASES[dataset]
        target = load_target_data(ctx, dataset)
        for eff_eps in eps_list:
            total_eps = eff_eps * base
            for idx in range(copies):
                ldp_out = shared_path(ctx, dataset, total_eps, "ldp", idx=idx)
                if validate_only:
                    pbar.update(1)
                    continue
                if not ldp_out.exists():
                    ldp = generate_ldp_dataset(target, eff_eps)
                    save_shared_data(ctx, ldp, dataset, total_eps, "ldp", idx=idx)
                pbar.update(1)
    pbar.close()


def maybe_generate_proposed_dp_maf(ctx: Context, datasets: list[DATASET], validate_only: bool, copies: int) -> None:
    total_maf = len(datasets) * len(MAF_EPS) * copies
    pbar_maf = tqdm(total=total_maf, desc="Generate proposed_dp_maf", dynamic_ncols=True)
    for dataset in datasets:
        target = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)
        for eps_maf in MAF_EPS:
            for idx in range(copies):
                out = shared_path(ctx, dataset, eps_maf, "proposed_dp_maf", idx=idx)
                if validate_only:
                    pbar_maf.update(1)
                    continue
                if ctx.dry_run or not out.exists():
                    proposed_maf = generate_proposed_dataset(target, reference, eps_maf)
                    save_shared_data(ctx, proposed_maf, dataset, eps_maf, "proposed_dp_maf", idx=idx)
                pbar_maf.update(1)
    pbar_maf.close()


def maybe_generate_core(ctx: Context, datasets: list[DATASET], include_large_mia: bool, validate_only: bool, copies: int) -> None:
    eps_lists = [STANDARD_EFFECTIVE_EPS]
    if include_large_mia:
        eps_lists.append(LARGE_SCALE_EFFECTIVE_EPS)

    total_main = len(datasets) * sum(len(x) for x in eps_lists) * copies
    total_maf = len(datasets) * len(MAF_EPS) * copies
    pbar_main = tqdm(total=total_main, desc="Generate proposed/ldp", dynamic_ncols=True)
    pbar_maf = tqdm(total=total_maf, desc="Generate proposed_dp_maf", dynamic_ncols=True)

    for dataset in datasets:
        base = EPSILON_BASES[dataset]
        target = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)

        for eps_list in eps_lists:
            for eff_eps in eps_list:
                total_eps = eff_eps * base
                for idx in range(copies):
                    proposed_out = shared_path(ctx, dataset, total_eps, "proposed", idx=idx)
                    ldp_out = shared_path(ctx, dataset, total_eps, "ldp", idx=idx)

                    if validate_only:
                        pbar_main.update(1)
                        continue
                    if ctx.dry_run or not proposed_out.exists():
                        proposed = generate_proposed_dataset(target, reference, total_eps)
                        save_shared_data(ctx, proposed, dataset, total_eps, "proposed", idx=idx)
                    if ctx.dry_run or not ldp_out.exists():
                        ldp = generate_ldp_dataset(target, eff_eps)
                        save_shared_data(ctx, ldp, dataset, total_eps, "ldp", idx=idx)
                    pbar_main.update(1)

        for eps_maf in MAF_EPS:
            for idx in range(copies):
                out = shared_path(ctx, dataset, eps_maf, "proposed_dp_maf", idx=idx)
                if validate_only:
                    pbar_maf.update(1)
                    continue
                if ctx.dry_run or not out.exists():
                    proposed_maf = generate_proposed_dataset(target, reference, eps_maf)
                    save_shared_data(ctx, proposed_maf, dataset, eps_maf, "proposed_dp_maf", idx=idx)
                pbar_maf.update(1)

    pbar_main.close()
    pbar_maf.close()


def maybe_generate_100_snp_methods(ctx: Context, datasets: list[DATASET], validate_only: bool, copies: int) -> None:
    # proposed for 100 SNP setting (epsilon total = 100 * [1..5])
    total_tasks = len(datasets) * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="Generate proposed 100-SNP", dynamic_ncols=True)
    for dataset in datasets:
        for eff_eps in STANDARD_EFFECTIVE_EPS:
            total_eps = 100 * eff_eps
            for idx in range(copies):
                out = shared_path(ctx, dataset, total_eps, "proposed", snp_count=100, idx=idx)
                if validate_only:
                    pbar.update(1)
                    continue
                if not out.exists():
                    data = load_target_data(ctx, dataset, snp_count=100, idx=idx)
                    ref = load_reference_data(ctx, dataset, snp_count=100, idx=idx)
                    proposed = generate_proposed_dataset(data, ref, total_eps)
                    save_shared_data(ctx, proposed, dataset, total_eps, "proposed", snp_count=100, idx=idx)
                pbar.update(1)
    pbar.close()


def run_privbayes_generation(ctx: Context, datasets: list[DATASET], validate_only: bool, copies: int) -> None:
    script = ctx.root / "comparison_methods" / "PrivBayes" / "experiment.py"
    if not script.exists():
        print("[warn] Skipping PrivBayes generation: artifact/comparison_methods/PrivBayes/experiment.py not found")
        return

    total_tasks = len(datasets) * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="Generate PrivBayes 100-SNP", dynamic_ncols=True)
    for dataset in datasets:
        for eff_eps in STANDARD_EFFECTIVE_EPS:
            total_eps = float(100 * eff_eps)
            for idx in range(copies):
                out = shared_path(ctx, dataset, total_eps, "privbayes", snp_count=100, idx=idx)
                if out.exists():
                    pbar.update(1)
                    continue
                if validate_only:
                    pbar.update(1)
                    continue
                cmd = [
                    "python",
                    str(script),
                    dataset.value,
                    "100",
                    str(total_eps),
                    "100",
                    str(idx),
                ]
                subprocess.run(cmd, cwd=ctx.root / "comparison_methods" / "PrivBayes", check=True)
                pbar.update(1)
    pbar.close()


def run_dpsyn_generation(ctx: Context, datasets: list[DATASET], validate_only: bool, copies: int) -> None:
    script = ctx.root / "comparison_methods" / "DPSyn" / "experiment.py"
    if not script.exists():
        print("[warn] Skipping DPSyn generation: artifact/comparison_methods/DPSyn/experiment.py not found")
        return

    total_tasks = len(datasets) * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="Generate DPSyn 100-SNP", dynamic_ncols=True)
    for dataset in datasets:
        for eff_eps in STANDARD_EFFECTIVE_EPS:
            total_eps = float(100 * eff_eps)
            for idx in range(copies):
                out = shared_path(ctx, dataset, total_eps, "dpsyn", snp_count=100, idx=idx)
                if out.exists():
                    pbar.update(1)
                    continue
                if validate_only:
                    pbar.update(1)
                    continue

                priv_data = ctx.cleansed_dir / dataset.value / f"data_100_{idx}.csv"
                params = ctx.root / "comparison_methods" / "DPSyn" / "config" / f"parameters_{dataset.value}_{total_eps:.4f}_100_{idx}.json"
                datatype = ctx.root / "comparison_methods" / "DPSyn" / "config" / f"column_datatypes_{dataset.value}_{total_eps:.4f}_100_{idx}.json"
                marginal = ctx.root / "comparison_methods" / "DPSyn" / "config" / f"eps_{total_eps:.4f}.yaml"
                target_path = out

                missing = [p for p in [params, datatype, marginal] if not p.exists()]
                if missing:
                    print(f"[warn] Missing DPSyn config for {dataset.value} eps={total_eps} idx={idx}; skipping")
                    pbar.update(1)
                    continue

                cmd = [
                    "python",
                    str(script),
                    "--priv_data",
                    str(priv_data),
                    "--priv_data_name",
                    f"{dataset.value}_{total_eps:.4f}_100_{idx}",
                    "--config",
                    str(ctx.root / "comparison_methods" / "DPSyn" / "config" / "data.yaml"),
                    "--n",
                    "100",
                    "--params",
                    str(params),
                    "--datatype",
                    str(datatype),
                    "--marginal_config",
                    str(marginal),
                    "--target_path",
                    str(target_path),
                    "--synthetic_count",
                    "100",
                ]
                subprocess.run(cmd, cwd=ctx.root / "comparison_methods" / "DPSyn", check=True)
                pbar.update(1)
    pbar.close()


# --------------------
# Evaluation functions
# --------------------
def count(records: np.ndarray) -> tuple[int, int, int]:
    d = Counter(records)
    return d[0], d[1], d[2]


def calc_chi_pvalues(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    p_values = []
    for j in range(matrix_a.shape[1]):
        case_count = count(matrix_a[:, j])
        ctrl_count = count(matrix_b[:, j])
        cont_table = [[], []]
        for cs, ct in zip(case_count, ctrl_count):
            if cs != 0 or ct != 0:
                cont_table[0].append(cs)
                cont_table[1].append(ct)
        p_values.append(chi2_contingency(np.array(cont_table))[1])
    return np.array(p_values)


def calc_or_pvalues(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    p_values = []
    for j in range(matrix_a.shape[1]):
        dtr_case = [x + 0.5 for x in count(matrix_a[:, j])]
        dtr_ctrl = [x + 0.5 for x in count(matrix_b[:, j])]
        try:
            odd_ratio = (dtr_ctrl[0] * (dtr_case[1] + dtr_case[2])) / (dtr_case[0] * (dtr_ctrl[1] + dtr_ctrl[2]))
            se_ln_or = math.sqrt(
                1 / (dtr_case[1] + dtr_case[2])
                + 1 / dtr_case[0]
                + 1 / (dtr_ctrl[1] + dtr_ctrl[2])
                + 1 / dtr_ctrl[0]
            )
            z_value = math.log(odd_ratio) / se_ln_or
            p_value = norm.sf(abs(z_value)) * 2
        except (ZeroDivisionError, ValueError):
            p_value = 1.0
        p_values.append(p_value)
    return np.array(p_values)


def get_significant_snps(p_values: np.ndarray, top_metric: str, top_metric_value: float) -> dict[int, float]:
    if top_metric == "ratio":
        n = int(len(p_values) * top_metric_value)
        idx = np.argsort(p_values)[:n]
    elif top_metric == "threshold":
        idx = np.where(p_values < top_metric_value)[0]
    else:
        raise ValueError(f"Invalid top_metric: {top_metric}")
    return {i: p_values[i] for i in idx}


def report_error(p_values: np.ndarray, error_type: str, error_rate: float, top_metric: str, top_metric_value: float) -> dict[int, float]:
    if error_type == "shifting":
        snps = get_significant_snps(p_values, top_metric, top_metric_value)
        indices = list(snps.keys())
        n_err = int(len(indices) * error_rate)
        if n_err == 0:
            return snps
        error_indices = set(np.random.choice(indices, n_err, replace=False))
        out = {}
        for idx in indices:
            if idx in error_indices:
                shift = np.random.choice([-1, 1])
                new_idx = (idx + shift) % len(p_values)
                out[new_idx] = snps[idx]
            else:
                out[idx] = snps[idx]
        return out

    if error_type == "flipping":
        p_mod = p_values.copy()
        n_err = int(len(p_values) * error_rate)
        if n_err > 0:
            err_idx = np.random.choice(np.arange(len(p_values)), n_err, replace=False)
            p_mod[err_idx] = np.random.uniform(0, 1, n_err)
        return get_significant_snps(p_mod, top_metric, top_metric_value)

    if error_type == "noise":
        p_mod = np.clip(p_values + np.random.normal(0, error_rate, size=len(p_values)), 0, 1)
        return get_significant_snps(p_mod, top_metric, top_metric_value)

    raise ValueError(f"Unknown error type: {error_type}")


def calc_gwas_reproducibility(
    target_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    original_matrix: np.ndarray,
    gwas_metric: str,
    error_type: str,
    error_rate: float,
    top_metric: str = "threshold",
    top_metric_value: float = 0.05,
    tolerance_factor: float = 0.8,
) -> float:
    gwas_fn = calc_chi_pvalues if gwas_metric == "chi2" else calc_or_pvalues
    p_values = gwas_fn(original_matrix, reference_matrix)
    report_snps = report_error(p_values, error_type, error_rate, top_metric, top_metric_value)

    dp_p_values = gwas_fn(target_matrix, reference_matrix)
    if top_metric == "ratio":
        significant_snps_dp = get_significant_snps(dp_p_values, "ratio", top_metric_value / tolerance_factor)
    else:
        significant_snps_dp = get_significant_snps(dp_p_values, "threshold", top_metric_value / tolerance_factor)

    report_idx = set(report_snps.keys())
    dp_idx = set(significant_snps_dp.keys())
    retained = report_idx.intersection(dp_idx)
    return len(retained) / len(report_idx) if report_idx else 0.0


def calc_point_error(data_matrix: np.ndarray, dp_matrix: np.ndarray) -> float:
    return float(np.mean(data_matrix != dp_matrix))


def calc_sample_distance(data_matrix: np.ndarray, dp_matrix: np.ndarray) -> float:
    d = [cityblock(data_matrix[i], dp_matrix[i]) / data_matrix.shape[1] for i in range(data_matrix.shape[0])]
    return float(np.mean(d))


def calc_mean_error(data_matrix: np.ndarray, dp_matrix: np.ndarray) -> float:
    return float(np.abs(np.mean(dp_matrix, axis=0) - np.mean(data_matrix, axis=0)).mean())


def calc_variance_error(data_matrix: np.ndarray, dp_matrix: np.ndarray) -> float:
    return float(np.abs(np.var(dp_matrix, axis=0) - np.var(data_matrix, axis=0)).mean())


def get_prob(arr: np.ndarray) -> list[float]:
    c = count(arr)
    s = sum(c)
    return [x / s for x in c]


def get_probs(matrix: np.ndarray) -> np.ndarray:
    return np.array([get_prob(matrix[:, i]) for i in range(matrix.shape[1])])


def get_maf(matrix: np.ndarray) -> np.ndarray:
    return np.array([(2 * x[2] + x[1]) / 2 for x in get_probs(matrix)])


def log_likelihood_test(sample: np.ndarray, case_mafs: np.ndarray, control_mafs: np.ndarray) -> float:
    score = 0.0
    for i, x in enumerate(sample):
        if 0.05 < case_mafs[i] < 1 and 0 < control_mafs[i] < 1:
            score += x * math.log(case_mafs[i] / control_mafs[i]) + (1 - x) * math.log((1 - case_mafs[i]) / (1 - control_mafs[i]))
    return score


def get_lrt_threshold(control_matrix: np.ndarray, case_mafs: np.ndarray, control_mafs: np.ndarray, err_rate: float = 0.05) -> float:
    scores = [log_likelihood_test(control_matrix[i], case_mafs, control_mafs) for i in range(control_matrix.shape[0])]
    return sorted(scores)[int(len(scores) * (1 - err_rate))]


def calc_log_likelihood_infr_power(case_matrix: np.ndarray, control_matrix: np.ndarray, dp_matrix: np.ndarray, err_rate: float = 0.05) -> float:
    case_mafs = get_maf(dp_matrix)
    control_mafs = get_maf(control_matrix)
    threshold = get_lrt_threshold(control_matrix, case_mafs, control_mafs, err_rate)
    scores = [log_likelihood_test(case_matrix[i], case_mafs, control_mafs) for i in range(case_matrix.shape[0])]
    return float(np.count_nonzero(np.array(scores) > threshold) / len(scores))


def get_min_hd(target: np.ndarray, group: np.ndarray) -> int:
    return min(np.sum(np.abs(target - group[idx, :])) for idx in range(group.shape[0]))


def hamming_distance(case_matrix: np.ndarray, control_matrix: np.ndarray, dp_matrix: np.ndarray, err_rate: float = 0.05) -> float:
    hd = np.array([get_min_hd(control_matrix[idx], dp_matrix) for idx in range(control_matrix.shape[0])])
    gamma_hd = sorted(hd, reverse=True)[int((1 - err_rate) * len(hd)) - 1]
    eval_results = np.array([get_min_hd(case_matrix[idx], dp_matrix) for idx in range(case_matrix.shape[0])])
    return float(np.count_nonzero(eval_results < gamma_hd) / len(eval_results))


def neural_network(X_train: np.ndarray, y_train: np.ndarray):
    from tensorflow.keras.layers import Dense, LeakyReLU
    from tensorflow.keras.models import Sequential

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model, scaler


def run_mia_experiment_with_split(target_data: np.ndarray, shared_data: np.ndarray, reference_data: np.ndarray, experiment_name: str) -> float:
    if experiment_name == "hamming_distance":
        shared_train, shared_test = train_test_split(shared_data, test_size=0.2, random_state=42)
        test_indices = range(shared_test.shape[0])
        X_test = target_data[test_indices]
        return hamming_distance(X_test, reference_data, shared_train)

    if experiment_name == "log_likelihood":
        shared_train, shared_test = train_test_split(shared_data, test_size=0.2, random_state=42)
        test_indices = range(shared_test.shape[0])
        X_test = target_data[test_indices]
        return calc_log_likelihood_infr_power(X_test, reference_data, shared_train)

    shared_train, shared_test = train_test_split(shared_data, test_size=0.2, random_state=42)
    X_train = np.vstack([shared_train, reference_data])
    y_train = np.hstack([np.ones(shared_train.shape[0]), np.zeros(reference_data.shape[0])])
    test_indices = range(shared_test.shape[0])
    X_test = target_data[test_indices]
    y_test = np.ones(X_test.shape[0])

    if experiment_name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)
        pred = model.predict(X_test)
    elif experiment_name == "decision_tree":
        model = DecisionTreeClassifier().fit(X_train, y_train)
        pred = model.predict(X_test)
    elif experiment_name == "random_forest":
        model = RandomForestClassifier(max_depth=2).fit(X_train, y_train)
        pred = model.predict(X_test)
    elif experiment_name == "svm":
        model = make_pipeline(StandardScaler(), SVC(gamma="auto")).fit(X_train, y_train)
        pred = model.predict(X_test)
    elif experiment_name == "nn":
        model, scaler = neural_network(X_train, y_train)
        pred = (model.predict(scaler.transform(X_test), verbose=0) > 0.5).astype(int).flatten()
    else:
        raise ValueError(f"Unknown MIA method: {experiment_name}")

    pred = (np.array(pred) > 0.5).astype(int)
    return float(accuracy_score(y_test, pred))


def evaluate_gwas(ctx: Context, datasets: list[DATASET], copies: int) -> None:
    rows = []
    total_tasks = len(datasets) * 2 * 2 * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="GWAS standard", dynamic_ncols=True)
    for dataset in datasets:
        base = EPSILON_BASES[dataset]
        original = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)
        for gwas_type in ["chi2", "odds"]:
            for error_type in ["flipping", "noise"]:
                for method in ["proposed"]:
                    for eff_eps in STANDARD_EFFECTIVE_EPS:
                        total_eps = eff_eps * base
                        for idx in range(copies):
                            try:
                                target = load_shared_data(ctx, dataset, total_eps, method, idx=idx)
                            except FileNotFoundError:
                                pbar.update(1)
                                continue
                            base_ret = calc_gwas_reproducibility(target, reference, original, gwas_type, error_type, 0.0)
                            rows.append(
                                {
                                    "Dataset": dataset.value,
                                    "GWAS Type": gwas_type,
                                    "Error Type": error_type,
                                    "Error Rate": 0.0,
                                    "Approach": method,
                                    "Epsilon": eff_eps,
                                    "Retention Ratio": base_ret,
                                }
                            )
                            for er in [x / 10 for x in range(1, 11)]:
                                val = calc_gwas_reproducibility(target, reference, original, gwas_type, error_type, er)
                                rows.append(
                                    {
                                        "Dataset": dataset.value,
                                        "GWAS Type": gwas_type,
                                        "Error Type": error_type,
                                        "Error Rate": er,
                                        "Approach": method,
                                        "Epsilon": eff_eps,
                                        "Retention Ratio": base_ret - val,
                                    }
                                )
                            pbar.update(1)

    ensure_dir(ctx.results_dir)
    merged = merge_result_rows(ctx, rows, "gwas_df_full_baselines.csv", datasets)
    out = write_results_dataframe(ctx, "gwas_df_full.csv", merged)
    pbar.close()
    print(f"[done] {out}")
    if ctx.dry_run:
        print_dry_run_preview("GWAS standard", merged)


def evaluate_gwas_maf(ctx: Context, datasets: list[DATASET], copies: int) -> None:
    rows = []
    total_tasks = len(datasets) * 2 * 2 * len(MAF_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="GWAS MAF", dynamic_ncols=True)
    for dataset in datasets:
        original = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)
        for gwas_type in ["chi2", "odds"]:
            for error_type in ["flipping", "noise"]:
                for eps_maf in MAF_EPS:
                    for idx in range(copies):
                        try:
                            target = load_shared_data(ctx, dataset, eps_maf, "proposed_dp_maf", idx=idx)
                        except FileNotFoundError:
                            pbar.update(1)
                            continue
                        base_ret = calc_gwas_reproducibility(target, reference, original, gwas_type, error_type, 0.0)
                        rows.append(
                            {
                                "Dataset": dataset.value,
                                "GWAS Type": gwas_type,
                                "Error Type": error_type,
                                "Error Rate": 0.0,
                                "Approach": "proposed_dp_maf",
                                "Epsilon": eps_maf,
                                "Retention Ratio": base_ret,
                            }
                        )
                        for er in [x / 10 for x in range(1, 11)]:
                            val = calc_gwas_reproducibility(target, reference, original, gwas_type, error_type, er)
                            rows.append(
                                {
                                    "Dataset": dataset.value,
                                    "GWAS Type": gwas_type,
                                    "Error Type": error_type,
                                    "Error Rate": er,
                                    "Approach": "proposed_dp_maf",
                                    "Epsilon": eps_maf,
                                    "Retention Ratio": base_ret - val,
                                }
                            )
                        pbar.update(1)

    out = write_results_csv(ctx, "gwas_df_full_maf.csv", rows)
    pbar.close()
    print(f"[done] {out}")
    if ctx.dry_run:
        print_dry_run_preview("GWAS MAF", pd.DataFrame(rows))


def evaluate_mia(ctx: Context, datasets: list[DATASET], effective_eps: Iterable[float], out_name: str, copies: int) -> None:
    rows = []
    n_methods = sum(6 if d == DATASET.eye else 5 for d in datasets)
    total_tasks = n_methods * len(list(effective_eps)) * copies
    pbar = tqdm(total=total_tasks, desc=f"MIA ({out_name})", dynamic_ncols=True)
    for dataset in datasets:
        base = EPSILON_BASES[dataset]
        target = load_target_data(ctx, dataset)
        reference = load_reference_data(ctx, dataset)
        for method_name in MIA_METHODS:
            if method_name == "nn" and dataset != DATASET.eye:
                continue
            for eff_eps in effective_eps:
                total_eps = eff_eps * base
                for idx in range(copies):
                    try:
                        shared = load_shared_data(ctx, dataset, total_eps, "proposed", idx=idx)
                    except FileNotFoundError:
                        shared = None
                    if shared is not None:
                        val = run_mia_experiment_with_split(target, shared, reference, method_name)
                        rows.append(
                            {
                                "Dataset": dataset.value,
                                "Epsilon": eff_eps,
                                "Approach": "proposed",
                                "MIAMethod": method_name,
                                "MIAResult": val,
                                "Group": idx,
                            }
                        )

                    pbar.update(1)

    merged = merge_result_rows(ctx, rows, out_name.replace(".csv", "_baselines.csv"), datasets)
    out = write_results_dataframe(ctx, out_name, merged)
    pbar.close()
    print(f"[done] {out}")
    if ctx.dry_run:
        print_dry_run_preview(out_name, merged)


def print_utility_summary(out: Path, df: pd.DataFrame, label: str) -> None:
    df = clean_dataframe(df)
    if df.empty:
        print(f"[warn] no utility rows produced for {label}")
        return

    summary = (
        df.groupby(["Dataset", "Utility Metric", "Approach", "Epsilon"], as_index=False)["Utility"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "Std", "count": "Count"})
    )
    summary["Std"] = summary["Std"].fillna(0.0)
    print(f"[summary] utility results: {label}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"[csv] {out}")


def evaluate_utility(ctx: Context, datasets: list[DATASET], copies: int) -> None:
    metrics = {
        "point_error": calc_point_error,
        "calc_sample_distance": calc_sample_distance,
        "mean_error": calc_mean_error,
        "variance_error": calc_variance_error,
    }

    rows = []
    total_tasks = len(datasets) * len(metrics) * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="Utility standard", dynamic_ncols=True)
    for dataset in datasets:
        original = load_target_data(ctx, dataset)
        base = EPSILON_BASES[dataset]
        for metric_name, metric_fn in metrics.items():
            for method in ["proposed"]:
                for eff_eps in STANDARD_EFFECTIVE_EPS:
                    total_eps = eff_eps * base
                    for idx in range(copies):
                        try:
                            shared = load_shared_data(ctx, dataset, total_eps, method, idx=idx)
                        except FileNotFoundError:
                            pbar.update(1)
                            continue
                        rows.append(
                            {
                                "Dataset": dataset.value,
                                "Epsilon": eff_eps,
                                "Approach": method,
                                "Utility Metric": metric_name,
                                "Utility": metric_fn(original, shared),
                            }
                        )
                        pbar.update(1)

    merged = merge_result_rows(ctx, rows, "utility_df_full_baselines.csv", datasets)
    out = write_results_dataframe(ctx, "utility_df_full.csv", merged)
    pbar.close()
    print(f"[done] {out}")
    print_utility_summary(out, merged, "standard")


def evaluate_utility_100(ctx: Context, datasets: list[DATASET], copies: int) -> None:
    metrics = {
        "point_error": calc_point_error,
        "calc_sample_distance": calc_sample_distance,
        "mean_error": calc_mean_error,
        "variance_error": calc_variance_error,
    }

    rows = []
    total_tasks = len(datasets) * len(metrics) * len(STANDARD_EFFECTIVE_EPS) * copies
    pbar = tqdm(total=total_tasks, desc="Utility 100-SNP", dynamic_ncols=True)
    for dataset in datasets:
        for metric_name, metric_fn in metrics.items():
            for method in ["proposed"]:
                for eff_eps in STANDARD_EFFECTIVE_EPS:
                    total_eps = 100 * eff_eps
                    for idx in range(copies):
                        try:
                            original = load_target_data(ctx, dataset, snp_count=100, idx=idx)
                            shared = load_shared_data(ctx, dataset, total_eps, method, snp_count=100, idx=idx)
                        except FileNotFoundError:
                            pbar.update(1)
                            continue
                        rows.append(
                            {
                                "Dataset": dataset.value,
                                "Epsilon": eff_eps,
                                "Approach": method,
                                "Utility Metric": metric_name,
                                "Utility": metric_fn(original, shared),
                            }
                        )
                        pbar.update(1)

    merged = merge_result_rows(ctx, rows, "utility_100_df_full_baselines.csv", datasets)
    out = write_results_dataframe(ctx, "utility_100_df_full.csv", merged)
    pbar.close()
    print(f"[done] {out}")
    print_utility_summary(out, merged, "100-SNP")


def validate_inputs(ctx: Context, datasets: list[DATASET], include_large_mia: bool, only_100_snp: bool, copies: int) -> int:
    missing = []
    for dataset in datasets:
        d = ctx.cleansed_dir / dataset.value
        if not only_100_snp:
            for name in ["data.csv", "reference.csv"]:
                if not (d / name).exists():
                    missing.append(str(d / name))
        for idx in range(copies):
            for name in [f"data_100_{idx}.csv", f"reference_100_{idx}.csv"]:
                if not (d / name).exists():
                    missing.append(str(d / name))

    if include_large_mia:
        print("[info] large-scale MIA enabled; this requires generating additional shared data at eps=[1e-2,1e-1,1,10,100].")

    if missing:
        print("[error] missing required inputs:")
        for p in missing[:40]:
            print(" -", p)
        if len(missing) > 40:
            print(f" - ... and {len(missing)-40} more")
        return 1

    print("[ok] input validation passed")
    return 0


def main() -> int:
    experiment_choices = [
        "all",
        "gwas_standard",
        "gwas_maf",
        "mia_standard",
        "mia_large",
        "utility_standard",
        "utility_100",
    ]
    parser = argparse.ArgumentParser(description="Run standalone artifact pipeline from artifact/ folder")
    parser.add_argument("--mode", choices=["validate", "generate", "evaluate", "all"], default="validate")
    parser.add_argument("--include-large-mia", action="store_true", help="also generate/evaluate MIA at eps 1e-2..1e2")
    parser.add_argument(
        "--only-100-snp",
        action="store_true",
        help="run/validate only the 100-SNP branch (proposed_100 generation and utility_100 evaluation with precomputed baselines)",
    )
    parser.add_argument("--copies", type=int, default=10, help="number of replicate groups to process (default: 10)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--dry-run", action="store_true", help="execute computations without writing generated data or result CSVs")
    parser.add_argument(
        "--no-overwrite-results",
        action="store_true",
        help="write result CSVs to new suffixed files instead of overwriting existing files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="hair,lactose,eye",
        help="comma-separated datasets: hair,lactose,eye",
    )
    parser.add_argument(
        "--experiment",
        choices=experiment_choices,
        default="all",
        help="evaluation target when mode is evaluate/all",
    )
    parser.add_argument(
        "--generation-target",
        choices=["all", "proposed", "proposed_dp_maf", "proposed_100"],
        default="all",
        help="generation target when mode is generate/all",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ctx = Context(root=root, workers=args.workers, no_overwrite_results=args.no_overwrite_results, dry_run=args.dry_run)
    if not ctx.dry_run:
        ensure_dir(ctx.results_dir)
        ensure_dir(root / "figures")

    try:
        selected = [DATASET[s.strip()] for s in args.datasets.split(",") if s.strip()]
    except KeyError as exc:
        print(f"[error] Unknown dataset in --datasets: {exc}")
        return 2
    if not selected:
        print("[error] --datasets produced an empty selection")
        return 2

    print(f"[info] Selected datasets: {[d.value for d in selected]}")

    if args.copies <= 0:
        print("[error] --copies must be >= 1")
        return 2

    code = validate_inputs(
        ctx,
        selected,
        include_large_mia=args.include_large_mia,
        only_100_snp=args.only_100_snp,
        copies=args.copies,
    )
    if code != 0:
        return code

    if args.mode in {"generate", "all"}:
        gen = args.generation_target
        if args.only_100_snp:
            if gen in {"all", "proposed_100"}:
                maybe_generate_100_snp_methods(ctx, selected, validate_only=False, copies=args.copies)
        else:
            if gen in {"all", "proposed"}:
                maybe_generate_proposed(ctx, selected, include_large_mia=args.include_large_mia, validate_only=False, copies=args.copies)
            if gen in {"all", "proposed_dp_maf"}:
                maybe_generate_proposed_dp_maf(ctx, selected, validate_only=False, copies=args.copies)
            if gen in {"all", "proposed_100"}:
                maybe_generate_100_snp_methods(ctx, selected, validate_only=False, copies=args.copies)

    if args.mode in {"evaluate", "all"}:
        exp = args.experiment
        if args.only_100_snp and exp not in {"all", "utility_100"}:
            print("[error] --only-100-snp is only compatible with --experiment all|utility_100")
            return 2

        if args.only_100_snp:
            evaluate_utility_100(ctx, selected, copies=args.copies)
        else:
            if exp in {"all", "gwas_standard"}:
                evaluate_gwas(ctx, selected, copies=args.copies)
            if exp in {"all", "gwas_maf"}:
                evaluate_gwas_maf(ctx, selected, copies=args.copies)
            if exp in {"all", "mia_standard"}:
                evaluate_mia(ctx, selected, STANDARD_EFFECTIVE_EPS, "mia_experiments_results_full.csv", copies=args.copies)
            if exp in {"all", "mia_large"}:
                if args.include_large_mia:
                    evaluate_mia(
                        ctx,
                        selected,
                        LARGE_SCALE_EFFECTIVE_EPS,
                        "mia_experiments_results_large_scale.csv",
                        copies=args.copies,
                    )
                else:
                    print("[warn] Skipping mia_large because --include-large-mia was not provided")
            if exp in {"all", "utility_standard"}:
                evaluate_utility(ctx, selected, copies=args.copies)
            if exp in {"all", "utility_100"}:
                evaluate_utility_100(ctx, selected, copies=args.copies)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
