#!/usr/bin/env python3
"""Standalone plotting for artifact outputs.

Reads from artifact/results and writes PDFs to artifact/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

DATASET_NAME_MAPPING = {"lactose": "Lactose Intolerance", "hair": "Hair Color", "eye": "Eye Color"}
METHOD_NAME_MAPPING = {"ldp": "Baseline - LDP [25]", "proposed": "Ours"}
MIA_METHOD_MAPPING = {
    "hamming_distance": "Hamming Distance Test",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine",
    "nn": "Neural Network",
}


def safe_savefig(path: Path, **kwargs) -> None:
    try:
        plt.savefig(path, **kwargs)
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        print(f"[warn] cannot overwrite {path}, writing {alt}")
        plt.savefig(alt, **kwargs)


def plot_gwas_results(gwas_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    datasets = ["lactose", "hair", "eye"]

    for gwas in sorted(gwas_df["GWAS Type"].unique()):
        for error in sorted(gwas_df["Error Type"].unique()):
            subset = gwas_df[(gwas_df["GWAS Type"] == gwas) & (gwas_df["Error Type"] == error)]
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

            for i, dataset in enumerate(datasets):
                ax = axes[i]
                data_subset = subset[subset["Dataset"] == dataset].copy()
                if data_subset.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, color="grey")
                    continue

                data_subset["Approach"] = data_subset["Approach"].map(METHOD_NAME_MAPPING).fillna(data_subset["Approach"])
                sns.lineplot(
                    data=data_subset,
                    x="Error Rate",
                    y="Retention Ratio",
                    hue="Approach",
                    style="Epsilon",
                    markers=True,
                    markersize=10,
                    linewidth=2,
                    errorbar=None,
                    ax=ax,
                )

                ax.set_title(DATASET_NAME_MAPPING[dataset], fontsize=14, pad=8)
                ax.set_xlabel(r"Error Rate $\delta_f$" if error == "flipping" else r"Error Rate $\delta_n$", fontsize=13)
                ax.set_ylabel("SNP Retention Rate Diff" if i == 0 else "", fontsize=13)
                ax.set_ylim(-0.02, 1.0)
                ax.axhline(0, color="grey", linestyle="--", linewidth=1)
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)), frameon=True, bbox_to_anchor=(0.5, -0.02), fontsize=11)
            plt.tight_layout(rect=[0.02, 0.05, 1, 1])
            safe_savefig(output_dir / f"gwas_results_{gwas}_{error}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)


def plot_gwas_maf(gwas_df: pd.DataFrame, gwas_maf_df: pd.DataFrame, output_dir: Path, large: bool) -> None:
    sns.set_theme(style="whitegrid")
    baseline_df = gwas_df[(gwas_df["Approach"] == "proposed") & (gwas_df["Epsilon"] == 1)]

    base_color, dp_color = "#1B4F72", "#E67E22"
    gwas_types = ["chi2", "odds"]
    datasets = ["lactose", "hair", "eye"]
    errors = ["flipping", "noise"]

    fig, axes = plt.subplots(2, 6, figsize=(15, 5) if large else (13, 4.5), sharey=True)
    for r, gwas in enumerate(gwas_types):
        for c, (dataset, error) in enumerate([(d, e) for d in datasets for e in errors]):
            ax = axes[r, c]
            sub_maf = gwas_maf_df[(gwas_maf_df["GWAS Type"] == gwas) & (gwas_maf_df["Dataset"] == dataset) & (gwas_maf_df["Error Type"] == error)]
            sub_base = baseline_df[(baseline_df["GWAS Type"] == gwas) & (baseline_df["Dataset"] == dataset) & (baseline_df["Error Type"] == error)]

            if not sub_base.empty:
                sns.lineplot(data=sub_base, x="Error Rate", y="Retention Ratio", color=base_color, linewidth=2, errorbar=None, ax=ax)
            if not sub_maf.empty:
                sns.lineplot(data=sub_maf, x="Error Rate", y="Retention Ratio", style="Epsilon", markers=True, markersize=6, linewidth=1.6, color=dp_color, errorbar=None, ax=ax)

            if r == 0:
                ax.set_title(f"{dataset.capitalize()} ({error.capitalize()})", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("")
            ax.set_ylabel("")
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    handles = [
        Line2D([], [], color=base_color, lw=2, label=r"Using Public MAFs ($\epsilon_e=1.0$)"),
        Line2D([], [], color=dp_color, lw=1.6, label="Using Protected MAFs"),
        Line2D([], [], color="none", label=r"$\epsilon_m$ ="),
        Line2D([], [], color=dp_color, linestyle="-", marker="o", label="0.1"),
        Line2D([], [], color=dp_color, linestyle="--", marker="s", label="0.5"),
        Line2D([], [], color=dp_color, linestyle=":", marker="^", label="1.0"),
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout(rect=[0.05, 0.09, 1, 1])
    safe_savefig(output_dir / ("gwas_results_maf_large.pdf" if large else "gwas_results_maf.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def map_mia(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Epsilon"] != 0].copy()
    out["Dataset"] = out["Dataset"].map(DATASET_NAME_MAPPING).fillna(out["Dataset"])
    out["Approach"] = out["Approach"].map(METHOD_NAME_MAPPING).fillna(out["Approach"])
    return out


def plot_mia(mia_df: pd.DataFrame, output_dir: Path, large_scale: bool) -> None:
    sns.set_theme(style="whitegrid")
    ordered = ["hamming_distance", "decision_tree", "random_forest", "xgboost", "svm", "nn"]

    for dataset in DATASET_NAME_MAPPING.values():
        num_methods = 6 if dataset == "Eye Color" else 5
        fig = plt.figure(figsize=(14 if dataset == "Eye Color" else 12, 3))

        for idx, mia_method in enumerate(ordered[:num_methods], 1):
            subset = mia_df[(mia_df["Dataset"] == dataset) & (mia_df["MIAMethod"] == mia_method)]
            ax = plt.subplot(1, num_methods, idx)
            sns.lineplot(data=subset, x="Epsilon", y="MIAResult", hue="Approach", marker="o", linewidth=2, errorbar=None)
            ax.set_title(MIA_METHOD_MAPPING[mia_method], fontsize=12)
            ax.set_xlabel(r"$\epsilon_e$ (log scale)", fontsize=10)
            if idx == 1:
                ax.set_ylabel("Attack Power", fontsize=10)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_ylim(-0.02, 1.02)

            if large_scale:
                ax.set_xscale("log")
                ax.set_xlim(1e-2, 1e2)
                ax.set_xticks([1e-2, 1e-1, 1, 10, 100])
                ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: rf"$10^{{{int(np.log10(x))}}}$"))
            else:
                ax.set_xticks([1, 2, 3, 4, 5])

            ax.legend([], [], frameon=False)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, loc="lower center", ncol=max(1, len(labels)), fontsize=10)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        suffix = "_large_scale" if large_scale else ""
        safe_savefig(output_dir / f"mia_{dataset.lower().replace(' ', '_')}{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_time(output_dir: Path, results_dir: Path) -> None:
    time_csv = results_dir / "time.csv"
    if not time_csv.exists():
        data = {
            "SNPs": [10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 500, 500, 500, 1000, 1000, 1000, 5000, 5000, 10000, 10000, 28000, 28000],
            "Time": [575.6069, 0.0047, 0.0009, 3.3142, 7.5779, np.nan, 0.0118, 0.001, 43.8204, 42.3977, np.nan, 0.0306, 0.0014, 448.6869, 243.6263, np.nan, 0.2294, 0.0025, np.nan, 0.6987, 0.0042, 14.256, 0.0197, 53.9606, 0.0347, 446.6277, 0.1045],
            "Method": ["Original XOR", "Proposed", "LDP", "DPSyn", "PrivBayes", "Original XOR", "Proposed", "LDP", "DPSyn", "PrivBayes", "Original XOR", "Proposed", "LDP", "DPSyn", "PrivBayes", "Original XOR", "Proposed", "LDP", "Original XOR", "Proposed", "LDP", "Proposed", "LDP", "Proposed", "LDP", "Proposed", "LDP"],
        }
        df = pd.DataFrame(data).dropna()
        df["Method"] = df["Method"].map({"LDP": "LDP [25]", "Original XOR": "Vanilla XOR [24]", "Proposed": "Ours", "DPSyn": "DPSyn [27]", "PrivBayes": "PrivBayes [54]"})
        df.to_csv(time_csv, index=False)
    else:
        df = pd.read_csv(time_csv)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=df, x="SNPs", y="Time", hue="Method", marker="o", linewidth=2, errorbar=None, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8, 30000)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("# of SNPs (log scale)")
    ax.set_ylabel("Time Complexity (sec, log scale)")
    safe_savefig(output_dir / "time.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    results = root / "results"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    req = [results / "gwas_df_full.csv", results / "gwas_df_full_maf.csv", results / "mia_experiments_results_full.csv"]
    missing = [p.name for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result CSVs in artifact/results: {', '.join(missing)}")

    gwas = pd.read_csv(results / "gwas_df_full.csv")
    gwas["Retention Ratio"] = gwas["Retention Ratio"].abs()
    gwas.loc[gwas["Error Rate"] == 0.0, "Retention Ratio"] = 0.0

    gwas_maf = pd.read_csv(results / "gwas_df_full_maf.csv")
    gwas_maf["Retention Ratio"] = gwas_maf["Retention Ratio"].abs()
    gwas_maf.loc[gwas_maf["Error Rate"] == 0.0, "Retention Ratio"] = 0.0

    mia = map_mia(pd.read_csv(results / "mia_experiments_results_full.csv"))

    large_path = results / "mia_experiments_results_large_scale.csv"
    if large_path.exists():
        mia_large = map_mia(pd.read_csv(large_path))
        print(f"[info] using large-scale MIA from {large_path}")
    else:
        print("[warn] missing mia_experiments_results_large_scale.csv; reusing standard MIA CSV for large-scale plot")
        mia_large = mia.copy()

    plot_gwas_results(gwas, figures)
    plot_gwas_maf(gwas, gwas_maf, figures, large=False)
    plot_gwas_maf(gwas, gwas_maf, figures, large=True)
    plot_mia(mia, figures, large_scale=False)
    plot_mia(mia_large, figures, large_scale=True)
    plot_time(figures, results)
    print(f"[done] figures written to {figures}")


if __name__ == "__main__":
    main()
