"""Shared helpers for experiment-specific evaluation and plotting entrypoints."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_evaluation.run_experiments import (  # noqa: E402
    Context,
    DATASET,
    LARGE_SCALE_EFFECTIVE_EPS,
    STANDARD_EFFECTIVE_EPS,
    evaluate_gwas,
    evaluate_gwas_maf,
    evaluate_mia,
    evaluate_utility,
    evaluate_utility_100,
    validate_inputs,
)

DEFAULT_COPIES = 10
DEFAULT_DATASETS = "hair,lactose,eye"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PLOTS_DIR = "plots"

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
TIME_DATA = {
    "SNPs": [10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 500, 500, 500, 1000, 1000, 1000, 5000, 5000, 10000, 10000, 28000, 28000],
    "Time": [575.6069, 0.0047, 0.0009, 3.3142, 7.5779, np.nan, 0.0118, 0.001, 43.8204, 42.3977, np.nan, 0.0306, 0.0014, 448.6869, 243.6263, np.nan, 0.2294, 0.0025, np.nan, 0.6987, 0.0042, 14.256, 0.0197, 53.9606, 0.0347, 446.6277, 0.1045],
    "Method": [
        "Original XOR",
        "Proposed",
        "LDP",
        "DPSyn",
        "PrivBayes",
        "Original XOR",
        "Proposed",
        "LDP",
        "DPSyn",
        "PrivBayes",
        "Original XOR",
        "Proposed",
        "LDP",
        "DPSyn",
        "PrivBayes",
        "Original XOR",
        "Proposed",
        "LDP",
        "Original XOR",
        "Proposed",
        "LDP",
        "Proposed",
        "LDP",
        "Proposed",
        "LDP",
        "Proposed",
        "LDP",
    ],
}


def build_context(dry_run: bool = False, no_overwrite_results: bool = False, workers: int | None = None) -> Context:
    return Context(
        root=ROOT,
        workers=workers or max(1, mp.cpu_count() // 2),
        no_overwrite_results=no_overwrite_results,
        dry_run=dry_run,
    )


def parse_datasets(raw: str) -> list[DATASET]:
    return [DATASET[item.strip()] for item in raw.split(",") if item.strip()]


def validate_evaluation(
    ctx: Context,
    datasets: list[DATASET],
    include_large_mia: bool,
    only_100_snp: bool,
    copies: int,
) -> int:
    return validate_inputs(
        ctx,
        datasets,
        include_large_mia=include_large_mia,
        only_100_snp=only_100_snp,
        copies=copies,
    )


def report_evaluation_dry_run(ctx: Context, filename: str, datasets: list[DATASET], copies: int) -> None:
    names = ",".join(dataset.name for dataset in datasets)
    print(f"[dry-run] validated datasets={names}, copies={copies}, workers={ctx.workers}")
    print(f"[dry-run] would write {ctx.results_dir / filename}")


def resolve_results_dir(raw: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_plots_dir(raw: str | Path = DEFAULT_PLOTS_DIR, dry_run: bool = False) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)
    return path


def safe_savefig(path: Path, dry_run: bool = False, **kwargs) -> Path:
    if dry_run:
        print(f"[dry-run] would write figure {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(path, **kwargs)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        print(f"[warn] cannot overwrite {path}, writing {alt}")
        plt.savefig(alt, **kwargs)
        return alt


def load_gwas_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gwas_path = results_dir / "gwas_df_full.csv"
    gwas_maf_path = results_dir / "gwas_df_full_maf.csv"
    missing = [p.name for p in [gwas_path, gwas_maf_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required GWAS result CSVs in {results_dir}: {', '.join(missing)}")

    gwas_df = pd.read_csv(gwas_path)
    gwas_df["Retention Ratio"] = gwas_df["Retention Ratio"].abs()
    gwas_df.loc[gwas_df["Error Rate"] == 0.0, "Retention Ratio"] = 0.0

    gwas_maf_df = pd.read_csv(gwas_maf_path)
    gwas_maf_df["Retention Ratio"] = gwas_maf_df["Retention Ratio"].abs()
    gwas_maf_df.loc[gwas_maf_df["Error Rate"] == 0.0, "Retention Ratio"] = 0.0
    return gwas_df, gwas_maf_df


def map_mia(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Epsilon"] != 0].copy()
    out["Dataset"] = out["Dataset"].map(DATASET_NAME_MAPPING).fillna(out["Dataset"])
    out["Approach"] = out["Approach"].map(METHOD_NAME_MAPPING).fillna(out["Approach"])
    return out


def load_mia_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_path = results_dir / "mia_experiments_results_full.csv"
    if not standard_path.exists():
        raise FileNotFoundError(f"Missing required MIA result CSV in {results_dir}: {standard_path.name}")

    standard_df = map_mia(pd.read_csv(standard_path))
    large_path = results_dir / "mia_experiments_results_large_scale.csv"
    if large_path.exists():
        large_df = map_mia(pd.read_csv(large_path))
        print(f"[info] using large-scale MIA from {large_path}")
    else:
        print("[warn] missing mia_experiments_results_large_scale.csv; reusing standard MIA CSV for large-scale plot")
        large_df = standard_df.copy()
    return standard_df, large_df


def plot_gwas_results(gwas_df: pd.DataFrame, output_dir: Path, dry_run: bool = False) -> list[Path]:
    sns.set_theme(style="whitegrid")
    datasets = ["lactose", "hair", "eye"]
    written: list[Path] = []

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
                    estimator="mean",
                    errorbar=("ci", 95),
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
            written.append(safe_savefig(output_dir / f"gwas_results_{gwas}_{error}.pdf", dry_run=dry_run, bbox_inches="tight", dpi=300))
            plt.close(fig)
    return written


def plot_gwas_maf(
    gwas_df: pd.DataFrame,
    gwas_maf_df: pd.DataFrame,
    output_dir: Path,
    large: bool,
    dry_run: bool = False,
) -> Path:
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
            sub_maf = gwas_maf_df[
                (gwas_maf_df["GWAS Type"] == gwas)
                & (gwas_maf_df["Dataset"] == dataset)
                & (gwas_maf_df["Error Type"] == error)
            ]
            sub_base = baseline_df[
                (baseline_df["GWAS Type"] == gwas)
                & (baseline_df["Dataset"] == dataset)
                & (baseline_df["Error Type"] == error)
            ]

            if not sub_base.empty:
                sns.lineplot(
                    data=sub_base,
                    x="Error Rate",
                    y="Retention Ratio",
                    color=base_color,
                    linewidth=2,
                    estimator="mean",
                    errorbar=("ci", 95),
                    ax=ax,
                )
            if not sub_maf.empty:
                sns.lineplot(
                    data=sub_maf,
                    x="Error Rate",
                    y="Retention Ratio",
                    style="Epsilon",
                    markers=True,
                    markersize=6,
                    linewidth=1.6,
                    color=dp_color,
                    estimator="mean",
                    errorbar=("ci", 95),
                    ax=ax,
                )

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
    out = safe_savefig(
        output_dir / ("gwas_results_maf_large.pdf" if large else "gwas_results_maf.pdf"),
        dry_run=dry_run,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
    return out


def plot_mia(mia_df: pd.DataFrame, output_dir: Path, large_scale: bool, dry_run: bool = False) -> list[Path]:
    sns.set_theme(style="whitegrid")
    ordered = ["hamming_distance", "decision_tree", "random_forest", "xgboost", "svm", "nn"]
    written: list[Path] = []

    for dataset in DATASET_NAME_MAPPING.values():
        num_methods = 6 if dataset == "Eye Color" else 5
        fig = plt.figure(figsize=(14 if dataset == "Eye Color" else 12, 3))

        for idx, mia_method in enumerate(ordered[:num_methods], 1):
            subset = mia_df[(mia_df["Dataset"] == dataset) & (mia_df["MIAMethod"] == mia_method)]
            ax = plt.subplot(1, num_methods, idx)
            sns.lineplot(
                data=subset,
                x="Epsilon",
                y="MIAResult",
                hue="Approach",
                marker="o",
                linewidth=2,
                estimator="mean",
                errorbar=("ci", 95),
                ax=ax,
            )
            ax.set_title(MIA_METHOD_MAPPING[mia_method], fontsize=12)
            ax.set_xlabel(r"$\epsilon_e$ (log scale)" if large_scale else r"$\epsilon_e$", fontsize=10)
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
                ax.set_xticks(STANDARD_EFFECTIVE_EPS)

            ax.legend([], [], frameon=False)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, loc="lower center", ncol=max(1, len(labels)), fontsize=10)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        suffix = "_large_scale" if large_scale else ""
        written.append(
            safe_savefig(
                output_dir / f"mia_{dataset.lower().replace(' ', '_')}{suffix}.pdf",
                dry_run=dry_run,
                bbox_inches="tight",
            )
        )
        plt.close(fig)
    return written


def ensure_time_results(results_dir: Path, dry_run: bool = False) -> Path:
    out = results_dir / "time.csv"
    if out.exists():
        return out

    df = pd.DataFrame(TIME_DATA).dropna()
    df["Method"] = df["Method"].map(
        {
            "LDP": "LDP [25]",
            "Original XOR": "Vanilla XOR [24]",
            "Proposed": "Ours",
            "DPSyn": "DPSyn [27]",
            "PrivBayes": "PrivBayes [54]",
        }
    )
    if dry_run:
        print(f"[dry-run] would write {out}")
        return out

    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def plot_time(results_dir: Path, output_dir: Path, dry_run: bool = False) -> Path:
    time_csv = ensure_time_results(results_dir, dry_run=dry_run)
    if dry_run and not time_csv.exists():
        df = pd.DataFrame(TIME_DATA).dropna()
        df["Method"] = df["Method"].map(
            {
                "LDP": "LDP [25]",
                "Original XOR": "Vanilla XOR [24]",
                "Proposed": "Ours",
                "DPSyn": "DPSyn [27]",
                "PrivBayes": "PrivBayes [54]",
            }
        )
    else:
        df = pd.read_csv(time_csv)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=df, x="SNPs", y="Time", hue="Method", marker="o", linewidth=2, estimator="mean", errorbar=None, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8, 30000)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("# of SNPs (log scale)")
    ax.set_ylabel("Time Complexity (sec, log scale)")
    out = safe_savefig(output_dir / "time.pdf", dry_run=dry_run, bbox_inches="tight", dpi=400)
    plt.close(fig)
    return out


def print_utility_plot_hint(label: str, results_dir: Path, filename: str) -> None:
    csv_path = results_dir / filename
    print(f"[info] No dedicated utility figure is included for {label}.")
    print(f"[info] Review the terminal summary from the experiment step and the CSV at {csv_path.resolve()}")


__all__ = [
    "Context",
    "DATASET",
    "DEFAULT_COPIES",
    "DEFAULT_DATASETS",
    "DEFAULT_PLOTS_DIR",
    "DEFAULT_RESULTS_DIR",
    "LARGE_SCALE_EFFECTIVE_EPS",
    "ROOT",
    "STANDARD_EFFECTIVE_EPS",
    "build_context",
    "evaluate_gwas",
    "evaluate_gwas_maf",
    "evaluate_mia",
    "evaluate_utility",
    "evaluate_utility_100",
    "ensure_time_results",
    "load_gwas_results",
    "load_mia_results",
    "parse_datasets",
    "plot_gwas_maf",
    "plot_gwas_results",
    "plot_mia",
    "plot_time",
    "print_utility_plot_hint",
    "resolve_plots_dir",
    "resolve_results_dir",
    "safe_savefig",
    "validate_evaluation",
]
