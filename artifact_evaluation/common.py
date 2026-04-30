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
UTILITY_METHOD_NAME_MAPPING = {
    "proposed": "Ours",
    "ldp": "LDP [25]",
    "privbayes": "PrivBayes [54]",
    "dpsyn": "DPSyn [27]",
}
UTILITY_METHOD_ORDER = ["proposed", "ldp", "privbayes", "dpsyn"]
UTILITY_METRIC_NAME_MAPPING = {
    "point_error": "Point Error",
    "calc_sample_distance": "Sample Distance",
    "mean_error": "Mean Error",
    "variance_error": "Variance Error",
}
UTILITY_METRIC_ORDER = ["point_error", "calc_sample_distance", "mean_error", "variance_error"]
MIA_METHOD_MAPPING = {
    "hamming_distance": "Hamming Distance Test",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine",
    "nn": "Neural Network",
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


def _read_required_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{label} at {path} is empty. This usually means the corresponding evaluation ran "
            "before all required generated datasets were available. Regenerate the missing "
            "datasets, rerun the evaluation, and then rerun plotting."
        ) from exc


def load_gwas_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gwas_path = results_dir / "gwas_df_full.csv"
    gwas_maf_path = results_dir / "gwas_df_full_maf.csv"
    missing = [p.name for p in [gwas_path, gwas_maf_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required GWAS result CSVs in {results_dir}: {', '.join(missing)}")

    gwas_df = _read_required_csv(gwas_path, "GWAS standard result CSV")
    gwas_df["Retention Ratio"] = gwas_df["Retention Ratio"].abs()
    gwas_df.loc[gwas_df["Error Rate"] == 0.0, "Retention Ratio"] = 0.0

    gwas_maf_df = _read_required_csv(gwas_maf_path, "GWAS MAF result CSV")
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

    standard_df = map_mia(_read_required_csv(standard_path, "MIA standard result CSV"))
    large_path = results_dir / "mia_experiments_results_large_scale.csv"
    if large_path.exists():
        large_df = map_mia(_read_required_csv(large_path, "MIA large-scale result CSV"))
        print(f"[info] using large-scale MIA from {large_path}")
    else:
        print("[warn] missing mia_experiments_results_large_scale.csv; reusing standard MIA CSV for large-scale plot")
        large_df = standard_df.copy()
    return standard_df, large_df


def plot_gwas_results(gwas_df: pd.DataFrame, output_dir: Path, dry_run: bool = False) -> list[Path]:
    sns.set_theme(style="whitegrid")
    datasets = [d for d in ["lactose", "hair", "eye"] if d in set(gwas_df["Dataset"].dropna())]
    written: list[Path] = []

    for gwas in sorted(gwas_df["GWAS Type"].unique()):
        for error in sorted(gwas_df["Error Type"].unique()):
            subset = gwas_df[(gwas_df["GWAS Type"] == gwas) & (gwas_df["Error Type"] == error)]
            fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.5), sharey=True)
            axes = np.atleast_1d(axes)

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
    dry_run: bool = False,
) -> Path:
    sns.set_theme(style="whitegrid")
    baseline_df = gwas_df[(gwas_df["Approach"] == "proposed") & (gwas_df["Epsilon"] == 1)]

    base_color, dp_color = "#1B4F72", "#E67E22"
    gwas_types = ["chi2", "odds"]
    available = set(gwas_maf_df["Dataset"].dropna()).union(set(baseline_df["Dataset"].dropna()))
    datasets = [d for d in ["lactose", "hair", "eye"] if d in available]
    errors = ["flipping", "noise"]

    fig, axes = plt.subplots(2, len(datasets) * len(errors), figsize=(2.2 * len(datasets) * len(errors), 4.5), sharey=True)
    axes = np.atleast_2d(axes)
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
    out = safe_savefig(output_dir / "gwas_results_maf.pdf", dry_run=dry_run, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_mia(mia_df: pd.DataFrame, output_dir: Path, large_scale: bool, dry_run: bool = False) -> list[Path]:
    sns.set_theme(style="whitegrid")
    ordered = ["hamming_distance", "decision_tree", "random_forest", "xgboost", "svm", "nn"]
    written: list[Path] = []

    available = set(mia_df["Dataset"].dropna())
    for dataset in [d for d in DATASET_NAME_MAPPING.values() if d in available]:
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


def summarize_utility_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]].copy()
    summary = (
        df.groupby(["Dataset", "Utility Metric", "Approach", "Epsilon"], as_index=False)["Utility"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "Std", "count": "Count"})
    )
    summary["Std"] = summary["Std"].fillna(0.0)
    return summary


def _format_utility_cell(mean: float, std: float, count: int) -> str:
    if count > 1:
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"{mean:.4f}"


def _build_utility_latex(summary: pd.DataFrame, label: str) -> str:
    lines = [
        "% Auto-generated by artifact_evaluation utility plotting.",
        "% Requires \\usepackage{booktabs}.",
        "",
    ]
    eps_values = sorted(summary["Epsilon"].unique())
    datasets = list(dict.fromkeys(summary["Dataset"]))

    for dataset in datasets:
        dataset_name = DATASET_NAME_MAPPING.get(dataset, dataset)
        sub = summary[summary["Dataset"] == dataset].copy()
        header = "Utility Metric & Approach & " + " & ".join([rf"$\epsilon={eps:g}$" for eps in eps_values]) + r" \\"
        lines.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                rf"\caption{{{label} utility results for {dataset_name}. Lower is better.}}",
                rf"\label{{tab:{label.lower().replace(' ', '-')}-{dataset}}}",
                r"\begin{tabular}{ll" + ("c" * len(eps_values)) + r"}",
                r"\toprule",
                header,
                r"\midrule",
            ]
        )

        for metric in UTILITY_METRIC_ORDER:
            metric_rows = sub[sub["Utility Metric"] == metric]
            if metric_rows.empty:
                continue
            for method in UTILITY_METHOD_ORDER:
                row = metric_rows[metric_rows["Approach"] == method]
                if row.empty:
                    continue
                cells = [UTILITY_METRIC_NAME_MAPPING.get(metric, metric), UTILITY_METHOD_NAME_MAPPING.get(method, method)]
                for eps in eps_values:
                    eps_row = row[row["Epsilon"] == eps]
                    if eps_row.empty:
                        cells.append("--")
                        continue
                    item = eps_row.iloc[0]
                    cells.append(_format_utility_cell(float(item["Mean"]), float(item["Std"]), int(item["Count"])))
                lines.append(" & ".join(cells) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    return "\n".join(lines)


def _utility_table_dataframe(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    eps_values = sorted(summary["Epsilon"].unique())
    sub = summary[summary["Dataset"] == dataset].copy()
    rows: list[list[str]] = []

    for metric in UTILITY_METRIC_ORDER:
        metric_rows = sub[sub["Utility Metric"] == metric]
        if metric_rows.empty:
            continue
        for method in UTILITY_METHOD_ORDER:
            row = metric_rows[metric_rows["Approach"] == method]
            if row.empty:
                continue
            cells = [UTILITY_METRIC_NAME_MAPPING.get(metric, metric), UTILITY_METHOD_NAME_MAPPING.get(method, method)]
            for eps in eps_values:
                eps_row = row[row["Epsilon"] == eps]
                if eps_row.empty:
                    cells.append("--")
                    continue
                item = eps_row.iloc[0]
                cells.append(_format_utility_cell(float(item["Mean"]), float(item["Std"]), int(item["Count"])))
            rows.append(cells)

    columns = ["Utility Metric", "Approach", *[rf"$\epsilon={eps:g}$" for eps in eps_values]]
    return pd.DataFrame(rows, columns=columns)


def _draw_utility_table(ax, table_df: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10, loc="left")
    col_widths = [0.24, 0.18] + [0.12] * (len(table_df.columns) - 2)
    tab = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)

    header_color = "#16324F"
    stripe_light = "#F7F4EA"
    stripe_dark = "#ECE6D8"
    ours_color = "#FFF4D6"

    ncols = len(table_df.columns)
    nrows = len(table_df)

    for col in range(ncols):
        cell = tab[(0, col)]
        cell.set_facecolor(header_color)
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_edgecolor("white")

    for row_idx in range(1, nrows + 1):
        metric_name = table_df.iloc[row_idx - 1, 0]
        metric_pos = UTILITY_METRIC_ORDER.index(next(k for k, v in UTILITY_METRIC_NAME_MAPPING.items() if v == metric_name))
        base_color = stripe_light if metric_pos % 2 == 0 else stripe_dark
        is_ours = table_df.iloc[row_idx - 1, 1] == "Ours"

        for col in range(ncols):
            cell = tab[(row_idx, col)]
            cell.set_edgecolor("white")
            cell.set_facecolor(ours_color if is_ours and col >= 1 else base_color)
            if col in {0, 1}:
                cell.get_text().set_weight("bold")


def render_utility_table_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    label: str,
    dry_run: bool = False,
) -> Path:
    datasets = list(dict.fromkeys(summary["Dataset"]))
    fig_height = 3.8 * max(1, len(datasets))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11.5, fig_height))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        dataset_name = DATASET_NAME_MAPPING.get(dataset, dataset)
        table_df = _utility_table_dataframe(summary, dataset)
        _draw_utility_table(ax, table_df, f"{label} Utility: {dataset_name}")

    fig.suptitle(f"{label} Utility Summary", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    out = safe_savefig(output_dir / f"{stem}_table.pdf", dry_run=dry_run, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def render_utility_tables(
    results_dir: Path,
    output_dir: Path,
    *,
    filename: str,
    stem: str,
    label: str,
    dry_run: bool = False,
) -> list[Path]:
    csv_path = results_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required utility result CSV in {results_dir}: {csv_path.name}")

    df = _read_required_csv(csv_path, f"{label} utility result CSV")
    summary = summarize_utility_results(df)
    summary_out = output_dir / f"{stem}_summary.csv"
    tex_out = output_dir / f"{stem}_table.tex"
    fig_out = output_dir / f"{stem}_table.pdf"

    if dry_run:
        print(f"[dry-run] would write {summary_out}")
        print(f"[dry-run] would write {tex_out}")
        print(f"[dry-run] would write {fig_out}")
        return [summary_out, tex_out, fig_out]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    tex_out.write_text(_build_utility_latex(summary, label), encoding="utf-8")
    written_fig = render_utility_table_figure(summary, output_dir, stem=stem, label=label, dry_run=dry_run)
    print(f"[done] {summary_out}")
    print(f"[done] {tex_out}")
    print(f"[done] {written_fig}")
    return [summary_out, tex_out, written_fig]


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
    "load_gwas_results",
    "load_mia_results",
    "parse_datasets",
    "plot_gwas_maf",
    "plot_gwas_results",
    "plot_mia",
    "render_utility_tables",
    "resolve_plots_dir",
    "resolve_results_dir",
    "safe_savefig",
    "summarize_utility_results",
    "validate_evaluation",
]
