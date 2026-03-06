#!/usr/bin/env python3
"""
Scatter plots of WT residue importance vs avg dTm per site across thresholds.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data_examples"
RESULTS_DIR = PACKAGE_ROOT / "figures/generated"


def site_from_mutant(mutant):
    m = str(mutant)
    if m == "WT":
        return None
    digits = re.findall(r"\d+", m)
    if not digits:
        return None
    return int(digits[0])


def avg_dtm_per_site(tm_csv):
    tm = pd.read_csv(tm_csv)
    tm["Mutant"] = tm["Mutant"].astype(str)
    if "WT" not in tm["Mutant"].values:
        raise ValueError("WT not found in Tm_table.csv")

    wt_tm = float(tm.loc[tm["Mutant"] == "WT", "Tm"].iloc[0])
    tm = tm[tm["Mutant"] != "WT"].copy()
    tm["site"] = tm["Mutant"].apply(site_from_mutant)
    tm = tm.dropna(subset=["site"])
    tm["site"] = tm["site"].astype(int)
    tm["dTm"] = tm["Tm"] - wt_tm

    grp = tm.groupby("site")
    avg_signed = grp["dTm"].mean()
    avg_abs = grp["dTm"].apply(lambda x: x.abs().mean())

    sites = sorted(avg_signed.index.tolist())
    return sites, avg_signed.loc[sites].to_numpy(), avg_abs.loc[sites].to_numpy()


DESC_COUNTS = {0: 7, 1: 6, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 6, 9: 7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default=str(DATA_DIR / "residue/wt_residue_importance.csv"),
    )
    ap.add_argument("--tm_csv", default=str(DATA_DIR / "Tm_table.csv"))
    ap.add_argument("--case", default=None)
    ap.add_argument("--out_dir", default=str(RESULTS_DIR / "residue_scatter"))
    ap.add_argument("--metric", choices=["signed", "abs", "both"], default="signed")
    ap.add_argument("--imp_mode", choices=["raw", "norm"], default="raw")
    ap.add_argument("--xlabel", default=None)
    ap.add_argument("--ylabel", default=None)
    ap.add_argument("--label_size", type=int, default=None)
    ap.add_argument("--point_size", type=float, default=None)
    ap.add_argument("--edge_width", type=float, default=None)
    ap.add_argument("--max_plots", type=int, default=None)
    ap.add_argument("--no_annotate", action="store_true")
    ap.add_argument("--fit_line", action="store_true", help="Add a linear fit line.")
    ap.add_argument(
        "--color_by_residue",
        action="store_true",
        help="Color points by residue with a fixed palette.",
    )
    ap.add_argument("--thrF", type=float, default=None, help="Filter to a specific thrF.")
    ap.add_argument("--thrU", type=float, default=None, help="Filter to a specific thrU.")
    ap.add_argument("--no_title", action="store_true", help="Omit plot title.")
    ap.add_argument(
        "--legend_stats",
        action="store_true",
        help="Show correlation stats in a legend instead of the title.",
    )
    ap.add_argument(
        "--no_legend",
        action="store_true",
        help="Do not show legend stats even if enabled.",
    )
    ap.add_argument(
        "--legend_pearson_only",
        action="store_true",
        help="If set with --legend_stats, only show Pearson stats in the legend.",
    )
    ap.add_argument(
        "--editorial",
        action="store_true",
        help="Apply a clean, publication-style aesthetic.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    if args.case and "case" in df.columns:
        df = df[df["case"] == args.case].copy()
        if df.empty:
            raise SystemExit(f"No rows for case '{args.case}' in {args.summary_csv}")

    sites, avg_signed, avg_abs = avg_dtm_per_site(args.tm_csv)

    if args.thrF is not None:
        df = df[df["thrF"] == args.thrF].copy()
    if args.thrU is not None:
        df = df[df["thrU"] == args.thrU].copy()
    if df.empty:
        raise SystemExit("No rows after thrF/thrU filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_rows = []
    plotted = 0

    for _, row in df.iterrows():
        thrF = float(row["thrF"])
        thrU = float(row["thrU"])
        case_label = row.get("case")
        if case_label is None:
            case_label = args.case

        x = np.array([row.get(f"res{r}") for r in sites], dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3:
            continue
        x = x[mask]
        sites_use = [sites[i] for i in range(len(sites)) if mask[i]]
        if args.imp_mode == "norm":
            counts = np.array([DESC_COUNTS.get(r, np.nan) for r in sites_use], dtype=float)
            if np.any(np.isnan(counts)) or np.any(counts == 0):
                raise SystemExit(f"Missing desc_counts for sites: {sites_use}")
            x = x / counts

        def do_plot(y, metric_label):
            nonlocal plotted
            y = y[mask]
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)

            corr_rows.append({
                "case": case_label,
                "thrF": thrF,
                "thrU": thrU,
                "imp_mode": args.imp_mode,
                "metric": metric_label,
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_r": r_s,
                "spearman_p": p_s,
            })

            if args.max_plots is not None and plotted >= args.max_plots:
                return

            if args.editorial:
                plt.rcParams.update({
                    "font.size": 9,
                    "axes.labelsize": 10,
                    "axes.titlesize": 10,
                })
            plt.figure(figsize=(5.2, 4.2) if args.editorial else (4.8, 4.2))
            fig = plt.gcf()
            ax = plt.gca()

            base_size = args.point_size if args.point_size is not None else (110 if args.editorial else 80)
            edge_width = args.edge_width
            if edge_width is None and args.editorial:
                edge_width = 0.9
            if edge_width is None:
                edge_width = 0.6
            if args.color_by_residue:
                color_map = {
                    2: "yellow",
                    4: "black",
                    3: "cyan",
                    9: "green",
                    0: "purple",
                    5: "pink",
                    7: "blue",
                }
                colors = [color_map.get(r, "gray") for r in sites_use]
                scatter_kwargs = {"s": base_size, "c": colors}
            else:
                scatter_kwargs = {"s": base_size}

            if args.editorial:
                scatter_kwargs.update({"alpha": 0.9, "edgecolors": "#111111", "linewidths": edge_width})
            elif args.edge_width is not None:
                scatter_kwargs.update({"edgecolors": "#111111", "linewidths": edge_width})
            plt.scatter(x, y, **scatter_kwargs)

            if args.fit_line and len(x) >= 2:
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_fit = np.array([x.min(), x.max()])
                    y_fit = slope * x_fit + intercept
                    plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.2)
                except Exception:
                    pass

            if not args.no_annotate:
                for i, r in enumerate(sites_use):
                    plt.text(x[i], y[i], f"{r}", fontsize=9, fontweight="bold",
                             ha="right", va="bottom")

            if args.xlabel:
                xlabel = args.xlabel
            else:
                xlabel = "WT residue importance (extended)"
                if args.imp_mode == "norm":
                    xlabel = "WT residue importance (normalized)"
            label_kwargs = {}
            if args.label_size:
                label_kwargs["fontsize"] = args.label_size
            if args.editorial:
                plt.xlabel(xlabel, fontweight="bold", **label_kwargs)
            else:
                plt.xlabel(xlabel, **label_kwargs)
            if args.ylabel:
                ylabel = args.ylabel
            else:
                ylabel = "avg ΔTm" if metric_label == "signed" else "avg |ΔTm|"
            if args.editorial:
                plt.ylabel(ylabel, fontweight="bold", **label_kwargs)
            else:
                plt.ylabel(ylabel, **label_kwargs)
            def fmt_p(pval):
                return f"{pval:.6f}"

            if not args.no_title:
                plt.title(
                    f"{case_label} | F={thrF:.2f}, U={thrU:.2f}\n"
                    f"Pearson r={r_p:.2f} (p={fmt_p(p_p)})  "
                    f"Spearman ρ={r_s:.2f} (p={fmt_p(p_s)})",
                    fontsize=9
                )
            if args.legend_stats and not args.no_legend:
                from matplotlib.lines import Line2D
                if args.legend_pearson_only:
                    stats_label = f"Pearson r={r_p:.2f} (p={fmt_p(p_p)})"
                else:
                    stats_label = (
                        f"Pearson r={r_p:.2f} (p={fmt_p(p_p)})\n"
                        f"Spearman rho={r_s:.2f} (p={fmt_p(p_s)})"
                    )
                stats_handle = Line2D([], [], linestyle="none", marker=None, label=stats_label)
                plt.legend(
                    handles=[stats_handle],
                    loc="best",
                    frameon=False,
                    handlelength=0,
                    handletextpad=0.0,
                    borderpad=0.2,
                    fontsize=9 if args.editorial else 8,
                )
            if args.editorial:
                for spine in ("top", "right", "left", "bottom"):
                    ax.spines[spine].set_visible(True)
                    ax.spines[spine].set_linewidth(1.1)
                ax.tick_params(axis="both", width=1.0, length=4.5, direction="out")
            plt.tight_layout()

            tag = f"F{thrF:.2f}_U{thrU:.2f}".replace(".", "p")
            out_png = out_dir / f"scatter_{case_label}_{args.imp_mode}_{metric_label}_{tag}.png"
            dpi = 300 if args.editorial else 200
            plt.savefig(out_png, dpi=dpi)
            plt.close()
            plotted += 1

        if args.metric in ("signed", "both"):
            do_plot(avg_signed, "signed")
        if args.metric in ("abs", "both"):
            do_plot(avg_abs, "abs")

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = out_dir / f"correlation_summary_{args.imp_mode}.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved {corr_csv}")


if __name__ == "__main__":
    main()
