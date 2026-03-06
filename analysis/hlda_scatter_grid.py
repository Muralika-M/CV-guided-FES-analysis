#!/usr/bin/env python3
"""
Generate scatter plots for all threshold pairs:
  dHLDA vs dTm
  dHLDA vs abs(dTm)
  abs(dHLDA) vs dTm
  abs(dHLDA) vs abs(dTm)
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
import seaborn as sns

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data_examples"
RESULTS_DIR = PACKAGE_ROOT / "figures/generated"


SITE_COLORS_BY_LABEL = {
    "D2": "orange",
    "Y9": "green",
    "T7": "blue",
    "E4": "black",
    "T5": "pink",
    "Y0": "purple",
    "P3": "cyan",
}


def site_label(mutant):
    m = re.match(r"([A-Z])(\d+)", str(mutant))
    return f"{m.group(1)}{m.group(2)}" if m else None


def stats_text(x, y):
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return (
        f"Pearson r = {pr:.2f}\n"
        f"p = {pp:.2e}\n"
        f"Spearman ρ = {sr:.2f}\n"
        f"p = {sp:.2e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ev_csv",
        default=str(DATA_DIR / "ev/per_mutant_hlda_EV_indexed.csv"),
    )
    ap.add_argument("--tm_csv", default=str(DATA_DIR / "Tm_table.csv"))
    ap.add_argument("--out_dir", default=str(RESULTS_DIR / "ev_scatter_grid"))
    ap.add_argument("--no_title", action="store_true")
    ap.add_argument("--no_legend", action="store_true")
    ap.add_argument("--point_size", type=float, default=55.0)
    ap.add_argument("--point_alpha", type=float, default=0.9)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--annotate_size", type=float, default=7.0)
    ap.add_argument("--label_size", type=float, default=12.0)
    ap.add_argument("--tick_size", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev_df = pd.read_csv(args.ev_csv)
    tm_df = pd.read_csv(args.tm_csv)
    ev_df["Mutant"] = ev_df["Mutant"].astype(str)
    tm_df["Mutant"] = tm_df["Mutant"].astype(str)

    df = ev_df.merge(tm_df, on="Mutant", how="inner")
    if "WT" not in df["Mutant"].values:
        raise SystemExit("WT not found in merged data.")

    wt_ev_map = df[df["Mutant"] == "WT"].set_index(["Thr_F", "Thr_UF"])["EV"].to_dict()
    wt_tm = float(df.loc[df["Mutant"] == "WT", "Tm"].iloc[0])

    df["dHLDA"] = df.apply(
        lambda r: r["EV"] - wt_ev_map.get((r["Thr_F"], r["Thr_UF"]), np.nan),
        axis=1,
    )
    df["abs_dHLDA"] = df["dHLDA"].abs()
    df["dTm"] = df["Tm"] - wt_tm
    df["abs_dTm"] = df["dTm"].abs()

    thrF_vals = sorted(df["Thr_F"].dropna().unique())
    thrU_vals = sorted(df["Thr_UF"].dropna().unique())

    combos = [
        ("dHLDA", "dTm", r"$\Delta$HLDA Eigenvalue", r"$\Delta T_m$", "dHLDA_dTm"),
        ("dHLDA", "abs_dTm", r"$\Delta$HLDA Eigenvalue", r"|$\Delta T_m$|", "dHLDA_absdTm"),
        ("abs_dHLDA", "dTm", r"|$\Delta$HLDA|", r"$\Delta T_m$", "absdHLDA_dTm"),
        ("abs_dHLDA", "abs_dTm", r"|$\Delta$HLDA|", r"|$\Delta T_m$|", "absdHLDA_absdTm"),
    ]

    sns.set_style("whitegrid")

    for thrF in thrF_vals:
        for thrU in thrU_vals:
            grp = df[(df["Thr_F"] == thrF) & (df["Thr_UF"] == thrU)].copy()
            grp = grp[grp["Mutant"] != "WT"]
            grp = grp.dropna(subset=["dHLDA", "dTm"])
            if len(grp) < 3:
                continue

            grp["Site"] = grp["Mutant"].apply(site_label)
            colors = [SITE_COLORS_BY_LABEL.get(s, "gray") for s in grp["Site"].tolist()]

            for x_col, y_col, xlab, ylab, tag in combos:
                x = grp[x_col].to_numpy(dtype=float)
                y = grp[y_col].to_numpy(dtype=float)
                if len(x) < 3:
                    continue

                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.regplot(
                    x=x, y=y, ax=ax,
                    scatter_kws={"s": args.point_size, "edgecolor": "k", "alpha": args.point_alpha},
                    line_kws={"color": "black", "linestyle": "--", "linewidth": 1.2},
                    ci=None,
                )
                scatter = ax.collections[0]
                scatter.set_facecolor(colors)
                scatter.set_edgecolor("k")

                if args.annotate:
                    for i, m in enumerate(grp["Mutant"].tolist()):
                        ax.text(
                            x[i], y[i], str(m),
                            fontsize=args.annotate_size,
                            fontweight="bold",
                            ha="center",
                            va="bottom",
                        )

                ax.text(
                    0.05, 0.95, stats_text(x, y),
                    transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

                ax.set_xlabel(xlab, fontsize=args.label_size, fontweight="bold")
                ax.set_ylabel(ylab, fontsize=args.label_size, fontweight="bold")
                ax.tick_params(axis="both", labelsize=args.tick_size)

                if not args.no_title:
                    ax.set_title(f"thrF {thrF:.2f}  thrUF {thrU:.2f}")

                if not args.no_legend:
                    for site, color in SITE_COLORS_BY_LABEL.items():
                        ax.scatter([], [], color=color, label=site, edgecolor="k", s=40)
                    ax.legend(title="Site", fontsize=7, title_fontsize=8)

                plt.tight_layout()
                out_png = out_dir / f"scatter_{tag}_thrF{thrF:.2f}_thrUF{thrU:.2f}.png"
                plt.savefig(out_png, dpi=300)
                plt.close(fig)


if __name__ == "__main__":
    main()
