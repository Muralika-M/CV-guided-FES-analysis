#!/usr/bin/env python3
"""
Generate 4 heatmaps and save them under pear_combined/ by default.

Input:
    --ev_csv : CSV with columns [Mutant,Thr_F,Thr_UF,EV]
    --tm_csv : CSV with columns [Mutant,Tm]
Output:
    4 PNG heatmaps
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data_examples"
RESULTS_DIR = PACKAGE_ROOT / "figures/generated"

ap = argparse.ArgumentParser(description="Plot dHLDA/dTm heatmaps")
ap.add_argument("--ev_csv", default=str(DATA_DIR / "ev/per_mutant_hlda_EV_indexed_pearson_intersection.csv"),
                help="CSV containing: Mutant,Thr_F,Thr_UF,EV")
ap.add_argument("--tm_csv", default=str(DATA_DIR / "Tm_table.csv"),
                help="CSV containing: Mutant,Tm")
ap.add_argument("--out_dir", default=str(RESULTS_DIR / "pear_combined"),
                help="Output directory for PNGs")
ap.add_argument("--out_prefix", default="pearson_hlda_tm",
                help="Prefix for heatmap PNGs")
ap.add_argument("--case_label", default="",
                help="Optional label to prepend to plot titles")
ap.add_argument("--thrUF_max", type=float, default=None,
                help="Optional max thrUF to include in plots")
ap.add_argument("--abs_corr", action="store_true",
                help="Plot absolute correlation values instead of signed r")
ap.add_argument("--no_title", action="store_true",
                help="Omit plot titles")
ap.add_argument("--x_label", default="Unfolded threshold",
                help="Label for x-axis")
ap.add_argument("--y_label", default="Folded threshold",
                help="Label for y-axis")
ap.add_argument("--transpose_axes", action="store_true",
                help="Plot Folded threshold on x-axis and Unfolded threshold on y-axis")
ap.add_argument("--publish_style", action="store_true",
                help="Use larger, bolder styling for publication-quality figures")
args = ap.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

prefix = out_dir / args.out_prefix

ev_df = pd.read_csv(args.ev_csv)
tm_df = pd.read_csv(args.tm_csv)

ev_df["Mutant"] = ev_df["Mutant"].astype(str)
tm_df["Mutant"] = tm_df["Mutant"].astype(str)

df = ev_df.merge(tm_df, on="Mutant", how="inner")
expected_mutants = set(tm_df["Mutant"])

if "WT" not in df["Mutant"].values:
    raise ValueError("WT must be present in Mutant column.")

wt_ev_map = df[df["Mutant"] == "WT"].set_index(["Thr_F", "Thr_UF"])["EV"].to_dict()
wt_tm = float(df.loc[df["Mutant"] == "WT", "Tm"].iloc[0])

def delta_ev(row):
    key = (row["Thr_F"], row["Thr_UF"])
    return row["EV"] - wt_ev_map.get(key, np.nan)

df["dHLDA"] = df.apply(delta_ev, axis=1)
df["abs_dHLDA"] = df["dHLDA"].abs()
df["dTm"] = df["Tm"] - wt_tm
df["abs_dTm"] = df["dTm"].abs()

stats = []

for (thrF, thrUF), grp in df.groupby(["Thr_F", "Thr_UF"]):
    present_mutants = set(grp["Mutant"])
    if present_mutants != expected_mutants:
        continue
    grp = grp[grp["Mutant"] != "WT"]
    if len(grp) < 3:
        continue

    try:
        r1, _ = pearsonr(grp["dHLDA"], grp["dTm"])
        r2, _ = pearsonr(grp["dHLDA"], grp["abs_dTm"])
        r3, _ = pearsonr(grp["abs_dHLDA"], grp["dTm"])
        r4, _ = pearsonr(grp["abs_dHLDA"], grp["abs_dTm"])
    except Exception:
        r1 = r2 = r3 = r4 = np.nan

    stats.append({
        "Thr_F": thrF,
        "Thr_UF": thrUF,
        "r_dHLDA_dTm": r1,
        "r_dHLDA_absdTm": r2,
        "r_absdHLDA_dTm": r3,
        "r_absdHLDA_absdTm": r4,
    })

stats_df = pd.DataFrame(stats)
if args.thrUF_max is not None:
    stats_df = stats_df[stats_df["Thr_UF"] <= args.thrUF_max].copy()

sns.set_context("talk")

if args.publish_style:
    annot_size = 10
    axis_label_size = 26
    tick_label_size = 16
    title_size = 22
    cbar_label_size = 22
    cbar_tick_size = 15
    save_dpi = 450
    cell_w = 0.62
    cell_h = 0.50
    min_w, max_w = 10.0, 18.0
    min_h, max_h = 8.0, 14.0
else:
    annot_size = 7
    axis_label_size = 14
    tick_label_size = 10
    title_size = 14
    cbar_label_size = 14
    cbar_tick_size = 10
    save_dpi = 300
    cell_w = 0.48
    cell_h = 0.40
    min_w, max_w = 8.0, 14.0
    min_h, max_h = 6.0, 11.0

heatmaps = {
    "dHLDA vs dTm": "r_dHLDA_dTm",
    "dHLDA vs abs(dTm)": "r_dHLDA_absdTm",
    "abs(dHLDA) vs dTm": "r_absdHLDA_dTm",
    "abs(dHLDA) vs abs(dTm)": "r_absdHLDA_absdTm",
}

for title, col in heatmaps.items():
    if args.transpose_axes:
        pivot = stats_df.pivot(index="Thr_UF", columns="Thr_F", values=col)
    else:
        pivot = stats_df.pivot(index="Thr_F", columns="Thr_UF", values=col)
    plot_data = pivot.abs() if args.abs_corr else pivot
    n_rows, n_cols = plot_data.shape
    fig_w = min(max_w, max(min_w, n_cols * cell_w + 3.0))
    fig_h = min(max_h, max(min_h, n_rows * cell_h + 1.6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cbar_label = "Pearson r"
    cmap = "coolwarm"
    center = 0
    vmin = vmax = None
    if args.abs_corr:
        cbar_label = "abs(Pearson r)"
        center = None
        data_vals = plot_data.to_numpy(dtype=float)
        if np.isfinite(data_vals).any():
            vmin = np.nanmin(data_vals)
            vmax = np.nanmax(data_vals)
    hm = sns.heatmap(
        plot_data,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        annot_kws={"size": annot_size},
        fmt=".2f",
        cbar_kws={"label": cbar_label}
    )
    plot_title = title
    if args.case_label:
        plot_title = f"{args.case_label}: {title}"
    if not args.no_title:
        ax.set_title(plot_title, fontsize=title_size, fontweight="bold" if args.publish_style else None)
    ax.set_xlabel(args.x_label, fontsize=axis_label_size, fontweight="bold" if args.publish_style else None)
    ax.set_ylabel(args.y_label, fontsize=axis_label_size, fontweight="bold" if args.publish_style else None)
    ax.tick_params(axis="x", labelrotation=45, labelsize=tick_label_size)
    ax.tick_params(axis="y", labelrotation=0, labelsize=tick_label_size)

    cbar = hm.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=cbar_label_size, fontweight="bold" if args.publish_style else None)
    cbar.ax.tick_params(labelsize=cbar_tick_size)

    plt.tight_layout()

    out_png = f"{prefix}_{col}.png"
    plt.savefig(out_png, dpi=save_dpi)
    plt.close()
    print(f"Saved {out_png}")

print("\n4 heatmaps generated successfully.\n")
