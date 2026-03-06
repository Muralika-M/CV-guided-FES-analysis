#!/usr/bin/env python3
"""
Plot correlation heatmaps between WT residue importance and avg dTm per site.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default=str(DATA_DIR / "residue/wt_residue_importance.csv"),
    )
    ap.add_argument("--tm_csv", default=str(DATA_DIR / "Tm_table.csv"))
#    ap.add_argument("--case", default="pearson_intersection_foldedmap")
#    ap.add_argument("--out_prefix", default="pearson_intersection_foldedmap")
    ap.add_argument("--out_dir", default=str(RESULTS_DIR / "residue_heatmaps"))
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    if args.case and "case" in df.columns:
        df = df[df["case"] == args.case].copy()
        if df.empty:
            raise SystemExit(f"No rows for case '{args.case}' in {args.summary_csv}")

    sites, avg_signed, avg_abs = avg_dtm_per_site(args.tm_csv)
    desc_counts = {0: 7, 1: 6, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 6, 9: 7}

    thrF_vals = sorted(df["thrF"].unique())
    thrU_vals = sorted(df["thrU"].unique())

    pearson_signed = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    pearson_abs = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    spearman_signed = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    spearman_abs = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    pearson_signed_norm = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    pearson_abs_norm = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    spearman_signed_norm = np.full((len(thrF_vals), len(thrU_vals)), np.nan)
    spearman_abs_norm = np.full((len(thrF_vals), len(thrU_vals)), np.nan)

    for _, row in df.iterrows():
        imp = np.array([row.get(f"res{r}") for r in sites], dtype=float)
        mask = np.isfinite(imp)
        if mask.sum() < 3:
            continue
        imp_use = imp[mask]
        signed_use = avg_signed[mask]
        abs_use = avg_abs[mask]

        pr_s, _ = pearsonr(imp_use, signed_use)
        pr_a, _ = pearsonr(imp_use, abs_use)
        sr_s, _ = spearmanr(imp_use, signed_use)
        sr_a, _ = spearmanr(imp_use, abs_use)

        counts = np.array([desc_counts[sites[i]] for i in range(len(sites))], dtype=float)
        imp_norm = imp_use / counts[mask]
        pr_s_n, _ = pearsonr(imp_norm, signed_use)
        pr_a_n, _ = pearsonr(imp_norm, abs_use)
        sr_s_n, _ = spearmanr(imp_norm, signed_use)
        sr_a_n, _ = spearmanr(imp_norm, abs_use)

        i = thrF_vals.index(row["thrF"])
        j = thrU_vals.index(row["thrU"])
        pearson_signed[i, j] = pr_s
        pearson_abs[i, j] = pr_a
        spearman_signed[i, j] = sr_s
        spearman_abs[i, j] = sr_a
        pearson_signed_norm[i, j] = pr_s_n
        pearson_abs_norm[i, j] = pr_a_n
        spearman_signed_norm[i, j] = sr_s_n
        spearman_abs_norm[i, j] = sr_a_n

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annot_font = max(4, int(40 / max(len(thrF_vals), len(thrU_vals))))
    heatmap_params = dict(
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_font},
        cbar=False,
        square=True,
    )

    case_label = args.case if args.case else "all"

    def draw_two_panel(matrix_left, matrix_right, title_left, title_right, out_name):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        sns.heatmap(
            matrix_left,
            ax=axes[0],
            xticklabels=[f"{x:.2f}" for x in thrU_vals],
            yticklabels=[f"{y:.2f}" for y in thrF_vals],
            **heatmap_params,
        )
        axes[0].set_title(title_left, fontsize=8)

        sns.heatmap(
            matrix_right,
            ax=axes[1],
            xticklabels=[f"{x:.2f}" for x in thrU_vals],
            yticklabels=[f"{y:.2f}" for y in thrF_vals],
            **heatmap_params,
        )
        axes[1].set_title(title_right, fontsize=8)

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)

        plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.18, wspace=0.15)
        out_path = out_dir / out_name
        plt.savefig(out_path, dpi=450)
        plt.close()
        print(f"Saved {out_path}")

    draw_two_panel(
        pearson_signed,
        pearson_abs,
        f"{case_label}: residue imp vs avg dTm (Pearson)",
        f"{case_label}: residue imp vs avg |dTm| (Pearson)",
        f"{args.out_prefix}_pearson.png",
    )

    draw_two_panel(
        spearman_signed,
        spearman_abs,
        f"{case_label}: residue imp vs avg dTm (Spearman)",
        f"{case_label}: residue imp vs avg |dTm| (Spearman)",
        f"{args.out_prefix}_spearman.png",
    )

    draw_two_panel(
        pearson_signed_norm,
        pearson_abs_norm,
        f"{case_label}: norm imp vs avg dTm (Pearson)",
        f"{case_label}: norm imp vs avg |dTm| (Pearson)",
        f"{args.out_prefix}_pearson_norm.png",
    )

    draw_two_panel(
        spearman_signed_norm,
        spearman_abs_norm,
        f"{case_label}: norm imp vs avg dTm (Spearman)",
        f"{case_label}: norm imp vs avg |dTm| (Spearman)",
        f"{args.out_prefix}_spearman_norm.png",
    )

    corr_rows = []
    for i, thrF in enumerate(thrF_vals):
        for j, thrU in enumerate(thrU_vals):
            corr_rows.append({
                "thrF": thrF,
                "thrU": thrU,
                "pearson_signed": pearson_signed[i, j],
                "pearson_abs": pearson_abs[i, j],
                "spearman_signed": spearman_signed[i, j],
                "spearman_abs": spearman_abs[i, j],
                "pearson_signed_norm": pearson_signed_norm[i, j],
                "pearson_abs_norm": pearson_abs_norm[i, j],
                "spearman_signed_norm": spearman_signed_norm[i, j],
                "spearman_abs_norm": spearman_abs_norm[i, j],
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = out_dir / f"{args.out_prefix}_corr_summary.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved {corr_csv}")


if __name__ == "__main__":
    main()
