#!/usr/bin/env python3
"""
HLDA sweep using RMSD index caches (no per-threshold feather files).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data_examples"


def parse_range(s):
    vals = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            a, b, c = map(float, chunk.split(":"))
            x = a
            while x <= b + 1e-9:
                vals.append(round(x, 6))
                x += c
        else:
            vals.append(float(chunk))
    return sorted(set(vals))


def read_header(path: Path):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith("#! FIELDS"):
                header = line.strip().split()[2:]
                return header, i + 1
    raise ValueError(f"No '#! FIELDS' in {path}")


def load_desc_df(path: Path) -> pd.DataFrame:
    header, skip = read_header(path)
    idxs = [i for i, name in enumerate(header) if name.startswith("d")]
    if not idxs:
        return pd.DataFrame()
    df = pd.read_csv(path, sep=r"\s+", skiprows=skip, header=None, usecols=idxs)
    df.columns = [header[i] for i in idxs]
    return df


def drop_features(df, cv_tol=0.0, corr_tol=0.93):
    tmp = df.copy(deep=True)
    tmp = tmp.loc[:, (tmp.std() / tmp.mean()).abs() >= cv_tol]
    corr_matrix = tmp.corr(method="spearman").abs()
    lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
    to_drop = [column for column in lower.columns if any(lower[column] > corr_tol)]
    tmp.drop(to_drop, axis=1, inplace=True)
    return tmp.columns


def compute_hlda(F_df, U_df, row_skip, npoints, corr_tol):
    desc = list(F_df.columns)
    if not desc:
        return None, None, "no_descriptors"

    try:
        df_all = pd.concat([F_df[desc], U_df[desc]], ignore_index=True)
        selected = list(drop_features(df_all, cv_tol=0.0, corr_tol=corr_tol))
    except Exception as e:
        return None, None, f"drop_features_error:{e}"

    if len(selected) == 0:
        return None, None, "zero_after_filter"

    fF = F_df[selected].values[row_skip:]
    fU = U_df[selected].values[row_skip:]

    if len(fF) < 20 or len(fU) < 20:
        return None, None, "too_few_frames"

    nstep_F = len(fF) // npoints
    nstep_U = len(fU) // npoints
    if nstep_F < 5 or nstep_U < 5:
        return None, None, "chunk_small"

    j = npoints
    F_use = fF[:j * nstep_F]
    U_use = fU[:j * nstep_U]

    muF = F_use.mean(0)
    muU = U_use.mean(0)

    CovF = np.cov(F_use.T)
    CovU = np.cov(U_use.T)

    try:
        Cov_inv = np.linalg.inv(CovF) + np.linalg.inv(CovU)
    except Exception:
        return None, None, "singular_cov"

    mu_tot = 0.5 * (muF + muU)
    SB = (np.outer(muF - mu_tot, muF - mu_tot) +
          np.outer(muU - mu_tot, muU - mu_tot))

    M = Cov_inv @ SB

    try:
        eigvals, eigvecs = np.linalg.eig(M)
    except Exception:
        return None, None, "eig_error"

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    idx = eigvals.argsort()[::-1]
    top_ev = float(eigvals[idx][0])

    if np.isnan(top_ev):
        return None, None, "nan_ev"

    return top_ev, eigvecs[:, idx][:, 0], "ok"


def process_mutant(args):
    name, mdir, idx_dir, thrF_list, thrUF_list, row_skip, npoints, corr_tol = args

    cache = idx_dir / name
    folded_npz = cache / "folded_idx.npz"
    unfolded_npz = cache / "unfolded_idx.npz"
    if not folded_npz.exists() or not unfolded_npz.exists():
        return []

    F_path = mdir / "COLVAR_CV_F"
    U_path = mdir / "COLVAR_CV_UF"
    if not F_path.exists() or not U_path.exists():
        return []

    F_df = load_desc_df(F_path)
    U_df = load_desc_df(U_path)
    if F_df.empty or U_df.empty:
        return []

    rows = []
    with np.load(folded_npz) as f_idx, np.load(unfolded_npz) as u_idx:
        for thrF in thrF_list:
            keyF = f"{thrF:.2f}"
            if keyF not in f_idx:
                continue
            idxF = f_idx[keyF]
            if len(idxF) == 0:
                continue
            Fd = F_df.iloc[idxF]

            for thrUF in thrUF_list:
                keyU = f"{thrUF:.2f}"
                if keyU not in u_idx:
                    continue
                idxU = u_idx[keyU]
                if len(idxU) == 0:
                    continue
                Ud = U_df.iloc[idxU]

                ev, vec, status = compute_hlda(Fd, Ud, row_skip, npoints, corr_tol)
                if status != "ok":
                    continue

                rows.append({
                    "Mutant": name,
                    "Thr_F": thrF,
                    "Thr_UF": thrUF,
                    "EV": ev
                })
    return rows


def main():
    default_data = Path(__file__).resolve().parents[2] / "minH_pdbs_fast"

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(default_data))
    ap.add_argument("--idx_dir", default=str(DATA_DIR / "cache/threshold_idx_cache"))
    ap.add_argument("--tm_csv", default=str(DATA_DIR / "Tm_table.csv"))
    ap.add_argument("--thrF", default="0.15:0.45:0.02")
    ap.add_argument("--thrUF", default="0.45:0.96:0.02")
    ap.add_argument("--out_csv", default=str(DATA_DIR / "ev/hlda_tm_heatmap_indexed.csv"))
    ap.add_argument("--per_mutant_csv", default=str(DATA_DIR / "ev/per_mutant_hlda_EV_indexed.csv"))
    ap.add_argument("--row_skip", type=int, default=5)
    ap.add_argument("--npoints", type=int, default=10)
    ap.add_argument("--corr_tol", type=float, default=0.98)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    idx_dir = Path(args.idx_dir)
    tm_csv = Path(args.tm_csv) if args.tm_csv else data_dir / "Tm_table.csv"

    thrF_list = parse_range(args.thrF)
    thrUF_list = parse_range(args.thrUF)

    tm = pd.read_csv(tm_csv)
    tm["Mutant"] = tm["Mutant"].astype(str)
    valid_mutants = set(tm["Mutant"])

    per_mutant_rows = []

    mutants = []
    for mdir in sorted(data_dir.iterdir()):
        if not mdir.is_dir():
            continue
        name = mdir.name
        if name not in valid_mutants:
            continue
        mutants.append((name, mdir))

    args_list = [
        (name, mdir, idx_dir, thrF_list, thrUF_list, args.row_skip, args.npoints, args.corr_tol)
        for name, mdir in mutants
    ]

    total = len(args_list)
    done = 0

    if args.n_jobs > 1 and total > 0:
        n_jobs = min(args.n_jobs, total)
        pool = None
        use_pool = True
        try:
            pool = Pool(processes=n_jobs)
        except (PermissionError, OSError) as exc:
            use_pool = False
            print(f"Pool init failed ({exc}); using ThreadPoolExecutor with {n_jobs} workers.")
        if use_pool:
            try:
                for rows in pool.imap_unordered(process_mutant, args_list):
                    if rows:
                        per_mutant_rows.extend(rows)
                    done += 1
                    if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
                        print(f"Processed {done}/{total} mutants")
            finally:
                pool.close()
                pool.join()
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(process_mutant, arg) for arg in args_list]
                for fut in as_completed(futures):
                    rows = fut.result()
                    if rows:
                        per_mutant_rows.extend(rows)
                    done += 1
                    if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
                        print(f"Processed {done}/{total} mutants")
    else:
        for arg in args_list:
            rows = process_mutant(arg)
            if rows:
                per_mutant_rows.extend(rows)
            done += 1
            if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
                print(f"Processed {done}/{total} mutants")

    per_mutant_df = pd.DataFrame(per_mutant_rows)
    per_mutant_df.to_csv(args.per_mutant_csv, index=False)
    print(f"{args.per_mutant_csv} written (for scatter & dHLDA heatmaps)\n")

    if per_mutant_df.empty:
        heatmap_df = pd.DataFrame(columns=[
            "Thr_F", "Thr_UF", "N", "Pearson_r", "Pearson_p", "Spearman_rho", "Spearman_p"
        ])
        heatmap_df.to_csv(args.out_csv, index=False)
        print(f"\nThreshold heatmap CSV  {args.out_csv}")
        return

    merged_all = per_mutant_df.merge(tm, on="Mutant")

    heatmap_rows = []
    grouped = merged_all.groupby(["Thr_F", "Thr_UF"])
    stats = {(a, b): g for (a, b), g in grouped}

    for thrF in thrF_list:
        for thrUF in thrUF_list:
            key = (thrF, thrUF)
            if key not in stats:
                heatmap_rows.append({
                    "Thr_F": thrF,
                    "Thr_UF": thrUF,
                    "N": 0,
                    "Pearson_r": np.nan,
                    "Pearson_p": np.nan,
                    "Spearman_rho": np.nan,
                    "Spearman_p": np.nan
                })
                continue

            grp = stats[key]
            if len(grp) >= 3:
                pr, pp = pearsonr(grp["EV"], grp["Tm"])
                sr, sp = spearmanr(grp["EV"], grp["Tm"])
            else:
                pr = pp = sr = sp = np.nan

            heatmap_rows.append({
                "Thr_F": thrF,
                "Thr_UF": thrUF,
                "N": len(grp),
                "Pearson_r": pr,
                "Pearson_p": pp,
                "Spearman_rho": sr,
                "Spearman_p": sp
            })

    heatmap_df = pd.DataFrame(heatmap_rows)
    heatmap_df.to_csv(args.out_csv, index=False)
    print(f"\nThreshold heatmap CSV  {args.out_csv}")


if __name__ == "__main__":
    main()
