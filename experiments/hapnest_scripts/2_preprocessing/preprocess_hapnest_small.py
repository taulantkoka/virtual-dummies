"""
HAPNEST Preprocessing — Small-scale (chromosome 1 only)
========================================================
Loads QC'd PLINK genotypes for chromosome 1 via pandas_plink, clusters
SNPs by |corr| > threshold, picks one representative per cluster, and
saves the pruned matrix as Fortran-order memmap files.

CLI mirrors preprocess_hapnest_full.py so the same Slurm wrapper works
for both scales.

Usage:
    # All runs sequentially:
    python preprocess_hapnest_small.py /path/to/outputs /path/to/preprocessed 100

    # Single run (for SLURM --array):
    python preprocess_hapnest_small.py /path/to/outputs /path/to/preprocessed 100 --run-id 1
"""
from __future__ import annotations

import os, sys, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

from pandas_plink import read_plink
import logging
logging.getLogger("pandas_plink").setLevel(logging.ERROR)

EPS = 1e-12
MAX_N_FOR_CLUSTERING = 3000


# =============================================================
# Utilities
# =============================================================

def center_unitL2(X, eps=EPS):
    X = X - X.mean(axis=0, keepdims=True)
    n2 = np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return X / n2


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1


def cluster_components_threshold(Xc, *, threshold=0.7, block_size=2000):
    n, p = Xc.shape
    uf = UnionFind(p)
    total_edges = 0
    for start in range(0, p, block_size):
        end = min(start + block_size, p)
        C_block = Xc[:, start:end].T @ Xc
        for j in range(end - start):
            C_block[j, start + j] = 0.0
        rows, cols = np.where(np.abs(C_block) > threshold)
        for r, c in zip(rows, cols):
            global_r = start + r
            if global_r < c:
                uf.union(global_r, c)
                total_edges += 1
    roots = np.array([uf.find(j) for j in range(p)], dtype=np.int32)
    clusters = {}
    for j in range(p):
        clusters.setdefault(int(roots[j]), []).append(j)
    n_clusters = len(clusters)
    singletons = sum(1 for c in clusters.values() if len(c) == 1)
    max_cluster = max(len(c) for c in clusters.values())
    print(f"    {total_edges} pairs |corr|>{threshold} -> "
          f"{n_clusters} clusters (singletons: {singletons}, max: {max_cluster})",
          flush=True)
    return clusters


def choose_representatives(clusters, *, mode, rng):
    reps = []
    for members in clusters.values():
        if mode == "first":
            reps.append(int(min(members)))
        elif mode == "random":
            reps.append(int(members[rng.integers(len(members))]))
    return np.array(sorted(reps), dtype=int)


# =============================================================
# Data loading (chromosome 1 only, via pandas_plink)
# =============================================================

def load_chr1_genotypes(run_dir, run_id):
    prefix = run_dir / f"run_{run_id}_chr-1"
    if not Path(str(prefix) + ".bim").exists():
        return None
    bim, fam, G = read_plink(str(prefix), verbose=False)
    X = G.compute().T.astype(np.float64)   # (n, p), 0/1/2 dosage
    np.nan_to_num(X, copy=False, nan=0.0)
    snp_ids = bim["snp"].tolist()
    return X, snp_ids


# =============================================================
# Process one run
# =============================================================

def process_one_run(run_id, base_dir, out_dir, *,
                    corr_threshold=0.7, seed0=42, block_size=2000,
                    rep_mode="random"):
    run_dir = Path(base_dir) / f"run_{run_id}"
    if not run_dir.exists():
        print(f"  Run {run_id}: directory not found, skipping")
        return False

    loaded = load_chr1_genotypes(run_dir, run_id)
    if loaded is None:
        print(f"  Run {run_id}: chr-1 PLINK files not found, skipping")
        return False

    X_raw, snp_ids = loaded
    n, p_raw = X_raw.shape
    print(f"\n  Run {run_id}: loaded chr 1, {n} x {p_raw}", flush=True)

    seed = int(seed0 + run_id * 10_007)
    rng = np.random.default_rng(seed)

    # Subsample rows for clustering, re-standardize
    if n > MAX_N_FOR_CLUSTERING:
        sub_idx = rng.choice(n, size=MAX_N_FOR_CLUSTERING, replace=False)
        Xc_clust = center_unitL2(X_raw[sub_idx, :])
        print(f"    Subsampling {MAX_N_FOR_CLUSTERING}/{n} rows for clustering",
              flush=True)
    else:
        Xc_clust = center_unitL2(X_raw)

    clusters = cluster_components_threshold(
        Xc_clust, threshold=corr_threshold, block_size=block_size
    )
    reps = choose_representatives(clusters, mode=rep_mode, rng=rng)
    p_rep = int(reps.size)

    # Pruned matrices (full n rows)
    X_raw_reps = X_raw[:, reps]
    Xc_reps = center_unitL2(X_raw_reps)
    rep_snp_ids = [snp_ids[j] for j in reps]

    # Spot-check max |corr|
    check_n = min(n, MAX_N_FOR_CLUSTERING)
    check_idx = rng.choice(n, size=check_n, replace=False)
    Xc_check = center_unitL2(X_raw_reps[check_idx, :])
    if p_rep > 5000:
        check_cols = rng.choice(p_rep, size=5000, replace=False)
        C = Xc_check[:, check_cols].T @ Xc_check[:, check_cols]
    else:
        C = Xc_check.T @ Xc_check
    np.fill_diagonal(C, 0)
    max_corr = float(np.abs(C).max())
    print(f"    After pruning: {n} x {p_rep}, max |corr| = {max_corr:.4f}",
          flush=True)

    # Write memmap files (same format as full-scale)
    run_out = out_dir / f"run_{run_id}"
    run_out.mkdir(parents=True, exist_ok=True)

    X_std_mm = np.memmap(
        run_out / "X_std.dat", dtype="float64", mode="w+",
        shape=(n, p_rep), order="F",
    )
    X_raw_mm = np.memmap(
        run_out / "X_raw.dat", dtype="float64", mode="w+",
        shape=(n, p_rep), order="F",
    )
    X_std_mm[:] = np.asfortranarray(Xc_reps)
    X_raw_mm[:] = np.asfortranarray(X_raw_reps)
    X_std_mm.flush(); X_raw_mm.flush()
    del X_std_mm, X_raw_mm

    # Metadata (keys match full-scale for benchmark_worker.py compatibility)
    meta = {
        "run_id": run_id,
        "seed": seed,
        "n_samples": int(n),
        "p_raw_total": int(p_raw),
        "p_pruned_total": int(p_rep),
        "n_chromosomes": 1,
        "corr_threshold": float(corr_threshold),
        "rep_mode": rep_mode,
        "max_n_for_clustering": MAX_N_FOR_CLUSTERING,
        "max_corr_after_pruning": float(max_corr),
        "per_chromosome": [
            {"chrom": 1, "p_raw": int(p_raw), "p_pruned": int(p_rep),
             "n_clusters": int(len(clusters))},
        ],
    }
    with open(run_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(run_out / "kept_snps.txt", "w") as f:
        f.write("\n".join(rep_snp_ids) + "\n")

    print(f"  Run {run_id}: done — {n} x {p_rep}, saved to {run_out}", flush=True)
    return True


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAPNEST small-scale preprocessing (chromosome 1 only)")
    parser.add_argument("base_dir",
                        help="Directory with run_*/run_*_chr-1.bed files")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("n_runs", type=int, help="Total number of runs")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Process a single run (for SLURM array jobs)")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--rep-mode", default="random",
                        choices=["random", "first"])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HAPNEST Preprocessing — Small-scale (chr 1, pandas_plink)")
    print("=" * 60)
    print(f"Input:           {base_dir}")
    print(f"Output:          {out_dir}")
    print(f"corr_threshold:  {args.corr_threshold}")
    print(f"rep_mode:        {args.rep_mode}")
    print(f"seed0:           {args.seed0}")
    print(f"max_n_for_clust: {MAX_N_FOR_CLUSTERING}")
    print("=" * 60)

    if args.run_id is not None:
        ok = process_one_run(
            args.run_id, base_dir, out_dir,
            corr_threshold=args.corr_threshold,
            seed0=args.seed0,
            rep_mode=args.rep_mode,
        )
        sys.exit(0 if ok else 1)
    else:
        success = failed = 0
        for rid in range(1, args.n_runs + 1):
            ok = process_one_run(
                rid, base_dir, out_dir,
                corr_threshold=args.corr_threshold,
                seed0=args.seed0,
                rep_mode=args.rep_mode,
            )
            if ok:
                success += 1
            else:
                failed += 1

        global_meta = {
            "timestamp": datetime.now().isoformat(),
            "base_dir": str(base_dir),
            "scale": "small (chr 1 only)",
            "n_runs_processed": success,
            "n_failed": failed,
            "settings": {
                "corr_threshold": args.corr_threshold,
                "seed0": args.seed0,
                "rep_mode": args.rep_mode,
                "max_n_for_clustering": MAX_N_FOR_CLUSTERING,
            },
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(global_meta, f, indent=2)

        print(f"\nDone! Success: {success}, Failed: {failed}")
        print(f"Output: {out_dir}/run_*/{{X_std.dat, X_raw.dat, meta.json}}")